#!/usr/bin/env bash
# Raw runner for the CUDA-METRO speed benchmark. No scheduler, no queue.
# For a machine you control: log in, start it, close the terminal, come back.
#
#   ./run.sh setup            create the venv and install everything
#   ./run.sh preflight        7 checks in seconds; do this before a long run
#   ./run.sh smoke            ~2 min sanity sweep, prints to the terminal
#   ./run.sh start            full sweep, DETACHED (survives logout)
#   ./run.sh status           running / completed / failed / died, + progress
#   ./run.sh wait             block until it ends, then report how it ended
#   ./run.sh log              follow the live log
#   ./run.sh stop             kill it
#   ./run.sh resume           restart, skipping points already measured
#   ./run.sh analyze          frontier + usable-P ceiling from the CSV
#
# Pick a GPU:   GPU=2 ./run.sh start        (default: GPU 0)
# Override the sweep:
#   SIZES=64,128 BLOCKS=64,256,1024 ATTEMPTS=2e6 REPEATS=1 ./run.sh start
#
# --- edit if your layout differs ---------------------------------------
VENV="${VENV:-$HOME/venv-cudametro}"
PYTHON="${PYTHON:-python3}"
CUDA_HOME_HINT="${CUDA_HOME:-}"   # set CUDA_HOME=/usr/local/cuda-12.9 to pin a toolkit
# -----------------------------------------------------------------------

# pycuda has no wheels: every install compiles a Boost-Python subset from
# source, using whatever CXX the interpreter's sysconfig recorded when IT was
# built. A Homebrew Python records Homebrew's compiler (g++-12), which does not
# exist on a Fedora box - so the build dies on a compiler nobody chose. Prefer
# a distro interpreter, and force CC/CXX to something that is actually here.
pick_python() {
    [ -n "${PYTHON_EXPLICIT:-}" ] && return 0
    local p
    for p in /usr/bin/python3.12 /usr/bin/python3.13 /usr/bin/python3.11 \
             /usr/bin/python3 "$PYTHON"; do
        [ -x "$p" ] || continue
        case "$("$p" -c 'import sys;print(sys.version_info[:2]>=(3,9) and sys.version_info[:2]<(3,14))' 2>/dev/null)" in
            True) PYTHON="$p"; return 0 ;;
        esac
    done
    return 0                       # fall through to whatever was set
}

fix_compilers() {
    # If sysconfig names a compiler that is not installed, fall back to the
    # plain names. Explicit CC/CXX from the caller always win.
    local sys_cxx
    sys_cxx="$("$PYTHON" -c 'import sysconfig;print(sysconfig.get_config_var("CXX") or "")' 2>/dev/null | awk '{print $1}')"
    if [ -n "$sys_cxx" ] && ! command -v "$sys_cxx" >/dev/null 2>&1; then
        echo "note: this interpreter wants '$sys_cxx', which is not installed."
        export CC="${CC:-gcc}" CXX="${CXX:-g++}"
        echo "      forcing CC=$CC CXX=$CXX"
    else
        export CC="${CC:-gcc}" CXX="${CXX:-g++}"
    fi
}

# nvcc refuses host compilers newer than it supports ("unsupported GNU
# version"). That bites at RUNTIME here, inside SourceModule, long after
# install. Test it now and pin an older g++ through pycuda's own flag hook.
check_nvcc_hostcc() {
    local ENVFILE="$HERE/.run/env"   # $HERE is only set further down
    local tmp; tmp="$(mktemp -d)"
    printf '__global__ void k(){}\nint main(){return 0;}\n' > "$tmp/t.cu"
    if nvcc -o "$tmp/t" "$tmp/t.cu" >"$tmp/err" 2>&1; then
        rm -rf "$tmp"; return 0
    fi
    if ! grep -qi "unsupported GNU version\|requires GCC\|is not supported" "$tmp/err"; then
        echo "nvcc test compile failed for another reason:"; sed -n '1,6p' "$tmp/err"
        rm -rf "$tmp"; return 1
    fi
    echo "nvcc rejects the default host compiler ($(g++ -dumpversion 2>/dev/null))."
    local g
    for g in $(ls /usr/bin/g++-* /usr/bin/g++1? 2>/dev/null | sort -Vr); do
        if nvcc -ccbin="$g" -o "$tmp/t" "$tmp/t.cu" >/dev/null 2>&1; then
            echo "  using $g for device compilation"
            printf 'export PYCUDA_DEFAULT_NVCC_FLAGS="-ccbin=%s"\n' "$g" >> "$ENVFILE"
            export PYCUDA_DEFAULT_NVCC_FLAGS="-ccbin=$g"
            rm -rf "$tmp"; return 0
        fi
    done
    echo "  no installed g++ is accepted by this nvcc."
    echo "  Fedora:  sudo dnf install gcc13-c++    (then re-run setup)"
    echo "  or pin an older toolkit: CUDA_HOME=/usr/local/cuda-12.9 ./run.sh setup"
    rm -rf "$tmp"; return 1
}

# CUDA-METRO compiles its kernels at RUNTIME through pycuda's SourceModule, so
# nvcc must be on PATH every time the benchmark starts - not merely when pycuda
# was installed. A detached run inheriting a shell without it dies seconds in,
# which is a miserable way to lose an overnight sweep. Find it here instead of
# trusting the environment.
find_cuda() {
    # CUDA_HOME first. An nvcc already on PATH must NOT win: the user exports
    # /usr/local/cuda/bin once, and every later attempt to pin a different
    # toolkit is then silently ignored.
    if [ -n "$CUDA_HOME_HINT" ] && [ -x "$CUDA_HOME_HINT/bin/nvcc" ]; then
        CUDA_ROOT="$CUDA_HOME_HINT"
        export PATH="$CUDA_ROOT/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
        return 0
    fi
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_ROOT="$(dirname "$(dirname "$(command -v nvcc)")")"
        return 0
    fi
    local c
    for c in /usr/local/cuda /opt/cuda \
             $(ls -d /usr/local/cuda-* 2>/dev/null | sort -Vr); do
        [ -n "$c" ] && [ -x "$c/bin/nvcc" ] || continue
        CUDA_ROOT="$c"
        export PATH="$c/bin:$PATH"
        export LD_LIBRARY_PATH="$c/lib64:${LD_LIBRARY_PATH:-}"
        return 0
    done
    return 1
}

# CUDA 13 removed Volta (sm_70) and everything older. Auto-picking "the newest
# toolkit" therefore picks one that CANNOT build for a V100 - and the failure
# arrives as "nvcc fatal: Unsupported gpu architecture 'sm_70'" at runtime,
# inside SourceModule. Ask the DEVICE what it is, then pick a toolkit that
# still speaks that architecture.
device_arch() {
    [ -x "$VENV/bin/python" ] || return 1
    "$VENV/bin/python" - <<'PYX' 2>/dev/null
import pycuda.autoinit, pycuda.driver as d      # noqa: F401
print("%d%d" % d.Device(0).compute_capability())
PYX
}

check_arch() {
    local ENVFILE="$HERE/.run/env" arch tmp cuda cc ex why pf shim ccc
    arch="$(device_arch)" || { echo "note: cannot query the GPU yet (venv not built); skipping arch check"; return 0; }
    [ -n "$arch" ] || { echo "note: could not read compute capability; skipping arch check"; return 0; }
    echo "device compute capability: sm_$arch"

    tmp="$(mktemp -d)"; printf '#include <math.h>\n__global__ void k(){}\n' > "$tmp/t.cu"

    # Three axes, and they are NOT independent:
    #   toolkit      - CUDA 13 dropped Volta, so 13.x cannot target sm_70
    #   host gcc     - each toolkit accepts only up to some gcc version
    #   glibc headers- glibc >= 2.41 added the C23 functions cospi/sinpi/tanpi;
    #                  CUDA's crt/math_functions.h redeclares them without the
    #                  matching noexcept, so every compile dies with
    #                  "exception specification is incompatible". -ccbin does
    #                  NOT fix that: an older g++ still reads the same
    #                  /usr/include. Only a define or a newer CUDA does.
    # So try the full matrix and report the real refusal per toolkit.
    local extras=(""
        "-D__GLIBC_USE_IEC_60559_FUNCS_EXT_C23=0"
        "-D__GLIBC_USE_IEC_60559_FUNCS_EXT_C2X=0"
        "-D__GLIBC_USE_IEC_60559_FUNCS_EXT_C23=0 -D__GLIBC_USE_IEC_60559_FUNCS_EXT_C2X=0")
    local cudas=() ccs=("")
    [ -n "$CUDA_HOME_HINT" ] && cudas+=("$CUDA_HOME_HINT")
    command -v nvcc >/dev/null 2>&1 && cudas+=("$(dirname "$(dirname "$(command -v nvcc)")")")
    local d; for d in $(ls -d /usr/local/cuda-*.* 2>/dev/null | sort -Vr) /usr/local/cuda /opt/cuda; do
        [ -x "$d/bin/nvcc" ] && cudas+=("$d")
    done
    local g; for g in $(ls /usr/bin/g++-1? /usr/bin/g++1? 2>/dev/null | sort -Vr) /usr/bin/clang++; do
        [ -x "$g" ] && ccs+=("$g")
    done

    echo "  toolkits: $(printf '%s\n' "${cudas[@]}" | awk '!s[$0]++' | tr '\n' ' ')"
    echo "  host compilers: default $(printf '%s ' "${ccs[@]:1}")"
    echo "  glibc $(ldd --version 2>/dev/null | head -1 | awk '{print $NF}')  gcc $(gcc -dumpversion 2>/dev/null)"

    for cuda in $(printf '%s\n' "${cudas[@]}" | awk '!s[$0]++'); do
        why=""
        for cc in "${ccs[@]}"; do
            for ex in "${extras[@]}"; do
                local flags=(-arch="sm_$arch" -cubin -Wno-deprecated-gpu-targets -o "$tmp/t.cubin")
                [ -n "$cc" ] && flags+=(-ccbin="$cc")
                if [ -n "$ex" ]; then
                    local exarr; read -r -a exarr <<< "$ex"; flags+=("${exarr[@]}")
                fi
                [ -n "${NVCC_ALLOW_UNSUPPORTED:-}" ] && flags+=(-allow-unsupported-compiler)

                if "$cuda/bin/nvcc" "${flags[@]}" "$tmp/t.cu" >"$tmp/err" 2>&1; then
                    echo "  OK: $cuda${cc:+   ccbin $cc}${ex:+   $ex}"
                    CUDA_ROOT="$cuda"
                    export CUDA_HOME="$cuda"
                    export PATH="$cuda/bin:$PATH"
                    export LD_LIBRARY_PATH="$cuda/lib64:${LD_LIBRARY_PATH:-}"
                    pf="-Wno-deprecated-gpu-targets"
                    [ -n "$cc" ] && pf="$pf -ccbin=$cc"
                    [ -n "$ex" ] && pf="$pf $ex"
                    [ -n "${NVCC_ALLOW_UNSUPPORTED:-}" ] && pf="$pf -allow-unsupported-compiler"
                    export PYCUDA_DEFAULT_NVCC_FLAGS="$pf"

                    # PYCUDA_DEFAULT_NVCC_FLAGS alone is NOT enough. pycuda
                    # honours it only when the caller passes no options, and
                    # montecarlo.py calls
                    #     SourceModule(..., options=["-std=c++17","-O0"])
                    # so -ccbin never reaches nvcc for the real kernels.
                    # Absent -ccbin, nvcc takes the host compiler as plain
                    # `g++` from PATH - so put the right one there under that
                    # name. Works however nvcc ends up being invoked.
                    if [ -n "$cc" ]; then
                        shim="$HERE/.run/hostcc"
                        mkdir -p "$shim"
                        ln -sf "$cc" "$shim/g++"
                        ln -sf "$cc" "$shim/c++"
                        ccc="${cc/g++/gcc}"
                        if [ -x "$ccc" ]; then
                            ln -sf "$ccc" "$shim/gcc"
                            ln -sf "$ccc" "$shim/cc"
                        fi
                        export PATH="$shim:$PATH"
                        echo "  host-compiler shim: $shim/g++ -> $cc"
                    fi

                    {   printf 'export CUDA_HOME=%s\n' "$cuda"
                        [ -n "$cc" ] && printf 'export PATH=%s/.run/hostcc:$PATH\n' "$HERE"
                        printf 'export PATH=%s/bin:$PATH\n' "$cuda"
                        printf 'export LD_LIBRARY_PATH=%s/lib64:${LD_LIBRARY_PATH:-}\n' "$cuda"
                        printf 'export PYCUDA_DEFAULT_NVCC_FLAGS="%s"\n' "$pf"
                    } >> "$ENVFILE"
                    rm -rf "$tmp"; return 0
                fi
                why="$(grep -m1 -E 'error|fatal error|nvcc fatal' "$tmp/err" | cut -c1-150)"
                [ -n "$why" ] || why="$(head -1 "$tmp/err" | cut -c1-150)"
            done
        done
        echo "  no: $(basename "$cuda") -> $why"
    done

    echo
    echo "No working (toolkit, compiler, flags) combination. Decode the refusal:"
    echo "  'Unsupported gpu architecture sm_$arch'"
    echo "      CUDA 13 dropped Volta. Only a 12.x toolkit can target this card."
    echo "  'unsupported GNU version'"
    echo "      host gcc too new for that toolkit -> sudo dnf install gcc14-c++"
    echo "  'exception specification is incompatible ... cospi'"
    echo "      glibc >= 2.41 vs CUDA headers. NOT a gcc-version problem: an"
    echo "      older g++ reads the same /usr/include, and -D of the glibc"
    echo "      feature macro does not survive (features.h redefines it)."
    echo "      ==> FIX IT:  python3 fix_cuda_glibc.py --dry-run"
    echo "                   sudo python3 fix_cuda_glibc.py --cuda /usr/local/cuda-12.9"
    echo "                   (adds the missing noexcept to CUDA's declarations;"
    echo "                    backs up, and --revert restores byte-identically)"
    rm -rf "$tmp"; return 1
}
cuda_or_die() {
    find_cuda || {
        echo "nvcc not found."
        echo "Installed toolkits: $(ls -d /usr/local/cuda-* 2>/dev/null | tr '\n' ' ')"
        echo "Pin one and retry, e.g.:"
        echo "  CUDA_HOME=/usr/local/cuda-12.9 ./run.sh $1"
        exit 1; }
    export CUDA_HOME="$CUDA_ROOT" CUDA_ROOT
}

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
RESULTS="$HERE/results"
RUN="$HERE/.run"                 # pid / current-csv pointers live here
PYTHON_EXPLICIT="${PYTHON+set}"
GPU_EXPLICIT="${GPU+set}"      # was GPU= given on this command line?
GPU="${GPU:-0}"

SIZES="${SIZES:-64,128,256,512}"
BLOCKS="${BLOCKS:-64,128,256,512,1024,2048,4096,8192,16384}"
ATTEMPTS="${ATTEMPTS:-2e7}"
TEMP="${TEMP:-30.0}"
REPEATS="${REPEATS:-3}"

mkdir -p "$RESULTS" "$RUN"
# shellcheck disable=SC1090
[ -f "$HERE/.run/env" ] && . "$HERE/.run/env"
PIDFILE="$RUN/bench.pid"
CSVPTR="$RUN/bench.csv.path"
EXITFILE="$RUN/bench.exit"
GPUPTR="$RUN/bench.gpu"
LOGPTR="$RUN/bench.log.path"

have_venv() { [ -x "$VENV/bin/python" ]; }
activate()  { have_venv && . "$VENV/bin/activate"; }

# A detached background process must not inherit the terminal. setsid puts it
# in a new session with no controlling tty, so the SIGHUP sent when the shell
# exits never reaches it; nohup is belt and braces; </dev/null stops it ever
# blocking on a read from a terminal that no longer exists.
#
# The child writes its OWN pid and then execs. Do not use $! here: when the
# background process is already a process-group leader, setsid forks and the
# parent exits immediately, so $! names a corpse and every later `kill -0`
# reports the run as dead while it is happily running.
spawn() {
    local log="$1" pidfile="$2"; shift 2
    local launcher=(); command -v setsid >/dev/null 2>&1 && launcher=(setsid)
    "${launcher[@]}" nohup bash -c '
        pidfile="$1"; exitfile="$2"; shift 2
        printf "%s\n" "$$" > "$pidfile"
        rm -f "$exitfile"
        "$@"; code=$?
        printf "%s %s\n" "$code" "$(date +%s)" > "$exitfile"
        exit $code' _ "$pidfile" "$EXITFILE" "$@" </dev/null >"$log" 2>&1 &
    # give the child a moment to write the file before anyone reads it
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        [ -s "$pidfile" ] && break
        sleep 0.3
    done
}

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-help}" in

setup)
    echo "repo: $REPO"
    cuda_or_die setup
    echo "nvcc: $(command -v nvcc)   ($(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/CUDA \1/p' | head -1))"
    if command -v nvidia-smi >/dev/null 2>&1; then
        DRV="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)"
        echo "driver supports up to CUDA $DRV"
        echo "  (if the toolkit above is NEWER than that, the driver will refuse"
        echo "   the binaries it produces - pin an older one with CUDA_HOME=)"
    fi
    pick_python
    echo "python: $PYTHON  ($("$PYTHON" -V 2>&1))"
    fix_compilers
    check_nvcc_hostcc || exit 1
    ARCH_AFTER_VENV=1
    if have_venv; then
        # a venv built on the wrong interpreter is the g++-12 trap; say so
        cur="$("$VENV/bin/python" -c 'import sys;print(sys.executable)' 2>/dev/null)"
        echo "reusing venv at $VENV ($cur)"
        echo "  if pycuda failed here before: rm -rf $VENV and re-run setup"
    else
        "$PYTHON" -m venv "$VENV"
    fi
    . "$VENV/bin/activate"
    pip install -q --upgrade pip wheel
    pip install -q numpy pandas matplotlib seaborn tqdm || exit 1
    echo "installing pycuda (compiles against the CUDA toolkit; slow) ..."
    pip install -q pycuda || {
        echo "pycuda failed. Almost always nvcc not on PATH, or no matching"
        echo "compiler. Check: which nvcc && nvcc --version && gcc --version"
        exit 1; }
    pip install -q "$REPO/"
    echo
    echo "setup done -> $VENV"
    "$VENV/bin/python" -c "import pycuda, numpy, construct" 2>/dev/null \
        && echo "imports OK" || echo "note: run ./run.sh preflight to see why imports fail"
    echo "next: ./run.sh preflight"
    ;;

preflight)
    cuda_or_die preflight
    activate || { echo "no venv at $VENV - run ./run.sh setup"; exit 1; }
    cd "$HERE"
    if nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>/dev/null; then
        :
    else
        echo "nvidia-smi unavailable (see NVML note below) - continuing, CUDA does not need it"
    fi
    check_arch || exit 1
    echo
    CUDA_VISIBLE_DEVICES="$GPU" python preflight_gpu.py
    ;;

smoke)
    cuda_or_die smoke
    activate || { echo "no venv - run ./run.sh setup"; exit 1; }
    cd "$HERE"
    check_arch || exit 1
    echo "quick sweep on GPU $GPU, output to /tmp/smoke.csv"
    CUDA_VISIBLE_DEVICES="$GPU" python bench_speed.py \
        --sizes 64 --blocks 64,256,1024 --attempts 2e6 --repeats 1 \
        --out /tmp/smoke.csv
    ;;

start|resume)
    cuda_or_die start
    activate || { echo "no venv - run ./run.sh setup"; exit 1; }
    if running; then
        echo "already running as pid $(cat "$PIDFILE"). ./run.sh status"
        exit 1
    fi
    cd "$HERE"
    check_arch || exit 1
    if [ "$1" = resume ] && [ -f "$CSVPTR" ]; then
        CSV="$(cat "$CSVPTR")"
        LOG="$(cat "$LOGPTR" 2>/dev/null || echo "${CSV%.csv}.log")"
        RESUME_ARG=(--resume "$CSV")
        # go back to the card the run started on, unless GPU= was given now
        if [ -f "$GPUPTR" ] && [ -z "${GPU_EXPLICIT:-}" ]; then
            GPU="$(cat "$GPUPTR")"
        fi
        echo "resuming into $CSV (GPU $GPU)"
    else
        STAMP="$(date +%Y%m%d-%H%M%S)"
        TAG="$(hostname -s)_gpu${GPU}_${STAMP}"
        CSV="$RESULTS/bench_speed_${TAG}.csv"
        LOG="$RESULTS/bench_speed_${TAG}.log"
        RESUME_ARG=(--resume "$CSV")     # harmless when the file does not exist
        echo "new run -> $CSV"
    fi
    echo "$CSV" > "$CSVPTR"; echo "$LOG" > "$LOGPTR"; echo "$GPU" > "$GPUPTR"

    rm -f "$PIDFILE" "$EXITFILE"
    spawn "$LOG" "$PIDFILE" \
        env "CUDA_VISIBLE_DEVICES=$GPU" "PATH=$PATH" \
            "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" "CUDA_HOME=${CUDA_HOME:-}" \
            "PYCUDA_DEFAULT_NVCC_FLAGS=${PYCUDA_DEFAULT_NVCC_FLAGS:-}" \
        "$VENV/bin/python" -u "$HERE/bench_speed.py" \
            --sizes "$SIZES" --blocks "$BLOCKS" --attempts "$ATTEMPTS" \
            --temp "$TEMP" --repeats "$REPEATS" \
            --out "$CSV" "${RESUME_ARG[@]}"
    [ -s "$PIDFILE" ] || { echo "child never reported a pid; log:"; tail -20 "$LOG"; exit 1; }
    PID="$(cat "$PIDFILE")"
    sleep 2
    if kill -0 "$PID" 2>/dev/null; then
        echo "started, pid $PID on GPU $GPU"
        echo "  ./run.sh status      ./run.sh log      ./run.sh stop"
        echo
        echo "Safe to close the terminal or log out now - the process has its"
        echo "own session and no controlling terminal."
    else
        echo "died immediately. Last lines of $LOG:"; tail -20 "$LOG"; exit 1
    fi
    ;;

status)
    CSV=""; [ -f "$CSVPTR" ] && CSV="$(cat "$CSVPTR")"
    LOG=""; [ -f "$LOGPTR" ] && LOG="$(cat "$LOGPTR")"

    if running; then
        VERDICT="RUNNING   pid $(cat "$PIDFILE")"
    elif [ -f "$EXITFILE" ]; then
        read -r CODE WHEN < "$EXITFILE"
        if [ "$CODE" = 0 ]; then
            VERDICT="COMPLETED  (exit 0, $(date -d "@$WHEN" 2>/dev/null || echo "at $WHEN"))"
        else
            VERDICT="FAILED     (exit $CODE) - read the log, then ./run.sh resume"
        fi
    elif [ -n "$LOG" ]; then
        VERDICT="DIED       (no exit status recorded: killed, OOM, or reboot)
           nothing measured is lost - ./run.sh resume"
    else
        VERDICT="no run yet"
    fi
    echo "$VERDICT"

    if [ -n "$CSV" ] && [ -f "$CSV" ]; then
        echo "csv       $CSV"
        echo "measured  $(($(wc -l < "$CSV") - 1)) points"
    fi
    if [ -n "$LOG" ] && [ -f "$LOG" ]; then
        # bench_speed prints "[n/total] ..." per point
        PROG="$(grep -o '\[[0-9]\+/[0-9]\+\]' "$LOG" | tail -1 | tr -d '[]')"
        [ -n "$PROG" ] && echo "progress  ${PROG%/*} of ${PROG#*/} points"
        echo "--- last 6 log lines ---"
        tail -6 "$LOG"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "--- gpu ---"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    fi
    exit 0
    ;;

wait)
    # Block until the run ends, then say how it ended. Handy over ssh:
    #   ./run.sh wait && ./run.sh analyze
    running || { echo "not running"; exit 0; }
    echo "waiting for pid $(cat "$PIDFILE") ..."
    while running; do sleep 30; done
    if [ -f "$EXITFILE" ]; then
        read -r CODE WHEN < "$EXITFILE"
        [ "$CODE" = 0 ] && echo "COMPLETED" || { echo "FAILED (exit $CODE)"; exit "$CODE"; }
    else
        echo "DIED without recording an exit status - ./run.sh resume"; exit 1
    fi
    ;;

log)
    [ -f "$LOGPTR" ] || { echo "no run yet"; exit 1; }
    tail -f "$(cat "$LOGPTR")"
    ;;

stop)
    running || { echo "not running"; exit 0; }
    PID="$(cat "$PIDFILE")"
    # setsid made the wrapper a session leader, so pid == process-group id:
    # kill the GROUP or python survives its parent.
    kill -TERM -"$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null
    echo "sent TERM to process group $PID"
    sleep 3
    if kill -0 "$PID" 2>/dev/null; then
        kill -9 -"$PID" 2>/dev/null || kill -9 "$PID" 2>/dev/null
        echo "SIGKILLed"
    fi
    rm -f "$PIDFILE"
    echo "the CSV is fsynced per row - ./run.sh resume picks up where it stopped"
    ;;

analyze)
    activate || { echo "no venv - run ./run.sh setup"; exit 1; }
    cd "$HERE"
    shopt -s nullglob
    CSVS=("$RESULTS"/bench_speed_*.csv)
    [ ${#CSVS[@]} -gt 0 ] || { echo "no CSVs in $RESULTS"; exit 1; }
    echo "analyzing: ${CSVS[*]}"; echo
    python analyze_bench.py "${CSVS[@]}"
    ;;

*)
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;
esac
