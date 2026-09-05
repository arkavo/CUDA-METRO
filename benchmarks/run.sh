#!/usr/bin/env bash
# Raw runner for the CUDA-METRO speed benchmark. No scheduler, no queue.
# For a machine you control: log in, start it, close the terminal, come back.
#
#   ./run.sh setup            create the venv and install everything
#   ./run.sh preflight        7 checks in seconds; do this before a long run
#   ./run.sh smoke            ~2 min sanity sweep, prints to the terminal
#   ./run.sh start            full sweep, DETACHED (survives logout)
#   ./run.sh status           alive? how far along?
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
# -----------------------------------------------------------------------

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
RESULTS="$HERE/results"
RUN="$HERE/.run"                 # pid / current-csv pointers live here
GPU_EXPLICIT="${GPU+set}"      # was GPU= given on this command line?
GPU="${GPU:-0}"

SIZES="${SIZES:-64,128,256,512}"
BLOCKS="${BLOCKS:-64,128,256,512,1024,2048,4096,8192,16384}"
ATTEMPTS="${ATTEMPTS:-2e7}"
TEMP="${TEMP:-30.0}"
REPEATS="${REPEATS:-3}"

mkdir -p "$RESULTS" "$RUN"
PIDFILE="$RUN/bench.pid"
CSVPTR="$RUN/bench.csv.path"
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
        printf "%s\n" "$$" > "$1"; shift
        exec "$@"' _ "$pidfile" "$@" </dev/null >"$log" 2>&1 &
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
    command -v nvcc >/dev/null 2>&1 \
        && echo "nvcc: $(command -v nvcc)" \
        || echo "nvcc: NOT ON PATH  <- pycuda will fail to build; fix this first"
    have_venv || "$PYTHON" -m venv "$VENV"
    . "$VENV/bin/activate"
    pip install -q --upgrade pip wheel
    pip install -q numpy pandas matplotlib || exit 1
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
    activate || { echo "no venv at $VENV - run ./run.sh setup"; exit 1; }
    cd "$HERE"
    command -v nvidia-smi >/dev/null 2>&1 || { echo "no nvidia-smi on this host"; exit 1; }
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
    echo
    CUDA_VISIBLE_DEVICES="$GPU" python preflight_gpu.py
    ;;

smoke)
    activate || { echo "no venv - run ./run.sh setup"; exit 1; }
    cd "$HERE"
    echo "quick sweep on GPU $GPU, output to /tmp/smoke.csv"
    CUDA_VISIBLE_DEVICES="$GPU" python bench_speed.py \
        --sizes 64 --blocks 64,256,1024 --attempts 2e6 --repeats 1 \
        --out /tmp/smoke.csv
    ;;

start|resume)
    activate || { echo "no venv - run ./run.sh setup"; exit 1; }
    if running; then
        echo "already running as pid $(cat "$PIDFILE"). ./run.sh status"
        exit 1
    fi
    cd "$HERE"
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

    rm -f "$PIDFILE"
    spawn "$LOG" "$PIDFILE" \
        env "CUDA_VISIBLE_DEVICES=$GPU" \
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
    if running; then
        echo "RUNNING  pid $(cat "$PIDFILE")"
    else
        echo "not running"
    fi
    if [ -f "$CSVPTR" ]; then
        CSV="$(cat "$CSVPTR")"
        if [ -f "$CSV" ]; then
            echo "csv      $CSV"
            echo "measured $(($(wc -l < "$CSV") - 1)) points"
        fi
    fi
    [ -f "$LOGPTR" ] && { echo "--- last 8 log lines ---"; tail -8 "$(cat "$LOGPTR")"; }
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "--- gpu ---"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    fi
    exit 0
    ;;

log)
    [ -f "$LOGPTR" ] || { echo "no run yet"; exit 1; }
    tail -f "$(cat "$LOGPTR")"
    ;;

stop)
    running || { echo "not running"; exit 0; }
    PID="$(cat "$PIDFILE")"
    kill "$PID" && echo "sent TERM to $PID"
    sleep 3
    kill -0 "$PID" 2>/dev/null && { kill -9 "$PID"; echo "SIGKILLed"; }
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
