#!/bin/bash
# One-shot deploy of the CUDA-METRO speed benchmark onto a GPU cluster.
#
#   ./deploy_gpu.sh setup      # clone/venv/install, once per cluster
#   ./deploy_gpu.sh preflight  # verify on a GPU node before spending a queue slot
#   ./deploy_gpu.sh run        # sbatch the sweep (detached; scheduler owns it)
#   ./deploy_gpu.sh status     # what is my job doing
#   ./deploy_gpu.sh analyze    # frontier from whatever CSV exists
#
# --- edit these four, then never again ---------------------------------
REPO=~/CUDA-METRO
VENV=~/venv-cudametro
RESULTS=~/bench-results
CUDA_MODULE="cuda/12.2"          # `module avail cuda` to find yours
PY_MODULE="python/3.11"          # `module spider python`
# -----------------------------------------------------------------------

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD="${1:-help}"

load_modules() {
    if command -v module >/dev/null 2>&1; then
        module load "$CUDA_MODULE" 2>/dev/null || echo "note: $CUDA_MODULE not loadable, continuing"
        module load "$PY_MODULE"   2>/dev/null || echo "note: $PY_MODULE not loadable, continuing"
    fi
}

case "$CMD" in

setup)
    load_modules
    if [ ! -d "$REPO" ]; then
        git clone https://github.com/arkavo/CUDA-METRO.git "$REPO"
    fi
    if [ ! -d "$VENV" ]; then
        python -m venv "$VENV"
    fi
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    pip install -q --upgrade pip wheel
    # pycuda builds against the CUDA toolkit the module just put on PATH.
    # If this fails, the usual cause is nvcc not being visible - check
    # `which nvcc` before blaming pip.
    pip install -q pycuda numpy pandas matplotlib seaborn tqdm
    pip install -q "$REPO/"    # editable is fine too: pip install -e "$REPO/"
    mkdir -p "$RESULTS"
    echo
    echo "setup done."
    echo "  repo   $REPO"
    echo "  venv   $VENV"
    echo "  nvcc   $(command -v nvcc || echo 'NOT ON PATH - pycuda will have failed')"
    echo
    echo "next:  ./deploy_gpu.sh preflight     (needs a GPU: srun --gres=gpu:1 --pty bash)"
    ;;

preflight)
    load_modules
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    cd "$HERE"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "no nvidia-smi here - you are probably on a login node."
        echo "get a GPU first:  srun --gres=gpu:1 --time=00:15:00 --pty bash"
        exit 1
    fi
    python preflight_gpu.py
    ;;

run)
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch not found. Without a scheduler, use:"
        echo "  ./run_detached.sh local"
        exit 1
    fi
    cd "$HERE"
    JOB=$(sbatch --parsable submit_bench.slurm)
    echo "submitted job $JOB"
    echo
    echo "The scheduler owns it. Close the terminal, log out, shut the laptop -"
    echo "none of that touches a queued or running batch job."
    echo
    echo "  squeue -j $JOB"
    echo "  tail -f bench_${JOB}.out"
    echo "  scancel $JOB"
    ;;

status)
    squeue -u "$USER" -o "%.10i %.20j %.9P %.8T %.10M %.6D %R" || true
    echo
    sacct -u "$USER" --starttime today \
          --format=JobID%12,JobName%22,State%12,Elapsed,MaxRSS,ExitCode 2>/dev/null || true
    ;;

analyze)
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    cd "$HERE"
    shopt -s nullglob
    CSVS=(results/bench_speed_*.csv bench_speed_*.csv "$RESULTS"/bench_speed_*.csv)
    if [ ${#CSVS[@]} -eq 0 ]; then
        echo "no bench_speed_*.csv found in $HERE/results, $HERE, or $RESULTS"
        exit 1
    fi
    echo "analyzing: ${CSVS[*]}"
    python analyze_bench.py "${CSVS[@]}"
    ;;

*)
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;
esac
