#!/bin/bash
# Launch the CUDA-METRO speed benchmark so it survives you closing the terminal.
#
#   ./run_detached.sh              # auto: sbatch if SLURM is there, else nohup
#   ./run_detached.sh slurm        # force the scheduler path
#   ./run_detached.sh local        # force the nohup path (interactive GPU node)
#
# Both paths are genuinely detached: the process is reparented away from your
# login shell and ignores SIGHUP, so logging out cannot kill it.
#
# Edit these three to match your site, then never touch them again.
REPO=~/CUDA-METRO
VENV=~/venv-cudametro
RESULTS=~/bench-results

set -euo pipefail
MODE="${1:-auto}"
mkdir -p "$RESULTS"
STAMP=$(date +%Y%m%d-%H%M%S)

if [ "$MODE" = auto ]; then
    if command -v sbatch >/dev/null 2>&1; then MODE=slurm; else MODE=local; fi
fi

# ---------------------------------------------------------------- SLURM ----
if [ "$MODE" = slurm ]; then
    JOB=$(sbatch --parsable submit_bench.slurm)
    cat <<EOF
submitted as job $JOB

The scheduler owns it now. Close the terminal, log out, shut the laptop —
none of that touches a queued or running batch job. That is the whole point
of sbatch; nohup would be redundant here.

  squeue -j $JOB                 # queued / running / gone
  sacct  -j $JOB --format=JobID,State,Elapsed,MaxRSS,ExitCode
  tail -f bench_${JOB}.out       # live log once it starts
  scancel $JOB                   # stop it

Results land in \$SLURM_SUBMIT_DIR as bench_speed_${JOB}.csv, written and
flushed row by row — a job killed at the time limit still leaves you every
point it managed to measure.
EOF
    exit 0
fi

# ---------------------------------------------------------------- local ----
# For an interactive GPU node (salloc/srun --pty) or a plain GPU box.
LOG="$RESULTS/bench_${STAMP}.log"
CSV="$RESULTS/bench_speed_${STAMP}.csv"
PIDFILE="$RESULTS/bench_${STAMP}.pid"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$REPO/benchmarks"             # _repo.py handles sys.path

# setsid  -> new session, no controlling terminal, cannot receive the tty's SIGHUP
# nohup   -> belt and braces if setsid is missing
# < /dev/null -> never blocks on a read from a terminal that no longer exists
setsid nohup python -u bench_speed.py \
    --sizes 64,128,256,512 \
    --blocks 64,128,256,512,1024,2048,4096,8192,16384 \
    --attempts 2e7 \
    --repeats 3 \
    --out "$CSV" \
    < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PIDFILE"
disown "$PID" 2>/dev/null || true

cat <<EOF
started detached, pid $PID

  tail -f $LOG
  ps -p $PID                     # still alive?
  kill $PID                      # stop it

CSV (flushed per row, safe to read while running):
  $CSV

Caveat you should know rather than discover: on many clusters an interactive
allocation is torn down when your srun/salloc session ends, and the scheduler
kills every process in the cgroup regardless of setsid. If this node came from
salloc, use the slurm path instead — sbatch is the only genuinely durable one.
EOF
