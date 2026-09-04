#!/bin/bash
#SBATCH --job-name=cudametro-bench
#SBATCH --partition=gpu              # <-- your cluster's GPU partition
#SBATCH --gres=gpu:1                 # one GPU is all this needs
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.err

set -euo pipefail

# NOTE ON DETACHMENT: an sbatch job is already fully detached. The scheduler,
# not your shell, owns it — closing the terminal, logging out, or losing the
# VPN cannot touch a queued or running batch job. Do not wrap this in nohup;
# that would only detach it from a shell it was never attached to.

# --- environment -------------------------------------------------------
# Names vary by site; `module avail cuda` and `module spider python` first.
module load cuda/12.2
module load python/3.11

VENV=~/venv-cudametro
if [ ! -d "$VENV" ]; then
    echo "creating $VENV (first run only)"
    python -m venv "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    pip install -q --upgrade pip
    pip install -q pycuda numpy pandas matplotlib seaborn tqdm
    pip install -q "$SLURM_SUBMIT_DIR/.."
else
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo

# --- run ---------------------------------------------------------------
# Run from benchmarks/ - _repo.py puts src/ and src/cudametro/ on sys.path,
# so nothing is ever copied into the package and no copy can go stale.
cd "$SLURM_SUBMIT_DIR"

# Gate the sweep on the preflight: if an attribute name or the config path is
# wrong, fail in seconds rather than after the sweep has burned the allocation.
python preflight_gpu.py

python bench_speed.py \
  --sizes 64,128,256,512 \
  --blocks 64,128,256,512,1024,2048,4096,8192,16384 \
  --attempts 2e7 \
  --temp 30.0 \
  --repeats 3 \
  --out "$SLURM_SUBMIT_DIR/bench_speed_${SLURM_JOB_ID}.csv"

echo "done"
