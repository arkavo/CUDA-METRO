#!/bin/bash
#SBATCH --job-name=msu-cpu-scaling
#SBATCH --output=bench_cpu_%j.out
#SBATCH --error=bench_cpu_%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64      # ask for every core you intend to test
#SBATCH --mem=16G
# no --gres: this job needs no GPU, so it queues on the general partition
# and will usually start far sooner than a GPU job.

set -euo pipefail

module load python/3.12 2>/dev/null || true
python -m venv "$SLURM_TMPDIR/venv" 2>/dev/null || python -m venv "$HOME/msu-venv"
source "${SLURM_TMPDIR:-$HOME}/venv/bin/activate" 2>/dev/null || source "$HOME/msu-venv/bin/activate"
pip install -q --upgrade pip
pip install -q numpy numba

# One thread pool sized to the allocation; numba.set_num_threads() then walks
# DOWN from it inside the benchmark. Setting it below the max you want to test
# silently truncates the sweep.
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Keep BLAS out of the way - it is not used here, but its pool competes for cores.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

THREADS=$(python - <<'EOF'
import os
n = int(os.environ["SLURM_CPUS_PER_TASK"])
t, out = 1, []
while t <= n:
    out.append(t); t *= 2
if out[-1] != n:
    out.append(n)
print(",".join(map(str, out)))
EOF
)

echo "allocation: $SLURM_CPUS_PER_TASK cores | sweeping threads: $THREADS"

python bench_cpu.py \
    --threads "$THREADS" \
    --L 256 \
    --P 16384 \
    --attempts 4e8 \
    --repeats 3 \
    --out "bench_cpu_${SLURM_JOB_ID}.csv"
