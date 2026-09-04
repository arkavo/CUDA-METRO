# AGENTS.md — automation contract for benchmarks/

Instructions for an agent driving these benchmarks on a cluster. Read
`README.md` first for what the measurement is; this file is only about running
it without a human watching.

## Ground rules

1. **Never edit files under `src/cudametro/` to make a benchmark work.**
   `_repo.py` puts `<repo>/src` and `<repo>/src/cudametro` on `sys.path`.
   If an import fails, the fix is the path bootstrap or the venv, never a copy
   of a script into the package. Two copies means one goes stale silently.
2. **Never submit without a green preflight.** `preflight_gpu.py` exits 0 or
   non-zero and is cheap. `submit_bench.slurm` runs it first by design; do not
   remove that line to "save time".
3. **Do not tune parameters to make a run finish.** If a sweep OOMs, reduce
   `--attempts` and say so in the report — do not silently drop lattice sizes
   or block counts, because the (N, P) grid is what makes the results
   comparable with the accuracy study.
4. **Results are append-only.** The CSV is opened in append mode with a header
   written once. Never delete or rewrite a CSV; start a new one.
5. **Report negative results.** A flat speedup curve is a finding, not a
   failure to be retried until it looks better.

## Environment assumptions

- `sbatch` present → use it. It is already detached; the scheduler owns the
  job and terminal loss is irrelevant. Do not wrap it in `nohup`.
- `sbatch` absent → `./run_detached.sh local`. Do **not** use this inside an
  `salloc`/`srun --pty` allocation: the cgroup is torn down when the session
  ends and `setsid` will not save the process.
- Site-specific values live at the top of `deploy_gpu.sh` (`REPO`, `VENV`,
  `RESULTS`, `CUDA_MODULE`, `PY_MODULE`) and in the `#SBATCH` header of
  `submit_bench.slurm` (`--partition`, `--gres`, `--time`). These are the only
  lines an agent should edit, and it should say which it changed.

## Happy path

```bash
cd <repo>/benchmarks && chmod +x *.sh
bash check_slurm.sh                          # capture output into the report
./deploy_gpu.sh setup
srun --gres=gpu:1 --time=00:15:00 python preflight_gpu.py   # must exit 0
./deploy_gpu.sh run                          # capture the job id
# poll; when State=COMPLETED:
./deploy_gpu.sh analyze
```

## Failure playbook

| symptom | cause | action |
|---|---|---|
| `ModuleNotFoundError: construct` | not run from `benchmarks/`, or `_repo.py` missing | run from `benchmarks/`; confirm it sits at the repo root beside `src/` |
| `ModuleNotFoundError: pycuda` | wrong venv | `source $VENV/bin/activate`; if it is not installed, `which nvcc` before blaming pip |
| pycuda build fails in `setup` | CUDA toolkit not on PATH | `module load` the CUDA module, check `which nvcc`, then retry |
| `no nvidia-smi` in preflight | on a login node | get a GPU: `srun --gres=gpu:1 --time=00:15:00 --pty bash` |
| preflight: "the lattice did not change" | β unwritten, or every move rejected | do **not** proceed. Check `--temp`; the sweep is meaningless without this |
| `cuda.MemoryError` mid-sweep | RNG buffers too big | lower `--attempts` (buffers ≈ `attempts × 36 bytes`); record the new value |
| job hits the time limit | sweep too long | the CSV is flushed per row, so **keep it**; resubmit the remaining points with narrowed `--sizes`/`--blocks` and append |
| `AttributeError` on any `m.<attr>` | repo changed under us | stop. Re-run preflight, report which attribute vanished; do not guess a replacement |
| `Invalid account / association` from sbatch | no GPU allocation on this account | `check_slurm.sh` output has the associations; escalate to a human |

## What to report back

Always, whether it worked or not:

- the `check_slurm.sh` summary (partitions, gres, QoS limits)
- GPU model, SM count, and the `maxBlocks/SM × SMs` product from preflight
- job id, final `sacct` State, Elapsed, ExitCode
- the CSV path and its row count
- `analyze_bench.py` stdout in full — specifically, for each lattice size, the
  P at which parallel efficiency drops below 50%, and whether the hardware or
  the accuracy ceiling binds
- any parameter you changed from the defaults, and why

## The one interpretive question

Compare the P where parallel efficiency falls below 50% against
`maxBlocks/SM × SMs` from preflight.

- **They agree** → the launch geometry (one site per block, `Threads = 2`) is
  the binding constraint, not CUDA core count. This contradicts the framing
  "a GPU has a fixed core count, so P is bought once", and the remedy is a
  kernel decomposition with multiple sites per block, not a larger card.
- **Efficiency holds well past that product** → occupancy is not the limit;
  look at memory traffic (`GRID_COPY` writes) or the RNG share of total time,
  both of which are already columns in the CSV.

Flag whichever it is explicitly. Do not average the two or hedge — this is the
single most informative number the run produces.
