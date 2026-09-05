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

Default is the **raw runner** `./run.sh` — a machine the user controls, no
scheduler. It detaches with `setsid`, the child reports its own PID, and the
CSV is `fsync`ed per row so `./run.sh resume` is always safe.

- Never poll faster than once a minute; `./run.sh status` is cheap but the log
  is the thing to read, not re-run.
- After any abnormal exit, `./run.sh resume` — never `start`, which opens a new
  CSV and re-measures everything.
- `GPU=n` selects a card. If several are free, one sweep per card with
  different `GPU=` values is fine; they write separate CSVs and
  `analyze_bench.py` takes them all.
- Only if the site actually has SLURM: `sbatch submit_bench.slurm`. That is
  already detached — the scheduler owns it; do not wrap it in `nohup`. Never
  run `./run.sh start` inside an `salloc`/`srun --pty` allocation: the cgroup
  is torn down when the session ends and `setsid` will not save it.
- Site-specific values live at the top of `deploy_gpu.sh` (`REPO`, `VENV`,
  `RESULTS`, `CUDA_MODULE`, `PY_MODULE`) and in the `#SBATCH` header of
  `submit_bench.slurm` (`--partition`, `--gres`, `--time`). These are the only
  lines an agent should edit, and it should say which it changed.

## Happy path

```bash
cd <repo>/benchmarks && chmod +x *.sh
./run.sh setup
./run.sh preflight                 # must exit 0 and print PREFLIGHT PASSED
./run.sh smoke                     # must produce 3 sane rows
./run.sh start
# poll ./run.sh status until "not running" AND the log ends with "wrote <csv>"
./run.sh analyze
```

"not running" alone does not mean success — a crash also stops the process.
Confirm the log's last line is `wrote <csv>` before reporting completion.

## Failure playbook

| symptom | cause | action |
|---|---|---|
| `ModuleNotFoundError: construct` | not run from `benchmarks/`, or `_repo.py` missing | run from `benchmarks/`; confirm it sits at the repo root beside `src/` |
| `ModuleNotFoundError: pycuda` | wrong venv | `source $VENV/bin/activate`; if it is not installed, `which nvcc` before blaming pip |
| pycuda build fails in `setup` | CUDA toolkit not on PATH | `module load` the CUDA module, check `which nvcc`, then retry |
| `no nvidia-smi` on this host | wrong machine, or driver down | check `nvidia-smi` by hand; do not proceed |
| process vanished mid-sweep | OOM killer, driver reset, reboot | `./run.sh resume` — never `start` |
| preflight: "the lattice did not change" | β unwritten, or every move rejected | do **not** proceed. Check `--temp`; the sweep is meaningless without this |
| `cuda.MemoryError` mid-sweep | RNG buffers too big | lower `--attempts` (buffers ≈ `attempts × 36 bytes`); record the new value |
| run interrupted for any reason | anything | `./run.sh resume`; rows are fsynced, nothing measured is lost |
| `AttributeError` on any `m.<attr>` | repo changed under us | stop. Re-run preflight, report which attribute vanished; do not guess a replacement |
| `Invalid account / association` from sbatch | no GPU allocation on this account | `check_slurm.sh` output has the associations; escalate to a human |

## What to report back

Always, whether it worked or not:

- host name, GPU model, SM count, and the `maxBlocks/SM × SMs` product
  from preflight
- the PID, whether the log ends with `wrote <csv>`, and any resume you did
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
