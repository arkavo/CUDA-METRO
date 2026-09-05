# benchmarks/

Speed and scaling measurements for CUDA-METRO's parallel Metropolis update.

The accuracy side of the multi-spin-update question is settled elsewhere: the
error of a P-site synchronous update against single-update Metropolis follows
`ε ≈ 0.0276 (P/N)` in the ordered phase, fitted across a 64× range of lattice
size. This folder measures the other half — what the parallelism actually buys
in wall-clock — over the **same (N, P) grid**, so the two combine into a
speed–accuracy frontier rather than two unrelated plots.

## Layout

```
benchmarks/
  run.sh                THE RUNNER for a machine you control: setup/preflight/
                        smoke/start/status/log/stop/resume/analyze
  _repo.py              path bootstrap; import it and `import construct` works from any cwd
  check_slurm.sh        SLURM only: can you submit a GPU job at all?
  preflight_gpu.py      7 checks in seconds; gate the sweep on this
  bench_speed.py        the GPU sweep: wall-clock vs Blocks (=P) vs lattice size
  submit_bench.slurm    SLURM only: batch job (runs preflight first)
  deploy_gpu.sh         SLURM only: setup / preflight / run / status / analyze
  run_detached.sh       SLURM only: alternative launcher
  analyze_bench.py      measured speed × fitted error law → frontier + usable-P ceiling
  cpu/                  shared-memory CPU thread-scaling variant (no GPU needed)
```

Nothing here is copied into `src/cudametro/`. `_repo.py` puts `<repo>/src` and
`<repo>/src/cudametro` on `sys.path`, so there is exactly one copy of every
script and it is the one that runs. Scripts work from any working directory.

## Quick start — machine you control, no scheduler

```bash
git clone https://github.com/arkavo/CUDA-METRO.git && cd CUDA-METRO/benchmarks
chmod +x *.sh

./run.sh setup          # venv + pycuda + the package
./run.sh preflight      # must end: PREFLIGHT PASSED
./run.sh smoke          # ~2 min, prints to the terminal
./run.sh start          # full sweep, detached — close the terminal
./run.sh status         # alive? how many points measured?
./run.sh analyze
```

Pick a card with `GPU=2 ./run.sh start`. Narrow the sweep with environment
variables: `SIZES=64,128 BLOCKS=64,256,1024 ATTEMPTS=2e6 REPEATS=1`.

If a run dies — power cut, driver reset, someone else's job — `./run.sh
resume` restarts it on the same card and skips every `(size, P, repeat)`
already in the CSV. Rows are `fsync`ed as they are written, so nothing
measured is ever lost.

### Why `start` really survives logout

`spawn()` launches the child under `setsid`, giving it a new session with no
controlling terminal, so the `SIGHUP` your shell sends on exit never reaches
it. `nohup` is belt and braces, and `</dev/null` keeps it from ever blocking
on a read from a terminal that no longer exists.

The child writes its **own** PID and then `exec`s. `$!` is wrong here: when
the background process is already a process-group leader, `setsid` forks and
the parent exits immediately, so `$!` names a corpse and every later
`kill -0` would report a healthy run as dead.

## If you are on a shared, scheduled cluster instead

`check_slurm.sh`, `submit_bench.slurm`, `deploy_gpu.sh` and `run_detached.sh`
are the SLURM path, kept for portability. `sbatch` is already detached — the
scheduler owns the job, so do not wrap it in `nohup`. Never use `run.sh` or
`run_detached.sh local` inside an `salloc`/`srun --pty` allocation: most sites
tear the allocation's cgroup down when the session ends and kill every process
in it regardless of `setsid`.

## What the sweep measures, and why each choice

- **Total flip-attempts held constant** across every (N, P) point, so
  wall-clock differences are pure speedup rather than a different amount of
  work. A useful consequence: `C1 = Blocks × stability_runs` equals
  `--attempts` at every point, so the RNG buffers are the same size throughout
  the sweep (~0.67 GB at the 2e7 default).
- **CUDA events, not `datetime`.** Kernel launches are asynchronous; host-side
  wall-clock measures how fast you can enqueue, not how fast the GPU executes.
- **Host-side RNG timed separately.** It is a real cost, but it is not the
  parallel update, and at large P it can dominate. One combined number hides
  that.
- **CSV flushed per row**, so a job killed at the time limit still leaves every
  point it managed to measure.
- **`del m; gc.collect()` each iteration.** The sweep allocates hundreds of MB
  per point; left to chance, a long sweep OOMs a card that had room for it.

The kernel-call block in `bench_speed.py` is copied verbatim from
`construct.run_mc_tc_3636` — argument order, slice arithmetic, block/grid
shapes — so this times the code path the science actually runs, not a
re-implementation of it.

### The β trap

`mc_init()` only **allocates** `BJ` (`drv.mem_alloc`). The temperature is
written by `run_mc_tc_3636` via `memcpy_htod(self.BJ, beta)`. Any script that
bypasses `run_mc_tc_3636` to time the kernel loop directly — as this one does —
must do that memcpy itself, or the kernel reads uninitialised device memory as
inverse temperature. It does not crash. It silently changes the acceptance
rate, and therefore the write traffic to `GPU_TRANS` and the branch divergence
inside the kernel, so the timings are of the wrong thing and look plausible.
`bench_speed.py` writes β explicitly; `--temp` defaults to 30 K and is recorded
in every CSV row. `preflight_gpu.py` catches the failure by asserting the
lattice actually changes.

## What `analyze_bench.py` tells you

Two ceilings bound the useful P, and the smaller one wins:

- **Hardware.** Speedup stops growing once the GPU saturates. The launch
  geometry is one lattice site per **block** with `Threads = 2`, so occupancy is
  bounded by resident *blocks* per SM, not by CUDA core count.
  `preflight_gpu.py` prints `maxBlocks/SM × SMs` — that product is where the
  curve should flatten.
- **Physics.** For a target accuracy A, `P/N ≤ (1 − A)/0.0276`.

It reports which one binds per lattice size and writes `frontier.png`.

**The result worth watching for:** if saturation arrives near
`maxBlocks/SM × SMs` rather than near the core count, then "a GPU has a fixed
core count, so P is bought once" is the wrong framing — the binding constraint
is launch geometry, and the fix is a different kernel decomposition (multiple
sites per block) rather than a bigger card.

## cpu/

A shared-memory CPU version of the same update: the DECIDE loop is a bare
`prange` because no site's acceptance depends on another's, and COMMIT is a
serial last-write-wins pass matching the GPU's `cp_grid`. Validated against the
identical serial baseline (M = 1.0689 ± 0.0037 vs 1.0755 ± 0.0017, inside 1.6σ).

It demonstrates that the update parallelises and fits an Amdahl serial
fraction. It is **not** a substitute for the GPU numbers: a CPU reaches tens of
threads where CUDA-METRO runs P into the thousands, and none of the
GPU-specific ceilings appear. `cpu/submit_bench_cpu.slurm` asks for no GPU, so
it queues on the general partition and usually starts far sooner.

## Fixed in this branch

Three bugs in `src/cudametro/bench.py`, verified present before fixing:

1. `bench_MC.save_direcotry` on the last line. `construct.py` spells it
   `save_directory`; the misspelling survives only under `deprecated/`. The
   line raised `AttributeError` *after* all the timing work, so every result
   was lost. **`curie_temp.py` had the same typo in four places** (lines 19,
   20, 26, 30) and was losing its output the same way. Both corrected.
2. `bench.py` defines `sizes = [64,128,256,512]` and never uses it — only
   `size = 256` was ever benchmarked. `bench_speed.py` sweeps sizes properly.
3. `bench.py` passes `config="../configs/bench.json"`, which from
   `src/cudametro/` resolves to a non-existent `src/configs/`. Handled here by
   `_repo.config()`, which is cwd-independent.
