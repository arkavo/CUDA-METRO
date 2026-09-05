# benchmarks/cpu/

Two independent CPU measurements. Both are pure numba — no GPU, no CUDA
toolkit, nothing to patch.

| file | what it measures |
|---|---|
| `critical_scaling_mp.py` | **accuracy at criticality**: does `P*/N` stay size-independent near `T_c`? |
| `bench_cpu.py` + `parallel_cpu_mc.py` | **thread scaling** of the synchronous update on a shared-memory machine |
| `fast_mc.py` | the Ising engine both accuracy studies use (`run_su`, `run_mu_sync`) |
| `plot_cpu_scaling.py` | turns the thread-scaling CSV into a figure |

## The critical-scaling run (the one that matters right now)

```bash
pip install numpy numba
python critical_scaling_mp.py --jobs $(nproc) \
       --sizes 8,12,16,24,32,48 --betas 0.40,0.4407
```

Resumable: rerun after any interruption and finished points are skipped.
Results append to `critical_size_scaling.csv`.

**Cost.** Critical slowing down makes this scale as `L^(2+z)` with z ≈ 2.17,
so L=48 alone is 80% of the work:

| L | sweeps/chain | G spin-updates |
|---|---|---|
| 8 | 80,000 | 0.26 |
| 16 | 360,018 | 4.6 |
| 24 | 867,846 | 17.5 |
| 32 | 1,620,168 | 58 |
| 48 | 3,905,514 | 315 |

0.79 T spin-updates for both temperatures. At ~11 M updates/s/core that is
~19 min on 64 cores, ~75 min on 16. Drop `--sizes 48` if you want it quick,
or `--tau-mult 400` to halve the equilibration (the control will tell you if
that was too aggressive).

**It also reports CPU throughput** in ns/spin-update, aggregate and per core —
directly comparable to the GPU numbers (3.2 ns/update Blackwell, 8.0 ns/update
V100). Note the GPU figure is for the 3-vector Heisenberg kernel with four
neighbour shells; this engine is Ising and does far less work per update, so
the comparison is indicative of the machines, not a like-for-like algorithmic
benchmark.

## Getting the output back out

The run ends with a full analysis printed to the terminal — the same style as
`analyze_bench.py` on the GPU side: every (β, L) accuracy table with the
control z-score, the interpolated `P*/N` per size (explicitly flagged when a
size was **censored**, i.e. still above 95% at the largest P tried, so it is
not a limit), a size-independence verdict, and the extrapolation to production
lattices against both measured GPU ceilings. Copy the whole block out; no CSV
needed.

It reads the CSV, not the in-memory results, so a resumed or partial run still
prints a complete summary. To re-print it later without recomputing anything:

```bash
python critical_scaling_mp.py --report-only
```

Redirect if you would rather not scroll:

```bash
python -u critical_scaling_mp.py --jobs $(nproc) \
       --sizes 8,12,16,24,32,48 --betas 0.40,0.4407 2>&1 | tee critical_run.log
```

## Why the question exists

The speed argument concludes: at production lattice sizes the hardware ceiling
(`P* = L·T`, independent of N) binds long before the accuracy budget
(`P ≤ (1−A)/0.0276 × N`, proportional to N). But 0.0276 is fitted in the
**ordered phase**. Near `T_c` the usable fraction collapses — at L=8 an earlier
sweep found `P*/N ≈ 0.12` at β=0.40 against ≈0.74 deep in the ordered phase.

If `P*/N ≈ 0.11` also holds at L=2048 then `P* ≈ 460,000` at criticality,
still tens of times above what either GPU can use, and "hardware binds first"
survives even at the worst temperature. Measured so far (β=0.40):

| L | N | P*/N |
|---|---|---|
| 8 | 64 | 0.1204 |
| 12 | 144 | 0.1097 |
| 16 | 256 | 0.1081 |

−10% from L=8→12, then −1.5% from 12→16: a finite-size effect at L=8 settling
toward ≈0.107. L=24, 32 and 48 decide whether that is flat.

## The control, and why it is not a fixed threshold

`MU_sync(1)` is mathematically identical to single-update Metropolis, so the
P=1 row must reproduce the SU baseline. It is judged against the **combined
standard error** (`z = |E_P − E_SU| / σ`, must be < 3), not a fixed accuracy
cutoff: `acc` is normalised by 2N, which is small at L=8 while the statistical
error is not, so a raw "acc > 0.999" test flags a perfectly equilibrated small
lattice as broken. If a control fails, that size is under-equilibrated and
every other number at that size is worthless — raise `--tau-mult`.
