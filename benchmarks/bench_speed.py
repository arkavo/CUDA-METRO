"""
Speed benchmark for CUDA-METRO: wall-clock vs Blocks (=P) across lattice sizes.

This is the measurement missing from the accuracy study. It sweeps the SAME
(N, P) grid the accuracy law was fitted on, so the two combine into a
speed-accuracy frontier afterwards (see analyze_bench.py).

The kernel-call pattern below is copied verbatim from construct.run_mc_tc_3636
(argument order, slice arithmetic, block/grid shapes) so this measures exactly
the code path the science runs on - not a re-implementation of it.

Differences from the repo's src/cudametro/bench.py:
  * that script has `bench_MC.save_direcotry` on its LAST line. The current
    construct.py spells the attribute `save_directory`, so it raises
    AttributeError *after* all the timing work and every result is lost.
    (curie_temp.py has the same typo in four places.)
  * that script defines `sizes = [64,128,256,512]` and never uses it; only
    size 256 is ever benched.
  * that script's `--config ../configs/bench.json` resolves to a non-existent
    src/configs/. From src/cudametro/ the correct path is ../../configs/.
  * timing is per-point CSV-flushed, so a job killed by the scheduler still
    leaves you everything measured up to that point.
  * host-side RNG generation is timed separately from the kernel loop. It is a
    real cost but it is not the parallel update, and at large P it can dominate
    -- reporting one number hides that.
  * timing uses CUDA events, not datetime. Kernel launches are asynchronous;
    host wall-clock measures enqueue time, not execution time.

Run from src/cudametro/ (so `import construct` resolves):
    python bench_speed.py [--sizes 64,128,256] [--blocks 64,...] [--out b.csv]
"""
import argparse
import csv
import datetime as dt
import gc
import os
import sys

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda

import _repo                      # noqa: F401  (puts src/ on sys.path)
import construct as cst
import cudametro.montecarlo as mc

KB = 8.6173e-2          # meV/K, as construct.py uses

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None,
                help="defaults to <repo>/configs/bench.json")
ap.add_argument("--sizes", default="64,128,256,512")
ap.add_argument("--blocks", default="64,128,256,512,1024,2048,4096,8192,16384")
ap.add_argument("--attempts", type=float, default=2e7,
                help="total flip-attempts per (N,P) point; held CONSTANT so "
                     "wall-clock differences are pure speedup")
ap.add_argument("--temp", type=float, default=30.0,
                help="temperature in K. MUST be set: mc_init only ALLOCATES "
                     "BJ, it never writes it - see the memcpy_htod below.")
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--out", default="bench_speed.csv")
args = ap.parse_args()

CONFIG = args.config or _repo.config("bench.json")
if not os.path.exists(CONFIG):
    sys.exit(f"config not found: {CONFIG}\n"
             f"pass --config with an explicit path.")

SIZES = [int(x) for x in args.sizes.split(",")]
BLOCKS = [int(x) for x in args.blocks.split(",")]
TARGET_ATTEMPTS = int(args.attempts)
BETA = np.array([1.0 / (args.temp * KB)], dtype=np.float32)

dev = cuda.Device(0)
free, total = cuda.mem_get_info()
attr = cuda.device_attribute
print(f"GPU: {dev.name()}  CC {dev.compute_capability()}  "
      f"{dev.total_memory()/1024**3:.1f} GB  ({free*100/total:.1f}% free)")
print(f"SMs: {dev.get_attribute(attr.MULTIPROCESSOR_COUNT)}   "
      f"max threads/SM: {dev.get_attribute(attr.MAX_THREADS_PER_MULTIPROCESSOR)}")
# The launch geometry is one lattice site per BLOCK with Threads=2, so
# occupancy is bounded by resident BLOCKS per SM, not by CUDA core count.
# If this attribute exists, print it - it is the number that predicts where
# the speedup curve flattens.
if hasattr(attr, "MAX_BLOCKS_PER_MULTIPROCESSOR"):
    print(f"max blocks/SM: {dev.get_attribute(attr.MAX_BLOCKS_PER_MULTIPROCESSOR)}")
print(f"T = {args.temp} K  ->  beta = {float(BETA[0]):.5f} 1/meV")
print(f"total attempts per point: {TARGET_ATTEMPTS:,}")

# The RNG arrays are sized by C1 = Blocks * stability_runs, which this script
# holds equal to TARGET_ATTEMPTS at every point. Five device arrays of C1
# float32 plus four more from gen_uniform:
approx_gb = TARGET_ATTEMPTS * 4 * 9 / 1024**3
print(f"RNG buffers: ~{approx_gb:.2f} GB per point "
      f"(scale --attempts down if that does not fit)\n")

new = not os.path.exists(args.out)
fh = open(args.out, "a", newline="")
w = csv.writer(fh)
if new:
    w.writerow(["gpu", "temp_K", "size", "N", "blocks_P", "N_over_P", "rounds",
                "attempts", "repeat", "kernel_s", "rng_s", "total_s",
                "attempts_per_s", "rounds_per_s"])
    fh.flush()

for size in SIZES:
    N = size * size
    for P in BLOCKS:
        if P > N:
            print(f"skip size={size} P={P} (P > N)")
            continue

        rounds = max(1, TARGET_ATTEMPTS // P)

        for rep in range(args.repeats):
            m = None
            try:
                m = cst.MonteCarlo(config=CONFIG)
                m.size = size
                m.Blocks = P                 # must precede mc_init: it sizes
                m.stability_runs = rounds    # TMATRIX = Blocks*4 and GPU_TRANS
                m.S_Wrap = 1
                m.C1 = m.Blocks * m.stability_runs
                m.mc_init()
                m.grid_reset()

                # ---------------------------------------------------------
                # BJ is only ALLOCATED by mc_init (drv.mem_alloc). The
                # temperature is written by run_mc_tc_3636, which this script
                # deliberately bypasses to time the kernel loop directly.
                # Without this memcpy the kernel reads uninitialised device
                # memory as beta: not a crash, but the acceptance rate - and
                # therefore the write traffic to GPU_TRANS and the branch
                # divergence in the kernel - would be meaningless.
                # ---------------------------------------------------------
                cuda.memcpy_htod(m.BJ, BETA)

                # --- host-side RNG generation, timed on its own ---
                t0 = dt.datetime.now()
                m.generate_random_numbers(1)
                cuda.Context.synchronize()
                rng_s = (dt.datetime.now() - t0).total_seconds()

                # --- the parallel update itself, timed with CUDA events ---
                start, end = cuda.Event(), cuda.Event()
                cuda.Context.synchronize()
                start.record()
                for j in range(rounds):
                    sl = slice(j * P, (j + 1) * P)
                    mc.METROPOLIS_MC_DM0_3_6_3_6(
                        m.GPU_MAT, m.GRID_GPU, m.BJ,
                        m.NFULL[sl], m.S1FULL[sl], m.S2FULL[sl], m.S3FULL[sl],
                        m.RLIST[sl], m.GPU_TRANS, m.B_GPU, m.GSIZE,
                        block=(m.Threads, 1, 1), grid=(P, 1, 1))
                    mc.GRID_COPY(m.GRID_GPU, m.GPU_TRANS,
                                 block=(2, 1, 1), grid=(P, 1, 1))
                end.record()
                end.synchronize()
                kernel_s = start.time_till(end) * 1e-3   # ms -> s

                attempts = rounds * P
                w.writerow([dev.name(), args.temp, size, N, P, N / P, rounds,
                            attempts, rep,
                            f"{kernel_s:.6f}", f"{rng_s:.6f}",
                            f"{kernel_s + rng_s:.6f}",
                            f"{attempts / kernel_s:.1f}",
                            f"{rounds / kernel_s:.1f}"])
                fh.flush()   # survive a scheduler kill
                print(f"size={size:4d} P={P:6d} N/P={N/P:8.1f} rep={rep} "
                      f"kernel={kernel_s:8.3f}s rng={rng_s:7.3f}s "
                      f"{attempts/kernel_s/1e6:8.2f} Mattempt/s")

            except cuda.MemoryError:
                print(f"OOM at size={size} P={P} - skipping this P")
                break
            except Exception as e:                       # keep the sweep alive
                print(f"FAILED size={size} P={P} rep={rep}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                break
            finally:
                # PyCUDA frees DeviceAllocation and GPUArray on collection, but
                # this loop allocates hundreds of MB per iteration - leave it to
                # chance and a long sweep OOMs on a card that had room for it.
                del m
                gc.collect()

fh.close()
print(f"\nwrote {args.out}")
