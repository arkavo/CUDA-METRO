"""
Speed benchmark for CUDA-METRO: wall-clock vs Blocks (=P) across lattice sizes.

This is the measurement missing from the accuracy study. It sweeps the SAME
(N, P) grid the accuracy law was fitted on, so the two combine into a
speed-accuracy frontier afterwards (see analyze_bench.py).

The kernel-call block below is copied verbatim from construct.run_mc_tc_3636
(argument order, slice arithmetic, block/grid shapes) so this times the code
path the science actually runs, not a re-implementation of it.

Run it through ./run.sh, or directly:
    python bench_speed.py --sizes 64,128 --blocks 64,256,1024 --out out.csv

Choosing the GPU: this uses pycuda.autoinit, which takes device 0 of whatever
is visible. Select a card with the environment, not a flag:
    CUDA_VISIBLE_DEVICES=2 python bench_speed.py ...

Resuming: --resume <csv> reads an existing file and skips every (size, P,
repeat) already in it, then appends. A run killed for any reason restarts
where it stopped.
"""
import argparse
import csv
import datetime as dt
import gc
import json
import os
import shutil
import socket
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
ap.add_argument("--resume", metavar="CSV", default=None,
                help="skip (size,P,repeat) rows already present in this file "
                     "(usually the same path as --out)")
args = ap.parse_args()

CONFIG = args.config or _repo.config("bench.json")
if not os.path.exists(CONFIG):
    sys.exit(f"config not found: {CONFIG}\npass --config with an explicit path.")

# MonteCarlo.__init__ writes Output_<prefix>_<material>_<timestamp>/ into the
# CURRENT directory, and os.mkdir (not makedirs) raises FileExistsError when
# two constructions land in the same second - which they will, on fast points.
# Two defences: run from a disposable scratch dir, and give every point a
# unique Prefix so the names cannot collide in the first place.
SCRATCH = _repo.scratch()
shutil.rmtree(SCRATCH, ignore_errors=True)
os.makedirs(SCRATCH, exist_ok=True)
_BASE_CFG = json.load(open(CONFIG))


def config_for(size, P, rep, rounds):
    """A per-point config.

    SIZE and Blocks MUST go through the config, not be assigned afterwards:
    __init__ copies SIZE into the device buffer GSIZE exactly once, and the
    RNG draws site indices as floor(u * GSIZE^2). Setting m.size later leaves
    GSIZE at the config value, so the kernel indexes a lattice larger than the
    one allocated -> illegal memory access. Prefix keeps save_directory unique."""
    c = dict(_BASE_CFG)
    c["SIZE"] = size
    c["Blocks"] = P
    c["stability_runs"] = rounds
    c["stability_wrap"] = 1
    c["Prefix"] = f"b{size}_{P}_{rep}"
    p = os.path.join(SCRATCH, f"cfg_{size}_{P}_{rep}.json")
    with open(p, "w") as fh:
        json.dump(c, fh)
    return p


SIZES = [int(x) for x in args.sizes.split(",")]
BLOCKS = [int(x) for x in args.blocks.split(",")]
TARGET_ATTEMPTS = int(args.attempts)
BETA = np.array([1.0 / (args.temp * KB)], dtype=np.float32)
HOST = socket.gethostname()

# ---- resume: which points are already measured? ----
done = set()
if args.resume and os.path.exists(args.resume):
    with open(args.resume) as fh:
        for r in csv.DictReader(fh):
            try:
                done.add((int(r["size"]), int(r["blocks_P"]), int(r["repeat"])))
            except (KeyError, ValueError):
                continue          # tolerate a truncated last row
    print(f"resume: {len(done)} points already measured in {args.resume}")

dev = cuda.Device(0)
free, total = cuda.mem_get_info()
attr = cuda.device_attribute
SMS = dev.get_attribute(attr.MULTIPROCESSOR_COUNT)
print(f"host: {HOST}   visible devices: "
      f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
print(f"GPU: {dev.name()}  CC {dev.compute_capability()}  "
      f"{dev.total_memory()/1024**3:.1f} GB  ({free*100/total:.1f}% free)")
print(f"SMs: {SMS}   max threads/SM: "
      f"{dev.get_attribute(attr.MAX_THREADS_PER_MULTIPROCESSOR)}")
# The launch geometry is one lattice site per BLOCK with Threads=2, so
# occupancy is bounded by resident BLOCKS per SM, not by CUDA core count.
if hasattr(attr, "MAX_BLOCKS_PER_MULTIPROCESSOR"):
    mb = dev.get_attribute(attr.MAX_BLOCKS_PER_MULTIPROCESSOR)
    print(f"max blocks/SM: {mb}  ->  ~{mb*SMS} concurrent blocks = the P "
          f"beyond which speedup should stop growing")
print(f"T = {args.temp} K  ->  beta = {float(BETA[0]):.5f} 1/meV")
print(f"attempts per point: {TARGET_ATTEMPTS:,}   "
      f"RNG buffers ~{TARGET_ATTEMPTS*4*9/1024**3:.2f} GB\n", flush=True)

args.out = os.path.abspath(args.out)     # resolve BEFORE the chdir below
if args.resume:
    args.resume = os.path.abspath(args.resume)
os.chdir(SCRATCH)

new = not os.path.exists(args.out)
fh = open(args.out, "a", newline="")
w = csv.writer(fh)
if new:
    w.writerow(["host", "gpu", "temp_K", "size", "N", "blocks_P", "N_over_P",
                "rounds", "attempts", "repeat", "kernel_s", "rng_s", "total_s",
                "attempts_per_s", "rounds_per_s"])
    fh.flush()

planned = sum(1 for s in SIZES for P in BLOCKS if P <= s * s) * args.repeats
n = 0
for size in SIZES:
    N = size * size
    for P in BLOCKS:
        if P > N:
            continue
        rounds = max(1, TARGET_ATTEMPTS // P)

        for rep in range(args.repeats):
            n += 1
            if (size, P, rep) in done:
                print(f"[{n}/{planned}] skip (done) size={size} P={P} rep={rep}",
                      flush=True)
                continue
            m = None
            try:
                m = cst.MonteCarlo(config=config_for(size, P, rep, rounds),
                                   input_folder=_repo.inputs())
                assert m.size == size and m.Blocks == P
                m.C1 = m.Blocks * m.stability_runs
                m.mc_init()
                m.grid_reset()

                # ---------------------------------------------------------
                # mc_init only ALLOCATES BJ. run_mc_tc_3636 is what writes the
                # temperature into it, and this script bypasses that function
                # to time the kernel loop directly - so it must do the memcpy
                # itself. Without this the kernel reads uninitialised device
                # memory as beta: no crash, but the acceptance rate (hence the
                # write traffic to GPU_TRANS and the branch divergence in the
                # kernel) is meaningless, and the timings look plausible.
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
                w.writerow([HOST, dev.name(), args.temp, size, N, P, N / P,
                            rounds, attempts, rep,
                            f"{kernel_s:.6f}", f"{rng_s:.6f}",
                            f"{kernel_s + rng_s:.6f}",
                            f"{attempts / kernel_s:.1f}",
                            f"{rounds / kernel_s:.1f}"])
                fh.flush()
                os.fsync(fh.fileno())        # survive a hard kill, not just exit
                print(f"[{n}/{planned}] size={size:4d} P={P:6d} "
                      f"N/P={N/P:8.1f} rep={rep} kernel={kernel_s:8.3f}s "
                      f"rng={rng_s:7.3f}s {attempts/kernel_s/1e6:8.2f} Mattempt/s",
                      flush=True)

            except cuda.MemoryError:
                print(f"OOM at size={size} P={P} - skipping this P", flush=True)
                break
            except Exception as e:                       # keep the sweep alive
                print(f"FAILED size={size} P={P} rep={rep}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
                break
            finally:
                # This loop allocates hundreds of MB per point. Left to chance,
                # a long sweep OOMs a card that had room for it.
                del m
                gc.collect()

fh.close()
print(f"\nwrote {args.out}")
