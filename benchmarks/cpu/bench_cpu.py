"""
Thread-scaling benchmark for the synchronous multi-spin update on CPU.

Measures what the algorithm claims: that removing the sequential dependency
makes the update parallel. Total flip-attempts are held CONSTANT across every
point, so wall-clock differences are pure speedup.

Run on a many-core node:
    python bench_cpu.py --threads 1,2,4,8,16,32,64 --L 256 --P 16384
On SLURM ask for the cores you intend to use:
    srun -c 64 --mem=16G --time=00:30:00 python bench_cpu.py --threads 1,2,4,8,16,32,64

Honest scope, keep it attached to the numbers: this measures CPU thread
scaling, not GPU speedup. A CPU reaches tens of threads; CUDA-METRO runs P
into the thousands, and none of the GPU-specific ceilings appear here.
"""
import argparse
import csv
import os
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--threads", default="1,2,4,8,16,32")
ap.add_argument("--L", type=int, default=256)
ap.add_argument("--P", type=int, default=16384,
                help="sites decided per round; must be large enough that the "
                     "per-round work exceeds thread-pool overhead")
ap.add_argument("--attempts", type=float, default=4e8,
                help="total flip-attempts per point, held constant")
ap.add_argument("--T", type=float, default=30.0)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--out", default="bench_cpu.csv")
args = ap.parse_args()

THREADS = [int(x) for x in args.threads.split(",")]
os.environ.setdefault("NUMBA_NUM_THREADS", str(max(THREADS)))

import numba                                        # noqa: E402  (after env var)
from parallel_cpu_mc import run, scratch, init_lattice, round_parallel, \
    round_serial, SPIN, J1, K1Z, AZ, KB              # noqa: E402

MAXT = numba.config.NUMBA_NUM_THREADS
THREADS = [t for t in THREADS if t <= MAXT]
rounds = max(1, int(args.attempts) // args.P)

print(f"CPU: {os.cpu_count()} logical cores | numba max threads: {MAXT}")
print(f"L={args.L} (N={args.L**2:,})  P={args.P:,}  rounds={rounds:,}  "
      f"attempts/point={rounds*args.P:,}")
print(f"threads to test: {THREADS}\n")

# warm the JIT so compilation never lands inside a timed region
_S = init_lattice(8, 0, True, SPIN)
_b = scratch(16)
round_serial(_S, 8, 16, 1.0, SPIN, J1, K1Z, AZ, *_b)
round_parallel(_S, 8, 16, 1.0, SPIN, J1, K1Z, AZ, *_b)

new = not os.path.exists(args.out)
fh = open(args.out, "a", newline="")
w = csv.writer(fh)
if new:
    w.writerow(["threads", "L", "N", "P", "rounds", "attempts", "repeat",
                "seconds", "attempts_per_s"])
    fh.flush()

beta = 1.0 / (args.T * KB)
baseline = None
for nt in THREADS:
    numba.set_num_threads(nt)
    times = []
    for rep in range(args.repeats):
        S = init_lattice(args.L, 1234 + rep, True, SPIN)
        buf = scratch(args.P)
        t0 = time.perf_counter()
        for _ in range(rounds):
            round_parallel(S, args.L, args.P, beta, SPIN, J1, K1Z, AZ, *buf)
        dt = time.perf_counter() - t0
        times.append(dt)
        w.writerow([nt, args.L, args.L**2, args.P, rounds, rounds * args.P,
                    rep, f"{dt:.4f}", f"{rounds*args.P/dt:.1f}"])
        fh.flush()
    best = min(times)                                # min = least contended
    if baseline is None:
        baseline = best
    print(f"threads={nt:3d}  {best:8.3f} s  "
          f"{rounds*args.P/best/1e6:8.2f} Mattempt/s  "
          f"speedup={baseline/best:6.2f}x  "
          f"efficiency={(baseline/best)/nt:6.1%}")

fh.close()
print(f"\nwrote {args.out}")
