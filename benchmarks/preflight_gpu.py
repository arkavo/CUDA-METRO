"""
Preflight for the CUDA-METRO speed benchmark. Run this on a GPU node BEFORE
submitting the sweep - it takes seconds and fails loudly on exactly the things
that otherwise waste a queue slot and twenty minutes.

    python preflight_gpu.py            # from benchmarks/, any cwd

Checks, in the order they would bite you:
  1. imports resolve (_repo.py puts src/ and src/cudametro/ on sys.path)
  2. a GPU is visible and a context comes up
  3. the config file exists at the path given
  4. every MonteCarlo attribute bench_speed.py touches actually exists
  5. the two kernels exist and accept the exact argument list we pass
  6. one real round runs and CHANGES the lattice - i.e. the kernel is doing
     work, not silently no-oping on an uninitialised beta

Exit code 0 = safe to sbatch. Anything else, read the message.
"""
import argparse
import json
import os
import sys
import traceback

import numpy as np

import _repo                 # puts <repo>/src and <repo>/src/cudametro on sys.path

KB = 8.6173e-2
FAIL = []


def check(label, fn):
    try:
        detail = fn()
        print(f"  ok    {label}" + (f"  [{detail}]" if detail else ""))
        return True
    except Exception as e:
        print(f"  FAIL  {label}\n          {type(e).__name__}: {e}")
        FAIL.append(label)
        return False


ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None,
                help="defaults to <repo>/configs/bench.json")
ap.add_argument("--size", type=int, default=64)
ap.add_argument("--blocks", type=int, default=256)
ap.add_argument("--temp", type=float, default=30.0)
args = ap.parse_args()

print("1. imports")
try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import construct as cst
    import cudametro.montecarlo as mc
    print("  ok    pycuda, construct, cudametro.montecarlo")
except ModuleNotFoundError as e:
    # Distinguish "the package isn't installed" from "the path is wrong".
    # construct.py imports seaborn/tqdm at module scope, and the repo's
    # pyproject declares only numpy, so a plain install leaves them missing.
    missing = e.name or "?"
    if missing in ("construct", "cudametro"):
        sys.exit(f"\ncannot import {missing}: this is a PATH problem. Check that "
                 f"benchmarks/ sits at the repo root beside src/, and that "
                 f"_repo.py is next to this script.")
    sys.exit(f"\nmissing package: {missing}\n"
             f"  install it into the venv running this script:\n"
             f"    {sys.executable} -m pip install {missing}\n"
             f"  (or re-run ./run.sh setup, which installs the full set)")
except Exception:
    traceback.print_exc()
    sys.exit("\nimports failed for a reason other than a missing module - "
             "the traceback above is the thing to read.")

print("\n2. device")
dev = cuda.Device(0)
free, total = cuda.mem_get_info()
attr = cuda.device_attribute
print(f"  ok    {dev.name()}  CC {dev.compute_capability()}  "
      f"{dev.total_memory()/1024**3:.1f} GB, {free*100/total:.0f}% free")
print(f"  info  SMs={dev.get_attribute(attr.MULTIPROCESSOR_COUNT)}  "
      f"maxThreads/SM={dev.get_attribute(attr.MAX_THREADS_PER_MULTIPROCESSOR)}")
if hasattr(attr, "MAX_BLOCKS_PER_MULTIPROCESSOR"):
    b = dev.get_attribute(attr.MAX_BLOCKS_PER_MULTIPROCESSOR)
    sm = dev.get_attribute(attr.MULTIPROCESSOR_COUNT)
    print(f"  info  maxBlocks/SM={b}  ->  ~{b*sm} concurrent blocks = the P "
          f"beyond which speedup should stop growing")

print("\n3. config")
CONFIG = args.config or _repo.config("bench.json")
check(f"{CONFIG} exists", lambda: "found"
      if os.path.exists(CONFIG) else (_ for _ in ()).throw(
          FileNotFoundError(f"not found (cwd={os.getcwd()})")))
if FAIL:
    sys.exit("\nfix the config path first; nothing below can run.")

print("\n4. MonteCarlo object and attributes")
os.chdir(_repo.scratch())        # contain the Output_* dir it creates
# SIZE/Blocks must travel through the config: __init__ copies SIZE into the
# device buffer GSIZE once, and assigning m.size afterwards leaves GSIZE stale.
_cfg = json.load(open(CONFIG))
_cfg.update(SIZE=args.size, Blocks=args.blocks, stability_runs=8,
            stability_wrap=1, Prefix="preflight")
_cfgp = os.path.join(_repo.scratch(), "preflight_cfg.json")
json.dump(_cfg, open(_cfgp, "w"))
m = cst.MonteCarlo(config=_cfgp, input_folder=_repo.inputs())
m.C1 = m.Blocks * m.stability_runs
m.mc_init()
m.grid_reset()

for a in ["GPU_MAT", "GRID_GPU", "BJ", "GPU_TRANS", "B_GPU", "GSIZE",
          "Threads", "Blocks", "size", "spin", "grid"]:
    check(f"m.{a}", lambda a=a: type(getattr(m, a)).__name__)

# GSIZE is the device's idea of the lattice edge. If it disagrees with
# m.size, the RNG draws indices outside the allocated grid and the first
# kernel launch dies with an illegal memory access.
def _gsize():
    v = np.zeros(1, dtype=np.int32)
    cuda.memcpy_dtoh(v, m.GSIZE)
    if int(v[0]) != m.size:
        raise AssertionError(f"device GSIZE={int(v[0])} but m.size={m.size} "
                             f"- SIZE must be set via the config, not after")
    return f"GSIZE={int(v[0])} == m.size"
check("device GSIZE matches m.size", _gsize)

check("m.generate_random_numbers(1)", lambda: m.generate_random_numbers(1))
for a in ["NFULL", "S1FULL", "S2FULL", "S3FULL", "RLIST"]:
    check(f"m.{a} (after RNG)", lambda a=a: f"len {len(getattr(m, a))}")

# The typo that eats results in the repo's own bench.py / curie_temp.py.
print("  info  save_directory present:",
      hasattr(m, "save_directory"),
      "| save_direcotry present:", hasattr(m, "save_direcotry"),
      "  <- bench.py and curie_temp.py reference the misspelling")

print("\n5. kernels")
check("mc.METROPOLIS_MC_DM0_3_6_3_6", lambda: "found")
check("mc.GRID_COPY", lambda: "found")
_ = mc.METROPOLIS_MC_DM0_3_6_3_6, mc.GRID_COPY

print("\n6. one real round (the part that actually proves it works)")
beta = np.array([1.0 / (args.temp * KB)], dtype=np.float32)
cuda.memcpy_htod(m.BJ, beta)

before = np.zeros_like(m.grid)
cuda.memcpy_dtoh(before, m.GRID_GPU)

P = m.Blocks
try:
    for j in range(m.stability_runs):
        sl = slice(j * P, (j + 1) * P)
        mc.METROPOLIS_MC_DM0_3_6_3_6(
            m.GPU_MAT, m.GRID_GPU, m.BJ,
            m.NFULL[sl], m.S1FULL[sl], m.S2FULL[sl], m.S3FULL[sl],
            m.RLIST[sl], m.GPU_TRANS, m.B_GPU, m.GSIZE,
            block=(m.Threads, 1, 1), grid=(P, 1, 1))
        mc.GRID_COPY(m.GRID_GPU, m.GPU_TRANS, block=(2, 1, 1), grid=(P, 1, 1))
    cuda.Context.synchronize()
    print("  ok    kernel launch + sync, argument list accepted")
except Exception as e:
    print(f"  FAIL  kernel launch\n          {type(e).__name__}: {e}")
    print("\n  The CUDA context is dead; anything further would only print"
          "\n  more cleanup noise. Stopping here.")
    if "illegal memory access" in str(e):
        print("  An illegal access at this point almost always means a device"
              "\n  buffer disagrees with a host-side size. Check GSIZE vs m.size.")
    sys.exit(1)

after = np.zeros_like(m.grid)
cuda.memcpy_dtoh(after, m.GRID_GPU)
changed = int(np.sum(np.abs(after - before) > 1e-6))
norm = np.linalg.norm(after.reshape(-1, 3), axis=1)
print(f"  info  {changed} of {after.size} grid components changed")
print(f"  info  spin norm: mean {norm.mean():.4f}, target {m.spin} "
      f"(drift here means the proposal is not unit-sphere)")
if changed == 0:
    FAIL.append("kernel produced no change")
    print("  FAIL  the lattice did not change. Either every move was rejected "
          "(check beta / temperature) or the kernel is not writing back.")
elif abs(norm.mean() - m.spin) > 0.05 * m.spin:
    print("  WARN  spin magnitude drifted from S - worth a look, but not fatal "
          "for a timing run.")

# CUDA-event timing, the thing the benchmark depends on
print("\n7. CUDA event timing")
start, end = cuda.Event(), cuda.Event()
cuda.Context.synchronize()
start.record()
for j in range(m.stability_runs):
    sl = slice(j * P, (j + 1) * P)
    mc.METROPOLIS_MC_DM0_3_6_3_6(
        m.GPU_MAT, m.GRID_GPU, m.BJ,
        m.NFULL[sl], m.S1FULL[sl], m.S2FULL[sl], m.S3FULL[sl],
        m.RLIST[sl], m.GPU_TRANS, m.B_GPU, m.GSIZE,
        block=(m.Threads, 1, 1), grid=(P, 1, 1))
    mc.GRID_COPY(m.GRID_GPU, m.GPU_TRANS, block=(2, 1, 1), grid=(P, 1, 1))
end.record()
end.synchronize()
ms = start.time_till(end)
print(f"  ok    {m.stability_runs} rounds at P={P} in {ms:.3f} ms "
      f"({m.stability_runs*P/(ms*1e-3)/1e6:.2f} Mattempt/s)")
if ms <= 0:
    FAIL.append("event timer returned <= 0")

print("\n" + "=" * 60)
if FAIL:
    print("PREFLIGHT FAILED:")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
print("PREFLIGHT PASSED - safe to sbatch the sweep.")
