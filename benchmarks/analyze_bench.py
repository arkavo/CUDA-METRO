"""
Close the loop: combine measured GPU speed with the measured accuracy law
into the speed-accuracy frontier, and read off the optimal P.

Speed side (from bench_speed.py):   speedup(P) = t_kernel(P=1) / t_kernel(P)
Accuracy side (from the MC study):  error(P/N) ~ 0.0276 * (P/N)   [ordered phase]

Two ceilings bound the useful P, and the smaller one wins:
  * HARDWARE  - speedup stops growing once the GPU saturates. With the
    current launch geometry (one lattice site per BLOCK, 2 threads per
    block) that ceiling is set by resident-blocks-per-SM, not by core
    count, so it may arrive far earlier than the core count suggests.
  * PHYSICS   - P must stay under the accuracy budget: for a target
    accuracy A, P/N <= (1 - A)/0.0276.

Run:  python analyze_bench.py bench_speed_*.csv
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_ERR = 0.0276          # fitted error coefficient, ordered phase
TARGETS = [0.99, 0.95]

paths = sys.argv[1:] or ["bench_speed.csv"]
df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
df = df.groupby(["size", "N", "blocks_P", "N_over_P"], as_index=False).agg(
    kernel_s=("kernel_s", "median"), rng_s=("rng_s", "median"),
    attempts_per_s=("attempts_per_s", "median"))

print("=== measured throughput ===")
for size, g in df.groupby("size"):
    g = g.sort_values("blocks_P")
    base = g[g.blocks_P == g.blocks_P.min()]
    if base.empty:
        continue
    t1, p1 = float(base.kernel_s.iloc[0]), int(base.blocks_P.iloc[0])
    print(f"\nL={size} (N={size**2}), speedup relative to P={p1}:")
    for _, r in g.iterrows():
        su = t1 / r.kernel_s
        ideal = r.blocks_P / p1
        eff = su / ideal
        rng_frac = r.rng_s / (r.rng_s + r.kernel_s)
        print(f"  P={int(r.blocks_P):6d}  kernel={r.kernel_s:8.3f}s  "
              f"speedup={su:7.2f}x  ideal={ideal:7.1f}x  "
              f"parallel-efficiency={eff:5.1%}  RNG share={rng_frac:4.0%}")

    # where does it stop scaling? last P whose efficiency is still >= 50%
    g = g.assign(speedup=t1 / g.kernel_s, ideal=g.blocks_P / p1)
    g = g.assign(eff=g.speedup / g.ideal)
    sat = g[g.eff >= 0.5].blocks_P.max()
    print(f"  -> hardware ceiling (parallel efficiency >= 50%): P ~ {sat}")

    N = size ** 2
    for A in TARGETS:
        p_acc = (1 - A) / C_ERR * N
        print(f"  -> accuracy ceiling at {A:.0%}: P <= {p_acc:,.0f}"
              f"   ==> usable P = {min(sat, p_acc):,.0f}"
              f"  ({'hardware' if sat < p_acc else 'accuracy'}-limited)")

# ---- the frontier plot: speedup you actually get vs accuracy you keep ----
fig, ax = plt.subplots(figsize=(8, 5.5))
for size, g in df.groupby("size"):
    g = g.sort_values("blocks_P")
    t1 = float(g.kernel_s.iloc[0]); p1 = float(g.blocks_P.iloc[0])
    acc = 1 - C_ERR * (g.blocks_P / (size ** 2))
    ax.plot(acc, t1 / g.kernel_s, "o-", markersize=7, label=f"L={size}")
for A in TARGETS:
    ax.axvline(A, color="crimson", ls="--", lw=1)
    ax.text(A, ax.get_ylim()[1] * 0.97, f" {A:.0%}", color="crimson", fontsize=9)
ax.set_xlabel("predicted accuracy (from the fitted error law)")
ax.set_ylabel("measured speedup vs smallest P")
ax.set_title("Speed-accuracy frontier: measured time, predicted accuracy")
ax.legend(); ax.grid(alpha=0.3)
ax.invert_xaxis()
fig.tight_layout()
fig.savefig("frontier.png", dpi=150)
print("\nwrote frontier.png")
