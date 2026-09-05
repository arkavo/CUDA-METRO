"""
Close the loop: measured GPU speed x the measured accuracy law -> the
speed-accuracy frontier, and the usable-P ceiling.

    python analyze_bench.py results/*.csv

Speed side (bench_speed.py):        speedup(P) = t_kernel(P_min) / t_kernel(P)
Accuracy side (the MC study):       error(P/N) ~ 0.0276 * (P/N)  [ordered phase]

Two ceilings bound the useful P, and the smaller one wins:
  HARDWARE - speedup stops growing once the GPU saturates. The launch geometry
    is one lattice site per BLOCK with Threads=2, so the ceiling is set by
    resident BLOCKS per SM (maxBlocks/SM x SMs), NOT by CUDA core count. That
    product is what preflight_gpu.py prints.
  PHYSICS  - for a target accuracy A, P/N <= (1 - A)/0.0276.

EVERYTHING IS PER GPU. Medians are taken within one GPU model; pooling two
different cards would average away the very effect being measured.
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
if "gpu" not in df.columns:
    df["gpu"] = "unknown GPU"
df["gpu"] = df["gpu"].astype(str).str.strip()

# NOTE the "gpu" key. Without it, two machines' timings for the same (size, P)
# are medianed into one meaningless number.
# "attempts" MUST be a grouping key. Sweeps run with different --attempts
# produce different kernel times for the same (size, P), and pooling them
# medians incomparable numbers: a P present in only one sweep is then compared
# against a P whose median came from another, and the curve grows spikes that
# look like hardware behaviour and are pure bookkeeping.
g = df.groupby(["gpu", "attempts", "size", "N", "blocks_P", "N_over_P"],
               as_index=False).agg(
    kernel_s=("kernel_s", "median"), rng_s=("rng_s", "median"),
    rounds=("rounds", "median"),
    attempts_per_s=("attempts_per_s", "median"), n=("kernel_s", "size"))

print(f"{len(df)} rows from {len(paths)} file(s); "
      f"{df.gpu.nunique()} GPU model(s): {', '.join(sorted(df.gpu.unique()))}\n")

summary = {}
for (gpu, att), gd in g.groupby(["gpu", "attempts"]):
    print("=" * 70)
    print(f"GPU: {gpu}    attempts/point = {int(att):,}")
    print("=" * 70)
    for size, s in gd.groupby("size"):
        s = s.sort_values("blocks_P")
        t1 = float(s.kernel_s.iloc[0]); p1 = float(s.blocks_P.iloc[0])
        s = s.assign(speedup=t1 / s.kernel_s,
                     ideal=s.blocks_P / p1)
        s = s.assign(eff=s.speedup / s.ideal,
                     rng_frac=s.rng_s / (s.rng_s + s.kernel_s))
        print(f"\n  L={size} (N={size**2:,}), relative to P={int(p1)}:")
        for _, r in s.iterrows():
            print(f"    P={int(r.blocks_P):7d}  kernel={r.kernel_s:9.4f}s  "
                  f"speedup={r.speedup:8.2f}x  ideal={r.ideal:8.1f}x  "
                  f"eff={r.eff:6.1%}  RNG={r.rng_frac:4.0%}")
        sat = s[s.eff >= 0.5].blocks_P.max()
        pmax = s.blocks_P.max()
        # If efficiency never fell below 50%, the "ceiling" is just the largest
        # P we tried - the measurement is CENSORED, not saturated. Reporting it
        # as a hardware ceiling invents a limit the data does not show. This
        # bites hardest at small L, where P <= N caps the sweep on its own.
        censored = bool(sat == pmax)
        peak = s.loc[s.attempts_per_s.idxmax()]
        if censored:
            cap = " (P capped by N)" if pmax >= size ** 2 else ""
            print(f"    -> ceiling NOT REACHED: efficiency still "
                  f"{float(s.eff.iloc[-1]):.0%} at the largest P tested "
                  f"({int(pmax)}){cap}")
        else:
            print(f"    -> hardware ceiling (eff >= 50%): P ~ {int(sat)}")
        # ---- the two-parameter model ------------------------------------
        # A round costs a fixed launch+sync latency L until the work in it
        # exceeds what the GPU can do; after that it costs work/throughput.
        #     t(P) = rounds * L                 while P < P*
        #     t(P) = attempts / T               once P > P*
        # Setting them equal gives the crossover and the speedup ceiling:
        #     P* = L * T          speedup_max = P* / P_min
        # Two measured constants predict the entire curve.
        s = s.assign(us_per_round=1e6 * s.kernel_s / s.rounds)
        L_us = float(s.us_per_round.min())            # flat region = pure latency
        T = float(s.attempts_per_s.max())             # plateau = peak throughput
        p_star = L_us * 1e-6 * T
        print(f"    -> per-round latency L = {L_us:.1f} us   "
              f"peak throughput T = {T/1e6:.1f} Mattempt/s")
        print(f"    -> model: P* = L x T = {p_star:,.0f}   "
              f"speedup ceiling = P*/{int(p1)} = {p_star/p1:.0f}x  "
              f"(observed max {float(s.speedup.max()):.0f}x)")
        print(f"    -> peak throughput {peak.attempts_per_s/1e6:.1f} Mattempt/s "
              f"at P={int(peak.blocks_P)}")
        summary.setdefault(f"{gpu}  [attempts={int(att):,}]", []).append(
            (size, int(sat), censored))
        N = size ** 2
        for A in TARGETS:
            p_acc = (1 - A) / C_ERR * N
            lim = "hardware" if sat < p_acc else "accuracy"
            print(f"    -> {A:.0%} accuracy allows P <= {p_acc:,.0f}"
                  f"  ==> usable P = {min(sat, p_acc):,.0f}  ({lim}-limited)")

print("\n" + "=" * 70)
print("SATURATION SUMMARY  (compare against maxBlocks/SM x SMs from preflight)")
print("=" * 70)
for gpu, rows in summary.items():
    real = [s for _, s, c in rows if not c]
    cens = [s for _, s, c in rows if c]
    print(f"  {gpu}")
    if real:
        print(f"    measured ceilings (eff dropped below 50%): {real}")
        print(f"    median {int(np.median(real))}")
    if cens:
        print(f"    NOT reached at these sizes (largest P tried): {cens}"
              f"  <- censored, not a limit")
    if not real:
        print("    no ceiling observed anywhere: the sweep never went far "
              "enough in P. Run ./run.sh big.")

# ---- frontier: measured speedup vs predicted accuracy, one line per (gpu,size)
fig, ax = plt.subplots(figsize=(9, 5.5))
styles = ["-o", "-s", "-^", "-D", "-v", "-P"]
for gi, ((gpu, att), gd) in enumerate(g.groupby(["gpu", "attempts"])):
    for si, (size, s) in enumerate(gd.groupby("size")):
        s = s.sort_values("blocks_P")
        t1 = float(s.kernel_s.iloc[0])
        acc = 1 - C_ERR * (s.blocks_P / (size ** 2))
        ax.plot(acc, t1 / s.kernel_s, styles[si % len(styles)], markersize=6,
                alpha=0.85, label=f"{gpu.split()[-1] if ' ' in gpu else gpu} L={size}")
for A in TARGETS:
    ax.axvline(A, color="crimson", ls="--", lw=1)
    ax.text(A, ax.get_ylim()[1] * 0.97, f" {A:.0%}", color="crimson", fontsize=9)
ax.set_xlabel("predicted accuracy (from the fitted error law)")
ax.set_ylabel("measured speedup vs smallest P")
ax.set_title("Speed-accuracy frontier: measured time, predicted accuracy")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
ax.invert_xaxis()
fig.tight_layout()
fig.savefig("frontier.png", dpi=150)
print("\nwrote frontier.png")
