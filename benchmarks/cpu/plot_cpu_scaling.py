"""Turn bench_cpu*.csv into the poster's thread-scaling panel.

    python plot_cpu_scaling.py bench_cpu_1234567.csv

Writes figD_cpu.svg in the poster's own visual language (transparent ground,
recessive chrome, no legend box) so it drops straight into the layout.

The panel plots measured speedup against thread count with the ideal line
underneath. The gap between them IS the result: how much of the update is
genuinely parallel. Amdahl's serial fraction is fitted from the curve and
printed - quote it, it is the honest one-number summary.
"""
import sys
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, FAINT, GRID, GROUND = "#16161A", "#55555C", "#8B8B93", "#E2E2DE", "#F4F4F2"
RUST, TEAL = "#C0451F", "#0B6E63"

plt.rcParams.update({
    "font.size": 20, "axes.edgecolor": GRID, "axes.linewidth": 1.6,
    "axes.labelcolor": MUTED, "axes.labelsize": 20, "text.color": INK,
    "xtick.color": FAINT, "ytick.color": FAINT,
    "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
    "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "svg.fonttype": "none",
})

path = sys.argv[1] if len(sys.argv) > 1 else "bench_cpu.csv"
best = defaultdict(lambda: float("inf"))
meta = {}
with open(path) as fh:
    for r in csv.DictReader(fh):
        t = int(r["threads"])
        best[t] = min(best[t], float(r["seconds"]))       # min = least contended
        meta = r

th = np.array(sorted(best))
tm = np.array([best[t] for t in th])
sp = tm[0] / tm

# Amdahl: S(p) = 1 / (f + (1-f)/p). Least-squares on the measured points.
def amdahl_resid(f):
    return np.sum((sp - 1.0 / (f + (1 - f) / th)) ** 2)
fs = np.linspace(0.0, 0.5, 5001)
f = fs[np.argmin([amdahl_resid(x) for x in fs])]

print(f"threads:  {list(th)}")
print(f"speedup:  {np.round(sp, 2).tolist()}")
print(f"efficiency: {np.round(sp / th, 3).tolist()}")
print(f"fitted serial fraction f = {f:.4f}  ->  ceiling {1/f:.1f}x" if f > 0
      else "fitted serial fraction f = 0 (no measurable serial part)")

fig, ax = plt.subplots(figsize=(7.6, 4.35))
ax.plot(th, th, color=FAINT, lw=1.6, ls=(0, (4, 4)), zorder=1)
ax.text(th[-1], th[-1], " ideal", color=FAINT, fontsize=17, va="center")

pf = np.linspace(th[0], th[-1], 200)
ax.plot(pf, 1.0 / (f + (1 - f) / pf), "-", color=INK, lw=2.2, alpha=0.5, zorder=2)
ax.plot(th, sp, "-", color=RUST, lw=2.6, zorder=3)
ax.plot(th, sp, "o", color=RUST, markersize=11, markeredgecolor=GROUND,
        markeredgewidth=1.6, zorder=4)

ax.text(0.03, 0.95, f"Amdahl fit:  serial fraction  f = {f:.3f}",
        transform=ax.transAxes, fontsize=17, color=INK, va="top")
ax.text(0.03, 0.87, f"L = {meta.get('L','?')},  P = {int(meta.get('P',0)):,} "
        f"sites decided per round", transform=ax.transAxes,
        fontsize=17, color=FAINT, va="top")

ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xticks(th); ax.set_xticklabels([str(t) for t in th])
ax.set_yticks(th); ax.set_yticklabels([str(t) for t in th])
ax.set_xlabel("threads", labelpad=10)
ax.set_ylabel("speedup vs 1 thread")
ax.grid(alpha=0.55, color=GRID, lw=1.1)
ax.set_axisbelow(True)

fig.savefig("figD_cpu.svg", format="svg", bbox_inches="tight", transparent=True)
print("wrote figD_cpu.svg")
