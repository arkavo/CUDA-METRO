"""
M vs H hysteresis loop for CrPS4-ML (4-layer multilayer).

Field sweep per temperature:  0 → +H_max → -H_max → +H_max
"""
import os, datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import seaborn as sns
import pycuda.driver as drv
import construct as cst

# ── Material constants for emu/g conversion ───────────────────────────────────
MU_B_CGS   = 9.2741e-21   # emu  (= erg/G, CGS Bohr magneton)
N_A        = 6.02214e23
G_LANDE    = 2.0
SPIN       = 1.5
M_MOLAR    = 51.996 + 30.974 + 4*32.06   # CrPS4  = 211.21 g/mol
# Saturation magnetisation in emu/g (M_norm = 1 → fully aligned)
EMU_G_SAT  = G_LANDE * MU_B_CGS * SPIN * N_A / M_MOLAR   # ≈ 79.3 emu/g

# ── Experiment parameters ─────────────────────────────────────────────────────
LAYERS       = 12
TEMPERATURES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]   # K
H_MAX        = 1.5                        # normalised field (same units as spin S)
N_STEPS      = 4                          # field points per branch
EQ_WRAP      = 10                         # stability_wrap passes per field point

# Full sweep: 0 → +H_max → -H_max → +H_max  (three branches)
h1 = np.linspace(0,      H_MAX,  N_STEPS,   endpoint=False)
h2 = np.linspace(H_MAX, -H_MAX, 2*N_STEPS,  endpoint=False) # 2 x N_STEPS
h3 = np.linspace(-H_MAX, H_MAX, 2*N_STEPS,  endpoint=True)  # 2 x N_STEPS
H_sweep = np.concatenate([h1, h2, h3])
n1, n2, n3 = len(h1), len(h2), len(h3)

SNAP_INDICES = [int(np.argmin(np.abs(H_sweep - 0.0)))]

# ── Output directory ──────────────────────────────────────────────────────────
out_dir = "Hysteresis_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.makedirs(out_dir, exist_ok=True)

# ── Main loop ─────────────────────────────────────────────────────────────────
results = {}

for T in TEMPERATURES:
    print(f"\n{'─'*55}")
    print(f"  T = {T} K   ({len(H_sweep)} field points)")
    print(f"{'─'*55}")

    sim = cst.MonteCarlo(config="hysteresis_config.json")
    sim.mc_init_multilayer(layers=LAYERS)

    M_total_list  = []
    M_layers_list = []
    snapshots     = {}   # {actual_H_value: grid_copy}

    for idx, h_val in enumerate(H_sweep):
        # Broadcast a uniform field to all SIZE columns on the GPU.
        # For a spatially varying field use sim.set_field_zones() instead;
        # see multilayer.py for an example.
        sim.set_field(h_val)

        for _ in range(EQ_WRAP):
            sim.generate_random_numbers_multilayer(None)
            sim.run_mc_3636_multilayer(T, layers=LAYERS)

        M_tot, M_lay = sim.get_magnetization(LAYERS)
        M_total_list.append(M_tot)
        M_layers_list.append(M_lay.copy())

        if idx in SNAP_INDICES:
            snapshots[h_val] = sim.grid.copy()

        branch = "0→+" if idx < n1 else ("+→-" if idx < n1+n2 else "-→+")
        print(f"  [{branch}] H = {h_val:+6.3f}  Mz = {M_tot:+.4f}  "
              f"({M_tot*EMU_G_SAT:+.2f} emu/g)", end="\r")

    print()
    results[T] = {
        "H":         H_sweep,
        "M_total":   np.array(M_total_list),
        "M_layers":  np.array(M_layers_list),   # shape (N_H, 12)
        "snapshots": snapshots,                  # 5 grids at key field values
    }

# ── Save raw data ─────────────────────────────────────────────────────────────
np.save(os.path.join(out_dir, "hysteresis_data.npy"),
        results, allow_pickle=True)

# ── Plot 1: total M/M_sat + emu/g vs H, all temperatures ─────────────────────
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax2_r = ax.twinx()                          # right axis: emu/g
colors = cm.plasma(np.linspace(0.1, 0.85, len(TEMPERATURES)))

for T, col in zip(TEMPERATURES, colors):
    r = results[T]
    H = r["H"]
    M = r["M_total"]
    ax.plot(H[:n1],      M[:n1],      color=col, lw=1.8, label=f"T = {T} K")
    ax.plot(H[n1:n1+n2], M[n1:n1+n2], color=col, lw=1.8, ls="--")
    ax.plot(H[n1+n2:],   M[n1+n2:],   color=col, lw=1.8, ls=":")

ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
ax.set_xlabel("H (normalised)", fontsize=12)
ax.set_ylabel(r"$M_z / M_\mathrm{sat}$", fontsize=12)

# Mirror the left axis limits onto emu/g scale
ylo, yhi = ax.get_ylim()
ax2_r.set_ylim(ylo * EMU_G_SAT, yhi * EMU_G_SAT)
ax2_r.set_ylabel(r"$M_z$ (emu/g)", fontsize=12)

ax.set_title(r"$M_z$–H Hysteresis Loop — CrPS4-ML (4 layers)", fontsize=13)

style_legend = [
    Line2D([0],[0], color="grey", lw=1.5, ls="-",  label="0 → +H"),
    Line2D([0],[0], color="grey", lw=1.5, ls="--", label="+H → -H"),
    Line2D([0],[0], color="grey", lw=1.5, ls=":",  label="-H → +H"),
]
ax.legend(handles=ax.get_legend_handles_labels()[0] + style_legend, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "MH_all_temps.png"))
plt.close()
print("Saved MH_all_temps.png")

# ── Plot 2: per-layer M vs H — one figure per temperature ─────────────────────
layer_colors = cm.tab10(np.linspace(0, 0.4, LAYERS))
for T_ref in TEMPERATURES:
    r  = results[T_ref]
    H  = r["H"]
    Ml = r["M_layers"]

    fig2, axes = plt.subplots(1, LAYERS, figsize=(5*LAYERS, 4), dpi=200, sharey=True)
    for i in range(LAYERS):
        ax2 = axes[i]
        ax2.plot(H[:n1],      Ml[:n1, i],      color=layer_colors[i], lw=1.8)
        ax2.plot(H[n1:n1+n2], Ml[n1:n1+n2, i], color=layer_colors[i], lw=1.8, ls="--")
        ax2.plot(H[n1+n2:],   Ml[n1+n2:,  i],  color=layer_colors[i], lw=1.8, ls=":")
        ax2.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax2.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax2.set_title(f"Layer {i+1}", fontsize=11)
        ax2.set_xlabel("H (normalised)", fontsize=10)
    axes[0].set_ylabel(r"$M_z / M_\mathrm{sat}$", fontsize=11)
    fig2.suptitle(r"Per-layer $M_z$–H — CrPS4-ML, T = " + f"{T_ref} K", fontsize=12)
    plt.tight_layout()
    fname = f"MH_per_layer_T{T_ref}K.png"
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()
    print(f"Saved {fname}")

# ── Plot 3: spin heatmaps at 5 field snapshots — one figure per (T, H) ────────
comp_labels = ["Sx", "Sy", "Sz"]
spin_val = float(sim.spin)
for T_ref in TEMPERATURES:
    h_snap, grid = next(iter(results[T_ref]["snapshots"].items()))
    size = int(np.sqrt(grid.shape[0] // (LAYERS * 3)))
    g    = grid.reshape(LAYERS, size, size, 3)

    fig3, axs = plt.subplots(3, LAYERS, figsize=(5*LAYERS, 15), dpi=150)
    if LAYERS == 1:
        axs = axs.reshape(3, 1)
    fig3.suptitle(
        f"Spin State — CrPS4-ML, T = {T_ref} K,  H = {h_snap:+.3f}",
        fontsize=13)
    for i in range(LAYERS):
        for c, clabel in enumerate(comp_labels):
            sns.heatmap(g[i, :, :, c], ax=axs[c][i],
                        cmap="coolwarm", vmin=-spin_val, vmax=spin_val,
                        square=True, xticklabels=False, yticklabels=False,
                        cbar=False)
            axs[c][i].set_title(f"Layer {i+1} — {clabel}", fontsize=10)
    plt.tight_layout()
    fname = f"spinstate_T{T_ref}K_H0.png"
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()
    print(f"Saved {fname}")

print(f"\nAll results saved in  {out_dir}/")
