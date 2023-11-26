import numpy as np
import seaborn as sns

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

vl = 1.5

for i in range(5):
    data = np.load(f"BL{256*2**i}_grid_12.5.npy")
    data = np.reshape(data, (64, 64, 3))
    figure = sns.heatmap(data[:,:,2], cmap="coolwarm", vmin=-vl, vmax=vl)
    figure.set_title(f"BL{256*2**i} at 12.50 K")
    figure.figure.savefig(f"BL{256*2**i}_grid_12.50.png")
    figure.figure.clf()
    data = np.load(f"BL{256*2**i}_grid_10.0.npy")
    data = np.reshape(data, (64, 64, 3))
    figure = sns.heatmap(data[:,:,2], cmap="coolwarm", vmin=-vl, vmax=vl)
    figure.set_title(f"BL{256*2**i} at 10.00 K")
    figure.figure.savefig(f"BL{256*2**i}_grid_10.00.png")
    figure.figure.clf()

for i in range(5):
    data = np.load(f"BQ{256*2**i}_grid_12.5.npy")
    data = np.reshape(data, (64, 64, 3))
    figure = sns.heatmap(data[:,:,2], cmap="coolwarm", vmin=-vl, vmax=vl)
    figure.set_title(f"BQ{256*2**i} at 12.50 K")
    figure.figure.savefig(f"BQ{256*2**i}_grid_12.50.png")
    figure.figure.clf()
    data = np.load(f"BQ{256*2**i}_grid_10.0.npy")
    data = np.reshape(data, (64, 64, 3))
    figure = sns.heatmap(data[:,:,2], cmap="coolwarm", vmin=-vl, vmax=vl)
    figure.set_title(f"BQ{256*2**i} at 10.00 K")
    figure.figure.savefig(f"BQ{256*2**i}_grid_10.00.png")
    figure.figure.clf()
