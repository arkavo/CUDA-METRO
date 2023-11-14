import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


flist = []
for f in os.listdir():
    if f.endswith(".npy") and f.startswith("grid"):
        flist.append(f)
flist.sort()
#shape = data.shape
ct = 0
#data = data.reshape((64,64,3))
for file in flist:
    data = np.load(file)
    print(data)
    print(data.shape)
    fig = plt.figure(figsize=(20,11), dpi=300)
    data = data.reshape((128,128,3))
    spinx = data[:,:,0]
    spiny = data[:,:,1]
    spinz = data[:,:,2]
    vmin, vmax = -2.0, 2.0
    ax = fig.add_subplot(131)
    ax = sns.heatmap(spinx, cbar=False, square=True, cmap="RdBu", vmin=vmin, vmax=vmax)
    ax = fig.add_subplot(132)
    ax = sns.heatmap(spiny, cbar=False, square=True, cmap="RdBu", vmin=vmin, vmax=vmax)
    ax = fig.add_subplot(133)
    ax = sns.heatmap(spinz, cbar=False, square=True, cmap="RdBu", vmin=vmin, vmax=vmax)
    ct += 1
    plt.savefig(f"FePS3_{ct}.png")
    plt.close()