import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

files = os.listdir()
flist = []
flist = [file for file in files if file.endswith(".npy")]


for file in flist:
    grid = np.load(file)
    shape = grid.shape
    grid = grid.reshape((int(np.sqrt(shape[0]/3)), int(np.sqrt(shape[0]/3)), 3))
    spinx, spiny, spinz = grid[:,:,0], grid[:,:,1], grid[:,:,2]
    figure = plt.figure(figsize=(12,10), dpi=600)
    plt.title("Spin Configuration")
    sns.heatmap(spinz, cbar=True, cmap="coolwarm", square=True, xticklabels=False, yticklabels=False)
    plt.savefig("Sz_"+file+".png")
    plt.close()