import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.colors import Normalize

files = os.listdir()
flist = []
flist = [file for file in files if file.endswith(".npy")]


for file in flist:
    grid = np.load(file)
    shape = grid.shape
    grid = grid.reshape((int(np.sqrt(shape[0]/3)), int(np.sqrt(shape[0]/3)), 3))
    spinx, spiny, spinz = grid[:,:,0], grid[:,:,1], grid[:,:,2]
    figure = plt.figure(figsize=(30,25), dpi=600)
    #plt.title("Spin Configuration")
    ax = sns.heatmap(spinz, cbar=True, cmap="coolwarm", square=True, xticklabels=False, yticklabels=False)
    plt.savefig("Sz_"+file+".png")
    plt.close()
    
    x_mesh, y_mesh = np.meshgrid(np.arange(0,spinx.shape[0]), np.arange(0,spinx.shape[1]))
    figure = plt.figure(figsize=(12,10), dpi=600)
    plt.title("Quiver Configuration")
    spinz = np.reshape(spinz, int(shape[0]/3))
    norm = Normalize()
    norm.autoscale(spinz)
    colormap = plt.cm.seismic
    plt.quiver(x_mesh, y_mesh, spinx, spiny, scale=1.61805625, scale_units="xy", pivot="mid", color=colormap(norm(spinz)), width=0.01, headwidth=3, headlength=4, headaxislength=3, minlength=0.1, minshaft=1)
    plt.savefig("Quiver_"+file+".png")
    plt.close()
    
    