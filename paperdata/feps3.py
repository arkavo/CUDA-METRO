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
    data = data.reshape((64,64,3))
    spinx = data[:,:,0]
    spiny = data[:,:,1]
    spinz = data[:,:,2]
    vmin, vmax = -2.0, 2.0
    ax = fig.add_subplot(111)
    ax = sns.heatmap(spinx, cbar=True, square=True, cmap="RdBu", vmin=vmin, vmax=vmax)
    '''
    ax = fig.add_subplot(132)
    ax = sns.heatmap(spiny, cbar=False, square=True, cmap="RdBu", vmin=vmin, vmax=vmax)
    ax = fig.add_subplot(133)
    ax = sns.heatmap(spinz, cbar=False, square=True, cmap="RdBu", vmin=vmin, vmax=vmax)
    ct += 1
    '''
    plt.savefig(f"FePS3_{ct}.png")
    plt.close()
    fig = plt.figure(figsize=(10,10), dpi=300)
    ax = fig.add_subplot(111)
    data = data.reshape((4096,3))
    x_mesh, y_mesh = np.meshgrid(np.arange(0, 64, 1), np.arange(0, 64, 1))
    figure = plt.figure(figsize=(10,10),dpi=400)
    plt.title("Spin Configuration at T = "+str(ctr))
    ax = figure.add_subplot(111)
    rgba = np.zeros((4096,4))
    spinz = np.reshape(spinz, 64)
    for i in range(4096):
        rgba[i][3] = 1.0
        rgba[i][1] = 0.0
        if spinz[i] > 0:
            rgba[i][0] = spinz[i]/self.spin
            rgba[i][2] = 0.0
        else:
            rgba[i][0] = 0.0
            rgba[i][2] = -spinz[i]/self.spin
    plt.quiver(x_mesh, y_mesh, spinx, spiny, scale=self.spin, scale_units="xy", pivot="mid", color=rgba, width=0.01, headwidth=3, headlength=4, headaxislength=3, minlength=0.1, minshaft=1)