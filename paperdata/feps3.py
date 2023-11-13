import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = np.load("grid_0000.npy")
shape = data.shape
data = data.reshape((256,256,3))

fig = plt.figure(figsize=(20,11), dpi=300)
spinx = data[:,:,0]
spiny = data[:,:,1]
spinz = data[:,:,2]

ax = fig.add_subplot(131)
ax = sns.heatmap(spinx, cbar=False, square=True)
ax = fig.add_subplot(132)
ax = sns.heatmap(spiny, cbar=False, square=True)
ax = fig.add_subplot(133)
ax = sns.heatmap(spinz, cbar=False, square=True)

plt.savefig("FePS3.png")