import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


l1,l2 = 10,10
r1,r2 = 110,110


grid = np.load("grid_0098.npy")
shape = grid.shape

grid = grid.reshape((int(np.sqrt(shape[0]/3)), int(np.sqrt(shape[0]/3)), 3))
print(grid.shape)
grid = grid[l1:r1, l2:r2, :]
#print(grid)
figure = plt.figure(figsize=(20,20), dpi=400)
ax = figure.add_subplot(111)
sns.heatmap(grid[:,:,2], cmap="coolwarm", cbar=False)
plt.savefig("Scutter.png")
plt.close()

figure = plt.figure(figsize=(20,20), dpi=400)
ax = figure.add_subplot(111)
x_mesh, y_mesh = np.meshgrid(np.arange(0,r1-l1,1), np.arange(0,r2-l2,1))
print(grid.shape)
rgba = np.zeros(((r1-l1) * (r2-l2), 4))
spinz = grid[:,:,2].reshape(((r1-l1) * (r2-l2)))
sp_max = np.max(np.abs(spinz))
for i in range((r1-l1)&(r2-l2)):
    rgba[i][3] = 1.0
    rgba[i][1] = 0.0
    if spinz[i] > 0:
        rgba[i][0] = spinz[i]/4.6
        rgba[i][2] = 0.0
    else:
        rgba[i][0] = 0.0
        rgba[i][2] = -spinz[i]/4.6    
plt.quiver(x_mesh, y_mesh, grid[:,:,0], grid[:,:,1], scale=1.526, scale_units="xy", pivot="mid", width=0.01, headwidth=3, headlength=4, headaxislength=3, minlength=0.1, minshaft=1)
plt.savefig("Qcutter.png")
plt.close()