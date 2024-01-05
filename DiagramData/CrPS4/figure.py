import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 42
#set axis tick size
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24

M = np.load("BQM_64_256_625.npy")/np.load("BQM_64_256_625.npy").max()
X = np.load("BQX_64_256_625.npy")
X[0] = 0.0
X = X/X.max()
T = np.linspace(0.01,93.3,21)

figure = plt.figure(figsize=(18,16),dpi=600)
ax = figure.add_subplot(111)
ax.plot(T,M,"r-",label="Magnetization",linewidth=3.0)
ax.plot(T,X,"b-",label="Susceptibility",linewidth=3.0)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Normalized Magnetization/Susceptibility")
ax.legend()
ax.set_xlim(0,93.3)
ax.axvline(x=41.992,color="k",linestyle="-",linewidth=3.0)
figure.savefig("BQM_X.png",dpi=600,bbox_inches="tight",pad_inches=0.1,transparent=False)