import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.size"] = 35
matplotlib.rcParams["lines.linewidth"] = 3
matplotlib.rcParams["axes.linewidth"] = 3
#set xrange to 0-2000
matplotlib.rcParams["axes.spines.right"] = True
matplotlib.rcParams["axes.spines.top"] = True
matplotlib.rcParams["xtick.minor.size"] = 0
matplotlib.rcParams["ytick.minor.size"] = 0
matplotlib.rcParams["xtick.minor.width"] = 0
matplotlib.rcParams["ytick.minor.width"] = 0
matplotlib.rcParams["xtick.direction"] = "out"
matplotlib.rcParams["ytick.direction"] = "out"
matplotlib.rcParams["xtick.labelsize"] = 35
matplotlib.rcParams["ytick.labelsize"] = 35
matplotlib.rcParams["axes.labelpad"] = 20
matplotlib.rcParams["axes.labelsize"] = 35
matplotlib.rcParams["legend.frameon"] = False
matplotlib.rcParams["legend.fontsize"] = 35
plt.rcParams["legend.loc"] = "upper right"
matplotlib.rcParams["legend.handlelength"] = 1.5
matplotlib.rcParams["legend.handletextpad"] = 0.5


data = np.load("Timings.npy")
X = np.ones_like(data)
for i in range(len(data)):
    X[i] = 64*2**i
XLabel = X.astype(int)
X = np.log(X)/np.log(2)
data = np.log(data)
fig = plt.figure(figsize=(40,30), dpi=400)
ax = fig.add_subplot(111)
plt.plot(X, data)
ax.set_xscale("log", base=2)
#ax.set_yscale("log")
plt.xlabel("Size")
plt.ylabel("Time for 1E5 MCS(s)")
plt.xticks(X, XLabel)
plt.savefig("Timings.png")