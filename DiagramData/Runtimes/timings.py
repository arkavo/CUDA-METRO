import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.size"] = 55
matplotlib.rcParams["lines.linewidth"] = 5
matplotlib.rcParams["axes.linewidth"] = 5
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
plt.rcParams["legend.loc"] = "upper left"
matplotlib.rcParams["legend.handlelength"] = 1.5
matplotlib.rcParams["legend.handletextpad"] = 0.5

x = np.linspace(2, 12, 6)
x = x*128 + 256

timings_4070ti = np.load("Timings_4070Ti.npy")
timings_a100 = np.load("Timings_A100SXM4.npy")
timings_v100 = np.load("Timings_V100SXM2.npy")

fig = plt.figure(figsize=(40,30), dpi=600)
ax = fig.add_subplot(111)

plt.plot(x, timings_4070ti, label="RTX 4070 Ti")
plt.plot(x, timings_a100, label="A100 SXM4")
plt.plot(x, timings_v100, label="V100 SXM2")
plt.legend(["RTX 4070 Ti", "A100 SXM4", "V100 SXM2"])
#plt.title("Timings vs Grid Size")
plt.xlabel("Grid Size (n)")
plt.ylabel("Time to stablize nxn lattice (s)")
plt.savefig("Timings.png")
plt.close()

runs_4070ti = np.load("Runs_4070Ti.npy")
runs_a100 = np.load("Runs_A100SXM4.npy")
runs_v100 = np.load("Runs_V100SXM2.npy")

fig = plt.figure(figsize=(40,30), dpi=600)

plt.plot(x,10.0/12.0* runs_4070ti/2310.0, label="RTX 4070 Ti")
plt.plot(x,10.0/40.0* runs_a100/1095.0, label="A100 SXM4")
plt.plot(x,10.0/32.0* runs_v100/1245.0, label="V100 SXM2")

plt.legend(["RTX 4070 Ti", "A100 SXM4", "V100 SXM2"])   
#plt.title("Runs vs Grid Size")
plt.xlabel("Grid Size (n)")
plt.ylabel("Time to stablize nxn lattice (s)")
plt.savefig("Runs.png")
plt.close()

