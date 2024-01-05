import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import matplotlib.mlab as mlab

BINS = 1000
afont = {'fontname': 'Arial'}
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 35
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["axes.linewidth"] = 3
#set xrange to 0-2000
plt.rcParams["axes.spines.right"] = True
plt.rcParams["axes.spines.top"] = True
plt.rcParams["xtick.major.size"] = 10
plt.rcParams["ytick.major.size"] = 10
plt.rcParams["xtick.major.width"] = 3
plt.rcParams["ytick.major.width"] = 3
plt.rcParams["xtick.minor.size"] = 5
plt.rcParams["ytick.minor.size"] = 5
plt.rcParams["xtick.minor.width"] = 3
plt.rcParams["ytick.minor.width"] = 3
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["xtick.labelsize"] = 35
plt.rcParams["ytick.labelsize"] = 35
plt.rcParams["xtick.major.pad"] = 10
plt.rcParams["ytick.major.pad"] = 10
plt.rcParams["axes.labelpad"] = 20
plt.rcParams["axes.labelsize"] = 35
plt.rcParams["legend.frameon"] = False
plt.rcParams["legend.fontsize"] = 35
#plt.rcParams["legend.loc"] = "upper right"
plt.rcParams["legend.handlelength"] = 1.5
plt.rcParams["legend.handletextpad"] = 0.5
plt.rcParams["legend.labelspacing"] = 0.5
plt.rcParams["legend.borderpad"] = 0.5
plt.rcParams["hist.bins"] = 1000

fig = plt.figure(figsize = (20,18), dpi=600)
ax = fig.add_subplot(111)
ax.set_xlabel("MCS")
ax.set_ylabel("Energy/atom (meV)")
ax.set_xlim(1,2000)
ax.set_ylim(-25.0,.0)
for i in range(5):
    data = np.load(f"BL_Energy_{256*2**i}_12.50.npy")
    plt.plot(-data)
plt.legend(["6.25%","12.5%","25.0%","50.0%","100.0%"])
ax.set_xscale("log")
plt.savefig(f"BL_Energy_evol_{12.50}.png")
plt.close()
fig = plt.figure(figsize = (20,18), dpi=600)
ax = fig.add_subplot(111)
ax.set_xlabel("MCS")
ax.set_ylabel("Correlation Ratio")
ax.set_xlim(1,2000)
ax.set_ylim(0.0,0.4)
for i in range(5):
    data = np.load(f"BL_Correlation_{256*2**i}_12.50.npy")
    plt.plot(data)
plt.legend(["6.25%","12.5%","25.0%","50.0%","100.0%"])
ax.set_xscale("log")
plt.savefig(f"BL_Correlation_evol_{12.50}.png")
plt.close()
fig = plt.figure(figsize = (20,18), dpi=600)
ax = fig.add_subplot(111)
ax.set_xlabel("MCS")
ax.set_ylabel("Acceptance Ratio")
ax.set_xlim(1,2000)
ax.set_ylim(0.6,1.05)
for i in range(5):
    data = np.load(f"BL_Acceptance_{256*2**i}_12.50.npy")
    plt.plot(data)
plt.legend(["6.25%","12.5%","25.0%","50.0%","100.0%"])
ax.set_xscale("log")
plt.savefig(f"BL_Acceptance_evol_{12.50}.png")
plt.close()
fig = plt.figure(figsize = (20,18), dpi=600)
for i in range(5):
    dE = np.gradient(-np.load(f"BL_Energy_{256*2**i}_12.50.npy"))
    m, s = norm.fit(dE)
    x = np.linspace(-1.0, 1.0, BINS)
    p = norm.pdf(x, m, s)
    plt.plot(x, 150*p)
plt.legend(["6.25%","12.5%","25.0%","50.0%","100.0%"])
plt.xlabel("ΔE/atom")
plt.ylim(0,1200)
plt.ylabel("Count")
plt.savefig(f"BL_dE_dist_{12.50}.png")
plt.close()

fig = plt.figure(figsize = (20,18), dpi=600)
ax = fig.add_subplot(111)
ax.set_xlabel("MCS")
ax.set_ylabel("Energy/atom (meV)")
ax.set_xlim(1,2000)
ax.set_ylim(-25.0,0)
for i in range(5):
    data = np.load(f"BL_Energy_{256*2**i}_10.00.npy")
    plt.plot(-data)
plt.legend(["6.25%","12.5%","25.0%","50.0%","100.0%"])
ax.set_xscale("log")
plt.savefig(f"BL_Energy_evol_{10.00}.png")
plt.close()
fig = plt.figure(figsize = (20,18), dpi=600)
ax = fig.add_subplot(111)
ax.set_xlabel("MCS")
ax.set_ylabel("Correlation Ratio")
ax.set_xlim(1,2000)
ax.set_ylim(0.0,0.4)
for i in range(5):
    data = np.load(f"BL_Correlation_{256*2**i}_10.00.npy")
    plt.plot(data)
plt.legend(["6.25%","12.5%","25.0%","50.0%","100.0%"])
ax.set_xscale("log")
plt.savefig(f"BL_Correlation_evol_{10.00}.png")
plt.close()
fig = plt.figure(figsize = (20,18), dpi=600)
ax = fig.add_subplot(111)
ax.set_xlabel("MCS")
ax.set_ylabel("Acceptance Ratio")
ax.set_xlim(1,2000)
ax.set_ylim(0.6,1.05)
for i in range(5):
    data = np.load(f"BL_Acceptance_{256*2**i}_10.00.npy")
    plt.plot(data)  
plt.legend(["6.25%","12.5%","25.0%","50.0%","100.0%"])
ax.set_xscale("log")
plt.savefig(f"BL_Acceptance_evol_{10.00}.png")
plt.close()
fig = plt.figure(figsize = (20,18), dpi=600)
for i in range(5):
    dE = np.gradient(-np.load(f"BL_Energy_{256*2**i}_10.00.npy"))
    m, s = norm.fit(dE)
    x = np.linspace(-1.0, 1.0, BINS)
    p = norm.pdf(x, m, s)
    plt.plot(x, p*150)
plt.legend(["6.25%","12.5%","25.0%","50.0%","100.0%"])
plt.ylim(0,1200)
plt.savefig(f"BL_dE_dist_{10.00}.png")
plt.xlabel("ΔE/atom")
plt.ylabel("Count")
plt.close()


fig = plt.figure(figsize = (20,18), dpi=600)
ax = fig.add_subplot(111)
ax.set_xlabel("MCS")
ax.set_ylabel("Energy/atom (meV)")
ax.set_xlim(1,2000)
ax.set_ylim(-25.0,.0)
for i in range(5):
    data = np.load(f"BQ_Energy_{256*2**i}_12.50.npy")
    plt.plot(-data)
plt.legend(["6.25%","12.5%","25.0%","50.0%","100.0%"])
ax.set_xscale("log")
plt.savefig(f"BQ_Energy_evol_{12.50}.png")
plt.close()
fig = plt.figure(figsize = (20,18), dpi=600)
ax = fig.add_subplot(111)
ax.set_xlabel("MCS")
ax.set_ylabel("Correlation Ratio")
ax.set_xlim(1,2000)
ax.set_ylim(0.0,0.4)
for i in range(5):
    data = np.load(f"BQ_Correlation_{256*2**i}_12.50.npy")
    plt.plot(data)
plt.legend(["6.25%","12.5%","25.0%","50.0%","100.0%"])
ax.set_xscale("log")
plt.savefig(f"BQ_Correlation_evol_{12.50}.png")
plt.close()
fig = plt.figure(figsize = (20,18), dpi=600)
ax = fig.add_subplot(111)
ax.set_xlabel("MCS")
ax.set_ylabel("Acceptance Ratio")
ax.set_xlim(1,2000)
ax.set_ylim(0.6,1.05)
for i in range(5):
    data = np.load(f"BQ_Acceptance_{256*2**i}_12.50.npy")
    plt.plot(data)
plt.legend(["6.25%","12.5%","25.0%","50.0%","100.0%"])
ax.set_xscale("log")
plt.savefig(f"BQ_Acceptance_evol_{12.50}.png")
plt.close()
fig = plt.figure(figsize = (20,18), dpi=600)
for i in range(5):
    dE = np.gradient(-np.load(f"BQ_Energy_{256*2**i}_12.50.npy"))
    m, s = norm.fit(dE)
    x = np.linspace(-1.0, 1.0, BINS)
    p = norm.pdf(x, m, s)
    plt.plot(x, p*150)
plt.legend(["6.25%","12.5%","25.0%","50.0%","100.0%"])
plt.xlabel("ΔE/atom")
plt.ylabel("Count")
plt.ylim(0,1200)
plt.savefig(f"BQ_dE_dist_{12.50}.png")
plt.close()

fig = plt.figure(figsize = (20,18), dpi=600)
ax = fig.add_subplot(111)
ax.set_xlabel("MCS")
ax.set_ylabel("Energy/atom (meV)")
ax.set_xlim(1,2000)
ax.set_ylim(-25.0,0)
for i in range(5):
    data = np.load(f"BQ_Energy_{256*2**i}_10.00.npy")
    plt.plot(-data)
plt.legend(["6.25%","12.5%","25.0%","50.0%","100.0%"])
ax.set_xscale("log")
plt.savefig(f"BQ_Energy_evol_{10.00}.png")
plt.close()
fig = plt.figure(figsize = (20,18), dpi=600)
ax = fig.add_subplot(111)
ax.set_xlabel("MCS")
ax.set_ylabel("Correlation Ratio")
ax.set_xlim(1,2000)
ax.set_ylim(0.0,0.4)
for i in range(5):
    data = np.load(f"BQ_Correlation_{256*2**i}_10.00.npy")
    plt.plot(data)
plt.legend(["6.25%","12.5%","25.0%","50.0%","100.0%"])
ax.set_xscale("log")
plt.savefig(f"BQ_Correlation_evol_{10.00}.png")
plt.close()
fig = plt.figure(figsize = (20,18), dpi=600)
ax = fig.add_subplot(111)
ax.set_xlabel("MCS")
ax.set_ylabel("Acceptance Ratio")
ax.set_xlim(1,2000)
ax.set_ylim(0.6,1.05)
for i in range(5):
    data = np.load(f"BQ_Acceptance_{256*2**i}_10.00.npy")
    plt.plot(data)
plt.legend(["6.25%","12.5%","25.0%","50.0%","100.0%"])
ax.set_xscale("log")
plt.savefig(f"BQ_Acceptance_evol_{10.00}.png")
plt.close()
fig = plt.figure(figsize = (20,18), dpi=600)
for i in range(5):
    dE = np.gradient(-np.load(f"BQ_Energy_{256*2**i}_10.00.npy"))
    m, s = norm.fit(dE)
    x = np.linspace(-1.0, 1.0, BINS)
    p = norm.pdf(x, m, s)
    plt.plot(x, p)
plt.legend(["6.25%","12.5%","25.0%","50.0%","100.0%"])
plt.xlabel("ΔE/atom")
plt.ylabel("Count")
plt.savefig(f"BQ_dE_dist_{10.00}.png")
plt.close()


