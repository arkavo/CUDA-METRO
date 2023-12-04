import numpy as np
import matplotlib.pyplot as plt



x = np.linspace(2, 12, 6)
x = x*128 + 256

timings_4070ti = np.load("Timings_4070Ti.npy")
timings_a100 = np.load("Timings_A100SXM4.npy")
timings_v100 = np.load("Timings_V100SXM2.npy")

plt.plot(x, timings_4070ti, label="RTX 4070 Ti")
plt.plot(x, timings_a100, label="A100 SXM4")
plt.plot(x, timings_v100, label="V100 SXM2")
plt.legend(["RTX 4070 Ti", "A100 SXM4", "V100 SXM2"])
plt.title("Timings vs Grid Size")
plt.xlabel("Grid Size")
plt.ylabel("Timings (s)")
plt.savefig("Timings.png")
plt.close()

runs_4070ti = np.load("Runs_4070Ti.npy")
runs_a100 = np.load("Runs_A100SXM4.npy")
runs_v100 = np.load("Runs_V100SXM2.npy")

plt.plot(x,10.0/12.0* runs_4070ti/2310.0, label="RTX 4070 Ti")
plt.plot(x,10.0/40.0* runs_a100/1095.0, label="A100 SXM4")
plt.plot(x,10.0/32.0* runs_v100/1245.0, label="V100 SXM2")

plt.legend(["RTX 4070 Ti", "A100 SXM4", "V100 SXM2"])   
plt.title("Runs vs Grid Size")
plt.xlabel("Grid Size")
plt.ylabel("Time(s)")
plt.savefig("Runs.png")
plt.close()

