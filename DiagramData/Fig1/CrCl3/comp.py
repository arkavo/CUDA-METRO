import numpy as np
import matplotlib.pyplot as plt

bqm, bqx = np.loadtxt("GREENCLM.txt", delimiter=","), np.loadtxt("GREENCLX.txt", delimiter=",")

gm, gx = np.load("BQM_220_512.npy"), np.load("BQX_220_512.npy")
T = np.linspace(0.01, 25.5, 41)
Trefm = bqm[:,0]
Trefx = bqx[:,0] - 9.0
plt.plot(Trefm, bqm[:,1], label="BQM")
plt.plot(T, gm/1.5, label="BQM_220_512")
plt.legend(["BQM", "BQM_220_512"])
plt.title("Magnetization vs Temperature of CrCl3 at size 512x512")
plt.savefig("M_512.png")
plt.close()

plt.plot(Trefx, np.exp(bqx[:,1])/np.max(np.exp(bqx[:,1])), label="BQX")
plt.plot(T, gx/np.max(gx), label="BQX_220_512")
plt.legend(["BQX", "BQX_220_512"])
plt.title("Susceptibility vs Temperature of CrCl3 at size 512x512")
plt.savefig("X_512.png")
plt.close()


