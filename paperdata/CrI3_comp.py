import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (17,10), dpi=300)
T_res = np.linspace(0.01, 80.00, 31)
data_ref = np.genfromtxt("BLUE.txt", delimiter=',')
T = data_ref[:,0]
M = data_ref[:,1]
for i in range(4):
    data = np.load(f"M_{64*2**i}.npy")
    plt.plot(T_res[0:23],data[0:23]/1.5)

plt.legend(["64","128","256","512"])
plt.scatter(T,M)

plt.savefig("CrI3.png")