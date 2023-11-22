import numpy as np
import matplotlib.pyplot as plt

folder0 = "256"


data0 = np.load(f"{folder0}/Energy_256_0.01.npy")
data1 = np.load(f"{folder0}/Energy_256_11.25.npy")
data2 = np.load(f"{folder0}/Energy_256_22.50.npy")
data3 = np.load(f"{folder0}/Energy_256_45.00.npy")

plt.plot(data0[0:150])
plt.plot(data1[0:150])
#plt.plot(data2[0:150])
#plt.plot(data3[0:150])

plt.title(f"En Evolution")
plt.savefig(f"En evolution.png")
plt.close()
data00 = np.zeros(150)
data01 = np.zeros(150)
#data02 = np.zeros(150)
#data03 = np.zeros(150)

for i in range(150):
    data00[i] = data0[i+1] - data0[i]
    data01[i] = data1[i+1] - data1[i]
    #data02[i] = data2[i+1] - data2[i]
    #data03[i] = data3[i+1] - data3[i]

plt.plot(data00)
plt.plot(data01)
#plt.plot(data02)
#plt.plot(data03)
plt.title(f"dE")
plt.savefig(f"dE.png")
plt.close()
plt.hist(data00, bins=10, alpha=0.4)
plt.hist(data01, bins=10, alpha=0.4)
#plt.hist(data02, bins=100, alpha=0.4)
#plt.hist(data03, bins=100, alpha=0.4)
plt.title(f"dE Hist")
plt.savefig(f"dE Hist.png")
plt.close()