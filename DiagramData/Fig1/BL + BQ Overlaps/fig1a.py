import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

blue_data = np.loadtxt("BLUE.txt", delimiter=",")
green_data = np.loadtxt("GREEN.txt", delimiter=",")

Figure_1A = plt.figure(figsize=(8, 6))
ax = Figure_1A.add_subplot(111)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Magnetization (μB)")
ax.set_title("Magnetization vs Temperature of CRI3-BL at size 64x64")
T_blue = blue_data[:, 0]
T_ref = np.linspace(0.01, 60.00, 41)
ax.scatter(T_blue, blue_data[:, 1], label="M")

blue0 = np.load("BLM_64_512.npy")
blue1 = np.load("BLM_64_1024.npy")
blue2 = np.load("BLM_64_2048.npy")
blue3 = np.load("BLM_64_4096.npy")

ax.plot(T_ref,blue0/1.5)
ax.plot(T_ref,blue1/1.5)
ax.plot(T_ref,blue2/1.5)
ax.plot(T_ref,blue3/1.5)

plt.legend(["Ref", "512", "1024", "2048", "4096"])
plt.savefig("Figure_1A.png")
plt.close()

Figure_1B = plt.figure(figsize=(8, 6))
ax = Figure_1B.add_subplot(111)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Magnetization (μB)")
ax.set_title("Magnetization vs Temperature of CRI3-BQ at size 64x64")
T_green = green_data[:, 0]
T_ref = np.linspace(0.01, 60.00, 41)
ax.scatter(T_green, green_data[:, 1], label="M")

green0 = np.load("BQM_64_512.npy")
green1 = np.load("BQM_64_1024.npy")
green2 = np.load("BQM_64_2048.npy")
green3 = np.load("BQM_64_4096.npy")

ax.plot(T_ref,green0/1.5)
ax.plot(T_ref,green1/1.5)
ax.plot(T_ref,green2/1.5)
ax.plot(T_ref,green3/1.5)

plt.legend(["Ref", "512", "1024", "2048", "4096"])
plt.savefig("Figure_1B.png")
plt.close()

