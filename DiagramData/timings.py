import numpy as np
import matplotlib.pyplot as plt

timings = np.load("Timings.npy")
runs = np.load("Runs.npy")

x = np.linspace(2, 12, 6)
x = x*128 + 256

plt.plot(x, timings, label="Timings")
plt.title("Timings vs Grid Size")
plt.xlabel("Grid Size")
plt.ylabel("Timings (s)")
plt.savefig("Timings.png")
plt.close()

plt.plot(x, runs, label="Runs")
plt.title("Runs vs Grid Size")
plt.xlabel("Grid Size")
plt.ylabel("Runs")
plt.savefig("Runs.png")
plt.close()

