import numpy as np
import matplotlib.pyplot as plt

data = np.load("Timings.npy")
X = np.ones_like(data)
for i in range(len(data)):
    X[i] = 64*2**i
XLabel = X.astype(int)
X = np.log(X)/np.log(2)
data = np.log(data)
fig = plt.figure(figsize=(20,10), dpi=400)
plt.plot(X, data)
plt.title("Timings on RTX 4070Ti")
plt.xlabel("Size")
plt.ylabel("Time (s)")
plt.xticks(X, XLabel)
plt.savefig("Timings.png")