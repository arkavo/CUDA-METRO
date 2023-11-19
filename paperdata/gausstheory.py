import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-1.0, 1.0, 401)
y = np.zeros_like(x)
for i in range(401):
    if x[i] < 0:
        y[i] = 1.0
    else:
        y[i] = np.exp(-x[i])

def En_L(L):
    dE = 0.0
    for i in range(L):
        e = np.random.randint(401)
        if np.random.rand() < y[e]:
            dE += x[e]
    return -dE/L

Ex0, Ex1, Ex2, Ex3, Ex4 = np.zeros(1000), np.zeros(1000), np.zeros(1000), np.zeros(1000), np.zeros(1000)

for i in range(1000):
    Ex0[i] = En_L(256)
    Ex1[i] = En_L(512)
    Ex2[i] = En_L(1024)
    Ex3[i] = En_L(2048)
    Ex4[i] = En_L(4096)

fig = plt.figure(figsize=(20,10), dpi=400)
fig.add_subplot(231)
plt.hist(Ex0,bins=50, alpha=0.8)
plt.title("256 6.25")
fig.add_subplot(232)
plt.hist(Ex1,bins=50, alpha=0.8)
plt.title("512 12.5")
fig.add_subplot(233)
plt.hist(Ex2,bins=50, alpha=0.8)
plt.title("1024 25")
fig.add_subplot(234)
plt.hist(Ex3,bins=50, alpha=0.8)
plt.title("2048 50")
fig.add_subplot(235)
plt.hist(Ex4,bins=50, alpha=0.8)
plt.title("4096 100")
fig.add_subplot(236)
mu0, sigma0 = norm.fit(Ex0)
mu1, sigma1 = norm.fit(Ex1)
mu2, sigma2 = norm.fit(Ex2)
mu3, sigma3 = norm.fit(Ex3)
mu4, sigma4 = norm.fit(Ex4)
plt.plot(x, y, 'r-', linewidth=1)
plt.plot(x, norm.pdf(x, mu0, sigma0), 'b-', linewidth=0.9)
plt.plot(x, norm.pdf(x, mu1, sigma1), 'g-', linewidth=0.9)
plt.plot(x, norm.pdf(x, mu2, sigma2), 'y-', linewidth=0.9)
plt.plot(x, norm.pdf(x, mu3, sigma3), 'c-', linewidth=0.9)
plt.plot(x, norm.pdf(x, mu4, sigma4), 'm-', linewidth=0.9)
plt.title("Gaussian fit")
plt.legend(["Target","256","512","1024","2048","4096"])
plt.xlim(-0.25,0.25)
plt.savefig(f"En.png")
