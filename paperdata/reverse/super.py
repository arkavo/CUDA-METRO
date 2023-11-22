import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab

folders = ["10.00","12.50"]
b = 50
a = 0.8
for folder in folders:
    data1 = np.load(f"{folder}/Energy_512_{folder}.npy")
    data2 = np.load(f"{folder}/Energy_1024_{folder}.npy")
    data3 = np.load(f"{folder}/Energy_2048_{folder}.npy")
    data4 = np.load(f"{folder}/Energy_4096_{folder}.npy")
    data0 = np.load(f"{folder}/Energy_256_{folder}.npy")

    fig = plt.figure(figsize=(20,10), dpi=400)
    plt.plot(-data0)
    plt.plot(-data1)
    plt.plot(-data2)
    plt.plot(-data3)
    plt.plot(-data4)
    plt.legend(["6.25","12.5","25","50","100"])
    plt.title(f"Energy evolution at {folder}K")
    plt.savefig(f"En_{folder}.png")
    plt.close()

    fig = plt.figure(figsize=(20,10), dpi=400)
    plt.plot(-np.gradient(data0))
    plt.plot(-np.gradient(data1))
    plt.plot(-np.gradient(data2))
    plt.plot(-np.gradient(data3))
    plt.plot(-np.gradient(data4))


    plt.legend(["6.25","12.5","25","50","100"])
    plt.title(f"Energy difference at {folder}K")
    plt.savefig(f"dE_{folder}.png")
    plt.close()

    fig = plt.figure(figsize=(20,10), dpi=400)
    fig.add_subplot(232)
    plt.hist(-np.gradient(data0),bins=b, alpha=a)
    plt.title("512 12.5")
    fig.add_subplot(233)
    plt.hist(-np.gradient(data1),bins=b, alpha=a)
    plt.title("1024 25")
    fig.add_subplot(234)
    plt.hist(-np.gradient(data2),bins=b, alpha=a)
    plt.title("2048 50")
    fig.add_subplot(235)
    plt.hist(-np.gradient(data3),bins=b, alpha=a)
    plt.title("4096 100")
    fig.add_subplot(231)
    plt.hist(-np.gradient(data4),bins=b, alpha=a)
    plt.title("256 6.25")
    fig.add_subplot(236)
    xmin, xmax = -1.0, 1.0
    x = np.linspace(xmin, xmax, b)
    m1, s1 = norm.fit(-np.gradient(data0))
    m2, s2 = norm.fit(-np.gradient(data1))
    m3, s3 = norm.fit(-np.gradient(data2))
    m4, s4 = norm.fit(-np.gradient(data3))
    m5, s5 = norm.fit(-np.gradient(data4))
    p1 = norm.pdf(x, m1, s1)
    p2 = norm.pdf(x, m2, s2)
    p3 = norm.pdf(x, m3, s3)
    p4 = norm.pdf(x, m4, s4)
    p5 = norm.pdf(x, m5, s5)
    plt.plot(x, p1,  linewidth=1)
    plt.plot(x, p2,  linewidth=1)
    plt.plot(x, p3,  linewidth=1)
    plt.plot(x, p4,  linewidth=1)
    plt.plot(x, p5,  linewidth=1)
    plt.title(f"Gaussian fit of dE at {folder}K")
    plt.legend(["256","512","1024","2048","4096"])
    plt.savefig(f"<H>dE_{folder}.png")
    plt.close()

    
