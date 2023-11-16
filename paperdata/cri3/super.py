import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab

folders = ["0.01","11.25","22.50","45.00"]
b = 50
a = 0.8
for folder in folders:
    data0 = np.load(f"{folder}/Energy_512_{folder}.npy")
    data1 = np.load(f"{folder}/Energy_1024_{folder}.npy")
    data2 = np.load(f"{folder}/Energy_2048_{folder}.npy")
    data3 = np.load(f"{folder}/Energy_4096_{folder}.npy")
    data4 = np.load(f"{folder}/Energy_256_{folder}.npy")

    fig = plt.figure(figsize=(20,10), dpi=400)
    plt.plot(-data0)
    plt.plot(-data1)
    plt.plot(-data2)
    plt.plot(-data3)
    plt.plot(-data4)
    plt.legend(["12.5","25","50","100","6.25"])
    plt.title(f"Energy evolution at {folder}K")
    plt.savefig(f"En_{folder}.png")
    plt.close()

    fig = plt.figure(figsize=(20,10), dpi=400)
    plt.plot(-np.gradient(data0))
    plt.plot(-np.gradient(data1))
    plt.plot(-np.gradient(data2))
    plt.plot(-np.gradient(data3))
    plt.plot(-np.gradient(data4))


    plt.legend(["12.5","25","50","100","6.25"])
    plt.title(f"Energy difference at {folder}K")
    plt.savefig(f"dE_{folder}.png")
    plt.close()

    fig = plt.figure(figsize=(20,10), dpi=400)
    fig.add_subplot(231)
    plt.hist(-np.gradient(data0),bins=b, alpha=a)
    plt.title("512 12.5")
    fig.add_subplot(232)
    plt.hist(-np.gradient(data1),bins=b, alpha=a)
    plt.title("1024 25")
    fig.add_subplot(233)
    plt.hist(-np.gradient(data2),bins=b, alpha=a)
    plt.title("2048 50")
    fig.add_subplot(234)
    plt.hist(-np.gradient(data3),bins=b, alpha=a)
    plt.title("4096 100")
    fig.add_subplot(235)
    plt.hist(-np.gradient(data4),bins=b, alpha=a)
    plt.title("256 6.25")
    plt.savefig(f"<H>dE_{folder}.png")
    plt.close()

    
