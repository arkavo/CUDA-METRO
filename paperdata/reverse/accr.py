import numpy as np
import matplotlib.pyplot as plt

folders = ["0.01","11.25","22.50","45.00"]

for folder in folders:
    ac0, co0 = np.load(f"{folder}/Acceptance_256_{folder}.npy"), 3.0*np.load(f"{folder}/Correlation_256_{folder}.npy")
    ac1, co1 = np.load(f"{folder}/Acceptance_512_{folder}.npy"), 3.0*np.load(f"{folder}/Correlation_512_{folder}.npy")
    ac2, co2 = np.load(f"{folder}/Acceptance_1024_{folder}.npy"), 3.0*np.load(f"{folder}/Correlation_1024_{folder}.npy")
    ac3, co3 = np.load(f"{folder}/Acceptance_2048_{folder}.npy"), 3.0*np.load(f"{folder}/Correlation_2048_{folder}.npy")
    ac4, co4 = np.load(f"{folder}/Acceptance_4096_{folder}.npy"), 3.0*np.load(f"{folder}/Correlation_4096_{folder}.npy")

    fig = plt.figure(figsize=(20,10), dpi=400)
    fig.add_subplot(231)
    plt.plot(ac0)
    
    plt.title("256 6.25")
    fig.add_subplot(232)
    plt.plot(ac1)
    
    plt.title("512 12.5")
    fig.add_subplot(233)
    plt.plot(ac2)
    
    plt.title("1024 25")
    fig.add_subplot(234)
    plt.plot(ac3)
    
    plt.title("2048 50")
    fig.add_subplot(235)
    plt.plot(ac4)
    
    plt.title("4096 100")
    fig.add_subplot(236)
    plt.plot(ac0)
    plt.plot(ac1)
    plt.plot(ac2)
    plt.plot(ac3)
    plt.plot(ac4)
    
    plt.title(f"Acceptance evolution at {folder}K")
    plt.legend(["6.25","12.5","25","50","100"])
    plt.savefig(f"Ac_{folder}.png")
    plt.close()

    fig = plt.figure(figsize=(20,10), dpi=400)
    fig.add_subplot(231)
    plt.plot(co0)
    
    plt.title("256 6.25")
    fig.add_subplot(232)
    plt.plot(co1)
    
    plt.title("512 12.5")
    fig.add_subplot(233)
    plt.plot(co2)
    
    plt.title("1024 25")
    fig.add_subplot(234)
    plt.plot(co3)
    
    plt.title("2048 50")
    fig.add_subplot(235)
    plt.plot(co4)
    
    plt.title("4096 100")
    fig.add_subplot(236)
    plt.plot(co0)
    plt.plot(co1)
    plt.plot(co2)
    plt.plot(co3)
    plt.plot(co4)
    
    plt.title(f"Correlation evolution at {folder}K")
    plt.legend(["6.25","12.5","25","50","100"])
    plt.savefig(f"Co_{folder}.png")
