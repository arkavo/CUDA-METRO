import numpy as np
import pandas as pd


T = np.linspace(0.01, 25.5, 41)

BlueL = np.loadtxt("BLUE.txt", delimiter=",")
blm_512 = np.load("BLM_64_512.npy")
blm_1024 = np.load("BLM_64_1024.npy")
blm_2048 = np.load("BLM_64_2048.npy")
blm_4096 = np.load("BLM_64_4096.npy")

bl = pd.DataFrame(np.transpose(np.array([blm_512,blm_1024,blm_2048,blm_4096])), columns=["512","1024","2048","4096"])
bl.to_csv(f"csvs/BLM_CrI3.csv", index=False)

BlueX = np.loadtxt("BLUEX.txt", delimiter=",")
blx_512 = np.load("BLX_64_512.npy")
blx_1024 = np.load("BLX_64_1024.npy")
blx_2048 = np.load("BLX_64_2048.npy")
blx_4096 = np.load("BLX_64_4096.npy")

bl = pd.DataFrame(np.transpose(np.array([blx_512,blx_1024,blx_2048,blx_4096])), columns=["512","1024","2048","4096"])
bl.to_csv(f"csvs/BLX_CrI3.csv", index=False)

GreenL = np.loadtxt("GREEN.txt", delimiter=",")
grm_512 = np.load("BQM_64_512.npy")
grm_1024 = np.load("BQM_64_1024.npy")
grm_2048 = np.load("BQM_64_2048.npy")
grm_4096 = np.load("BQM_64_4096.npy")

gr = pd.DataFrame(np.transpose(np.array([grm_512,grm_1024,grm_2048,grm_4096])), columns=["512","1024","2048","4096"])
gr.to_csv(f"csvs/GRM_CrI3.csv", index=False)

GreenX = np.loadtxt("GREENX.txt", delimiter=",")
grx_512 = np.load("BQX_64_512.npy")
grx_1024 = np.load("BQX_64_1024.npy")
grx_2048 = np.load("BQX_64_2048.npy")
grx_4096 = np.load("BQX_64_4096.npy")

gr = pd.DataFrame(np.transpose(np.array([grx_512,grx_1024,grx_2048,grx_4096])), columns=["512","1024","2048","4096"])
gr.to_csv(f"csvs/GRX_CrI3.csv", index=False)
