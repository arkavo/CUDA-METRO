import numpy as np

T = np.linspace(0.01, 60.00, 41)
np.savetxt("csvs/T.txt", np.transpose(np.array([T])), delimiter=",")
for i in range(4):
    blue_m_data = np.load(f"BLM_64_{512*2**i}.npy")
    green_m_data = np.load(f"BQM_64_{512*2**i}.npy")
    blue_m_data = blue_m_data/1.5
    green_m_data = green_m_data/1.5
    blue_x_data = np.load(f"BLX_64_{512*2**i}.npy")
    green_x_data = np.load(f"BQX_64_{512*2**i}.npy")
    blue_x_data = blue_x_data/blue_x_data.max()
    green_x_data = green_x_data/green_x_data.max()

    np.savetxt(f"csvs/BLM_{512*2**i}.txt", np.transpose(np.array([blue_m_data])), delimiter=",")
    np.savetxt(f"csvs/BQM_{512*2**i}.txt", np.transpose(np.array([green_m_data])), delimiter=",")
    np.savetxt(f"csvs/BLX_{512*2**i}.txt", np.transpose(np.array([blue_x_data])), delimiter=",")
    np.savetxt(f"csvs/BQX_{512*2**i}.txt", np.transpose(np.array([green_x_data])), delimiter=",")

