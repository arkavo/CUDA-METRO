import cudametro.montecarlo as montecarlo
from cudametro import construct as cst
import numpy as np
import matplotlib.pyplot as plt

fpath = "../configs/fig1_configs/"
fname = "p1.json"


test_mc0 = cst.MonteCarlo(config=fname)
test_mc0.mc_init()
test_mc0.display_material()
M, X = np.array([]), np.array([])
for t in test_mc0.T:
    test_mc0.grid_reset()
    test_mc0.generate_random_numbers(test_mc0.S_Wrap)
    m, x = test_mc0.run_mc_tc_3636(t)
    M = np.append(M, m)
    X = np.append(X, x)
np.save(f"{test_mc0.save_directory}/BQM_{test_mc0.size}_{test_mc0.Blocks}", M)
np.save(f"{test_mc0.save_directory}/BQX_{test_mc0.size}_{test_mc0.Blocks}", X)

X[0] = 0.0

plt.plot(test_mc0.T, M/test_mc0.spin, label="M")
plt.title(f"Magnetization vs Temperature of {test_mc0.MAT_NAME} at size {test_mc0.size}x{test_mc0.size}")
plt.savefig(f"{test_mc0.save_directory}/M_{test_mc0.size}.png")
plt.close()
plt.plot(test_mc0.T, X/X.max(), label="X")
plt.title(f"Susceptibility vs Temperature of {test_mc0.MAT_NAME} at size {test_mc0.size}x{test_mc0.size}")
plt.savefig(f"{test_mc0.save_directory}/X_{test_mc0.size}.png")
plt.close()
M_1, X_1 = M, X

M_1, X_1 = np.load("BQM_64_512.npy"), np.load("BQX_64_512.npy")
M_2, X_2 = np.load("BQM_64_1024.npy"), np.load("BQX_64_1024.npy")
M_3, X_3 = np.load("BQM_64_2048.npy"), np.load("BQX_64_2048.npy")
M_4, X_4 = np.load("BQM_64_4096.npy"), np.load("BQX_64_4096.npy")

Figure = plt.figure()
plt.plot(test_mc0.T, M_1/test_mc0.spin, label="M_12.5")
plt.plot(test_mc0.T, M_2/test_mc0.spin, label="M_25")
plt.plot(test_mc0.T, M_3/test_mc0.spin, label="M_50")
plt.plot(test_mc0.T, M_4/test_mc0.spin, label="M_100")
plt.title(f"Magnetization vs Temperature of {test_mc0.MAT_NAME} at size {test_mc0.size}x{test_mc0.size}")
plt.legend()
plt.savefig(f"{test_mc0.save_directory}/M_{test_mc0.size}_all.png")
plt.close()

Figure = plt.figure()
plt.plot(test_mc0.T, X_1/X_1.max(), label="X_12.5")
plt.plot(test_mc0.T, X_2/X_2.max(), label="X_25")
plt.plot(test_mc0.T, X_3/X_3.max(), label="X_50")
plt.plot(test_mc0.T, X_4/X_4.max(), label="X_100")
plt.title(f"Susceptibility vs Temperature of {test_mc0.MAT_NAME} at size {test_mc0.size}x{test_mc0.size}")
plt.legend()
plt.savefig(f"{test_mc0.save_directory}/X_{test_mc0.size}_all.png")
plt.close()