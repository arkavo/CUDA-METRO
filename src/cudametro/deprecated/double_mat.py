import alt_mat_1_construct as cst
import cudametro.montecarlo as mc
import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt

mc2 = cst.alt_Montecarlo(config1="../configs/test_config.json", config2="../configs/test_config.json")
mc2.MAT_NAME = "MnCrI6"
mc2.mc_init(S1=1.50, S2=2.00)
print(mc2.T)
'''
for i in tqdm.tqdm(range(mc2.S_Wrap), desc="Stability Runs", unit="runs", colour="green"):
    mc2.generate_random_numbers(mc2.S_Wrap)
    np.save(f"{mc2.save_direcotry}/grid_{i:04d}", mc2.run_mc_3636(mc2.T[0]))
mc2.Analyze()
mc2.spin_view()
#mc2.quiver_view()
'''

M, X = np.array([]), np.array([])
for t in mc2.T:
    #mc2.grid_reset()
    mc2.generate_random_numbers(mc2.S_Wrap)
    m, x = mc2.run_mc_tc_3636(t)
    M = np.append(M, m)
    X = np.append(X, x)

np.save(f"{mc2.save_direcotry}/M_{mc2.size}", M)
np.save(f"{mc2.save_direcotry}/X_{mc2.size}", X)

plt.plot(mc2.T, M/mc2.spin, label="M")
plt.plot(mc2.T, X/X.max(), label="X")
plt.title(f"Magnetization vs Temperature of {mc2.MAT_NAME} at size {mc2.size}x{mc2.size}")
plt.savefig(f"{mc2.save_direcotry}/M_{mc2.size}.png")