# This file is used to calculate the Curie temperature of a material using Monte Carlo simulation.
#--------------------------------------------

import construct as cst
import cudametro.montecarlo as mc
import numpy as np
import matplotlib.pyplot as plt

test_mc0 = cst.MonteCarlo(config="../../configs/tc_config.json")
test_mc0.mc_init()
test_mc0.display_material()
M, X = np.array([]), np.array([])
for t in test_mc0.T:
    test_mc0.grid_reset()
    test_mc0.generate_random_numbers(test_mc0.S_Wrap)
    m, x = test_mc0.run_mc_tc_3636(t)
    M = np.append(M, m)
    X = np.append(X, x)
np.save(f"{test_mc0.save_direcotry}/BQM_{test_mc0.size}_{test_mc0.Blocks}", M)
np.save(f"{test_mc0.save_direcotry}/BQX_{test_mc0.size}_{test_mc0.Blocks}", X)

X[0] = 0.0

plt.plot(test_mc0.T, M/test_mc0.spin, label="M")
plt.title(f"Magnetization vs Temperature of {test_mc0.MAT_NAME} at size {test_mc0.size}x{test_mc0.size}")
plt.savefig(f"{test_mc0.save_direcotry}/M_{test_mc0.size}.png")
plt.close()
plt.plot(test_mc0.T, X/X.max(), label="X")
plt.title(f"Susceptibility vs Temperature of {test_mc0.MAT_NAME} at size {test_mc0.size}x{test_mc0.size}")
plt.savefig(f"{test_mc0.save_direcotry}/X_{test_mc0.size}.png")
plt.close()
