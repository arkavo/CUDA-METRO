from cudametro import construct as cst
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm

# =============================================================================
# Edit the following lines to change the parameters of the simulation
# =============================================================================
fname = "fig1_configs"          # Configuration file [options p1.json, p2.json, p3.json, p4.json]
in_dir = "input_parameters/"                             # Input directory
# =============================================================================


# Universal parameters
save_dir_flag = False
T_arr = None
spin = 1.0
mat = None
size = 32

for i in range(4):
    fs_name = f"fig1_configs/p{i+1}.json"
    print(f"Running {fs_name}")
    test_mc0 = cst.MonteCarlo(config=fs_name, input_folder=in_dir)
    test_mc0.mc_init()
    test_mc0.display_material()
    M, X = np.array([]), np.array([])
    for t in test_mc0.T:
        test_mc0.grid_reset()
        test_mc0.generate_random_numbers(test_mc0.S_Wrap)
        m, x = test_mc0.run_mc_tc_3636(t)
        M = np.append(M, m)
        X = np.append(X, x)
    if not save_dir_flag:
        unified_save_dir = test_mc0.save_directory
        save_dir_flag = True
        save_dir = test_mc0.save_directory
        T_arr = test_mc0.T
        spin = test_mc0.spin
        mat = test_mc0.MAT_NAME
        size = test_mc0.size
    np.save(f"{unified_save_dir}/BQM_{test_mc0.size}_{test_mc0.Blocks}", M)
    np.save(f"{unified_save_dir}/BQX_{test_mc0.size}_{test_mc0.Blocks}", X)


    X[0] = 0.0

    plt.plot(T_arr, M/test_mc0.spin, label="M")
    plt.title(f"Magnetization vs Temperature of {mat} at size {test_mc0.size}x{test_mc0.size}")
    plt.savefig(f"{unified_save_dir}/M_{test_mc0.size}.png")
    plt.close()
    plt.plot(T_arr, X/X.max(), label="X")
    plt.title(f"Susceptibility vs Temperature of {mat} at size {test_mc0.size}x{test_mc0.size}")
    plt.savefig(f"{unified_save_dir}/X_{test_mc0.size}.png")
    plt.close()
    test_mc0 = None

M_1, X_1 = np.load(f"{unified_save_dir}/BQM_32_128.npy"), np.load(f"{unified_save_dir}/BQX_32_128.npy")
M_2, X_2 = np.load(f"{unified_save_dir}/BQM_32_256.npy"), np.load(f"{unified_save_dir}/BQX_32_256.npy")
M_3, X_3 = np.load(f"{unified_save_dir}/BQM_32_512.npy"), np.load(f"{unified_save_dir}/BQX_32_512.npy")
M_4, X_4 = np.load(f"{unified_save_dir}/BQM_32_1024.npy"), np.load(f"{unified_save_dir}/BQX_32_1024.npy")

Figure = plt.figure()
plt.plot(T_arr, M_1/spin, label="M_12.5")
plt.plot(T_arr, M_2/spin, label="M_25")
plt.plot(T_arr, M_3/spin, label="M_50")
plt.plot(T_arr, M_4/spin, label="M_100")
plt.title(f"Magnetization vs Temperature of {mat} at size {size}x{size}")
plt.legend()
plt.savefig(f"{unified_save_dir}/M_{size}_all.png")
plt.close()

Figure = plt.figure()
plt.plot(T_arr[1:], X_1[1:]/X_1.max(), label="X_12.5")
plt.plot(T_arr[1:], X_2[1:]/X_2.max(), label="X_25")
plt.plot(T_arr[1:], X_3[1:]/X_3.max(), label="X_50")
plt.plot(T_arr[1:], X_4[1:]/X_4.max(), label="X_100")
plt.title(f"Susceptibility vs Temperature of {mat} at size {size}x{size}")
plt.legend()
plt.savefig(f"{unified_save_dir}/X_{size}_all.png")
plt.close()