import construct as cst
import montecarlo as mc
import numpy as np
import matplotlib.pyplot as plt


metric = 10
test_mc0 = cst.MonteCarlo(config="../configs/CRI3.json")
test_mc0.mc_init()
test_mc0.display_material()
Mf, Xf, Ef = np.array([]), np.array([]), np.array([])
M, X = np.array([]), np.array([])
steps = int(metric*(test_mc0.size**2/(test_mc0.Blocks*test_mc0.stability_runs)))
unit = int(steps/metric)
test_mc0.sampler()
for t in test_mc0.T:
    test_mc0.grid_reset()
    E = np.array([])
    for i in range(steps):
        for j in range(unit):
            test_mc0.generate_random_numbers(test_mc0.S_Wrap)
        m, x = test_mc0.run_mc_tc_3636(t)
        M = np.append(M, m)
        X = np.append(X, x)
        test_mc0.sampler()
        e = np.mean(test_mc0.en_3636(t))
        E = np.append(E, e)
    print(f"Energy/site avg: {np.mean(e):.3f} meV")
    plt.plot(E/test_mc0.size**2, label=f"{t:.2f} K")
    np.save(f"{test_mc0.save_direcotry}/Energy_{test_mc0.Blocks}_{t:.2f}", E)
    Mf = np.append(Mf, np.mean(M))
    Xf = np.append(Xf, np.mean(X))
    Ef = np.append(Ef, np.mean(E))
Xf[0] = 0.0
plt.savefig(f"{test_mc0.save_direcotry}/Energy_{test_mc0.size}.png")
plt.legend(str(test_mc0.T),shadow=True, fancybox=True)
plt.title("Energy vs Step evolution")
plt.close()
np.save(f"{test_mc0.save_direcotry}/M_{test_mc0.size}", Mf)
np.save(f"{test_mc0.save_direcotry}/X_{test_mc0.size}", Xf)
np.save(f"{test_mc0.save_direcotry}/E_{test_mc0.size}", Ef)

plt.plot(test_mc0.T, Mf/test_mc0.spin, label="M")
plt.plot(test_mc0.T, Xf/Xf.max(), label="X")
plt.plot(test_mc0.T, np.gradient(Ef), label="E")
plt.title(f"M/X/E vs T of {test_mc0.MAT_NAME} at size {test_mc0.size}x{test_mc0.size}")
plt.savefig(f"{test_mc0.save_direcotry}/M_{test_mc0.size}.png")
