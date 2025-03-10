import construct as cst
import cudametro.montecarlo as mc
import numpy as np
import matplotlib.pyplot as plt



def corr(g1,g2):
    ac = 0
    corr = 0.0
    for i in range(len(g1)):
        for j in range(3):
            cur = g1[i][j]*g2[i][j]
            if cur < 1.0:
                ac += 1
            corr += cur
    
    return corr/(len(g1)*3), ac/(len(g1)*3)



metric = 2000
test_mc0 = cst.MonteCarlo(config="../configs/CRI3.json")
test_mc0.mc_init()
test_mc0.display_material()
gprev = np.reshape(test_mc0.grid, (test_mc0.size**2, 3))/test_mc0.spin
Mf, Xf, Ef = np.array([]), np.array([]), np.array([])
M, X = np.array([]), np.array([])
steps = int(metric*(test_mc0.size**2/(test_mc0.Blocks*test_mc0.stability_runs)))
unit = int(steps/metric)
test_mc0.sampler()
for t in test_mc0.T:
    test_mc0.grid_reset()
    E, C, Ac = np.array([]), np.array([]), np.array([])
    for i in range(steps):
        for j in range(unit):
            test_mc0.generate_random_numbers(test_mc0.S_Wrap)
            m, x = test_mc0.run_mc_tc_3636(t)
        gnext = np.reshape(test_mc0.grid, (test_mc0.size**2, 3))/test_mc0.spin
        c, ac = corr(gprev, gnext)
        gprev = gnext
        M = np.append(M, m)
        X = np.append(X, x)
        test_mc0.sampler()
        e = np.mean(test_mc0.en_3636(t))
        E = np.append(E, e)
        C = np.append(C, c)
        Ac = np.append(Ac, ac)
    print(f"Correlation avg: {np.mean(C):.3f}")
    print(f"Acceptance avg: {np.mean(Ac):.3f}")
    print(f"Energy/site avg: {np.mean(e):.3f} meV")
    plt.plot(E/test_mc0.size**2, label=f"{t:.2f} K")
    np.save(f"{test_mc0.save_direcotry}/Energy_{test_mc0.Blocks}_{t:.2f}", E)
    plt.close()
    plt.plot(C, label=f"{t:.2f} K")
    plt.title(f"Correlation vs Step evolution at {t:.2f} K")
    plt.savefig(f"{test_mc0.save_direcotry}/Correlation_{test_mc0.Blocks}_{t:.2f}.png")
    np.save(f"{test_mc0.save_direcotry}/Correlation_{test_mc0.Blocks}_{t:.2f}", C)
    plt.close()
    plt.plot(Ac, label=f"{t:.2f} K")
    plt.title(f"Acceptance vs Step evolution at {t:.2f} K")
    plt.savefig(f"{test_mc0.save_direcotry}/Acceptance_{test_mc0.Blocks}_{t:.2f}.png")
    np.save(f"{test_mc0.save_direcotry}/Acceptance_{test_mc0.Blocks}_{t:.2f}", Ac)
    plt.close()
    np.save(f"../DiagramData/Fig1/BL_Correlation_{test_mc0.Blocks}_{t:.2f}", C)
    np.save(f"../DiagramData/Fig1/BL_Acceptance_{test_mc0.Blocks}_{t:.2f}", Ac)
    np.save(f"../DiagramData/Fig1/BL_Energy_{test_mc0.Blocks}_{t:.2f}", E)
    Mf = np.append(Mf, np.mean(M))
    Xf = np.append(Xf, np.mean(X))
    Ef = np.append(Ef, np.mean(E))
    np.save(f"{test_mc0.save_direcotry}/grid_{t}", test_mc0.grid.reshape((3*test_mc0.size**2)))
Xf[0] = 0.0
plt.savefig(f"{test_mc0.save_direcotry}/Energy_{test_mc0.size}.png")
plt.legend(str(test_mc0.T),shadow=True, fancybox=True)
plt.title("Energy vs Step evolution")
plt.close()
np.save(f"{test_mc0.save_direcotry}/M_{test_mc0.size}", Mf)
np.save(f"{test_mc0.save_direcotry}/X_{test_mc0.size}", Xf)

plt.plot(test_mc0.T, Mf/test_mc0.spin, label="M")
plt.plot(test_mc0.T, Xf/Xf.max(), label="X")
plt.plot(test_mc0.T, np.gradient(Ef), label="E")
plt.title(f"M/X/E vs T of {test_mc0.MAT_NAME} at size {test_mc0.size}x{test_mc0.size}")
plt.savefig(f"{test_mc0.save_direcotry}/M_{test_mc0.size}.png")
