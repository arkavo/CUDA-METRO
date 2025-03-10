import construct as cst
import cudametro.montecarlo as mc
import numpy as np
import tqdm as tqdm
import datetime as dt
import pycuda.autoinit
import pycuda.driver as cuda
import sys

tol = 1e-2


timings, runs = np.array([]), np.array([])
for i in range(6):
    mc_gb = cst.MonteCarlo(config=f"../../configs/grid_benchs/grid_bench_{i}.json")
    mc_gb.mc_init()
    mc_gb.display_material()
    mc_gb.grid_reset()
    print(mc_gb.size, mc_gb.T)
    print(f"Begin Bench at {mc_gb.size} grid")
    t_curr = dt.datetime.now()
    ct = 0
    R_FLAG = False
    while not R_FLAG:
        mc_gb.generate_random_numbers(mc_gb.S_Wrap)
        m,x = mc_gb.run_mc_tc_3636(mc_gb.T[0])
        ct += 1
        if m < tol:
            R_FLAG = True
            pass
    t_end = dt.datetime.now()
    print(f"Time taken for {mc_gb.size} grid {t_end-t_curr}")
    print(f"Number of runs: {ct}")
    timings = np.append(timings, (t_end-t_curr).total_seconds())
    runs = np.append(runs, ct)

np.save(f"Timings", timings)
np.save(f"Runs", runs)

        