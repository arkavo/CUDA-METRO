import construct as cst
import montecarlo as mc
import numpy as np
import tqdm as tqdm
import datetime as dt
import pycuda.autoinit
import pycuda.driver as cuda
import sys

tol = 1e-2

mc_gb = cst.MonteCarlo(config="../configs/grid_bench.json")
grids = [256]

for g in grids:
    mc_gb.mc_init()
    mc_gb.size = g
    mc_gb.grid_reset()
    for t in mc_gb.T:
        #
        t_curr = dt.datetime.now()
        ct = 0
        R_FLAG = False
        while not R_FLAG:
            M = np.array([])
            
            for i in range(10):
                mc_gb.generate_random_numbers(mc_gb.S_Wrap)
                m,x = mc_gb.run_mc_tc_3636(t)
                M = np.append(M, m)
            std_M = np.std(M)
            ct += 1
            if std_M < tol:
                R_FLAG = True
        t_end = dt.datetime.now()
        print(f"Time taken for {g} grid at {t} K: {t_end-t_curr}")
        print(f"Number of runs: {ct}")

        