# Benchmarking script for the Monte Carlo simulation

#--------------------------------------------
import construct as cst
import cudametro.montecarlo as mc
import numpy as np
import tqdm as tqdm
import datetime as dt
import pycuda.autoinit
import pycuda.driver as cuda
import sys


tol = 1e-3

(free,total)=cuda.mem_get_info()
print("Global memory occupancy:%f%% free"%(free*100/total))

device=cuda.Device(0)
print(f"Name: {device.name()}")
print(f"Compute Capability: {device.compute_capability()}")
print(f"Total Memory: {device.total_memory()/1024**3:.1f} GB")
print("=====================================")

sizes = [64,128,256,512]
blocks = [64,128,256,512,1024,2048,4096,8192,16384]

print("Benching with approx 90.0% memory usage")

bench_MC = cst.MonteCarlo(config="../configs/bench.json")
base_stability_runs = int(device.total_memory()*0.90/(10000*sys.getsizeof(np.float32())))
print(f"clean base stability runs on single spin update: {base_stability_runs}")


bench_MC.size = 256
bench_MC.mc_init()
Time = np.array([])
print("Begin Bench")
for i in range(len(blocks)):
    if blocks[i] > base_stability_runs:
        break
    bench_MC.Blocks = blocks[i]
    bench_MC.stability_runs = 10000*base_stability_runs//bench_MC.Blocks
    print(f"Blocks: {bench_MC.Blocks}, Stability Runs: {bench_MC.stability_runs}")
    bench_MC.C1 = bench_MC.Blocks*bench_MC.stability_runs
    t_curr = dt.datetime.now()
    for j in range(bench_MC.S_Wrap):
        bench_MC.generate_random_numbers(1)
        bench_MC.run_mc_tc_3636(bench_MC.T[0])
    t_end = dt.datetime.now()
    Time = np.append(Time, (t_end-t_curr).total_seconds())
    print(f"Time taken for {blocks[i]} blocks: {t_end-t_curr}")

np.save(f"{bench_MC.save_direcotry}/Timings", Time)
print("=====================================")
