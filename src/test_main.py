import construct as cst
import montecarlo as mc
import numpy as np
import tqdm as tqdm

test_mc0 = cst.MonteCarlo(config="../configs/test_config.json")

test_mc0.mc_init()
for i in tqdm.tqdm(range(test_mc0.S_Wrap), desc="Stability Runs", unit="runs", colour="green"):
    test_mc0.generate_random_numbers(test_mc0.stability_runs)
    np.save(f"{test_mc0.save_direcotry}/grid_{i:04d}", test_mc0.run_mc_tc_3636(test_mc0.T[0]))