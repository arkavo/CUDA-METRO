# Main file to run the Monte Carlo simulation 
#--------------------------------------------

import construct as cst
import montecarlo as mc
import numpy as np
import tqdm as tqdm

# Set your config file here
test_mc0 = cst.MonteCarlo(config="../configs/test_config.json")

# Initialize the Monte Carlo simulation
test_mc0.mc_init()
test_mc0.display_material()
for i in tqdm.tqdm(range(test_mc0.S_Wrap), desc="Stability Runs", unit="runs", colour="green"):
    test_mc0.generate_random_numbers(test_mc0.stability_runs)
    np.save(f"{test_mc0.save_direcotry}/grid_{i:04d}", test_mc0.run_mc_dmi_66612(test_mc0.T[0]))

# Save the final state of the simulation 
# To visualize the final state, use the "visualize.py" script with appropriate directory