from montecarlo import construct as cst
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm

# Use fig2a for MnBr2 and fig2b for CrInSe3
config_name = "fig2a.json"
config_path = "../../configs/fig2_configs/"
# Run once for each configuration

# Set your config file here
test_mc0 = cst.MonteCarlo(config=config_path+config_name)

if config_name == "fig2a.json":

    # Initialize the Monte Carlo simulation
    test_mc0.mc_init()
    test_mc0.display_material()
    for i in tqdm.tqdm(range(test_mc0.S_Wrap), desc="Stability Runs", unit="runs", colour="green"):
        test_mc0.generate_random_numbers(test_mc0.stability_runs)
        np.save(f"{test_mc0.save_directory}/grid_{i:04d}", test_mc0.run_mc_dmi_4448(test_mc0.T[0]))
else:
    # Initialize the Monte Carlo simulation
    # NOTE: PLEASE EDIT THE MODE OF EXECUTION WITH THE CORRECT CRYSTAL CLASS SIMULATOR
    test_mc0.mc_init()
    test_mc0.display_material()
    for i in tqdm.tqdm(range(test_mc0.S_Wrap), desc="Stability Runs", unit="runs", colour="green"):
        test_mc0.generate_random_numbers(test_mc0.stability_runs)
        np.save(f"{test_mc0.save_directory}/grid_{i:04d}", test_mc0.run_mc_dmi_66612(test_mc0.T[0]))
# Save the final state of the simulation 
# To visualize the final state, use the "visualize.py" script with appropriate directory

output_folder_name = test_mc0.save_directory

folder = output_folder_name

viewer = cst.Analyze(folder, reverse=False)

# Visualize the results
# Spin configuration at the end of the simulation
viewer.spin_view()
# Quiver plot of the spins
viewer.quiver_view()

# Your Diagrams are now saved in the output folder. Use the 'reverse' flag if you want to quickly visualize the end state.
# You can even delete data points should you wish to speed up your visualizer.