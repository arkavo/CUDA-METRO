from cudametro import construct as cst
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm


# =============================================================================
# Edit the following lines to change the parameters of the simulation
# =============================================================================
fname = "fig3_configs/fig3.json"          # Configuration file [options fig2a.json, fig2b.json]
in_dir = "input_parameters/"               # Input directory
# =============================================================================

# Set your config file here
test_mc0 = cst.MonteCarlo(config=fname, input_folder=in_dir)

# Initialize the Monte Carlo simulation
test_mc0.mc_init()
test_mc0.display_material()
for i in tqdm.tqdm(range(test_mc0.S_Wrap), desc="Stability Runs", unit="runs", colour="green"):
    test_mc0.generate_random_numbers(test_mc0.stability_runs)
    np.save(f"{test_mc0.save_directory}/grid_{i:04d}", test_mc0.run_mc_dmi_66612(test_mc0.T[0]))

# Save the final state of the simulation 
# To visualize the final state, use the "visualize.py" script with appropriate directory

output_folder_name = test_mc0.save_directory

folder = output_folder_name

viewer = cst.Analyze(folder, reverse=False, input_folder=in_dir)

# Visualize the results
# Spin configuration at the end of the simulation
viewer.spin_view()
# Quiver plot of the spins
viewer.quiver_view()
# The energy of the system
viewer.en_66612()

# Your Diagrams are now saved in the output folder. Use the 'reverse' flag if you want to quickly visualize the end state. 
# You can even delete data points should you wish to speed up your visualizer.
# Note that your energy will also be counted in reverse if you used the reverse flag.