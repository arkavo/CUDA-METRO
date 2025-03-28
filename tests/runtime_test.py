from cudametro import construct as cst
import numpy as np
import tqdm as tqdm

fname = "runtime_test_config.json"
in_dir = ""
# =============================================================================

print("=== Testing the CUDAMETRO package ===")

test_mc0 = None

try:
    test_mc0 = cst.MonteCarlo(config=fname, input_folder=in_dir)
except Exception as e:
    print(f"Error initializing MonteCarlo: {e}")
# Initialize the Monte Carlo simulation

try:
    test_mc0.mc_init()
    test_mc0.display_material()
except Exception as e:
    print(f"Error initializing Monte Carlo simulation: {e}")

try:
    for i in tqdm.tqdm(range(test_mc0.S_Wrap), desc="Stability Runs", unit="runs", colour="green"):
        test_mc0.generate_random_numbers(test_mc0.stability_runs)
        np.save(f"{test_mc0.save_directory}/grid_{i:04d}", test_mc0.run_mc_dmi_66612(test_mc0.T[0]))
except Exception as e:
    print(f"Error during Monte Carlo simulation: {e}")
# Save the final state of the simulation
# To visualize the final state, use the "visualize.py" script with appropriate directory
output_folder_name = test_mc0.save_directory
folder = output_folder_name
viewer = None
try:
    viewer = cst.Analyze(folder, reverse=False, input_folder=in_dir)
except Exception as e:
    print(f"Error initializing Analyze: {e}")
# Visualize the results
try:
    # Spin configuration at the end of the simulation
    viewer.spin_view()
    # Quiver plot of the spins
    viewer.quiver_view()
except Exception as e:
    print(f"Error during visualization: {e}")

print("=== CUDAMETRO package test completed ===")
# =============================================================================
# Your Diagrams are now saved in the output folder. Use the 'reverse' flag if you want to quickly visualize the end state.
# You can even delete data points should you wish to speed up your visualizer.
# Note that your energy will also be counted in reverse if you used the reverse flag.
# =============================================================================
# End of script
# =============================================================================
# This script is designed to test the runtime of the CUDAMETRO package.
# It initializes the Monte Carlo simulation, runs it, and visualizes the results.
# Make sure to adjust the configuration file and input directory as needed.
# =============================================================================