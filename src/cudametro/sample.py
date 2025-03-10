# Main file to run the Monte Carlo simulation 
#--------------------------------------------

from cudametro import construct as cst
import numpy as np
import tqdm as tqdm

# Set your config file here
test_mc0 = cst.MonteCarlo(config="sample_config.json")

# Initialize the Monte Carlo simulation
test_mc0.mc_init()
test_mc0.display_material()
for i in tqdm.tqdm(range(test_mc0.S_Wrap), desc="Stability Runs", unit="runs", colour="green"):
    test_mc0.generate_random_numbers(test_mc0.stability_runs)
    np.save(f"{test_mc0.save_directory}/grid_{i:04d}", test_mc0.run_mc_dmi_66612(test_mc0.T[0]))

# Save the final state of the simulation

print("Simulation Completed")
print(f"Final state saved in {test_mc0.save_directory}")
# To visualize the final state, use the "visualize.py" script with appropriate directory
print("To visualize the final state, use the visualize.py script with appropriate directory")
print(f"Example: python visualize.py {test_mc0.save_directory}")
inp = input("Press Y/N to continue to analysis mode: ")
#--------------------------------------------

if inp == "Y" or "y" or "yes" or "Yes":
    viewer = cst.Analyze(test_mc0.save_directory, reverse=False)
    viewer.spin_view()

#--------------------------------------------

if inp == "N" or "n" or "no" or "No":
    print("Exiting")
    exit(0)