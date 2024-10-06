# Description: This script is used to visualize the results of the simulation. It takes the folder containing the results as input and generates the following plots:
# 1. Spin configuration at the end of the simulation
# 2. Quiver plot of the spins
#--------------------------------------------

import construct as cst
import sys
import os

args = sys.argv
if len(args) != 2:
    print("Usage: python3 test_main.py <config_file>")
    exit(1)
folder = args[1]

viewer = cst.Analyze(folder, reverse=False)

# Visualize the results
# Spin configuration at the end of the simulation
viewer.spin_view()
# Quiver plot of the spins
viewer.quiver_view()
