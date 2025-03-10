import construct as cst
import cudametro.montecarlo as mc
import numpy as np
import tqdm as tqdm
import sys
import os

test_mc0 = cst.MonteCarlo(config="../../configs/test_config.json")
test_mc0.mc_init()

args = sys.argv
if len(args) != 2:
    print("Usage: python3 test_main.py <config_file>")
    exit(1)
folder = args[1]

viewer = cst.Analyze(folder, reverse=True)
viewer.spin_view()
viewer.quiver_view()