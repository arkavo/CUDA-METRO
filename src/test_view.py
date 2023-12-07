import construct as cst
import sys
import os

args = sys.argv
if len(args) != 2:
    print("Usage: python3 test_main.py <config_file>")
    exit(1)
folder = args[1]

viewer = cst.Analyze(folder, reverse=False)
#viewer.spin_view()
#viewer.quiver_view()
viewer.en_66612()
