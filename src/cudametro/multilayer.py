import construct as cst
import numpy as np
import tqdm as tqdm
import pycuda.driver as drv

layers = 8

test_ml = cst.MonteCarlo(config="multilayer_config.json")
test_ml.mc_init_multilayer(layers=layers)
test_ml.display_material()

def vram_used_mb():
    free, total = drv.mem_get_info()
    return (total - free) / 1024**2

# ── Field zone configuration ──────────────────────────────────────────────────
# Divide the lattice columns into N zones by providing N-1 boundary indices.
# Zones 0 and N-1 act as buffer regions (same field); middle zones are the
# experiment region. Adjust boundaries and fields as needed.
#
# Example: 5 zones across SIZE=300 columns
#   columns:  [  0, 60)  [60,120)  [120,180)  [180,240)  [240,300)
#   role:       buffer    zone 2     zone 3     zone 4     buffer
#
SIZE = test_ml.size
n_zones    = 5
boundaries = [10, 145, 155, 290]   # 4 boundaries for 5 zones; adjust as needed
fields     = [0.0, 0.3, 0.1, 0.4, 0.0]  # one value per zone; first/last are buffers
# For a uniform field instead, comment the above and use:
#   test_ml.set_field(0.0)
test_ml.set_field_zones(boundaries, fields)
print(f"Field zones set: boundaries={boundaries}, fields={fields}")
# ─────────────────────────────────────────────────────────────────────────────

free, total = drv.mem_get_info()
print(f"\n  ── GPU memory ──────────────────────────────")
print(f"     Total VRAM : {total/1024**2:,.0f} MB")
print(f"     Used       : {(total-free)/1024**2:,.0f} MB  (after full pre-allocation)")
print(f"     Free       : {free/1024**2:,.0f} MB")
print(f"  ────────────────────────────────────────────\n")

for i in tqdm.tqdm(range(test_ml.S_Wrap), desc="Stability Runs", unit="runs", colour="blue"):
    test_ml.generate_random_numbers_multilayer(test_ml.stability_runs)
    np.save(f"{test_ml.save_directory}/grid_{i:04d}", test_ml.run_mc_3636_multilayer(test_ml.T[0], layers=layers))

print("Multilayer Simulation Completed")
print(f"Final state saved in {test_ml.save_directory}")
print("To visualize the final state, use the visualize.py script with appropriate directory")
print(f"Example: python visualize.py {test_ml.save_directory}")
inp = input("Press S/Q/SQ/N to continue to analysis mode: ")
if inp == "S" or inp == "s":
    viewer = cst.Analyze(test_ml.save_directory, reverse=True)
    viewer.spin_view_multilayer()
elif inp == "Q" or inp == "q":
    viewer = cst.Analyze(test_ml.save_directory, reverse=False)
    viewer.quiver_view_multilayer()
elif inp == "SQ" or inp == "sq" or inp == "Sq" or inp == "sQ":
    viewer = cst.Analyze(test_ml.save_directory, reverse=False)
    viewer.spin_view_multilayer()
    viewer.quiver_view_multilayer()
elif inp == "N" or inp == "n" or inp == "no" or inp == "No":
    print("Exiting")
    exit(0)
