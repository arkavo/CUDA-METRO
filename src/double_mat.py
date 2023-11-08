import alt_mat_1_construct as cst
import montecarlo as mc
import numpy as np
import tqdm as tqdm

mc2 = cst.alt_Montecarlo(config1="../configs/test_config.json", config2="../configs/test_config.json")

mc2.mc_init(S1=2.00, S2=1.50)
print(mc2.T)
'''
for i in tqdm.tqdm(range(mc2.S_Wrap), desc="Stability Runs", unit="runs", colour="green"):
    mc2.generate_random_numbers(mc2.S_Wrap)
    np.save(f"{mc2.save_direcotry}/grid_{i:04d}", mc2.run_mc_3636(mc2.T[0]))
mc2.Analyze()
mc2.spin_view()
#mc2.quiver_view()
'''
mc2.generate_random_numbers(mc2.S_Wrap)
mc2.run_mc_tc_3636(mc2.T)
