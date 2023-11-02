import alt_mat_1_construct as cst
import montecarlo as mc
import numpy as np

mc2 = cst.alt_Montecarlo(config1="../configs/test_config.json", config2="../configs/test_config.json")
mc2.generate_random_numbers(mc2.S_Wrap)
mc2.mc_init()
mc2.S1 = np.array([1.5], dtype=np.float32)
mc2.S2 = np.array([1.7], dtype=np.float32)
print(mc2.T[0])