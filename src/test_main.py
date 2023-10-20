import construct as cst
import montecarlo as mc

test_mc0 = cst.MonteCarlo(config="../configs/test_config.json")
test_mc0.generate_random_numbers(test_mc0.S_Wrap)
test_mc0.mc_init()
print(test_mc0.T[0])
test_mc0.run_mc_tc(test_mc0.T[0])