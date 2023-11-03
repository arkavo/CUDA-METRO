import construct as cst
import montecarlo as mc

test_mc0 = cst.MonteCarlo(config="../configs/tc_config.json")
test_mc0.generate_random_numbers(test_mc0.S_Wrap)
test_mc0.mc_init()
test_mc0.run_mc_tc_3636(test_mc0.T)