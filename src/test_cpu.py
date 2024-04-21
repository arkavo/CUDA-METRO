import sub_construct as sc
import cupy as cp

test_mc_cpu = sc.serial_MonteCarlo(config="../configs/tc_config.json")
test_mc_cpu.display_material()
test_mc_cpu.mc_init()
#test_mc_cpu.mc_run_6_6_6_12_dm1()
#test_mc_cpu.mc_tc_3_6_3_6_dm0()

