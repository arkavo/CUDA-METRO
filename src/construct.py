import pycuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.curandom import rand as curand 

rg = pycuda.curandom.XORWOWRandomNumberGenerator()

import numpy as np 
from numpy import random as rd
import seaborn as sns 
import matplotlib.pyplot as plt 

import os 
import sys 
sys.path.append("../utilities/")
import Material_Reader as rm 
import montecarlo as mc
from tqdm import tqdm 

import csv
import json
import datetime

class MonteCarlo:
    def __init__(self, config):
        self.config_read(config)
        
    def config_read(config):
        with open(config, 'r') as f:
            CONFIG = json.load(f)
        # FLAGS
        self.Single_MAT_Flag = CONFIG["Single_Mat_Flag"]
        self.Animation_Flags = CONFIG["Animation_Flags"]
        self.Static_T_Flag = CONFIG["Static_T_Flag"]
        self.Temps = np.float32(CONFIG["Temps"])
        self.FM_Flag = CONFIG["FM_Flag"]
        self.DMI_Flag = CONFIG["DMI_Flag"]
        self.TC_Flag = CONFIG["TC_Flag"]
        # CONSTANTS AND PATHS
        self.Material = CONFIG["Material"]
        self.Multiple_Materials = CONFIG["Multiple_Materials"]
        self.Input_Folder = "Inputs/"
        self.Output_Folder = "Outputs/"
        self.B_C = np.float32(CONFIG["B"])
        self.__SIZE = CONFIG["SIZE"]
        self.Box = CONFIG["Box"]
        self.Blocks = CONFIG["Blocks"]
        self.Threads = CONFIG["Threads"]
        self.stability_runs = CONFIG["stability_runs"]
        self.calculation_runs = CONFIG["calculation_runs"]
        self.Cmpl = CONFIG["Cmpl_Flag"]
        self.S_Wrap = CONFIG["stability_wrap"]
        self.C_Wrap = CONFIG["calculation_wrap"]
        self.Prefix = CONFIG["Prefix"]
        self.dump_location = "dumps/"
        self.Input_flag = CONFIG["Input_flag"]
        self.Input_File = CONFIG["Input_File"]
    
    def generate_random_numbers(self, size, grain):
        self.NLIST = rg.gen_uniform((size), np.float32)
        self.ULIST = rg.gen_uniform((size), np.float32)
        self.VLIST = rg.gen_uniform((size), np.float32)
        self.RLIST = rg.gen_uniform((size), np.float32)

        self.NFULL = pycuda.gpuarray.zeros((size), dtype=np.int32)
        self.S1FULL = pycuda.gpuarray.zeros((size), dtype=np.float32)
        self.S2FULL = pycuda.gpuarray.zeros((size), dtype=np.float32)
        self.S3FULL = pycuda.gpuarray.zeros((size), dtype=np.float32)

        mc.NPREC(self.NLIST, self.NFULL, self.__SIZE, block=(1,1,1), grid=(grain,1))
        mc.VPREC(self.ULIST, self.VLIST, self.S1FULL, self.S2FULL, self.S3FULL, self.Spin, block=(1,1,1), grid=(grain,1))

    def mc_init(self):
        self.grid = np.zeros((self.size*self.size), dtype=np.float32)
        if self.FM_Flag:
            mc.FM_N(self.grid)
        else:
            mc.AFM_N(self.grid)
        self.grid *= self.spin
        self.GPU_MAT = drv.mem_alloc(self.MAT_PARAMS.nbytes)
        self.GRID_GPU = drv.mem_alloc(self.grid.nbytes)
        self.BJ = drv.mem_alloc(self.T[0].nbytes)
        drv.memcpy_htod(self.GPU_MAT, self.MAT_PARAMS)
        drv.memcpy_htod(self.GRID_GPU, self.grid)
    
    def run_mc_tc(self, T):
        for i in tqdm(range(self.S_Wrap), desc="Stabilizing...", colour="blue"):
            mag_fluc =  np.zeros(self.calculation_runs)
            M = 0.0
            X = 0.0
            generate_random_numbers(self, self.stability_runs*self.stability_wrap, self.stability_runs*self.Blocks)
            dev.memcpy_htod(self.BJ,T)
            for j in range(stability_runs):
                mc.METROPOLIS_MC_DM1_6_6_6_12(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[i*self.Blocks:(i+1)*self.Blocks-1], self.S1FULL[i*self.Blocks:(i+1)*self.Blocks-1], self.S2FULL[i*self.Blocks:(i+1)*self.Blocks-1], self.S3FULL[i*self.Blocks:(i+1)*self.Blocks-1], self.RLIST[i*self.Blocks:(i+1)*self.Blocks-1], self.GPU_TRANS, self.GPU_UVEC, self.b, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
                mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(Threads,1,1), grid=(self.stability_runs*self.Blocks,1,1))

