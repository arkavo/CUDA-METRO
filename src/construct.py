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
    
    def generate_random_numbers(self, size):
        self.NLIST = rg.gen_uniform((size), np.float32)
        self.ULIST = rg.gen_uniform((size), np.float32)
        self.VLIST = rg.gen_uniform((size), np.float32)
        self.RLIST = rg.gen_uniform((size), np.float32)

        self.NFULL = pycuda.gpuarray.zeros((size), dtype=np.int32)
        self.S1FULL = pycuda.gpuarray.zeros((size), dtype=np.float32)
        self.S2FULL = pycuda.gpuarray.zeros((size), dtype=np.float32)
        self.S3FULL = pycuda.gpuarray.zeros((size), dtype=np.float32)

        mc.NPREC(self.NLIST, self.NFULL, self.__SIZE, block=(self.Blocks,1,1), grid=(self.Threads,1))
        mc.VPREC(self.ULIST, self.VLIST, self.S1FULL, self.S2FULL, self.S3FULL, self.Spin, block=(self.Blocks,1,1), grid=(self.Threads,1))
