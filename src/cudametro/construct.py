# Constructors and wrappers for the Monte Carlo simulation CUDA code in 'montecarlo.py'
# Create additional directives and algorithms here to run the simulation, more information in the README.md
#--------------------------------------------

import pycuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.curandom import rand as curand 

rg = pycuda.curandom.XORWOWRandomNumberGenerator()

import numpy as np 
from numpy import random as rd
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import os 
import sys 
import re
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
import cudametro.montecarlo as mc
from tqdm import tqdm

import csv
import json
import datetime

def read_2dmat(filename):
    '''
    Reads the 2D material file from the ../../Inputs folder
    '''
    namelist = []
    params_list = np.array([])
    with open(filename, 'r') as f:
        data = f.readlines()
        for item in data:
            subdata = re.split(r'[,|+]', item)
            subdata[-1] = subdata[-1].strip("\n")
            namelist.append(subdata[0])
            params_list = np.append(params_list, np.float32(subdata[1:]))
    return namelist, np.array(params_list, dtype=np.float32)

class MonteCarlo:
    '''
    Default MC class, use this to instantiate the Simulation object
    '''
    def __init__(self, config, input_folder="../../inputs/", output_folder="outputs/"):
        with open(config, 'r') as f:
            CONFIG = json.load(f)
        # FLAGS
        self.Single_MAT_Flag = CONFIG["Single_Mat_Flag"]
        self.Static_T_Flag = CONFIG["Static_T_Flag"]
        self.Temps = [np.float32(item) for item in CONFIG["Temps"]]
        self.FM_Flag = CONFIG["FM_Flag"]
        self.DMI_Flag = CONFIG["DMI_Flag"]
        self.TC_Flag = CONFIG["TC_Flag"]
        
        # CONSTANTS AND PATHS
        self.Material = CONFIG["Material"]
        self.Multiple_Materials = CONFIG["Multiple_Materials"]
        self.Input_Folder = input_folder
        self.Output_Folder = output_folder
        self.B_C = np.float32(CONFIG["B"])
        self.size = CONFIG["SIZE"]
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
        self.C1 = self.Blocks*self.stability_runs
        self.C2 = self.Blocks*self.calculation_runs
        size_int = np.array([self.size]).astype(np.int32)
        self.b = np.array([self.B_C]).astype(np.float32)
        self.GSIZE = drv.mem_alloc(size_int.nbytes)
        self.B_GPU = drv.mem_alloc(self.b.nbytes)
        drv.memcpy_htod(self.GSIZE, size_int)
        drv.memcpy_htod(self.B_GPU, self.b)
        self.dmi_3 = np.load(script_dir+"/dmi_3.npy")
        self.dmi_4 = np.load(script_dir+"/dmi_4.npy")
        self.dmi_6 = np.load(script_dir+"/dmi_6.npy")
        self.GPU_DMI_3 = drv.mem_alloc(self.dmi_3.nbytes)
        self.GPU_DMI_4 = drv.mem_alloc(self.dmi_4.nbytes)
        self.GPU_DMI_6 = drv.mem_alloc(self.dmi_6.nbytes)
        drv.memcpy_htod(self.GPU_DMI_3, self.dmi_3)
        drv.memcpy_htod(self.GPU_DMI_4, self.dmi_4)
        drv.memcpy_htod(self.GPU_DMI_6, self.dmi_6)
        print("Inputs Folder default path: ", self.Input_Folder)
        self.MAT_NAME, self.MAT_PARAMS = read_2dmat(self.Input_Folder+"TC_"+self.Material+".csv")
        self.spin = self.MAT_PARAMS[0]
        spin_gpu = np.array([self.spin]).astype(np.float32)
        self.SGPU = drv.mem_alloc(spin_gpu.nbytes)
        drv.memcpy_htod(self.SGPU, spin_gpu)
        self.save_directory = "Output_"+self.Prefix+"_"+self.Material+"_"+str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        self.NBS = int(self.MAT_PARAMS[20]), int(self.MAT_PARAMS[21]), int(self.MAT_PARAMS[22]), int(self.MAT_PARAMS[23])
        print("Output Folder default path: ", self.save_directory)
        self.metadata = {
            "Material": self.Material,
            "Size": self.size,
            "Box": self.Box,
            "Blocks": self.Blocks,
            "Threads": 2,
            "stability_runs": self.stability_runs,
            "calculation_runs": self.calculation_runs,
            "Cmpl_Flag": self.Cmpl,
            "stability_wrap": self.S_Wrap,
            "calculation_wrap": self.C_Wrap,
            "Prefix": self.Prefix,
            "B": str(self.B_C),
            "spin": str(self.spin),
            "Temps": str(self.Temps),
            "FM_Flag": self.FM_Flag,
            "DMI_Flag": self.DMI_Flag,
            "TC_Flag": self.TC_Flag,
            "Input_flag": self.Input_flag,
            "Input_File": self.Input_File
        }
        metadata_file = json.dumps(self.metadata, indent=4)
        os.mkdir(self.save_directory)
        
        self.metadata_file = self.save_directory+"/metadata.json"
        with open(self.metadata_file, 'w+') as f:
            f.write(metadata_file)
        
    def display_material(self):
        '''
        Print material properties
        '''
        print(f"\tMaterial: {self.Material}")
        print(f"\tSize: {self.size}")
        print(f"\tSpin: {self.spin}")
        print(f"\tJ1: {self.MAT_PARAMS[1]:.4f} J2: {self.MAT_PARAMS[2]:.4f} J3: {self.MAT_PARAMS[3]:.4f} J4: {self.MAT_PARAMS[4]:.4f}")
        print(f"\tK1x: {self.MAT_PARAMS[5]:.4f} K1y: {self.MAT_PARAMS[6]:.4f} K1z: {self.MAT_PARAMS[7]:.4f}")
        print(f"\tK2x: {self.MAT_PARAMS[8]:.4f} K2y: {self.MAT_PARAMS[9]:.4f} K2z: {self.MAT_PARAMS[10]:.4f}")
        print(f"\tK3x: {self.MAT_PARAMS[11]:.4f} K3y: {self.MAT_PARAMS[12]:.4f} K3z: {self.MAT_PARAMS[13]:.4f}")
        print(f"\tK4x: {self.MAT_PARAMS[14]:.4f} K4y: {self.MAT_PARAMS[15]:.4f} K4z: {self.MAT_PARAMS[16]:.4f}")
        print(f"\tAx: {self.MAT_PARAMS[17]:.4f} Ay: {self.MAT_PARAMS[18]:.4f} Az: {self.MAT_PARAMS[19]:.4f}")
        print(f"\tNBS System: {self.NBS}")
        print(f"\tFile TC/NC: {self.MAT_PARAMS[24]}")
        print(f"\tConfig Temps: {self.Temps}")
        
    def generate_random_numbers(self, deprecate):
        '''
        Generate 4 uniform random number arrays for the simulation and copies them to the GPU
        '''
        self.NLIST = rg.gen_uniform((self.C1), np.float32)
        self.ULIST = rg.gen_uniform((self.C1), np.float32)
        self.VLIST = rg.gen_uniform((self.C1), np.float32)
        self.RLIST = rg.gen_uniform((self.C1), np.float32)

        self.NFULL = pycuda.gpuarray.zeros((self.C1), dtype=np.int32)
        self.S1FULL = pycuda.gpuarray.zeros((self.C1), dtype=np.float32)
        self.S2FULL = pycuda.gpuarray.zeros((self.C1), dtype=np.float32)
        self.S3FULL = pycuda.gpuarray.zeros((self.C1), dtype=np.float32)

        mc.NPREC(self.NLIST, self.NFULL, self.GSIZE, block=(2,1,1), grid=(self.C1,1,1))
        mc.VPREC(self.ULIST, self.VLIST, self.S1FULL, self.S2FULL, self.S3FULL, self.SGPU, block=(2,1,1), grid=(self.C1,1,1))

    def generate_ising_numbers(self, deprecate):
        '''
        Generate 4 uniform random number arrays for the simulation and copies them to the GPU
        '''
        self.NLIST = rg.gen_uniform((self.C1), np.float32)
        self.ULIST = rg.gen_uniform((self.C1), np.float32)
        self.VLIST = rg.gen_uniform((self.C1), np.float32)
        self.RLIST = rg.gen_uniform((self.C1), np.float32)

        self.NFULL = pycuda.gpuarray.zeros((self.C1), dtype=np.int32)
        self.S1FULL = pycuda.gpuarray.zeros((self.C1), dtype=np.float32)
        self.S2FULL = pycuda.gpuarray.zeros((self.C1), dtype=np.float32)
        self.S3FULL = pycuda.gpuarray.zeros((self.C1), dtype=np.float32)

        mc.NPREC(self.NLIST, self.NFULL, self.GSIZE, block=(2,1,1), grid=(self.C1,1,1))
        mc.ISING(self.ULIST, self.VLIST, self.S1FULL, self.S2FULL, self.S3FULL, self.SGPU, block=(2,1,1), grid=(self.C1,1,1))
    
    def sampler(self):
        self.N_SAMPLE = rg.gen_uniform((self.Blocks), np.float32)
        self.GPU_N_SAMPLE = drv.mem_alloc(self.N_SAMPLE.nbytes)
        mc.NPREC(self.N_SAMPLE, self.GPU_N_SAMPLE, self.GSIZE, block=(2,1,1), grid=(self.Blocks,1,1))

    def mc_init(self):
        '''
        Initialize the simulation
        '''
        self.grid = np.zeros((self.size*self.size*3)).astype(np.float32)

        self.GRID_GPU = drv.mem_alloc(self.grid.nbytes)
        self.TMATRIX = np.zeros((self.Blocks, 4)).astype(np.float32)

        self.GPU_TRANS = drv.mem_alloc(self.TMATRIX.nbytes)
        
        if self.FM_Flag:
            mc.FM_N(self.grid, self.size)
        else:
            mc.AFM_N(self.grid, self.size)
        self.grid *= self.spin

        self.GPU_MAT = drv.mem_alloc(self.MAT_PARAMS.nbytes)
        
        if self.Static_T_Flag:
            self.T = self.Temps
        else:
            self.T = np.linspace(0.01, np.float32(2.0*self.MAT_PARAMS[24]), 11)

        self.BJ = drv.mem_alloc(self.T[0].nbytes)
        drv.memcpy_htod(self.GPU_MAT, self.MAT_PARAMS)
        drv.memcpy_htod(self.GRID_GPU, self.grid)
 
    def grid_reset(self):
        self.grid = np.zeros((self.size*self.size*3)).astype(np.float32)
        if self.FM_Flag:
            mc.FM_N(self.grid, self.size)
        else:
            mc.AFM_N(self.grid, self.size)
        self.grid *= self.spin
        drv.memcpy_htod(self.GRID_GPU, self.grid)
    '''
    DMI SECTOR
    This section has the MC simulation algorithms for materials that use a DMI term
    In order, this contains the following functions:
    run_mc_dmi_66612: Runs the MC simulation for a 6 6 6 12 crystal structure material
    run_mc_dmi_4448: Runs the MC simulation for a 4 4 4 8 crystal structure material
    run_mc_dmi_3636: Runs the MC simulation for a 3 6 3 6 crystal structure material
    '''
    def run_mc_dmi_66612(self, T):
        beta = np.array([1.0 / (T * 8.6173e-2)],dtype=np.float32)
        drv.memcpy_htod(self.BJ,beta[0])
        for j in range(self.stability_runs):
            mc.METROPOLIS_MC_DM1_6_6_6_12(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[j*self.Blocks:(j+1)*self.Blocks], self.S1FULL[j*self.Blocks:(j+1)*self.Blocks], self.S2FULL[j*self.Blocks:(j+1)*self.Blocks], self.S3FULL[j*self.Blocks:(j+1)*self.Blocks], self.RLIST[j*self.Blocks:(j+1)*self.Blocks], self.GPU_TRANS, self.GPU_DMI_6, self.B_GPU, self.GSIZE, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
            mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(2,1,1), grid=(self.Blocks,1,1))
        drv.memcpy_dtoh(self.grid, self.GRID_GPU)
        drv.Context.synchronize()
        return self.grid


    def run_mc_dmi_4448(self, T):
        beta = np.array([1.0 / (T * 8.6173e-2)],dtype=np.float32)
        drv.memcpy_htod(self.BJ,beta[0])
        for j in range(self.stability_runs):
            mc.METROPOLIS_MC_DM1_4_4_4_8(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[j*self.Blocks:(j+1)*self.Blocks], self.S1FULL[j*self.Blocks:(j+1)*self.Blocks], self.S2FULL[j*self.Blocks:(j+1)*self.Blocks], self.S3FULL[j*self.Blocks:(j+1)*self.Blocks], self.RLIST[j*self.Blocks:(j+1)*self.Blocks], self.GPU_TRANS, self.GPU_DMI_6, self.B_GPU, self.GSIZE, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
            mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(2,1,1), grid=(self.Blocks,1,1))
        drv.memcpy_dtoh(self.grid, self.GRID_GPU)
        drv.Context.synchronize()
        return self.grid
    
    def run_mc_dmi_3636(self, T):
        beta = np.array([1.0 / (T * 8.6173e-2)],dtype=np.float32)
        drv.memcpy_htod(self.BJ,beta[0])
        for j in range(self.stability_runs):
            mc.METROPOLIS_MC_DM1_3_6_3_6(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[j*self.Blocks:(j+1)*self.Blocks], self.S1FULL[j*self.Blocks:(j+1)*self.Blocks], self.S2FULL[j*self.Blocks:(j+1)*self.Blocks], self.S3FULL[j*self.Blocks:(j+1)*self.Blocks], self.RLIST[j*self.Blocks:(j+1)*self.Blocks], self.GPU_TRANS, self.GPU_DMI_3, self.B_GPU, self.GSIZE, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
            mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(2,1,1), grid=(self.Blocks,1,1))
        drv.memcpy_dtoh(self.grid, self.GRID_GPU)
        drv.Context.synchronize()
        return self.grid
    
    
    # DMI SECTOR END
    
    # TC SECTOR
    # This section has the MC simulation algorithms for finding the TC of a material
    
    def run_mc_tc_4448(self, T):
        '''
        Run the MC simulation for a 4 4 4 8 crystal structure material
        '''
        Mt, Xt = 0.0, 0.0
        ct = 0
        M, X = np.zeros(self.S_Wrap), np.zeros(self.S_Wrap)
        for i in tqdm(range(self.S_Wrap), desc=f"Stabilizing at {T:.3f}", colour="blue"):
            mag_fluc =  np.zeros(self.stability_runs)
            beta = np.array([1.0 / (T * 8.6173e-2)],dtype=np.float32)
            drv.memcpy_htod(self.BJ, beta)
            for j in range(self.stability_runs):
                mc.METROPOLIS_MC_DM0_4_4_4_8(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[j*self.Blocks:(j+1)*self.Blocks], self.S1FULL[j*self.Blocks:(j+1)*self.Blocks], self.S2FULL[j*self.Blocks:(j+1)*self.Blocks], self.S3FULL[j*self.Blocks:(j+1)*self.Blocks], self.RLIST[j*self.Blocks:(j+1)*self.Blocks], self.GPU_TRANS, self.B_GPU, self.GSIZE, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
                mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(2,1,1), grid=(self.Blocks,1,1))
            drv.memcpy_dtoh(self.grid, self.GRID_GPU)
            self.grid = self.grid.reshape((self.size, self.size, 3))
            magx, magy, magz = self.grid[:,:,0], self.grid[:,:,1], self.grid[:,:,2]
            mag = np.array([np.sum(magx) , np.sum(magy) , np.sum(magz)])/(self.size**2)
            M[i] = np.abs(np.linalg.norm(mag))
        Mt, Xt = np.mean(M), np.std(M)/T
        print(f"Mean Magnetization at {T:.3f} = {Mt:.3f}")
        print(f"Mean Susceptibility at {T:.3f} = {Xt:.3f}")
        drv.Context.synchronize()
        return Mt, Xt
    
    def run_mc_tc_66612(self, T):
        '''
        Run the MC simulation for a 6 6 6 12 crystal structure material
        '''
        Mt, Xt = 0.0, 0.0
        ct = 0
        M, X = np.zeros(self.S_Wrap), np.zeros(self.S_Wrap)
        for i in tqdm(range(self.S_Wrap), desc=f"Stabilizing at {T:.3f}", colour="blue"):
            mag_fluc =  np.zeros(self.stability_runs)
            beta = np.array([1.0 / (T * 8.6173e-2)],dtype=np.float32)
            drv.memcpy_htod(self.BJ, beta)
            for j in range(self.stability_runs):
                mc.METROPOLIS_MC_DM0_6_6_6_12(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[j*self.Blocks:(j+1)*self.Blocks], self.S1FULL[j*self.Blocks:(j+1)*self.Blocks], self.S2FULL[j*self.Blocks:(j+1)*self.Blocks], self.S3FULL[j*self.Blocks:(j+1)*self.Blocks], self.RLIST[j*self.Blocks:(j+1)*self.Blocks], self.GPU_TRANS, self.B_GPU, self.GSIZE, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
                mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(2,1,1), grid=(self.Blocks,1,1))
            drv.memcpy_dtoh(self.grid, self.GRID_GPU)
            self.grid = self.grid.reshape((self.size, self.size, 3))
            magx, magy, magz = self.grid[:,:,0], self.grid[:,:,1], self.grid[:,:,2]
            mag = np.array([np.sum(magx) , np.sum(magy) , np.sum(magz)])/(self.size**2)
            M[i] = np.abs(np.linalg.norm(mag))
        Mt, Xt = np.mean(M), np.std(M)/T
        print(f"Mean Magnetization at {T:.3f} = {Mt:.3f}")
        print(f"Mean Susceptibility at {T:.3f} = {Xt:.3f}")
        drv.Context.synchronize()
        return Mt, Xt
    
    def run_mc_tc_3636(self, T):
        '''
        Run the MC simulation for a 3 6 3 6 crystal structure material
        '''
        Mt, Xt = 0.0, 0.0
        ct = 0
        M = np.zeros(self.S_Wrap)
        for i in tqdm(range(self.S_Wrap), desc=f"Stabilizing at {T:.3f}", colour="blue"):
            mag_fluc =  np.zeros(self.stability_runs)
            beta = np.array([1.0 / (T * 8.6173e-2)],dtype=np.float32)
            drv.memcpy_htod(self.BJ, beta)
            for j in range(self.stability_runs):
                mc.METROPOLIS_MC_DM0_3_6_3_6(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[j*self.Blocks:(j+1)*self.Blocks], self.S1FULL[j*self.Blocks:(j+1)*self.Blocks], self.S2FULL[j*self.Blocks:(j+1)*self.Blocks], self.S3FULL[j*self.Blocks:(j+1)*self.Blocks], self.RLIST[j*self.Blocks:(j+1)*self.Blocks], self.GPU_TRANS, self.B_GPU, self.GSIZE, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
                mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(2,1,1), grid=(self.Blocks,1,1))
            drv.memcpy_dtoh(self.grid, self.GRID_GPU)
            self.grid = self.grid.reshape((self.size, self.size, 3))
            magx, magy, magz = self.grid[:,:,0], self.grid[:,:,1], self.grid[:,:,2]
            mag = np.array([np.sum(magx) , np.sum(magy) , np.sum(magz)])/(self.size**2)
            M[i] = np.abs(np.linalg.norm(mag))
        Mt, Xt = np.mean(M), np.std(M)/T
        print(f"Mean Magnetization at {T:.3f} = {Mt:.3f}")
        print(f"Mean Susceptibility at {T:.3f} = {Xt:.3f}")
        drv.Context.synchronize()
        return Mt, Xt
    
    '''
    # Deprecated function
    def run_mc_tc_3636_2(self, T):
        Mt, Xt = 0.0, 0.0
        ct = 0
        M = np.zeros(self.S_Wrap)
        for i in tqdm(range(self.S_Wrap), desc=f"Stabilizing at {T:.3f}", colour="blue"):
            mag_fluc =  np.zeros(self.stability_runs)
            beta = np.array([1.0 / (T * 8.6173e-2)],dtype=np.float32)
            drv.memcpy_htod(self.BJ, beta)
            for j in range(self.stability_runs):
                mc.METROPOLIS_MC_DM2_3_6_3_6(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[j*self.Blocks:(j+1)*self.Blocks], self.S1FULL[j*self.Blocks:(j+1)*self.Blocks], self.S2FULL[j*self.Blocks:(j+1)*self.Blocks], self.S3FULL[j*self.Blocks:(j+1)*self.Blocks], self.RLIST[j*self.Blocks:(j+1)*self.Blocks], self.GPU_TRANS, self.B_GPU, self.GSIZE, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
                mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(2,1,1), grid=(self.Blocks,1,1))
            drv.memcpy_dtoh(self.grid, self.GRID_GPU)
            self.grid = self.grid.reshape((self.size, self.size, 3))
            magx, magy, magz = self.grid[:,:,0], self.grid[:,:,1], self.grid[:,:,2]
            mag = np.array([np.sum(magx) , np.sum(magy) , np.sum(magz)])/(self.size**2)
            M[i] = np.abs(np.linalg.norm(mag))
        Mt, Xt = np.mean(M), np.std(M)/T
        print(f"Mean Magnetization at {T:.3f} = {Mt:.3f}")
        print(f"Mean Susceptibility at {T:.3f} = {Xt:.3f}")
        
        return Mt, Xt
    '''

    def run_mc_tc_2242(self, T):
        '''
        Run the MC simulation for a 2 2 4 2 crystal structure material
        '''
        Mt, Xt = 0.0, 0.0
        ct = 0
        M = np.zeros(self.S_Wrap)
        for i in tqdm(range(self.S_Wrap), desc=f"Stabilizing at {T:.3f}", colour="blue"):
            mag_fluc =  np.zeros(self.stability_runs)
            beta = np.array([1.0 / (T * 8.6173e-2)],dtype=np.float32)
            drv.memcpy_htod(self.BJ, beta)
            for j in range(self.stability_runs):
                mc.METROPOLIS_MC_DM0_2_2_4_2(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[j*self.Blocks:(j+1)*self.Blocks], self.S1FULL[j*self.Blocks:(j+1)*self.Blocks], self.S2FULL[j*self.Blocks:(j+1)*self.Blocks], self.S3FULL[j*self.Blocks:(j+1)*self.Blocks], self.RLIST[j*self.Blocks:(j+1)*self.Blocks], self.GPU_TRANS, self.B_GPU, self.GSIZE, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
                mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(2,1,1), grid=(self.Blocks,1,1))
            drv.memcpy_dtoh(self.grid, self.GRID_GPU)
            self.grid = self.grid.reshape((self.size, self.size, 3))
            magx, magy, magz = self.grid[:,:,0], self.grid[:,:,1], self.grid[:,:,2]
            mag = np.array([np.sum(magx) , np.sum(magy) , np.sum(magz)])/(self.size**2)
            M[i] = np.abs(np.linalg.norm(mag))
        Mt, Xt = np.mean(M), np.std(M)/T
        print(f"Mean Magnetization at {T:.3f} = {Mt:.3f}")
        print(f"Mean Susceptibility at {T:.3f} = {Xt:.3f}")
        drv.Context.synchronize()
        return Mt, Xt
    
    def run_mc_tc_2424(self, T):
        '''
        Run the MC simulation for a 2 4 2 4 crystal structure material
        '''
        Mt, Xt = 0.0, 0.0
        ct = 0
        M = np.zeros(self.S_Wrap)
        for i in tqdm(range(self.S_Wrap), desc=f"Stabilizing at {T:.3f}", colour="blue"):
            mag_fluc =  np.zeros(self.stability_runs)
            beta = np.array([1.0 / (T * 8.6173e-2)],dtype=np.float32)
            drv.memcpy_htod(self.BJ, beta)
            for j in range(self.stability_runs):
                mc.METROPOLIS_MC_DM0_2_4_2_4(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[j*self.Blocks:(j+1)*self.Blocks], self.S1FULL[j*self.Blocks:(j+1)*self.Blocks], self.S2FULL[j*self.Blocks:(j+1)*self.Blocks], self.S3FULL[j*self.Blocks:(j+1)*self.Blocks], self.RLIST[j*self.Blocks:(j+1)*self.Blocks], self.GPU_TRANS, self.B_GPU, self.GSIZE, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
                mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(2,1,1), grid=(self.Blocks,1,1))
            drv.memcpy_dtoh(self.grid, self.GRID_GPU)
            self.grid = self.grid.reshape((self.size, self.size, 3))
            magx, magy, magz = self.grid[:,:,0], self.grid[:,:,1], self.grid[:,:,2]
            mag = np.array([np.sum(magx) , np.sum(magy) , np.sum(magz)])/(self.size**2)
            M[i] = np.abs(np.linalg.norm(mag))
        Mt, Xt = np.mean(M), np.std(M)/T
        print(f"Mean Magnetization at {T:.3f} = {Mt:.3f}")
        print(f"Mean Susceptibility at {T:.3f} = {Xt:.3f}")
        drv.Context.synchronize()
        return Mt, Xt

    # TC SECTOR END
    
    # TC EN SECTOR
    # This section has the MC simulation algorithms for finding the Energy/atom of a material while simulating.
    # In order, this contains the following functions:

    def run_mc_tc_en_3636(self, T):
        '''
        Run the MC simulation for a 3 6 3 6 crystal structure material while calculating the Energy/atom
        '''
        Mt, Xt = 0.0, 0.0
        ct = 0
        M = np.zeros(self.S_Wrap)
        E = np.zeros(self.S_Wrap)
        Et = np.zeros(self.Blocks).astype(np.float32)
        GPU_ET = drv.mem_alloc(Et.nbytes)

        for i in tqdm(range(self.S_Wrap), desc=f"Stabilizing at {T:.3f}", colour="blue"):
            mag_fluc =  np.zeros(self.stability_runs)
            beta = np.array([1.0 / (T * 8.6173e-2)],dtype=np.float32)
            drv.memcpy_htod(self.BJ, beta)
            ET_S = np.zeros(self.stability_runs)
            for j in range(self.stability_runs):
                mc.METROPOLIS_MC_DM0_3_6_3_6(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[j*self.Blocks:(j+1)*self.Blocks], self.S1FULL[j*self.Blocks:(j+1)*self.Blocks], self.S2FULL[j*self.Blocks:(j+1)*self.Blocks], self.S3FULL[j*self.Blocks:(j+1)*self.Blocks], self.RLIST[j*self.Blocks:(j+1)*self.Blocks], self.GPU_TRANS, self.B_GPU, self.GSIZE, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
                mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(2,1,1), grid=(self.Blocks,1,1))
                self.sampler()
                mc.EN_CALC_3_6_3_6(self.GPU_MAT, self.GRID_GPU, self.GPU_N_SAMPLE, GPU_ET, self.B_GPU, self.GSIZE, block=(2,1,1), grid=(self.size*self.size,1,1))
                drv.memcpy_dtoh(Et, GPU_ET)
                ET_S[j] = np.mean(Et)
            drv.memcpy_dtoh(self.grid, self.GRID_GPU)
            self.grid = self.grid.reshape((self.size, self.size, 3))
            magx, magy, magz = self.grid[:,:,0], self.grid[:,:,1], self.grid[:,:,2]
            mag = np.array([np.sum(magx) , np.sum(magy) , np.sum(magz)])/(self.size**2)
            M[i] = np.abs(np.linalg.norm(mag))
            E[i] = np.mean(ET_S)
        Mt, Xt, Et = np.mean(M), np.std(M)/T, np.std(E[-10:])/T**2
        np.save(f"{self.save_directory}/En_{T:.3f}", E)
        print(f"Mean Magnetization at {T:.3f} = {Mt:.3f}")
        print(f"Mean Susceptibility at {T:.3f} = {Xt:.3f}")
        print(f"Mean Specific Heat at {T:.3f} = {Et:.3f}")
        drv.Context.synchronize()
        return Mt, Xt, Et
    '''
    def run_mc_tc_en_3636_2(self, T):
        Mt, Xt = 0.0, 0.0
        ct = 0
        M = np.zeros(self.S_Wrap)
        E = np.zeros(self.S_Wrap)
        Et = np.zeros(self.Blocks).astype(np.float32)
        GPU_ET = drv.mem_alloc(Et.nbytes)

        for i in tqdm(range(self.S_Wrap), desc=f"Stabilizing at {T:.3f}", colour="blue"):
            mag_fluc =  np.zeros(self.stability_runs)
            beta = np.array([1.0 / (T * 8.6173e-2)],dtype=np.float32)
            drv.memcpy_htod(self.BJ, beta)
            ET_S = np.zeros(self.stability_runs)
            for j in range(self.stability_runs):
                mc.METROPOLIS_MC_DM1_3_6_3_6(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[j*self.Blocks:(j+1)*self.Blocks], self.S1FULL[j*self.Blocks:(j+1)*self.Blocks], self.S2FULL[j*self.Blocks:(j+1)*self.Blocks], self.S3FULL[j*self.Blocks:(j+1)*self.Blocks], self.RLIST[j*self.Blocks:(j+1)*self.Blocks], self.GPU_TRANS, self.B_GPU, self.GSIZE, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
                mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(2,1,1), grid=(self.Blocks,1,1))
                self.sampler()
                mc.EN_CALC_3_6_3_6_2(self.GPU_MAT, self.GRID_GPU, self.GPU_N_SAMPLE, GPU_ET, self.B_GPU, self.GSIZE, block=(2,1,1), grid=(self.size*self.size,1,1))
                drv.memcpy_dtoh(Et, GPU_ET)
                ET_S[j] = np.mean(Et)
            drv.memcpy_dtoh(self.grid, self.GRID_GPU)
            self.grid = self.grid.reshape((self.size, self.size, 3))
            magx, magy, magz = self.grid[:,:,0], self.grid[:,:,1], self.grid[:,:,2]
            mag = np.array([np.sum(magx) , np.sum(magy) , np.sum(magz)])/(self.size**2)
            M[i] = np.abs(np.linalg.norm(mag))
            E[i] = np.mean(ET_S)
        Mt, Xt, Et = np.mean(M), np.std(M)/T, np.std(E[-10:])/T**2
        np.save(f"{self.save_directory}/En_{T:.3f}", E)
        print(f"Mean Magnetization at {T:.3f} = {Mt:.3f}")
        print(f"Mean Susceptibility at {T:.3f} = {Xt:.3f}")
        print(f"Mean Specific Heat at {T:.3f} = {Et:.3f}")
        return Mt, Xt, Et
    '''
    def en_3636(self, T):
        '''
        Calculate the Energy/atom for a 3 6 3 6 crystal structure material
        '''
        Et = np.zeros(self.Blocks).astype(np.float32)
        GPU_ET = drv.mem_alloc(Et.nbytes)
        mc.EN_CALC_3_6_3_6(self.GPU_MAT, self.GRID_GPU, self.B_GPU, self.GPU_N_SAMPLE, self.GSIZE, GPU_ET, block=(2,1,1), grid=(self.Blocks,1,1))
        drv.memcpy_dtoh(Et, GPU_ET)
        drv.Context.synchronize()
        return Et
    
    '''
    def en_3636_2(self, T):
        Et = np.zeros(self.Blocks).astype(np.float32)
        GPU_ET = drv.mem_alloc(Et.nbytes)
        mc.EN_CALC_3_6_3_6_2(self.GPU_MAT, self.GRID_GPU, self.B_GPU, self.GPU_N_SAMPLE, self.GSIZE, GPU_ET, block=(2,1,1), grid=(self.Blocks,1,1))
        drv.memcpy_dtoh(Et, GPU_ET)
        drv.Context.synchronize()
        return Et
    '''
    def run_mc_tc_en_66612(self, T):
        '''
        Calculate the Energy/atom for a 6 6 6 12 crystal structure material
        '''
        Et = np.zeros(self.Blocks).astype(np.float32)
        GPU_ET = drv.mem_alloc(Et.nbytes)
        mc.EN_CALC_6_6_6_12(self.GPU_MAT, self.GRID_GPU, self.B_GPU, self.GPU_N_SAMPLE, self.GSIZE, GPU_ET, block=(2,1,1), grid=(self.Blocks,1,1))
        drv.memcpy_dtoh(Et, GPU_ET)
        drv.Context.synchronize()
        return Et
        
    # TC EN SECTOR END
    
    # MISC SECTOR
    
    '''
    def run_mc_dmi_36362(self, T):
        beta = np.array([1.0 / (T * 8.6173e-2)],dtype=np.float32)
        drv.memcpy_htod(self.BJ,beta[0])
        for j in range(self.stability_runs):
            mc.METROPOLIS_MC_DM2_3_6_3_6(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[j*self.Blocks:(j+1)*self.Blocks], self.S1FULL[j*self.Blocks:(j+1)*self.Blocks], self.S2FULL[j*self.Blocks:(j+1)*self.Blocks], self.S3FULL[j*self.Blocks:(j+1)*self.Blocks], self.RLIST[j*self.Blocks:(j+1)*self.Blocks], self.GPU_TRANS, self.B_GPU, self.GSIZE, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
            mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(2,1,1), grid=(self.Blocks,1,1))
        drv.memcpy_dtoh(self.grid, self.GRID_GPU)
        return self.grid
    
    def dump_state():
        pass
    '''
    
    def cleanup(self):
        '''
        Clean up the GPU GRID memory
        '''
        self.GRID_GPU.free()

class Analyze():
    def __init__(self, directory, reverse=False, input_folder="../../inputs/"):
        '''
        Initialize the Analyze class, used to analyze the output of the MC simulation
        '''
        self.flist = os.listdir(directory)
        self.flist = [file for file in self.flist if file.endswith(".npy") and file.startswith("grid")]
        self.flist.sort(reverse=reverse)
        self.directory = directory
        if not os.path.exists(self.directory+"/spin"):
            os.mkdir(self.directory+"/spin")
        if not os.path.exists(self.directory+"/quiver"):
            os.mkdir(self.directory+"/quiver")
        print(f"{len(self.flist)} files found.....Analyzing")
        self.metadata = json.load(open(self.directory+"/metadata.json", 'r'))
        self.size = self.metadata["Size"]
        self.spin = np.float32(self.metadata["spin"])
        self.Blocks = self.metadata["Blocks"]
        self.Material = self.metadata["Material"]
        self.Input_Folder = input_folder
        self.MAT_NAME, self.MAT_PARAMS = read_2dmat(self.Input_Folder+"TC_"+self.Material+".csv")
        self.GPU_MAT = drv.mem_alloc(self.MAT_PARAMS.nbytes)
        drv.memcpy_htod(self.GPU_MAT, self.MAT_PARAMS)
        self.B_GPU = drv.mem_alloc(self.MAT_PARAMS[0].nbytes)
        drv.memcpy_htod(self.B_GPU, np.array([self.metadata["B"]]).astype(np.float32))
        self.GSIZE = drv.mem_alloc(np.array([self.size]).astype(np.int32).nbytes)
        drv.memcpy_htod(self.GSIZE, np.array([self.size]).astype(np.int32))
        # To fix energy plot issue
        self.GPU_DMI_6 = np.ones((3,3)).astype(np.float32)

    def spin_view(self):
        '''
        Generate a heatmap of the spin configuration at each time step
        '''
        ctr = 0
        for file in self.flist:
            print(f"Processing {file} mode: spin heatmap", end="\r")
            grid = np.load(self.directory+"/"+file)
            shape = grid.shape
            with open(self.directory+"/metadata.json", 'r') as f:
                metadata = json.load(f)
            self.spin = np.float32(metadata["spin"])
            grid = grid.reshape((int(np.sqrt(shape[0]/3)), int(np.sqrt(shape[0]/3)), 3))
            #grid = grid.reshape((64, 64, 3))
            spinx, spiny, spinz = grid[:,:,0], grid[:,:,1], grid[:,:,2]
            figure = plt.figure(dpi=400)
            plt.title("Spin Configuration at T = "+str(ctr))
            ax = figure.add_subplot(131)
            sns.heatmap(spinz, cbar=False, cmap="coolwarm", square=True, xticklabels=False, yticklabels=False, vmin=-self.spin, vmax=self.spin)
            ax.set_xlabel("Z")
            ax = figure.add_subplot(132)
            sns.heatmap(spiny, cbar=False, cmap="coolwarm", square=True, xticklabels=False, yticklabels=False, vmin=-self.spin, vmax=self.spin)
            ax.set_xlabel("Y")
            ax = figure.add_subplot(133)
            sns.heatmap(spinx, cbar=False, cmap="coolwarm", square=True, xticklabels=False, yticklabels=False, vmin=-self.spin, vmax=self.spin)
            ax.set_xlabel("X")
            plt.savefig(self.directory+"/spin/spin_"+str(ctr)+".png")
            plt.close()
            ctr += 1
        print("\n")
        print("Exiting spin view mode")
    def quiver_view(self):
        '''
        Generate a quiver plot of the spin configuration at each time step
        '''
        ctr = 0
        for file in self.flist:
            print(f"Processing {file} mode: quiver", end="\r")
            grid = np.load(self.directory+"/"+file)
            shape = grid.shape
            grid = grid.reshape((int(np.sqrt(shape[0]/3)), int(np.sqrt(shape[0]/3)), 3))
            spinx, spiny, spinz = grid[:,:,0], grid[:,:,1], grid[:,:,2]
            x_mesh, y_mesh = np.meshgrid(np.arange(0, int(np.sqrt(shape[0]/3)), 1), np.arange(0, int(np.sqrt(shape[0]/3)), 1))
            figure = plt.figure(figsize=(10,10),dpi=400)
            plt.title("Spin Configuration at T = "+str(ctr))
            ax = figure.add_subplot(111)
            rgba = np.zeros((shape[0],4))
            spinz = np.reshape(spinz, int(shape[0]/3))
            norm = Normalize()
            norm.autoscale(spinz)
            colormap = cm.bwr
            plt.quiver(x_mesh, y_mesh, spinx, spiny, scale=self.spin, scale_units="xy", pivot="mid", color=colormap(norm(spinz)), width=0.01, headwidth=3, headlength=4, headaxislength=3, minlength=0.1, minshaft=1)
            plt.savefig(self.directory+"/quiver/quiver_"+str(ctr)+".png")
            plt.close()
            ctr += 1
        print("\n")
        print("Exiting quiver view mode")
    def en_66612(self):
        '''
        Visualize the Energy/atom for a 6 6 6 12 crystal structure material
        '''
        ctr = 0
        E_f = np.zeros(len(self.flist))
        for file in self.flist:
            print(f"Processing {file}", end="\r")
            grid = np.load(self.directory+"/"+file)
            shape = grid.shape
            grid = grid.reshape((int(np.sqrt(shape[0]/3)), int(np.sqrt(shape[0]/3)), 3))
            Et = np.zeros(self.Blocks).astype(np.float32)
            self.GRID_GPU = drv.mem_alloc(grid.nbytes)
            drv.memcpy_htod(self.GRID_GPU, grid)
            GPU_ET = drv.mem_alloc(Et.nbytes)
            mc.EN_CALC_6_6_6_12(self.GPU_MAT, self.GRID_GPU, self.B_GPU, self.GSIZE, GPU_ET, self.GPU_DMI_6, block=(2,1,1), grid=(self.Blocks,1,1))
            drv.memcpy_dtoh(Et, GPU_ET)
            print(np.mean(-Et))
            E_f[ctr] = -np.mean(Et)
            ctr += 1
        np.save(self.directory+"/En", E_f)
        plt.plot(E_f)
        plt.savefig(self.directory+"/En.png")
        np.savetxt(self.directory+"/En.txt", E_f)
        


    
    
    