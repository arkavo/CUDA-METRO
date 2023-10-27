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
        self.Input_Folder = "inputs/"
        self.Output_Folder = "outputs/"
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
        #placeholder spin = 1.0
        self.spin = 1.0
        spin_gpu = np.array([self.spin]).astype(np.float32)
        size_int = np.array([self.size]).astype(np.int32)
        self.b = np.array([self.B_C]).astype(np.float32)
        self.SGPU = drv.mem_alloc(spin_gpu.nbytes)
        self.GSIZE = drv.mem_alloc(size_int.nbytes)
        self.B_GPU = drv.mem_alloc(self.b.nbytes)
        drv.memcpy_htod(self.SGPU, spin_gpu)
        drv.memcpy_htod(self.GSIZE, size_int)
        drv.memcpy_htod(self.B_GPU, self.b)
        self.dmi_4 = np.load("dmi_4.npy")
        self.dmi_6 = np.load("dmi_6.npy")
        self.GPU_DMI_4 = drv.mem_alloc(self.dmi_4.nbytes)
        self.GPU_DMI_6 = drv.mem_alloc(self.dmi_6.nbytes)
        drv.memcpy_htod(self.GPU_DMI_4, self.dmi_4)
        drv.memcpy_htod(self.GPU_DMI_6, self.dmi_6)
        self.save_direcotry = "../"+self.Output_Folder+self.Prefix+"_"+self.Material+"_"+str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        os.mkdir(self.save_direcotry)

    def generate_random_numbers(self, mult):
        self.NLIST = rg.gen_uniform((self.C1), np.float32)
        self.ULIST = rg.gen_uniform((self.C1), np.float32)
        self.VLIST = rg.gen_uniform((self.C1), np.float32)
        self.RLIST = rg.gen_uniform((self.C1), np.float32)

        self.NFULL = pycuda.gpuarray.zeros((self.C1), dtype=np.int32)
        self.S1FULL = pycuda.gpuarray.zeros((self.C1), dtype=np.float32)
        self.S2FULL = pycuda.gpuarray.zeros((self.C1), dtype=np.float32)
        self.S3FULL = pycuda.gpuarray.zeros((self.C1), dtype=np.float32)

        mc.NPREC(self.NLIST, self.NFULL, self.GSIZE, block=(1,1,1), grid=(self.C1,1,1))
        mc.VPREC(self.ULIST, self.VLIST, self.S1FULL, self.S2FULL, self.S3FULL, self.SGPU, block=(1,1,1), grid=(self.C1,1,1))

    def mc_init(self):
        self.grid = np.zeros(((self.size*self.size),3), dtype=np.float32)
        self.TMATRIX = np.zeros((self.Blocks, 4)).astype(np.float32)
        self.GPU_TRANS = drv.mem_alloc(self.TMATRIX.nbytes)
        self.MAT_NAME, self.MAT_PARAMS = rm.read_2dmat("../"+self.Input_Folder+"TC_"+self.Material+".csv")
        
        if self.FM_Flag:
            mc.FM_N(self.grid)
        else:
            mc.AFM_N(self.grid)
        self.grid *= self.spin
        self.GPU_MAT = drv.mem_alloc(self.MAT_PARAMS.nbytes)
        self.GRID_GPU = drv.mem_alloc(self.grid.nbytes)
        if self.Static_T_Flag:
            self.T = np.array([self.Temps]).astype(np.float32)
        else:
            self.T = np.linspace(0.01, np.float32(2.0*self.MAT_PARAMS[24]), 31)
        self.BJ = drv.mem_alloc(self.T[0].nbytes)
        drv.memcpy_htod(self.GPU_MAT, self.MAT_PARAMS)
        drv.memcpy_htod(self.GRID_GPU, self.grid)
    
    def run_mc_tc_66612(self, T):
        for i in tqdm(range(self.S_Wrap), desc="Stabilizing...", colour="blue"):
            mag_fluc =  np.zeros(self.calculation_runs)
            M = 0.0
            X = 0.0
            #generate_random_numbers(self, self.stability_runs*self.stability_wrap, self.stability_runs*self.Blocks)
            drv.memcpy_htod(self.BJ,T[0])
            for j in range(self.stability_runs):
                mc.METROPOLIS_MC_DM1_6_6_6_12(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[j*self.Blocks:(j+1)*self.Blocks-1], self.S1FULL[j*self.Blocks:(j+1)*self.Blocks-1], self.S2FULL[j*self.Blocks:(j+1)*self.Blocks-1], self.S3FULL[j*self.Blocks:(j+1)*self.Blocks-1], self.RLIST[j*self.Blocks:(j+1)*self.Blocks-1], self.GPU_TRANS, self.GPU_DMI_6, self.B_GPU, self.GSIZE, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
                mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
            drv.memcpy_dtoh(self.grid, self.GRID_GPU)
            np.save(f"{self.save_direcotry}/grid_{i:04d}", self.grid)
    
    def run_mc_tc_4448(self, T):
        for i in tqdm(range(self.S_Wrap), desc="Stabilizing...", colour="blue"):
            mag_fluc =  np.zeros(self.calculation_runs)
            M = 0.0
            X = 0.0
            #generate_random_numbers(self, self.stability_runs*self.stability_wrap, self.stability_runs*self.Blocks)
            drv.memcpy_htod(self.BJ,T[0])
            for j in range(self.stability_runs):
                mc.METROPOLIS_MC_DM1_4_4_4_8(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[j*self.Blocks:(j+1)*self.Blocks-1], self.S1FULL[j*self.Blocks:(j+1)*self.Blocks-1], self.S2FULL[j*self.Blocks:(j+1)*self.Blocks-1], self.S3FULL[j*self.Blocks:(j+1)*self.Blocks-1], self.RLIST[j*self.Blocks:(j+1)*self.Blocks-1], self.GPU_TRANS, self.GPU_DMI_4, self.B_GPU, self.GSIZE, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
                mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
            drv.memcpy_dtoh(self.grid, self.GRID_GPU)
            np.save(f"{self.save_direcotry}/grid_{i:04d}", self.grid)
        
    
    def dump_state():
        pass

class Analyze():
    def __init__(self, directory):
        self.flist = os.listdir(directory)
        self.flist = [file for file in self.flist if file.endswith(".npy")]
        self.flist.sort()
        self.directory = directory
        print(self.flist)

    def spin_view(self):
        ctr = 0
        for file in self.flist:
            print(file)
            grid = np.load(self.directory+"/"+file)
            shape = grid.shape
            grid = grid.reshape((int(np.sqrt(shape[0])), int(np.sqrt(shape[0])), 3))
            spinx, spiny, spinz = grid[:,:,0], grid[:,:,1], grid[:,:,2]
            figure = plt.figure(dpi=400)
            plt.title("Spin Configuration at T = "+str(ctr))
            ax = figure.add_subplot(131)
            sns.heatmap(spinz, cbar=False, cmap="coolwarm", square=True, xticklabels=False, yticklabels=False, vmin=-1.0, vmax=1.0)
            ax.set_xlabel("Z")
            ax = figure.add_subplot(132)
            sns.heatmap(spiny, cbar=False, cmap="coolwarm", square=True, xticklabels=False, yticklabels=False)
            ax.set_xlabel("Y")
            ax = figure.add_subplot(133)
            sns.heatmap(spinx, cbar=False, cmap="coolwarm", square=True, xticklabels=False, yticklabels=False)
            ax.set_xlabel("X")
            plt.savefig(self.directory+"/spin_"+str(ctr)+".png")
            plt.close()
            ctr += 1
    
    def quiver_view(self):
        ctr = 0
        for file in self.flist:
            print(file)
            grid = np.load(self.directory+"/"+file)
            shape = grid.shape
            grid = grid.reshape((int(np.sqrt(shape[0])), int(np.sqrt(shape[0])), 3))
            spinx, spiny, spinz = grid[:,:,0], grid[:,:,1], grid[:,:,2]
            x_mesh, y_mesh = np.meshgrid(np.arange(0, int(np.sqrt(shape[0])), 1), np.arange(0, int(np.sqrt(shape[0])), 1))
            figure = plt.figure(dpi=400)
            plt.title("Spin Configuration at T = "+str(ctr))
            ax = figure.add_subplot(111)
            rgba = np.zeros((shape[0],4))
            spinz = np.reshape(spinz, shape[0])/1.5
            for i in range(shape[0]):
                rgba[i][3] = 1.0
                rgba[i][1] = 0.0#np.abs(spinz[i])
                if spinz[i] > 0:
                    rgba[i][0] = spinz[i]
                    rgba[i][2] = 0.0
                else:
                    rgba[i][0] = 0.0
                    rgba[i][2] = -spinz[i]
            plt.quiver(x_mesh, y_mesh, spinx, spiny, scale=1.5, scale_units="xy", pivot="mid", color=rgba, width=0.01, headwidth=3, headlength=4, headaxislength=3, minlength=0.1, minshaft=1)
            plt.savefig(self.directory+"/quiver_"+str(ctr)+".png")
            plt.close()
            ctr += 1