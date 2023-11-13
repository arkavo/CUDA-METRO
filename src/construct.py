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
        size_int = np.array([self.size]).astype(np.int32)
        self.b = np.array([self.B_C]).astype(np.float32)
        self.GSIZE = drv.mem_alloc(size_int.nbytes)
        self.B_GPU = drv.mem_alloc(self.b.nbytes)
        drv.memcpy_htod(self.GSIZE, size_int)
        drv.memcpy_htod(self.B_GPU, self.b)
        self.dmi_4 = np.load("dmi_4.npy")
        self.dmi_6 = np.load("dmi_6.npy")
        self.GPU_DMI_4 = drv.mem_alloc(self.dmi_4.nbytes)
        self.GPU_DMI_6 = drv.mem_alloc(self.dmi_6.nbytes)
        drv.memcpy_htod(self.GPU_DMI_4, self.dmi_4)
        drv.memcpy_htod(self.GPU_DMI_6, self.dmi_6)
        self.MAT_NAME, self.MAT_PARAMS = rm.read_2dmat("../"+self.Input_Folder+"TC_"+self.Material+".csv")
        self.spin = self.MAT_PARAMS[0]
        spin_gpu = np.array([self.spin]).astype(np.float32)
        self.SGPU = drv.mem_alloc(spin_gpu.nbytes)
        drv.memcpy_htod(self.SGPU, spin_gpu)
        self.save_direcotry = "../"+self.Output_Folder+self.Prefix+"_"+self.Material+"_"+str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        self.NBS = int(self.MAT_PARAMS[20]), int(self.MAT_PARAMS[21]), int(self.MAT_PARAMS[22]), int(self.MAT_PARAMS[23])

        self.metadata = {
            "Material": self.Material,
            "Size": self.size,
            "Box": self.Box,
            "Blocks": self.Blocks,
            "Threads": self.Threads,
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
        os.mkdir(self.save_direcotry)
        
        self.metadata_file = self.save_direcotry+"/metadata.json"
        with open(self.metadata_file, 'w+') as f:
            f.write(metadata_file)
        

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
        
        print(self.GRID_GPU)
        if self.Static_T_Flag:
            self.T = np.array([self.Temps]).astype(np.float32)
        else:
            self.T = np.linspace(0.01, np.float32(2.0*self.MAT_PARAMS[24]), 31)
        self.BJ = drv.mem_alloc(self.T[0].nbytes)
        drv.memcpy_htod(self.GPU_MAT, self.MAT_PARAMS)
        drv.memcpy_htod(self.GRID_GPU, self.grid)
    
    def grid_reset(self):
        if self.FM_Flag:
            mc.FM_N(self.grid)
        else:
            mc.AFM_N(self.grid)
        self.grid *= self.spin
        drv.memcpy_htod(self.GRID_GPU, self.grid)

    def run_mc_dmi_66612(self, T):
        beta = np.array([1.0 / (T[0] * 8.6173e-2)],dtype=np.float32)
        drv.memcpy_htod(self.BJ,beta[0])
        for j in range(self.stability_runs):
            mc.METROPOLIS_MC_DM1_6_6_6_12(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[j*self.Blocks:(j+1)*self.Blocks-1], self.S1FULL[j*self.Blocks:(j+1)*self.Blocks-1], self.S2FULL[j*self.Blocks:(j+1)*self.Blocks-1], self.S3FULL[j*self.Blocks:(j+1)*self.Blocks-1], self.RLIST[j*self.Blocks:(j+1)*self.Blocks-1], self.GPU_TRANS, self.GPU_DMI_6, self.B_GPU, self.GSIZE, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
            mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(1,1,1), grid=(self.Blocks,1,1))
        drv.memcpy_dtoh(self.grid, self.GRID_GPU)
        return self.grid

    def run_mc_tc_4448(self, T):
        for i in tqdm(range(self.S_Wrap), desc="Stabilizing...", colour="blue"):
            mag_fluc =  np.zeros(self.calculation_runs)
            M = 0.0
            X = 0.0
            beta = np.array([1.0 / (T[0] * 8.6173e-2)],dtype=np.float32)
            drv.memcpy_htod(self.BJ,beta[0])
            for j in range(self.stability_runs):
                mc.METROPOLIS_MC_DM1_4_4_4_8(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[j*self.Blocks:(j+1)*self.Blocks-1], self.S1FULL[j*self.Blocks:(j+1)*self.Blocks-1], self.S2FULL[j*self.Blocks:(j+1)*self.Blocks-1], self.S3FULL[j*self.Blocks:(j+1)*self.Blocks-1], self.RLIST[j*self.Blocks:(j+1)*self.Blocks-1], self.GPU_TRANS, self.GPU_DMI_4, self.B_GPU, self.GSIZE, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
                mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(1,1,1), grid=(self.Blocks,1,1))
            drv.memcpy_dtoh(self.grid, self.GRID_GPU)
            np.save(f"{self.save_direcotry}/grid_{i:04d}", self.grid)
    
    def run_mc_tc_3636(self, T):
        Mt, Xt, Et = 0.0, 0.0, 0.0
        ct = 0
        M = np.zeros(self.S_Wrap)
        for i in tqdm(range(self.S_Wrap), desc=f"Stabilizing at {T:2f}", colour="blue"):
            mag_fluc =  np.zeros(self.stability_runs)
            beta = np.array([1.0 / (T * 8.6173e-2)],dtype=np.float32)
            drv.memcpy_htod(self.BJ, beta)
            for j in range(self.stability_runs):
                mc.METROPOLIS_MC_DM0_3_6_3_6(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[j*self.Blocks:(j+1)*self.Blocks-1], self.S1FULL[j*self.Blocks:(j+1)*self.Blocks-1], self.S2FULL[j*self.Blocks:(j+1)*self.Blocks-1], self.S3FULL[j*self.Blocks:(j+1)*self.Blocks-1], self.RLIST[j*self.Blocks:(j+1)*self.Blocks-1], self.GPU_TRANS, self.B_GPU, self.GSIZE, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
                mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(1,1,1), grid=(self.Blocks,1,1))
            drv.memcpy_dtoh(self.grid, self.GRID_GPU)
            self.grid = self.grid.reshape((self.size, self.size, 3))
            magx, magy, magz = self.grid[:,:,0], self.grid[:,:,1], self.grid[:,:,2]
            mag = np.array([np.sum(magx) , np.sum(magy) , np.sum(magz)])/(self.size**2)
            M[i] = np.abs(np.linalg.norm(mag))
        Mt, Xt = np.mean(M), np.std(M)/T
        print(f"Mean Magnetization at {T} = {Mt}")
        print(f"Mean Susceptibility at {T} = {Xt}")
        return Mt, Xt
    
    def run_mc_tc_66612(self, T):
        Mt, Xt = 0.0, 0.0
        ct = 0
        M, X = np.zeros(self.S_Wrap), np.zeros(self.S_Wrap)
        for i in tqdm(range(self.S_Wrap), desc=f"Stabilizing at {T}", colour="blue"):
            mag_fluc =  np.zeros(self.stability_runs)
            beta = np.array([1.0 / (T * 8.6173e-2)],dtype=np.float32)
            drv.memcpy_htod(self.BJ, beta)
            for j in range(self.stability_runs):
                mc.METROPOLIS_MC_DM0_6_6_6_12(self.GPU_MAT, self.GRID_GPU, self.BJ, self.NFULL[j*self.Blocks:(j+1)*self.Blocks-1], self.S1FULL[j*self.Blocks:(j+1)*self.Blocks-1], self.S2FULL[j*self.Blocks:(j+1)*self.Blocks-1], self.S3FULL[j*self.Blocks:(j+1)*self.Blocks-1], self.RLIST[j*self.Blocks:(j+1)*self.Blocks-1], self.GPU_TRANS, self.B_GPU, self.GSIZE, block=(self.Threads,1,1), grid=(self.Blocks,1,1))
                mc.GRID_COPY(self.GRID_GPU, self.GPU_TRANS, block=(1,1,1), grid=(self.Blocks,1,1))
            drv.memcpy_dtoh(self.grid, self.GRID_GPU)
            magx, magy, magz = self.grid[:,0], self.grid[:,1], self.grid[:,2]
            mag = np.array([np.sum(magx) , np.sum(magy) , np.sum(magz)])/(self.size**2)
            M[i] = np.sum(magz)/(self.spin*self.size**2)
            X[i] = np.abs(np.linalg.norm(mag)**2)
        Mt, Xt = np.mean(M), np.mean(X)
        print(f"Mean Magnetization at {T} = {Mt}")
        print(f"Mean Susceptibility at {T} = {Xt}")
        return Mt, Xt


    
    def dump_state():
        pass

class Analyze():
    def __init__(self, directory, reverse=False):
        self.flist = os.listdir(directory)
        self.flist = [file for file in self.flist if file.endswith(".npy")]
        self.flist.sort(reverse=reverse)
        self.directory = directory
        if not os.path.exists(self.directory+"/spin"):
            os.mkdir(self.directory+"/spin")
        if not os.path.exists(self.directory+"/quiver"):
            os.mkdir(self.directory+"/quiver")
        print(self.flist)

    def spin_view(self):
        ctr = 0
        for file in self.flist:
            print(file)
            grid = np.load(self.directory+"/"+file)
            shape = grid.shape
            print(shape)
            grid = grid.reshape((int(np.sqrt(shape[0]/3)), int(np.sqrt(shape[0]/3)), 3))
            spinx, spiny, spinz = grid[:,:,0], grid[:,:,1], grid[:,:,2]
            figure = plt.figure(dpi=400)
            plt.title("Spin Configuration at T = "+str(ctr))
            ax = figure.add_subplot(131)
            sns.heatmap(spinz, cbar=False, cmap="coolwarm", square=True, xticklabels=False, yticklabels=False)
            ax.set_xlabel("Z")
            ax = figure.add_subplot(132)
            sns.heatmap(spiny, cbar=False, cmap="coolwarm", square=True, xticklabels=False, yticklabels=False)
            ax.set_xlabel("Y")
            ax = figure.add_subplot(133)
            sns.heatmap(spinx, cbar=False, cmap="coolwarm", square=True, xticklabels=False, yticklabels=False)
            ax.set_xlabel("X")
            plt.savefig(self.directory+"/spin/spin_"+str(ctr)+".png")
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
            plt.savefig(self.directory+"/quiver/quiver_"+str(ctr)+".png")
            plt.close()
            ctr += 1