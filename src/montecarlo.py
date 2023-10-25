import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from numpy import random as rd
from pycuda.compiler import SourceModule

def PREC_N(n,box,size):
    NLIST = np.zeros(size)
    for i in range(4):
        for j in range(4):
            NLIST[i+4*j] = n + box*(i%4) + j*box*size
    return NLIST.astype(np.int32)


def FM_N(grid):
    for i in range(len(grid)):
        grid[i][2] = 1.0

def AFM_N(grid):
    for i in range(len(grid)):
        u,v = np.random.random(),np.random.random()
        theta, phi = 2*np.pi*u, np.arccos(2*v-1)
        grid[i][0] = np.sin(phi)*np.cos(theta)
        grid[i][1] = np.sin(phi)*np.sin(theta)
        grid[i][2] = np.cos(phi)

dev_hamiltonian = SourceModule("""
//cuda
#include <curand.h>
#include <cuda_runtime.h>
//Lattice Utility
__global__ void cp_grid(float_t* grid, float_t* tf)
{
    int idx = blockIdx.x;
    int threadID = idx;
    grid[int(tf[threadID*4])*3] = tf[threadID*4+1];
    grid[int(tf[threadID*4])*3+1] = tf[threadID*4+2];
    grid[int(tf[threadID*4])*3+2] = tf[threadID*4+3];
}
//Vector preprocessing
__global__ void uvec_processor(float* u, float* v, float* s1, float* s2, float* s3, float* spin)
{
    int idx = blockIdx.x;
    float phi = 2.0*3.14159265359*u[idx];
    float theta = acosf(2.0*v[idx] - 1.0);
    s1[idx] = spin[0]*sinf(theta)*cosf(phi);
    s2[idx] = spin[0]*sinf(theta)*sinf(phi);
    s3[idx] = spin[0]*cosf(theta);
}

__global__ void NList_processor(float* nlist, int* res, int* __SIZE)
{
    int Idx = blockIdx.x;
    res[Idx] = __float2uint_ru(__SIZE[0]*__SIZE[0]*nlist[Idx]);
}

//Neighbour mapping
__device__ void N1_6_6_6_12(int n, int size, int* NLIST)
{
    int row = n/size;
    int col = n%size;
    NLIST[0] = ((row-1)*size + col-1 + size*size)%(size*size);
    NLIST[1] = (row*size     + col+1 + size*size)%(size*size);
    NLIST[2] = ((row-1)*size + col   + size*size)%(size*size);
    NLIST[3] = ((row)*size   + col-1 + size*size)%(size*size);
    NLIST[4] = ((row+1)*size + col+1 + size*size)%(size*size);
    NLIST[5] = ((row+1)*size + col   + size*size)%(size*size);
}

__device__ void N2_6_6_6_12(int n, int size, int* NLIST)
{
    int row = n/size;
    int col = n%size;
    NLIST[0] = ((row-2)*size + col-1 + size*size)%(size*size);
    NLIST[1] = ((row-1)*size + col+1 + size*size)%(size*size);
    NLIST[2] = ((row+1)*size + col-1 + size*size)%(size*size);
    NLIST[3] = ((row-1)*size + col-2 + size*size)%(size*size);
    NLIST[4] = ((row+1)*size + col+1 + size*size)%(size*size);
    NLIST[5] = ((row+1)*size + col+2 + size*size)%(size*size);
}

__device__ void N3_6_6_6_12(int n, int size, int* NLIST)
{
    int row = n/size;
    int col = n%size;
    NLIST[0] = ((row-2)*size + col-2 + size*size)%(size*size);
    NLIST[1] = ((row)*size   + col+2 + size*size)%(size*size);
    NLIST[2] = ((row-2)*size + col   + size*size)%(size*size);
    NLIST[3] = ((row)*size   + col-2 + size*size)%(size*size);
    NLIST[4] = ((row+2)*size + col+2 + size*size)%(size*size);
    NLIST[5] = ((row+2)*size + col   + size*size)%(size*size);
}

__device__ void N4_6_6_6_12(int n, int size, int* NLIST)
{
    int row = n/size;
    int col = n%size;
    NLIST[0] = ((row-2)*size + col-3 + size*size)%(size*size);
    NLIST[1] = ((row-3)*size + col-1 + size*size)%(size*size);
    NLIST[2] = ((row-3)*size + col-2 + size*size)%(size*size);
    NLIST[3] = ((row+1)*size + col+3 + size*size)%(size*size);
    NLIST[4] = ((row-1)*size + col+2 + size*size)%(size*size);
    NLIST[5] = ((row-2)*size + col+1 + size*size)%(size*size);
    NLIST[6] = ((row+2)*size + col-1 + size*size)%(size*size);
    NLIST[7] = ((row+1)*size + col-2 + size*size)%(size*size);
    NLIST[8] = ((row-1)*size + col-3 + size*size)%(size*size);
    NLIST[9] = ((row+3)*size + col+2 + size*size)%(size*size);
    NLIST[10] = ((row+1)*size+ col+3 + size*size)%(size*size);
    NLIST[11] = ((row+2)*size+ col+3 + size*size)%(size*size);
}

__device__ void N1_4_4_4_8(int n, int size, int* NLIST)
{
    int row = n/size;
    int col = n%size;
    NLIST[0] = ((row-1)*size + col   + size*size)%(size*size);
    NLIST[1] = ((row  )*size + col-1 + size*size)%(size*size);
    NLIST[2] = ((row+1)*size + col   + size*size)%(size*size);
    NLIST[3] = ((row  )*size + col+1 + size*size)%(size*size);
}

__device__ void N2_4_4_4_8(int n, int size, int* NLIST)
{
    int row = n/size;
    int col = n%size;
    NLIST[0] = ((row-1)*size + col-1 + size*size)%(size*size);
    NLIST[1] = ((row-1)*size + col+1 + size*size)%(size*size);
    NLIST[2] = ((row+1)*size + col-1 + size*size)%(size*size);
    NLIST[3] = ((row+1)*size + col+1 + size*size)%(size*size);
}

__device__ void N3_4_4_4_8(int n, int size, int* NLIST)
{
    int row = n/size;
    int col = n%size;
    NLIST[0] = ((row-2)*size + col   + size*size)%(size*size);
    NLIST[1] = ((row  )*size + col-2 + size*size)%(size*size);
    NLIST[2] = ((row+2)*size + col   + size*size)%(size*size);
    NLIST[3] = ((row  )*size + col+2 + size*size)%(size*size);
}

__device__ void N4_4_4_4_8(int n, int size, int* NLIST)
{
    int row = n/size;
    int col = n%size;
    NLIST[0] = ((row-2)*size + col-1 + size*size)%(size*size);
    NLIST[1] = ((row-1)*size + col-2 + size*size)%(size*size);
    NLIST[2] = ((row-2)*size + col+1 + size*size)%(size*size);
    NLIST[3] = ((row-1)*size + col+2 + size*size)%(size*size);
    NLIST[4] = ((row+1)*size + col-2 + size*size)%(size*size);
    NLIST[5] = ((row+2)*size + col-1 + size*size)%(size*size);
    NLIST[6] = ((row+1)*size + col+2 + size*size)%(size*size);
    NLIST[7] = ((row+2)*size + col+1 + size*size)%(size*size);
}

__device__ void N1_2_4_2_4(int n, int size, int* NLIST)
{
    int row = n/size;
    int col = n%size;
    NLIST[0] = ((row  )*size + col-1 + size*size)%(size*size);
    NLIST[1] = ((row  )*size + col+1 + size*size)%(size*size);
}

__device__ void N2_2_4_2_4(int n, int size, int* NLIST)
{
    int row = n/size;
    int col = n%size;
    NLIST[0] = ((row+(size/2)*(row<(size/2)) + -(row>(size/2))*(1+(size/2)))*size + col -(row<(size/2)) + size*size)%(size*size);
    NLIST[1] = ((row+(size/2)*(row<(size/2)) + -(row>(size/2))*(1+(size/2)))*size + col +                 size*size)%(size*size);
}

//Hamiltonians
__device__ float_t hamiltonian_tc_2d_6_6_6_12_dm1(float_t* mat, float_t* sheet, int pti, float_t spinx, float_t spiny, float_t spinz, float* NVEC, float* b, int size)
{
    float H = 0.0;
    float d_plane = mat[25];

    int n1list[6];
    int n2list[6];
    int n3list[6];
    int n4list[12];

    N1_6_6_6_12(pti, size, n1list);
    N2_6_6_6_12(pti, size, n2list);
    N3_6_6_6_12(pti, size, n3list);
    N4_6_6_6_12(pti, size, n4list);

    for(int i=0; i<6; i++)
    {
        H += mat[1]*(spinx*sheet[n1list[i]*3] + spiny*sheet[n1list[i]*3+1] + spinz*sheet[n1list[i]*3+2]) + mat[5]*spinx*sheet[n1list[i]*3] + mat[6]*spiny*sheet[n1list[i]*3+1] + mat[7]*spinz*sheet[n1list[i]*3+2];
        float CRijx = (spinz*sheet[n1list[i]*3+1] - spiny*sheet[n1list[i]*3+2]) * NVEC[i*3];
        float CRijy = (spinx*sheet[n1list[i]*3+2] - spinz*sheet[n1list[i]*3]) * NVEC[i*3+1];
        float CRijz = (spiny*sheet[n1list[i]*3] - spinx*sheet[n1list[i]*3+1]) * NVEC[i*3+2];
        H += d_plane*(CRijx+CRijy+CRijz);
    }
    for(int i=0; i<6; i++)
    {
        H += mat[2]*(spinx*sheet[n2list[i]*3] + spiny*sheet[n2list[i]*3+1] + spinz*sheet[n2list[i]*3+2]) + mat[8]*spinx*sheet[n2list[i]*3] + mat[9]*spiny*sheet[n2list[i]*3+1] + mat[10]*spinz*sheet[n2list[i]*3+2];
    }
    for(int i=0; i<6; i++)
    {
        H += mat[3]*(spinx*sheet[n3list[i]*3] + spiny*sheet[n3list[i]*3+1] + spinz*sheet[n3list[i]*3+2]) + mat[11]*spinx*sheet[n3list[i]*3] + mat[12]*spiny*sheet[n3list[i]*3+1] + mat[13]*spinz*sheet[n3list[i]*3+2];
    }
    for(int i=0; i<12; i++)
    {
        H += mat[4]*(spinx*sheet[n4list[i]*3] + spiny*sheet[n4list[i]*3+1] + spinz*sheet[n4list[i]*3+2]) + mat[14]*spinx*sheet[n4list[i]*3] + mat[15]*spiny*sheet[n4list[i]*3+1] + mat[16]*spinz*sheet[n4list[i]*3+2];
    }
    H += mat[17]*spinx*spinx + mat[18]*spiny*spiny + mat[19]*spinz*spinz;
    H += b[0]*spinz;
    return -1.0*H;
}

__device__ float_t hamiltonian_tc_2d_6_6_6_12_dm0(float_t* mat, float_t* sheet, int pti, float_t spinx, float_t spiny, float_t spinz, float* n_vec, float* b, int size)
{
    float H = 0.0;

    int n1list[6];
    int n2list[6];
    int n3list[6];
    int n4list[12];
    
    N1_6_6_6_12(pti, size, n1list);
    N2_6_6_6_12(pti, size, n2list);
    N3_6_6_6_12(pti, size, n3list);
    N4_6_6_6_12(pti, size, n4list);

    for(int i=0; i<6; i++)
    {
        H += mat[1]*(spinx*sheet[n1list[i]*3] + spiny*sheet[n1list[i]*3+1] + spinz*sheet[n1list[i]*3+2]) + mat[5]*spinx*sheet[n1list[i]*3] + mat[6]*spiny*sheet[n1list[i]*3+1] + mat[7]*spinz*sheet[n1list[i]*3+2];
    }
    for(int i=0; i<6; i++)
    {
        H += mat[2]*(spinx*sheet[n2list[i]*3] + spiny*sheet[n2list[i]*3+1] + spinz*sheet[n2list[i]*3+2]) + mat[8]*spinx*sheet[n2list[i]*3] + mat[9]*spiny*sheet[n2list[i]*3+1] + mat[10]*spinz*sheet[n2list[i]*3+2];
    }
    for(int i=0; i<6; i++)
    {
        H += mat[3]*(spinx*sheet[n3list[i]*3] + spiny*sheet[n3list[i]*3+1] + spinz*sheet[n3list[i]*3+2]) + mat[11]*spinx*sheet[n3list[i]*3] + mat[12]*spiny*sheet[n3list[i]*3+1] + mat[13]*spinz*sheet[n3list[i]*3+2];
    }
    for(int i=0; i<12; i++)
    {
        H += mat[4]*(spinx*sheet[n4list[i]*3] + spiny*sheet[n4list[i]*3+1] + spinz*sheet[n4list[i]*3+2]) + mat[14]*spinx*sheet[n4list[i]*3] + mat[15]*spiny*sheet[n4list[i]*3+1] + mat[16]*spinz*sheet[n4list[i]*3+2];
    }
    H += mat[17]*spinx*spinx + mat[18]*spiny*spiny + mat[19]*spinz*spinz;
    H += b[0]*spinz;
    return -1.0*H;
}

//Monte Carlo
__global__ void metropolis_mc_dm1_6_6_6_12(float_t *mat, float_t *sheet, float_t *T, int* N, float_t* S1, float_t* S2,float_t* S3, float_t* R, float* tf, float* NVEC, float* B, int* size)
    {
        __shared__ float L0;
        __shared__ float L1;
        __shared__ float dE;
        int idx = blockIdx.x;
        int threadID = idx;
        int pt_thread = N[threadID];
        int tidx = threadIdx.x;
        if (tidx == 0)
        {  
            L0 = hamiltonian_tc_2d_6_6_6_12_dm1(mat, sheet, pt_thread, sheet[pt_thread*3], sheet[pt_thread*3+1], sheet[pt_thread*3+2], NVEC, B, size[0]);
        }
        if (tidx == 1)
        {
            L1 = hamiltonian_tc_2d_6_6_6_12_dm1(mat, sheet, pt_thread, S1[threadID], S2[threadID], S3[threadID], NVEC, B, size[0]);
        }
        __syncthreads();

        dE = L1 - L0;
        
        if (dE < 0)
        {
            tf[threadID*4] = pt_thread;
            tf[threadID*4+1] = S1[threadID];
            tf[threadID*4+2] = S2[threadID];
            tf[threadID*4+3] = S3[threadID];
        }
        else if (expf(-1.0*dE*T[0]) > R[threadID])
        {
            tf[threadID*4] = pt_thread;
            tf[threadID*4+1] = S1[threadID];
            tf[threadID*4+2] = S2[threadID];
            tf[threadID*4+3] = S3[threadID];
        }
    }
//!cuda
""")

GRID_COPY = dev_hamiltonian.get_function("cp_grid")

NPREC = dev_hamiltonian.get_function("NList_processor")
VPREC = dev_hamiltonian.get_function("uvec_processor")


#METROPOLIS_MC_DM0_6_6_6_12 = dev_hamiltonian.get_function("metropolis_mc_dm0_6_6_6_12")
METROPOLIS_MC_DM1_6_6_6_12 = dev_hamiltonian.get_function("metropolis_mc_dm1_6_6_6_12")

