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


def FM_N(grid, size):
    for i in range(size*size):
        grid[i*3+0] = 0.0
        grid[i*3+1] = 0.0
        grid[i*3+2] = 1.0

def AFM_N(grid, size):
    for i in range(size*size):
        u,v = np.random.random(),np.random.random()
        theta, phi = 2*np.pi*u, np.arccos(2*v-1)
        grid[i*3+0] = np.sin(phi)*np.cos(theta)
        grid[i*3+1] = np.sin(phi)*np.sin(theta)
        grid[i*3+2] = np.cos(phi)

def alt_FM_N(grid, s1, s2, size):
    for i in range(size*size):
        grid[i*4+0] = 0.0
        grid[i*4+1] = 0.0
        grid[i*4+2] = s1*(grid[i*4+3]==1) + s2*(grid[i*4+3]==2)

def alt_AFM_N(grid, s1, s2, size):
    for i in range(size*size):
        u,v = np.random.random(),np.random.random()
        theta, phi = 2*np.pi*u, np.arccos(2*v-1)
        grid[i*4+0] = np.sin(phi)*np.cos(theta)*(s1*(grid[i*4+3]) + s2*(1-grid[i*4+3]))
        grid[i*4+1] = np.sin(phi)*np.sin(theta)*(s1*(grid[i*4+3]) + s2*(1-grid[i*4+3]))
        grid[i*4+2] = np.cos(phi)*(s1*(grid[i*4+3]) + s2*(1-grid[i*4+3]))


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
__global__ void alt_cp_grid(float_t* grid, float_t* tf)
{
    int idx = blockIdx.x;
    int threadID = idx;
    grid[int(tf[threadID*4])*4] = tf[threadID*4+1];
    grid[int(tf[threadID*4])*4+1] = tf[threadID*4+2];
    grid[int(tf[threadID*4])*4+2] = tf[threadID*4+3];
}
//Vector preprocessing
__global__ void uvec_processor(float_t* u, float_t* v, float_t* s1, float_t* s2, float_t* s3, float_t* spin)
{
    int idx = blockIdx.x;
    float_t phi = 2.0*3.14159265359*u[idx];
    float_t theta = acosf(2.0*v[idx] - 1.0);
    s1[idx] = spin[0]*sinf(theta)*cosf(phi);
    s2[idx] = spin[0]*sinf(theta)*sinf(phi);
    s3[idx] = spin[0]*cosf(theta);
}

__global__ void NList_processor(float_t* nlist, int* res, int* __SIZE)
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

__device__ void N1_3_6_3_6(int n, int size, int* NLIST)
{
    int row = n/size;
    int col = n%size;
    NLIST[0] = ((row+1)*size + col+0 + size*size)%(size*size);
    NLIST[1] = ((row-1)*size + col+1 + size*size)%(size*size);
    NLIST[2] = ((row-0)*size + col-1 + size*size)%(size*size);
}

__device__ void N2_3_6_3_6(int n, int size, int* NLIST)
{
    int row = n/size;
    int col = n%size;
    NLIST[0] = ((row+1)*size + col+1 + size*size)%(size*size);
    NLIST[1] = ((row-1)*size + col+2 + size*size)%(size*size);
    NLIST[2] = ((row-2)*size + col+1 + size*size)%(size*size);
    NLIST[3] = ((row-1)*size + col-1 + size*size)%(size*size);
    NLIST[4] = ((row+1)*size + col-2 + size*size)%(size*size);
    NLIST[5] = ((row+2)*size + col-1 + size*size)%(size*size);
}

__device__ void N3_3_6_3_6(int n, int size, int* NLIST)
{
    int row = n/size;
    int col = n%size;
    NLIST[0] = ((row+2)*size + col   + size*size)%(size*size);
    NLIST[1] = ((row-2)*size + col+2 + size*size)%(size*size);
    NLIST[2] = ((row  )*size + col-2 + size*size)%(size*size);
}

__device__ void N4_3_6_3_6(int n, int size, int* NLIST)
{
    int row = n/size;
    int col = n%size;
    NLIST[0] = ((row-1)*size + col+3 + size*size)%(size*size);
    NLIST[1] = ((row-3)*size + col+1 + size*size)%(size*size);
    NLIST[2] = ((row-2)*size + col-1 + size*size)%(size*size);
    NLIST[3] = ((row+2)*size + col-3 + size*size)%(size*size);
    NLIST[4] = ((row-3)*size + col+2 + size*size)%(size*size);
    NLIST[5] = ((row+1)*size + col+2 + size*size)%(size*size);
}

//Hamiltonians
__device__ float_t hamiltonian_tc_2d_6_6_6_12_dm1(float_t* mat, float_t* sheet, int pti, float_t spinx, float_t spiny, float_t spinz, float_t* NVEC, float_t* b, int size)
{
    float_t H = 0.0;
    float_t d_plane = mat[25];

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
        float_t CRijx = (spinz*sheet[n1list[i]*3+1] - spiny*sheet[n1list[i]*3+2]) * NVEC[i*3];
        float_t CRijy = (spinx*sheet[n1list[i]*3+2] - spinz*sheet[n1list[i]*3]) * NVEC[i*3+1];
        float_t CRijz = (spiny*sheet[n1list[i]*3] - spinx*sheet[n1list[i]*3+1]) * NVEC[i*3+2];
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

__device__ float_t hamiltonian_tc_2d_6_6_6_12_dm0(float_t* mat, float_t* sheet, int pti, float_t spinx, float_t spiny, float_t spinz, float_t* b, int size)
{
    float_t H = 0.0;

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

__device__ float_t hamiltonian_tc_2d_4_4_4_8_dm1(float_t* mat, float_t* sheet, int pti, float_t spinx, float_t spiny, float_t spinz, float_t* NVEC, float_t* b, int size)
{
    float_t H = 0.0;
    float_t d_plane = mat[25];

    int n1list[6];
    int n2list[6];
    int n3list[6];
    int n4list[12];

    N1_4_4_4_8(pti, size, n1list);
    N2_4_4_4_8(pti, size, n2list);
    N3_4_4_4_8(pti, size, n3list);
    N4_4_4_4_8(pti, size, n4list);

    for(int i=0; i<6; i++)
    {
        H += mat[1]*(spinx*sheet[n1list[i]*3] + spiny*sheet[n1list[i]*3+1] + spinz*sheet[n1list[i]*3+2]) + mat[5]*spinx*sheet[n1list[i]*3] + mat[6]*spiny*sheet[n1list[i]*3+1] + mat[7]*spinz*sheet[n1list[i]*3+2];
        float_t CRijx = (spinz*sheet[n1list[i]*3+1] - spiny*sheet[n1list[i]*3+2]) * NVEC[i*3];
        float_t CRijy = (spinx*sheet[n1list[i]*3+2] - spinz*sheet[n1list[i]*3]) * NVEC[i*3+1];
        float_t CRijz = (spiny*sheet[n1list[i]*3] - spinx*sheet[n1list[i]*3+1]) * NVEC[i*3+2];
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

__device__ float_t hamiltonian_tc_2d_4_4_4_8_dm0(float_t* mat, float_t* sheet, int pti, float_t spinx, float_t spiny, float_t spinz, float_t* n_vec, float_t* b, int size)
{
    float_t H = 0.0;

    int n1list[6];
    int n2list[6];
    int n3list[6];
    int n4list[12];
    
    N1_4_4_4_8(pti, size, n1list);
    N2_4_4_4_8(pti, size, n2list);
    N3_4_4_4_8(pti, size, n3list);
    N4_4_4_4_8(pti, size, n4list);

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

__device__ float_t hamiltonian_tc_2d_3_6_3_6_dm0(float_t* mat, float_t* sheet, int pti, float_t spinx, float_t spiny, float_t spinz, float_t* b, int size)
{
    float_t H = 0.0;

    int n1list[3];
    int n2list[6];
    int n3list[3];
    int n4list[6];

    N1_3_6_3_6(pti, size, n1list);
    N2_3_6_3_6(pti, size, n2list);
    N3_3_6_3_6(pti, size, n3list);
    N4_3_6_3_6(pti, size, n4list);

    for(int i=0; i<3; i++)
    {
        H += mat[1]*(spinx*sheet[n1list[i]*3] + spiny*sheet[n1list[i]*3+1] + spinz*sheet[n1list[i]*3+2]) + mat[5]*spinx*sheet[n1list[i]*3] + mat[6]*spiny*sheet[n1list[i]*3+1] + mat[7]*spinz*sheet[n1list[i]*3+2];
    }
    for(int i=0; i<6; i++)
    {
        H += mat[2]*(spinx*sheet[n2list[i]*3] + spiny*sheet[n2list[i]*3+1] + spinz*sheet[n2list[i]*3+2]) + mat[8]*spinx*sheet[n2list[i]*3] + mat[9]*spiny*sheet[n2list[i]*3+1] + mat[10]*spinz*sheet[n2list[i]*3+2];
    }
    for(int i=0; i<3; i++)
    {
        H += mat[3]*(spinx*sheet[n3list[i]*3] + spiny*sheet[n3list[i]*3+1] + spinz*sheet[n3list[i]*3+2]) + mat[11]*spinx*sheet[n3list[i]*3] + mat[12]*spiny*sheet[n3list[i]*3+1] + mat[13]*spinz*sheet[n3list[i]*3+2];
    }
    for(int i=0; i<6; i++)
    {
        H += mat[4]*(spinx*sheet[n4list[i]*3] + spiny*sheet[n4list[i]*3+1] + spinz*sheet[n4list[i]*3+2]) + mat[14]*spinx*sheet[n4list[i]*3] + mat[15]*spiny*sheet[n4list[i]*3+1] + mat[16]*spinz*sheet[n4list[i]*3+2];
    }
    H += mat[17]*spinx*spinx + mat[18]*spiny*spiny + mat[19]*spinz*spinz;
    H += b[0]*spinz;
    return -1.0*H;
}

__device__ float_t hamiltonian_tc_2d_3_6_3_6_dm1(float_t* mat, float_t* sheet, int pti, float_t spinx, float_t spiny, float_t spinz, float_t* b, int size)
{
    float_t H = 0.0;

    int n1list[3];
    int n2list[6];
    int n3list[3];
    int n4list[6];

    N1_3_6_3_6(pti, size, n1list);
    N2_3_6_3_6(pti, size, n2list);
    N3_3_6_3_6(pti, size, n3list);
    N4_3_6_3_6(pti, size, n4list);

    for(int i=0; i<3; i++)
    {
        H += mat[1]*(spinx*sheet[n1list[i]*3] + spiny*sheet[n1list[i]*3+1] + spinz*sheet[n1list[i]*3+2]) + mat[5]*spinx*sheet[n1list[i]*3] + mat[6]*spiny*sheet[n1list[i]*3+1] + mat[7]*spinz*sheet[n1list[i]*3+2];
        H += 0.21*(spinx*sheet[n1list[i]*3] + spiny*sheet[n1list[i]*3+1] + spinz*sheet[n1list[i]*3+2])*(spinx*sheet[n1list[i]*3] + spiny*sheet[n1list[i]*3+1] + spinz*sheet[n1list[i]*3+2]);
    }
    for(int i=0; i<6; i++)
    {
        H += mat[2]*(spinx*sheet[n2list[i]*3] + spiny*sheet[n2list[i]*3+1] + spinz*sheet[n2list[i]*3+2]) + mat[8]*spinx*sheet[n2list[i]*3] + mat[9]*spiny*sheet[n2list[i]*3+1] + mat[10]*spinz*sheet[n2list[i]*3+2];
    }
    for(int i=0; i<3; i++)
    {
        H += mat[3]*(spinx*sheet[n3list[i]*3] + spiny*sheet[n3list[i]*3+1] + spinz*sheet[n3list[i]*3+2]) + mat[11]*spinx*sheet[n3list[i]*3] + mat[12]*spiny*sheet[n3list[i]*3+1] + mat[13]*spinz*sheet[n3list[i]*3+2];
    }
    for(int i=0; i<6; i++)
    {
        H += mat[4]*(spinx*sheet[n4list[i]*3] + spiny*sheet[n4list[i]*3+1] + spinz*sheet[n4list[i]*3+2]) + mat[14]*spinx*sheet[n4list[i]*3] + mat[15]*spiny*sheet[n4list[i]*3+1] + mat[16]*spinz*sheet[n4list[i]*3+2];
    }
    H += mat[17]*spinx*spinx + mat[18]*spiny*spiny + mat[19]*spinz*spinz;
    H += b[0]*spinz;
    return -1.0*H;
}

__device__ float_t alt_hamiltonian_MnCr_3_6_3_6_dm1(float_t* sheet, int pti, float_t spinx, float_t spiny, float_t spinz, float_t* b, int size)
{
    float_t H = 0.0;
    int n1list[3];
    int n2list[6];
    int n3list[3];
    
    float_t J1_Cr_Mn[3][3] = {{1.15,0.0,0.0},
                        {0.0,0.37,0.40},
                        {0.0,0.40,0.51}};
    
    float_t D_Cr_Mn[3] = {0.21,0.0,0.0};

    float_t J2_Cr_Cr[3][3] = {{0.49,0.0,0.0},
                        {0.0,0.44,0.06},
                        {0.0,0.06,0.42}};

    float_t D_Cr_Cr[3] = {0.0,0.06,-0.19};

    float_t J3_Mn_Mn[3][3] = {{-0.12,0.0,0.0},
                            {0.0,0.93,-0.28},
                            {0.0,-0.28,0.24}};
    
    float_t D_Mn_Mn[3] = {0.0,0.27,-0.19};

    float_t J4_Mn_Cr[3][3] = {{0.19,0.0,0.0},
                            {0.0,0.14,0.09},
                            {0.0,0.9,0.22}};
    
    float_t D_Mn_Cr[3] = {-0.03,0.0,0.0};

    float_t Azz_Cr = 0.28;
    float_t Azz_Mn = 0.32;
    
    N1_3_6_3_6(pti, size, n1list);
    N2_3_6_3_6(pti, size, n2list);
    N3_3_6_3_6(pti, size, n3list);
    if(sheet[pti*4+3]-1) //Cr
    {
        for(int i=0; i<3; i++)
        {
            int pt = n1list[i];
            H += spinx*sheet[pt*4]*J1_Cr_Mn[0][0] + spiny*sheet[pt*4+1]*J1_Cr_Mn[0][1] + spinz*sheet[pt*4+2]*J1_Cr_Mn[0][2];
            H += D_Cr_Mn[0]*(spiny*sheet[pt*4+2] - spinz*sheet[pt*4+1]);
            H += D_Cr_Mn[1]*(spinz*sheet[pt*4] - spinx*sheet[pt*4+2]);
            H += D_Cr_Mn[2]*(spinx*sheet[pt*4+1] - spiny*sheet[pt*4]);
        }

        for(int i=0; i<6; i++)
        {
            int pt = n2list[i];
            H += spinx*sheet[pt*4]*J2_Cr_Cr[0][0] + spiny*sheet[pt*4+1]*J2_Cr_Cr[0][1] + spinz*sheet[pt*4+2]*J2_Cr_Cr[0][2];
            H += D_Cr_Cr[0]*(spiny*sheet[pt*4+2] - spinz*sheet[pt*4+1]);
            H += D_Cr_Cr[1]*(spinz*sheet[pt*4] - spinx*sheet[pt*4+2]);
            H += D_Cr_Cr[2]*(spinx*sheet[pt*4+1] - spiny*sheet[pt*4]);
        }

        for(int i=0; i<3; i++)
        {
            int pt = n3list[i];
            H += spinx*sheet[pt*4]*J4_Mn_Cr[0][0] + spiny*sheet[pt*4+1]*J4_Mn_Cr[0][1] + spinz*sheet[pt*4+2]*J4_Mn_Cr[0][2];
            H += D_Mn_Cr[0]*(spiny*sheet[pt*4+2] - spinz*sheet[pt*4+1]);
            H += D_Mn_Cr[1]*(spinz*sheet[pt*4] - spinx*sheet[pt*4+2]);
            H += D_Mn_Cr[2]*(spinx*sheet[pt*4+1] - spiny*sheet[pt*4]);
        }

        H += Azz_Cr*spinz*spinz;
    }
    else //Mn   
    {
        for(int i=0; i<3; i++)
        {
            int pt = n1list[i];
            H += spinx*sheet[pt*4]*J1_Cr_Mn[0][0] + spiny*sheet[pt*4+1]*J1_Cr_Mn[0][1] + spinz*sheet[pt*4+2]*J1_Cr_Mn[0][2];
            H += D_Cr_Mn[0]*(spiny*sheet[pt*4+2] - spinz*sheet[pt*4+1]);
            H += D_Cr_Mn[1]*(spinz*sheet[pt*4] - spinx*sheet[pt*4+2]);
            H += D_Cr_Mn[2]*(spinx*sheet[pt*4+1] - spiny*sheet[pt*4]);
        }

        for(int i=0; i<6; i++)
        {
            int pt = n2list[i];
            H += spinx*sheet[pt*4]*J3_Mn_Mn[0][0] + spiny*sheet[pt*4+1]*J3_Mn_Mn[0][1] + spinz*sheet[pt*4+2]*J3_Mn_Mn[0][2];
            H += D_Mn_Mn[0]*(spiny*sheet[pt*4+2] - spinz*sheet[pt*4+1]);
            H += D_Mn_Mn[1]*(spinz*sheet[pt*4] - spinx*sheet[pt*4+2]);
            H += D_Mn_Mn[2]*(spinx*sheet[pt*4+1] - spiny*sheet[pt*4]);
        }

        for(int i=0; i<3; i++)
        {
            int pt = n3list[i];
            H += spinx*sheet[pt*4]*J4_Mn_Cr[0][0] + spiny*sheet[pt*4+1]*J4_Mn_Cr[0][1] + spinz*sheet[pt*4+2]*J4_Mn_Cr[0][2];
            H += D_Mn_Cr[0]*(spiny*sheet[pt*4+2] - spinz*sheet[pt*4+1]);
            H += D_Mn_Cr[1]*(spinz*sheet[pt*4] - spinx*sheet[pt*4+2]);
            H += D_Mn_Cr[2]*(spinx*sheet[pt*4+1] - spiny*sheet[pt*4]);
        }

        H += Azz_Mn*spinz*spinz;
    }
    H += b[0]*spinz;
    return -1.0*H;
}

__device__ float_t alt_hamiltonian_MnCr_3_6_3_6_dm0(float_t* sheet, int pti, float_t spinx, float_t spiny, float_t spinz, float_t* b, int size)
{
    float_t H = 0.0;
    int n1list[3];
    int n2list[6];
    int n3list[3];
    //MnCrI6 System---------------------------------------------------
    
    float_t J1_Cr_Mn[3][3] = {{1.15,0.0,0.0},
                        {0.0,0.37,0.40},
                        {0.0,0.40,0.51}};
    
    float_t D_Cr_Mn[3] = {0.21,0.0,0.0};

    float_t J2_Cr_Cr[3][3] = {{0.49,0.0,0.0},
                        {0.0,0.44,0.06},
                        {0.0,0.06,0.42}};

    float_t D_Cr_Cr[3] = {0.0,0.06,-0.19};

    float_t J3_Mn_Mn[3][3] = {{-0.12,0.0,0.0},
                            {0.0,0.93,-0.28},
                            {0.0,-0.28,0.24}};
    
    float_t D_Mn_Mn[3] = {0.0,0.27,-0.19};

    float_t J4_Mn_Cr[3][3] = {{0.19,0.0,0.0},
                            {0.0,0.14,0.09},
                            {0.0,0.9,0.22}};
    
    float_t D_Mn_Cr[3] = {-0.03,0.0,0.0};

    float_t Azz_Cr = 0.28;
    float_t Azz_Mn = 0.32;
    

    //Test CrI3-CrI3 System-------------------------------------------
    // float_t J1_Cr_Mn[3][3] = {{1.15,0.0,0.0},
    //                     {0.0,0.37,0.40},
    //                     {0.0,0.40,0.51}};
    
    // float_t D_Cr_Mn[3] = {0.21,0.0,0.0};

    // float_t J2_Cr_Cr[3][3] = {{0.49,0.0,0.0},
    //                     {0.0,0.44,0.06},
    //                     {0.0,0.06,0.42}};

    // float_t D_Cr_Cr[3] = {0.0,0.06,-0.19};

    // float_t J3_Mn_Mn[3][3] = {{-0.12,0.0,0.0},
    //                         {0.0,0.93,-0.28},
    //                         {0.0,-0.28,0.24}};
    
    // float_t D_Mn_Mn[3] = {0.0,0.27,-0.19};

    // float_t J4_Mn_Cr[3][3] = {{0.19,0.0,0.0},
    //                         {0.0,0.14,0.09},
    //                         {0.0,0.9,0.22}};
    
    // float_t D_Mn_Cr[3] = {-0.03,0.0,0.0};

    // float_t Azz_Cr = 0.28;
    // float_t Azz_Mn = 0.32;

    N1_3_6_3_6(pti, size, n1list);
    N2_3_6_3_6(pti, size, n2list);
    N3_3_6_3_6(pti, size, n3list);
    if(sheet[pti*4+3]-1) //Cr
    {
        for(int i=0; i<3; i++)
        {
            int pt = n1list[i];
            H += spinx*sheet[pt*4]*J1_Cr_Mn[0][0] + spiny*sheet[pt*4+1]*J1_Cr_Mn[0][1] + spinz*sheet[pt*4+2]*J1_Cr_Mn[0][2];
        }

        for(int i=0; i<6; i++)
        {
            int pt = n2list[i];
            H += spinx*sheet[pt*4]*J2_Cr_Cr[0][0] + spiny*sheet[pt*4+1]*J2_Cr_Cr[0][1] + spinz*sheet[pt*4+2]*J2_Cr_Cr[0][2];
        }

        for(int i=0; i<3; i++)
        {
            int pt = n3list[i];
            H += spinx*sheet[pt*4]*J4_Mn_Cr[0][0] + spiny*sheet[pt*4+1]*J4_Mn_Cr[0][1] + spinz*sheet[pt*4+2]*J4_Mn_Cr[0][2];
        }

        H += Azz_Cr*spinz*spinz;
    }
    else //Mn   
    {
        for(int i=0; i<3; i++)
        {
            int pt = n1list[i];
            H += spinx*sheet[pt*4]*J1_Cr_Mn[0][0] + spiny*sheet[pt*4+1]*J1_Cr_Mn[0][1] + spinz*sheet[pt*4+2]*J1_Cr_Mn[0][2];
        }

        for(int i=0; i<6; i++)
        {
            int pt = n2list[i];
            H += spinx*sheet[pt*4]*J3_Mn_Mn[0][0] + spiny*sheet[pt*4+1]*J3_Mn_Mn[0][1] + spinz*sheet[pt*4+2]*J3_Mn_Mn[0][2];
        }

        for(int i=0; i<3; i++)
        {
            int pt = n3list[i];
            H += spinx*sheet[pt*4]*J4_Mn_Cr[0][0] + spiny*sheet[pt*4+1]*J4_Mn_Cr[0][1] + spinz*sheet[pt*4+2]*J4_Mn_Cr[0][2];
        }

        H += Azz_Mn*spinz*spinz;
    }
    H += b[0]*spinz;
    return -1.0*H;
}
//Monte Carlo
__global__ void metropolis_mc_dm1_6_6_6_12(float_t *mat, float_t *sheet, float_t *T, int* N, float_t* S1, float_t* S2,float_t* S3, float_t* R, float_t* tf, float_t* NVEC, float_t* B, int* size)
{
    __shared__ float_t L0;
    __shared__ float_t L1;
    __shared__ float_t dE;
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

__global__ void metropolis_mc_dm0_6_6_6_12(float_t *mat, float_t *sheet, float_t *T, int* N, float_t* S1, float_t* S2,float_t* S3, float_t* R, float_t* tf, float_t* B, int* size)
{
    __shared__ float_t L0;
    __shared__ float_t L1;
    __shared__ float_t dE;
    int idx = blockIdx.x;
    int threadID = idx;
    int pt_thread = N[threadID];
    int tidx = threadIdx.x;
    if (tidx == 0)
    {  
        L0 = hamiltonian_tc_2d_6_6_6_12_dm0(mat, sheet, pt_thread, sheet[pt_thread*3], sheet[pt_thread*3+1], sheet[pt_thread*3+2],  B, size[0]);
    }
    if (tidx == 1)
    {
        L1 = hamiltonian_tc_2d_6_6_6_12_dm0(mat, sheet, pt_thread, S1[threadID], S2[threadID], S3[threadID], B, size[0]);
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

__global__ void metropolis_mc_dm0_3_6_3_6(float_t *mat, float_t *sheet, float_t *T, int* N, float_t* S1, float_t* S2,float_t* S3, float_t* R, float_t* tf, float_t* B, int* size)
{
    __shared__ float_t L0;
    __shared__ float_t L1;
    __shared__ float_t dE;
    int idx = blockIdx.x;
    int threadID = idx;
    int pt_thread = N[threadID];
    int tidx = threadIdx.x;
    if (tidx == 0)
    {  
        L0 = hamiltonian_tc_2d_3_6_3_6_dm0(mat, sheet, pt_thread, sheet[pt_thread*3], sheet[pt_thread*3+1], sheet[pt_thread*3+2],  B, size[0]);
    }
    if (tidx == 1)
    {
        L1 = hamiltonian_tc_2d_3_6_3_6_dm0(mat, sheet, pt_thread, S1[threadID], S2[threadID], S3[threadID], B, size[0]);
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

__global__ void metropolis_mc_dm1_3_6_3_6(float_t *mat, float_t *sheet, float_t *T, int* N, float_t* S1, float_t* S2,float_t* S3, float_t* R, float_t* tf, float_t* B, int* size)
{
    __shared__ float_t L0;
    __shared__ float_t L1;
    __shared__ float_t dE;
    int idx = blockIdx.x;
    int threadID = idx;
    int pt_thread = N[threadID];
    int tidx = threadIdx.x;
    if (tidx == 0)
    {  
        L0 = hamiltonian_tc_2d_3_6_3_6_dm1(mat, sheet, pt_thread, sheet[pt_thread*3], sheet[pt_thread*3+1], sheet[pt_thread*3+2],  B, size[0]);
    }
    if (tidx == 1)
    {
        L1 = hamiltonian_tc_2d_3_6_3_6_dm1(mat, sheet, pt_thread, S1[threadID], S2[threadID], S3[threadID], B, size[0]);
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

__global__ void metropolis_mc_dm1_4_4_4_8(float_t *mat, float_t *sheet, float_t *T, int* N, float_t* S1, float_t* S2,float_t* S3, float_t* R, float_t* tf, float_t* NVEC, float_t* B, int* size)
{
    __shared__ float_t L0;
    __shared__ float_t L1;
    __shared__ float_t dE;
    int idx = blockIdx.x;
    int threadID = idx;
    int pt_thread = N[threadID];
    int tidx = threadIdx.x;
    if (tidx == 0)
    {  
        L0 = hamiltonian_tc_2d_4_4_4_8_dm1(mat, sheet, pt_thread, sheet[pt_thread*3], sheet[pt_thread*3+1], sheet[pt_thread*3+2], NVEC, B, size[0]);
    }
    if (tidx == 1)
    {
        L1 = hamiltonian_tc_2d_4_4_4_8_dm1(mat, sheet, pt_thread, S1[threadID], S2[threadID], S3[threadID], NVEC, B, size[0]);
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

__global__ void metropolis_mc_dm0_4_4_4_8(float_t *mat, float_t *sheet, float_t *T, int* N, float_t* S1, float_t* S2,float_t* S3, float_t* R, float_t* tf, float_t* NVEC, float_t* B, int* size)
{
    __shared__ float_t L0;
    __shared__ float_t L1;
    __shared__ float_t dE;
    int idx = blockIdx.x;
    int threadID = idx;
    int pt_thread = N[threadID];
    int tidx = threadIdx.x;
    if (tidx == 0)
    {  
        L0 = hamiltonian_tc_2d_4_4_4_8_dm0(mat, sheet, pt_thread, sheet[pt_thread*3], sheet[pt_thread*3+1], sheet[pt_thread*3+2], NVEC, B, size[0]);
    }
    if (tidx == 1)
    {
        L1 = hamiltonian_tc_2d_4_4_4_8_dm0(mat, sheet, pt_thread, S1[threadID], S2[threadID], S3[threadID], NVEC, B, size[0]);
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

__global__ void alt_metropolis(float_t *sheet, float_t *T, int* N, float_t* S1, float_t* S2,float_t* S3, float_t* R, float_t* tf, float_t* B, int* size, float_t* spin1, float_t* spin2)
{
    __shared__ float_t L0;
    __shared__ float_t L1;
    __shared__ float_t dE;
    int idx = blockIdx.x;
    int threadID = idx;
    int pt_thread = N[threadID];
    int tidx = threadIdx.x;
    if (tidx == 0)
    {
        L0 = alt_hamiltonian_MnCr_3_6_3_6_dm1(sheet, pt_thread, sheet[pt_thread*4], sheet[pt_thread*4+1], sheet[pt_thread*4+2], B, size[0]);
    }
    if (tidx == 1)
    {
        L1 = alt_hamiltonian_MnCr_3_6_3_6_dm1(sheet, pt_thread, S1[threadID], S2[threadID], S3[threadID], B, size[0]);
    }

    __syncthreads();

    dE = L1 - L0;
    float_t mult = 1.0;
    if ((sheet[pt_thread*4+3] - 1.0)*(sheet[pt_thread*4+3] - 1.0) < 0.0001)
        mult = spin1[0];
    else
        mult = spin2[0];
    if (dE < 0)
    {
        tf[threadID*4] = pt_thread;
        tf[threadID*4+1] = mult*S1[threadID];
        tf[threadID*4+2] = mult*S2[threadID];
        tf[threadID*4+3] = mult*S3[threadID];
    }
    else if (expf(-1.0*dE*T[0]) > R[threadID])
    {
        tf[threadID*4] = pt_thread;
        tf[threadID*4+1] = mult*S1[threadID];
        tf[threadID*4+2] = mult*S2[threadID];
        tf[threadID*4+3] = mult*S3[threadID];
    }
}

__global__ void alt_metropolis_TC(float_t *sheet, float_t *T, int* N, float_t* S1, float_t* S2,float_t* S3, float_t* R, float_t* tf, float_t* B, int* size, float_t* spin1, float_t* spin2)
{
    __shared__ float_t L0;
    __shared__ float_t L1;
    __shared__ float_t dE;
    int idx = blockIdx.x;
    int threadID = idx;
    int pt_thread = N[threadID];
    int tidx = threadIdx.x;
    if (tidx == 0)
    {
        L0 = alt_hamiltonian_MnCr_3_6_3_6_dm0(sheet, pt_thread, sheet[pt_thread*4], sheet[pt_thread*4+1], sheet[pt_thread*4+2], B, size[0]);
    }
    if (tidx == 1)
    {
        L1 = alt_hamiltonian_MnCr_3_6_3_6_dm0(sheet, pt_thread, S1[threadID], S2[threadID], S3[threadID], B, size[0]);
    }

    __syncthreads();

    dE = L1 - L0;
    float_t mult = 1.0;
    if ((sheet[pt_thread*4+3] - 1.0)*(sheet[pt_thread*4+3] - 1.0) < 0.0001)
        mult = spin2[0];
    else
        mult = spin1[0];
    if (dE < 0)
    {
        tf[threadID*4] = pt_thread;
        tf[threadID*4+1] = mult*S1[threadID];
        tf[threadID*4+2] = mult*S2[threadID];
        tf[threadID*4+3] = mult*S3[threadID];
    }
    else if (expf(-1.0*dE*T[0]) > R[threadID])
    {
        tf[threadID*4] = pt_thread;
        tf[threadID*4+1] = mult*S1[threadID];
        tf[threadID*4+2] = mult*S2[threadID];
        tf[threadID*4+3] = mult*S3[threadID];
    }
}
//Double Material Study
__device__ int alt_populate(float_t* sheet, int pt, int size)
{
    int n1list[3];
    N1_3_6_3_6(pt, size, n1list);
    if((sheet[pt*4+3] - 1)*(sheet[pt*4+3] - 1) <= 0.0001)
    {
        for(int j = 0; j < 3; j++)
        {
            if((sheet[n1list[j]*4+3] - 2)*(sheet[n1list[j]*4+3] - 2) <= 0.0001)
            {
                return 1;
            }
            else    
            {    
                sheet[n1list[j]*4+3] = 2;
                int s = alt_populate(sheet, n1list[j], size);
                return s;
            }
        }
    } 
    else if((sheet[pt*4+3] - 2)*(sheet[pt*4+3] - 2) <= 0.0001)
    {
        for(int j = 0; j < 3; j++)
        {
            if((sheet[n1list[j]*4+3] - 1)*(sheet[n1list[j]*4+3] - 1) <= 0.0001)
            {
                return 1;
            }
            else    
            {
                sheet[n1list[j]*4+3] = 1;
                int s = alt_populate(sheet, n1list[j], size);
                return s;
            }
        }
    }
}

__global__ void alt_grid(int* size, float_t* sheet, int* debug, float_t* spins)
{
    sheet[3] = 1;
    debug[0] = alt_populate(sheet, 0, size[0]);
    for(int i=0;i<size[0]*size[0];i++)
    {
        if(sheet[i*4+3]==1)
        {
            sheet[i*4] *= spins[0];
            sheet[i*4+1] *= spins[0];
            sheet[i*4+2] *= spins[0];
        }
        else if(sheet[i*4+3]==2)
        {
            sheet[i*4] *= spins[1];
            sheet[i*4+1] *= spins[1];
            sheet[i*4+2] *= spins[1];
        }
    }
}

//Energy Calculators
__global__ void energy_metropolis_mc_dm0_3_6_3_6(float_t *mat, float_t *sheet, float_t *T, int* N, float_t* S1, float_t* S2,float_t* S3, float_t* R, float_t* tf, float_t* B, int* size, float_t* en)
{
    __shared__ float_t L0;
    __shared__ float_t L1;
    __shared__ float_t dE;
    int idx = blockIdx.x;
    int threadID = idx;
    int pt_thread = N[threadID];
    int tidx = threadIdx.x;
    if (tidx == 0)
    {  
        L0 = hamiltonian_tc_2d_3_6_3_6_dm0(mat, sheet, pt_thread, sheet[pt_thread*3], sheet[pt_thread*3+1], sheet[pt_thread*3+2],  B, size[0]);
    }
    if (tidx == 1)
    {
        L1 = hamiltonian_tc_2d_3_6_3_6_dm0(mat, sheet, pt_thread, S1[threadID], S2[threadID], S3[threadID], B, size[0]);
    }
    __syncthreads();

    dE = L1 - L0;

    if (dE < 0)
    {
        tf[threadID*4] = pt_thread;
        tf[threadID*4+1] = S1[threadID];
        tf[threadID*4+2] = S2[threadID];
        tf[threadID*4+3] = S3[threadID];
        en[0] += dE;
    }
    else if (expf(-1.0*dE*T[0]) > R[threadID])
    {
        tf[threadID*4] = pt_thread;
        tf[threadID*4+1] = S1[threadID];
        tf[threadID*4+2] = S2[threadID];
        tf[threadID*4+3] = S3[threadID];
        en[0] += dE;
    }
}

__global__ void encalc_3636(float_t* mat, float_t* sheet, float_t* B, int* N, int* size, float_t* en)
{
    __shared__ float_t E;
    int idx = blockIdx.x;
    int threadID = idx;
    int pt_thread = threadID;
    en[idx] = -1.0*hamiltonian_tc_2d_3_6_3_6_dm0(mat, sheet, pt_thread, sheet[pt_thread*3], sheet[pt_thread*3+1], sheet[pt_thread*3+2], B, size[0]);
}

__global__ void encalc_3636_2(float_t* mat, float_t* sheet, float_t* B, int* N, int* size, float_t* en)
{
    __shared__ float_t E;
    int idx = blockIdx.x;
    int threadID = idx;
    int pt_thread = threadID;
    en[idx] = -1.0*hamiltonian_tc_2d_3_6_3_6_dm1(mat, sheet, pt_thread, sheet[pt_thread*3], sheet[pt_thread*3+1], sheet[pt_thread*3+2], B, size[0]);
}

//!cuda
""")

GRID_COPY = dev_hamiltonian.get_function("cp_grid")
ALT_GRID_COPY = dev_hamiltonian.get_function("alt_cp_grid")

NPREC = dev_hamiltonian.get_function("NList_processor")
VPREC = dev_hamiltonian.get_function("uvec_processor")
ALT_GRID = dev_hamiltonian.get_function("alt_grid")


METROPOLIS_MC_DM0_3_6_3_6 = dev_hamiltonian.get_function("metropolis_mc_dm0_3_6_3_6")
METROPOLIS_MC_DM1_3_6_3_6 = dev_hamiltonian.get_function("metropolis_mc_dm1_3_6_3_6")

METROPOLIS_MC_DM1_6_6_6_12 = dev_hamiltonian.get_function("metropolis_mc_dm1_6_6_6_12")
METROPOLIS_MC_DM0_6_6_6_12 = dev_hamiltonian.get_function("metropolis_mc_dm0_6_6_6_12")

METROPOLIS_MC_DM1_4_4_4_8  = dev_hamiltonian.get_function("metropolis_mc_dm1_4_4_4_8")
METROPOLIS_MC_DM0_4_4_4_8  = dev_hamiltonian.get_function("metropolis_mc_dm0_4_4_4_8")

METROPOLIS_ALT_MnCr_3_6_3_6 = dev_hamiltonian.get_function("alt_metropolis")
METROPOLIS_MALT_MnCr_3_6_3_6 = dev_hamiltonian.get_function("alt_metropolis_TC")

EN_CALC_3_6_3_6 = dev_hamiltonian.get_function("encalc_3636")
EN_CALC_3_6_3_6_2 = dev_hamiltonian.get_function("encalc_3636_2")