import numpy as np
import cupy as cp
import json
import os
import sys
sys.path.append("../utilities/")
import Material_Reader as rm
import datetime
import tqdm as tqdm

kb = 8.617333262145e-2 # meV/K

# Definitions of neighbours
def n1_6_6_6_12(pt, size):
    nbs = np.zeros(6).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row-1)*size + col-1 + size*size)%(size*size)
    nbs[1] = (row*size     + col+1 + size*size)%(size*size)
    nbs[2] = ((row-1)*size + col   + size*size)%(size*size)
    nbs[3] = ((row)*size   + col-1 + size*size)%(size*size)
    nbs[4] = ((row+1)*size + col+1 + size*size)%(size*size)
    nbs[5] = ((row+1)*size + col   + size*size)%(size*size)
    return nbs

def n2_6_6_6_12(pt, size):
    nbs = np.zeros(6).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row-2)*size + col-1 + size*size)%(size*size)
    nbs[1] = ((row-1)*size + col+1 + size*size)%(size*size)
    nbs[2] = ((row+1)*size + col-1 + size*size)%(size*size)
    nbs[3] = ((row-1)*size + col-2 + size*size)%(size*size)
    nbs[4] = ((row+1)*size + col+1 + size*size)%(size*size)
    nbs[5] = ((row+1)*size + col+2 + size*size)%(size*size)
    return nbs

def n3_6_6_6_12(pt, size):
    nbs = np.zeros(6).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row-2)*size + col-2 + size*size)%(size*size)
    nbs[1] = ((row)*size   + col+2 + size*size)%(size*size)
    nbs[2] = ((row-2)*size + col   + size*size)%(size*size)
    nbs[3] = ((row)*size   + col-2 + size*size)%(size*size)
    nbs[4] = ((row+2)*size + col+2 + size*size)%(size*size)
    nbs[5] = ((row+2)*size + col   + size*size)%(size*size)
    return nbs

def n4_6_6_6_12(pt, size):
    nbs = np.zeros(12).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row-2)*size + col-3 + size*size)%(size*size)
    nbs[1] = ((row-3)*size + col-1 + size*size)%(size*size)
    nbs[2] = ((row-3)*size + col-2 + size*size)%(size*size)
    nbs[3] = ((row+1)*size + col+3 + size*size)%(size*size)
    nbs[4] = ((row-1)*size + col+2 + size*size)%(size*size)
    nbs[5] = ((row-2)*size + col+1 + size*size)%(size*size)
    nbs[6] = ((row+2)*size + col-1 + size*size)%(size*size)
    nbs[7] = ((row+1)*size + col-2 + size*size)%(size*size)
    nbs[8] = ((row-1)*size + col-3 + size*size)%(size*size)
    nbs[9] = ((row+3)*size + col+2 + size*size)%(size*size)
    nbs[10] = ((row+1)*size+ col+3 + size*size)%(size*size)
    nbs[11] = ((row+2)*size+ col+3 + size*size)%(size*size)
    return nbs

def n1_4_4_4_8(pt, size):
    nbs = np.zeros(4).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row-1)*size + col-1 + size*size)%(size*size)
    nbs[1] = (row*size     + col+1 + size*size)%(size*size)
    nbs[2] = ((row-1)*size + col   + size*size)%(size*size)
    nbs[3] = ((row)*size   + col-1 + size*size)%(size*size)
    return nbs

def n2_4_4_4_8(pt, size):
    nbs = np.zeros(4).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row-1)*size + col-1 + size*size)%(size*size)
    nbs[1] = ((row-1)*size + col+1 + size*size)%(size*size)
    nbs[2] = ((row+1)*size + col-1 + size*size)%(size*size)
    nbs[3] = ((row+1)*size + col+1 + size*size)%(size*size)
    return nbs

def n3_4_4_4_8(pt, size):
    nbs = np.zeros(4).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row-2)*size + col   + size*size)%(size*size)
    nbs[1] = ((row  )*size + col-2 + size*size)%(size*size)
    nbs[2] = ((row+2)*size + col   + size*size)%(size*size)
    nbs[3] = ((row  )*size + col+2 + size*size)%(size*size)
    return nbs

def n4_4_4_4_8(pt, size):
    nbs = np.zeros(8).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row-2)*size + col-1 + size*size)%(size*size)
    nbs[1] = ((row-1)*size + col-2 + size*size)%(size*size)
    nbs[2] = ((row-2)*size + col+1 + size*size)%(size*size)
    nbs[3] = ((row-1)*size + col+2 + size*size)%(size*size)
    nbs[4] = ((row+1)*size + col-2 + size*size)%(size*size)
    nbs[5] = ((row+2)*size + col-1 + size*size)%(size*size)
    nbs[6] = ((row+1)*size + col+2 + size*size)%(size*size)
    nbs[7] = ((row+2)*size + col+1 + size*size)%(size*size)
    return nbs

def n1_2_2_4_2(pt, size):
    nbs = np.zeros(2).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row-1)*size + col   + size*size)%(size*size)
    nbs[1] = ((row+1)*size + col   + size*size)%(size*size)
    return nbs

def n2_2_2_4_2(pt, size):
    nbs = np.zeros(2).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row+1)*size + col-1 + size*size)%(size*size)
    nbs[1] = ((row+1)*size + col+1 + size*size)%(size*size)
    return nbs

def n3_2_2_4_2(pt, size):
    nbs = np.zeros(4).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row+1)*size + (col-1+size)%size + size*size)%(size*size)
    nbs[1] = ((row+1)*size + col+1 + size*size)%(size*size)
    nbs[2] = (size - (row-1)*size + col-1 + size*size)%(size*size)
    nbs[3] = ((row-1)*size + col+1 + size*size)%(size*size)
    return nbs

def n4_2_2_4_2(pt, size):
    nbs = np.zeros(2).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row-2)*size + col   + size*size)%(size*size)
    nbs[1] = ((row-2)*size + col   + size*size)%(size*size)
    return nbs

def n1_2_4_2_4(pt, size):
    nbs = np.zeros(2).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = (row*size + col-1   + size*size)%(size*size)
    nbs[1] = (row*size + col+1   + size*size)%(size*size)
    return nbs

def n2_2_4_2_4(pt, size):
    nbs = np.zeros(4).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row-1)*size + col   + size*size)%(size*size)
    nbs[1] = ((row+1)*size + col   + size*size)%(size*size)
    nbs[2] = ((row-1)*size + col-1 + size*size)%(size*size)
    nbs[3] = ((row+1)*size + col-1 + size*size)%(size*size)
    return nbs

def n3_2_4_2_4(pt, size):
    nbs = np.zeros(2).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row-2)*size + col   + size*size)%(size*size)
    nbs[1] = ((row+2)*size + col   + size*size)%(size*size)
    return nbs

def n4_2_4_2_4(pt, size):
    nbs = np.zeros(4).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row-2)*size + col-1   + size*size)%(size*size)
    nbs[1] = ((row+2)*size + col+1   + size*size)%(size*size)
    nbs[2] = ((row-2)*size + col-1   + size*size)%(size*size)
    nbs[3] = ((row+2)*size + col+1   + size*size)%(size*size)
    return nbs

def n1_3_6_3_6(pt, size):
    nbs = np.zeros(3).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row+1)*size + col+0 + size*size)%(size*size)
    nbs[1] = ((row-1)*size + col+1 + size*size)%(size*size)
    nbs[2] = ((row-0)*size + col-1 + size*size)%(size*size)
    return nbs

def n2_3_6_3_6(pt, size):
    nbs = np.zeros(6).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row+1)*size + col+1 + size*size)%(size*size)
    nbs[1] = ((row-1)*size + col+2 + size*size)%(size*size)
    nbs[2] = ((row-2)*size + col+1 + size*size)%(size*size)
    nbs[3] = ((row-1)*size + col-1 + size*size)%(size*size)
    nbs[4] = ((row+1)*size + col-2 + size*size)%(size*size)
    nbs[5] = ((row+2)*size + col-1 + size*size)%(size*size)
    return nbs

def n3_3_6_3_6(pt, size):
    nbs = np.zeros(3).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row+2)*size + col   + size*size)%(size*size)
    nbs[1] = ((row-2)*size + col+2 + size*size)%(size*size)
    nbs[2] = ((row  )*size + col-2 + size*size)%(size*size)
    return nbs

def n4_3_6_3_6(pt, size):
    nbs = np.zeros(6).astype(np.int32)
    row, col = pt//size, pt%size
    nbs[0] = ((row-1)*size + col+3 + size*size)%(size*size)
    nbs[1] = ((row-3)*size + col+1 + size*size)%(size*size)
    nbs[2] = ((row-2)*size + col-1 + size*size)%(size*size)
    nbs[3] = ((row+2)*size + col-3 + size*size)%(size*size)
    nbs[4] = ((row-3)*size + col+2 + size*size)%(size*size)
    nbs[5] = ((row+1)*size + col+2 + size*size)%(size*size)
    return nbs

# Definitions of Hamiltonians

def H_6_6_6_12_DM1(mat, grid, pt, sx, sy, sz, dmi, B, size):
    H = 0.0
    d_plane = mat[25]
    n1 = n1_6_6_6_12(pt, size)
    n2 = n2_6_6_6_12(pt, size)
    n3 = n3_6_6_6_12(pt, size)
    n4 = n4_6_6_6_12(pt, size)
    
    for i in range(6):
        H += mat[1]*(sx*grid[n1[i]*3] + sy*grid[n1[i]*3+1] + sz*grid[n1[i]*3+2]) + mat[5]*sx*grid[n1[i]*3] + mat[6]*sy*grid[n1[i]*3+1] + mat[7]*sz*grid[n1[i]*3+2]
        crij_x = (sz*grid[n1[i]*3+1] - sy*grid[n1[i]*3+2]) * dmi[i*3]
        crij_y = (sx*grid[n1[i]*3+2] - sz*grid[n1[i]*3]) * dmi[i*3+1]
        crij_z = (sy*grid[n1[i]*3] - sx*grid[n1[i]*3+1]) * dmi[i*3+2]
        H += (crij_x + crij_y + crij_z) * d_plane
    
    for i in range(6):
        H += mat[2]*(sx*grid[n2[i]*3] + sy*grid[n2[i]*3+1] + sz*grid[n2[i]*3+2]) + mat[8]*sx*grid[n2[i]*3] + mat[9]*sy*grid[n2[i]*3+1] + mat[10]*sz*grid[n2[i]*3+2]
    
    for i in range(6):
        H += mat[3]*(sx*grid[n3[i]*3] + sy*grid[n3[i]*3+1] + sz*grid[n3[i]*3+2]) + mat[11]*sx*grid[n3[i]*3] + mat[12]*sy*grid[n3[i]*3+1] + mat[13]*sz*grid[n3[i]*3+2]
    
    for i in range(12):
        H += mat[4]*(sx*grid[n4[i]*3] + sy*grid[n4[i]*3+1] + sz*grid[n4[i]*3+2]) + mat[14]*sx*grid[n4[i]*3] + mat[15]*sy*grid[n4[i]*3+1] + mat[16]*sz*grid[n4[i]*3+2]
        
    H += mat[17]*sx*sx + mat[18]*sy*sy + mat[19]*sz*sz
    H += B*sz
    return -H

def H_6_6_6_12_DM0(mat, grid, pt, sx, sy, sz, B, size):
    H = 0.0
    n1 = n1_6_6_6_12(pt, size)
    n2 = n2_6_6_6_12(pt, size)
    n3 = n3_6_6_6_12(pt, size)
    n4 = n4_6_6_6_12(pt, size)
    
    for i in range(6):
        H += mat[1]*(sx*grid[n1[i]*3] + sy*grid[n1[i]*3+1] + sz*grid[n1[i]*3+2]) + mat[5]*sx*grid[n1[i]*3] + mat[6]*sy*grid[n1[i]*3+1] + mat[7]*sz*grid[n1[i]*3+2]
    
    for i in range(6):
        H += mat[2]*(sx*grid[n2[i]*3] + sy*grid[n2[i]*3+1] + sz*grid[n2[i]*3+2]) + mat[8]*sx*grid[n2[i]*3] + mat[9]*sy*grid[n2[i]*3+1] + mat[10]*sz*grid[n2[i]*3+2]
    
    for i in range(6):
        H += mat[3]*(sx*grid[n3[i]*3] + sy*grid[n3[i]*3+1] + sz*grid[n3[i]*3+2]) + mat[11]*sx*grid[n3[i]*3] + mat[12]*sy*grid[n3[i]*3+1] + mat[13]*sz*grid[n3[i]*3+2]
    
    for i in range(12):
        H += mat[4]*(sx*grid[n4[i]*3] + sy*grid[n4[i]*3+1] + sz*grid[n4[i]*3+2]) + mat[14]*sx*grid[n4[i]*3] + mat[15]*sy*grid[n4[i]*3+1] + mat[16]*sz*grid[n4[i]*3+2]
        
    H += mat[17]*sx*sx + mat[18]*sy*sy + mat[19]*sz*sz
    H += B*sz
    return -H

def H_4_4_4_8_DM1(mat, grid, pt, sx, sy, sz, dmi, B, size):
    H = 0.0
    d_plane = mat[25]
    n1 = n1_4_4_4_8(pt, size)
    n2 = n2_4_4_4_8(pt, size)
    n3 = n3_4_4_4_8(pt, size)
    n4 = n4_4_4_4_8(pt, size)
    
    for i in range(4):
        H += mat[1]*(sx*grid[n1[i]*3] + sy*grid[n1[i]*3+1] + sz*grid[n1[i]*3+2]) + mat[5]*sx*grid[n1[i]*3] + mat[6]*sy*grid[n1[i]*3+1] + mat[7]*sz*grid[n1[i]*3+2]
        crij_x = (sz*grid[n1[i]*3+1] - sy*grid[n1[i]*3+2]) * dmi[i*3]
        crij_y = (sx*grid[n1[i]*3+2] - sz*grid[n1[i]*3]) * dmi[i*3+1]
        crij_z = (sy*grid[n1[i]*3] - sx*grid[n1[i]*3+1]) * dmi[i*3+2]
        H += (crij_x + crij_y + crij_z) * d_plane
    
    for i in range(4):
        H += mat[2]*(sx*grid[n2[i]*3] + sy*grid[n2[i]*3+1] + sz*grid[n2[i]*3+2]) + mat[8]*sx*grid[n2[i]*3] + mat[9]*sy*grid[n2[i]*3+1] + mat[10]*sz*grid[n2[i]*3+2]
    
    for i in range(4):
        H += mat[3]*(sx*grid[n3[i]*3] + sy*grid[n3[i]*3+1] + sz*grid[n3[i]*3+2]) + mat[11]*sx*grid[n3[i]*3] + mat[12]*sy*grid[n3[i]*3+1] + mat[13]*sz*grid[n3[i]*3+2]
    
    for i in range(8):
        H += mat[4]*(sx*grid[n4[i]*3] + sy*grid[n4[i]*3+1] + sz*grid[n4[i]*3+2]) + mat[14]*sx*grid[n4[i]*3] + mat[15]*sy*grid[n4[i]*3+1] + mat[16]*sz*grid[n4[i]*3+2]
    
    H += mat[17]*sx*sx + mat[18]*sy*sy + mat[19]*sz*sz
    H += B*sz
    return -H

def H_4_4_4_8_DM0(mat, grid, pt, sx, sy, sz, B, size):
    H = 0.0
    n1 = n1_4_4_4_8(pt, size)
    n2 = n2_4_4_4_8(pt, size)
    n3 = n3_4_4_4_8(pt, size)
    n4 = n4_4_4_4_8(pt, size)
    
    for i in range(4):
        H += mat[1]*(sx*grid[n1[i]*3] + sy*grid[n1[i]*3+1] + sz*grid[n1[i]*3+2]) + mat[5]*sx*grid[n1[i]*3] + mat[6]*sy*grid[n1[i]*3+1] + mat[7]*sz*grid[n1[i]*3+2]
    
    for i in range(4):
        H += mat[2]*(sx*grid[n2[i]*3] + sy*grid[n2[i]*3+1] + sz*grid[n2[i]*3+2]) + mat[8]*sx*grid[n2[i]*3] + mat[9]*sy*grid[n2[i]*3+1] + mat[10]*sz*grid[n2[i]*3+2]
    
    for i in range(4):
        H += mat[3]*(sx*grid[n3[i]*3] + sy*grid[n3[i]*3+1] + sz*grid[n3[i]*3+2]) + mat[11]*sx*grid[n3[i]*3] + mat[12]*sy*grid[n3[i]*3+1] + mat[13]*sz*grid[n3[i]*3+2]
    
    for i in range(8):
        H += mat[4]*(sx*grid[n4[i]*3] + sy*grid[n4[i]*3+1] + sz*grid[n4[i]*3+2]) + mat[14]*sx*grid[n4[i]*3] + mat[15]*sy*grid[n4[i]*3+1] + mat[16]*sz*grid[n4[i]*3+2]
    
    H += mat[17]*sx*sx + mat[18]*sy*sy + mat[19]*sz*sz
    H += B*sz
    return -H

def H_3_6_3_6_DM1(mat, grid, pt, sx, sy, sz, dmi, B, size):
    H = 0.0
    d_plane = mat[25]
    n1 = n1_3_6_3_6(pt, size)
    n2 = n2_3_6_3_6(pt, size)
    n3 = n3_3_6_3_6(pt, size)
    n4 = n4_3_6_3_6(pt, size)
    
    for i in range(3):
        H += mat[1]*(sx*grid[n1[i]*3] + sy*grid[n1[i]*3+1] + sz*grid[n1[i]*3+2]) + mat[5]*sx*grid[n1[i]*3] + mat[6]*sy*grid[n1[i]*3+1] + mat[7]*sz*grid[n1[i]*3+2]
        crij_x = (sz*grid[n1[i]*3+1] - sy*grid[n1[i]*3+2]) * dmi[i*3]
        crij_y = (sx*grid[n1[i]*3+2] - sz*grid[n1[i]*3]) * dmi[i*3+1]
        crij_z = (sy*grid[n1[i]*3] - sx*grid[n1[i]*3+1]) * dmi[i*3+2]
        H += (crij_x + crij_y + crij_z) * d_plane
    
    for i in range(3):
        H += mat[2]*(sx*grid[n2[i]*3] + sy*grid[n2[i]*3+1] + sz*grid[n2[i]*3+2]) + mat[8]*sx*grid[n2[i]*3] + mat[9]*sy*grid[n2[i]*3+1] + mat[10]*sz*grid[n2[i]*3+2]
    
    for i in range(3):
        H += mat[3]*(sx*grid[n3[i]*3] + sy*grid[n3[i]*3+1] + sz*grid[n3[i]*3+2]) + mat[11]*sx*grid[n3[i]*3] + mat[12]*sy*grid[n3[i]*3+1] + mat[13]*sz*grid[n3[i]*3+2]
    
    for i in range(6):
        H += mat[4]*(sx*grid[n4[i]*3] + sy*grid[n4[i]*3+1] + sz*grid[n4[i]*3+2]) + mat[14]*sx*grid[n4[i]*3] + mat[15]*sy*grid[n4[i]*3+1] + mat[16]*sz*grid[n4[i]*3+2]
    
    H += mat[17]*sx*sx + mat[18]*sy*sy + mat[19]*sz*sz
    H += B*sz
    return -H

def H_3_6_3_6_DM0(mat, grid, pt, sx, sy, sz, B, size):
    H = 0.0
    n1 = n1_3_6_3_6(pt, size)
    n2 = n2_3_6_3_6(pt, size)
    n3 = n3_3_6_3_6(pt, size)
    n4 = n4_3_6_3_6(pt, size)
    
    for i in range(3):
        H += mat[1]*(sx*grid[n1[i]*3] + sy*grid[n1[i]*3+1] + sz*grid[n1[i]*3+2]) + mat[5]*sx*grid[n1[i]*3] + mat[6]*sy*grid[n1[i]*3+1] + mat[7]*sz*grid[n1[i]*3+2]
    
    for i in range(3):
        H += mat[2]*(sx*grid[n2[i]*3] + sy*grid[n2[i]*3+1] + sz*grid[n2[i]*3+2]) + mat[8]*sx*grid[n2[i]*3] + mat[9]*sy*grid[n2[i]*3+1] + mat[10]*sz*grid[n2[i]*3+2]
    
    for i in range(3):
        H += mat[3]*(sx*grid[n3[i]*3] + sy*grid[n3[i]*3+1] + sz*grid[n3[i]*3+2]) + mat[11]*sx*grid[n3[i]*3] + mat[12]*sy*grid[n3[i]*3+1] + mat[13]*sz*grid[n3[i]*3+2]
    
    for i in range(6):
        H += mat[4]*(sx*grid[n4[i]*3] + sy*grid[n4[i]*3+1] + sz*grid[n4[i]*3+2]) + mat[14]*sx*grid[n4[i]*3] + mat[15]*sy*grid[n4[i]*3+1] + mat[16]*sz*grid[n4[i]*3+2]
    
    H += mat[17]*sx*sx + mat[18]*sy*sy + mat[19]*sz*sz
    H += B*sz
    return -H

def H_2_2_4_2_DM0(mat, grid, pt, sx, sy, sz, dmi, size):
    H = 0.0
    n1 = n1_2_2_4_2(pt, size)
    n2 = n2_2_2_4_2(pt, size)
    n3 = n3_2_2_4_2(pt, size)
    n4 = n4_2_2_4_2(pt, size)
    
    for i in range(2):
        H += mat[1]*(sx*grid[n1[i]*3] + sy*grid[n1[i]*3+1] + sz*grid[n1[i]*3+2]) + mat[5]*sx*grid[n1[i]*3] + mat[6]*sy*grid[n1[i]*3+1] + mat[7]*sz*grid[n1[i]*3+2]
    
    for i in range(2):
        H += mat[2]*(sx*grid[n2[i]*3] + sy*grid[n2[i]*3+1] + sz*grid[n2[i]*3+2]) + mat[8]*sx*grid[n2[i]*3] + mat[9]*sy*grid[n2[i]*3+1] + mat[10]*sz*grid[n2[i]*3+2]
    
    for i in range(4):
        H += mat[3]*(sx*grid[n3[i]*3] + sy*grid[n3[i]*3+1] + sz*grid[n3[i]*3+2]) + mat[11]*sx*grid[n3[i]*3] + mat[12]*sy*grid[n3[i]*3+1] + mat[13]*sz*grid[n3[i]*3+2]
    
    for i in range(2):
        H += mat[4]*(sx*grid[n4[i]*3] + sy*grid[n4[i]*3+1] + sz*grid[n4[i]*3+2]) + mat[14]*sx*grid[n4[i]*3] + mat[15]*sy*grid[n4[i]*3+1] + mat[16]*sz*grid[n4[i]*3+2]
        
    H += mat[17]*sx*sx + mat[18]*sy*sy + mat[19]*sz*sz
    H += B*sz
    return -H

def H_2_4_2_4_DM0(mat, grid, pt, sx, sy, sz, dmi, size):
    H = 0.0
    n1 = n1_2_4_2_4(pt, size)
    n2 = n2_2_4_2_4(pt, size)
    n3 = n3_2_4_2_4(pt, size)
    n4 = n4_2_4_2_4(pt, size)
    
    for i in range(2):
        H += mat[1]*(sx*grid[n1[i]*3] + sy*grid[n1[i]*3+1] + sz*grid[n1[i]*3+2]) + mat[5]*sx*grid[n1[i]*3] + mat[6]*sy*grid[n1[i]*3+1] + mat[7]*sz*grid[n1[i]*3+2]
    
    for i in range(4):
        H += mat[2]*(sx*grid[n2[i]*3] + sy*grid[n2[i]*3+1] + sz*grid[n2[i]*3+2]) + mat[8]*sx*grid[n2[i]*3] + mat[9]*sy*grid[n2[i]*3+1] + mat[10]*sz*grid[n2[i]*3+2]
    
    for i in range(2):
        H += mat[3]*(sx*grid[n3[i]*3] + sy*grid[n3[i]*3+1] + sz*grid[n3[i]*3+2]) + mat[11]*sx*grid[n3[i]*3] + mat[12]*sy*grid[n3[i]*3+1] + mat[13]*sz*grid[n3[i]*3+2]
    
    for i in range(4):
        H += mat[4]*(sx*grid[n4[i]*3] + sy*grid[n4[i]*3+1] + sz*grid[n4[i]*3+2]) + mat[14]*sx*grid[n4[i]*3] + mat[15]*sy*grid[n4[i]*3+1] + mat[16]*sz*grid[n4[i]*3+2]
        
    H += mat[17]*sx*sx + mat[18]*sy*sy + mat[19]*sz*sz
    H += B*sz
    return -H

def Serial_MC_6_6_6_12_DM1(mat, grid, size, dmi, T, BJ):
    n_rn = cp.random.randint(0, size*size, size=size*size).get()
    u_rn = cp.random.uniform(size=size*size).get()
    v_rn = cp.random.uniform(size=size*size).get()
    r_rn = cp.random.uniform(size=size*size).get()
    for j in range(size*size):
        pt = n_rn[j]
        H0 = H_6_6_6_12_DM1(mat, grid, pt, grid[pt*3], grid[pt*3+1], grid[pt*3+2], dmi, BJ, size)
        theta, phi = 2.0*np.pi*u_rn[j], np.arccos(2.0*v_rn[j]-1.0)
        sx, sy, sz = np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)
        H1 = H_6_6_6_12_DM1(mat, grid, pt, sx, sy, sz, dmi, BJ, size)
        if H1 < H0:
            grid[pt*3], grid[pt*3+1], grid[pt*3+2] = sx, sy, sz
        else:
            if r_rn[j] < np.exp(-(H1-H0)*T):
                grid[pt*3], grid[pt*3+1], grid[pt*3+2] = sx, sy, sz

def Serial_MC_6_6_6_12_DM0(mat, grid, size, T, BJ, spin):
    n_rn = cp.random.randint(0, size*size, size=size*size).get()
    u_rn = cp.random.uniform(size=size*size).get()
    v_rn = cp.random.uniform(size=size*size).get()
    r_rn = cp.random.uniform(size=size*size).get()
    for j in range(size*size):
        pt = n_rn[j]
        H0 = H_6_6_6_12_DM0(mat, grid, pt, grid[pt*3], grid[pt*3+1], grid[pt*3+2], BJ, size)
        theta, phi = 2.0*np.pi*u_rn[j], np.arccos(2.0*v_rn[j]-1.0)
        sx, sy, sz = np.sin(phi)*np.cos(theta)*spin, np.sin(phi)*np.sin(theta)*spin, np.cos(phi)*spin
        H1 = H_6_6_6_12_DM0(mat, grid, pt, sx, sy, sz, BJ, size)
        if H1 < H0:
            grid[pt*3], grid[pt*3+1], grid[pt*3+2] = sx, sy, sz
        else:
            if r_rn[j] < np.exp(-(H1-H0)*T):
                grid[pt*3], grid[pt*3+1], grid[pt*3+2] = sx, sy, sz

def Serial_MC_3_6_3_6_DM0(mat, grid, size, T, BJ, spin):
    n_rn = cp.random.randint(0, size*size, size=size*size).get()
    u_rn = cp.random.uniform(size=size*size).get()
    v_rn = cp.random.uniform(size=size*size).get()
    r_rn = cp.random.uniform(size=size*size).get()
    for j in range(size*size):
        pt = n_rn[j]
        H0 = H_3_6_3_6_DM0(mat, grid, pt, grid[pt*3], grid[pt*3+1], grid[pt*3+2], BJ, size)
        theta, phi = 2.0*np.pi*u_rn[j], np.arccos(2.0*v_rn[j]-1.0)
        sx, sy, sz = np.sin(phi)*np.cos(theta)*spin, np.sin(phi)*np.sin(theta)*spin, np.cos(phi)*spin
        H1 = H_3_6_3_6_DM0(mat, grid, pt, sx, sy, sz, BJ, size)
        if H1 < H0:
            grid[pt*3], grid[pt*3+1], grid[pt*3+2] = sx, sy, sz
        else:
            if r_rn[j] < np.exp(-(H1-H0)*T):
                grid[pt*3], grid[pt*3+1], grid[pt*3+2] = sx, sy, sz

class serial_MonteCarlo:
    def __init__(self, config):
        with open(config, "r") as f:
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
        self.dmi_3 = np.load("dmi_3.npy")
        self.dmi_4 = np.load("dmi_4.npy")
        self.dmi_6 = np.load("dmi_6.npy")
        self.MAT_NAME, self.MAT_PARAMS = rm.read_2dmat("../"+self.Input_Folder+"TC_"+self.Material+".csv")
        self.spin = self.MAT_PARAMS[0]
        self.save_direcotry = "../"+self.Output_Folder+self.Prefix+"_"+self.Material+"_"+str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+ "/")
        self.NBS = int(self.MAT_PARAMS[20]), int(self.MAT_PARAMS[21]), int(self.MAT_PARAMS[22]), int(self.MAT_PARAMS[23])
        
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
        os.mkdir(self.save_direcotry)
        
        self.metadata_file = self.save_direcotry+"/metadata.json"
        with open(self.metadata_file, 'w+') as f:
            f.write(metadata_file)
    
    def display_material(self):
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
    
    def mc_init(self):
        self.size = int(self.size)
        if self.FM_Flag:
            self.grid = np.ones((self.size*self.size*3)).astype(np.float32) * self.spin
        else:
            u, v = np.random.uniform(size=self.size*self.size), np.random.uniform(size=self.size*self.size)
            theta, phi = 2.0*np.pi*u, np.arccos(2.0*v-1.0)
            self.grid = np.zeros((self.size*self.size*3)).astype(np.float32)
            self.grid[0::3] = np.sin(phi)*np.cos(theta)
            self.grid[1::3] = np.sin(phi)*np.sin(theta)
            self.grid[2::3] = np.cos(phi)
            self.grid *= self.spin
        
        if self.Static_T_Flag:
            self.T = self.Temps
        else:
            self.T = np.linspace(0.01, np.float32(2.0*self.MAT_PARAMS[24]), 11)
        self.T_copy = self.T.copy()
        for i in range(len(self.T)):
            self.T[i] = 1/np.float32(self.T[i]*kb)
        
    def grid_reset(self):
        self.grid = np.zeros((self.size*self.size*3)).astype(np.float32)
        if self.FM_Flag:
            mc.FM_N(self.grid, self.size)
        else:
            mc.AFM_N(self.grid, self.size)
        self.grid *= self.spin
    
    def mc_run_6_6_6_12_dm1(self):
        for i in range(self.stability_runs):
            Serial_MC_6_6_6_12_DM1(self.MAT_PARAMS, self.grid, self.size, self.dmi_6.reshape(18), self.T[0], self.B_C)
            if i%10 == 0:
                print("Stability Run: ", i)
            np.save(self.save_direcotry + "grid_" + str(i), self.grid)
    
    def mc_tc_6_6_6_12_dm0(self):
        M, X = [], []
        for j in range(len(self.T)):
            for i in tqdm.tqdm(range(self.calculation_runs)):
                Serial_MC_6_6_6_12_DM0(self.MAT_PARAMS, self.grid, self.size, self.T[j], self.B_C, self.spin)
            
            np.save(self.save_direcotry + "grid_" + str(j), self.grid)
            m = np.sum(self.grid[2::3])/self.size/self.size
            M.append(m)
            X.append(m*m)
            print(f"Temp: {self.T_copy[j]:.4f} M: {m:.4f}")
            print(f"Temp: {self.T_copy[j]:.4f} X: {m*m:.4f}")
            print("-"*20)
        np.save(self.save_direcotry + "M", M)
        np.save(self.save_direcotry + "X", X)
    
    def mc_tc_3_6_3_6_dm0(self):
        M, X = [], []
        for j in range(len(self.T)):
            for i in tqdm.tqdm(range(self.calculation_runs)):
                Serial_MC_3_6_3_6_DM0(self.MAT_PARAMS, self.grid, self.size, self.T[j], self.B_C, self.spin)
            
            np.save(self.save_direcotry + "grid_" + str(j), self.grid)
            m = np.sum(self.grid[2::3])/self.size/self.size
            M.append(m)
            X.append(m*m)
            print(f"Temp: {self.T_copy[j]:.4f} M: {m:.4f}")
            print(f"Temp: {self.T_copy[j]:.4f} X: {m*m:.4f}")
            print("-"*20)
        np.save(self.save_direcotry + "M", M)
        np.save(self.save_direcotry + "X", X)
