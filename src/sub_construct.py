import numpy as np

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

def Serial_MC_6_6_6_12_DM1(grid, size, T, BJ, TMATRIX, stability_runs):
    for i in range(stability_runs):
        for j in range(size*size):
            pt = np.random.randint(0, size*size)
            sx, sy, sz = grid[pt*3], grid[pt*3+1], grid[pt*3+2]
            H = H_6_6_6_12_DM1(TMATRIX, grid, pt, sx, sy, sz, BJ, size)
            if H < 0:
                grid[pt*3] *= -1
                grid[pt*3+1] *= -1
                grid[pt*3+2] *= -1
            elif np.exp(-H/T) > np.random.rand():
                grid[pt*3] *= -1
                grid[pt*3+1] *= -1
                grid[pt*3+2] *= -1
    return grid

class serial_MonteCarlo:
    def __init__(self, config):
        self.size = config["Size"]
        self.spin = config["spin"]
        self.FM_Flag = config["FM"]
        self.Blocks = config["Blocks"]
        self.stability_runs = config["Stability"]
        self.save_direcotry = config["Directory"]
        self.MAT_NAME, self.MAT_PARAMS = rm.read_2dmat("../inputs/"+"TC_"+config["Material"]+".csv")
        self.B = config["B"]
        
    def mc_init(self):
        self.grid = np.zeros((self.size*self.size*3)).astype(np.float32)
        self.TMATRIX = np.zeros((self.Blocks, 4)).astype(np.float32)
        if self.FM_Flag:
            mc.FM_N(self.grid, self.size)
        else:
            mc.AFM_N(self.grid, self.size)
        self.grid *= self.spin
        self.T = np.linspace(0.01, np.float32(2.0*self.MAT_PARAMS[24]), 11)
        self.BJ = np.array([1.0 / (self.T * 8.6173e-2)],dtype=np.float32)
        self.BJ = self.BJ[0]
        
    def grid_reset(self):
        self.grid = np.zeros((self.size*self.size*3)).astype(np.float32)
        if self.FM_Flag:
            mc.FM_N(self.grid, self.size)
        else:
            mc.AFM_N(self.grid, self.size)
        self.grid *= self.spin
    
    
    