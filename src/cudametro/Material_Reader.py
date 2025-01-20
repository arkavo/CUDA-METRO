# This file contains the function to read the material properties from a file
# Sample file structures are in the 'inputs' folder
#--------------------------------------------
import numpy as np
import re

def read_2dmat(filename):
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


