import numpy as np

data = np.load("../outputs/0_VZr3C3II_2023_11_11_00_59_53/grid_0000.npy")
data = data.reshape((300,300,3))
print(data[0])