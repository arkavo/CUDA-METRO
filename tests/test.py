# Testing script for the CUDAMETRO package
# Currently in version 1.2.1


# Importing the necessary packages

print("=== Testing the CUDAMETRO package ===")
print("=== Testing the import of the necessary packages ===")
print(" Searching for CUDAMETRO")
try:
    import cudametro as cm
except ImportError:
    print("Please install the package using the following command: pip install cudametro")

if "cm" in globals():
    print(f"CUDA METRO package is installed. Version: {cm.__version__}")

print(" Searching for numpy")
try:
    import numpy as np
except ImportError:
    print("Please install the package using the following command: pip install numpy")
if "np" in globals():
    print(f"NumPy package is installed. Version: {np.__version__}")

print(" Searching for matplotlib")
try:
    import matplotlib as plt
except ImportError:
    print("Please install the package using the following command: pip install matplotlib")
if "plt" in globals():
    print(f"Matplotlib package is installed. Version: {plt.__version__}")

print(" Searching for tqdm")
try:
    import tqdm as tqdm
except ImportError:
    print("Please install the package using the following command: pip install tqdm")
if "tqdm" in globals():
    print(f"TQDM package is installed. Version: {tqdm.__version__}")

print(" Searching for seaborn")
try:
    import seaborn
except ImportError:
    print("Please install the package using the following command: pip install seaborn")
if "seaborn" in globals():
    print(f"Seaborn package is installed. Version: {seaborn.__version__}")

print(" Searching for pyCUDA")    
try:
    import pycuda
except ImportError:
    print("Please install the package using the following command: pip install pycuda")
    print("Please refer to the following link for installation instructions: https://wiki.tiker.net/PyCuda/Installation/")
if "pycuda" in globals():
    print(f"PyCUDA package is installed. Version: {pycuda.VERSION_TEXT}")

print("=== All packages are installed ===")

# Testing GPU availability
print("=== Testing GPU availability ===")
print(" Searching for GPU")
try:
    import pycuda.driver as cuda
    cuda.init()
    print(f"GPU found: {cuda.Device.count()} devices found")
    for i in range(cuda.Device.count()):
        dev = cuda.Device(i)
        print(f"Device {i}: {dev.name()}")
except ImportError:
    print("Please install the package using the following command: pip install pycuda")
    print("Please refer to the following link for installation instructions: https://wiki.tiker.net/PyCuda/Installation/")

# Print GPU information
print("=== GPU Information ===")
print('\x1b[6;30;42m')

for i in range(cuda.Device.count()):
    print(f"Device {i}: {cuda.Device(i).name()}")
    print(f"Compute Capability: {cuda.Device(i).compute_capability()}")
    print(f"Total Memory: {cuda.Device(i).total_memory()/1024**2} MB")
    print(f"Max Threads per Block: {cuda.Device(i).max_threads_per_block}")
    print(f"Max Threads per Multiprocessor: {cuda.Device(i).max_threads_per_multiprocessor}")
    print(f"Max Grid Size: {cuda.Device(i).max_grid_dim_x}, {cuda.Device(i).max_grid_dim_y}, {cuda.Device(i).max_grid_dim_z}")
    print(f"Max Threads per Block: {cuda.Device(i).max_threads_per_block}")

print('\x1b[0m')

if cuda.Device(i).compute_capability() < (5,0):
    print("Please ensure that the GPU has a compute capability of at least 5.0")
    print("Please refer to the following link for further information: https://developer.nvidia.com/cuda-gpus")

print("CHECK COMPLETE")
print("=== All tests are complete ===")
print("=== Please refer to the README.md file for further instructions ===")
print("=== If you encounter any issues, please refer to the documentation ===")
print("=== For any further queries, please contact the author ===")
print("=== Thank you for using the CUDAMETRO package ===")