# CUDA METRO
[![DOI](https://zenodo.org/badge/705754637.svg)](https://zenodo.org/doi/10.5281/zenodo.10472803)

A pyCUDA based Metropolis Monte Carlo simulator for 2D-magnetism systems. This code uses NVIDIA CUDA architecture wrapped by a simple python wrapper to work.
To use, clone the repository using
```
git clone https://github.com/arkavo/CUDA-METRO.git
```
into a directory of choice. Creating a new python3 environment is recommended to run the code.
To setup the environment after creation, simply run ```pip install -r requirements.pip```

Some template codes have already been provided under folder ```/src``` along with sample input files in ```/inputs``` and running configurations in ```/configs```

## Vector analysis
To run a vector analysis, simple execute ```python test_main.py <config file name>``` (main config file is ```/configs/test_config.json```)

After execution, an output folder will be created with name ```<prefix>_<material>_<date>_<time>``` with ```.npy``` files in them containing spin data in 1D array form as [s<sub>1</sub><sup>x</sup> s<sub>1</sub><sup>y</sup> s<sub>1</sub><sup>z</sup> s<sub>2</sub><sup>x</sup> s<sub>2</sub><sup>y</sup>.........s<sub>n</sub><sup>y</sup> s<sub>n</sub><sup>z</sup>] and an additional ```metadata.json``` file containing the details of the run.

For casual viewing, it is advised to use the in-built viewer as ```python test_view.py <folder name>``` to provide visuals of the spin magnitudes in 3 directions as well as a top-down vector map of the spins.

## Curie temperature analysis
To run a Curie temperature analysis, execute ```python tc_sims.py``` after configuring the appropriate config file in ```/configs/tc_config.json```.

After the run, a graph of temperature vs magnetization and temperature vs susceptibility will be auto generated.
