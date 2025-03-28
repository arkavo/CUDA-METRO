# CUDA METRO

[![PyPI version](https://badge.fury.io/py/cudametro.svg)](https://badge.fury.io/py/cudametro)

# Description/Installation

A pyCUDA based Metropolis Monte Carlo simulator for 2D-magnetism systems. This code uses NVIDIA CUDA architecture wrapped by a simple python wrapper to work.

To use, clone the repository using

```
git clone https://github.com/arkavo/CUDA-METRO.git
```

into a directory of choice. Creating a new python3(>=3.8) environment is recommended to run the code.

To test your installation, please go to the ```/tests``` folder and run the test files. The test files are designed to run on a GPU. If you have installed the code correctly, the tests should pass without any errors.

To setup the environment after creation, simply run 
				```pip install -r requirements.pip```

Some template codes have already been provided under folder ```/src``` along with sample input files in ```/inputs``` and running configurations in ```/configs```

There are 4 seperate files to design your simulations. To run a simple simulation, use ```python main.py <input_file_name>```.

# Executing a simulation
This is a standalone code with ```pyCuda``` backend. An "Input" to this code's kernel would be a matrix which has 3D spins of lattices as an array with the structure ```[site1_spin_x site1_spin_y site1_spin_z site2_spin_x ..... siteN_spin_z]```. Executing the simulation will then access this array and perform a Metropolis Monte Carlo simulation which will then give an output which will be automatically saved in the save directory with the same array structure in a ```.npy``` format.

## Setup
To execute a simulation, you will need 3 files.
<list>
<li>Config file (examples in /configs/test_config.json)
<li>Material parameters (examples in /inputs)
<li>Script file (examples in /src/cudametro)
</list>

The ```Config file``` determines the parameters for the simulation, including the amount of VRAM used by your GPU if you are facing crashes or out of memory issues.

The ```Material parameters``` file contains all the properties of the material(interaction parameters) as a vector. The crystal configuration is also stored in this file. Note that this file has to be referenced in the input.

<details>
<summary>Vector structure</summary>
<br>
[name, spin, J1, J2, J3, J4, K1x, K1y, K1z, K2x, K2y, K2z, K3x, K3y, K3z, K4x, K4y, K4z, Ax, Ay, Az, Tc(experimental), structure, DMI]
</details>

## Usage

To execute a fresh simulation, run the ```script file``` with an ```input file```. The script file must have the construct library bindings as defined in [Construct MonteCarlo](#construct-montecarlo). This will also generate an output folder which contains the lattice of spins in the format ```[site1_spin_x site1_spin_y site1_spin_z site2_spin_x ..... siteN_spin_z]``` with $N=n^2$, the last spin. The contents of this folder can be easily visualized by using the bindings of the provided [Construct Analyze](#construct-analyze) library. Note that a seperate script has to be wrtitten. A simple sample of such a script is given in ```/src/visualize.py``` which is ready to run and can analyze any given folder.

To execute a simulation, run

```python <script>```

To visualize a simulation, run

```python <visual_script> <folder_path>```

if you are running a custom job but simply running
```python visualize.py <folder_path>``` will work for most cases.

To run a Critical temperature analysis, execute ```python tc_sims.py``` after tweaking the appropriate config file in ```/configs/tc_config.json```. The script will create its own graph at runtime, no additional script needed. If facing out of memory error, please consider lowering "Block" count in the config file.

### A basic example for MnSeTe is given in ```/src/cudametro/sample.py```.

This example is a simple script that initializes the simulation and runs it for a single temperature. The output is saved in the ```Output``` folder, created in the same directory.

There are more scripts inside the ```sample_scripts``` directory which can be used to run simulations for different materials as given in the accompanying paper. These are portable scripts and can be run from anywhere.

# Custom input files

You can also create your custom input and script files. Instructions below.

## Config

If one wants to create their own ```config``` file from scratch, please copy this json template:
```
{
    "Single_Mat_Flag" : 1 if single material 0 otherwise (int),
    "Animation_Flags" : DEPRECATED,
    "DMI_Flag"        : 1 if simulation in DMI mode 0 otherwise (int),
    "TC_Flag"         : 1 if simulation is in Critical Temperature mode 0 otherwise (int),
    "Static_T_Flag"   : 1 if simulation has a single temperature 0 otherwise (int),
    "FM_Flag"         : 1 if starting state is FM 0 otherwise,
    "Input_flag"      : 1 if starting from a specified <input.npy> state 0 otherwise,
    "Input_File"      : String value(use quotes please) for starting state file name (if applicable),
    "Temps"           : Array form of temperatures used for simulation (only in Critical Temperature mode),
    "Material"        : Material name (omit the ".csv"),
    "Multiple_Materials" : File name with multiple materials (omit the ".csv"),
    "SIZE"    :  Lattice size (int),
    "Box"       : DEPRECATED,
    "Blocks"    : How much to parallelize (int),
    "Threads"   : 2 (dont change this),
    "B"         : external magnetic field (use quotes, double),
    "stability_runs"    : MC Phase 1 batch runs,
    "calculation_runs"  : MC Phase 2 batch runs,
    "stability_wrap"    : MC Phase 1 batch size,
    "calculation_wrap"  : MC Phase 2 batch size,
    "Cmpl_Flag"         : DEPRECATED,
    "Prefix"            : String value to appear in FRONT of output folder
}
```
> **_NOTE 1:_** The simulator can work in *either* Critical Temperature or DMI mode, because of the different Hamiltonians, while it is easy to see its an OR condition, please do not attempt to use 1 on both.
>  
> **_NOTE 2:_** The total number of raw MC steps will be ```Blocks x (MC Phase 1 runs x MC Phase 1 size + MC Phase 2 runs x MC Phase 2 size)```. We typically divide the phases to study the Critical temperature, since that typically gives the simulation time to settle down in ```Phase 1``` and then find out the statistical properties in ```Phase 2```(which is our data collection phase). For any raw simulation, where the evolution of states are required from start to finish, one may keep any one phase and omit the other.

## Script

A main ```script.py``` is also required to actually "run" the simulation. This file will contain the simulaiton initializer (with parameters drawn from the config file) and the material in question (with parameters drawn from the corresponding file in ```/inputs```). A sample file is given at ```/src/cudametro/main.py``` which contains a template with basic functionality. Additional functionality is given below in [functions](#functions).

Similarly, for visualization, a file in ```/src/cudametro/visualize.py``` is supplied as a template. Additional functionality is given below in [analyze](#construct-analyze). Note that this does not require a ```config file``` to run but rather a ```folder_path```.


# Functions

A template file is given as ```main.py```, import the requisite 2 libraries to work as ```construct``` and ```montecarlo```. 

MonteCarlo is a class object which is defined in ```construct.py``` as the Main code with ```montecarlo.py``` having all the Hamiltonian constructs as the GPU kernel (written in CUDA cpp).


## Construct MonteCarlo

```construct.MonteCarlo(config_file, input_folder, output_folder)``` is the MonteCarlo class construct. The base class for working. The ```input_folder``` and ```output_folder``` flags are defaulted to work if you run ```sample.py``` from inside its own directory. These can be set to any folder of choice.

```construct.MonteCarlo.load_material()``` loads the material parameters from the input file.

```construct.MonteCarlo.load_config()``` loads the config file.

```construct.MonteCarlo.load_materials()``` loads the multiple materials file.

```construct.MonteCarlo.load_config()``` loads the config

```construct.MonteCarlo.mc_init()``` initializes the simulation with the parameter file(but does not run it yet).

```construct.MonteCarlo.display_material()``` prints out the current material properties.

```construct.MonteCarlo.grid_reset()``` resets the grid to ALL $(0,0,1)$ if ```FM_Flag=1``` else randomizes all spins to an diamagnetic state.

```construct.MonteCarlo.generate_random_numbers(int size)``` creates 4 GPU allocated arrays of size ```size``` using the pyCUDA XORWOW random number generator.

```construct.MonteCarlo.generate_ising_numbers(int size)``` creates 4 GPU allocated arrays of size ```size``` using the pyCUDA XORWOW random number generator but the spin vectors are either $(0,0,1)$ or $(0,0,-1)$.

```construct.MonteCarlo.run_mc_dmi_66612(double T)``` runs a single ```Phase 1``` batch size run, with the return as a ```np.array(NxNx3)``` individual spin directions as raw output. This can be saved using the ```np.save``` command.

Other variations of ```run_mc_dmi_66612(T)``` are ```run_mc_<tc/dmi>_<mode>(T)```

```tc``` and ```dmi``` mode both contain modes for ```66612```, ```4448```, ```3636```, ```2424``` and ```2242```, which are the primary lattice types explored in this code. ```dmi``` can only be invoked by the configs ```66612, 4448, 3636```, for the rest, if you wish to open a running simulation, use ```tc``` mode with single temperature.

## Construct Analyze

The template file to run a visual analyzer is given in ```visualize.py```, this will compile images for all given state ```*.npy``` files in a given folder.

```construct.Analyze(<folder_path>, reverse=False)``` to create the Analyzer instance with an option to start from the opposite end (in case you want end results first)

```construct.Analyze.spin_view()``` creates a subfolder in the main folder called "spins" with the spin vector images inside. They are split into components as $z = s(x,y)$ as the functional form.

```construct.Ananlyze.quiver_view()``` creates a subfolder in the main folder called "quiver" with the spin vector images inside. They show only the planar(xy) part of the spins on a flat surface. This is useful for finding patterns in results.

# Theory

We consider a lattice system with a periodic arrangement of atoms, where each atom is represented by a 3D spin vector.  This atomistic spin model is founded on the spin Hamiltonian, which delineates the essential spin-dependent interactions at the atomic scale, excluding the influences of potential and kinetic energy and electron correlations. The spin Hamiltonian is conventionally articulated as:

## $$H_i=  -\sum_j J_1s_i\cdot s_j - \sum_j K^x_1 s^x_i s^x_j-\sum_j K^y_1 s^y_i s^y_j-\sum_j K^z_1 s^z_i s^z_j-\sum_k J_2 s_i\cdot s_k $$ 

## $$-\sum_k K^x_2 s^x_i s^x_k -\sum_k K^y_2 s^y_i s^y_k -\sum_k K^z_2 s^z_i s^z_k -\sum_l J_3s_i\cdot s_l  - \sum_l K^x_3 s^x_i s^x_l $$

## $$-\sum_l K^y_3 s^y_i s^y_l-\sum_l K^z_3 s^z_i s^z_l -\sum_m J_4s_i\cdot s_m - \sum_m K^x_4 s^x_i s^x_m-\sum_m K^y_4 s^y_i s^y_m  $$ 

## $$-\sum_m K^z_4 s^z_i s^z_m - A s_i \cdot s_i-\sum_j \lambda(s_i\cdot s_j)^2-\sum_j D_{ij}\cdot (s_i \times s_j) -\mu B \cdot s_i$$

Where $J$ is the isotropic exchange parameter, the $K$ s are the anisotropic exchange parameters, with the superscript denoting the spin direction, $A$ is the single ion exchange parameter, $\lambda$ is the biquadratic parameter, $D$ is the Dyzaloshinskii-Moriya Interaction(DMI) parameter. $\mu$ is the dipole moment of a single atom and $B$ is the external magnetic field. $s_i,s_j$ are individual atomic spin vectors. $\{s_j\}$ are the first set of neighbours, $\{s_k\}$ are the second set of neighbours and so on. The subscripts below all $J$s and $K$s denote the neighbour set, $J_1$ denotes the first neighbours, $J_2$ the second and so on. From this equation, we see that energy of an atom depends on its interactions with the neighbours. In our code, we have limited the number of neighbour sets to be 4 since it is expected for 2D materials that the interaction energy dies down beyond that. All these above parameters except $B$ are material specific parameters that are the inputs to our MC code.

# Results

First, we report the "meron" and "anti-meron", which are structures with topological numbers $-1/2$ and $+1/2$ respectively observed in CrCl<sub>3</sub> by LLG equations of spin dynamics. The results of our simulation are shown in Fig 2. CrCl<sub>3</sub> has a honeycomb lattice structure and for this simulation, we have considered up to the third nearest neighbour with biquadratic exchange. We can see all 4 types of meron and anti-meron structures as found in the previous report[@augustin_properties_2021]. This simulation was conducted in a $500 \times 500(143 \times 143 nm^2)$ supercell and took 300s to stabilize these topological spin textures at a parallelization of $3\%$ conducted on an A100-SXM4 processor.

Secondly, we simulate skyrmions in MnBr<sub>2</sub>[@acs_nanolett] as shown in Fig 2. MnBr<sub>2</sub> is a square lattice and for this simulation, we have considered up to the second nearest neighbour. This material exhibits anisotropic DMI with an anti-ferromagnetic ground state. An anti-ferromagnetic skyrmion spin texture is accurately reproduced in our simulation. Anti-ferromagnetic skyrmions are technologically important since they do not exhibit skyrmion Hall effect. We further study the material CrInSe<sub>3</sub> [@du_spontaneous_2022] which has a hexagonal lattice. This simulation was conducted considering only the nearest neighbours and the formation of skyrmions is shown in Fig 3. Once again our results are in agreement with the original report. All these simulations were conducted in a $200 \times 200(49 \times 49nm^2)$ supercell and took 30s to stabilize these topological spin textures at a parallelization of $20\%$ conducted on a V100-SXM2 processor.

In Fig 4 we demonstrate the skyrmion neucleation process for the material MnSTe [@liang_very_2020], which has a hexagonal lattice. While we first observe several skyrmions, with evolving MCS, they disappear and the whole lattice eventually becomes uniformly ferromagnetic,which happens to be the direction of the applied magnetic field. This has not been reported in the original literature[@liang_very_2020], possibly because of the high computational time required for a traditional SSU scheme. 

In Fig 5, we further show a similar life cycle evolution for a giant skyrmion of diameter $21nm$ hosted in the material VZr<sub>3</sub>C<sub>3</sub>II [@kabiraj_realizing_2023]. To host such a large skyrmion, the simulation was conducted in a supercell of size $750\times 750$ with a parallelization ratio of $1\%$ utilizing $70\%$ VRAM of an A100-SXM4 GPU. As mentioned before, our parallelization is limited by the number of CUDA cores and so we cannot go more than $1\%$ parallelization for this simulation. However, even with this low parallelization ratio, we can still access 8000 lattice points simultaneously and by careful tuning of our parameter $\Gamma$, we can observe the ground state of a $750\times750$ supercell in $9$ hours using an A100-SXM4 GPU. The formation of the skyrmion roughly takes $100$ mins.

|![Figure 1](figures/Figure_1.png)|
|:--:|
| *Fig 1:  Discrepancy between simulation and [reference](http://dx.doi.org/10.1038/s41524-020-00416-1) results at differing levels of parallelization. At $10\%$, the simulation results are almost indistinguishable from the reference data.* |

|![Figure 2](figures/Figure_2.png)|
|:--:|
| *Fig 2: Presence of anti-merons and merons in CrCl<sub>3</sub>. The color bar represents normalized spin vectors in the z direction.* |

|![Figure 3](figures/Figure_3.png)|
|:--:|
| *Fig 3: Presence of skyrmions in MnBr<sub>2</sub> and CrInSe<sub>3</sub>. The color bar represents normalized spin vectors in the z direction. Note that the spins of MnBr<sub>2</sub> appear purple because there are "red-blue" spin pairs for the vast majority.* |

|![Figure 4](figures/Figure_4.png)|
|:--:|
| *Fig 4: Lifetime of a skyrmion in MnSTe, from its creation to annihilation. The graph denotes the average energy per atom. As we approach the global minima, the entire field becomes aligned to the magnetic field as expected. Total time: 30s on a V100-SXM2.* |

|![Figure 5](figures/Figure_5.PNG)|
|:--:|
| *Fig 5: Lifetime of a skyrmion in VZr<sub>3</sub>C<sub>3</sub>II, from its creation to annihilation. The graph denotes the average energy per atom. Note how the entire field is now blue (as opposed to red as in Fig 4), this is because unlike the simulation in Fig 4, there is no external magnetic field applied, this means that the ground state would either be all spins up(red) or all spins down(blue) with a 50\% probability for either. Total time: 9 hrs on an A100-SXM4.* |


All these results and more are explored more in the attached ```JOSS Paper.md``` file, which forms the cover for the project.

# Contributions

Contributions to this project are always welcome and greatly appreciated. To find out how you can contribute to this project, please read our [contribution guidelines](https://github.com/arkavo/CUDA-METRO/blob/main/CONTRIBUTING.md)


# Code of Conduct

To read our code of conduct, please visit [CUDA-METRO Code of Conduct](https://github.com/arkavo/CUDA-METRO/blob/main/CODE_OF_CONDUCT.md).
