
---
title: 'CUDA-METRO: Parallel Metropolis Monte-Carlo for 2D Atomistic Spin Texture Simulation '
tags:
  -  Python
  -  Monte Carlo
  -  CUDA
  -  spintronics
  -  2D
  authors:
  -  name: Arkavo Hait
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1 
  -  name: Santanu Mahapatra
  - orcid: 0000-0000-0000-0000
    equal-contrib: true 
    affiliation: 1
  - affiliations:
  -  name: Indian Institute of Science
   index: 1
   ror: [04dese585](https://ror.org/04dese585)
  -  name: Indian Institute of Science, India
date: 30 Sept 2024
bibliography: references.bib
---

# Statement of need

Atomistic spin texture simulations are crucial for understanding and predicting the behaviour of magnetic materials at the nanoscale. These simulations provide insights into fundamental properties like magnetic phase transition and are thus useful for exploring novel materials (npj Computational Materials (2023) 9:173). The Metropolis Monte-Carlo method is frequently utilised for atomistic spin texture simulations as a sampling algorithm to investigate the phase space of a system and is especially effective for calculating equilibrium properties (J. Phys.: Condens. Matter 26, 103202 (2014), Physical Review B. 99, 224414 (2019)).
Efficient parallelization of Metropolis Monte Carlo simulation is challenging since the evolving states are typically not independent because of the Markov property. Here we focus on simulating magnetic phase transition under the anisotropic Heisenberg Model in a very high dimensional space, which is important for emerging two-dimensional (2D) magnetism and nontrivial topological spin textures (Nature Communications. 12, 185 (2021,1)). Previous attempts for parallelization are restricted to the simpler Ising Model and not applicable to 2D materials because of their finite magneto crystalline anisotropy, complex crystal structures and long-range interactions. MC simulation of the anisotropic Heisenberg model is very complex owing to the large number of additional Hamiltonian calculations and interconnectivity between lattice points. The amount of calculations increases as n², where n represents the dimension of a square lattice. This becomes alarming when n exceeds 100, which is entirely justifiable for investigating topological spin textures (skyrmions, merons, etc.). 
Here we present CUDA-METRO, a graphical processing unit (GPU) based open source code for accelerated atomistic spin dynamics simulation. Our code works on the hypothesis that a state can be approximately evolved by updating several connected lattice atoms’ spin vectors while ignoring their dependency on neighbours and ensuring zero race condition effects by enforcing a dynamic batch size. We evaluated our code by precisely simulating complex topological spin textures and temperature-dependent magnetic phase transitions for diverse 2D crystal structures with long-range magnetic interactions. We demonstrate exceptional acceleration while finding the ground state of a $750\times750$ supercell in 9 hours using an A100-SXM4 GPU.

# Summary
We consider a lattice system with a periodic arrangement of atoms, where each atom is represented by a 3D spin vector.  This atomistic spin model is founded on the spin Hamiltonian, which delineates the essential spin-dependent interactions at the atomic scale, excluding the influences of potential and kinetic energy and electron correlations. The spin Hamiltonian is conventionally articulated as

$$
H=-\sum Js_i\cdot s_j - \sum K_x s_i \cdot s_j-\sum K_y s_i \cdot s_j-\sum K_z s_i \cdot s_j-\sum A s_i \cdot s_i-\mu B \cdot \sum s_i
$$

Where $J$ is the isotropic exchange parameter, the $K$s are the anisotropic exchange parameters, $A$ is the single ion exchange parameter, $\mu$ is the dipole moment of a single atom and $B$ is the external magnetic field. $s_i,s_j$ are individual atomic spin vectors. $s_j$ are the first set of neighbours. We continue this calculation for however many neighbours sets as required. Since these energies are short range in nature, they typically die out in a span of 3-4 bond lengths(10-15$\textup{~\AA})$

Starting from a random spin configuration, in this many-body problem, our objective is to find the orientation of spin vectors for every atom so that the energy of the entire lattice reaches to its minimum for a given magnetic field and temperature. 
Traditionally single spin update (SSU) scheme is employed to solve this problem, which satisfies the detailed balance condition. Very briefly explain SSU with Metropolis scheme.  In the SSU method of updating the state, a single atomic spin is chosen at random and changed, while noting down the energy shift. This new state is then accepted or rejected using the Metropolis criteria. It is imperative that SSU becomes extremely inefficient as the dimensionality increases. We propose the following parallel algorithm to find the ground state, which is inaccessible by SSU scheme.

In our method, we select multiple atomic spins at the same time and change them all at once, treating them as independent events. For any individual spin, they do not feel the effects of the other changed spins. For each of these points, we then invoke the Metropolis criteria and put in the changed or otherwise new spins. This becomes our new state.

A basic schematic of the algorithm is given below:

![Figure 1](figures/Figure_1.PNG)

At present, five different lattice types  (square, rectangular, centred-rectangular, hexagonal and honeycomb) are implemented in our code since most of the 2D magnetic materials fall into this category (Ref: Patterns), and for neighbour mapping, we use analytical relations (Ref: IOP paper).

For a lattice of size $N\times N$, $100\%$ parallelization would correspond to selecting $N^2$ spins at random. Since each spin selection and its consequent Metropolis criterion is evaluated on a separate CUDA core, it becomes apparent that we would need $N^2$ CUDA cores to achieve this $100\%$ parallelization. In most cases, we avoid this altogether because of the sheer size of larger lattices and accuracy problems discussed in the next section.

Since the proposed algorithm may not adhere to the detailed balance conditions, it yields approximate results, and there is a trade-off between parallelization/acceleration and accuracy. It is found that if the parallelization is limited to $10\%$ of the lattice size, we obtain very accurate results with significant acceleration. We can see the effect of excess parallelization in $CrI_3$ below with respect to the standard results. However, the problem disappears at larger grid sizes where even modern hardware cannot achieve higher levels of parallelization leaving us with satisfactory results highlighted below. Along with these, the additional speed gives us the chance to observe the lifetime of microstructures in these materials, from their conception to destruction, as is also highlighted below. For each image, the overall grid size, the parallelization percentage (how much of the total number of atoms we are changing at one step), total time taken and temperature are given. External magnetic field is $0.0T$ unless stated otherwise.

Below, we demonstrate some of the results we get from ```CUDA-METRO``` with respect to the formation of spin microstructures in some materials. All these results replicate previously observed experimental phenomena. We have shown 2 broad types of spin structures in skyrmions(vortex) and merons(anti-vortex) in Fig 2, note the topological difference between the . In Fig 3, we have shown anti-skyrmions and compared them to skyrmions, anti-skyrmions are the analogue of skyrmions in anti-ferromagnetic materials. In Fig 4, we show the increasing deviation from expected results the more points we take in parallel. Finally, in Fig 5 we demonstrate the lifetime of a skyrmion, which also shows that they are mostly a "local minima of energy" phenomena, and die out when the material approaches its global energy minima.


![Figure 2](figures/Figure_2.png)
Fig 2: Presence of Skyrmions and Merons in $CrCl_3$. [@augustin_properties_2021]. The material parameters are taken from [@]. The color bar represents normalized spin vectors in the z direction.

![Figure 3](figures/Figure_3.png)
Fig 3: Presence of anti-skyrmions in $MnBr_2$ and skyrmions in $CrInSe_3$. [@], The material parameters are taken from [@]. The color bar represents normalized spin vectors in the z direction. Note that the spins of $MnBr_2$ appear purple because there are "red-blue" spin pairs for the vast majority.

![Figure 4](figures/Figure_4.png)
Fig 4: Discrepancy between experimantal and simuation results at differing levels of parallelization. Note how at $10\%$, the simulation results are almost indistinguishable from the reference data. The material parameters are taken from [@].

![Figure 5](figures/Figure_5.PNG)
Fig 5: Lifetime of a skyrmion, from its creation to annihilation. The graph denotes the average energy per atom. As we approach the global minima, the entire field becomes aligned to the magnetic field as expected.


# Acknowledgements
This work is supported by the Core Research Grant (CRG) scheme of the Science and Engineering Research
Board (SERB), Government of India, under Grant No. CRG/2020/000758.

# References

