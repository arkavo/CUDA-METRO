
---
title: 'CUDA-METRO: A Python package for Heisenberg '
tags:
  -  Python
  -  Monte Carlo
  -  CUDA
  - spintronics
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
bibliography: paper.bib
---

# Summary
Efficient parallelization of Metropolis Monte Carlo simulation is challenging since the evolving states are typically
not independent because of the Markov property. Here we focus on simulating magnetic phase transition under the
anisotropic Heisenberg Model in a very high dimensional space, which is important for emerging two-dimensional
(2D) magnetism and nontrivial topological spin textures. Previous attempts for parallelization are restricted to the
simpler Ising Model and not applicable to 2D materials because of their finite magnetocrystalline anisotropy, complex
crystal structures and long-range interactions.

# Statement of need
`CUDA-METRO` is a python based tool to simulate and find out the ground states of a given 2D lattice of a number of crystallline structures, beginning with a given starting state. We wrap `CUDA C++` inside python for access to low level code. The kernel is written in `CUDA C++` while the data I/O is handled with `pyCUDA`. We additionally handle the large amount of random numbers needed with `pycuda.curandom.XORWOWRandomNumberGenerator()` which we then post-process according to our needs.
`CUDA-METRO`, being a Metropolis Monte Carlo simulation for any 2D matrix problem, can be used in different ways both to simulate materials and to other matrix based problems. This tool was primarily created to harness the huge computational power of GPUs(Graphics Processing Units) and their efficiency in evaluating matrix based computations. This tool provides both computation speed and capacity(with respect to grid sizes) for simulating such matrices/materials.

# Mathematics
According to the laws of thermodynamics, two different states with energies $E_1$ and $E_2$ will have relative probabilities of existence as 
$$\frac{p(E_2)}{p(E_1)}=\frac{e^{-\beta E_2}}{e^{-\beta E_1}}$$
where $\beta=(k_bT)^{-1}$, $k_b$ being the Boltzmann constant and $T$ being the temperature. The energy of the Heisenberg model is calculated as 
$$H=-\sum Js_i\cdot s_j - \sum K_x s_i \cdot s_j-\sum K_y s_i \cdot s_j-\sum K_z s_i \cdot s_j-\sum A s_i \cdot s_i-\mu B \cdot \sum s_i$$
where $s_j$ are the first set of neighbours. We continue this calculation for however many neighbours sets as required. Since these energies are short range in nature, they typically die out in a span of 3-4 bond lengths(10-15$\textup{~\AA}$
The Metropolis Monte Carlo works by randomly instantiating a state$(\Omega_1)$ close to a starting state$(\Omega_0)$ and then comparing the energies of the starting and the modified state. If the energy of this new state$(\Omega_1)$ is lower than the original state$(\Omega_0)$, then this new state$(\Omega_1)$ becomes our next $\Omega_0$, while if the energy of $\Omega_1$ is higher than $\Omega_0$, then we choose $\Omega_1$ with a probability of $e^{-\beta \Delta E}$ where $\Delta E=E(\Omega_1)-E(\Omega_0)$, the difference between the energies of 2 states. The Monte Carlo then proceeds to the next stage with the new state.

# Acknowledgements
This work is supported by the Core Research Grant (CRG) scheme of the Science and Engineering Research
Board (SERB), Government of India, under Grant No. CRG/2020/000758.
# References

