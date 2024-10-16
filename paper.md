---
title: 'CUDA-METRO: Parallel Metropolis Monte-Carlo for 2D Atomistic Spin Texture Simulation'
tags:
  - Python
  - Monte Carlo
  - 2D
authors:
  - name: Arkavo Hait
    orcid: 0009-0006-6741-9377
    affiliation: 1
affiliations:
 - name: Indian Institute of Science
   index: 1
header-includes:
 - \usepackage{algorithm}
 - \usepackage[noend]{algpseudocode}
 - \usepackage[utf8]{inputenc}
date: 30 September 2024
bibliography: references.bib

---

# Statement of need

Atomistic spin texture simulations are crucial for understanding and predicting the behaviour of magnetic materials at the nanoscale. These simulations provide insights into fundamental properties like magnetic phase transition and are thus useful for exploring novel materials [@kabiraj_realizing_2023]. The Metropolis[@metropolis_equation_1953] Monte-Carlo[@heisenberg_zur_1928] (MC) method is frequently utilised for atomistic spin texture simulations as a sampling algorithm to investigate the phase space of a system and is especially effective for calculating equilibrium properties [@evans_atomistic_2014;@PhysRevB.99.224414].
Efficient parallelization of Metropolis Monte Carlo simulation is challenging since the evolving states are typically not independent because of the Markov property. Here we focus on simulating magnetic phase transition under the anisotropic Heisenberg Model in a very high dimensional space, which is important for emerging two-dimensional (2D) magnetism and nontrivial topological spin textures [@augustin_properties_2021]. Previous attempts for parallelization are restricted to the simpler Ising Model and not applicable to 2D materials because of their finite magneto crystalline anisotropy, complex crystal structures and long-range interactions. MC simulation of the anisotropic Heisenberg model is very complex owing to the large number of additional Hamiltonian calculations and interconnectivity between lattice points. The amount of calculations increases as $N^2$, where $N$ represents the dimension of a square lattice. This becomes alarming when $N$ exceeds 100, which is entirely justifiable for investigating topological spin textures (skyrmions, merons, etc.)
Here we present CUDA-METRO, a graphical processing unit (GPU) based open source code for accelerated atomistic spin dynamics simulation. We evaluated our code by precisely simulating complex topological spin textures and temperature-dependent magnetic phase transitions for diverse 2D crystal structures with long-range magnetic interactions. We demonstrate exceptional acceleration while finding the ground state of a $750\times750$ supercell in 9 hours using an A100-SXM4 GPU.

# Summary
We consider a lattice system with a periodic arrangement of atoms, where each atom is represented by a 3D spin vector. This atomistic spin model is founded on the spin Hamiltonian, which delineates the essential spin-dependent interactions at the atomic scale, excluding the influences of potential and kinetic energy and electron correlations. The spin Hamiltonian of the $i^{th}$ atom is conventionally articulated as

Where $J$ is the isotropic exchange parameter, the $K$s are the anisotropic exchange parameters, with the superscript denoting the spin direction, $A$ is the single ion exchange parameter, $\lambda$ is the biquadratic parameter, $D$ is the Dyzaloshinskii-Moriya Interaction(DMI) parameter. $\mu$ is the dipole moment of a single atom and $B$ is the external magnetic field. $s_i,s_j$ are individual atomic spin vectors. $\{s_j\}$ are the first set of neighbours, $\{s_k\}$ are the second set of neighbours and so on. The subscripts below all $J$s and $K$s denote the neighbour set, $J_1$ denotes the first neighbours, $J_2$ the second and so on. From this equation, we see that energy of an atom depends on its interactions with the neighbours. In our code, we have limited the number of neighbour sets to be 4 since it is expected for 2D materials that the interaction energy dies down beyond that.

Starting from a random spin configuration, in this many-body problem, our objective is to find the orientation of spin vectors for every atom so that the energy of the entire lattice reaches to its minimum for a given magnetic field and temperature.
Traditionally single spin update(SSU) scheme is employed to solve this problem, which satisfies the detailed balance condition. In the SSU method of updating the state, a single atomic spin is chosen at random and changed, while noting down the energy shift. This new state is then accepted or rejected using the Metropolis criteria as shown in Algorithm 1. It is imperative that SSU becomes extremely inefficient as the dimensionality increases. We propose the following parallel algorithm to find the ground state, which is inaccessible by SSU scheme.