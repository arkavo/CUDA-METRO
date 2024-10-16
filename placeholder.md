
\begin{algorithm}[H]
    \caption{Metropolis Selection}
    \label{algorithm:MS}
    \begin{algorithmic}[0]
        \Procedure{M}{$H_f,H_i$}
            \If {$\Delta H < 0$}
            \State \texttt{Return 1 (ACCEPT)}
            \ElsIf {$e^{\beta \Delta H} < R$}\Comment{$R$ is uniformly random}
            \State \texttt{Return 1 (ACCEPT)}
            \Else
            \State \texttt{Return 0 (REJECT)}
            \EndIf
        \EndProcedure
    \end{algorithmic}
\end{algorithm}


\begin{algorithm}[t]
    \caption{Parallel Monte Carlo}
    \label{algorithm:step}
    \begin{algorithmic}[0]
        \Procedure{Step}
        \hspace*{4.5em}
        \State \hspace*{4.5em}{Read State $\Omega_i$}
        \State \hspace*{4.5em}{Create 4 $P\times B$ length uniform random arrays}
        \State \hspace*{4.5em}{Process 4 arrays into $N,\theta, \phi, R$}
        \For{\hspace*{4.5em}{$i<B$}}
            \State \hspace*{4.5em}{Create 4 sub-arrays as $(N,\theta,\phi,R)[P\times i:P\times (i+1)-1]$}
            \State \hspace*{4.5em}{Execute $P$ parallel BLOCKS with sub array $(N,\theta,\phi,R)[j]$}\Comment{$j\in [P\times i,P\times (i+1)]$}
            \For{In each BLOCK}
                \State \hspace*{4.5em}{Evaluate $H$ before(T0) and after(T1) spin change}\Comment{Multithreading}
                \State \hspace*{4.5em}{Select spins according to $S_{new} = S_f(M(H_f,H_i)) + S_i(1-M(H_f,H_i))$}
                \State \hspace*{4.5em}{Wait for all BLOCKS to finish}
            \EndFor
            \State \hspace*{4.5em}{Update all $P$ spins to state}
            \State \hspace*{4.5em}{$\Omega_{i+1} \leftarrow \Omega_{i}$}
        \EndFor
        \EndProcedure
    \end{algorithmic}
\end{algorithm}[t]

header-includes: |
  - \usepackage{algorithm}
  - \usepackage[noend]{algpseudocode}
  - \usepackage{chemformula}


---
title: 'CUDA-METRO: Parallel Metropolis Monte-Carlo for 2D Atomistic Spin Texture Simulation'
tags:
  -  Python
  -  Monte Carlo
  -  CUDA
  -  spintronics
  -  2D
authors:
  - name: Arkavo Hait
    orcid: 0009-0006-6741-9377
    equal-contrib: true
    affiliation: 1
  - name: Santanu Mahapatra
    orcid: 0000-0003-1112-8109
    equal-contrib: true
    affiliation: 1
affiliations:
  - name: Indian Institute of Science
    index: 1
    ror: [04dese585](https://ror.org/04dese585)
header-includes:
  - \usepackage{algorithm}
  - \usepackage[noend]{algpseudocode}
  - \usepackage{chemformula}
output:
  pdf_document
date: 30 Sept 2024
bibliography: references.bib

---