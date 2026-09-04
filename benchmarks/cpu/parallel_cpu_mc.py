"""
Shared-memory PARALLEL CPU implementation of the synchronous multi-spin update.

Why this exists: the algorithm's claim is that removing the sequential
dependency makes the update embarrassingly parallel. That claim is about the
ALGORITHM, not about CUDA - so it should be demonstrable on any parallel
substrate. A CPU thread-scaling curve tests it directly, needs no GPU queue,
and is more legible to an algorithms audience than a CUDA occupancy argument.

What it can and cannot show - state this plainly, do not let it drift:
  CAN  - that the update parallelises, and how efficiently, as a function of
         thread count on a shared-memory machine.
  CANNOT - stand in for the GPU numbers. A CPU reaches tens of threads;
         CUDA-METRO runs P into the thousands. The CPU curve measures a
         different (much smaller) region of the same axis, and none of the
         GPU-specific effects (occupancy limits from the 2-threads-per-block
         launch geometry, coalescing, kernel-launch overhead) appear here.

Model: the same anisotropic Heisenberg loss used in the accuracy study -
    L(s) = -[ J1 sum_<ij> s_i.s_j + K1z sum_<ij> s_iz s_jz + Az sum_i s_iz^2 ]
on a periodic square lattice, spins of fixed magnitude S.

Structure of one round (this is the whole point):
  1. DECIDE - P chosen sites each evaluate their Metropolis acceptance against
     the SAME pre-round snapshot. No site's decision depends on any other's,
     so this loop is `prange` with zero synchronisation.
  2. COMMIT - accepted moves are written. Serial and cheap (O(accepted)), and
     last-write-wins resolves a repeated site exactly as the GPU's cp_grid does.
Only step 1 is parallel, which is also what bounds the achievable speedup.
"""
import numpy as np
from numba import njit, prange

KB = 8.6173e-2          # meV/K
SPIN, J1, K1Z, AZ = 1.5, 2.01, 0.1068, 0.10882


@njit(cache=True, inline="always")
def _local_loss(sx, sy, sz, nx, ny, nz, J1, K1z, Az):
    return -(J1 * (sx * nx + sy * ny + sz * nz) + K1z * sz * nz + Az * sz * sz)


@njit(cache=True)
def init_lattice(L, seed, fm, spin):
    np.random.seed(seed)
    S = np.empty((L, L, 3), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            if fm:
                S[i, j, 0] = 0.0; S[i, j, 1] = 0.0; S[i, j, 2] = spin
            else:
                u, v = np.random.random(), np.random.random()
                phi, th = 2.0 * np.pi * u, np.arccos(2.0 * v - 1.0)
                S[i, j, 0] = spin * np.sin(th) * np.cos(phi)
                S[i, j, 1] = spin * np.sin(th) * np.sin(phi)
                S[i, j, 2] = spin * np.cos(th)
    return S


@njit(cache=True, parallel=True)
def round_parallel(S, L, P, beta, spin, J1, K1z, Az,
                   rows, cols, newx, newy, newz, acc):
    """One synchronous round. The decision loop carries no cross-iteration
    dependency, so it is a plain prange - this is the algorithmic claim,
    made executable."""
    N = L * L
    # ---- DECIDE: parallel, every site reads the same pre-round snapshot ----
    for p in prange(P):
        site = np.random.randint(0, N)
        i = site // L
        j = site % L
        rows[p] = i
        cols[p] = j

        u, v = np.random.random(), np.random.random()
        phi, th = 2.0 * np.pi * u, np.arccos(2.0 * v - 1.0)
        st = np.sin(th)
        sx = spin * st * np.cos(phi)
        sy = spin * st * np.sin(phi)
        sz = spin * np.cos(th)
        newx[p] = sx; newy[p] = sy; newz[p] = sz

        nx = S[(i + 1) % L, j, 0] + S[(i - 1) % L, j, 0] + S[i, (j + 1) % L, 0] + S[i, (j - 1) % L, 0]
        ny = S[(i + 1) % L, j, 1] + S[(i - 1) % L, j, 1] + S[i, (j + 1) % L, 1] + S[i, (j - 1) % L, 1]
        nz = S[(i + 1) % L, j, 2] + S[(i - 1) % L, j, 2] + S[i, (j + 1) % L, 2] + S[i, (j - 1) % L, 2]

        e_old = _local_loss(S[i, j, 0], S[i, j, 1], S[i, j, 2], nx, ny, nz, J1, K1z, Az)
        e_new = _local_loss(sx, sy, sz, nx, ny, nz, J1, K1z, Az)
        dL = e_new - e_old
        acc[p] = 1 if (dL < 0.0 or np.random.random() < np.exp(-beta * dL)) else 0

    # ---- COMMIT: serial, last-write-wins for a repeated site (as cp_grid) ----
    for p in range(P):
        if acc[p] == 1:
            i = rows[p]; j = cols[p]
            S[i, j, 0] = newx[p]; S[i, j, 1] = newy[p]; S[i, j, 2] = newz[p]


@njit(cache=True)
def round_serial(S, L, P, beta, spin, J1, K1z, Az,
                 rows, cols, newx, newy, newz, acc):
    """Identical round with the decision loop kept sequential - the baseline
    the speedup is measured against."""
    N = L * L
    for p in range(P):
        site = np.random.randint(0, N)
        i = site // L
        j = site % L
        rows[p] = i
        cols[p] = j
        u, v = np.random.random(), np.random.random()
        phi, th = 2.0 * np.pi * u, np.arccos(2.0 * v - 1.0)
        st = np.sin(th)
        sx = spin * st * np.cos(phi)
        sy = spin * st * np.sin(phi)
        sz = spin * np.cos(th)
        newx[p] = sx; newy[p] = sy; newz[p] = sz
        nx = S[(i + 1) % L, j, 0] + S[(i - 1) % L, j, 0] + S[i, (j + 1) % L, 0] + S[i, (j - 1) % L, 0]
        ny = S[(i + 1) % L, j, 1] + S[(i - 1) % L, j, 1] + S[i, (j + 1) % L, 1] + S[i, (j - 1) % L, 1]
        nz = S[(i + 1) % L, j, 2] + S[(i - 1) % L, j, 2] + S[i, (j + 1) % L, 2] + S[i, (j - 1) % L, 2]
        e_old = _local_loss(S[i, j, 0], S[i, j, 1], S[i, j, 2], nx, ny, nz, J1, K1z, Az)
        e_new = _local_loss(sx, sy, sz, nx, ny, nz, J1, K1z, Az)
        dL = e_new - e_old
        acc[p] = 1 if (dL < 0.0 or np.random.random() < np.exp(-beta * dL)) else 0
    for p in range(P):
        if acc[p] == 1:
            i = rows[p]; j = cols[p]
            S[i, j, 0] = newx[p]; S[i, j, 1] = newy[p]; S[i, j, 2] = newz[p]


@njit(cache=True)
def magnetization(S, L):
    mx = my = mz = 0.0
    for i in range(L):
        for j in range(L):
            mx += S[i, j, 0]; my += S[i, j, 1]; mz += S[i, j, 2]
    n = L * L
    return np.sqrt(mx * mx + my * my + mz * mz) / n


def scratch(P):
    return (np.empty(P, np.int64), np.empty(P, np.int64),
            np.empty(P, np.float64), np.empty(P, np.float64),
            np.empty(P, np.float64), np.empty(P, np.uint8))


def run(L, T, P, rounds, seed, parallel=True, fm=True, spin=SPIN,
        j1=J1, k1z=K1Z, az=AZ, sample_every=0):
    """Run `rounds` synchronous rounds; returns (final lattice, M samples)."""
    beta = 1.0 / (T * KB)
    S = init_lattice(L, seed, fm, spin)
    buf = scratch(P)
    step = round_parallel if parallel else round_serial
    out = []
    for t in range(rounds):
        step(S, L, P, beta, spin, j1, k1z, az, *buf)
        if sample_every and t % sample_every == 0:
            out.append(magnetization(S, L))
    return S, np.array(out)
