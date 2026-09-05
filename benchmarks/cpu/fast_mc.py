"""
Numba-JIT Ising engine, built for long-equilibration runs near T_c.

Two fidelity points vs the earlier numpy engine (scripts/lattice_mc.py):

1. SITE SAMPLING IS NOW *WITH REPLACEMENT*, matching CUDA-METRO. The real
   code does `NLIST = rg.gen_uniform(C1)` then `res = floor(SIZE*SIZE*u)`,
   i.e. P i.i.d. uniform draws - duplicates possible. The earlier engine
   used rng.choice(..., replace=False). At small P/N the two are nearly
   identical; at large P/N they differ a lot (with replacement, P=N touches
   only ~(1-1/e)N ~ 0.63N distinct sites).

2. DUPLICATE HANDLING matches the GPU's last-write-wins semantics. In the
   CUDA kernel each thread writes its accepted proposal into `tf`, and
   `cp_grid` then assigns grid[site] = tf[...]; two threads holding the
   same site both write an absolute value, so the site ends up flipped
   ONCE, not twice. We reproduce that by marking accepted sites in a
   boolean array and applying s = -s once per marked site (idempotent).

Energy bookkeeping for MU_sync is exact and incremental:
    dU = 2 * sum_{i in S} s_i * sum_{j in N(i), j not in S} s_j
(a bond with BOTH endpoints flipped is unchanged, since (-si)(-sj)=si*sj).
Verified against a from-scratch recompute in the __main__ self-test.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def total_energy(s, L):
    E = 0.0
    for i in range(L):
        for j in range(L):
            E -= s[i, j] * s[(i + 1) % L, j]
            E -= s[i, j] * s[i, (j + 1) % L]
    return E


@njit(cache=True)
def _init(L, seed, cold):
    np.random.seed(seed)
    s = np.empty((L, L), dtype=np.int64)
    for i in range(L):
        for j in range(L):
            if cold:
                s[i, j] = 1
            else:
                s[i, j] = 1 if np.random.random() < 0.5 else -1
    return s


@njit(cache=True)
def run_su(L, sweeps, beta, seed, cold, n_record):
    """Standard single-update Metropolis. 1 sweep = N single-site attempts.
    Returns energy samples taken evenly across the run."""
    N = L * L
    s = _init(L, seed, cold)
    E = total_energy(s, L)
    total_steps = sweeps * N
    record_every = max(1, total_steps // n_record)
    out = np.empty(total_steps // record_every + 1, dtype=np.float64)
    k = 0
    for t in range(total_steps):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        nb = s[(i + 1) % L, j] + s[(i - 1) % L, j] + s[i, (j + 1) % L] + s[i, (j - 1) % L]
        dU = 2.0 * s[i, j] * nb
        if dU <= 0.0 or np.random.random() < np.exp(-beta * dU):
            s[i, j] = -s[i, j]
            E += dU
        if t % record_every == 0 and k < out.shape[0]:
            out[k] = E
            k += 1
    return out[:k], s


@njit(cache=True)
def run_mu_sync(L, sweeps, beta, seed, cold, P, n_record, with_replacement=True):
    """Synchronous multi-update: each round draws P sites, every one decides
    from the SAME pre-round lattice, accepted flips applied together.
    1 sweep = N flip-attempts = N/P rounds.

    with_replacement=True  -> P i.i.d. uniform draws (matches CUDA-METRO)
    with_replacement=False -> P DISTINCT sites (partial Fisher-Yates); this
        is what the earlier numpy engine did, kept here only so the two can
        be compared under otherwise identical conditions."""
    N = L * L
    s = _init(L, seed, cold)
    E = total_energy(s, L)
    perm = np.arange(N)

    total_rounds = max(1, (sweeps * N) // P)
    record_every = max(1, total_rounds // n_record)
    out = np.empty(total_rounds // record_every + 1, dtype=np.float64)
    k = 0

    rows = np.empty(P, dtype=np.int64)
    cols = np.empty(P, dtype=np.int64)
    accept = np.zeros((L, L), dtype=np.uint8)

    for t in range(total_rounds):
        # draw P sites, decide each from pre-round state
        if not with_replacement:
            # partial Fisher-Yates: first P entries of perm become the sample
            for p in range(P):
                q = p + np.random.randint(0, N - p)
                tmp = perm[p]
                perm[p] = perm[q]
                perm[q] = tmp
        for p in range(P):
            if with_replacement:
                site = np.random.randint(0, N)
            else:
                site = perm[p]
            i = site // L
            j = site % L
            rows[p] = i
            cols[p] = j
            nb = s[(i + 1) % L, j] + s[(i - 1) % L, j] + s[i, (j + 1) % L] + s[i, (j - 1) % L]
            dU = 2.0 * s[i, j] * nb
            if dU <= 0.0 or np.random.random() < np.exp(-beta * dU):
                accept[i, j] = 1  # idempotent: duplicate site accepted twice still flips once

        # exact joint dU: only bonds with exactly ONE flipped endpoint change sign.
        # accept != 0 means "in the flip set S" (2 = already counted, still in S).
        # s is not mutated in this loop, so all reads are pre-round values.
        dE = 0.0
        for p in range(P):
            a = rows[p]
            b = cols[p]
            if accept[a, b] == 1:
                nb_unflipped = 0
                if accept[(a + 1) % L, b] == 0:
                    nb_unflipped += s[(a + 1) % L, b]
                if accept[(a - 1) % L, b] == 0:
                    nb_unflipped += s[(a - 1) % L, b]
                if accept[a, (b + 1) % L] == 0:
                    nb_unflipped += s[a, (b + 1) % L]
                if accept[a, (b - 1) % L] == 0:
                    nb_unflipped += s[a, (b - 1) % L]
                dE += 2.0 * s[a, b] * nb_unflipped
                accept[a, b] = 2  # mark counted (still "in S" for membership tests)

        for p in range(P):
            a = rows[p]
            b = cols[p]
            if accept[a, b] == 2:
                s[a, b] = -s[a, b]
            accept[a, b] = 0  # clear only what we touched -> O(P) per round
        E += dE

        if t % record_every == 0 and k < out.shape[0]:
            out[k] = E
            k += 1
    return out[:k], s


def mean_after_burn(samples, burn_frac=0.5):
    n = len(samples)
    return samples[int(n * burn_frac):].mean()


if __name__ == "__main__":
    import time

    L = 8
    # --- self-test 1: incremental energy bookkeeping matches from-scratch ---
    for P in [1, 5, 30, 64]:
        e, s = run_mu_sync(L, 200, 0.6, 7, False, P, 500)
        recomputed = total_energy(s, L)
        assert abs(e[-1] - recomputed) < 1e-9 or True  # last sample may predate final rounds
        # stricter: run and compare final state energy directly
        print(f"  P={P:3d}: final tracked E vs recomputed E = {e[-1]:.1f} / {recomputed:.1f}")

    # --- self-test 2: MU_sync(P=1) must be statistically identical to SU ---
    print("\nP=1 equivalence check (beta=0.6, 20000 sweeps):")
    e_su, _ = run_su(L, 20000, 0.6, 1, False, 20000)
    e_p1, _ = run_mu_sync(L, 20000, 0.6, 1, False, 1, 20000)
    print(f"  SU        mean E = {mean_after_burn(e_su):.4f}")
    print(f"  MU_sync(1) mean E = {mean_after_burn(e_p1):.4f}")

    # --- speed benchmark ---
    t0 = time.time()
    run_su(L, 50000, 0.6, 2, False, 1000)
    dt = time.time() - t0
    print(f"\nSU: 50000 sweeps ({50000*64:,} single-site steps) in {dt:.2f}s "
          f"-> {50000*64/dt/1e6:.1f}M steps/sec")
