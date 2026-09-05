"""
Does P*/N stay size-independent AT CRITICALITY?  (many-core version)

THE QUESTION
    The speed argument ends with: at production lattice sizes the hardware
    ceiling (P* = L.T, independent of N) binds long before the accuracy budget
    (P <= (1-A)/0.0276 * N, proportional to N). The soft spot is that 0.0276
    was fitted in the ORDERED phase. Near T_c the usable fraction collapses -
    at L=8 an earlier sweep found P*/N ~ 0.12 at beta=0.40 against ~0.74 deep
    in the ordered phase.

    If P*/N ~ 0.11 also holds at L=2048, then P* ~ 0.11 * 4.19e6 ~ 460,000 -
    still tens of times above what any GPU here can use, and "hardware binds
    first" survives even at the worst temperature. But that is an extrapolation
    from small lattices, and critical fluctuations are exactly where a
    size-independent ratio is least obvious. So: measure P*/N at beta = 0.40
    and beta_c = 0.4407 across L = 8..48 and see whether the ratio is flat.

WHY THIS FILE EXISTS SEPARATELY
    The single-threaded version is CPU-bound for hours at L >= 24 (critical
    slowing down: sweeps ~ L^2.17, work per sweep ~ L^2, so cost ~ L^4.17).
    Every (beta, L, P, seed) chain is completely independent, so this version
    hands them to a process pool. On a 64-core node it finishes in minutes.

    It also times itself, so the same run reports CPU spin-update throughput
    in ns/update - directly comparable to the GPU table.

USAGE
    python critical_scaling_mp.py                       # all cores
    python critical_scaling_mp.py --jobs 64
    python critical_scaling_mp.py --sizes 8,12,16,24,32,48 --betas 0.40,0.4407
    python critical_scaling_mp.py --tau-mult 400        # cheaper, still checked

    Resumable: rerun after any interruption and finished points are skipped.
    Results append to critical_size_scaling.csv.

METHOD
    Equilibration: tau ~ L^z, z ~ 2.17, so sweeps scale as (L/8)^2.17 from a
    base of TAU_MULT tau at L=8. The P=1 column is the CONTROL - MU_sync(1) is
    mathematically identical to single-update Metropolis, so E_P - E_SU must be
    consistent with zero. It is judged against the combined standard error, not
    a fixed accuracy cutoff: a raw "acc > 0.999" test flags a perfectly
    equilibrated small lattice as broken, because the 2N normaliser is small
    there while the statistical error is not.

    Metric: bond-normalised, acc = 1 - |E_P - E_SU| / (2N), i.e. error as a
    fraction of the full energy scale. Temperature-stable. The relative metric
    1 - |dE|/|E_SU| is also recorded, but |E_SU| shrinks toward zero as T rises
    so the same absolute error looks worse at high T - a metric artefact.
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=0, help="0 = all cores")
ap.add_argument("--sizes", default="8,12,16,24,32")
ap.add_argument("--betas", default="0.40,0.4407")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--tau-mult", type=int, default=800,
                help="sweeps = TAU_MULT * tau(L); 800 is generous, 400 is "
                     "usually plenty and the control will tell you")
ap.add_argument("--tau8", type=float, default=100.0)
ap.add_argument("--z", type=float, default=2.17)
ap.add_argument("--out", default="critical_size_scaling.csv")
ap.add_argument("--report-only", action="store_true",
                help="print the full analysis from the CSV and exit")
args = ap.parse_args()

SIZES = [int(x) for x in args.sizes.split(",")]
BETAS = [float(x) for x in args.betas.split(",")]
SEEDS = list(range(1, args.seeds + 1))
RATIOS = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
RATIOS_LARGE = [0.06, 0.08, 0.10, 0.12, 0.15]
LARGE_FROM = 24
N_REC = 4000


def sweeps_for(L):
    return int(max(80_000, args.tau_mult * args.tau8 * (L / 8.0) ** args.z))


def p_list(L):
    N = L * L
    ratios = RATIOS_LARGE if L >= LARGE_FROM else RATIOS
    out, seen = [], set()
    for P in [1] + [max(1, int(round(r * N))) for r in ratios]:
        if P not in seen:            # at small N, ratio*N can round down to 1
            seen.add(P)
            out.append(P)
    return out


# GPU ceilings measured on the same algorithm, for the extrapolation below
GPU_CEILINGS = [("Tesla V100-SXM2-32GB", 8192), ("RTX PRO 5000 Blackwell", 12600)]
ORDERED_PSTAR_N = 0.74      # Ising, beta=0.6, deep in the ordered phase


def p_star(pairs, target=0.95):
    """Interpolate P/N where bond-normalised accuracy crosses `target`."""
    v = sorted(pairs)
    xs, ys = [p for p, _ in v], [a for _, a in v]
    for i in range(1, len(ys)):
        if ys[i] < target <= ys[i - 1]:
            t = (ys[i - 1] - target) / (ys[i - 1] - ys[i])
            return xs[i - 1] + t * (xs[i] - xs[i - 1]), False
    return (max(xs), True) if min(ys) >= target else (None, False)


def report(path):
    """The end-of-run summary, verbose enough to paste straight out."""
    if not os.path.exists(path):
        print(f"no results at {path}")
        return
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print(f"{path} is empty")
        return
    by = {}
    for r in rows:
        k = (round(float(r["beta"]), 4), int(r["L"]))
        by.setdefault(k, []).append(r)

    print("=" * 74)
    print(f"CRITICAL ACCURACY SCALING - {len(rows)} points from {path}")
    print("=" * 74)

    stars = {}
    for (beta, L), rs in sorted(by.items()):
        N = L * L
        rs.sort(key=lambda r: int(r["P"]))
        S = int(rs[0]["sweeps"])
        E_su, sem = float(rs[0]["E_su"]), float(rs[0]["E_su_sem"])
        print(f"\nbeta = {beta:.4f}   L = {L}   N = {N:,}   "
              f"{S:,} sweeps/chain   {rs[0]['seeds']} seeds")
        print(f"  single-update reference  E_SU = {E_su:.3f} +- {sem:.3f}")
        for r in rs:
            P, pn = int(r["P"]), float(r["P_over_N"])
            ab, ar = float(r["acc_bond"]), float(r["acc_rel"])
            flag = ""
            if P == 1:
                sig = (sem ** 2 + float(r["E_p_sem"]) ** 2) ** 0.5
                z = abs(float(r["E_p"]) - E_su) / sig if sig > 0 else 0.0
                flag = (f"   <- CONTROL ok (z={z:.1f})" if z < 3
                        else f"   <- CONTROL FAILED (z={z:.1f})")
            print(f"    P={P:7d}  P/N={pn:7.4f}   E={float(r['E_p']):11.3f}   "
                  f"acc_bond={ab:7.4f}  acc_rel={ar:7.4f}{flag}")
        st, cens = p_star([(float(r["P_over_N"]), float(r["acc_bond"])) for r in rs])
        if st is None:
            print("    -> P*/N: accuracy never reaches 95% in the sampled range")
        elif cens:
            print(f"    -> P*/N NOT REACHED: still >=95% at the largest P/N "
                  f"tried ({st:.4f}) - censored, not a limit")
        else:
            stars[(beta, L)] = st
            print(f"    -> P*/N = {st:.4f}   (P* = {st*N:,.0f} at this size)")

    print("\n" + "=" * 74)
    print("SIZE-INDEPENDENCE OF P*/N")
    print("=" * 74)
    for beta in sorted({b for b, _ in stars}):
        pts = sorted((L, v) for (b, L), v in stars.items() if b == beta)
        if not pts:
            continue
        vals = [v for _, v in pts]
        print(f"\n  beta = {beta:.4f}")
        for L, v in pts:
            print(f"    L={L:3d}  N={L*L:8,}   P*/N = {v:.4f}")
        spread = (max(vals) - min(vals)) / np.mean(vals)
        print(f"    spread across sizes: {spread:.1%}")
        if len(pts) >= 3:
            drift = (pts[-1][1] - pts[-2][1]) / pts[-2][1]
            print(f"    drift over the last two sizes: {drift:+.1%}")
            verdict = ("FLAT - the ratio is size-independent, extrapolation justified"
                       if abs(drift) < 0.05 else
                       "STILL DRIFTING - quote only at the sizes measured")
            print(f"    verdict: {verdict}")
        big = np.mean(vals[-2:]) if len(vals) >= 2 else vals[-1]
        print(f"\n    taking P*/N = {big:.4f} (mean of the two largest sizes):")
        print(f"    vs ordered phase (beta=0.6, P*/N ~ {ORDERED_PSTAR_N}): "
              f"criticality costs {ORDERED_PSTAR_N/big:.1f}x")
        print(f"\n    {'lattice':>10} {'N':>12} {'P* at criticality':>19}   "
              f"binds first?")
        for L in (512, 750, 1024, 2048, 4096):
            N = L * L
            ps = big * N
            worst = min(c for _, c in GPU_CEILINGS)
            best = max(c for _, c in GPU_CEILINGS)
            print(f"    {L:>10} {N:>12,} {ps:>19,.0f}   hardware, by "
                  f"{ps/best:.0f}-{ps/worst:.0f}x")
        print(f"    (GPU ceilings: " +
              ", ".join(f"{g} P*={c:,}" for g, c in GPU_CEILINGS) + ")")
    print()


def chain(task):
    """One independent chain. Imported inside so each worker JITs/loads once."""
    from fast_mc import run_su, run_mu_sync, mean_after_burn
    kind, L, S, beta, seed, P = task
    t0 = time.perf_counter()
    if kind == "su":
        v = mean_after_burn(run_su(L, S, beta, seed, False, N_REC)[0])
    else:
        v = mean_after_burn(run_mu_sync(L, S, beta, seed, False, P, N_REC, True)[0])
    return task, float(v), time.perf_counter() - t0, S * L * L


def main():
    if args.report_only:
        report(args.out)
        return
    jobs = args.jobs or (os.cpu_count() or 1)
    print(f"cores: {os.cpu_count()}   workers: {jobs}")
    print(f"sizes {SIZES}   betas {BETAS}   seeds {len(SEEDS)}   "
          f"tau_mult {args.tau_mult}")
    for L in SIZES:
        print(f"  L={L:3d}  N={L*L:6d}  sweeps={sweeps_for(L):,}  "
              f"P values: {len(p_list(L))}")

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for r in csv.DictReader(f):
                try:
                    done.add((round(float(r["beta"]), 4), int(r["L"]), int(r["P"])))
                except (KeyError, ValueError):
                    continue         # tolerate a truncated final row
        print(f"resume: {len(done)} points already measured")

    # Build every chain up front, then run them all in one pool. Batching by
    # (beta, L) would idle most cores on the last few chains of each block.
    tasks, need_su = [], set()
    for beta in BETAS:
        for L in SIZES:
            todo = [P for P in p_list(L) if (round(beta, 4), L, P) not in done]
            if not todo:
                continue
            need_su.add((beta, L))
            S = sweeps_for(L)
            for sd in SEEDS:
                tasks.append(("su", L, S, beta, sd, 0))
            for P in todo:
                for sd in SEEDS:
                    tasks.append(("mu", L, S, beta, sd, P))
    tasks = list(dict.fromkeys(tasks))
    if not tasks:
        print("nothing to do - every point is already measured.")
        return
    # longest chains first: a 40-minute L=48 chain started last leaves 63 cores
    # idle waiting for it
    tasks.sort(key=lambda t: -(t[1] ** 2 * t[2]))
    total_updates = sum(t[2] * t[1] ** 2 for t in tasks)
    print(f"\n{len(tasks)} chains, {total_updates/1e9:.1f} G spin-updates total")

    t_start = time.perf_counter()
    res, wall, upd = {}, 0.0, 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, (task, val, dt, n) in enumerate(ex.map(chain, tasks), 1):
            res.setdefault(task[:5] if task[0] == "su" else task[:6], []).append(val)
            res.setdefault((task[0], task[1], task[3], task[5]), []).append(val)
            wall += dt
            upd += n
            if i % 20 == 0 or i == len(tasks):
                el = time.perf_counter() - t_start
                print(f"  {i}/{len(tasks)} chains   {el/60:5.1f} min elapsed   "
                      f"{upd/el/1e6:7.2f} M spin-updates/s aggregate", flush=True)

    new = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["beta", "L", "N", "P", "P_over_N", "sweeps", "seeds",
                    "E_su", "E_su_sem", "E_p", "E_p_sem", "acc_bond", "acc_rel"])

    print()
    for beta, L in sorted(need_su):
        N, S = L * L, sweeps_for(L)
        su = res[("su", L, beta, 0)]
        E_su, su_sem = float(np.mean(su)), float(np.std(su) / np.sqrt(len(su)))
        print(f"beta={beta:.4f} L={L:3d} N={N:6d} S={S:,}  E_su={E_su:10.3f} "
              f"+- {su_sem:.3f}")
        for P in p_list(L):
            key = ("mu", L, beta, P)
            if key not in res:
                continue
            m = res[key]
            E_p, p_sem = float(np.mean(m)), float(np.std(m) / np.sqrt(len(m)))
            acc_bond = 1 - abs(E_p - E_su) / (2 * N)
            acc_rel = 1 - abs(E_p - E_su) / abs(E_su)
            w.writerow([beta, L, N, P, P / N, S, len(SEEDS),
                        f"{E_su:.6f}", f"{su_sem:.6f}", f"{E_p:.6f}",
                        f"{p_sem:.6f}", f"{acc_bond:.6f}", f"{acc_rel:.6f}"])
            fh.flush()
            flag = ""
            if P == 1:
                sig = (su_sem ** 2 + p_sem ** 2) ** 0.5
                z = abs(E_p - E_su) / sig if sig > 0 else 0.0
                flag = (f"  <- CONTROL ok (z={z:.1f})" if z < 3 else
                        f"  <- CONTROL FAILED (z={z:.1f}): under-equilibrated")
            print(f"    P={P:7d}  P/N={P/N:6.3f}  E={E_p:10.3f}  "
                  f"acc_bond={acc_bond:7.4f}  acc_rel={acc_rel:7.4f}{flag}")
    fh.close()

    el = time.perf_counter() - t_start
    print(f"\nwrote {args.out}")
    print(f"wall {el/60:.1f} min   cpu {wall/60:.1f} min   "
          f"parallel speedup {wall/el:.1f}x on {jobs} workers")
    print(f"CPU throughput: {upd/el/1e6:.1f} M spin-updates/s aggregate, "
          f"{1e9*el/upd:.1f} ns/update")
    print(f"                {upd/wall/1e6:.2f} M/s per core, "
          f"{1e9*wall/upd:.1f} ns/update per core")
    print("\n(compare the GPU table: 3.2 ns/update Blackwell, 8.0 ns/update V100,\n"
          " for the Heisenberg kernel - this Ising engine does less work per\n"
          " update, so the numbers are indicative of the machines, not a\n"
          " like-for-like algorithmic comparison.)\n")
    report(args.out)


if __name__ == "__main__":
    main()
