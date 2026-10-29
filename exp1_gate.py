"""Experiment 1 (gate): omnibus chi-square versus cell-level tests at matched non-centrality.

Usage: python3 exp1_gate.py REPS OUTDIR [NPROC]
Writes one CSV per condition to OUTDIR/cells (resumable) and OUTDIR/exp1_gate.csv.
"""
import sys, os, itertools, numpy as np, pandas as pd
from scipy import stats, optimize
from multiprocessing import Pool

ALPHA = 0.05
SEED = 101


def margins(I, J, mg):
    if mg == "equal": return np.ones(I) / I, np.ones(J) / J
    p = 0.6 ** np.arange(I); q = 0.7 ** np.arange(J)
    return p / p.sum(), q / q.sum()


def ncp(P):
    r, c = P.sum(1), P.sum(0); E = np.outer(r, c)
    return ((P - E) ** 2 / E).sum()


def lam50(df):
    crit = stats.chi2.ppf(1 - ALPHA, df)
    return optimize.brentq(lambda l: stats.ncx2.sf(crit, df, l) - 0.5, 1e-6, 500)


def build(p, q, pattern, target, rng):
    I, J = len(p), len(q); base = np.outer(p, q); dev = np.zeros((I, J), bool)
    if pattern == "null": return base, dev
    if pattern in ("one_cell", "two_cells"):
        K = 1 if pattern == "one_cell" else 2
        dev[rng.choice(I, K, replace=False), rng.choice(J, K, replace=False)] = True
        f = lambda t: ncp((lambda P: P / P.sum())(base * np.where(dev, t, 1.0))) - target
        t = optimize.brentq(f, 1.0 + 1e-12, 1e5)
        P = base * np.where(dev, t, 1.0); return P / P.sum(), dev
    u = rng.standard_normal((I, J)); u -= u.mean(1, keepdims=True); u -= u.mean(0, keepdims=True)
    f = lambda s: ncp((lambda P: P / P.sum())(base * np.exp(s * u))) - target
    s = optimize.brentq(f, 1e-12, 50)
    P = base * np.exp(s * u); return P / P.sum(), dev


def adj(n):
    N = n.sum(); r = n.sum(1, keepdims=True); c = n.sum(0, keepdims=True); e = r * c / N
    return (n - e) / np.sqrt(np.maximum(e * (1 - r / N) * (1 - c / N), 1e-12)), e


def fisher_p(n):
    I, J = n.shape; N = n.sum(); r = n.sum(1); c = n.sum(0); P = np.ones((I, J))
    for i in range(I):
        for j in range(J):
            a = n[i, j]
            P[i, j] = stats.fisher_exact([[a, r[i] - a], [c[j] - a, N - r[i] - c[j] + a]])[1]
    return P


def ok(n): return (n.sum(1) > 0).all() and (n.sum(0) > 0).all()


def run(args):
    I, J, N, mg, pattern, reps, outdir, cid = args
    fn = os.path.join(outdir, "cells", f"c{cid:03d}.csv")
    if os.path.exists(fn): return pd.read_csv(fn, keep_default_na=False).to_dict("records")
    rng = np.random.default_rng([SEED, cid])
    p, q = margins(I, J, mg); df = (I - 1) * (J - 1); target = lam50(df) / N; k = I * J
    crng = np.random.default_rng([SEED, cid, 99]); base = np.outer(p, q).ravel(); mx = []
    while len(mx) < 20000:
        n = crng.multinomial(N, base).reshape(I, J).astype(float)
        if ok(n): mx.append(np.abs(adj(n)[0]).max())
    cut = np.quantile(mx, 1 - ALPHA); zb = stats.norm.ppf(1 - ALPHA / (2 * k))
    acc = dict(chisq=0, G=0, max_asr_mc=0, asr_bonf=0, fisher_bonf=0, true_found=0, true_found_chi_ns=0)
    used = skipped = 0
    for _ in range(reps):
        P, dev = build(p, q, pattern, target, rng)
        n = rng.multinomial(N, P.ravel()).reshape(I, J).astype(float)
        if not ok(n): skipped += 1; continue
        z, e = adj(n); chi = ((n - e) ** 2 / e).sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            G = 2 * np.nansum(np.where(n > 0, n * np.log(n / e), 0))
        fp = fisher_p(n)
        chi_sig = stats.chi2.sf(chi, df) < ALPHA
        acc["chisq"] += chi_sig; acc["G"] += stats.chi2.sf(G, df) < ALPHA
        acc["max_asr_mc"] += np.abs(z).max() > cut; acc["asr_bonf"] += np.abs(z).max() > zb
        acc["fisher_bonf"] += fp.min() < ALPHA / k
        if pattern == "one_cell":
            found = bool((fp[dev] < ALPHA / k).any())
            acc["true_found"] += found; acc["true_found_chi_ns"] += found and not chi_sig
        used += 1
    row = dict(table=f"{I}x{J}", I=I, J=J, df=df, N=N, margins=mg, pattern=pattern, reps=used, skipped=skipped,
               lambda50=lam50(df), mc_cut=cut, **{kk: int(v) for kk, v in acc.items()})
    pd.DataFrame([row]).to_csv(fn, index=False)
    return [row]


if __name__ == "__main__":
    reps, outdir = int(sys.argv[1]), sys.argv[2]; nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    os.makedirs(os.path.join(outdir, "cells"), exist_ok=True)
    grid = list(itertools.product([(2, 3), (3, 4), (4, 5), (6, 8), (8, 10)], [200, 1000, 5000],
                                  ["equal", "skewed"], ["null", "one_cell", "two_cells", "diffuse"]))
    tasks = [(I, J, N, mg, pat, reps, outdir, cid) for cid, ((I, J), N, mg, pat) in enumerate(grid)]
    tasks.sort(key=lambda t: -t[0] * t[1])          # big tables first for load balance
    with Pool(nproc) as pool: res = pool.map(run, tasks, chunksize=1)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(os.path.join(outdir, "exp1_gate.csv"), index=False)
    print("done", len(res))
