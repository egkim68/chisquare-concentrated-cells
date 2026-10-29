"""Experiment 2 (map): post hoc cell analysis after a chi-square test, with calibrated robust methods (confirmatory; seed base 2000).

True table: independence except K deviating cells whose probability is multiplied by theta.
Methods (each returns a boolean matrix of flagged cells):
  Pearson_unadj   |(n-e)/sqrt(e)| > 1.96                       (default printout in R/SPSS)
  ASR_unadj       adjusted residual |z| > 1.96
  ASR_bonf        adjusted residual, Bonferroni over IJ cells
  ASR_holm        adjusted residual, Holm
  ASR_BH          adjusted residual, Benjamini-Hochberg (FDR)
  Fisher_bonf     Fisher exact cell-versus-rest 2x2, Bonferroni (Shan & Gerstberger 2017)
  KH_mc           Kotze-Hawkins: median log odds ratio over all 2x2 subtables with the cell; MC-calibrated
  Robust_mc       median polish on log counts -> candidates -> quasi-independence refit -> z; MC-calibrated
  Backstep        backward stepping (Simonoff 1988), Bonferroni
MC calibration: B tables simulated from the quasi-independence fit of the observed table; cutoff is the
(1 - alpha) quantile of the maximum |statistic| over cells (familywise, Westfall-Young style).
Omnibus gate: Pearson chi-square p < .05, or a Monte Carlo exact p (fixed margins) when any expected count < 5.
Usage: python3 pilot3.py REPS B OUTCSV [NPROC]
"""
import sys, itertools, numpy as np, pandas as pd
from scipy import stats
from multiprocessing import Pool

ALPHA = 0.05


# ---------------- fits and statistics ----------------
def quasi_fit(n, w):
    I, J = n.shape
    a = np.ones(I); b = np.ones(J)
    R = (n * w).sum(1); C = (n * w).sum(0)
    for _ in range(500):
        a_new = R / np.maximum((w * b[None, :]).sum(1), 1e-12)
        b_new = C / np.maximum((w * a_new[:, None]).sum(0), 1e-12)
        done = np.allclose(a_new, a, rtol=1e-9) and np.allclose(b_new, b, rtol=1e-9)
        a, b = a_new, b_new
        if done: break
    return np.outer(a, b), R, C


def zscores(n, w):
    m, R, C = quasi_fit(n, w); T = (n * w).sum()
    Rm = np.maximum(R, 1e-9)[:, None]; Cm = np.maximum(C, 1e-9)[None, :]
    v_in = m * np.clip((1 - m / Rm) * (1 - m / Cm), 1e-6, None)
    v_out = m + m ** 2 * (1 / Rm + 1 / Cm - 1 / T)
    v = np.where(w > 0, v_in, v_out)
    return (n - m) / np.sqrt(np.maximum(v, 1e-12)), m


def adj_resid(n):
    N = n.sum(); r = n.sum(1, keepdims=True); c = n.sum(0, keepdims=True)
    e = r * c / N
    return (n - e) / np.sqrt(np.maximum(e * (1 - r / N) * (1 - c / N), 1e-12)), e


def median_polish_log(n, iters=20):
    y = np.log(n + 0.5); res = y.copy()
    rowe = np.zeros(y.shape[0]); cole = np.zeros(y.shape[1]); ov = 0.0
    for _ in range(iters):
        rm = np.median(res, 1); res -= rm[:, None]; rowe += rm
        m = np.median(cole); cole -= m; ov += m
        cm = np.median(res, 0); res -= cm[None, :]; cole += cm
        m = np.median(rowe); rowe -= m; ov += m
    return res  # log-scale residuals from the robust additive fit



def cap_rows_cols(cand, score):
    """Keep fewer than half of the cells of every row and column as candidates (the breakdown
    condition of the robust fit); drop the weakest candidates first."""
    cand = cand.copy(); I, J = cand.shape
    rmax, cmax = (J - 1) // 2, (I - 1) // 2
    for _ in range(I * J):
        bad = False
        for i in range(I):
            idx = np.where(cand[i])[0]
            if idx.size > rmax:
                drop = idx[np.argsort(score[i, idx])[: idx.size - rmax]]; cand[i, drop] = False; bad = True
        for j in range(J):
            idx = np.where(cand[:, j])[0]
            if idx.size > cmax:
                drop = idx[np.argsort(score[idx, j])[: idx.size - cmax]]; cand[drop, j] = False; bad = True
        if not bad: break
    return cand

def robust_stat(n):
    """Two-stage robust statistic: candidates are cells with large median-polish log residuals
    (more than 3 robust SDs); refit quasi-independence without them; z for every cell."""
    res = median_polish_log(n)
    s = 1.4826 * np.median(np.abs(res)) + 1e-9
    cand = np.abs(res) > 3 * s
    cand = cap_rows_cols(cand, np.abs(res))
    z, m = zscores(n, (~cand).astype(float))
    return z, m


def kh_stat(n):
    """Kotze-Hawkins: for each cell, median over all 2x2 subtables containing it of the
    standardized log odds ratio (cell, same row, same column, opposite corner)."""
    y = np.log(n + 0.5); I, J = n.shape
    inv = 1 / (n + 0.5)
    S = np.zeros((I, J))
    for i in range(I):
        oi = [k for k in range(I) if k != i]
        for j in range(J):
            oj = [k for k in range(J) if k != j]
            lor = y[i, j] - y[i, oj][None, :] - y[oi, j][:, None] + y[np.ix_(oi, oj)]
            se = np.sqrt(inv[i, j] + inv[i, oj][None, :] + inv[oi, j][:, None] + inv[np.ix_(oi, oj)])
            S[i, j] = np.median(lor / se)
    return S


def mc_cutoff(n, stat_fn, B, rng):
    """Cutoff = (1-alpha) quantile of max|stat| over B multinomial tables from the robust null fit."""
    _, m = robust_stat(n)
    p = (m / m.sum()).ravel(); N = int(n.sum())
    mx = []
    for _ in range(B):
        nb = rng.multinomial(N, p).reshape(n.shape).astype(float)
        nb = nb[nb.sum(1) > 0][:, nb.sum(0) > 0]          # drop empty rows/columns of the simulated table
        if min(nb.shape) < 2: continue
        S = stat_fn(nb); S = S[0] if isinstance(S, tuple) else S
        mx.append(np.max(np.abs(S)))
    return np.quantile(mx, 1 - ALPHA) if mx else np.inf


def holm(p):
    o = np.argsort(p.ravel()); k = p.size; rej = np.zeros(k, bool)
    for r, idx in enumerate(o):
        if p.ravel()[idx] <= ALPHA / (k - r): rej[idx] = True
        else: break
    return rej.reshape(p.shape)


def bh(p):
    o = np.argsort(p.ravel()); k = p.size; ps = p.ravel()[o]
    ok = np.where(ps <= ALPHA * np.arange(1, k + 1) / k)[0]
    rej = np.zeros(k, bool)
    if ok.size: rej[o[: ok.max() + 1]] = True
    return rej.reshape(p.shape)


def omnibus_p(n, rng, B=500):
    chi, p, _, e = stats.chi2_contingency(n, correction=False)
    if (e < 5).any():  # Monte Carlo exact test with fixed margins
        r, c = n.sum(1).astype(int), n.sum(0).astype(int)
        tabs = stats.random_table(r, c, seed=rng).rvs(B)
        ee = np.outer(r, c) / r.sum()
        sims = (((tabs - ee) ** 2) / ee).sum((1, 2))
        p = (1 + (sims >= chi - 1e-9).sum()) / (B + 1)
    return p


def methods(n, B, rng):
    I, J = n.shape; k = I * J
    zc_b = stats.norm.ppf(1 - ALPHA / (2 * k)); z196 = stats.norm.ppf(1 - ALPHA / 2)
    z, e = adj_resid(n)
    pz = 2 * stats.norm.sf(np.abs(z))
    out = {"Pearson_unadj": np.abs((n - e) / np.sqrt(e)) > z196,
           "ASR_unadj": np.abs(z) > z196, "ASR_bonf": np.abs(z) > zc_b,
           "ASR_holm": holm(pz), "ASR_BH": bh(pz)}
    N = n.sum(); r = n.sum(1); c = n.sum(0); f = np.zeros_like(n, bool)
    for i in range(I):
        for j in range(J):
            a = n[i, j]
            f[i, j] = stats.fisher_exact([[a, r[i] - a], [c[j] - a, N - r[i] - c[j] + a]])[1] < ALPHA / k
    out["Fisher_bonf"] = f
    S = kh_stat(n); out["KH_mc"] = np.abs(S) > mc_cutoff(n, kh_stat, B, rng)
    zr, _ = robust_stat(n); out["Robust_mc"] = np.abs(zr) > mc_cutoff(n, robust_stat, B, rng)
    flag = np.zeros_like(n, bool)
    for _ in range(k // 2):
        zz, _ = zscores(n, (~flag).astype(float)); zz[flag] = 0
        # respect the breakdown condition: a row/column keeps more than half of its cells unflagged
        full = (flag.sum(1)[:, None] >= (J - 1) // 2) | (flag.sum(0)[None, :] >= (I - 1) // 2)
        zz[full] = 0
        if not np.any(zz): break
        idx = np.unravel_index(np.argmax(np.abs(zz)), zz.shape)
        w2 = (~flag).astype(float); w2[idx] = 0
        if abs(zscores(n, w2)[0][idx]) > zc_b: flag[idx] = True
        else: break
    out["Backstep"] = flag
    return out, omnibus_p(n, rng)


# ---------------- data generation ----------------
def make_table(I, J, N, K, theta, margins, layout, rng):
    if margins == "equal": pr, pc = np.ones(I) / I, np.ones(J) / J
    else:
        pr = 0.6 ** np.arange(I); pc = 0.7 ** np.arange(J)
        pr, pc = rng.permutation(pr / pr.sum()), rng.permutation(pc / pc.sum())
    dev = np.zeros((I, J), bool)
    if K:
        if layout == "scattered":
            dev[rng.choice(I, K, replace=False), rng.choice(J, K, replace=False)] = True
        else:
            dev[rng.integers(I), rng.choice(J, K, replace=False)] = True
    p = np.outer(pr, pc) * np.where(dev, theta, 1.0); p /= p.sum()
    n = rng.multinomial(N, p.ravel()).reshape(I, J).astype(float)
    return n, dev


def run_cell(args):
    (I, J, N, K, theta, mg, lay, reps, B, seed, outdir) = args
    import os
    cellfile = os.path.join(outdir, f"cell_{seed}.csv")
    if os.path.exists(cellfile): return pd.read_csv(cellfile).to_dict("records")
    rng = np.random.default_rng(seed); rows = []
    for rep in range(reps):
        n, dev = make_table(I, J, N, K, theta, mg, lay, rng)
        if (n.sum(1) == 0).any() or (n.sum(0) == 0).any(): continue
        out, pomni = methods(n, B, rng)
        linked = (dev.any(1)[:, None] | dev.any(0)[None, :]) & ~dev
        for meth, fl in out.items():
            tp = int((fl & dev).sum()); fp = int((fl & ~dev).sum()); fn = int((~fl & dev).sum())
            rows.append(dict(I=I, J=J, N=N, K=K, theta=theta, margins=mg, layout=lay, rep=rep, method=meth,
                             tp=tp, fp=fp, fn=fn, f1=(2 * tp / (2 * tp + fp + fn) if K else np.nan),
                             rec=(tp / K if K else np.nan), fp_linked=int((fl & linked).sum()),
                             fp_other=int((fl & ~dev & ~linked).sum()), anyfp=int(fp > 0),
                             omni_sig=int(pomni < ALPHA), min_e=float((np.outer(n.sum(1), n.sum(0)) / n.sum()).min())))
    pd.DataFrame(rows).to_csv(cellfile, index=False)
    return rows


if __name__ == "__main__":
    reps, B, outp = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    grid = []
    for (I, J), N, K, mg in itertools.product([(4, 5), (6, 8), (8, 10)], [200, 1000, 5000], [0, 1, 3], ["equal", "skewed"]):
        for lay in (["scattered"] if K < 3 else ["scattered", "samerow"]):
            grid.append((I, J, N, K, 2.0, mg, lay))
    import os
    outdir = outp + "_cells"; os.makedirs(outdir, exist_ok=True)
    tasks = [g + (reps, B, 2000 + s, outdir) for s, g in enumerate(grid)]
    with Pool(nproc) as pool:
        res = pool.map(run_cell, tasks, chunksize=1)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(outp, index=False)
    print("saved", outp, sum(len(r) for r in res))
