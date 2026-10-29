"""Numerical checks of Results 1-4 (seed 7). Usage: python3 theory_checks.py OUTDIR"""
import sys, os, numpy as np, pandas as pd
from scipy import stats, optimize

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(7)


def margins(I, J, skew):
    if not skew: return np.ones(I) / I, np.ones(J) / J
    p = 0.6 ** np.arange(I); q = 0.7 ** np.arange(J); return p / p.sum(), q / q.sum()


def adjlam(P):
    r, c = P.sum(1), P.sum(0); E = np.outer(r, c)
    return (P - E) / np.sqrt(E * (1 - r)[:, None] * (1 - c)[None, :]), ((P - E) ** 2 / E).sum()


# Result 1: one-cell association, cell adjusted non-centrality^2 / chi-square non-centrality
r1 = []
for (I, J) in [(2, 3), (3, 4), (6, 8), (8, 10)]:
    for skew in [False, True]:
        for th in [1.2, 1.5, 2.0]:
            p, q = margins(I, J, skew); P = np.outer(p, q); P[1, 1] *= th; P /= P.sum()
            L, nc = adjlam(P); r1.append(dict(table=f"{I}x{J}", skew=skew, theta=th, ratio=L[1, 1] ** 2 / nc))
r1 = pd.DataFrame(r1); r1.to_csv(os.path.join(out, "result1_ratio.csv"), index=False)
# analytic power table
pw = []
for (I, J) in [(2, 2), (2, 3), (3, 4), (4, 5), (6, 8), (8, 10), (10, 12), (20, 20)]:
    df = (I - 1) * (J - 1); crit = stats.chi2.ppf(.95, df)
    l50 = optimize.brentq(lambda l: stats.ncx2.sf(crit, df, l) - .5, 1e-6, 2000)
    zc = stats.norm.ppf(1 - .05 / (2 * I * J))
    pw.append(dict(table=f"{I}x{J}", df=df, lambda50=l50,
                   cell_power=stats.norm.sf(zc - np.sqrt(l50)) + stats.norm.cdf(-zc - np.sqrt(l50)),
                   n_ratio=l50 / zc ** 2))
pd.DataFrame(pw).to_csv(os.path.join(out, "result1_power.csv"), index=False)


# Result 2: drift of innocent cells
def adj(n):
    N = n.sum(); r = n.sum(1, keepdims=True); c = n.sum(0, keepdims=True); e = r * c / N
    return (n - e) / np.sqrt(e * (1 - r / N) * (1 - c / N))


r2 = []
for (I, J) in [(4, 5), (6, 8)]:
    for K in [1, 3]:
        p, q = margins(I, J, False); dev = np.zeros((I, J), bool); dev[np.arange(K), np.arange(K)] = True
        P = np.outer(p, q) * np.where(dev, 2.0, 1.0); P /= P.sum(); L, _ = adjlam(P)
        linked = (dev.any(1)[:, None] | dev.any(0)[None, :]) & ~dev; other = ~dev & ~linked
        zc = stats.norm.ppf(1 - 0.05 / (2 * I * J))
        for N in [200, 1000, 5000, 20000]:
            Z = []
            while len(Z) < 4000:
                n = rng.multinomial(N, P.ravel()).reshape(I, J).astype(float)
                if (n.sum(1) > 0).all() and (n.sum(0) > 0).all(): Z.append(adj(n))
            Z = np.array(Z)
            for name, m in [("linked", linked), ("other", other)]:
                lm = L[m]
                r2.append(dict(table=f"{I}x{J}", K=K, N=N, cells=name, pred_mean=np.sqrt(N) * lm.mean(),
                               sim_mean=Z[:, m].mean(),
                               pred_flag=(stats.norm.cdf(-zc - np.sqrt(N) * lm) + stats.norm.cdf(-zc + np.sqrt(N) * lm)).mean(),
                               sim_flag=(np.abs(Z[:, m]) > zc).mean()))
pd.DataFrame(r2).to_csv(os.path.join(out, "result2_drift.csv"), index=False)


# Result 3: noise-free median polish
def mp_res(P):
    res = np.log(P).copy()
    for _ in range(100): res -= np.median(res, 1)[:, None]; res -= np.median(res, 0)[None, :]
    return res


r3 = []
for (I, J), K, lay in [((4, 5), 1, "scattered"), ((4, 5), 3, "scattered"), ((6, 8), 3, "scattered"),
                       ((4, 5), 2, "one row"), ((6, 8), 3, "one row"), ((6, 8), 4, "one row"), ((4, 5), 3, "one row")]:
    p, q = margins(I, J, True); dev = np.zeros((I, J), bool)
    if lay == "scattered": dev[np.arange(K), np.arange(K)] = True
    else: dev[0, :K] = True
    P = np.outer(p, q) * np.where(dev, 2.0, 1.0); P /= P.sum(); res = mp_res(P)
    r3.append(dict(table=f"{I}x{J}", K=K, layout=lay, share_of_row=K / J if lay == "one row" else 1 / J,
                   max_abs_null=np.abs(res[~dev]).max(), mean_dev=res[dev].mean()))
pd.DataFrame(r3).to_csv(os.path.join(out, "result3_breakdown.csv"), index=False)

# Result 4: column-proportion ratio
r4 = []
for R, C in [(4, 3), (6, 4)]:
    for th in [1.5, 2.0]:
        pr = 0.6 ** np.arange(R); pr /= pr.sum(); P = np.outer(pr, np.ones(C) / C); P[0, 0] *= th; P /= P.sum()
        cp = P / P.sum(0, keepdims=True)
        r4.append(dict(rows=R, cols=C, theta=th, ratio_exact=(cp[1:, 0] / cp[1:, 1]).mean(),
                       ratio_formula=1 / (1 + (th - 1) * pr[0])))
pd.DataFrame(r4).to_csv(os.path.join(out, "result4_closure.csv"), index=False)
print(r1.ratio.min(), r1.ratio.max())
print(pd.DataFrame(r2).round(3).to_string(index=False))
print(pd.DataFrame(r3).round(4).to_string(index=False)); print(pd.DataFrame(r4).round(4).to_string(index=False))
