"""Real-data illustration: British occupational mobility table (Goodman 1979; R datasets::occupationalStatus).

8 x 8 table of father's (origin) by son's (destination) occupational status, N = 3,498.
Immobility (the diagonal) is the classic concentrated departure from independence.
Usage: python3 illustration_mobility.py occupationalStatus.csv OUTDIR
"""
import sys, os, json, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp2_map as M

rng = np.random.default_rng(1979)
o = pd.read_csv(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
n = o.pivot(index="origin", columns="destination", values="Freq").values.astype(float)
I, J = n.shape; dist = np.abs(np.subtract.outer(np.arange(I), np.arange(J)))


def classify(fl):
    return dict(total=int(fl.sum()), diagonal=int((fl & (dist == 0)).sum()),
                adjacent=int((fl & (dist == 1)).sum()), far=int((fl & (dist >= 2)).sum()))


res = {"N": int(n.sum()), "chisq": float(stats.chi2_contingency(n, correction=False)[0]),
       "df": (I - 1) * (J - 1)}
out_m, _ = M.methods(n, 999, rng)
res["full"] = {k: classify(v) for k, v in out_m.items()}
z, _ = M.adj_resid(n); zr, _ = M.robust_stat(n)
pd.DataFrame(np.round(z, 1)).to_csv(os.path.join(out, "adjusted_residuals.csv"), index=False)
pd.DataFrame(np.round(zr, 1)).to_csv(os.path.join(out, "robust_residuals.csv"), index=False)
# subsamples without replacement from the 3,498 individuals
cells = np.repeat(np.arange(I * J), n.ravel().astype(int))
sub = []
for size in [500, 1000, 2000, 3498]:
    for rep in range(40 if size < 3498 else 1):
        pick = rng.choice(cells, size, replace=False) if size < 3498 else cells
        t = np.bincount(pick, minlength=I * J).reshape(I, J).astype(float)
        if (t.sum(1) == 0).any() or (t.sum(0) == 0).any(): continue
        o2, _ = M.methods(t, 199, rng)
        for meth in ["ASR_bonf", "Fisher_bonf", "Robust_mc"]:
            sub.append(dict(size=size, rep=rep, method=meth, **classify(o2[meth])))
sub = pd.DataFrame(sub); sub.to_csv(os.path.join(out, "subsamples.csv"), index=False)
res["subsample_means"] = sub.groupby(["method", "size"])[["total", "diagonal", "adjacent", "far"]].mean().round(2).reset_index().to_dict("records")
json.dump(res, open(os.path.join(out, "illustration.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
