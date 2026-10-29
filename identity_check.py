"""Check of Result 1 in 2,000 random tables (seed 1): with one standout cell, (a) the departure from
independence is rank one, D = k (e_r - r)(e_c - c)^T; (b) the chi-square non-centrality equals the
adjusted-residual non-centrality of the standout cell; (c) the collapsed 2 x 2 odds ratio of that cell equals theta; (d) median polish recovery rate.
Usage: python3 identity_check.py"""
import numpy as np
rng = np.random.default_rng(1); err = {"rank_one": 0, "noncentrality": 0, "odds_ratio": 0}
for _ in range(2000):
    I, J = rng.integers(2, 11, 2); p = rng.dirichlet(np.ones(I)); q = rng.dirichlet(np.ones(J))
    th = rng.uniform(1.05, 4); r0, c0 = rng.integers(I), rng.integers(J)
    P = np.outer(p, q); P[r0, c0] *= th; P /= P.sum(); r, c = P.sum(1), P.sum(0); D = P - np.outer(r, c)
    er, ec = np.eye(I)[r0] - r, np.eye(J)[c0] - c; k = D[r0, c0] / (er[r0] * ec[c0])
    err["rank_one"] = max(err["rank_one"], np.abs(D - k * np.outer(er, ec)).max())
    lam = (D ** 2 / np.outer(r, c)).sum(); cell = D[r0, c0] ** 2 / (r[r0] * c[c0] * (1 - r[r0]) * (1 - c[c0]))
    err["noncentrality"] = max(err["noncentrality"], abs(lam / cell - 1))
    a = P[r0, c0]; b = r[r0] - a; cc = c[c0] - a; d = 1 - r[r0] - c[c0] + a
    err["odds_ratio"] = max(err["odds_ratio"], abs(a * d / (b * cc) / th - 1))
print("largest deviations:", err)
# (d) median polish recovery when fewer than half of every row and column stand out (seed 3)
rng = np.random.default_rng(3); tot = miss = 0
for _ in range(20000):
    I, J = rng.integers(3, 11, 2); y = np.add.outer(rng.normal(0, 1, I), rng.normal(0, 1, J)); dev = np.zeros((I, J), bool)
    K = rng.integers(1, max(2, min(I, J)))
    for k in rng.permutation(I * J):
        i, j = divmod(k, J); d2 = dev.copy(); d2[i, j] = True
        if d2.sum(1).max() < J / 2 and d2.sum(0).max() < I / 2: dev = d2
        if dev.sum() >= K: break
    res = y + np.where(dev, np.log(rng.uniform(1.5, 4, (I, J))), 0)
    for _ in range(200): res -= np.median(res, 1)[:, None]; res -= np.median(res, 0)[None, :]
    tot += 1; miss += np.abs(res[~dev]).max() > 1e-6
print(f"median polish exact recovery: {tot - miss} of {tot} ({(tot - miss) / tot:.3%})")
