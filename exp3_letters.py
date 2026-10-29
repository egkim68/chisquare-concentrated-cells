"""Experiment 3 (letters, confirmatory; seed 303): column-proportion subscript tests (SPSS Custom Tables / Crosstabs 'Compare column proportions').

Rows are response categories, columns are groups. In each row, every pair of columns is compared
with a two-proportion z test (pooled), Bonferroni-adjusted within the row (SPSS default).
True table: independence except ONE cell (row r, column c) with odds multiplier theta.
Because column percentages sum to 100%, the excess in (r, c) lowers every other row's share in
column c by the same factor, so rows without any interaction get significant letters.
Outputs per table: letters in row r (true row), number of other rows with at least one letter,
and whether those letters all involve column c.
Theory: for a row i != r, P(i | c) / P(i | j) = 1 / (1 + (theta - 1) P0(r | c)), constant over i.
Usage: python3 pilot5_colprop.py REPS OUTCSV
"""
import sys, itertools, numpy as np, pandas as pd
from scipy import stats

rng = np.random.default_rng(303)
ALPHA = 0.05


def letters(n):
    R, C = n.shape; col = n.sum(0); pairs = list(itertools.combinations(range(C), 2))
    sig = np.zeros((R, len(pairs)), bool)
    for t, (j, k) in enumerate(pairs):
        p1, p2 = n[:, j] / col[j], n[:, k] / col[k]
        pp = (n[:, j] + n[:, k]) / (col[j] + col[k])
        se = np.sqrt(np.maximum(pp * (1 - pp) * (1 / col[j] + 1 / col[k]), 1e-12))
        p = 2 * stats.norm.sf(np.abs(p1 - p2) / se)
        sig[:, t] = np.minimum(p * len(pairs), 1) < ALPHA
    return sig, pairs


rows = []
for (R, C), N, theta, mg in itertools.product([(3, 3), (4, 3), (5, 4), (6, 4)], [300, 1000, 3000, 10000],
                                              [1.0, 1.5, 2.0], ["equal", "skewed"]):
    pr = np.ones(R) / R if mg == "equal" else (lambda v: v / v.sum())(0.6 ** np.arange(R))
    pc = np.ones(C) / C
    r, c = 0, 0
    P = np.outer(pr, pc); P[r, c] *= theta; P /= P.sum()
    cnt = dict(true_row=0, other_rows=0, any_other=0, any_row=0, other_only_colc=0, tables=0)
    for _ in range(int(sys.argv[1])):
        n = rng.multinomial(N, P.ravel()).reshape(R, C).astype(float)
        if (n.sum(0) == 0).any(): continue
        sig, pairs = letters(n)
        other = sig[np.arange(R) != r]
        cnt["true_row"] += int(sig[r].any()); cnt["other_rows"] += int(other.any(1).sum())
        cnt["any_other"] += int(other.any()); cnt["any_row"] += int(sig.any())
        involves_c = np.array([c in pr_ for pr_ in pairs])
        cnt["other_only_colc"] += int(other.any() and not other[:, ~involves_c].any())
        cnt["tables"] += 1
    T = cnt["tables"]
    ratio = 1 / (1 + (theta - 1) * pr[r])  # P(i|c)/P(i|j) for i != r (equal column margins)
    rows.append(dict(table=f"{R}x{C}", N=N, theta=theta, margins=mg, share_true_row=cnt["true_row"] / T,
                     mean_other_rows_flagged=cnt["other_rows"] / T, share_any_other=cnt["any_other"] / T, share_any_row=cnt["any_row"] / T,
                     share_other_only_colc=(cnt["other_only_colc"] / max(cnt["any_other"], 1)),
                     theory_ratio_other_rows=ratio))
d = pd.DataFrame(rows); d.to_csv(sys.argv[2], index=False)
pd.set_option("display.width", 220); print(d.round(3).to_string(index=False))
