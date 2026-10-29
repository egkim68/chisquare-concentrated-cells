"""Analysis of the confirmatory runs: summary tables and hypothesis tests (H1-H8).
Usage: python3 analysis.py OUTDIR   (expects OUTDIR/exp1/exp1_gate.csv, OUTDIR/exp2_map.csv, OUTDIR/exp3_letters.csv)
Writes OUTDIR/analysis/*.csv and hypotheses.json.
"""
import sys, os, json, numpy as np, pandas as pd
from scipy import stats

O = sys.argv[1]; A = os.path.join(O, "analysis"); os.makedirs(A, exist_ok=True)
rng = np.random.default_rng(2026)
H = {}

# ---------------- Experiment 1 ----------------
e1 = pd.read_csv(os.path.join(O, "exp1", "exp1_gate.csv"), keep_default_na=False)
for t in ["chisq", "G", "max_asr_mc", "asr_bonf", "fisher_bonf"]: e1[t + "_rate"] = e1[t] / e1.reps
e1.to_csv(os.path.join(A, "exp1_rates.csv"), index=False)
order = ["2x3", "3x4", "4x5", "6x8", "8x10"]
size = e1[e1.pattern == "null"].pivot_table(index="table", values=[c + "_rate" for c in ["chisq", "G", "max_asr_mc", "asr_bonf", "fisher_bonf"]],
                                            aggfunc=["mean", "max"]).reindex(order)
size.to_csv(os.path.join(A, "exp1_size.csv"))
pw = e1[e1.pattern != "null"].pivot_table(index=["pattern", "table"], values=["chisq_rate", "fisher_bonf_rate", "max_asr_mc_rate", "asr_bonf_rate"], aggfunc="mean")
pw.to_csv(os.path.join(A, "exp1_power.csv"))


def pooled_diff(df, a, b):
    """Pooled power difference a - b with binomial standard error (independent; conservative for paired tests)."""
    na, nb, n = df[a].sum(), df[b].sum(), df.reps.sum()
    pa, pb = na / n, nb / n
    return pa - pb, np.sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n)


big = e1[e1.table.isin(["6x8", "8x10"])]
d, se = pooled_diff(big[big.pattern == "one_cell"], "fisher_bonf", "chisq")
H["H1"] = dict(criterion="One cell, 6x8 and 8x10: Fisher-Bonferroni power minus chi-square power", type="Difference",
               threshold=">= 0.10", estimate=d, p=stats.norm.sf((d - 0.10) / se), passed=bool(d >= 0.10))
d, se = pooled_diff(big[big.pattern == "diffuse"], "chisq", "fisher_bonf")
H["H2"] = dict(criterion="Diffuse, 6x8 and 8x10: chi-square power minus Fisher-Bonferroni power", type="Difference",
               threshold=">= 0.15", estimate=d, p=stats.norm.sf((d - 0.15) / se), passed=bool(d >= 0.15))
x = big[(big.pattern == "one_cell") & (big.N == 1000)]
k, n = int(x.true_found_chi_ns.sum()), int(x.true_found.sum())
H["H3"] = dict(criterion="One cell, 6x8 and 8x10, N = 1,000: share of Fisher-detected true cells with non-significant chi-square",
               type="Share", threshold=">= 0.25", estimate=k / n, p=stats.binomtest(k, n, 0.25, alternative="greater").pvalue,
               passed=bool(k / n >= 0.25), n=n)
gate = e1[e1.pattern == "one_cell"].groupby(["table", "N"]).agg(found=("true_found", "sum"), ns=("true_found_chi_ns", "sum"),
                                                                 reps=("reps", "sum"), chi=("chisq", "sum")).reset_index()
gate["chi_power"] = gate.chi / gate.reps; gate["found_rate"] = gate.found / gate.reps; gate["share_ns"] = gate.ns / gate.found
gate.to_csv(os.path.join(A, "exp1_gate_cost.csv"), index=False)

# ---------------- Experiment 2 ----------------
e2 = pd.read_csv(os.path.join(O, "exp2_map.csv"))
e2["tab"] = e2.I.astype(str) + "x" + e2.J.astype(str)
sc = e2[(e2.K > 0) & (e2.layout == "scattered")]
m2 = sc.groupby(["method", "tab", "N"])[["f1", "rec", "fp_linked", "fp_other"]].mean().reset_index()
m2.to_csv(os.path.join(A, "exp2_scattered.csv"), index=False)
nullfw = e2[e2.K == 0].groupby(["method", "tab", "N", "margins"]).anyfp.mean().reset_index()
nullfw.to_csv(os.path.join(A, "exp2_null_fwer.csv"), index=False)
row = e2[e2.layout == "samerow"].groupby(["method", "tab", "N"]).f1.mean().reset_index()
row.to_csv(os.path.join(A, "exp2_samerow.csv"), index=False)
gated = sc[sc.omni_sig == 1].groupby(["method", "tab", "N"])[["f1", "fp_linked"]].mean().reset_index()
gated.to_csv(os.path.join(A, "exp2_gated.csv"), index=False)

a = sc[sc.method == "ASR_bonf"]
l5, l1 = a[a.N == 5000].fp_linked.values, a[a.N == 1000].fp_linked.values
ratio = l5.mean() / l1.mean()
boots = np.array([rng.choice(l5, l5.size).mean() / max(rng.choice(l1, l1.size).mean(), 1e-9) for _ in range(2000)])
H["H4"] = dict(criterion="Map, scattered: linked false flags of adjusted residuals (Bonferroni), N = 5,000 / N = 1,000",
               type="Ratio", threshold=">= 3", estimate=ratio, p=float((boots < 3).mean()), passed=bool(ratio >= 3))
s5 = sc[sc.N == 5000].pivot_table(index=["tab", "margins", "K", "rep"], columns="method", values="f1")
dd = (s5["Robust_mc"] - s5["ASR_bonf"]).dropna()
H["H5"] = dict(criterion="Map, scattered, N = 5,000: robust two-stage F1 minus adjusted residuals (Bonferroni) F1",
               type="Difference", threshold=">= 0.15", estimate=dd.mean(),
               p=stats.ttest_1samp(dd, 0.15, alternative="greater").pvalue, passed=bool(dd.mean() >= 0.15))
rb = e2[(e2.K == 0) & (e2.method == "Robust_mc")]
cond = rb.groupby(["tab", "N", "margins"]).anyfp.mean()
k, n = int(rb.anyfp.sum()), len(rb)
H["H6"] = dict(criterion="Map, no deviating cells: robust two-stage familywise error, mean and maximum over conditions",
               type="Share", threshold="<= 0.075 and <= 0.12", estimate=k / n, max=float(cond.max()),
               p=stats.binomtest(k, n, 0.075, alternative="less").pvalue,
               passed=bool(k / n <= 0.075 and cond.max() <= 0.12))

# ---------------- Experiment 3 ----------------
e3 = pd.read_csv(os.path.join(O, "exp3_letters.csv"))
e3["R"] = e3.table.str.split("x").str[0].astype(int)
e3["share_other_rows"] = e3.mean_other_rows_flagged / (e3.R - 1)
e3.to_csv(os.path.join(A, "exp3_letters_summary.csv"), index=False)
x = e3[(e3.theta == 1.5) & (e3.N == 10000)]
H["H7"] = dict(criterion="Letters, one cell with odds multiplier 1.5, N = 10,000: share of unaffected rows with a letter",
               type="Mean", threshold=">= 0.50", estimate=x.share_other_rows.mean(),
               p=stats.ttest_1samp(x.share_other_rows, 0.5, alternative="greater").pvalue,
               passed=bool(x.share_other_rows.mean() >= 0.5), conditions=len(x))
x = e3[(e3.theta == 1.0) & (e3.R >= 4)]
k = int(round((x.share_any_row * 5000).sum())); n = 5000 * len(x)
H["H8"] = dict(criterion="Letters, no association, 4 or more rows: share of tables with a letter in any row",
               type="Share", threshold=">= 0.10", estimate=k / n, p=stats.binomtest(k, n, 0.10, alternative="greater").pvalue,
               passed=bool(k / n >= 0.10))

for h in H.values():
    for kk, v in list(h.items()):
        if isinstance(v, (np.floating, np.integer)): h[kk] = float(v)
json.dump(H, open(os.path.join(A, "hypotheses.json"), "w"), indent=1)
print(json.dumps(H, indent=1))
