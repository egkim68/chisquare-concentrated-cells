"""Figures (PNG) from the confirmatory outputs. Separate from the simulation code so they can be redrawn.
Usage: python3 make_figures.py OUTDIR FIGDIR [DPI]
"""
import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

O, F = sys.argv[1], sys.argv[2]; DPI = int(sys.argv[3]) if len(sys.argv) > 3 else 300
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"chisq": "#000000", "one": "#1f4e9c", "two": "#5a8fd6", "diff": "#c0392b", "asr": "#c0392b",
     "fisher": "#e08e0b", "robust": "#1f4e9c", "back": "#7f7f7f"}
A = os.path.join(O, "analysis")

# Figure 1: gate
e1 = pd.read_csv(os.path.join(A, "exp1_rates.csv"), keep_default_na=False)
th = pd.read_csv(os.path.join(O, "theory", "result1_power.csv"))
fig, ax = plt.subplots(figsize=(5.2, 3.4))
g = e1[e1.pattern != "null"].groupby(["pattern", "df"])[["chisq_rate", "fisher_bonf_rate"]].mean().reset_index()
for pat, lab, col, mk in [("one_cell", "Cell test, one standout cell", C["one"], "o"),
                          ("two_cells", "Cell test, two standout cells", C["two"], "s"),
                          ("diffuse", "Cell test, spread out", C["diff"], "^")]:
    x = g[g.pattern == pat]; ax.plot(x.df, x.fisher_bonf_rate, marker=mk, color=col, label=lab, lw=1.4)
x = g.groupby("df").chisq_rate.mean(); ax.plot(x.index, x.values, color=C["chisq"], lw=1.4, ls="--", marker="x", label="Chi-square test, all patterns")
t = th[th.df.between(2, 63)]; ax.plot(t.df, t.cell_power, color=C["one"], lw=0.8, ls=":", label="Predicted, one standout cell")
ax.set_xscale("log"); ax.xaxis.set_minor_locator(NullLocator()); ax.set_xticks([2, 6, 12, 35, 63]); ax.set_xticklabels(["2×3\n(2)", "3×4\n(6)", "4×5\n(12)", "6×8\n(35)", "8×10\n(63)"])
ax.set_xlabel("Table size (degrees of freedom)"); ax.set_ylabel("Power (chi-square test set to 0.50)"); ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=7.5, loc="upper left"); fig.tight_layout(); fig.savefig(os.path.join(F, "Figure_3.png"), dpi=DPI); plt.close(fig)

# Figure 2: theory check of Result 2
r2 = pd.read_csv(os.path.join(O, "theory", "result2_drift.csv"))
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.9))
for cells, mk, col in [("linked", "o", C["asr"]), ("other", "s", C["robust"])]:
    x = r2[r2.cells == cells]
    ax[0].scatter(x.pred_mean, x.sim_mean, marker=mk, s=16, color=col, label=("Neighbor cells" if cells == "linked" else "Other ordinary cells"))
    ax[1].scatter(x.pred_flag, x.sim_flag, marker=mk, s=16, color=col)
for a_, lim in [(ax[0], (-7, 6)), (ax[1], (0, 1))]:
    a_.plot(lim, lim, color="black", lw=0.7)
ax[0].set_xlabel("Predicted mean adjusted residual"); ax[0].set_ylabel("Simulated mean"); ax[0].set_title("(a) Mean residual", loc="left", fontsize=9)
ax[1].set_xlabel("Predicted flag rate (Bonferroni)"); ax[1].set_ylabel("Simulated flag rate"); ax[1].set_title("(b) Flag rate", loc="left", fontsize=9)
ax[0].legend(frameon=False, fontsize=7.5); fig.tight_layout(); fig.savefig(os.path.join(F, "Figure_2.png"), dpi=DPI); plt.close(fig)

# Figure 3: map
m2 = pd.read_csv(os.path.join(A, "exp2_scattered.csv"))
g = m2.groupby(["method", "N"])[["f1", "fp_linked"]].mean().reset_index()
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.9))
for meth, lab, col, mk in [("ASR_bonf", "Adjusted residuals (Bonferroni)", C["asr"], "o"), ("Fisher_bonf", "Cell tests (Bonferroni)", C["fisher"], "s"),
                           ("Robust_mc", "Robust refit", C["robust"], "^"), ("Backstep", "Backward stepping", C["back"], "d")]:
    x = g[g.method == meth]
    ax[0].plot(x.N, x.f1, marker=mk, color=col, label=lab, lw=1.3); ax[1].plot(x.N, x.fp_linked, marker=mk, color=col, lw=1.3)
for a_ in ax: a_.set_xscale("log"); a_.xaxis.set_minor_locator(NullLocator()); a_.set_xticks([200, 1000, 5000]); a_.set_xticklabels(["200", "1,000", "5,000"]); a_.set_xlabel("Sample size N")
ax[0].set_ylabel("F1 score"); ax[0].set_ylim(0, 1); ax[0].set_title("(a) F1 score", loc="left", fontsize=9)
ax[1].set_ylabel("Neighbor flags per table"); ax[1].set_title("(b) Neighbor flags", loc="left", fontsize=9)
ax[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(F, "Figure_4.png"), dpi=DPI); plt.close(fig)

# Figure 4: letters
e3 = pd.read_csv(os.path.join(A, "exp3_letters_summary.csv"))
g = e3.groupby(["theta", "N"]).share_other_rows.mean().reset_index()
fig, ax = plt.subplots(figsize=(4.6, 3.1))
for thv, col, mk in [(1.0, "#7f7f7f", "x"), (1.5, C["two"], "s"), (2.0, C["one"], "o")]:
    x = g[g.theta == thv]; ax.plot(x.N, x.share_other_rows, marker=mk, color=col, lw=1.3,
                                   label="No association" if thv == 1 else f"One standout cell, θ = {thv:g}")
ax.set_xscale("log"); ax.xaxis.set_minor_locator(NullLocator()); ax.set_xticks([300, 1000, 3000, 10000]); ax.set_xticklabels(["300", "1,000", "3,000", "10,000"])
ax.set_xlabel("Sample size N"); ax.set_ylabel("Share of unaffected rows flagged"); ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=7.5); fig.tight_layout(); fig.savefig(os.path.join(F, "Figure_5.png"), dpi=DPI); plt.close(fig)
print("figures saved")
