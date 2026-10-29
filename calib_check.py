"""Stability of the robust refit's Monte Carlo cutoff: B = 99 (as in Experiment 2) versus B = 999.
For tables without standout cells (false alarms) and with three standout cells (accuracy), each table
is scored with both cutoffs, computed from independent simulation draws.
Usage: python3 calib_check.py REPS OUTCSV [NPROC]   (seed base 4040)
"""
import sys, os, itertools, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp2_map as M


def run(args):
    I, J, N, K, mg, reps, seed = args
    rng = np.random.default_rng(seed); rows = []
    for rep in range(reps):
        n, dev = M.make_table(I, J, N, K, 2.0, mg, "scattered", rng)
        if (n.sum(1) == 0).any() or (n.sum(0) == 0).any(): continue
        z, _ = M.robust_stat(n)
        for B in (99, 999):
            cut = M.mc_cutoff(n, M.robust_stat, B, rng); fl = np.abs(z) > cut
            tp = int((fl & dev).sum()); fp = int((fl & ~dev).sum()); fn = int((~fl & dev).sum())
            rows.append(dict(I=I, J=J, N=N, K=K, margins=mg, rep=rep, B=B, cut=cut, anyfp=int(fp > 0),
                             f1=(2 * tp / (2 * tp + fp + fn) if K else np.nan)))
    return rows


if __name__ == "__main__":
    reps, outp = int(sys.argv[1]), sys.argv[2]; nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    grid = [(I, J, N, K, mg) for (I, J), N, K, mg in itertools.product([(4, 5), (6, 8), (8, 10)], [200, 1000, 5000], [0], ["equal", "skewed"])]
    grid += [(I, J, 5000, 3, mg) for (I, J), mg in itertools.product([(4, 5), (6, 8), (8, 10)], ["equal", "skewed"])]
    tasks = [g + (reps, 4040 + s) for s, g in enumerate(grid)]
    with Pool(nproc) as p: res = p.map(run, tasks, chunksize=1)
    d = pd.DataFrame([r for rr in res for r in rr]); d.to_csv(outp, index=False); print("saved", len(d))
