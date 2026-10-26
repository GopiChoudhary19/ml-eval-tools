# paired_delta.py - paired per-row test: is candidate B better than baseline A on the SAME holdout rows?
# Mode 1 (per-row scores, any metric):  paired_delta.py cand.csv baseline.csv --col score [--lower]
# Mode 2 (predictions + truth):          paired_delta.py cand.csv baseline.csv --truth holdout_truth.csv --target y --metric auc
# Exit 0 = BETTER (favourable delta, t >= --tmin), 1 = DIFFERENT, not better.
import argparse, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comp_metrics as C

ap = argparse.ArgumentParser()
ap.add_argument("cand"); ap.add_argument("baseline")
ap.add_argument("--id", default="id")
ap.add_argument("--col", help="per-row score column (mode 1)")
ap.add_argument("--lower", action="store_true", help="mode 1: lower per-row score is better")
ap.add_argument("--truth"); ap.add_argument("--target"); ap.add_argument("--metric")
ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--tmin", type=float, default=0.7)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

if a.col:
    A = pd.read_csv(a.baseline).set_index(a.id)[a.col]; B = pd.read_csv(a.cand).set_index(a.id)[a.col]
    common = A.index.intersection(B.index)
    if len(common) < max(len(A), len(B)):
        print(f"WARNING: only {len(common)} shared ids (baseline {len(A)}, cand {len(B)}) - holdout rows differ, pairing is "
              f"only valid on the shared rows")
    sa, sb = A.loc[common].to_numpy(float), B.loc[common].to_numpy(float)
    sign = -1 if a.lower else 1
    d = sign * (sb - sa)
    n = len(d); mean = d.mean(); sem = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("inf")
    t = mean / sem if sem > 0 else (np.inf if mean > 0 else 0.0)
    better, worse = int((d > 0).sum()), int((d < 0).sum())
    print(f"rows {n} | baseline {sa.mean():.6f} | cand {sb.mean():.6f} | delta {sign*mean:+.6f} (favourable {mean:+.6f})")
    print(f"paired sem {sem:.6f} | t {t:+.2f} | rows better {better}/{n}, worse {worse}, tied {n-better-worse}")
else:
    if not (a.truth and a.target and a.metric):
        sys.exit("need --col, or --truth + --target + --metric")
    m = C.canon(a.metric); direction = C.DIRECTION[m]
    tr = C.load_truth(a.truth, a.id, a.target); ids = tr[a.id].tolist(); y = tr["__y"].to_numpy()
    pa, cls_a = C.as_matrix(C.load_pred(a.baseline, a.id, ids, a.target), y)
    pb, cls_b = C.as_matrix(C.load_pred(a.cand, a.id, ids, a.target), y)
    if (cls_a is None) != (cls_b is None) or (cls_a and cls_a != cls_b):
        sys.exit("prediction files have different column layouts")
    sa, sb = C.score(m, y, pa, cls_a), C.score(m, y, pb, cls_b)
    sign = 1 if direction == "max" else -1
    n = len(y)
    ra, rb = C.per_row(m, y, pa, cls_a), C.per_row(m, y, pb, cls_b)
    if ra is not None and m != "rmse":
        rs = -1 if m in ("logloss", "mse", "mae", "brier") else 1
        d = rs * (rb - ra); mean = d.mean(); sem = d.std(ddof=1) / np.sqrt(n)
        t = mean / sem if sem > 0 else 0.0; better, worse = int((d > 0).sum()), int((d < 0).sum())
        how = "per-row"
    else:
        ds = np.empty(a.boot)
        for i in range(a.boot):
            ix = rng.integers(0, n, n)
            try: ds[i] = sign * (C.score(m, y[ix], pb[ix], cls_b) - C.score(m, y[ix], pa[ix], cls_a))
            except ValueError: ds[i] = np.nan
        ds = ds[~np.isnan(ds)]; mean = sign * (sb - sa); sem = ds.std(ddof=1)
        t = mean / sem if sem > 0 else 0.0; better = int((ds > 0).sum()); worse = int((ds < 0).sum())
        how = f"paired bootstrap x{len(ds)} (same resampled rows for both)"
    print(f"metric {m} ({direction}) | rows {n} | baseline {sa:.6f} | cand {sb:.6f} | delta {sb-sa:+.6f}")
    print(f"{how}: favourable delta {mean:+.6f} | sd {sem:.6f} | t {t:+.2f} | "
          + (f"rows better {better}/{n}, worse {worse}" if how == "per-row" else f"P(cand better) {better/max(1,len(ds)):.2f}"))

if t >= a.tmin and mean > 0:
    print(f"VERDICT: BETTER (t {t:.2f} >= {a.tmin}).")
    sys.exit(0)
print(f"VERDICT: DIFFERENT, NOT BETTER (t {t:.2f} < {a.tmin} or delta <= 0).")
sys.exit(1)
