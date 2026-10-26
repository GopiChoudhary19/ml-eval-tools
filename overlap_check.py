# overlap_check.py - detect stride-k overlapping windows in a (rows, T, C) sequence dataset BEFORE any within-block CV split.
# When rows are sliding windows over one long series, neighbouring rows share most of their values, so a random-row
# split puts near-copies on both sides and validation scores look better than they are.
# usage: python overlap_check.py data.npz [--key X] [--rows 1500] [--maxshift 12]
#        or a csv with a ';'-separated sequence column: python overlap_check.py train.csv --col sequence --T 36 --C 2
# exit 1 = OVERLAP FOUND (use segment/group folds, never random rows) | 0 = no overlap detected
import argparse, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('path'); ap.add_argument('--key', default='X'); ap.add_argument('--col', default='sequence')
ap.add_argument('--T', type=int, default=0); ap.add_argument('--C', type=int, default=1)
ap.add_argument('--rows', type=int, default=1500); ap.add_argument('--maxshift', type=int, default=12)
a = ap.parse_args()
if a.path.endswith('.npz'):
    X = np.load(a.path, allow_pickle=True)[a.key].astype(np.float32)
else:
    import pandas as pd
    s = pd.read_csv(a.path, usecols=[a.col])[a.col]
    X = np.array([np.array(v.split(';'), dtype=np.float32) for v in s]).reshape(len(s), a.T, a.C)
if X.ndim == 2: X = X[..., None]
n, T = X.shape[:2]
sub = np.random.default_rng(0).choice(n, min(a.rows, n), replace=False)
hit = []
for k in range(1, min(a.maxshift, T - 4) + 1):
    A = X[sub][:, k:]; B = X[:, :T - k]
    A = (A - A.mean(1, keepdims=True)).reshape(len(sub), -1); B = (B - B.mean(1, keepdims=True)).reshape(n, -1)
    best = np.empty((len(sub), 2)); bb = (B ** 2).sum(1); m = A.shape[1]
    for i in range(0, len(sub), 200):
        Ai = A[i:i + 200]; d = ((Ai ** 2).sum(1)[:, None] + bb[None] - 2 * Ai @ B.T) / m; d[np.arange(len(d)), sub[i:i + 200]] = np.inf
        best[i:i + 200] = np.sort(d, 1)[:, :2]
    ratio = np.median(best[:, 1] / np.maximum(best[:, 0], 1e-9)); v = np.median(best[:, 0]) / max(float(A.var()), 1e-9)
    flag = ratio > 5 and v < 0.05
    print(f'shift {k:2d}: NN mse/var {v:.4f}  2nd/1st {ratio:8.2f}  {"<- OVERLAP" if flag else ""}')
    if flag: hit.append(k)
if hit:
    print(f'OVERLAP at shifts {hit}: rows are overlapping windows (stride {min(hit)}). Random-row CV LEAKS. Group rows into '
          'contiguous segments (chain by shifted-window match) and split by segment; keep test rows independent.')
    sys.exit(1)
print('no shifted-window overlap detected'); sys.exit(0)
