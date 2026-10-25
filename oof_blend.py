# oof_blend.py - is an ensemble worth it, and with which weights? Honest split-half check, dev-time only.
#   python oof_blend.py --truth holdout_truth.csv --target y --metric auc --preds a.csv b.csv c.csv [--id id] [--rank]
# Prints each member's score, pairwise correlation (errors for decodable metrics, predictions otherwise), equal-weight
# mean, and Caruana greedy weights fit on one half and scored on the OTHER half. Ship the weights only if they beat equal
# weights on the held half; bake them into solution.py as a FIXED list derived in-code or documented as the recipe.
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comp_metrics as C

ap = argparse.ArgumentParser()
ap.add_argument("--truth", required=True); ap.add_argument("--target", required=True)
ap.add_argument("--metric", required=True); ap.add_argument("--preds", nargs="+", required=True)
ap.add_argument("--id", default="id"); ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--rank", action="store_true", help="rank-average scalar predictions (AUC-type metrics only)")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
m = C.canon(a.metric); sgn = 1 if C.DIRECTION[m] == "max" else -1
tr = C.load_truth(a.truth, a.id, a.target); ids = tr[a.id].tolist(); y = tr["__y"].to_numpy()
P, cls = [], None
for f in a.preds:
    p, c = C.as_matrix(C.load_pred(f, a.id, ids, a.target), y)
    if P and c != cls:
        sys.exit(f"{f}: column layout differs from {a.preds[0]}")
    cls = c
    if a.rank and c is None:
        p = (np.argsort(np.argsort(p)) + 0.5) / len(p)
    P.append(p)
names = [os.path.basename(f) for f in a.preds]
n, k = len(y), len(P)
S = lambda idx, w: C.score(m, y[idx], sum(wi * P[i][idx] for i, wi in enumerate(w)) / sum(w), cls)
full = np.arange(n)

print(f"{m} ({C.DIRECTION[m]}) on {n} rows")
for nm, i in zip(names, range(k)):
    print(f"  {nm:40} {S(full, np.eye(k)[i]):.6f}")
eq = S(full, np.ones(k)); print(f"  {'EQUAL-WEIGHT MEAN':40} {eq:.6f}  (probability space; never average log-probs when seeds can collapse)")

rows = [C.per_row(m, y, p, cls) for p in P]
V = [r if r is not None else (p if p.ndim == 1 else p.ravel()) for r, p in zip(rows, P)]
what = "per-row loss" if rows[0] is not None else "predictions"
print(f"pairwise correlation of {what} (~0.8 across two FAMILIES pays; >=0.97 is the same model twice):")
for i in range(k):
    print("  " + " ".join(f"{np.corrcoef(V[i], V[j])[0, 1]:6.3f}" for j in range(k)) + f"  {names[i][:30]}")


def greedy(idx):
    w = np.zeros(k); best = None
    for _ in range(a.steps):
        cand = []
        for i in range(k):
            w2 = w.copy(); w2[i] += 1
            cand.append(sgn * S(idx, w2))
        i = int(np.argmax(cand))
        if best is not None and cand[i] <= best + 1e-12: break
        w[i] += 1; best = cand[i]
    return w


rng = np.random.default_rng(a.seed); perm = rng.permutation(n); A, B = perm[: n // 2], perm[n // 2:]
out = []
for fit, ev, tag in ((A, B, "fit A -> eval B"), (B, A, "fit B -> eval A")):
    w = greedy(fit); g, e = S(ev, w), S(ev, np.ones(k)); out.append(sgn * (g - e))
    print(f"  {tag}: greedy w={np.round(w / w.sum(), 3).tolist()} {g:.6f} vs equal {e:.6f} ({'+' if sgn*(g-e) >= 0 else ''}{sgn*(g-e):.6f})")
wf = greedy(full)
print(f"greedy on ALL rows (in-sample, optimistic): w={np.round(wf / wf.sum(), 3).tolist()} {S(full, wf):.6f}")
if min(out) > 0:
    print("VERDICT: learned weights beat equal weights on BOTH held halves - ship the full-data weights as a fixed list.")
else:
    print("VERDICT: learned weights do not beat equal weights out-of-half - ship EQUAL weights (or drop the weak member).")
best_single = max(sgn * S(full, np.eye(k)[i]) for i in range(k))
if sgn * eq <= best_single:
    print("NOTE: the equal-weight blend does not beat the best single member - the members are not diverse enough; "
          "add a different FAMILY / input view / target parameterisation rather than more seeds.")
