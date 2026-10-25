# comp_metrics.py - shared metric registry + prediction loader for paired_delta.py / oof_blend.py
import sys
import numpy as np
import pandas as pd

# name -> (direction, per_row_fn or None). per_row_fn(y, p) -> per-row contribution whose MEAN is the metric
# (only for decomposable metrics); everything else goes through the whole-set function in score().
DIRECTION = {
    "auc": "max", "acc": "max", "f1macro": "max", "f1micro": "max", "r2": "max", "ap": "max",
    "logloss": "min", "rmse": "min", "mae": "min", "mse": "min", "brier": "min",
}
ALIASES = {"accuracy": "acc", "roc_auc": "auc", "rocauc": "auc", "log_loss": "logloss", "ce": "logloss",
           "f1": "f1macro", "map": "ap", "average_precision": "ap"}


def canon(name):
    n = name.lower().replace("-", "_")
    n = ALIASES.get(n, n)
    if n not in DIRECTION:
        sys.exit(f"unknown metric '{name}'. Built in: {sorted(DIRECTION)}. For anything else (MAP@k, NDCG, IoU, a"
                 f" custom row score) write per-row scores to a CSV and use --col mode.")
    return n


def load_truth(path, idcol, target):
    t = pd.read_csv(path, keep_default_na=False)
    if target not in t.columns:
        sys.exit(f"{path}: no column '{target}' (have {list(t.columns)[:12]})")
    return t[[idcol, target]].rename(columns={target: "__y"})


def load_pred(path, idcol, truth_ids, target=None):
    p = pd.read_csv(path)
    if idcol not in p.columns:
        sys.exit(f"{path}: no id column '{idcol}'")
    p = p.set_index(idcol)
    missing = set(truth_ids) - set(p.index)
    if missing:
        sys.exit(f"{path}: {len(missing)} truth ids have no prediction (e.g. {list(missing)[:3]}) - pairing is void")
    p = p.loc[truth_ids]
    cols = [c for c in p.columns if c != target] if target in p.columns and p.shape[1] > 1 else list(p.columns)
    return p[cols]


def as_matrix(pred, y):
    # scalar prediction -> 1-D; several columns -> class-probability matrix aligned to sorted class labels
    if pred.shape[1] == 1:
        return pred.iloc[:, 0].to_numpy(dtype=float), None
    classes = [str(c) for c in pred.columns]
    return pred.to_numpy(dtype=float), classes


def score(metric, y, p, classes=None):
    from sklearn import metrics as M
    if metric in ("rmse", "mse", "mae", "r2"):
        y = np.asarray(y, dtype=float)
        if metric == "rmse": return float(np.sqrt(np.mean((y - p) ** 2)))
        if metric == "mse": return float(np.mean((y - p) ** 2))
        if metric == "mae": return float(np.mean(np.abs(y - p)))
        return float(M.r2_score(y, p))
    if classes is None:  # binary, p = P(positive)
        yb = _binary(y)
        if metric == "auc": return float(M.roc_auc_score(yb, p))
        if metric == "ap": return float(M.average_precision_score(yb, p))
        if metric == "logloss": return float(M.log_loss(yb, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]))
        if metric == "brier": return float(np.mean((yb - p) ** 2))
        pred = (p >= 0.5).astype(int)
        if metric == "acc": return float(np.mean(pred == yb))
        return float(M.f1_score(yb, pred, average="macro" if metric == "f1macro" else "micro"))
    ys = np.asarray([str(v) for v in y])
    idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([idx.get(v, -1) for v in ys])
    if (yi < 0).any():
        sys.exit(f"truth labels not among prediction columns: {sorted(set(ys[yi < 0]))[:5]}")
    P = np.clip(p, 1e-15, 1)
    P = P / P.sum(1, keepdims=True)
    if metric == "logloss": return float(-np.mean(np.log(P[np.arange(len(yi)), yi])))
    if metric == "brier": return float(np.mean(np.sum((P - np.eye(len(classes))[yi]) ** 2, 1)))
    if metric == "auc": return float(M.roc_auc_score(yi, P, multi_class="ovr", average="macro", labels=list(range(len(classes)))))
    pred = P.argmax(1)
    if metric == "acc": return float(np.mean(pred == yi))
    if metric in ("f1macro", "f1micro"): return float(M.f1_score(yi, pred, average=metric[2:]))
    sys.exit(f"metric {metric} not defined for multiclass here - use --col mode")


def per_row(metric, y, p, classes=None):
    # per-row contributions for decomposable metrics, else None (-> bootstrap)
    if metric in ("mse", "mae", "rmse"):
        y = np.asarray(y, dtype=float)
        return (y - p) ** 2 if metric != "mae" else np.abs(y - p)
    if metric in ("logloss", "acc", "brier"):
        if classes is None:
            yb = _binary(y); q = np.clip(p, 1e-15, 1 - 1e-15)
            if metric == "logloss": return -(yb * np.log(q) + (1 - yb) * np.log(1 - q))
            if metric == "brier": return (yb - p) ** 2
            return ((p >= 0.5).astype(int) == yb).astype(float)
        ys = [str(v) for v in y]; idx = {c: i for i, c in enumerate(classes)}; yi = np.array([idx[v] for v in ys])
        P = np.clip(p, 1e-15, 1); P = P / P.sum(1, keepdims=True)
        if metric == "logloss": return -np.log(P[np.arange(len(yi)), yi])
        if metric == "brier": return np.sum((P - np.eye(len(classes))[yi]) ** 2, 1)
        return (P.argmax(1) == yi).astype(float)
    return None


def _binary(y):
    v = pd.Series(y).astype(str)
    u = sorted(v.unique())
    if len(u) > 2:
        sys.exit(f"binary metric but {len(u)} label values - supply one probability column per class")
    pos = "1" if "1" in u else ("True" if "True" in u else u[-1])
    return (v == pos).astype(int).to_numpy()
