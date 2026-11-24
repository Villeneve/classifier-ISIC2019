"""Métricas de classificação binária em numpy (classe positiva = maligno)."""
import numpy as np


def _rankdata(a):
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    _, inv, counts = np.unique(sa, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    cs = np.cumsum(counts)
    avg = (cs - counts + 1 + cs) / 2.0
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = avg[inv]
    return ranks


def roc_auc(y, p):
    y = np.asarray(y).astype(int)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rankdata(np.asarray(p, dtype=np.float64))
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(y, p):
    """Retorna fpr, tpr, thresholds (predição positiva se p >= thr)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    tps = np.cumsum(ys)
    fps = np.cumsum(1 - ys)
    idx = np.r_[np.where(np.diff(ps))[0], len(ps) - 1]
    tpr = np.r_[0.0, tps[idx] / max(tps[-1], 1)]
    fpr = np.r_[0.0, fps[idx] / max(fps[-1], 1)]
    thr = np.r_[np.inf, ps[idx]]
    return fpr, tpr, thr


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def compute_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    return {
        "threshold": float(thr),
        "accuracy": (tp + tn) / max(len(y), 1),
        "balanced_accuracy": (sens + spec) / 2,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "auc": roc_auc(y, p),
        "confusion": [[tn, fp], [fn, tp]],  # linhas = real (benigno, maligno)
    }
