"""Small statistics: bootstrap intervals, AUC, isotonic regression and monotonicity tests."""
import numpy as np


def bootstrap_ci(x, n=2000, seed=0):
    """(mean, 2.5 % quantile, 97.5 % quantile) of the mean, by bootstrap over the entries of x."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def auc(score, label):
    """Area under the ROC curve = P(score of a positive > score of a negative), ties counted 1/2."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):                                     # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_ci(score, label, n=2000, seed=0):
    """(AUC, 2.5 %, 97.5 %), bootstrap over the items."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n):
        k = rng.integers(0, len(s), len(s))
        v = auc(s[k], y[k])
        if np.isfinite(v):
            b.append(v)
    return auc(s, y), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def conditional_auc(score, label, strata, n_strata=5):
    """AUC of ``score`` within quantile bins of ``strata`` (e.g. the threshold at fixed position), averaged with
    weights n_pos * n_neg of each bin."""
    s, y, z = map(np.asarray, (score, label, strata))
    edges = np.quantile(z, np.linspace(0, 1, n_strata + 1))
    num = den = 0.0
    for k in range(n_strata):
        sel = (z >= edges[k]) & ((z < edges[k + 1]) if k < n_strata - 1 else (z <= edges[k + 1]))
        w = float(y[sel].sum() * (~y[sel].astype(bool)).sum())
        if w > 0:
            num += w * auc(s[sel], y[sel])
            den += w
    return num / den if den else float("nan")


def isotonic(y, w, increasing=True):
    """Weighted isotonic regression (pool adjacent violators)."""
    y, w = np.asarray(y, float), np.asarray(w, float)
    if not increasing:
        return isotonic(y[::-1], w[::-1])[::-1]
    blocks = []
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, l2 = blocks.pop()
            m1, w1, l1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, l1 + l2])
    out = []
    for m_, _, l_ in blocks:
        out += [m_] * l_
    return np.array(out)


def max_drop(y):
    """Largest drop of a curve: max over t < t' of y_t - y_t' (0 if it never goes down)."""
    best, top = 0.0, -np.inf
    for v in y:
        top = max(top, v)
        best = max(best, top - v)
    return float(best)


def monotonicity_test(k, n, n_boot, rng):
    """Is the curve of binary proportions k_t / n_t increasing? Statistic: largest drop D. Parametric bootstrap
    from the isotonic fit (Bernoulli responses, same n_t), valid because the data are binary.
    Returns (p-value of D, isotonic fit, D)."""
    y = k / n
    iso = isotonic(y, n)
    D = max_drop(y)
    p = np.clip(iso, 0, 1)
    c = 0
    for _ in range(n_boot):
        c += max_drop(rng.binomial(n.astype(int), p) / n) >= D - 1e-12
    return (1 + c) / (1 + n_boot), iso, D


def benjamini_hochberg(p, alpha=0.10):
    """Number of rejections at false discovery rate alpha."""
    p = np.asarray(p)
    o = np.argsort(p)
    ok = p[o] <= alpha * np.arange(1, len(p) + 1) / len(p)
    return int(np.max(np.nonzero(ok)[0]) + 1) if ok.any() else 0


def unit(v):
    v = np.asarray(v, float).reshape(-1)
    return v / max(np.linalg.norm(v), 1e-30)
