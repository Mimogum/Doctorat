import numpy as np

from .aa import _fit
from .simplex import solve

# ============================================================
# Robust Archetypal Analysis, row-wise Huber (Chen et al. 2014, Alg. 3)
#
#   min sum_i ||r_i|| via IRLS: scale v_i = max(||r_i||, epsilon),
#   weight w_i = 1 / v_i, then the aa core with those weights.
# ============================================================


def robust_aa(X, k, max_robust_iter=20, max_inner_iter=50, epsilon=0.01,
              tol=1e-5, seed=42, verbose=False, solver="nnls"):
    """
    Returns (S, C, Z, info) with info = {rss, weights, n_iter}.
    weights (n,) in [0, 1]: ~0 marks an outlier row. solver: see simplex.py.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")

    n = X.shape[0]
    # Random start, NOT FurthestSum: that one deliberately picks the most
    # extreme points, so outliers would start as archetypes with residual 0
    # and weight 1, and IRLS could no longer discard them.
    rng = np.random.default_rng(seed)
    C = np.zeros((k, n))
    C[np.arange(k), rng.choice(n, size=k, replace=n < k)] = 1.0
    v = np.ones(n)

    for t in range(max_robust_iter):
        v_old = v.copy()

        # 1. Per-row Huber scales from the current fit
        S = solve(X, (C @ X).T, solver)
        v = np.maximum(np.linalg.norm(X - S @ C @ X, axis=1), epsilon)

        # 2. Weighted AA with w = 1 / v (warm-started from the current C)
        S, C, history = _fit(X, C, 1.0 / v, max_inner_iter, tol, solver)

        # Outer stop: the scales barely changed
        weight_diff = np.linalg.norm(v - v_old) / (np.linalg.norm(v_old) + 1e-10)
        if verbose:
            print(f"Robust iteration {t + 1}/{max_robust_iter} "
                  f"(inner: {len(history)}) | weight diff: {weight_diff:.6f}")
        if weight_diff < tol:
            break

    Z = C @ X
    # Median normalisation: a single perfect row does not overshadow the rest
    weights = np.clip(np.median(v) / v, 0.0, 1.0)
    info = {"rss": np.sum((X - S @ Z) ** 2), "weights": weights, "n_iter": t + 1}
    return S, C, Z, info
