import numpy as np

from .biaa import _fit, _init, _update_S0

# ============================================================
# Robust Biarchetypal Analysis, cell-wise Huber (Chen et al. 2014)
#
#   Cell-by-cell IRLS: scale v_ij = max(|r_ij|, epsilon),
#   weight W_ij = 1 / v_ij, then the biaa core with those weights.
# ============================================================


def robust_biaa(X, k, c, max_robust_iter=20, max_inner_iter=50, epsilon=0.01,
                tol=1e-5, seed=42, verbose=False, solver="nnls"):
    """
    Returns (S0, C0, S1, C1, Z, info) with info = {rss, weights, n_iter}.
    weights (n, m) in [0, 1]: ~0 marks an outlier cell. solver: see simplex.py.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")

    C0, C1, S1 = _init(X, k, c, seed)
    V = np.ones_like(X)  # per-cell Huber scales

    for t in range(max_robust_iter):
        V_old = V.copy()

        # 1. Per-cell Huber scales from the current fit
        Z = C0 @ X @ C1.T
        S0 = _update_S0(X, Z, S1, 1.0 / V, solver)
        V = np.maximum(np.abs(X - S0 @ Z @ S1.T), epsilon)

        # 2. Weighted BiAA with W = 1 / V (warm-started from the current C0, C1, S1)
        S0, C0, S1, C1, history = _fit(X, C0, C1, S1, 1.0 / V, max_inner_iter, tol, solver)

        # Outer stop: the scales barely changed
        weight_diff = np.linalg.norm(V - V_old) / (np.linalg.norm(V_old) + 1e-10)
        if verbose:
            print(f"Robust iteration {t + 1}/{max_robust_iter} "
                  f"(inner: {len(history)}) | weight diff: {weight_diff:.6f}")
        if weight_diff < tol:
            break

    Z = C0 @ X @ C1.T
    # Median normalisation: a single perfect cell does not overshadow the rest
    weights = np.clip(np.median(V) / V, 0.0, 1.0)
    info = {"rss": np.sum((X - S0 @ Z @ S1.T) ** 2), "weights": weights, "n_iter": t + 1}
    return S0, C0, S1, C1, Z, info
