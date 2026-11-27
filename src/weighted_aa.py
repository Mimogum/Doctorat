import numpy as np

from .aa import aa
from .simplex import solve

# ============================================================
# Weighted Archetypal Analysis + post-processing
#
#   1. AA on the weighted data diag(w) X.
#   2. Post-processing on the original X:
#        C: min ||Z_w - C X||^2   (projects the weighted archetypes
#                                  onto the convex hull of X)
#        S: min ||X - S Z||^2     with Z = C X
# ============================================================


def _validate_weights(w, n):
    """Check w is a finite, non-negative (n,) vector with at least one weight > 0."""
    w = np.asarray(w, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"w must have shape ({n},).")
    if not np.all(np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("w must be finite, non-negative and have at least one weight > 0.")
    return w


def weighted_aa(X, k, w, max_iter=100, tol=1e-6, seed=42, verbose=False, solver="nnls"):
    """
    w : (n,) weight of each observation. solver: see simplex.py.

    Returns (S, C, Z, info) on the original X, with Z = C @ X.
    info = {rss, history, n_iter}: rss is on X; the rest describes the fit
    on the weighted data.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")
    w = _validate_weights(w, X.shape[0])

    # 1. Plain AA on the rescaled rows; its archetypes Z_w live in weighted space
    _, _, Z_w, info = aa(w[:, None] * X, k, max_iter=max_iter, tol=tol,
                         seed=seed, verbose=verbose, solver=solver)

    # 2. Bring the archetypes back to the original X and recompute S there
    C = solve(Z_w, X.T, solver)
    Z = C @ X
    S = solve(X, Z.T, solver)

    info = {**info, "rss": np.sum((X - S @ Z) ** 2)}
    return S, C, Z, info
