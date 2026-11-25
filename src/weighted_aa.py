import numpy as np

from .aa import aa
from .simplex import solve

# ============================================================
# Archetypal Analysis ponderado + postprocesado
#
#   1. AA sobre los datos ponderados diag(w) X.
#   2. Postprocesado sobre la X original:
#        C: min ||Z_w - C X||^2   (proyecta los arquetipos ponderados
#                                  sobre la envolvente convexa de X)
#        S: min ||X - S Z||^2     con Z = C X
# ============================================================


def _validate_weights(w, n):
    w = np.asarray(w, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"w debe tener forma ({n},).")
    if not np.all(np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("w debe ser finito, no negativo y con algún peso > 0.")
    return w


def weighted_aa(X, k, w, max_iter=100, tol=1e-6, seed=42, verbose=False, solver="nnls"):
    """
    w : (n,) peso de cada observación. solver: ver simplex.py.

    Devuelve (S, C, Z, info) sobre la X original, con Z = C @ X.
    info = {rss, history, n_iter}: rss sobre X; el resto es el ajuste
    sobre los datos ponderados.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X debe ser una matriz 2D.")
    w = _validate_weights(w, X.shape[0])

    _, _, Z_w, info = aa(w[:, None] * X, k, max_iter=max_iter, tol=tol,
                         seed=seed, verbose=verbose, solver=solver)

    C = solve(Z_w, X.T, solver)
    Z = C @ X
    S = solve(X, Z.T, solver)

    info = {**info, "rss": np.sum((X - S @ Z) ** 2)}
    return S, C, Z, info
