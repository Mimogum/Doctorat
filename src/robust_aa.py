import numpy as np

from .aa import _fit
from .simplex import solve

# ============================================================
# Archetypal Analysis robusto, Huber por filas (Chen et al. 2014, Alg. 3)
#
#   min sum_i ||r_i|| mediante IRLS: escala v_i = max(||r_i||, epsilon),
#   peso w_i = 1 / v_i, y el núcleo de aa con esos pesos.
# ============================================================


def robust_aa(X, k, max_robust_iter=20, max_inner_iter=50, epsilon=0.01,
              tol=1e-5, seed=42, verbose=False, solver="nnls"):
    """
    Devuelve (S, C, Z, info) con info = {rss, weights, n_iter}.
    weights (n,) en [0, 1]: ~0 indica fila outlier. solver: ver simplex.py.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X debe ser una matriz 2D.")

    n = X.shape[0]
    # Inicio aleatorio, NO FurthestSum: este elige a propósito los puntos más
    # extremos, los outliers empiezan como arquetipos con residuo 0 y peso 1,
    # y el IRLS ya no puede descartarlos.
    rng = np.random.default_rng(seed)
    C = np.zeros((k, n))
    C[np.arange(k), rng.choice(n, size=k, replace=n < k)] = 1.0
    v = np.ones(n)

    for t in range(max_robust_iter):
        v_old = v.copy()

        # 1. Escalas Huber por fila con el ajuste actual
        S = solve(X, (C @ X).T, solver)
        v = np.maximum(np.linalg.norm(X - S @ C @ X, axis=1), epsilon)

        # 2. AA ponderado con w = 1 / v
        S, C, history = _fit(X, C, 1.0 / v, max_inner_iter, tol, solver)

        weight_diff = np.linalg.norm(v - v_old) / (np.linalg.norm(v_old) + 1e-10)
        if verbose:
            print(f"Iteración robusta {t + 1}/{max_robust_iter} "
                  f"(internas: {len(history)}) | dif. pesos: {weight_diff:.6f}")
        if weight_diff < tol:
            break

    Z = C @ X
    # Normalización con la mediana: una sola fila perfecta no oscurece el resto
    weights = np.clip(np.median(v) / v, 0.0, 1.0)
    info = {"rss": np.sum((X - S @ Z) ** 2), "weights": weights, "n_iter": t + 1}
    return S, C, Z, info
