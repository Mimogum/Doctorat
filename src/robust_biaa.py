import numpy as np

from .biaa import _fit, _init, _update_S0

# ============================================================
# Biarchetypal Analysis robusto, Huber por celdas (Chen et al. 2014)
#
#   IRLS celda a celda: escala v_ij = max(|r_ij|, epsilon),
#   peso W_ij = 1 / v_ij, y el núcleo de biaa con esos pesos.
# ============================================================


def robust_biaa(X, k, c, max_robust_iter=20, max_inner_iter=50, epsilon=0.01,
                tol=1e-5, seed=42, verbose=False, solver="nnls"):
    """
    Devuelve (S0, C0, S1, C1, Z, info) con info = {rss, weights, n_iter}.
    weights (n, m) en [0, 1]: ~0 indica celda outlier. solver: ver simplex.py.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X debe ser una matriz 2D.")

    C0, C1, S1 = _init(X, k, c, seed)
    V = np.ones_like(X)

    for t in range(max_robust_iter):
        V_old = V.copy()

        # 1. Escalas Huber por celda con el ajuste actual
        Z = C0 @ X @ C1.T
        S0 = _update_S0(X, Z, S1, 1.0 / V, solver)
        V = np.maximum(np.abs(X - S0 @ Z @ S1.T), epsilon)

        # 2. BiAA ponderado con W = 1 / V
        S0, C0, S1, C1, history = _fit(X, C0, C1, S1, 1.0 / V, max_inner_iter, tol, solver)

        weight_diff = np.linalg.norm(V - V_old) / (np.linalg.norm(V_old) + 1e-10)
        if verbose:
            print(f"Iteración robusta {t + 1}/{max_robust_iter} "
                  f"(internas: {len(history)}) | dif. pesos: {weight_diff:.6f}")
        if weight_diff < tol:
            break

    Z = C0 @ X @ C1.T
    # Normalización con la mediana: una sola celda perfecta no oscurece el resto
    weights = np.clip(np.median(V) / V, 0.0, 1.0)
    info = {"rss": np.sum((X - S0 @ Z @ S1.T) ** 2), "weights": weights, "n_iter": t + 1}
    return S0, C0, S1, C1, Z, info
