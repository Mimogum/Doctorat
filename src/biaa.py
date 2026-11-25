import numpy as np

from .simplex import solve

# ============================================================
# Biarchetypal Analysis
#
#   X ≈ S0 Z S1.T,   Z = C0 X C1.T
#   X : (n, m)   Z : (k, c)
#   S0 : (n, k)  C0 : (k, n)   -> observaciones
#   S1 : (m, c)  C1 : (c, m)   -> variables
#   Todas >= 0 y cada fila suma 1.
#
# Alterna S0, S1 (proyecciones por filas/columnas) y C0, C1 (descenso por
# coordenadas). El núcleo admite pesos por celda W (n, m): biaa usa W = 1 y
# robust_biaa los pesos de Huber. Todo subproblema va por simplex.solve.
# ============================================================


def _update_S0(X, Z, S1, W, solver, S0=None):
    """Cada fila x_i sobre las k filas de Z S1.T."""
    return solve(X, (Z @ S1.T).T, solver, W, A0=S0)


def _update_S1(X, S0, Z, W, solver, S1=None):
    """Cada columna x_:j sobre las c columnas de S0 Z."""
    return solve(X.T, S0 @ Z, solver, W.T, A0=S1)


def _update_C0(X, S0, C0, S1, C1, W, solver):
    """
    C0 (k, n) por coordenadas. Con H = X C1.T S1.T, la fila g de Z S1.T es
    C0[g] H y, con el resto fijo:
      sum_ij W_ij (r_ij + s_ig (h_j - h'_j))^2 = sum_j denom_j (target_j - h'_j)^2
    """
    H = X @ C1.T @ S1.T
    R = X - S0 @ C0 @ H

    for g in range(C0.shape[0]):
        s = S0[:, [g]]
        h = C0[g] @ H
        denom = np.sum(W * s ** 2, axis=0)
        target = np.where(denom > 1e-12, np.sum(W * s * R, axis=0) / np.maximum(denom, 1e-12) + h, h)

        C0[g] = solve(target, H.T, solver, np.maximum(denom, 1e-12)[None], A0=C0[g])[0]
        R += s * (h - C0[g] @ H)

    return C0


def _update_C1(X, S0, C0, S1, C1, W, solver):
    """C1 (c, m) por coordenadas; columna h de S0 Z = B C1[h], con B = S0 C0 X."""
    B = S0 @ C0 @ X
    R = X - B @ C1.T @ S1.T

    for h in range(C1.shape[0]):
        s = S1[:, h][None, :]
        b = B @ C1[h]
        denom = np.sum(W * s ** 2, axis=1)
        target = np.where(denom > 1e-12, np.sum(W * R * s, axis=1) / np.maximum(denom, 1e-12) + b, b)

        C1[h] = solve(target, B, solver, np.maximum(denom, 1e-12)[None], A0=C1[h])[0]
        R += (b - B @ C1[h])[:, None] * s

    return C1


def _init(X, k, c, seed):
    n, m = X.shape
    if not 1 <= k <= n:
        raise ValueError("k debe cumplir 1 <= k <= n.")
    if not 1 <= c <= m:
        raise ValueError("c debe cumplir 1 <= c <= m.")
    rng = np.random.default_rng(seed)
    C0 = rng.dirichlet(np.ones(n), size=k)
    C1 = rng.dirichlet(np.ones(m), size=c)
    S1 = np.full((m, c), 1.0 / c)
    return C0, C1, S1


def _fit(X, C0, C1, S1, W, max_iter, tol, solver, verbose=False):
    """
    Núcleo compartido por biaa y robust_biaa: minimiza
    sum_ij W_ij (X - S0 C0 X C1.T S1.T)_ij^2. Devuelve S0, C0, S1, C1, historial.
    """
    history, S0 = [], None
    for it in range(max_iter):
        Z = C0 @ X @ C1.T
        S0 = _update_S0(X, Z, S1, W, solver, S0)
        S1 = _update_S1(X, S0, Z, W, solver, S1)
        C0 = _update_C0(X, S0, C0, S1, C1, W, solver)
        C1 = _update_C1(X, S0, C0, S1, C1, W, solver)
        history.append(np.sum(W * (X - S0 @ C0 @ X @ C1.T @ S1.T) ** 2))

        # Se para cuando Z deja de moverse (ver aa._fit).
        change = np.linalg.norm(C0 @ X @ C1.T - Z) / (np.linalg.norm(Z) + 1e-10)
        if verbose:
            print(f"Iteración {it + 1:3d} | pérdida = {history[-1]:.6e} | cambio = {change:.2e}")
        if change < tol:
            break

    # S0, S1 coherentes con la Z final
    Z = C0 @ X @ C1.T
    S0 = _update_S0(X, Z, S1, W, solver, S0)
    S1 = _update_S1(X, S0, Z, W, solver, S1)
    return S0, C0, S1, C1, np.asarray(history)


def biaa(X, k, c, max_iter=100, tol=1e-6, seed=42, verbose=False, solver="nnls"):
    """
    Biarchetypal Analysis con k arquetipos de observaciones y c de variables.
    solver: "nnls", "slsqp" o "pgd" (ver simplex.py).

    Devuelve (S0, C0, S1, C1, Z, info) con info = {rss, history, n_iter}.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X debe ser una matriz 2D.")

    C0, C1, S1 = _init(X, k, c, seed)
    S0, C0, S1, C1, history = _fit(X, C0, C1, S1, np.ones_like(X), max_iter, tol, solver, verbose)
    Z = C0 @ X @ C1.T

    info = {"rss": np.sum((X - S0 @ Z @ S1.T) ** 2), "history": history, "n_iter": len(history)}
    return S0, C0, S1, C1, Z, info
