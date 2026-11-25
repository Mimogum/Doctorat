import numpy as np

from .simplex import solve

# ============================================================
# Archetypal Analysis
#
#   X ≈ S Z,   Z = C X
#   X : (n, m)   S : (n, k)   C : (k, n)   Z : (k, m)
#   S, C >= 0 y cada fila suma 1.
#
# Alterna S (cada fila de X sobre los arquetipos) y C (descenso por
# coordenadas, arquetipo a arquetipo; Cutler & Breiman 1994). Todos los
# subproblemas los resuelve simplex.solve con el `solver` elegido.
# ============================================================


def furthest_sum(X, k, rng):
    """
    FURTHESTSUM (Mørup & Hansen 2012): k índices de observaciones extremas.
    Se parte de un punto aleatorio, que al final se descarta.
    """
    n = X.shape[0]
    if not 1 <= k <= n:
        raise ValueError("k debe cumplir 1 <= k <= n.")

    j = int(rng.integers(n))
    selected = [j]
    q = np.linalg.norm(X - X[j], axis=1)
    q_first = q.copy()

    for t in range(1, k + 1):
        if t == k:
            q = q - q_first  # se elimina la influencia del punto aleatorio
        available = np.ones(n, dtype=bool)
        available[selected] = False
        j = int(np.argmax(np.where(available, q, -np.inf)))
        selected.append(j)
        q = q + np.linalg.norm(X - X[j], axis=1)

    return np.asarray(selected[1:], dtype=int)


def _update_C(X, S, C, w, solver):
    """
    C (k, n) por descenso por coordenadas con pesos por fila w (n,).
    Para el arquetipo j, con el resto fijo:
      sum_i w_i ||r_i + s_ij (z_j - z')||^2 = denom ||target - z'||^2 + cte
    y z' = C[j] X se proyecta sobre la envolvente convexa de X.
    """
    Z = C @ X
    R = X - S @ Z

    for j in range(C.shape[0]):
        s_j = S[:, j]
        z_j = Z[j].copy()

        denom = np.sum(w * s_j ** 2)
        target = (w * s_j) @ R / denom + z_j if denom > 1e-12 else z_j

        C[j] = solve(target, X.T, solver, A0=C[j])[0]
        Z[j] = C[j] @ X
        R += np.outer(s_j, z_j - Z[j])

    return C


def _fit(X, C, w, max_iter, tol, solver, verbose=False):
    """
    Nucleo compartido por aa y robust_aa: minimiza sum_i w_i ||x_i - s_i C X||^2
    alternando S y C desde C. Devuelve S, C y el historial de la pérdida.
    """
    history, S = [], None
    for it in range(max_iter):
        Z_old = C @ X
        S = solve(X, Z_old.T, solver, A0=S)  # el peso de cada fila no cambia su óptimo
        C = _update_C(X, S, C, w, solver)
        history.append(np.sum(w * np.sum((X - S @ C @ X) ** 2, axis=1)))

        # Se para cuando Z deja de moverse: la pérdida puede estancarse un
        # momento con Z aún lejos (robust_aa paraba antes de tiempo).
        change = np.linalg.norm(C @ X - Z_old) / (np.linalg.norm(Z_old) + 1e-10)
        if verbose:
            print(f"Iteración {it + 1:4d} | pérdida = {history[-1]:.6e} | cambio = {change:.2e}")
        if change < tol:
            break

    S = solve(X, (C @ X).T, solver, A0=S)  # S coherente con la C final
    return S, C, np.asarray(history)


def aa(X, k, max_iter=100, tol=1e-6, seed=42, verbose=False, solver="nnls"):
    """
    Archetypal Analysis. solver: "nnls", "slsqp" o "pgd" (ver simplex.py).

    Devuelve (S, C, Z, info) con info = {rss, history, n_iter}.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X debe ser una matriz 2D.")

    n = X.shape[0]
    C = np.zeros((k, n))
    C[np.arange(k), furthest_sum(X, k, np.random.default_rng(seed))] = 1.0

    S, C, history = _fit(X, C, np.ones(n), max_iter, tol, solver, verbose)
    Z = C @ X

    info = {"rss": np.sum((X - S @ Z) ** 2), "history": history, "n_iter": len(history)}
    return S, C, Z, info
