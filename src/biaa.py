import numpy as np

from .simplex import solve

# ============================================================
# Biarchetypal Analysis
#
#   X ≈ S0 Z S1.T,   Z = C0 X C1.T
#   X : (n, m)   Z : (k, c)
#   S0 : (n, k)  C0 : (k, n)   -> observations
#   S1 : (m, c)  C1 : (c, m)   -> variables
#   All >= 0 and every row sums to 1.
#
# Alternates S0, S1 (row/column projections) and C0, C1 (coordinate
# descent). The core accepts per-cell weights W (n, m): biaa uses W = 1 and
# robust_biaa the Huber weights. Every subproblem goes through simplex.solve.
# ============================================================


def _update_S0(X, Z, S1, W, solver, S0=None):
    """Each row x_i onto the k rows of Z S1.T."""
    return solve(X, (Z @ S1.T).T, solver, W, A0=S0)


def _update_S1(X, S0, Z, W, solver, S1=None):
    """Each column x_:j onto the c columns of S0 Z."""
    return solve(X.T, S0 @ Z, solver, W.T, A0=S1)


def _update_C0(X, S0, C0, S1, C1, W, solver):
    """
    C0 (k, n) by coordinates. With H = X C1.T S1.T, row g of Z S1.T is
    C0[g] H and, with the rest fixed:
      sum_ij W_ij (r_ij + s_ig (h_j - h'_j))^2 = sum_j denom_j (target_j - h'_j)^2
    """
    H = X @ C1.T @ S1.T
    R = X - S0 @ C0 @ H  # current residual, kept up to date in the loop

    for g in range(C0.shape[0]):
        s = S0[:, [g]]  # (n, 1) usage of observation-archetype g
        h = C0[g] @ H
        # Per-column weights and target; columns with denom ~ 0 keep h
        denom = np.sum(W * s ** 2, axis=0)
        target = np.where(denom > 1e-12, np.sum(W * s * R, axis=0) / np.maximum(denom, 1e-12) + h, h)

        # Weighted projection of target onto the convex hull of the rows of H
        C0[g] = solve(target, H.T, solver, np.maximum(denom, 1e-12)[None], A0=C0[g])[0]
        R += s * (h - C0[g] @ H)  # incremental residual update

    return C0


def _update_C1(X, S0, C0, S1, C1, W, solver):
    """C1 (c, m) by coordinates; column h of S0 Z = B C1[h], with B = S0 C0 X.
    Mirror of _update_C0 acting on columns instead of rows."""
    B = S0 @ C0 @ X
    R = X - B @ C1.T @ S1.T

    for h in range(C1.shape[0]):
        s = S1[:, h][None, :]  # (1, m) usage of variable-archetype h
        b = B @ C1[h]
        denom = np.sum(W * s ** 2, axis=1)
        target = np.where(denom > 1e-12, np.sum(W * R * s, axis=1) / np.maximum(denom, 1e-12) + b, b)

        C1[h] = solve(target, B, solver, np.maximum(denom, 1e-12)[None], A0=C1[h])[0]
        R += (b - B @ C1[h])[:, None] * s

    return C1


def _init(X, k, c, seed):
    """Validate k, c and draw a random start: C0, C1 from a flat Dirichlet, S1 uniform."""
    n, m = X.shape
    if not 1 <= k <= n:
        raise ValueError("k must satisfy 1 <= k <= n.")
    if not 1 <= c <= m:
        raise ValueError("c must satisfy 1 <= c <= m.")
    rng = np.random.default_rng(seed)
    C0 = rng.dirichlet(np.ones(n), size=k)
    C1 = rng.dirichlet(np.ones(m), size=c)
    S1 = np.full((m, c), 1.0 / c)
    return C0, C1, S1


def _fit(X, C0, C1, S1, W, max_iter, tol, solver, verbose=False):
    """
    Core shared by biaa and robust_biaa: minimises
    sum_ij W_ij (X - S0 C0 X C1.T S1.T)_ij^2. Returns S0, C0, S1, C1, history.
    """
    history, S0 = [], None
    for it in range(max_iter):
        # One sweep: S0 -> S1 -> C0 -> C1, each with the others fixed
        Z = C0 @ X @ C1.T
        S0 = _update_S0(X, Z, S1, W, solver, S0)
        S1 = _update_S1(X, S0, Z, W, solver, S1)
        C0 = _update_C0(X, S0, C0, S1, C1, W, solver)
        C1 = _update_C1(X, S0, C0, S1, C1, W, solver)
        history.append(np.sum(W * (X - S0 @ C0 @ X @ C1.T @ S1.T) ** 2))

        # Stop when Z stops moving (see aa._fit).
        change = np.linalg.norm(C0 @ X @ C1.T - Z) / (np.linalg.norm(Z) + 1e-10)
        if verbose:
            print(f"Iteration {it + 1:3d} | loss = {history[-1]:.6e} | change = {change:.2e}")
        if change < tol:
            break

    # S0, S1 consistent with the final Z
    Z = C0 @ X @ C1.T
    S0 = _update_S0(X, Z, S1, W, solver, S0)
    S1 = _update_S1(X, S0, Z, W, solver, S1)
    return S0, C0, S1, C1, np.asarray(history)


def biaa(X, k, c, max_iter=100, tol=1e-6, seed=42, verbose=False, solver="nnls"):
    """
    Biarchetypal Analysis with k observation archetypes and c variable archetypes.
    solver: "nnls", "slsqp" or "pgd" (see simplex.py).

    Returns (S0, C0, S1, C1, Z, info) with info = {rss, history, n_iter}.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")

    C0, C1, S1 = _init(X, k, c, seed)
    S0, C0, S1, C1, history = _fit(X, C0, C1, S1, np.ones_like(X), max_iter, tol, solver, verbose)
    Z = C0 @ X @ C1.T

    info = {"rss": np.sum((X - S0 @ Z @ S1.T) ** 2), "history": history, "n_iter": len(history)}
    return S0, C0, S1, C1, Z, info
