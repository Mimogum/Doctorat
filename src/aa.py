import numpy as np

from .simplex import solve

# ============================================================
# Archetypal Analysis
#
#   X ≈ S Z,   Z = C X
#   X : (n, m)   S : (n, k)   C : (k, n)   Z : (k, m)
#   S, C >= 0 and every row sums to 1.
#
# Alternates S (each row of X onto the archetypes) and C (coordinate
# descent, one archetype at a time; Cutler & Breiman 1994). Every
# subproblem is solved by simplex.solve with the chosen `solver`.
# ============================================================


def furthest_sum(X, k, rng):
    """
    FURTHESTSUM (Mørup & Hansen 2012): k indices of extreme observations.
    It starts from a random point, which is discarded at the end.
    """
    n = X.shape[0]
    if not 1 <= k <= n:
        raise ValueError("k must satisfy 1 <= k <= n.")

    j = int(rng.integers(n))
    selected = [j]
    # q[i] = sum of distances from point i to every selected point
    q = np.linalg.norm(X - X[j], axis=1)
    q_first = q.copy()

    for t in range(1, k + 1):
        if t == k:
            q = q - q_first  # remove the influence of the random starting point
        available = np.ones(n, dtype=bool)
        available[selected] = False
        # Next point: the furthest (in summed distance) not yet selected
        j = int(np.argmax(np.where(available, q, -np.inf)))
        selected.append(j)
        q = q + np.linalg.norm(X - X[j], axis=1)

    return np.asarray(selected[1:], dtype=int)  # drop the random start


def _update_C(X, S, C, w, solver):
    """
    C (k, n) by coordinate descent with per-row weights w (n,).
    For archetype j, with the rest fixed:
      sum_i w_i ||r_i + s_ij (z_j - z')||^2 = denom ||target - z'||^2 + const
    and z' = C[j] X is projected onto the convex hull of X.
    """
    Z = C @ X
    R = X - S @ Z  # current residual, kept up to date in the loop

    for j in range(C.shape[0]):
        s_j = S[:, j]
        z_j = Z[j].copy()

        # Weighted least-squares target for archetype j; if no observation
        # uses it (denom ~ 0) it is left where it is.
        denom = np.sum(w * s_j ** 2)
        target = (w * s_j) @ R / denom + z_j if denom > 1e-12 else z_j

        # Closest point to target inside the convex hull of the rows of X
        C[j] = solve(target, X.T, solver, A0=C[j])[0]
        Z[j] = C[j] @ X
        R += np.outer(s_j, z_j - Z[j])  # incremental residual update

    return C


def _fit(X, C, w, max_iter, tol, solver, verbose=False):
    """
    Core shared by aa and robust_aa: minimises sum_i w_i ||x_i - s_i C X||^2
    by alternating S and C, starting from C. Returns S, C and the loss history.
    """
    history, S = [], None
    for it in range(max_iter):
        Z_old = C @ X
        S = solve(X, Z_old.T, solver, A0=S)  # a row's weight does not change its optimum
        C = _update_C(X, S, C, w, solver)
        history.append(np.sum(w * np.sum((X - S @ C @ X) ** 2, axis=1)))

        # Stop when Z stops moving: the loss can stall for a moment while Z
        # is still far off (robust_aa used to stop too early).
        change = np.linalg.norm(C @ X - Z_old) / (np.linalg.norm(Z_old) + 1e-10)
        if verbose:
            print(f"Iteration {it + 1:4d} | loss = {history[-1]:.6e} | change = {change:.2e}")
        if change < tol:
            break

    S = solve(X, (C @ X).T, solver, A0=S)  # S consistent with the final C
    return S, C, np.asarray(history)


def aa(X, k, max_iter=100, tol=1e-6, seed=42, verbose=False, solver="nnls"):
    """
    Archetypal Analysis. solver: "nnls", "slsqp" or "pgd" (see simplex.py).

    Returns (S, C, Z, info) with info = {rss, history, n_iter}.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")

    # Initialise each archetype as one extreme observation (one-hot rows of C)
    n = X.shape[0]
    C = np.zeros((k, n))
    C[np.arange(k), furthest_sum(X, k, np.random.default_rng(seed))] = 1.0

    S, C, history = _fit(X, C, np.ones(n), max_iter, tol, solver, verbose)
    Z = C @ X

    info = {"rss": np.sum((X - S @ Z) ** 2), "history": history, "n_iter": len(history)}
    return S, C, Z, info
