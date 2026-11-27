import numpy as np
from scipy.optimize import minimize, nnls

# ============================================================
# Row-wise least squares on the simplex:
#
#   A = solve(Y, V, solver, W)
#
#   Y : (r, d)   r vectors to represent.
#   V : (d, q)   its q columns are the candidates.
#   W : (r, d)   per-cell weights (optional).
#   A0: (r, q)   initial solution (optional); pgd and slsqp start from it.
#   A : (r, q)   row i minimises sum_j W_ij (Y_ij - (V a_i)_j)^2
#                subject to a_i >= 0, sum(a_i) = 1.
#
# Every method solves its subproblems with solve(); the solver decides
# whether to go row by row or in a single block:
#   "nnls"  : row by row. NNLS with the sum-to-one as an extra row. Default:
#             the most accurate (relative error ~1e-8).
#   "slsqp" : row by row. Generic scipy optimiser; the slowest, and it stops
#             on ftol (relative error ~4e-6).
#   "pgd"   : block. Accelerated projected gradient (FISTA) over all rows
#             at once (relative error ~2e-7).
# ============================================================


def _nnls_row(y, V, a0):  # NNLS cannot be warm-started: a0 is ignored
    """
    NNLS (Lawson-Hanson) with an extra row M*1 that enforces sum(a) = 1
    (Cutler & Breiman); M dominates the scale, so the constraint is
    effectively exact.

    A tiny ridge (1e-8 relative) breaks ties when there are infinitely
    many optima (e.g. C0/C1 in BiAA): without it NNLS returns a vertex and
    several archetypes can collapse onto the same point.
    """
    q = V.shape[1]
    scale = max(np.abs(V).max(), np.abs(y).max(), 1.0)
    M = 1e3 * scale
    # Augmented system: [V; M*1; ridge*I] a = [y; M; 0]
    A = np.vstack([V, np.full((1, q), M), 1e-4 * scale * np.eye(q)])
    b = np.concatenate([y, [M], np.zeros(q)])
    try:
        return nnls(A, b, maxiter=50 * q)[0]
    except RuntimeError:
        # NNLS hit maxiter: solve() falls back to the uniform vector.
        return np.zeros(q)


def _slsqp_row(y, V, a0):
    """Direct constrained minimisation of ||y - V a||^2 with SLSQP."""
    res = minimize(
        lambda a: np.sum((y - V @ a) ** 2), a0,
        jac=lambda a: -2.0 * V.T @ (y - V @ a), method="SLSQP",
        bounds=[(0.0, None)] * V.shape[1],
        constraints={"type": "eq", "fun": lambda a: a.sum() - 1.0},
        options={"ftol": 1e-9, "maxiter": 500},
    )
    return res.x


def _by_rows(row_solver):
    """Turn a single-vector solver into a matrix solver that works row by row."""
    def solve_rows(Y, V, W, A0):
        # Weighted LS equals plain LS after scaling residuals by sqrt(W).
        sqrt_W = np.ones_like(Y) if W is None else np.sqrt(W)
        return np.array([row_solver(sw * y, sw[:, None] * V, a0)
                         for y, sw, a0 in zip(Y, sqrt_W, A0)])
    return solve_rows


def _project_rows(P):
    """Euclidean projection of each row onto the simplex (Duchi et al. 2008)."""
    r, q = P.shape
    U = -np.sort(-P, axis=1)  # each row sorted in descending order
    css = np.cumsum(U, axis=1) - 1.0
    positive = U > css / np.arange(1, q + 1)
    rho = q - 1 - np.argmax(positive[:, ::-1], axis=1)  # last positive index
    theta = css[np.arange(r), rho] / (rho + 1.0)  # per-row shift
    return np.maximum(P - theta[:, None], 0.0)


def _pgd_block(Y, V, W, A0, max_iter=500, tol=1e-8):
    """
    FISTA over all rows at once. Step 1/L_i per row, with
    L_i = 2 max_j(W_ij) ||V||_2^2 >= Lipschitz constant of row i.
    """
    r, q = Y.shape[0], V.shape[1]
    W = np.ones_like(Y) if W is None else W
    L = 2.0 * W.max(axis=1, keepdims=True) * np.linalg.norm(V, 2) ** 2
    L[L == 0] = 1.0  # all-zero weights: any step works, avoid division by 0

    A = Ym = A0  # A: current iterate, Ym: extrapolated (momentum) point
    t = 1.0
    for _ in range(max_iter):
        G = 2.0 * (W * (Ym @ V.T - Y)) @ V  # gradient of the weighted loss
        A_new = _project_rows(Ym - G / L)
        # Nesterov momentum update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        Ym = A_new + (t - 1.0) / t_new * (A_new - A)
        if np.abs(A_new - A).max() < tol:
            return A_new
        A, t = A_new, t_new
    return A


SOLVERS = {"nnls": _by_rows(_nnls_row), "slsqp": _by_rows(_slsqp_row), "pgd": _pgd_block}


def solve(Y, V, solver="nnls", W=None, A0=None):
    """See the module header. Returns A : (r, q)."""
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {sorted(SOLVERS)}.")

    Y = np.atleast_2d(Y)  # a single vector is treated as one row
    q = V.shape[1]
    # Default start: barycentre of the simplex
    A0 = np.full((Y.shape[0], q), 1.0 / q) if A0 is None else np.atleast_2d(A0)
    A = np.maximum(SOLVERS[solver](Y, V, W, A0), 0.0)
    # Shared safeguard: negatives ~-1e-12 and sum != 1 from rounding.
    # Rows that came out all zero (failed solve) become uniform.
    s = A.sum(axis=1, keepdims=True)
    return np.where(s > 0, A / np.where(s > 0, s, 1.0), 1.0 / V.shape[1])
