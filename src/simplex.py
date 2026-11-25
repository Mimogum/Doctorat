import numpy as np
from scipy.optimize import minimize, nnls

# ============================================================
# Mínimos cuadrados en el símplex, por filas:
#
#   A = solve(Y, V, solver, W)
#
#   Y : (r, d)   r vectores a representar.
#   V : (d, q)   sus q columnas son los candidatos.
#   W : (r, d)   pesos por celda (opcional).
#   A0: (r, q)   solución inicial (opcional); pgd y slsqp arrancan de ella.
#   A : (r, q)   la fila i minimiza sum_j W_ij (Y_ij - (V a_i)_j)^2
#                con a_i >= 0, sum(a_i) = 1.
#
# Todos los métodos resuelven sus subproblemas con solve(); el solver
# decide si va fila a fila o en bloque:
#   "nnls"  : fila a fila. NNLS con la suma 1 como fila extra. Por defecto:
#             el más preciso (error relativo ~1e-8).
#   "slsqp" : fila a fila. Optimizador genérico de scipy; el más lento y
#             para por ftol (error relativo ~4e-6).
#   "pgd"   : en bloque. Gradiente proyectado acelerado (FISTA) sobre todas
#             las filas a la vez (error relativo ~2e-7).
# ============================================================


def _nnls_row(y, V, a0):  # NNLS no admite arranque: a0 se ignora
    """
    NNLS (Lawson-Hanson) con una fila extra M*1 que impone sum(a) = 1
    (Cutler & Breiman); M domina la escala, así que queda exacta.

    Un ridge minúsculo (1e-8 relativo) desempata cuando hay infinitos
    óptimos (p.ej. C0/C1 en BiAA): sin él NNLS devuelve un vértice y
    varios arquetipos pueden colapsar en el mismo punto.
    """
    q = V.shape[1]
    scale = max(np.abs(V).max(), np.abs(y).max(), 1.0)
    M = 1e3 * scale
    A = np.vstack([V, np.full((1, q), M), 1e-4 * scale * np.eye(q)])
    b = np.concatenate([y, [M], np.zeros(q)])
    try:
        return nnls(A, b, maxiter=50 * q)[0]
    except RuntimeError:
        return np.zeros(q)


def _slsqp_row(y, V, a0):
    res = minimize(
        lambda a: np.sum((y - V @ a) ** 2), a0,
        jac=lambda a: -2.0 * V.T @ (y - V @ a), method="SLSQP",
        bounds=[(0.0, None)] * V.shape[1],
        constraints={"type": "eq", "fun": lambda a: a.sum() - 1.0},
        options={"ftol": 1e-9, "maxiter": 500},
    )
    return res.x


def _by_rows(row_solver):
    """Convierte un solver de un vector en uno de matrices, fila a fila."""
    def solve_rows(Y, V, W, A0):
        sqrt_W = np.ones_like(Y) if W is None else np.sqrt(W)
        return np.array([row_solver(sw * y, sw[:, None] * V, a0)
                         for y, sw, a0 in zip(Y, sqrt_W, A0)])
    return solve_rows


def _project_rows(P):
    """Proyección euclídea de cada fila sobre el símplex (Duchi et al. 2008)."""
    r, q = P.shape
    U = -np.sort(-P, axis=1)
    css = np.cumsum(U, axis=1) - 1.0
    positive = U > css / np.arange(1, q + 1)
    rho = q - 1 - np.argmax(positive[:, ::-1], axis=1)  # último índice positivo
    theta = css[np.arange(r), rho] / (rho + 1.0)
    return np.maximum(P - theta[:, None], 0.0)


def _pgd_block(Y, V, W, A0, max_iter=500, tol=1e-8):
    """
    FISTA sobre todas las filas a la vez. Paso 1/L_i por fila, con
    L_i = 2 max_j(W_ij) ||V||_2^2 >= constante de Lipschitz de la fila i.
    """
    r, q = Y.shape[0], V.shape[1]
    W = np.ones_like(Y) if W is None else W
    L = 2.0 * W.max(axis=1, keepdims=True) * np.linalg.norm(V, 2) ** 2
    L[L == 0] = 1.0

    A = Ym = A0
    t = 1.0
    for _ in range(max_iter):
        G = 2.0 * (W * (Ym @ V.T - Y)) @ V
        A_new = _project_rows(Ym - G / L)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        Ym = A_new + (t - 1.0) / t_new * (A_new - A)
        if np.abs(A_new - A).max() < tol:
            return A_new
        A, t = A_new, t_new
    return A


SOLVERS = {"nnls": _by_rows(_nnls_row), "slsqp": _by_rows(_slsqp_row), "pgd": _pgd_block}


def solve(Y, V, solver="nnls", W=None, A0=None):
    """Ver la cabecera del módulo. Devuelve A : (r, q)."""
    if solver not in SOLVERS:
        raise ValueError(f"solver debe ser uno de {sorted(SOLVERS)}.")

    Y = np.atleast_2d(Y)
    q = V.shape[1]
    A0 = np.full((Y.shape[0], q), 1.0 / q) if A0 is None else np.atleast_2d(A0)
    A = np.maximum(SOLVERS[solver](Y, V, W, A0), 0.0)
    # Salvaguarda común: negativos ~-1e-12 y suma != 1 por redondeo.
    s = A.sum(axis=1, keepdims=True)
    return np.where(s > 0, A / np.where(s > 0, s, 1.0), 1.0 / V.shape[1])
