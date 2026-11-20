import numpy as np

# ============================================================
# 1. Active Set Simplex QP (Chen et al. 2014)
# ============================================================

def active_set_simplex_qp(Z, x):
    """
    Resuelve min_{alpha in Delta_p} ||x - Z alpha||_2^2 con proyección sobre el símplex.
    Z: forma (M, p)
    x: vector objetivo de dimensión (M,)
    """
    m, p = Z.shape
    alpha = np.ones(p, dtype=float) / p
    max_iter = 100

    norm_z = np.linalg.norm(Z, ord=2)
    lr = 1.0 / (norm_z**2 + 1e-8) if norm_z > 0 else 0.01

    for _ in range(max_iter):
        grad = -2 * Z.T @ (x - Z @ alpha)
        alpha_next = alpha - lr * grad

        # Proyección Euclidiana sobre el símplex Delta_p
        v = np.sort(alpha_next)[::-1]
        cssv = np.cumsum(v)
        rho_idx = np.nonzero(v * np.arange(1, p + 1) > (cssv - 1))[0]
        if len(rho_idx) == 0:
            break
        rho = rho_idx[-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        alpha_next = np.maximum(alpha_next - theta, 0)

        if np.linalg.norm(alpha_next - alpha) < 1e-6:
            break
        alpha = alpha_next

    return alpha


# ============================================================
# 2. Funciones de actualización BiAA ponderadas celda a celda
# ============================================================

def compute_alpha_cellwise(X, Z, gamma, W_mat):
    """
    Actualiza alpha (n, k) ponderando cada celda (i, j) con W_mat[i, j].
    """
    n = X.shape[0]
    observation_archetypes = Z @ gamma  # (k, m)
    alpha = np.zeros((n, Z.shape[0]))
    sqrt_W = np.sqrt(W_mat)  # (n, m)

    for i in range(n):
        sw_i = sqrt_W[i][:, np.newaxis]  # (m, 1)
        Z_aux_w = sw_i * observation_archetypes.T  # (m, k)
        x_i_w = sqrt_W[i] * X[i]  # (m,)
        alpha[i] = active_set_simplex_qp(Z_aux_w, x_i_w)

    return alpha


def compute_gamma_cellwise(X, alpha, Z, W_mat):
    """
    Actualiza gamma (c, m) ponderando cada celda (i, j) con W_mat[i, j].
    """
    variable_archetypes = alpha @ Z  # (n, c)
    m = X.shape[1]
    c = Z.shape[1]
    gamma = np.zeros((c, m))
    sqrt_W = np.sqrt(W_mat)  # (n, m)

    for j in range(m):
        sw_j = sqrt_W[:, j][:, np.newaxis]  # (n, 1)
        Z_aux_w = sw_j * variable_archetypes  # (n, c)
        x_j_w = sqrt_W[:, j] * X[:, j]  # (n,)
        gamma[:, j] = active_set_simplex_qp(Z_aux_w, x_j_w)

    return gamma


def compute_beta_chen_cellwise(X, alpha, beta, gamma, theta, W_mat):
    """
    Actualización de beta (k, n) por descenso por coordenadas ponderado celda a celda (W_mat).
    """
    k, n = beta.shape
    H = (X @ theta.T) @ gamma  # (n, m)
    Z = beta @ X @ theta.T
    R = X - alpha @ Z @ gamma  # Residuo global (n, m)

    for g in range(k):
        a_g = alpha[:, g][:, np.newaxis]  # (n, 1)
        h_g = beta[g, :] @ H  # (m,)

        # Denominador ponderado por fila y columna: sum_i W[i, j] * a_g[i]^2 -> forma (m,)
        denom = np.sum(W_mat * (a_g ** 2), axis=0)  # (m,)
        num = np.sum(W_mat * a_g * R, axis=0)  # (m,)

        target_vector = np.where(denom > 1e-12, (num / denom) + h_g, h_g)

        # Ponderamos el QP por sqrt(denom) en cada coordenada m
        sqrt_d = np.sqrt(np.maximum(denom, 1e-12))
        H_w = sqrt_d[:, np.newaxis] * H.T  # (m, n)
        target_w = sqrt_d * target_vector  # (m,)

        beta[g, :] = active_set_simplex_qp(H_w, target_w)

        # Actualizar residuo global R
        h_g_new = beta[g, :] @ H
        R += a_g * (h_g - h_g_new)

    return beta


def compute_theta_chen_cellwise(X, alpha, beta, gamma, theta, W_mat):
    """
    Actualización de theta (c, m) por descenso por coordenadas ponderado celda a celda (W_mat).
    """
    c, m = theta.shape
    B = alpha @ beta @ X  # (n, m)
    Z = beta @ X @ theta.T
    R = X - alpha @ Z @ gamma  # Residuo global (n, m)

    for h in range(c):
        g_h = gamma[h, :][np.newaxis, :]  # (1, m)
        b_h = B @ theta[h, :]  # (n,)

        # Denominador ponderado por columna: sum_j W[i, j] * g_h[j]^2 -> forma (n,)
        denom = np.sum(W_mat * (g_h ** 2), axis=1)  # (n,)
        num = np.sum(W_mat * R * g_h, axis=1)  # (n,)

        target_vector = np.where(denom > 1e-12, (num / denom) + b_h, b_h)

        # Ponderamos el QP por sqrt(denom) en cada observación n
        sqrt_d = np.sqrt(np.maximum(denom, 1e-12))
        B_w = sqrt_d[:, np.newaxis] * B  # (n, m)
        target_w = sqrt_d * target_vector  # (n,)

        theta[h, :] = active_set_simplex_qp(B_w, target_w)

        # Actualizar residuo global R
        b_h_new = B @ theta[h, :]
        R += (b_h - b_h_new)[:, np.newaxis] * g_h

    return theta


# ============================================================
# 3. Robust Biarchetypal Analysis Cellwise (Chen + Huber N x M)
# ============================================================

def robust_biarchetypal_analysis_huber_cellwise(
    X,
    k,
    c,
    max_robust_iter=20,
    max_inner_iter=50,
    epsilon=0.01,
    tol=1e-5,
    seed=42,
    verbose=False
):
    """
    Biarchetypal Analysis robustecido con Huber elemento a elemento (N x M)
    utilizando descenso por coordenadas estilo Chen et al. (2014).
    Devuelve alpha, beta, gamma, theta, Z y la matriz de pesos efectivos (N, M).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional matrix.")

    n, m = X.shape
    rng = np.random.default_rng(seed)

    # Inicialización estilo biaa.py
    beta = rng.dirichlet(np.ones(n), size=k)
    theta = rng.dirichlet(np.ones(m), size=c)
    Z = beta @ X @ theta.T
    gamma = np.full((c, m), 1.0 / c)

    # Matriz de escalas Huber celda a celda (N, M)
    W_huber = np.ones((n, m), dtype=float)

    for t in range(max_robust_iter):
        W_huber_old = W_huber.copy()

        # 1. Actualización de alpha y cálculo de pesos Huber celda a celda
        alpha = compute_alpha_cellwise(X, Z, gamma, 1.0 / W_huber)
        X_reconstructed = alpha @ Z @ gamma
        W_huber = np.maximum(np.abs(X - X_reconstructed), epsilon)

        # Matriz de pesos W_mat = 1 / W_huber_ij
        W_mat = 1.0 / W_huber

        # 2. Bucle interno de actualización alternante ponderada celda a celda
        for inner_iter in range(max_inner_iter):
            Z_old = Z.copy()

            alpha = compute_alpha_cellwise(X, Z, gamma, W_mat)
            gamma = compute_gamma_cellwise(X, alpha, Z, W_mat)
            beta = compute_beta_chen_cellwise(X, alpha, beta, gamma, theta, W_mat)
            theta = compute_theta_chen_cellwise(X, alpha, beta, gamma, theta, W_mat)
            Z = beta @ X @ theta.T

            if np.linalg.norm(Z - Z_old, ord='fro') / (np.linalg.norm(Z_old, ord='fro') + 1e-10) < tol:
                break

        weight_diff = np.linalg.norm(W_huber - W_huber_old, ord='fro') / (np.linalg.norm(W_huber_old, ord='fro') + 1e-10)

        if verbose:
            print(
                f"Iteración Robust Huber Cellwise {t+1}/{max_robust_iter} "
                f"(inner iters: {inner_iter+1}) | Dif. pesos: {weight_diff:.6f}"
            )

        if weight_diff < tol:
            break

    # Normalización en [0, 1] usando la mediana como referencia de celda limpia
    # (evita que una sola celda perfecta oscurezca todo el mapa de calor)
    ref_scale = np.median(W_huber)
    effective_weights = np.clip(ref_scale / W_huber, 0.0, 1.0)

    return alpha, beta, gamma, theta, Z, effective_weights