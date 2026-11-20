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
# 2. Funciones de actualización BiAA (Estilo biaa.py + Chen)
# ============================================================

def compute_alpha(X, Z, gamma):
    """
    Actualiza alpha (n, k): cada fila x_i se proyecta en el símplex
    sobre observation_archetypes = Z @ gamma (k, m).
    """
    n = X.shape[0]
    observation_archetypes = Z @ gamma  # (k, m)
    alpha = np.zeros((n, Z.shape[0]))

    for i in range(n):
        alpha[i] = active_set_simplex_qp(observation_archetypes.T, X[i])

    return alpha


def compute_gamma_weighted(X, alpha, Z, gamma_vec):
    """
    Actualiza gamma (c, m) ponderando las observaciones por gamma_vec = 1 / w_huber.
    Minimiza sum_i gamma_vec[i] * (X[i, j] - (alpha @ Z)[i] @ gamma[:, j])^2.
    """
    sqrt_w = np.sqrt(gamma_vec)[:, np.newaxis]  # (n, 1)
    variable_archetypes_w = sqrt_w * (alpha @ Z)  # (n, c)
    X_w = sqrt_w * X  # (n, m)

    m = X.shape[1]
    c = Z.shape[1]
    gamma = np.zeros((c, m))

    for j in range(m):
        gamma[:, j] = active_set_simplex_qp(variable_archetypes_w, X_w[:, j])

    return gamma


def compute_beta_chen(X, alpha, beta, gamma, theta, gamma_vec):
    """
    Actualización de beta (k, n) por descenso por coordenadas ponderado con gamma_vec
    (equivalente al paso 2 de Chen et al. 2014 para C).
    Z = beta @ X @ theta.T  =>  X ~= alpha @ beta @ (X @ theta.T) @ gamma
    """
    k, n = beta.shape
    H = (X @ theta.T) @ gamma  # (n, m): candidatos de observación proyectados al espacio original
    Z = beta @ X @ theta.T
    R = X - alpha @ Z @ gamma  # Residuo global (n, m)

    for g in range(k):
        a_g = alpha[:, g]  # (n,)
        h_g = beta[g, :] @ H  # (m,) fila g-ésima de Z @ gamma

        denom = np.sum((a_g ** 2) * gamma_vec)
        if denom > 1e-12:
            num = (gamma_vec * a_g) @ R  # (m,)
            target_vector = (num / denom) + h_g
        else:
            target_vector = h_g

        # Optimizar beta[g, :] en Delta_n usando H.T de forma (m, n)
        beta[g, :] = active_set_simplex_qp(H.T, target_vector)

        # Actualizar residuo R con el nuevo beta[g, :]
        h_g_new = beta[g, :] @ H
        R += np.outer(a_g, h_g - h_g_new)

    return beta


def compute_theta_chen(X, alpha, beta, gamma, theta, gamma_vec):
    """
    Actualización de theta (c, m) por descenso por coordenadas ponderado con gamma_vec
    (análogo de Chen et al. 2014 para la dimensión de variables).
    X ~= (alpha @ beta @ X) @ theta.T @ gamma
    """
    c, m = theta.shape
    B = alpha @ beta @ X  # (n, m): datos proyectados por los arquetipos de observación
    sqrt_w = np.sqrt(gamma_vec)[:, np.newaxis]
    B_w = sqrt_w * B  # (n, m)

    Z = beta @ X @ theta.T
    R_w = sqrt_w * (X - alpha @ Z @ gamma)  # Residuo ponderado (n, m)

    for h in range(c):
        g_h = gamma[h, :]  # (m,)
        b_h_w = B_w @ theta[h, :]  # (n,)

        denom = np.sum(g_h ** 2)
        if denom > 1e-12:
            num = R_w @ g_h  # (n,)
            target_vector = (num / denom) + b_h_w
        else:
            target_vector = b_h_w

        # Optimizar theta[h, :] en Delta_m usando B_w de forma (n, m)
        theta[h, :] = active_set_simplex_qp(B_w, target_vector)

        # Actualizar residuo ponderado R_w
        b_h_w_new = B_w @ theta[h, :]
        R_w += np.outer(b_h_w - b_h_w_new, g_h)

    return theta


# ============================================================
# 3. Robust Biarchetypal Analysis (Chen et al. 2014 - Huber)
# ============================================================

def robust_biarchetypal_analysis_huber(
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
    Algoritmo 3 de Chen et al. (2014) extendido a Biarchetypal Analysis (BiAA).
    Devuelve alpha, beta, gamma, theta, Z y los pesos efectivos por observación.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional matrix.")

    n, m = X.shape
    rng = np.random.default_rng(seed)

    # Inicialización estilo biaa.py (Dirichlet en el símplex)
    beta = rng.dirichlet(np.ones(n), size=k)
    theta = rng.dirichlet(np.ones(m), size=c)
    Z = beta @ X @ theta.T
    gamma = np.full((c, m), 1.0 / c)

    # Pesos Huber por observación (1 / w_huber atenúa outliers)
    w_huber = np.ones(n, dtype=float)

    for t in range(max_robust_iter):
        w_huber_old = w_huber.copy()

        # 1. Actualización de alpha (por fila) y cálculo de pesos Huber v_i = max(||r_i||_2, epsilon)
        alpha = compute_alpha(X, Z, gamma)
        X_reconstructed = alpha @ Z @ gamma
        for i in range(n):
            res_i = X[i, :] - X_reconstructed[i, :]
            w_huber[i] = max(np.linalg.norm(res_i), epsilon)

        # gamma_vec = 1 / w_huber_i (outliers tienen peso muy pequeño)
        gamma_vec = 1.0 / w_huber

        # 2. Bucle interno de actualización alternante ponderada (estilo biaa.py + Chen)
        for inner_iter in range(max_inner_iter):
            Z_old = Z.copy()

            alpha = compute_alpha(X, Z, gamma)
            gamma = compute_gamma_weighted(X, alpha, Z, gamma_vec)
            beta = compute_beta_chen(X, alpha, beta, gamma, theta, gamma_vec)
            theta = compute_theta_chen(X, alpha, beta, gamma, theta, gamma_vec)
            Z = beta @ X @ theta.T

            if np.linalg.norm(Z - Z_old, ord='fro') / (np.linalg.norm(Z_old, ord='fro') + 1e-10) < tol:
                break

        weight_diff = np.linalg.norm(w_huber - w_huber_old) / (np.linalg.norm(w_huber_old) + 1e-10)

        if verbose:
            print(
                f"Iteración Robust Huber {t+1}/{max_robust_iter} "
                f"(inner iters: {inner_iter+1}) | Dif. pesos: {weight_diff:.6f}"
            )

        if weight_diff < tol:
            break

    # Normalizar pesos en rango [0, 1] donde ~0 es outlier
    effective_weights = 1.0 / (w_huber / np.min(w_huber))

    return alpha, beta, gamma, theta, Z, effective_weights