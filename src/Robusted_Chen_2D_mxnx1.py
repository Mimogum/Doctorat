import numpy as np

def active_set_simplex_qp(Z, x):
    """
    Resuelve min_{alpha in Delta_p} ||x - Z alpha||_2^2 con proyección sobre el símplex.
    Z: forma (M, p) o (M, N)
    x: vector objetivo de dimensión (M,)
    """
    m, p = Z.shape
    alpha = np.ones(p, dtype=float) / p
    max_iter = 100
    
    # Tasa de aprendizaje adaptativa basada en la norma espectral
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


def robust_archetypal_analysis_huber(X, p, max_robust_iter=20, epsilon=0.01, 
                                     tol=1e-5, random_state=42, verbose=False):
    """
    Algoritmo 3 (Chen et al. 2014) adaptado a matriz X de forma (N observaciones, M características).
    Devuelve C (equivalente a S en Chen), S (equivalente a A en Chen), Z y los pesos por observación.
    """
    X = np.asarray(X, dtype=float)
    N, M = X.shape  # N = filas (observaciones), M = columnas (características)

    np.random.seed(random_state)
    replace_flag = False if N >= p else True
    init_indices = np.random.choice(N, size=p, replace=replace_flag)
    
    # C de forma (p, N) para expresar Z = C @ X (p arquetipos en espacio M)
    C = np.zeros((p, N), dtype=float)
    for j, idx in enumerate(init_indices):
        C[j, idx] = 1.0
    
    Z = C @ X  # Forma (p, M)
    S = np.ones((N, p), dtype=float) / p  # Forma (N, p), coeficientes de reconstrucción
    
    # Pesos huber para observaciones (1/w_i atenuará outliers)
    w_huber = np.ones(N, dtype=float)

    for t in range(max_robust_iter):
        w_huber_old = w_huber.copy()

        # 1. Actualización de coeficientes por fila (S) y cálculo de residuales por observación
        for i in range(N):
            x_i = X[i, :]  # (M,)
            # Proyectar sobre los p arquetipos Z (p, M) -> Z.T es (M, p)
            S[i, :] = active_set_simplex_qp(Z.T, x_i)
            
            # residuo r_i y peso Huber v_i = max(||r_i||_2, epsilon)
            res_i = x_i - (S[i, :] @ Z)
            w_huber[i] = max(np.linalg.norm(res_i), epsilon)

        # Gamma_i = 1 / w_huber_i (los outliers tendrán peso muy pequeño en gamma)
        gamma_vec = 1.0 / w_huber
        
        # Residuo global R = X - S @ Z (N, M)
        R = X - (S @ Z)

        # 2. Actualización de C (columnas de arquetipos)
        for j in range(p):
            s_j = S[:, j]  # Coeficiente del j-ésimo arquetipo en las N observaciones (N,)
            
            denom = np.sum((s_j ** 2) * gamma_vec)
            if denom > 1e-12:
                # Numerador ponderado por gamma
                num = (gamma_vec * s_j) @ R  # Forma (M,)
                target_vector = (num / denom) + Z[j, :]
            else:
                target_vector = Z[j, :]

            # Optimizar C[j, :] en el símplex Delta_N utilizando X.T (M, N)
            C[j, :] = active_set_simplex_qp(X.T, target_vector)
            
            # Actualizar arquetipo z_j y residuo R
            z_j_new = C[j, :] @ X
            R += np.outer(s_j, Z[j, :] - z_j_new)
            Z[j, :] = z_j_new

        weight_diff = np.linalg.norm(w_huber - w_huber_old) / (np.linalg.norm(w_huber_old) + 1e-10)
        
        if verbose:
            print(f"Iteración Robust Huber {t+1}/{max_robust_iter} | Dif. pesos: {weight_diff:.6f}")
            
        if weight_diff < tol:
            break

    # Normalizar pesos en rango [0, 1] donde ~0 es outlier (para compatibilidad con Bisquare)
    effective_weights = 1.0 / (w_huber / np.min(w_huber))

    return C, S, Z, effective_weights


def robust_archetypal_analysis_huber_mxnx1(X, p, max_robust_iter=20, epsilon=0.01, 
                                           tol=1e-5, random_state=42, verbose=False):
    """
    Retorna la versión de Huber coincidente con la interfaz 3D (N, M, 1).
    """
    C, S, Z, weights = robust_archetypal_analysis_huber(
        X, p=p, max_robust_iter=max_robust_iter, 
        epsilon=epsilon, tol=tol, random_state=random_state, verbose=verbose
    )

    # Re-ponderación de la matriz X por filas con los pesos efectivos
    X_weighted_2d = weights[:, np.newaxis] * X
    X_weighted_3d = X_weighted_2d[:, :, np.newaxis]

    return C, S, Z, weights, X_weighted_3d