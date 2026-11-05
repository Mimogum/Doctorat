import numpy as np

# ============================================================
# 1. Solucionador de Conjunto Activo para el Símplex (Chen et al., Algoritmo 2)
# ============================================================

def active_set_qp_simplex(x, Z, max_iter=100, tol=1e-8):
    """
    Resuelve el problema QP con restricciones en el símplex:
    min_{alpha in Delta_p} ||x - Z * alpha||_2^2
    Conforme al Algoritmo 2 de Chen et al. (2014).
    """
    m, p = Z.shape
    
    # Inicialización con el vértice más cercano (punto factible)
    distances = np.linalg.norm(Z - x[:, np.newaxis], axis=0)
    best_idx = np.argmin(distances)
    
    alpha = np.zeros(p, dtype=float)
    alpha[best_idx] = 1.0
    active_set = [best_idx]
    
    for k in range(max_iter):
        Z_A = Z[:, active_set]
        s_A = len(active_set)
        
        if s_A == 1:
            q_A = np.zeros(1)
        else:
            r = x - Z @ alpha
            Gram_A = Z_A.T @ Z_A
            rhs_grad = Z_A.T @ r
            
            KKT = np.zeros((s_A + 1, s_A + 1))
            KKT[:s_A, :s_A] = Gram_A
            KKT[:s_A, s_A] = 1.0
            KKT[s_A, :s_A] = 1.0
            
            rhs = np.zeros(s_A + 1)
            rhs[:s_A] = rhs_grad
            
            try:
                sol = np.linalg.solve(KKT + 1e-10 * np.eye(s_A + 1), rhs)
                q_A = sol[:s_A]
            except np.linalg.LinAlgError:
                q_A = np.zeros(s_A)

        q = np.zeros(p, dtype=float)
        q[active_set] = q_A
        
        if np.linalg.norm(q) < tol:
            grad = -2.0 * Z.T @ (x - Z @ alpha)
            inactive_set = [j for j in range(p) if j not in active_set]
            if not inactive_set:
                break
                
            min_inactive_idx = inactive_set[np.argmin(grad[inactive_set])]
            lambda_val = np.mean(grad[active_set])
            
            if grad[min_inactive_idx] >= lambda_val - tol:
                break
            else:
                active_set.append(min_inactive_idx)
                continue
        else:
            gamma = 1.0
            blocking_idx = None
            
            for idx in active_set:
                if q[idx] < -tol:
                    alpha_ratio = -alpha[idx] / q[idx]
                    if alpha_ratio < gamma:
                        gamma = alpha_ratio
                        blocking_idx = idx
            
            gamma = max(0.0, min(1.0, gamma))
            alpha += gamma * q
            
            alpha[alpha < 1e-14] = 0.0
            sum_a = np.sum(alpha)
            if sum_a > 0:
                alpha /= sum_a
            
            if gamma < 1.0 and blocking_idx is not None:
                if blocking_idx in active_set:
                    active_set.remove(blocking_idx)
            
            active_set = [j for j in range(p) if alpha[j] > 0]
            if len(active_set) == 0:
                active_set = [best_idx]
                alpha[best_idx] = 1.0

    return alpha


# ============================================================
# 2. Análisis de Arquetipos Robusto (Chen et al., Algoritmo 3)
# ============================================================

def robust_archetypal_analysis_chen(
    X,
    p,
    max_iter=100,
    epsilon=0.01,
    tol=1e-5,
    random_state=42,
    verbose=True
):
    """
    Implementación al 100% del Algoritmo 3 de Chen et al. (2014) para 
    Análisis de Arquetipos Robusto usando pérdida de Huber y pesos por muestra w_i.
    
    Parameters:
    -----------
    X : ndarray de forma (m, n)
        Matriz de datos (m atributos, n muestras). Si se pasa como (n, m), se traspone.
    p : int
        Número de arquetipos.
    epsilon : int/float
        Parámetro de umbral para la pérdida de Huber (típicamente 0.01 para datos normalizados).
    """
    X = np.asarray(X, dtype=float)
    if X.shape[0] > X.shape[1]:
        X = X.T  # Asegurar formato (m x n) donde columnas son muestras
        
    m, n = X.shape
    rng = np.random.default_rng(random_state)

    # 1. Inicialización de Z con columnas aleatorias de X (Algoritmo 3, Línea 2)
    init_indices = rng.choice(n, size=p, replace=False)
    Z = X[:, init_indices].copy()

    # 2. Inicialización de B tal que Z = X * B (Línea 3) -> Estimación inicial por mínimos cuadrados o identidad aproximada
    # Para simplificar de forma robusta, inicializamos B aleatoriamente en el símplex delta_n por columna
    B = rng.dirichlet(np.ones(n), size=p).T  # (n x p)
    Z = X @ B

    # 3. Inicialización de w en R^n con w_i = 1 para todo i (Línea 4)
    w = np.ones(n, dtype=float)
    
    A = np.zeros((p, n), dtype=float)
    previous_rss = np.inf

    for t in range(max_iter):
        
        # --- Actualización de alpha_i y pesos w_i (Líneas 6-8)[cite: 16] ---
        for i in range(n):
            x_i = X[:, i]
            # Resolver alpha_i en el símplex Delta_p (Línea 6)[cite: 16]
            A[:, i] = active_set_qp_simplex(x_i, Z)
            
            # w_i = max(||x_i - Z * alpha_i||_2, epsilon) (Línea 7)[cite: 16]
            residual_norm = np.linalg.norm(x_i - Z @ A[:, i])
            w[i] = max(residual_norm, epsilon)

        # --- Matriz de escala Gamma = diag(w)^{-1} (Línea 9)[cite: 16] ---
        gamma_vec = 1.0 / w
        
        # Residuos globales R = X - Z * A (Línea 10)[cite: 16]
        R = X - Z @ A

        # --- Actualización de beta_j y arquetipos z_j (Líneas 11-16)[cite: 16] ---
        for j in range(p):
            alpha_j = A[j, :]  # Fila j de A (1 x n)
            
            # Denominador: alpha^j * Gamma * alpha^{j T}
            denom = np.dot(alpha_j * gamma_vec, alpha_j)
            
            if denom > 1e-12:
                # Ecuación de actualización de beta_j (Línea 12)[cite: 16]
                numer = R @ (alpha_j * gamma_vec)
                v_j = (numer / denom) + Z[:, j]
            else:
                v_j = Z[:, j]
                
            # Resolver beta_j en argmin ||v_j - X * beta||_2^2 s.t. beta in Delta_n (Línea 12)[cite: 16]
            B[:, j] = active_set_qp_simplex(v_j, X)
            
            # Actualización del arquetipo z_j y residuo R (Líneas 13-14)[cite: 16]
            z_j_new = X @ B[:, j]
            R += np.outer(Z[:, j] - z_j_new, alpha_j)
            Z[:, j] = z_j_new

        # Calcular RSS ponderado por Huber/pesos
        rss = np.sum(w * np.sum((X - Z @ A) ** 2, axis=0))
        improvement = previous_rss - rss

        if verbose:
            outliers_count = np.sum(w == epsilon)
            print(
                f"Iteración Chen {t+1}/{max_iter} | "
                f"Huber RSS = {rss:.6f} | "
                f"Outliers detectados (w=epsilon): {outliers_count}"
            )

        if abs(improvement) < tol and t > 5:
            break
            
        previous_rss = rss

    return A, B, Z, w