import numpy as np
import matplotlib.pyplot as plt

from .aagradient import archetypal_analysis as aagradient_analysis
from .aagradient import compute_archetypes

def objective_S_given_Z(X, Z, S):
    """Calcula el error ||X - S @ Z||_F^2"""
    residual = X - S @ Z
    return np.sum(residual ** 2)

def gradient_S_given_Z(ZZT, XZT, S):
    """
    Gradiente de ||X - S @ Z||_F^2 respecto a S con corrección de normalización por filas.
    G = S @ (Z @ Z.T) - X @ Z.T
    """
    G = S @ ZZT - XZT
    correction = np.sum(G * S, axis=1, keepdims=True)
    return G - correction

def update_S_given_Z(ZZT, XZT, S, step):
    G = gradient_S_given_Z(ZZT, XZT, S)
    S_new = S - step * G
    S_new = np.maximum(S_new, 0.0)
    
    # Normalización L1 por filas (sum_d S[d, n] = 1)
    row_sums = S_new.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return S_new / row_sums

def line_search_S_given_Z(X, Z, ZZT, XZT, S, initial_step=1.0, max_search=100):
    current_loss = objective_S_given_Z(X, Z, S)
    step = initial_step
    for _ in range(max_search):
        candidate = update_S_given_Z(ZZT, XZT, S, step)
        candidate_loss = objective_S_given_Z(X, Z, candidate)
        if candidate_loss < current_loss:
            return candidate, step, candidate_loss
        step *= 0.5
    return S.copy(), step, current_loss

def objective_C_given_X(Z, X, C):
    """Calcula el error ||Z - C @ X||_F^2"""
    residual = Z - C @ X
    return np.sum(residual ** 2)

def gradient_C_given_X(XXT, ZXT, C):
    """
    Gradiente de ||Z - C @ X||_F^2 respecto a C con corrección de normalización por columnas.
    G = C @ (X @ X.T) - Z @ X.T
    """
    G = C @ XXT - ZXT
    correction = np.sum(G * C, axis=1, keepdims=True)
    return G - correction

def update_C_given_X(XXT, ZXT, C, step):
    G = gradient_C_given_X(XXT, ZXT, C)
    C_new = C - step * G
    C_new = np.maximum(C_new, 0.0)
    
    # Normalización L1 por columnas (sum_n C[n, d] = 1)
    col_sums = C_new.sum(axis=1, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    return C_new / col_sums

def line_search_C_given_X(Z, X, XXT, ZXT, C, initial_step=1.0, max_search=100):
    current_loss = objective_C_given_X(Z, X, C)
    step = initial_step
    for _ in range(max_search):
        candidate = update_C_given_X(XXT, ZXT, C, step)
        candidate_loss = objective_C_given_X(Z, X, candidate)
        if candidate_loss < current_loss:
            return candidate, step, candidate_loss
        step *= 0.5
    return C.copy(), step, current_loss

def validate_weights(weights, N):
    """
    Valida y prepara el vector o matriz de pesos W.
    """
    if weights is None:
        return np.ones(N, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) != N:
        raise ValueError(f"weights debe tener forma ({N},).")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights contiene valores no finitos.")
    if np.any(w < 0):
        raise ValueError("weights debe ser no negativo.")
    if not np.any(w > 0):
        raise ValueError("Al menos un peso debe ser estrictamente positivo.")
    return w

def weighted_archetypal_analysis(X, p, W, max_iter=100, max_inner_iter=10, tol=1e-8, random_state=42, verbose=False):
    """
    1. Ejecuta wAA encontrando la solución inicial en el espacio ponderado.
    2. Aplica el postprocesamiento exacto sobre la X original optimizando 
       ||X - SZ||^2 (para S) y ||Z - CX||^2 (para C) con sus respectivos gradientes.
    """
    X = np.asarray(X, dtype=float)
    
    # 1. Preparar la matriz de pesos
    if isinstance(W, np.ndarray) and W.ndim == 1:
        W_matrix = np.diag(W)
    else:
        W_matrix = W

    # Datos ponderados para la fase de búsqueda principal
    X_weighted = W_matrix @ X

    # 2. Ejecutar aagradient sobre los datos ponderados
    C_w, S_w, Z_w, hist_w, selected_w = aagradient_analysis(
        X_weighted, p=p, max_iter=max_iter, max_inner_iter=max_inner_iter, 
        tol=tol, random_state=random_state, verbose=verbose
    )

    # 3. POSTPROCESAMIENTO: Recálculo de ambas matrices sobre el espacio original X
    # Para asegurar que las matrices finales C y S se ajusten perfectamente a los datos reales X,
    # ejecutamos la optimización sobre X (o un recálculo directo de ambas matrices).
    C_final = C_w.copy()
    S_final = S_w.copy()
    Z_final = Z_w.copy()  # Arquetipos iniciales proyectados sobre la X real

    step_S = 1.0
    step_C = 1.0
    XXT = X @ X.T

    for _ in range(max_inner_iter):
        ZZT = Z_final @ Z_final.T
        XZT = X @ Z_final.T
        ZXT = Z_final @ X.T
        
        # Recalcular C minimizando ||Z - CX||^2
        C_final, step_C, _ = line_search_C_given_X(Z_final, X, XXT, ZXT, C_final, initial_step=step_C)
        
        # Actualizar arquetipos con el nuevo C y la X original
        Z_final = C_final @ X

        # Recalcular S minimizando ||X - SZ||^2
        S_final, step_S, _ = line_search_S_given_Z(X, Z_final, ZZT, XZT, S_final, initial_step=step_S)

    if verbose:
        print("-> Postprocesamiento completado: Matrices C y S recalculadas sobre X con gradientes dedicados.")

    return C_final, S_final, Z_w, hist_w, selected_w

