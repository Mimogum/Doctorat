import numpy as np
import matplotlib.pyplot as plt

from .aagradient import archetypal_analysis as aagradient_analysis
from .Weighted import line_search_C_given_X, line_search_S_given_Z

def bisquare_weights(residuals, c_factor=6.0):
    """
    Calcula la matriz de pesos W utilizando el estimador M Bisquare de Tukey
    elemento a elemento (para cada celda de la matriz).
    
    Parameters:
    -----------
    residuals : ndarray de forma (N, M)
        Matriz de residuos R = X - S @ Z
    c_factor : float
        Factor de ajuste. c = c_factor * median(|residuals|).
    
    Returns:
    --------
    W : ndarray de forma (N, M)
        Matriz de pesos celda a celda en R^1 con dimensiones (N, M).
    """
    # 1. Residuos en valor absoluto para cada celda (N, M)
    abs_residuals = np.abs(residuals) # (N, M)
    
    # 2. Calcular la mediana de los residuos no nulos a nivel global (todas las celdas)
    non_zero_r = abs_residuals[abs_residuals > 0]
    if len(non_zero_r) == 0:
        s = 1.0
    else:
        s = np.median(non_zero_r)
    
    c = c_factor * s
    if c == 0:
        c = 1e-8

    # 3. Función de pesos Bisquare elemento a elemento:
    # w_ij = (1 - (r_ij/c)^2)^2 si |r_ij| < c, de lo contrario 0
    W = np.zeros_like(abs_residuals)
    mask = abs_residuals < c
    W[mask] = (1.0 - (abs_residuals[mask] / c) ** 2) ** 2

    return W


def robust_archetypal_analysis(X, p, max_robust_iter=20, max_inner_iter=10, 
                               c_factor=6.0, tol=1e-5, random_state=42, verbose=False):
    """
    Ejecuta el Robust AA por IRLS conforme a Eugster & Leisch (2011),
    utilizando ponderación elemento a elemento en X.
    """
    X = np.asarray(X, dtype=float)
    N, M = X.shape

    # 1. Inicialización: Pesos unitarios de dimensión (N, M) en la primera iteración
    W_matrix = np.ones((N, M), dtype=float)
    
    C_final, S_final, Z_final = None, None, None

    for t in range(max_robust_iter):
        # 2. Ponderación de datos elemento a elemento (multiplicación Hadamard W * X)
        X_weighted = W_matrix * X

        # 3. Optimización con gradiente sobre los datos reponderados
        C_t, S_t, Z_t, hist_t, selected_t = aagradient_analysis(
            X_weighted, p=p, max_iter=max_inner_iter, 
            tol=tol, random_state=random_state + t, verbose=False
        )
        
        # Proyección de arquetipos sobre la X original
        Z_t = C_t @ X
        
        # 4. Residuos sobre los datos reales: R = X - S @ Z (dimensión N x M)
        residuals = X - (S_t @ Z_t)

        # 5. Actualización adaptativa de pesos mediante Bisquare celda a celda
        W_matrix_new = bisquare_weights(residuals, c_factor=c_factor)

        # Criterio de convergencia de la matriz de pesos
        weight_diff = np.linalg.norm(W_matrix_new - W_matrix) / np.linalg.norm(W_matrix)
        
        C_final, S_final, Z_final = C_t, S_t, Z_t
        W_matrix = W_matrix_new

        if verbose:
            print(f"Iteración Robust {t+1}/{max_robust_iter} | Dif. pesos: {weight_diff:.6f} | Outliers en celdas (w=0): {np.sum(W_matrix == 0.0)}")

        if weight_diff < tol:
            break

    # 6. Recalculación final sobre X sin ponderar (Paso 4 del artículo)
    Z_final = C_final @ X
    
    # Ajuste final de S sobre los arquetipos Z_final fijos
    ZZT = Z_final @ Z_final.T
    XZT = X @ Z_final.T
    S_final, _, _ = line_search_S_given_Z(X, Z_final, ZZT, XZT, S_final)

    # Ajuste final de C sobre X sin alterar Z_final
    XXT = X @ X.T
    ZXT = Z_final @ X.T
    C_final, _, _ = line_search_C_given_X(Z_final, X, XXT, ZXT, C_final)

    return C_final, S_final, Z_final, W_matrix