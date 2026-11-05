import numpy as np
import matplotlib.pyplot as plt

from .aagradient import archetypal_analysis as aagradient_analysis
from .Weighted import line_search_C_given_X, line_search_S_given_Z

def bisquare_weights(residuals, c_factor=6.0):
    """
    Calcula el vector de pesos w utilizando el estimador M Bisquare de Tukey por filas.
    """
    # 1. Calcular la norma L2 de los residuos por observación (filas)
    r_norms = np.linalg.norm(residuals, ord=2, axis=1) # (N,)
    
    # 2. Calcular la mediana de los residuos no nulos
    non_zero_r = r_norms[r_norms > 0]
    if len(non_zero_r) == 0:
        s = 1.0
    else:
        s = np.median(non_zero_r)
    
    c = c_factor * s
    if c == 0:
        c = 1e-8

    # 3. Función de pesos Bisquare: w(r) = (1 - (r/c)^2)^2 si r < c, de lo contrario 0
    w = np.zeros_like(r_norms)
    mask = r_norms < c
    w[mask] = (1.0 - (r_norms[mask] / c) ** 2) ** 2

    return w


def robust_archetypal_analysis(X, p, max_robust_iter=20, max_inner_iter=10, 
                               c_factor=6.0, tol=1e-5, random_state=42, verbose=False):
    """
    Ejecuta el Robust AA por IRLS utilizando filtrado de outliers (trimmmed robust AA).
    """
    X = np.asarray(X, dtype=float)
    N, M = X.shape

    # 1. Inicialización: Pesos unitarios en la primera iteración
    w_vec = np.ones(N, dtype=float)
    
    C_final, S_final, Z_final = None, None, None

    for t in range(max_robust_iter):
        # 1. Ponderación de los datos: X_t = W * X (multiplicando cada fila por su peso w_i)
        X_t = w_vec[:, np.newaxis] * X

        # 2. Optimización con gradiente estándar
        C_t, S_t, Z_t, hist_t, selected_t = aagradient_analysis(
            X_t, p=p, max_iter=max_inner_iter, 
            tol=tol, random_state=random_state + t, verbose=False
        )
               
        # 3. Residuos sobre los datos reales completos: R = X - S @ Z
        residuals = X - (S_t @ Z_t)

        # 4. Actualización adaptativa de pesos mediante Bisquare
        w_vec_new = bisquare_weights(residuals, c_factor=c_factor)

        # 5. Criterio de convergencia de los pesos
        weight_diff = np.linalg.norm(w_vec_new - w_vec) / (np.linalg.norm(w_vec) + 1e-10)
        
        C_final, S_final, Z_final = C_t, S_t, Z_t
        w_vec = w_vec_new

        if verbose:
            print(f"Iteración Robust {t+1}/{max_robust_iter} | Dif. pesos: {weight_diff:.6f} | Outliers (w=0): {np.sum(w_vec == 0.0)}")

        if weight_diff < tol:
            break

    # 6. Recalculación final sobre X sin ponderar (Paso 4 del artículo)
    Z_final = C_final @ X
    
    ZZT = Z_final @ Z_final.T
    XZT = X @ Z_final.T
    S_final, _, _ = line_search_S_given_Z(X, Z_final, ZZT, XZT, S_final)

    XXT = X @ X.T
    ZXT = Z_final @ X.T
    C_final, _, _ = line_search_C_given_X(Z_final, X, XXT, ZXT, C_final)

    return C_final, S_final, Z_final, w_vec