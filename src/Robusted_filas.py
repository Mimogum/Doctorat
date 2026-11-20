import numpy as np
import matplotlib.pyplot as plt

from .aagradient import archetypal_analysis as aagradient_analysis
from .Weighted import line_search_C_given_X, line_search_S_given_Z

def bisquare_weights(residuals, c_factor=6.0):
    """
    Calcula el vector de pesos w utilizando el estimador M Bisquare de Tukey por filas.
    """
    # 1. Calcular la norma L1 de los residuos por observación (filas)
    r_norms = np.linalg.norm(residuals, ord=1, axis=1) # (N,)

    for i, ri in enumerate(r_norms):
        if ri >= c_factor:
            r_norms[i] = 0
        else:
            r_norms[i] = (1- np.linalg.norm(residuals[i]/c_factor, ord=1)**2)**2

    return r_norms


def robust_archetypal_analysis(X, p, max_robust_iter=20, max_inner_iter=10, 
                               c_factor=2.0, tol=1e-5, w_init=None, random_state=42, verbose=False):
    """
    Ejecuta el Robust AA por IRLS utilizando filtrado de outliers (trimmmed robust AA).
    """
    X = np.asarray(X, dtype=float)
    N, M = X.shape

    # 1. Inicialización: Pesos unitarios en la primera iteración
    if w_init is not None:
        w_vec = np.asarray(w_init, dtype=float).copy()
    else:
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
    X_weighted = w_vec[:, np.newaxis] * X
    Z_final = C_final @ X_weighted
    
    ZZT = Z_final @ Z_final.T
    XZT = X_weighted @ Z_final.T
    S_final, _, _ = line_search_S_given_Z(X_weighted, Z_final, ZZT, XZT, S_final)

    XXT = X_weighted @ X_weighted.T
    ZXT = Z_final @ X_weighted.T
    C_final, _, _ = line_search_C_given_X(Z_final, X_weighted, XXT, ZXT, C_final)

    return C_final, S_final, Z_final, w_vec