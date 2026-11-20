import numpy as np
import matplotlib.pyplot as plt

from .biaa import biarchetypal_analysis, compute_alpha, compute_gamma, compute_Z_from_alpha, compute_beta, compute_theta

def bisquare_weights(residuals, c_factor=6.0):
    """
    Calcula el vector de pesos w utilizando el estimador M Bisquare de Tukey por filas.
    """
    abs_res = np.abs(residuals)
    mad = np.median(abs_res)
    scale = 1.4826 * mad + 1e-8

    u = abs_res / (c_factor * scale)
    w = np.where(u >= 1.0, 0.0, (1.0 - u**2)**2)
    return w


def robust_biarchetypal_analysis_mxnx1(X, k, c, max_robust_iter=20, max_inner_iter=50, 
                               c_factor=6.0, tol=1e-5, w_init=None, random_state=42, verbose=False):
    """
    Ejecuta el Robust AA por IRLS utilizando filtrado de outliers (trimmmed robust AA).
    Retorna la matriz ponderada con dimensiones (m, n, 1).
    """
    X = np.asarray(X, dtype=float)
    N, M = X.shape  # N = m (filas), M = n (columnas)

    # 1. Inicialización: Pesos unitarios en la primera iteración
    if w_init is not None:
        W_mat = np.asarray(w_init, dtype=float).copy()
        if W_mat.shape != (N, M):
            raise ValueError(f"w_init debe tener dimensiones ({N}, {M})")
    else:
        W_mat = np.ones((N, M), dtype=float)

    alpha, beta, gamma, theta, Z, _ = biarchetypal_analysis(X, k=k, c=c, max_iter=max_inner_iter, tol=tol, seed=random_state, verbose=False)

    for t in range(max_robust_iter):
        # 1. Residuos celda por celda sobre los datos reales originales: R = X - alpha @ Z @ gamma
        X_reconstructed = alpha @ Z @ gamma
        residuals = X - X_reconstructed

        # 2. Actualización adaptativa celda por celda mediante Bisquare
        W_mat_new = bisquare_weights(residuals, c_factor=c_factor)

        # 3. Criterio de convergencia de la matriz de pesos (Norma de Frobenius)
        weight_diff = np.linalg.norm(W_mat_new - W_mat, ord='fro') / (np.linalg.norm(W_mat, ord='fro') + 1e-10)

        W_mat = W_mat_new

        # 4. Actualización de los datos ponderados y recálculo de coeficientes
        X_weighted = W_mat * X
        alpha, beta, gamma, theta, Z, _ = biarchetypal_analysis(
            X_weighted, 
            k=k, 
            c=c, 
            max_iter=1, 
            tol=tol, 
            seed=random_state, 
            verbose=False
        )

        if verbose:
            zero_cells = np.sum(W_mat == 0.0)
            total_cells = N * M
            print(
                f"Iteración Robust {t+1}/{max_robust_iter} | "
                f"Dif. pesos: {weight_diff:.6f} | "
                f"Celdas Outliers (w=0): {zero_cells}/{total_cells} ({(zero_cells/total_cells)*100:.2f}%)"
            )

        if weight_diff < tol:
            break

    # Devolvemos los factores de biarquetipos, los pesos matriciales W 
    return alpha, beta, gamma, theta, Z, W_mat