import numpy as np
import pytest


@pytest.fixture
def X_aa():
    """200 puntos en 5D dentro de 4 arquetipos, con ruido."""
    rng = np.random.default_rng(0)
    Z_true = rng.normal(size=(4, 5)) * 10
    return rng.dirichlet(np.ones(4) * 0.5, 200) @ Z_true + rng.normal(size=(200, 5)) * 0.5


@pytest.fixture
def triangle_with_outlier():
    """Triángulo 2D limpio (vértices incluidos) + 1 outlier fuera."""
    rng = np.random.default_rng(0)
    Z_true = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    X = np.vstack([Z_true, rng.dirichlet(np.ones(3), size=57) @ Z_true, [[30.0, 30.0]]])
    return X, Z_true


@pytest.fixture
def X_biaa():
    """X = S0 Z S1.T exacto, con filas y columnas puras."""
    rng = np.random.default_rng(1)
    n, m, k, c = 60, 8, 3, 2
    Z_true = rng.random((k, c)) * 10
    S0 = np.vstack([np.eye(k), rng.dirichlet(np.ones(k), size=n - k)])
    S1 = np.vstack([np.eye(c), rng.dirichlet(np.ones(c), size=m - c)])
    return S0 @ Z_true @ S1.T, Z_true


def match_rows(Z, Z_true):
    """Distancia máxima de cada arquetipo real al estimado más cercano."""
    return np.linalg.norm(Z[:, None] - Z_true[None], axis=2).min(axis=0).max()


def check_simplex(*Ms):
    for M in Ms:
        assert (M >= 0).all() and np.allclose(M.sum(axis=1), 1.0)
