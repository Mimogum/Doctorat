import numpy as np
import pytest

from src.biaa import biaa
from src.robust_biaa import robust_biaa
from tests.conftest import check_simplex


@pytest.mark.parametrize("seed", [42, 0, 1])
def test_biaa_recovers_exact_biarchetypal_data(X_biaa, seed):
    X, Z_true = X_biaa
    S0, C0, S1, C1, Z, info = biaa(X, 3, 2, seed=seed)

    check_simplex(S0, C0, S1, C1)
    assert np.allclose(Z, C0 @ X @ C1.T)
    assert info["rss"] / np.sum(X ** 2) < 1e-3
    # Mismos valores salvo permutación de filas/columnas.
    assert np.allclose(np.sort(Z.ravel()), np.sort(Z_true.ravel()), atol=0.1)


def test_biaa_loss_never_increases(X_biaa):
    # El descenso por coordenadas es monótono (el antiguo paso pinv no lo era).
    X, _ = X_biaa
    *_, info = biaa(X + np.random.default_rng(5).normal(size=X.shape), 3, 2)
    assert np.all(np.diff(info["history"]) <= 1e-9 * info["history"][0])


def test_robust_biaa_recovers_clean_data(X_biaa):
    X, Z_true = X_biaa
    S0, C0, S1, C1, Z, info = robust_biaa(X, 3, 2)

    check_simplex(S0, C0, S1, C1)
    assert np.allclose(np.sort(Z.ravel()), np.sort(Z_true.ravel()), atol=0.1)


def test_robust_biaa_flags_outlier_cells(X_biaa):
    # Solo se comprueban los pesos: la Z recuperada depende de la semilla
    # (óptimos locales con error de 1.5 a 25 en este caso).
    X, _ = X_biaa
    X = X.copy()
    cells = [(10, 3), (25, 5), (40, 6)]
    for i, j in cells:
        X[i, j] += 50

    *_, info = robust_biaa(X, 3, 2, max_robust_iter=10)

    w = info["weights"]
    assert w.shape == X.shape
    assert max(w[i, j] for i, j in cells) < 0.01 and np.median(w) == 1.0
