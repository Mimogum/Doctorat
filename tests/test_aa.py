import numpy as np
import pytest

from src.aa import aa
from src.robust_aa import robust_aa
from src.weighted_aa import weighted_aa
from tests.conftest import check_simplex, match_rows


def test_aa_recovers_archetypes(triangle_with_outlier):
    X, Z_true = triangle_with_outlier
    X = X[:-1]
    S, C, Z, info = aa(X, 3, tol=1e-8)
    check_simplex(S, C)
    assert np.allclose(Z, C @ X)
    assert info["rss"] / np.sum(X ** 2) < 1e-3
    assert match_rows(Z, Z_true) < 0.3


@pytest.mark.parametrize("solver", ["pgd", "slsqp"])
def test_aa_reaches_same_optimum_with_every_solver(X_aa, solver):
    # Mismo óptimo (159.18) que los antiguos aa_gradient y aa_nnls.
    *_, ref = aa(X_aa, 4, tol=1e-8)
    *_, info = aa(X_aa, 4, tol=1e-8, solver=solver)
    assert info["rss"] <= ref["rss"] * 1.01


def test_weighted_aa_outputs_consistent_on_original_X(X_aa):
    w = np.random.default_rng(2).uniform(0.5, 1.5, size=len(X_aa))
    S, C, Z, info = weighted_aa(X_aa, 4, w)
    check_simplex(S, C)
    assert np.allclose(Z, C @ X_aa)
    assert np.isclose(info["rss"], np.sum((X_aa - S @ Z) ** 2))
    # Con pesos uniformes el postprocesado no pierde ajuste frente a aa.
    *_, info_1 = weighted_aa(X_aa, 4, np.ones(len(X_aa)))
    *_, info_aa = aa(X_aa, 4)
    assert info_1["rss"] <= info_aa["rss"] * 1.01


def test_robust_aa_ignores_outlier(triangle_with_outlier):
    # 3 arquetipos no cubren las 4 esquinas: aa se tuerce (dist ~6.5).
    X, Z_true = triangle_with_outlier
    S, C, Z, info = robust_aa(X, 3, seed=1)
    check_simplex(S, C)
    w = info["weights"]
    assert w[-1] < 0.01 and np.median(w[:-1]) == 1.0
    assert match_rows(Z, Z_true) < 0.5
