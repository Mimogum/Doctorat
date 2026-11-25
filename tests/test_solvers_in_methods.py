import numpy as np
import pytest

from src.aa import aa
from src.biaa import biaa
from src.robust_aa import robust_aa
from src.robust_biaa import robust_biaa
from src.simplex import SOLVERS
from src.weighted_aa import weighted_aa
from tests.conftest import check_simplex

rng = np.random.default_rng(0)
X_AA = rng.dirichlet(np.ones(3), 30) @ (rng.normal(size=(3, 4)) * 10)
X_BI = rng.random((15, 8))

METHODS = {
    "aa": lambda s: aa(X_AA, 3, max_iter=20, solver=s),
    "weighted_aa": lambda s: weighted_aa(X_AA, 3, np.ones(30), max_iter=20, solver=s),
    "robust_aa": lambda s: robust_aa(X_AA, 3, max_robust_iter=3, max_inner_iter=5, solver=s),
    "biaa": lambda s: biaa(X_BI, 2, 2, max_iter=20, solver=s),
    "robust_biaa": lambda s: robust_biaa(X_BI, 2, 2, max_robust_iter=3, max_inner_iter=5, solver=s),
}


@pytest.mark.parametrize("method", METHODS)
def test_every_method_runs_with_every_solver(method):
    rss = {}
    for solver in SOLVERS:
        *factors, Z, info = METHODS[method](solver)
        check_simplex(*factors)
        rss[solver] = info["rss"]
    # Los tres llegan a ajustes comparables (no idénticos: óptimos locales).
    assert max(rss.values()) <= 2 * min(rss.values()) + 1e-6
