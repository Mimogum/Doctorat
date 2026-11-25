import numpy as np
import pytest

from src.simplex import SOLVERS, solve

TRIANGLE = np.array([[0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])  # columnas = vértices


def err(Y, V, A):
    return np.sum((np.atleast_2d(Y) - A @ V.T) ** 2, axis=1)


@pytest.mark.parametrize("solver", SOLVERS)
def test_projection_outside_triangle_matches_closed_form(solver):
    # (10, 10) se proyecta en el punto medio de la hipotenusa: (5, 5).
    A = solve([10.0, 10.0], TRIANGLE, solver)
    assert np.allclose(A, [[0.0, 0.5, 0.5]], atol=1e-6)


@pytest.mark.parametrize("solver", SOLVERS)
def test_interior_points_reconstruct_exactly(solver):
    # nnls: falla si su ridge crece x10 (error ~1e-6 en vez de ~1e-8).
    # slsqp para por ftol (~4e-6 medido) y pgd por tolerancia.
    rng = np.random.default_rng(0)
    V = rng.normal(size=(5, 20)) * 10
    Y = rng.dirichlet(np.ones(20), 50) @ V.T  # 50 filas, en bloque
    A = solve(Y, V, solver)
    rel = np.linalg.norm(Y - A @ V.T, axis=1) / np.linalg.norm(Y, axis=1)
    assert rel.max() < {"nnls": 1e-7, "slsqp": 1e-5, "pgd": 1e-5}[solver]


@pytest.mark.parametrize("solver", SOLVERS)
def test_ties_split_instead_of_picking_a_vertex(solver):
    # Dos candidatos idénticos: repartir el peso evita que BiAA colapse
    # dos arquetipos en el mismo punto.
    v0, v1 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    A = solve(v0, np.column_stack([v0, v0, v1]), solver)
    assert np.allclose(A, [[0.5, 0.5, 0.0]], atol=1e-4)


@pytest.mark.parametrize("solver", SOLVERS)
def test_cell_weights_equal_scaled_problem(solver):
    # Pesos W en solve == escalar cada fila por sqrt(W) a mano.
    rng = np.random.default_rng(2)
    V = rng.normal(size=(6, 4)) * 5
    Y = rng.normal(size=(8, 6)) * 5
    W = rng.uniform(0.1, 3.0, size=Y.shape)
    A = solve(Y, V, solver, W)
    A_ref = np.vstack([solve(np.sqrt(w) * y, np.sqrt(w)[:, None] * V, "nnls") for y, w in zip(Y, W)])
    e = lambda A: np.sum(W * (Y - A @ V.T) ** 2, axis=1)
    assert np.allclose(e(A), e(A_ref), rtol=1e-4, atol=1e-8)


def test_solvers_agree_on_random_problems():
    rng = np.random.default_rng(1)
    for d, q in [(5, 4), (5, 30), (30, 10)]:
        V = rng.normal(size=(d, q)) * 10
        Y = rng.normal(size=(10, d)) * 15
        errs = [err(Y, V, solve(Y, V, s)) for s in SOLVERS]
        for e in errs[1:]:
            assert np.allclose(e, errs[0], rtol=1e-4, atol=1e-6)


def test_unknown_solver_is_rejected():
    with pytest.raises(ValueError):
        solve(np.zeros(2), TRIANGLE, "adam")
