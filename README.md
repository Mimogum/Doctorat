# doctorat-mireia

Archetypal Analysis (AA) y Biarchetypal Analysis (BiAA), con variantes
ponderadas y robustas.

## Métodos (`src/`)

| Fichero | Función | Método |
|---|---|---|
| `aa.py` | `aa(X, k)` | AA: alterna S y C (descenso por coordenadas, Cutler & Breiman); inicio FurthestSum |
| `weighted_aa.py` | `weighted_aa(X, k, w)` | AA sobre diag(w)·X + postprocesado sobre X |
| `robust_aa.py` | `robust_aa(X, k)` | AA robusto, Huber por filas (Chen et al. 2014): IRLS sobre el núcleo de `aa` |
| `biaa.py` | `biaa(X, k, c)` | BiAA: alterna S0, S1 y C0, C1 por coordenadas |
| `robust_biaa.py` | `robust_biaa(X, k, c)` | BiAA robusto, Huber por celdas: IRLS sobre el núcleo de `biaa` |
| `simplex.py` | `solve(Y, V, solver, W, A0)` | Mínimos cuadrados en el símplex, por filas (compartido) |

## Notación

- AA: `X ≈ S Z`, `Z = C X`, con `S` (n×k) y `C` (k×n).
- BiAA: `X ≈ S0 Z S1ᵀ`, `Z = C0 X C1ᵀ`, con `S0` (n×k), `C0` (k×n),
  `S1` (m×c), `C1` (c×m).

Todas las matrices `S*` y `C*` son no negativas y cada fila suma 1.

Los AA devuelven `(S, C, Z, info)` y los BiAA `(S0, C0, S1, C1, Z, info)`.
`info` es un dict con `rss` y, según el método, `history`, `n_iter` y
`weights` (robustos: ~0 indica outlier).

Parámetros comunes: `max_iter`, `max_inner_iter`, `max_robust_iter`, `tol`,
`epsilon`, `seed=42`, `verbose=False`.

## Solver

Todos los subproblemas de todos los métodos pasan por
`simplex.solve(Y, V, solver, W=None, A0=None)`: cada fila de `Y` como
combinación convexa de las columnas de `V`, con pesos por celda `W` y
solución inicial `A0` opcionales. El solver decide si va fila a fila o en
bloque:

| `solver` | Cómo | Algoritmo | Error relativo |
|---|---|---|---|
| `"nnls"` (defecto) | fila a fila | NNLS con la suma 1 como fila extra | ~1e-8 |
| `"slsqp"` | fila a fila | optimizador genérico de scipy (el más lento) | ~4e-6 |
| `"pgd"` | en bloque | gradiente proyectado acelerado (FISTA) | ~2e-7 |

Los tres llegan al mismo óptimo; solo cambia el tiempo (`notebooks/01_aa_velocidad.qmd`).

## Notebooks

| Fichero | Contenido |
|---|---|
| `notebooks/01_aa_velocidad.qmd` | Tiempo y RSS de cada AA con cada solver |
| `notebooks/02_aa_ponderado_robusto.qmd` | AA vs ponderado vs robusto con outliers |
| `notebooks/03_biaa.qmd` | BiAA en datos sintéticos |
| `notebooks/04_biaa_robusto.qmd` | BiAA vs BiAA robusto con celdas outlier |

## Uso

```sh
uv sync
uv run pytest
```

Quarto debe usar el Python del proyecto (`.venv`), no el de otro entorno
activo (p. ej. conda). Una vez por máquina:

```sh
# Kernel de Jupyter que usan los qmd (jupyter: doctorat-mireia)
uv run python -m ipykernel install --user --name doctorat-mireia
# Python con el que Quarto lanza Jupyter (no se sube a git)
echo "QUARTO_PYTHON=$PWD/.venv/bin/python" > _environment.local
```

Después:

```sh
quarto render notebooks/01_aa_velocidad.qmd
```

En Positron, elegir el intérprete `uv: doctorat-mireia` para ejecutar las
celdas de los qmd en la consola.
