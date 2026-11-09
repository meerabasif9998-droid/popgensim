"""Population-genetic models: genotype fitness and the replicator equation.

Conventions
-----------
* ``x``  : allele frequencies, shape (k,), non-negative, sums to 1.
* ``W``  : symmetric (k, k) fitness matrix. ``W[i, j]`` is the fitness of the
           diploid genotype carrying alleles i and j.
* Genotype order for a flat list of fitnesses is row-major over i <= j.
  For k = 2 this is (AA, Aa, aa); for k = 3 it is
  (11, 12, 13, 22, 23, 33).
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Callable, Optional, Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


# --------------------------------------------------------------------------
# Fitness setup
# --------------------------------------------------------------------------
def n_genotypes(n_alleles: int) -> int:
    """Number of distinct diploid genotypes for k alleles: k(k+1)/2."""
    return n_alleles * (n_alleles + 1) // 2


def infer_n_alleles(n_fitness_values: int) -> int:
    """Invert k(k+1)/2 = n. Raises if n is not a valid genotype count."""
    k = int(round((-1 + (1 + 8 * n_fitness_values) ** 0.5) / 2))
    if n_genotypes(k) != n_fitness_values:
        raise ValueError(
            f"{n_fitness_values} fitness values do not match any number of "
            "alleles (need 1, 3, 6, 10, ... values)."
        )
    return k


def fitness_matrix(
    genotype_fitness: ArrayLike, n_alleles: Optional[int] = None
) -> np.ndarray:
    """Build the symmetric fitness matrix W from a flat genotype list.

    Example (2 alleles): ``fitness_matrix([1.0, 0.95, 0.8])`` gives
    wAA = 1.0, wAa = 0.95, waa = 0.8.
    """
    w = np.asarray(genotype_fitness, dtype=float)
    if w.ndim != 1:
        raise ValueError("genotype_fitness must be a 1-D sequence.")
    if np.any(w < 0):
        raise ValueError("Fitness values must be non-negative.")

    k = infer_n_alleles(w.size) if n_alleles is None else n_alleles
    if w.size != n_genotypes(k):
        raise ValueError(
            f"{k} alleles need {n_genotypes(k)} genotype fitnesses, got {w.size}."
        )

    W = np.zeros((k, k))
    for value, (i, j) in zip(w, combinations_with_replacement(range(k), 2)):
        W[i, j] = W[j, i] = value
    return W


def check_frequencies(x: ArrayLike, atol: float = 1e-8) -> np.ndarray:
    """Validate that x is a probability vector and return it as an array."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Allele frequencies must be a 1-D array.")
    if np.any(x < -atol):
        raise ValueError("Allele frequencies must be non-negative.")
    if not np.isclose(x.sum(), 1.0, atol=atol):
        raise ValueError(f"Allele frequencies must sum to 1 (got {x.sum():.6f}).")
    return x


# --------------------------------------------------------------------------
# Fitness functions
# --------------------------------------------------------------------------
def allele_fitness(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Marginal fitness of each allele: f_i(x) = sum_j W[i, j] * x_j."""
    return W @ x


def mean_fitness(x: np.ndarray, W: np.ndarray) -> float:
    """Population mean fitness: f_bar(x) = sum_i x_i * f_i(x) = x^T W x."""
    return float(x @ W @ x)


# --------------------------------------------------------------------------
# Dynamics
# --------------------------------------------------------------------------
def replicator_rhs(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Right-hand side of the replicator equation.

        dx_i/dt = x_i * (f_i(x) - f_bar(x))

    Components sum to zero, so the simplex (sum x = 1) is preserved.
    """
    f = allele_fitness(x, W)
    return x * (f - x @ f)


def make_replicator(W: np.ndarray) -> Callable[[float, np.ndarray], np.ndarray]:
    """Return f(t, x) with W baked in, the signature the ODE solvers expect.

    The system is autonomous, so ``t`` is accepted but unused.
    """

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        return replicator_rhs(x, W)

    return rhs


def selection_step(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """One discrete generation of selection: x_i' = x_i * f_i(x) / f_bar(x).

    This is the discrete-time analogue of the replicator equation. The
    Wright-Fisher module will apply it before binomial/multinomial sampling.
    """
    f = allele_fitness(x, W)
    return x * f / (x @ f)


# --------------------------------------------------------------------------
# Genotype frequencies (used by the Hardy-Weinberg tests in stats.py)
# --------------------------------------------------------------------------
def genotype_frequencies(x: np.ndarray) -> np.ndarray:
    """Hardy-Weinberg genotype proportions in the same order as fitness input.

    Homozygote ii: x_i^2.  Heterozygote ij (i < j): 2 * x_i * x_j.
    """
    k = x.size
    out = np.empty(n_genotypes(k))
    for idx, (i, j) in enumerate(combinations_with_replacement(range(k), 2)):
        out[idx] = x[i] ** 2 if i == j else 2 * x[i] * x[j]
    return out