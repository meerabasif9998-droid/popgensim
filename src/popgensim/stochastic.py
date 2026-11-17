"""Stochastic population genetics: Wright-Fisher drift with selection and mutation.

Each generation applies, in order:

1. Selection  : x_i' = x_i * f_i(x) / f_bar(x)         (deterministic)
2. Mutation   : x'' = x' @ M                           (deterministic)
3. Drift      : counts ~ Multinomial(2N, x'')  ->  x''' = counts / 2N

A diploid population of N individuals carries 2N gene copies, so 2N is the
number of draws in the sampling step. Smaller N means stronger drift.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .models import check_frequencies, genotype_frequencies, selection_step

SeedLike = Union[None, int, np.random.Generator, np.random.SeedSequence]


# --------------------------------------------------------------------------
# Mutation
# --------------------------------------------------------------------------
def mutation_matrix(n_alleles: int, mu: float) -> np.ndarray:
    """Uniform mutation matrix. ``M[i, j]`` = P(allele i becomes allele j).

    Each allele mutates away with total probability ``mu`` per generation,
    split evenly among the other k - 1 alleles. Rows sum to 1.
    """
    if not 0.0 <= mu <= 1.0:
        raise ValueError("mu must be between 0 and 1.")
    if n_alleles < 2:
        raise ValueError("Need at least 2 alleles for mutation.")
    M = np.full((n_alleles, n_alleles), mu / (n_alleles - 1))
    np.fill_diagonal(M, 1.0 - mu)
    return M


def check_mutation_matrix(M: np.ndarray, n_alleles: int) -> np.ndarray:
    """Validate that M is a (k, k) row-stochastic matrix."""
    matrix = np.asarray(M, dtype=float)
    if matrix.shape != (n_alleles, n_alleles):
        raise ValueError(f"Mutation matrix must be {n_alleles}x{n_alleles}.")
    if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1.0):
        raise ValueError("Mutation matrix rows must be non-negative and sum to 1.")
    return matrix


# --------------------------------------------------------------------------
# One generation
# --------------------------------------------------------------------------
def wright_fisher_step(
    x: np.ndarray,
    W: np.ndarray,
    n_copies: int,
    rng: np.random.Generator,
    M: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Advance allele frequencies by one Wright-Fisher generation.

    Parameters
    ----------
    x : current allele frequencies, shape (k,).
    W : fitness matrix from ``models.fitness_matrix``.
    n_copies : number of gene copies sampled (2N for N diploid individuals).
    rng : NumPy random Generator.
    M : optional mutation matrix (rows sum to 1).
    """
    x = selection_step(x, W)
    if M is not None:
        x = x @ M
    x = np.clip(x, 0.0, None)
    x = x / x.sum()  # guard against floating-point drift before sampling
    return rng.multinomial(n_copies, x) / n_copies


# --------------------------------------------------------------------------
# Whole simulations
# --------------------------------------------------------------------------
def simulate(
    x0: np.ndarray,
    W: np.ndarray,
    pop_size: int,
    generations: int,
    mu: float = 0.0,
    M: Optional[np.ndarray] = None,
    seed: SeedLike = None,
) -> np.ndarray:
    """Run one Wright-Fisher trajectory.

    Parameters
    ----------
    pop_size : N, the number of diploid individuals (2N gene copies).
    generations : number of generations to simulate.
    mu : uniform mutation probability; ignored if ``M`` is given.
    M : custom mutation matrix; overrides ``mu``.
    seed : int, SeedSequence, Generator, or None for a fresh generator.

    Returns
    -------
    X : array of shape (generations + 1, k); row t is the state at generation t.
    """
    x = check_frequencies(x0)
    k = x.size
    if pop_size < 1 or generations < 0:
        raise ValueError("pop_size must be >= 1 and generations >= 0.")
    mutation = M
    if mutation is not None:
        mutation = check_mutation_matrix(mutation, k)
    elif mu > 0:
        mutation = mutation_matrix(k, mu)

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    n_copies = 2 * pop_size

    X = np.empty((generations + 1, k))
    X[0] = x
    for t in range(1, generations + 1):
        x = wright_fisher_step(x, W, n_copies, rng, mutation)
        X[t] = x
    return X


def simulate_replicates(
    x0: np.ndarray,
    W: np.ndarray,
    pop_size: int,
    generations: int,
    n_reps: int,
    mu: float = 0.0,
    M: Optional[np.ndarray] = None,
    seed: Union[None, int] = None,
) -> np.ndarray:
    """Run many independent replicate trajectories.

    Each replicate gets its own child stream from ``SeedSequence.spawn``, so
    results are reproducible for a given ``seed`` and replicates are
    statistically independent.

    Returns
    -------
    R : array of shape (n_reps, generations + 1, k).
    """
    if n_reps < 1:
        raise ValueError("n_reps must be >= 1.")
    children = np.random.SeedSequence(seed).spawn(n_reps)
    return np.stack(
        [
            simulate(x0, W, pop_size, generations, mu=mu, M=M,
                     seed=np.random.default_rng(child))
            for child in children
        ]
    )


# --------------------------------------------------------------------------
# Genotype sampling (feeds the Hardy-Weinberg tests in stats.py)
# --------------------------------------------------------------------------
def sample_genotype_counts(
    x: np.ndarray, pop_size: int, seed: SeedLike = None
) -> np.ndarray:
    """Sample N newborn genotypes under random mating.

    Zygotes are formed in Hardy-Weinberg proportions given the allele
    frequencies, so counts ~ Multinomial(N, genotype_frequencies(x)).
    Order matches ``models.fitness_matrix``: (AA, Aa, aa) for two alleles.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    return rng.multinomial(pop_size, genotype_frequencies(np.asarray(x, dtype=float)))