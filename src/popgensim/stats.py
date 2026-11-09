"""Biostatistics for the simulator.

* Hardy-Weinberg equilibrium (HWE) chi-square test on genotype counts.
* Genotype proportions after selection (how selection pushes away from HWE).
* Population fitness metrics along a trajectory.
* Monte Carlo summaries across replicate runs: percentile bands for allele
  frequencies and fixation probability with a Wilson confidence interval.

Genotype order everywhere matches ``models.fitness_matrix``: row-major over
i <= j, i.e. (AA, Aa, aa) for two alleles.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from .models import genotype_frequencies, infer_n_alleles


def _genotype_pairs(k: int) -> list[tuple[int, int]]:
    return list(combinations_with_replacement(range(k), 2))


# --------------------------------------------------------------------------
# Hardy-Weinberg
# --------------------------------------------------------------------------
def allele_frequencies_from_counts(counts: Sequence[float]) -> np.ndarray:
    """Estimate allele frequencies from diploid genotype counts.

    Each homozygote ii contributes two copies of allele i; each heterozygote
    ij contributes one copy each of i and j. Total copies = 2N.
    """
    c = np.asarray(counts, dtype=float)
    k = infer_n_alleles(c.size)
    x = np.zeros(k)
    for n, (i, j) in zip(c, _genotype_pairs(k)):
        x[i] += n
        x[j] += n
    return x / (2.0 * c.sum())


@dataclass(frozen=True)
class HWEResult:
    """Outcome of a Hardy-Weinberg chi-square test."""

    chi2: float
    dof: int
    p_value: float
    observed: np.ndarray
    expected: np.ndarray
    allele_freqs: np.ndarray
    low_expected: bool  # True if any expected count < 5 (chi-square unreliable)

    def reject(self, alpha: float = 0.05) -> bool:
        """True if HWE is rejected at significance level ``alpha``."""
        return self.p_value < alpha


def hardy_weinberg_test(counts: Sequence[float]) -> HWEResult:
    """Chi-square goodness-of-fit test for Hardy-Weinberg proportions.

    Allele frequencies are estimated from the counts, which costs degrees of
    freedom: dof = (#genotypes - 1) - (#alleles - 1) = m(m - 1)/2 for m
    segregating alleles (1 for the usual two-allele case).

    Alleles absent from the sample are dropped. Raises ValueError if fewer
    than two alleles segregate, because the test is undefined then.
    """
    obs = np.asarray(counts, dtype=float)
    if np.any(obs < 0):
        raise ValueError("Genotype counts must be non-negative.")
    total = obs.sum()
    if total <= 0:
        raise ValueError("Need at least one individual.")

    x = allele_frequencies_from_counts(obs)
    m = int((x > 0).sum())
    if m < 2:
        raise ValueError("HWE test is undefined for a monomorphic sample.")

    expected = total * genotype_frequencies(x)
    keep = expected > 0
    chi2 = float(np.sum((obs[keep] - expected[keep]) ** 2 / expected[keep]))
    dof = m * (m - 1) // 2
    p_value = float(sps.chi2.sf(chi2, dof))

    return HWEResult(
        chi2=chi2,
        dof=dof,
        p_value=p_value,
        observed=obs,
        expected=expected,
        allele_freqs=x,
        low_expected=bool((expected[keep] < 5).any()),
    )


def genotype_fitness_vector(W: np.ndarray) -> np.ndarray:
    """Flatten the fitness matrix into genotype order (i <= j)."""
    return np.array([W[i, j] for i, j in _genotype_pairs(W.shape[0])])


def genotype_frequencies_after_selection(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Genotype proportions among survivors: HW proportions * fitness / mean.

    Newborns are in HWE, but selection reweights genotypes, so survivors are
    generally *not*. Sampling counts from this and running
    ``hardy_weinberg_test`` shows selection-driven HWE deviation.
    """
    g = genotype_frequencies(np.asarray(x, dtype=float)) * genotype_fitness_vector(W)
    return g / g.sum()


# --------------------------------------------------------------------------
# Fitness metrics along a trajectory
# --------------------------------------------------------------------------
def fitness_summary(X: np.ndarray, W: np.ndarray) -> pd.DataFrame:
    """Per-generation population metrics for one trajectory.

    Columns: ``generation``, ``p0..p{k-1}`` (allele frequencies),
    ``mean_fitness`` (x^T W x), ``genetic_load`` (1 - mean_fitness / max W,
    the fitness lost relative to the best genotype) and ``heterozygosity``
    (expected, 1 - sum x_i^2).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != W.shape[0]:
        raise ValueError("X must have shape (generations + 1, k) matching W.")
    mean_w = np.einsum("ti,ij,tj->t", X, W, X)
    df = pd.DataFrame(X, columns=[f"p{i}" for i in range(X.shape[1])])
    df.insert(0, "generation", np.arange(X.shape[0]))
    df["mean_fitness"] = mean_w
    df["genetic_load"] = 1.0 - mean_w / W.max()
    df["heterozygosity"] = 1.0 - (X**2).sum(axis=1)
    return df


# --------------------------------------------------------------------------
# Monte Carlo summaries across replicates
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TrajectoryCI:
    """Percentile band for allele-frequency trajectories across replicates.

    All arrays have shape (generations + 1, k). The band shows where
    ``level`` of *individual populations* fall at each generation, i.e. the
    spread caused by drift, not the uncertainty of the mean.
    """

    level: float
    mean: np.ndarray
    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        """Tidy long-format table: generation, allele, mean, median, lower, upper."""
        T, k = self.mean.shape
        return pd.DataFrame(
            {
                "generation": np.repeat(np.arange(T), k),
                "allele": np.tile(np.arange(k), T),
                "mean": self.mean.ravel(),
                "median": self.median.ravel(),
                "lower": self.lower.ravel(),
                "upper": self.upper.ravel(),
            }
        )


def trajectory_ci(R: np.ndarray, level: float = 0.95) -> TrajectoryCI:
    """Monte Carlo percentile band from replicates of shape (reps, T+1, k)."""
    R = np.asarray(R, dtype=float)
    if R.ndim != 3:
        raise ValueError("R must have shape (n_reps, generations + 1, k).")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1.")
    tail = 100.0 * (1.0 - level) / 2.0
    return TrajectoryCI(
        level=level,
        mean=R.mean(axis=0),
        median=np.median(R, axis=0),
        lower=np.percentile(R, tail, axis=0),
        upper=np.percentile(R, 100.0 - tail, axis=0),
    )


@dataclass(frozen=True)
class ProportionEstimate:
    """A proportion with a confidence interval."""

    estimate: float
    lower: float
    upper: float
    n: int


def wilson_interval(successes: int, n: int, level: float = 0.95) -> ProportionEstimate:
    """Wilson score interval for a binomial proportion.

    Preferred over the plain normal interval because it behaves sensibly near
    0 and 1, exactly where fixation probabilities often sit.
    """
    if n < 1 or not 0 <= successes <= n:
        raise ValueError("Need n >= 1 and 0 <= successes <= n.")
    z = sps.norm.ppf(1.0 - (1.0 - level) / 2.0)
    phat = successes / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return ProportionEstimate(phat, float(center - half), float(center + half), n)


def fixation_probability(
    R: np.ndarray, allele: int = 0, threshold: float = 1.0, level: float = 0.95
) -> ProportionEstimate:
    """Fraction of replicates in which ``allele`` is fixed by the final generation.

    "Fixed" means final frequency >= ``threshold``. With mutation, alleles
    never reach exactly 1, so pass something like ``threshold=0.99``.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 3:
        raise ValueError("R must have shape (n_reps, generations + 1, k).")
    fixed = int((R[:, -1, allele] >= threshold - 1e-12).sum())
    return wilson_interval(fixed, R.shape[0], level)