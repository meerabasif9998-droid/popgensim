"""Plotting utilities: allele-frequency trajectories, drift ensembles, fitness heatmaps.

Functions take plain numpy arrays, so they work with any output from
solvers.py / stochastic.py / stats.py without importing them.

Conventions
-----------
- Deterministic trajectories: ``t`` has shape (n_steps,), ``x`` has shape
  (n_steps, n_alleles).
- Stochastic ensembles: ``runs`` has shape (n_runs, n_generations + 1) and
  holds the frequency of ONE focal allele.
"""

from __future__ import annotations

# Matplotlib and Seaborn stubs leave plotting member signatures partially untyped.
# pyright: reportUnknownMemberType=false

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
import seaborn as sns
from matplotlib.figure import Figure

sns.set_theme(style="whitegrid", context="notebook")


def _default_labels(n: int) -> list[str]:
    return [f"Allele {i + 1}" for i in range(n)]


def plot_trajectories(
    t: np.ndarray,
    x: np.ndarray,
    labels: Sequence[str] | None = None,
    title: str = "Allele frequency trajectories",
    xlabel: str = "Generation",
) -> Figure:
    """Line plot of every allele's frequency over time (replicator ODE output)."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.ndim != 1 or x.ndim != 2 or x.shape[0] != t.shape[0]:
        raise ValueError("x must have shape (len(t), n_alleles)")

    labels = list(labels) if labels is not None else _default_labels(x.shape[1])
    if len(labels) != x.shape[1]:
        raise ValueError("labels must have one entry per allele")

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, lab in enumerate(labels):
        ax.plot(t, x[:, i], lw=2, label=lab)
    ax.set(xlabel=xlabel, ylabel="Frequency", title=title, ylim=(-0.02, 1.02))
    ax.legend(frameon=True)
    fig.tight_layout()
    return fig


def plot_solver_comparison(
    t: np.ndarray,
    solutions: dict[str, np.ndarray],
    allele: int = 0,
    title: str = "Solver comparison",
) -> Figure:
    """Overlay solvers for one allele and plot their absolute differences."""
    t = np.asarray(t, dtype=float)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.5]}
    )
    for name, solution in solutions.items():
        ax1.plot(t, np.asarray(solution, dtype=float)[:, allele], lw=2, label=name)
    ax1.set(ylabel=f"Frequency (allele {allele + 1})", title=title)
    ax1.legend()

    names = list(solutions)
    if not names:
        raise ValueError("solutions must contain at least one solver")
    reference_name = names[-1]
    reference = np.asarray(solutions[reference_name], dtype=float)[:, allele]
    for name in names[:-1]:
        solution = np.asarray(solutions[name], dtype=float)[:, allele]
        error = np.abs(solution - reference)
        ax2.semilogy(t, error + 1e-16, label=f"|{name} - {reference_name}|")
    ax2.set(xlabel="Generation", ylabel="Abs. difference")
    if len(names) > 1:
        ax2.legend()
    fig.tight_layout()
    return fig


def plot_drift_ensemble(
    runs: np.ndarray,
    deterministic: np.ndarray | None = None,
    ci: tuple[np.ndarray, np.ndarray] | None = None,
    max_paths: int = 50,
    title: str = "Wright-Fisher drift ensemble",
) -> Figure:
    """Plot stochastic paths with an optional confidence band and ODE overlay."""
    run_values: NDArray[np.float64] = np.asarray(runs, dtype=np.float64)
    if run_values.ndim != 2:
        raise ValueError("runs must have shape (n_runs, n_generations + 1)")
    n_generations: int = run_values.shape[1]
    generations: NDArray[np.int64] = np.arange(n_generations, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(8, 5))
    for path in run_values[:max_paths]:
        ax.plot(generations, path, color="steelblue", alpha=0.2, lw=0.8)
    if ci is not None:
        lower: NDArray[np.float64] = np.asarray(ci[0], dtype=np.float64)
        upper: NDArray[np.float64] = np.asarray(ci[1], dtype=np.float64)
        ax.fill_between(
            generations, lower, upper, color="orange", alpha=0.3, label="Monte Carlo CI"
        )
    ensemble_mean: NDArray[np.float64] = run_values.mean(axis=0)
    ax.plot(generations, ensemble_mean, color="black", lw=2, label="Ensemble mean")
    if deterministic is not None:
        deterministic_values: NDArray[np.float64] = np.asarray(
            deterministic, dtype=np.float64
        )
        ax.plot(
            generations,
            deterministic_values,
            "r--",
            lw=2,
            label="Deterministic",
        )
    ax.set(
        xlabel="Generation",
        ylabel="Allele frequency",
        title=title,
        ylim=(-0.02, 1.02),
    )
    ax.legend()
    fig.tight_layout()
    return fig


def plot_fixation_histogram(
    final_freqs: np.ndarray,
    bins: int = 30,
    title: str = "Final allele frequency distribution",
) -> Figure:
    """Histogram of end-of-run frequencies (shows fixation or loss under drift)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(np.asarray(final_freqs, dtype=float), bins=bins, binrange=(0, 1), ax=ax)
    ax.set(xlabel="Final frequency", ylabel="Count", title=title)
    fig.tight_layout()
    return fig


def mean_fitness_grid(
    p_values: ArrayLike, s_values: ArrayLike, h: float = 0.0
) -> NDArray[np.float64]:
    """Diploid mean fitness grid; genotype fitnesses are 1, 1-h*s, and 1-s."""
    p = np.asarray(p_values, dtype=float)[None, :]
    s = np.asarray(s_values, dtype=float)[:, None]
    q = 1 - p
    return p**2 + 2 * p * q * (1 - h * s) + q**2 * (1 - s)


def plot_fitness_heatmap(
    values: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    xlabel: str = "Allele 1 frequency (p)",
    ylabel: str = "Selection coefficient (s)",
    title: str = "Mean population fitness",
    cmap: str = "viridis",
    n_ticks: int = 6,
) -> Figure:
    """Plot values as a heatmap with sparse x- and y-axis labels."""
    values = np.asarray(values, dtype=float)
    if values.shape != (len(y_values), len(x_values)):
        raise ValueError("values must have shape (len(y_values), len(x_values))")
    if n_ticks < 1:
        raise ValueError("n_ticks must be at least 1")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        values,
        ax=ax,
        cmap=cmap,
        cbar_kws={"label": "Mean fitness"},
        xticklabels=False,
        yticklabels=False,
    )
    x_indices = np.linspace(0, len(x_values) - 1, min(n_ticks, len(x_values))).astype(int)
    y_indices = np.linspace(0, len(y_values) - 1, min(n_ticks, len(y_values))).astype(int)
    ax.set_xticks(x_indices + 0.5)
    ax.set_xticklabels([f"{x_values[i]:.2f}" for i in x_indices], rotation=0)
    ax.set_yticks(y_indices + 0.5)
    ax.set_yticklabels([f"{y_values[i]:.2f}" for i in y_indices], rotation=0)
    ax.invert_yaxis()
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.tight_layout()
    return fig


def plot_hw_deviation(
    genotype_obs: np.ndarray,
    genotype_exp: np.ndarray,
    labels: Sequence[str] = ("AA", "Aa", "aa"),
    chi2: float | None = None,
    p_value: float | None = None,
) -> Figure:
    """Plot observed and Hardy-Weinberg expected genotype counts."""
    obs = np.asarray(genotype_obs, dtype=float)
    exp = np.asarray(genotype_exp, dtype=float)
    if obs.ndim != 1 or exp.ndim != 1 or len(obs) != len(labels) or len(exp) != len(labels):
        raise ValueError("genotype counts must be 1D arrays matching the number of labels")

    indices = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(indices - width / 2, obs, width, label="Observed")
    ax.bar(indices + width / 2, exp, width, label="HW expected")
    ax.set_xticks(indices)
    ax.set_xticklabels(labels)
    ax.set(ylabel="Count")
    title = "Hardy-Weinberg: observed vs expected"
    if chi2 is not None and p_value is not None:
        title += f"\n$\\chi^2$ = {chi2:.2f}, p = {p_value:.3g}"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def save_figure(
    fig: Figure, outdir: str | Path, name: str, dpi: int = 150, close: bool = True
) -> Path:
    """Save to ``outdir/name.png`` and return the path."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path
