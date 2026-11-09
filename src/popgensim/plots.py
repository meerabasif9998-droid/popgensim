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

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
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
    t = np.asarray(t)
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[0] != t.shape[0]:
        raise ValueError("x must have shape (len(t), n_alleles)")

    labels = list(labels) if labels is not None else _default_labels(x.shape[1])
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
    """Overlay several solvers (e.g. Euler vs RK4) for one allele, plus
    the absolute difference from the last solver in the dict (assumed reference)."""
    t = np.asarray(t)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.5]}
    )
    for name, x in solutions.items():
        ax1.plot(t, np.asarray(x)[:, allele], lw=2, label=name)
    ax1.set(ylabel=f"Frequency (allele {allele + 1})", title=title)
    ax1.legend()

    names = list(solutions)
    ref = np.asarray(solutions[names[-1]])[:, allele]
    for name in names[:-1]:
        err = np.abs(np.asarray(solutions[name])[:, allele] - ref)
        ax2.semilogy(t, err + 1e-16, label=f"|{name} - {names[-1]}|")
    ax2.set(xlabel="Generation", ylabel="Abs. difference")
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
    """Stochastic paths for one allele, optionally with a Monte Carlo CI band
    and the deterministic ODE trajectory overlaid."""
    runs = np.asarray(runs)
    if runs.ndim != 2:
        raise ValueError("runs must have shape (n_runs, n_generations + 1)")
    gens = np.arange(runs.shape[1])

    fig, ax = plt.subplots(figsize=(8, 5))
    for path in runs[:max_paths]:
        ax.plot(gens, path, color="steelblue", alpha=0.2, lw=0.8)
    if ci is not None:
        lo, hi = ci
        ax.fill_between(gens, lo, hi, color="orange", alpha=0.3, label="Monte Carlo CI")
    ax.plot(gens, runs.mean(axis=0), color="black", lw=2, label="Ensemble mean")
    if deterministic is not None:
        ax.plot(gens, np.asarray(deterministic), "r--", lw=2, label="Deterministic")
    ax.set(xlabel="Generation", ylabel="Allele frequency", title=title, ylim=(-0.02, 1.02))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_fixation_histogram(
    final_freqs: np.ndarray,
    bins: int = 30,
    title: str = "Final allele frequency distribution",
) -> Figure:
    """Histogram of end-of-run frequencies (shows fixation / loss under drift)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(np.asarray(final_freqs), bins=bins, binrange=(0, 1), ax=ax)
    ax.set(xlabel="Final frequency", ylabel="Count", title=title)
    fig.tight_layout()
    return fig


def mean_fitness_grid(
    p_values: np.ndarray, s_values: np.ndarray
) -> np.ndarray:
    """Mean fitness surface for a 2-allele haploid model.

    Allele 1 has fitness 1, allele 2 has fitness 1 - s, so
    w_bar(p, s) = p * 1 + (1 - p) * (1 - s).
    Returns an array of shape (len(s_values), len(p_values)).
    """
    p = np.asarray(p_values)[None, :]
    s = np.asarray(s_values)[:, None]
    return p + (1 - p) * (1 - s)


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
    """Generic heatmap; ``values`` has shape (len(y_values), len(x_values))."""
    values = np.asarray(values)
    if values.shape != (len(y_values), len(x_values)):
        raise ValueError("values must have shape (len(y_values), len(x_values))")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        values,
        ax=ax,
        cmap=cmap,
        cbar_kws={"label": "Mean fitness"},
        xticklabels=False,
        yticklabels=False,
    )
    # Readable sparse ticks instead of one label per cell.
    xi = np.linspace(0, len(x_values) - 1, n_ticks).astype(int)
    yi = np.linspace(0, len(y_values) - 1, n_ticks).astype(int)
    ax.set_xticks(xi + 0.5)
    ax.set_xticklabels([f"{x_values[i]:.2f}" for i in xi], rotation=0)
    ax.set_yticks(yi + 0.5)
    ax.set_yticklabels([f"{y_values[i]:.2f}" for i in yi], rotation=0)
    ax.invert_yaxis()  # low s at the bottom
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
    """Observed vs. Hardy-Weinberg expected genotype counts."""
    obs = np.asarray(genotype_obs, dtype=float)
    exp = np.asarray(genotype_exp, dtype=float)
    idx = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(idx - w / 2, obs, w, label="Observed")
    ax.bar(idx + w / 2, exp, w, label="HW expected")
    ax.set_xticks(idx)
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
    """Save to ``outdir/name.png`` (creating the directory) and return the path."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path