"""Command-line interface for popgensim using Typer."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import typer

from . import models, plots, solvers, stats, stochastic

app = typer.Typer(help="Population genomics and evolutionary fitness simulator.")


@app.callback()
def main() -> None:
    """Population genomics and evolutionary fitness simulator."""


@app.command()
def simulate(
    generations: int = typer.Option(100, min=1, help="Number of generations to simulate."),
    runs: int = typer.Option(10, min=1, help="Number of replicate Wright-Fisher runs."),
    seed: int = typer.Option(42, help="Random seed for reproducibility."),
    fitness: List[float] = typer.Option([1.0, 0.95, 0.8], help="Genotype fitness values."),
    dt: float = typer.Option(0.1, help="Time step for ODE solver."),
    solver: str = typer.Option("rk4", help="ODE solver method ('rk4' or 'euler')."),
    pop_size: int = typer.Option(200, min=1, help="Diploid population size N for drift."),
    mu: float = typer.Option(0.0, min=0.0, max=1.0, help="Per-generation mutation probability."),
    outdir: Path = typer.Option(Path("output"), help="Directory to save figures."),
):
    """Run deterministic + stochastic simulations and save plots."""
    try:
        W = models.fitness_matrix(fitness)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--fitness")
    k = W.shape[0]
    freqs = np.ones(k) / k  # start with uniform allele frequencies

    typer.echo(f"Solving replicator ODE ({solver}, dt={dt}) ...")
    try:
        t, x = solvers.integrate(
            models.make_replicator(W), freqs, (0, float(generations)), dt, method=solver
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    saved = [plots.save_figure(plots.plot_trajectories(t, x), outdir, "trajectories")]

    # Deterministic path sampled at integer generations, for overlay on drift runs.
    gens = np.arange(generations + 1)
    det = np.column_stack([np.interp(gens, t, x[:, i]) for i in range(k)])

    typer.echo(f"Running {runs} Wright-Fisher replicates (N={pop_size}, mu={mu}) ...")
    R = stochastic.simulate_replicates(
        freqs, W, pop_size, generations, runs, mu=mu, seed=seed
    )
    ci = stats.trajectory_ci(R)
    saved.append(
        plots.save_figure(
            plots.plot_drift_ensemble(
                R[:, :, 0], deterministic=det[:, 0], ci=(ci.lower[:, 0], ci.upper[:, 0])
            ),
            outdir,
            "drift_ensemble",
        )
    )
    saved.append(
        plots.save_figure(plots.plot_fixation_histogram(R[:, -1, 0]), outdir, "fixation_hist")
    )

    fix = stats.fixation_probability(R, allele=0, threshold=0.99 if mu > 0 else 1.0)
    typer.echo(
        f"P(allele 1 fixed) = {fix.estimate:.2f} "
        f"(95% CI {fix.lower:.2f}-{fix.upper:.2f}, n={fix.n})"
    )
    typer.echo(f"Simulation complete! Saved {len(saved)} figures to {outdir}.")


if __name__ == "__main__":
    app()
    