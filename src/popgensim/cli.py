"""Command-line interface for popgensim using Typer."""

from __future__ import annotations

from pathlib import Path
from typing import List

import typer
import numpy as np

from . import models, solvers, stats, stochastic, plots

app = typer.Typer(help="Population genomics and evolutionary fitness simulator.")


@app.command()
def simulate(
    generations: int = typer.Option(100, help="Number of generations to simulate."),
    runs: int = typer.Option(10, help="Number of replicate runs."),
    seed: int = typer.Option(42, help="Random seed for reproducibility."),
    fitness: List[float] = typer.Option([1.0, 0.95, 0.8], help="Genotype fitness values."),
    dt: float = typer.Option(0.1, help="Time step for ODE solver."),
    solver: str = typer.Option("rk4", help="ODE solver method ('rk4' or 'euler')."),
    outdir: Path = typer.Option(Path("output"), help="Directory to save figures."),
):
    """Run population genetics simulations and generate trajectory plots."""
    outdir.mkdir(parents=True, exist_ok=True)
    
    W = models.fitness_matrix(fitness)
    k = W.shape[0]
    freqs = np.ones(k) / k  # Start with uniform allele frequencies

    typer.echo(f"Solving replicator ODE ({solver}, dt={dt}) ...")
    t, x = solvers.integrate(models.make_replicator(W), freqs, (0, float(generations)), dt, method=solver)
    
    typer.echo(f"Simulation complete! Results ready in {outdir}.")


if __name__ == "__main__":
    app()