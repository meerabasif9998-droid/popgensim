# popgensim

A Python command-line simulator for population genetics. It compares deterministic allele-frequency trajectories with stochastic Wright–Fisher simulations and saves plots of the results.

## Features

- Deterministic replicator-equation simulation using Euler or RK4.
- Wright–Fisher simulations with genetic drift, selection, and mutation.
- Fixation probability and confidence-interval summaries.
- Plots for allele trajectories, drift ensembles, and final allele frequencies.

## Requirements

Python 3.10 or newer.

## Install

Clone the repository and enter its folder:

```bash
git clone https://github.com/meerabasif9998-droid/popgensim.git
cd popgensim
```

Create a virtual environment and install the project:

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run a simulation

```bash
sim-pop simulate
```

For a short run:

```bash
sim-pop simulate --generations 5 --runs 3
```

The simulator saves three PNG files in the `output/` directory:

- `trajectories.png` — deterministic allele-frequency trajectories.
- `drift_ensemble.png` — stochastic paths with a confidence band and deterministic comparison.
- `fixation_hist.png` — final frequency distribution across runs.

Choose a different output directory with `--outdir`. See all options with:

```bash
sim-pop simulate --help
```

## Run tests

```bash
python -m pytest
```
