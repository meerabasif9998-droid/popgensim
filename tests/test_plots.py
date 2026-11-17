import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from popgensim.plots import plot_trajectories


@pytest.mark.parametrize("labels", [["A"], ["A", "B", "C"]])
def test_plot_trajectories_requires_one_label_per_allele(labels):
    with pytest.raises(ValueError, match="one entry per allele"):
        plot_trajectories(np.array([0, 1]), np.ones((2, 2)), labels=labels)


def test_plot_trajectories_requires_one_dimensional_timepoints():
    with pytest.raises(ValueError, match=r"shape \(len\(t\), n_alleles\)"):
        plot_trajectories(np.array([[0, 1]]), np.ones((2, 2)))


def test_plot_trajectories_plots_each_allele_with_its_label():
    fig = plot_trajectories(
        np.array([0, 1]), np.ones((2, 2)), labels=["A", "B"]
    )

    assert [line.get_label() for line in fig.axes[0].lines] == ["A", "B"]
