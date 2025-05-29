"""The plotting tools for normal operations"""

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np

from libera_cam.constants import IntegrationTime
from libera_cam.plotting_tools.plotting_utils import add_colorbar_to_axes, remove_ticks_from_axes


def plot_observed_vs_true_plus_relative_difference(
    true_data: np.ndarray,
    observed_data: np.ndarray,
    integration_time: IntegrationTime,
    observation_time: datetime.datetime,
    subplot_titles: list[str] = ("Radiance Truth", "Radiance Observed", "Relative Difference (%)"),
    save_plot: bool = False,
    vmin: float = 0.0,
    vmax: float = 0.5,
    relative_diff_vmin: float = -30.0,
    relative_diff_vmax: float = 30.0,
    cmap: str = "jet",
    relative_diff_cmap: str = "seismic",
) -> plt.Figure:
    """Plots observed vs. true data with relative difference.

    Generates a figure with three subplots:
    1.  True data (ground truth).
    2.  Observed data.
    3.  Relative difference between observed and true data.

    All data are expected to be 2D arrays compatible with the `imshow` method.

    Parameters
    ----------
    true_data : np.ndarray
        The ground truth data (e.g., reference radiance).
    observed_data : np.ndarray
        The observed data (e.g., camera-derived radiance).
    integration_time : IntegrationTime
        The integration time used for the observation.
    observation_time : datetime.datetime
        The time of the observation.
    subplot_titles : Optional[List[str]], optional
        Titles for the three subplots. If None, default titles are used.
        Defaults to None.
    save_plot : bool, optional
        Whether to save the plot to a file. Defaults to False.
    vmin : float, optional
        The minimum value for the color scale of the true and observed data plots.
        Defaults to 0.0.
    vmax : float, optional
        The maximum value for the color scale of the true and observed data plots.
        Defaults to 0.5.
    relative_diff_vmin : float, optional
        The minimum value for the color scale of the relative difference plot.
        Defaults to -30.0.
    relative_diff_vmax : float, optional
        The maximum value for the color scale of the relative difference plot.
        Defaults to 30.0.
    cmap : str, optional
        The colormap to use for the true and observed data plots.
        Defaults to "jet".
    relative_diff_cmap : str, optional
        The colormap to use for the relative difference plot.
        Defaults to "seismic".

    Returns
    -------
    plt.Figure
        The generated Matplotlib figure.

    Raises
    ------
    ValueError
        If `true_data` and `observed_data` do not have the same shape.
    ValueError
        If `subplot_titles` is provided and does not contain exactly three strings.
    """
    if true_data.shape != observed_data.shape:
        raise ValueError(
            f"true_data and observed_data must have the same shape. Got {true_data.shape} and {observed_data.shape}."
        )

    if subplot_titles is None:
        subplot_titles = [
            "Radiance Truth",
            "Radiance Observed",
            "Relative Difference (%)",
        ]
    elif len(subplot_titles) != 3:
        raise ValueError(
            f"subplot_titles must contain exactly three strings. Got {len(subplot_titles)}: {subplot_titles}"
        )

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(f"{observation_time.strftime('%Y-%m-%d %H:%M:%S')} with Integration Time= {integration_time.value}ms")

    # True Data Plot
    ax1 = fig.add_subplot(131)
    cs = ax1.imshow(true_data.T, origin="lower", cmap=cmap, zorder=0, vmin=vmin, vmax=vmax)
    ax1.set_title(subplot_titles[0])
    add_colorbar_to_axes(fig, ax1, cs)
    remove_ticks_from_axes(ax1)

    # Observed Data Plot
    ax2 = fig.add_subplot(132)
    cs = ax2.imshow(observed_data.T, origin="lower", cmap=cmap, zorder=0, vmin=vmin, vmax=vmax)
    ax2.set_title(subplot_titles[1])
    add_colorbar_to_axes(fig, ax2, cs)
    remove_ticks_from_axes(ax2)

    # Relative Difference Plot
    relative_difference = (observed_data - true_data) / true_data * 100.0
    ax3 = fig.add_subplot(133)
    cs = ax3.imshow(
        relative_difference.T,
        origin="lower",
        cmap=relative_diff_cmap,
        zorder=0,
        vmin=relative_diff_vmin,
        vmax=relative_diff_vmax,
    )
    ax3.set_title(subplot_titles[2])
    add_colorbar_to_axes(fig, ax3, cs)
    remove_ticks_from_axes(ax3)

    if save_plot:
        fig.subplots_adjust(hspace=0.6, wspace=0.6)
        _metadata_ = {
            "Computer": os.uname()[1],
            "Script": os.path.abspath(__file__),
            "Function": "plot_observed_vs_true_plus_relative_difference",
            "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        plt.savefig(
            f"{_metadata_['Function']}_{observation_time.strftime('%Y-%m-%d_%H:%M:%S')}.png",
            bbox_inches="tight",
            metadata=_metadata_,
        )

    return fig
