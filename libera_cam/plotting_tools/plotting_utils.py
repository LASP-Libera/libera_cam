"""Utility Functions for plotting"""
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_colorbar_to_axes(fig: plt.Figure, ax: plt.Axes, cs: mpimg.AxesImage) -> None:
    """Adds a colorbar to the specified axes."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    fig.colorbar(cs, cax=cax)


def remove_ticks_from_axes(ax: plt.Axes) -> None:
    """Removes x and y ticks from the specified axes."""
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
