""" Plotting tools for visualizing different calibration values"""
import matplotlib.pyplot as plt
import numpy as np

from libera_cam.correction_tools.non_linearity_corrections import load_non_linearity_parameters
from libera_cam.plotting_tools.plotting_utils import add_colorbar_to_axes, remove_ticks_from_axes
from libera_cam.simulation_tools.reverse_non_linearity import (
    apply_reverse_non_linear_function,
    create_reverse_pixel_dependency,
)


def plot_synthetic_non_linearity():
    """Plots the synthetic non-linearity calculated exactly and the polynomial reconstruction as a comparison"""
    fig = plt.figure(figsize=(4, 10))

    # Define the possible output range (used as input for the reversal)
    output_dn = np.linspace(0.0, 1.0)

    # First subplot is the non-linearity
    ax1 = fig.add_subplot(311, aspect='equal')
    ax2 = fig.add_subplot(312, aspect='equal')

    # Make a subset of expected pixel dependencies for visualization
    pix_deps = np.linspace(0.9, 1.1, 512)
    colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, pix_deps.size))
    # Define the polynomial coefficients
    non_linearity_coefficients = load_non_linearity_parameters(use_synthetic=True)
    for i in range(pix_deps.size):
        # Create and the simulated observed dn
        input_dn = apply_reverse_non_linear_function(output_dn, pix_deps[i])
        # Plot the simulated data in a color to match its location in the dependency map
        ax1.plot(output_dn, input_dn, lw=0.1, color=colors[i, ...])
        # Create the polynomial estimated non-linearity (using a diagonal line to see across the image)
        estimated_output = (non_linearity_coefficients[i+1023, i+1023, 0]*input_dn**5 +
                            non_linearity_coefficients[i+1023, i+1023, 1]*input_dn**4 +
                            non_linearity_coefficients[i+1023, i+1023, 2]*input_dn**3 +
                            non_linearity_coefficients[i+1023, i+1023, 3]*input_dn**2 +
                            non_linearity_coefficients[i+1023, i+1023, 4]*input_dn +
                            non_linearity_coefficients[i+1023, i+1023, 5])
        # Plot color coded to match
        ax2.plot(estimated_output, input_dn, lw=0.1, color=colors[511-i, ...])
    ax1.plot([0, 1], [0, 1], color='gray', ls='--')
    ax1.set_xlim((0, 1))
    ax1.set_ylim((0, 1))
    ax1.set_xlabel('True (Normalized DN)')
    ax1.set_ylabel('Obs. (Normalized DN)')


    ax2.plot([0, 1], [0, 1], color='gray', ls='--')
    ax2.set_xlim((0, 1))
    ax2.set_ylim((0, 1))
    ax2.set_xlabel('Polynomial Est. (Normalized DN)')
    ax2.set_ylabel('Obs. (Normalized DN)')

    # Second subplot is the pixel dependency
    ax3 = fig.add_subplot(313)

    pix_dependency = create_reverse_pixel_dependency()
    # Plot the pixel dependency as an image
    cs = ax3.imshow(pix_dependency.T, origin='lower', cmap='turbo', zorder=0)

    add_colorbar_to_axes(fig, ax3, cs)
    remove_ticks_from_axes(ax3)

    return fig
