"""The module for simulating the non-linearity of the pixels."""
import numpy as np

from libera_cam.constants import BIT_COUNT, PIXEL_COUNT_X, PIXEL_COUNT_Y


def apply_reverse_non_linear_function(
        image_counts: np.ndarray,
        pixel_dependency: np.ndarray = None):
    """The non-linearity function applied to each pixel including its spatial dependency"""
    if pixel_dependency is None:
        pixel_dependency = np.ones_like(image_counts, dtype=np.float32)

    adjusted_counts = 0.8 * pixel_dependency * (2.0 * image_counts - np.tan(0.8 * pixel_dependency * image_counts))

    return adjusted_counts


def create_reverse_pixel_dependency():
    """The pixel to pixel spatial dependency estimated for simulations"""
    # pixel dependency
    x1d = np.arange(PIXEL_COUNT_X)
    y1d = np.arange(PIXEL_COUNT_Y)
    x2d, y2d = np.meshgrid(x1d, y1d, indexing='ij')
    xc = PIXEL_COUNT_X // 2
    yc = PIXEL_COUNT_Y // 2
    pix_dep = 0.9 + 0.2 * (1.0 - ((x2d-xc)**2.0 + (y2d-yc)**2.0)/(((xc+yc)/2.0)**2.0))

    return pix_dep


def create_reverse_non_linear_factor(true_counts: np.ndarray):
    """A simulation tool to create an estimated non-linearity factor"""
    true_scaled = true_counts / (2.0 ** BIT_COUNT)

    pix_dep = create_reverse_pixel_dependency()

    nlin = apply_reverse_non_linear_function(true_scaled, pixel_dependency=pix_dep)

    scale_factor = true_scaled / nlin

    return scale_factor
