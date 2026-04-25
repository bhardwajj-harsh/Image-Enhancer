"""
Visualization helpers: histograms and comparison grids.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _flatten_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale for histogram analysis.

    Parameters
    ----------
    image_rgb : np.ndarray
        Input RGB image.

    Returns
    -------
    np.ndarray
        Grayscale image.
    """
    if image_rgb.ndim == 2:
        return image_rgb
    r, g, b = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)


def plot_histograms_side_by_side(
    original_rgb: np.ndarray,
    enhanced_rgb: Optional[np.ndarray],
):
    """
    Plot intensity histograms of original and enhanced images.

    Parameters
    ----------
    original_rgb : np.ndarray
        Original RGB image.
    enhanced_rgb : np.ndarray, optional
        Enhanced RGB image.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the histograms.
    """
    orig_gray = _flatten_to_gray(original_rgb)
    enh_gray = _flatten_to_gray(enhanced_rgb) if enhanced_rgb is not None else None

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    axes[0].hist(orig_gray.ravel(), bins=256, range=(0, 255), color="blue", alpha=0.7)
    axes[0].set_title("Original Histogram")
    axes[0].set_xlabel("Intensity")
    axes[0].set_ylabel("Count")

    if enh_gray is not None:
        axes[1].hist(enh_gray.ravel(), bins=256, range=(0, 255), color="green", alpha=0.7)
        axes[1].set_title("Enhanced Histogram")
        axes[1].set_xlabel("Intensity")
        axes[1].set_ylabel("Count")
    else:
        axes[1].axis("off")

    fig.tight_layout()
    return fig


def create_comparison_grid(
    original_rgb: np.ndarray,
    noisy_rgb: np.ndarray,
    denoised_rgb: np.ndarray,
    enhanced_rgb: np.ndarray,
):
    """
    Create a 2x2 comparison grid figure.

    Layout:
        [Original   | Noisy   ]
        [Filtered   | Enhanced]

    Parameters
    ----------
    original_rgb : np.ndarray
        Original image.
    noisy_rgb : np.ndarray
        Noisy image.
    denoised_rgb : np.ndarray
        Filtered / denoised image.
    enhanced_rgb : np.ndarray
        Enhanced image.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with the 2x2 grid of images.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(noisy_rgb)
    axes[0, 1].set_title("Noisy")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(denoised_rgb)
    axes[1, 0].set_title("Filtered")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(enhanced_rgb)
    axes[1, 1].set_title("Enhanced")
    axes[1, 1].axis("off")

    fig.tight_layout()
    return fig

