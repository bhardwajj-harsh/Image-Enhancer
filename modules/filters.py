"""
Noise removal filters.

Each function implements a classical spatial / neighborhood filter.
"""

import cv2
import numpy as np


def apply_gaussian_blur(image: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur (low-pass filtering).

    Gaussian blur smooths the image and reduces high-frequency noise
    like Gaussian noise, at the cost of slightly blurring edges.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    ksize : int
        Kernel size (odd number, e.g. 3, 5, 7).
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Blurred image.
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)


def apply_median_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Apply median filtering.

    Median filter is very effective for **salt & pepper noise**
    because it replaces each pixel by the median of its neighborhood,
    which removes outliers (0 or 255) but keeps edges sharper.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or color).
    ksize : int
        Neighborhood size (odd number).

    Returns
    -------
    np.ndarray
        Filtered image.
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(image, ksize)


def apply_bilateral_filter(
    image: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0
) -> np.ndarray:
    """
    Apply bilateral filtering.

    Bilateral filter smooths the image while **preserving edges** by
    considering both spatial distance and intensity difference.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    d : int
        Diameter of each pixel neighborhood.
    sigma_color : float
        Filter sigma in the color space.
    sigma_space : float
        Filter sigma in the coordinate space.

    Returns
    -------
    np.ndarray
        Filtered image.
    """
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def apply_non_local_means_denoising(
    image: np.ndarray,
    h: float = 10.0,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> np.ndarray:
    """
    Apply Non-local Means (NLM) denoising.

    NLM compares small patches in a larger search window and
    averages similar patches, which can preserve texture better.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image (uint8).
    h : float
        Filter strength (higher removes more noise but may remove details).
    template_window_size : int
        Size of the template patch.
    search_window_size : int
        Size of the window used to search for similar patches.

    Returns
    -------
    np.ndarray
        Denoised image.
    """
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h,
        h,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size,
    )

