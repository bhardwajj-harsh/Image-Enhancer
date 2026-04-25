"""
Image quality metrics: MSE and PSNR.
"""

from typing import Optional

import numpy as np


def calculate_mse(reference: np.ndarray, target: Optional[np.ndarray]) -> float:
    """
    Compute Mean Squared Error (MSE) between two images.

    MSE measures average squared difference between corresponding pixels.
    Lower MSE means images are more similar.

    Parameters
    ----------
    reference : np.ndarray
        Reference image (e.g., original clean image).
    target : np.ndarray, optional
        Target image to compare (e.g., noisy or filtered).

    Returns
    -------
    float
        MSE value. Returns NaN if target is None or shapes mismatch.
    """
    if target is None or reference.shape != target.shape:
        return float("nan")
    diff = reference.astype(np.float32) - target.astype(np.float32)
    mse = np.mean(diff ** 2)
    return float(mse)


def calculate_psnr(reference: np.ndarray, target: Optional[np.ndarray]) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR is defined as:
        PSNR = 10 * log10( MAX^2 / MSE )
    where MAX is the maximum possible pixel value (255 for uint8).

    Higher PSNR (e.g. > 30 dB) usually indicates better quality.

    Parameters
    ----------
    reference : np.ndarray
        Reference image.
    target : np.ndarray, optional
        Target image to compare.

    Returns
    -------
    float
        PSNR in decibels (dB). Returns NaN if MSE is zero or invalid.
    """
    mse = calculate_mse(reference, target)
    if not np.isfinite(mse):
        return float("nan")
    if mse == 0:
        return float("inf")
    if mse < 0:
        return float("nan")
    max_val = 255.0
    psnr = 10.0 * np.log10((max_val ** 2) / mse)
    return float(psnr)

