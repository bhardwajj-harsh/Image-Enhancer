"""
Classical image enhancement techniques.

These operations improve visual quality or contrast, not necessarily
recover the original scene perfectly.
"""

import cv2
import numpy as np


def _rgb_to_ycrcb(image_rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to YCrCb (uint8).

    We do enhancement mainly on the Y (luminance) channel to preserve colors.
    """
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)


def _ycrcb_to_rgb(image_ycrcb: np.ndarray) -> np.ndarray:
    """Convert a YCrCb image back to RGB (uint8)."""
    return cv2.cvtColor(image_ycrcb, cv2.COLOR_YCrCb2RGB)


def apply_hist_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply global histogram equalization.

    Histogram equalization redistributes intensities to use the full
    range [0, 255], improving overall contrast, especially in dark
    or low-contrast images.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image (uint8).

    Returns
    -------
    np.ndarray
        Contrast-enhanced RGB image.
    """
    ycrcb = _rgb_to_ycrcb(image)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge([y_eq, cr, cb])
    return _ycrcb_to_rgb(merged)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    CLAHE works on small tiles of the image and limits contrast
    to avoid over-amplifying noise. It is useful for non-uniform
    illumination.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image.
    clip_limit : float
        Threshold for contrast limiting.
    tile_grid_size : int
        Size of the grid for histogram equalization.

    Returns
    -------
    np.ndarray
        Enhanced RGB image.
    """
    ycrcb = _rgb_to_ycrcb(image)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    y_clahe = clahe.apply(y)
    merged = cv2.merge([y_clahe, cr, cb])
    return _ycrcb_to_rgb(merged)


def apply_contrast_stretching(
    image: np.ndarray,
    low_perc: float = 2.0,
    high_perc: float = 98.0,
) -> np.ndarray:
    """
    Apply contrast stretching using percentile-based normalization.

    Intensities below the low percentile are mapped to 0,
    above the high percentile to 255, and the rest are stretched linearly.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image.python3 -m streamlit run app.py
    low_perc : float
        Low percentile (e.g., 2).
    high_perc : float
        High percentile (e.g., 98).

    Returns
    -------
    np.ndarray
        Contrast-stretched image.
    """
    ycrcb = _rgb_to_ycrcb(image)
    y, cr, cb = cv2.split(ycrcb)

    y_f = y.astype(np.float32)
    p_low = float(np.percentile(y_f, low_perc))
    p_high = float(np.percentile(y_f, high_perc))
    if p_high - p_low < 1e-3:
        return image.copy()

    y_stretched = (y_f - p_low) * (255.0 / (p_high - p_low))
    y_stretched = np.clip(y_stretched, 0, 255).astype(np.uint8)

    merged = cv2.merge([y_stretched, cr, cb])
    return _ycrcb_to_rgb(merged)


def apply_sharpening(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """
    Apply simple sharpening using an unsharp masking style kernel.

    Sharpening enhances edges and fine details, which can make the
    image appear crisper.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image.
    amount : float
        Sharpening strength.

    Returns
    -------
    np.ndarray
        Sharpened image.
    """
    if amount <= 0:
        return image.copy()

    img_f = image.astype(np.float32)
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
    blurred_f = blurred.astype(np.float32)

    # Unsharp mask: sharpened = image + amount * (image - blurred)
    sharpened = img_f + amount * (img_f - blurred_f)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

