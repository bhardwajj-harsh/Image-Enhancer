"""
Noise simulation utilities.

These functions add synthetic noise to images so that
we can test how well different filters remove them.
"""

import numpy as np


def _to_float_image(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to float32 in [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Input image (uint8).

    Returns
    -------
    np.ndarray
        Float32 image with values in [0, 1].
    """
    return image.astype(np.float32) / 255.0


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    """
    Convert a float32 image in [0, 1] back to uint8.

    Parameters
    ----------
    image : np.ndarray
        Float32 image.

    Returns
    -------
    np.ndarray
        uint8 image in [0, 255].
    """
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, std: float = 25.0) -> np.ndarray:
    """
    Add additive Gaussian noise to an image.

    Gaussian noise follows a normal distribution with
    mean (μ) and standard deviation (σ).

    Parameters
    ----------
    image : np.ndarray
        Input image (uint8).
    mean : float
        Mean of the Gaussian noise (in pixel units).
    std : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    np.ndarray
        Noisy image (uint8).
    """
    float_img = image.astype(np.float32)
    noise = np.random.normal(loc=mean, scale=std, size=float_img.shape).astype(np.float32)
    noisy = float_img + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


def add_salt_pepper_noise(
    image: np.ndarray,
    amount: float = 0.02,
    salt_vs_pepper: float = 0.5,
) -> np.ndarray:
    """
    Add salt & pepper impulse noise.

    - "Salt" pixels are set to white (255).
    - "Pepper" pixels are set to black (0).
    This noise is random and affects some pixels drastically.

    Parameters
    ----------
    image : np.ndarray
        Input image (uint8).
    amount : float
        Fraction of pixels to corrupt with noise.
    salt_vs_pepper : float
        Ratio of salt noise to pepper noise.

    Returns
    -------
    np.ndarray
        Noisy image (uint8).
    """
    noisy = image.copy()
    h, w = noisy.shape[:2]
    num_pixels = h * w

    num_salt = int(amount * num_pixels * salt_vs_pepper)
    num_pepper = int(amount * num_pixels * (1.0 - salt_vs_pepper))

    coords_salt = (
        np.random.randint(0, h, num_salt),
        np.random.randint(0, w, num_salt),
    )
    coords_pepper = (
        np.random.randint(0, h, num_pepper),
        np.random.randint(0, w, num_pepper),
    )

    if noisy.ndim == 2:
        noisy[coords_salt] = 255
        noisy[coords_pepper] = 0
    else:
        noisy[coords_salt[0], coords_salt[1], :] = 255
        noisy[coords_pepper[0], coords_pepper[1], :] = 0

    return noisy


def add_speckle_noise(image: np.ndarray, var: float = 0.05) -> np.ndarray:
    """
    Add multiplicative speckle noise to the image.

    Speckle noise is often modeled as:
        noisy = image + image * noise

    Parameters
    ----------
    image : np.ndarray
        Input image (uint8).
    var : float
        Variance factor controlling noise strength.

    Returns
    -------
    np.ndarray
        Noisy image (uint8).
    """
    float_img = _to_float_image(image)
    noise = np.random.randn(*float_img.shape).astype(np.float32) * np.sqrt(var)
    noisy = float_img + float_img * noise
    return _to_uint8_image(noisy)

