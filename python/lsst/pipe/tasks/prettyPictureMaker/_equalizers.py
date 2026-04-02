# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import cv2
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter


from ._localContrast import levelPadder, makeLapPyramid
from .types import FloatImagePlane


def eigf_variance_analysis_no_mask(guide: FloatImagePlane, sigma: float) -> NDArray:
    """
    Computes average and variance of guide using Gaussian filtering.

    Parameters
    ----------
    guide : `FloatImagePlane`
        2D array representing the guide image.
    sigma : `float`
        Standard deviation for Gaussian kernel.

    Returns
    -------
    `NDArray`
        Array where each pixel has [average, variance].
    """
    # Compute average of guide
    mu_guide = gaussian_filter(guide, sigma=sigma)

    # Compute average of squared guide values
    guide_squared = guide**2
    mu_guide_squared = gaussian_filter(guide_squared, sigma=sigma)

    # Calculate variance as E[guide^2] - (E[guide])^2
    var_guide = mu_guide_squared - mu_guide**2

    # Combine into an output array with shape (height, width, 2)
    output = np.stack((mu_guide, var_guide), axis=2)

    return output


def eigf_blending_no_mask(image: FloatImagePlane, av: NDArray, feathering: float, filter_type: int) -> None:
    """
    Applies blending without a mask using averages and variances.

    Parameters
    ----------
    image : `FloatImagePlane`
        2D input image array. Modified in-place.
    av : `NDArray`
        Array with shape (height, width, 2) containing averages and variances.
    feathering : `float`
        Feathering parameter for blending.
    filter_type : `int`
        Blending type: 0 for linear, 1 for geometric mean.
    """
    # Reshape 'av' to match image dimensions
    av_reshaped = av.reshape(image.shape[0], image.shape[1], -1)

    avg_g = av_reshaped[..., 0]
    var_g = av_reshaped[..., 1]

    norm_g = np.maximum(avg_g * image, 1e-6)
    normalized_var_guide = var_g / norm_g

    a = normalized_var_guide / (normalized_var_guide + feathering)
    b = avg_g - a * avg_g

    # Apply blending
    if filter_type == 0:  # Linear blending
        image[:] = np.maximum(image * a + b, np.finfo(float).min)
    else:  # Geometric mean blending
        image[:] *= np.maximum(image * a + b, np.finfo(float).min)
        image[:] = np.sqrt(image[:])


def fast_eigf_surface_blur(
    image: FloatImagePlane, sigma: float, feathering: float, iterations: int = 1, filter_type: int = 1
) -> None:
    """
    Applies exposure-independent guided blur with down-scaling and up-sampling.

    Parameters
    ----------
    image : `FloatImagePlane`
        Input image array of shape (height, width). Modified in-place.
    sigma : `float`
        Standard deviation for Gaussian kernel.
    feathering : `float`
        Feathering parameter.
    iterations : `int`, optional
        Number of iterations to model diffusion. Default is 1.
    filter_type : `int`, optional
        Blending type: 0 for linear, 1 for geometric mean. Default is 1.
    """
    scaling = np.maximum(np.minimum(sigma, 4.0), 1.0)
    ds_sigma = np.maximum(sigma / scaling, 1.0)

    # Down-sampling dimensions

    for _ in range(iterations):
        av = eigf_variance_analysis_no_mask(image, ds_sigma)
        eigf_blending_no_mask(image, av.reshape(-1, 2), feathering, filter_type)


def tone_equalizer(
    image: FloatImagePlane,
    tone_factors: list[float],
    weight: float,
    sigma: float,
    feathering: float,
    iterations: int = 1,
    filter_type: int = 1,
) -> FloatImagePlane:
    """Enhance image brightness using exposure-dependent correction.

    This function adjusts image brightness by applying exposure-dependent
    corrections based on tone factors. It uses exposure centers spanning from
    0 to 1 (10 levels) and applies Gaussian-weighted adjustments. A copy of
    the input image is made before processing.

    Parameters
    ----------
    image : `FloatImagePlane`
        Input image array of shape (height, width).
    tone_factors : `list` of `float`
        List of 10 tone correction factors, one for each exposure level.
    weight : `float`
        Width of the Gaussian kernel for exposure weighting.
    sigma : `float`
        Standard deviation for Gaussian blur of luminance.
    feathering : `float`
        Feathering parameter for exposure-independent guided blur.
    iterations : `int`, optional
        Number of iterations for the blur process. Default is 1.
    filter_type : `int`, optional
        Blending type: 0 for linear, 1 for geometric mean. Default is 1.

    Returns
    -------
    `FloatImagePlane`
        Image with brightness adjusted based on tone factors.
    """
    luminance = np.copy(image)
    fast_eigf_surface_blur(luminance, sigma, feathering, iterations, filter_type)
    exposure = luminance
    corrections = np.zeros_like(luminance)
    EXPOSURE_CENTERS = np.linspace(0, 1, 10)
    for eq_val, factor in zip(EXPOSURE_CENTERS, tone_factors):
        corrections += np.exp(-1 * (exposure - eq_val) ** 2 / (2 * weight**2)) * factor
    return image + corrections


def contrast_equalizer(image: FloatImagePlane, contrast_factors: list[float]) -> FloatImagePlane:
    """Enhance image contrast using Laplacian pyramid adjustment.

    This function performs contrast equalization by modifying the Laplacian
    pyramid coefficients of the input image. Each level of the pyramid
    corresponds to a different spatial scale, allowing for scale-dependent
    contrast adjustments. A padded copy of the input image is created for
    processing.

    Parameters
    ----------
    image : `FloatImagePlane`
        Input image array of shape (height, width).
    contrast_factors : `list` of `float`
        List of factors to multiply each pyramid level. Values > 1 increase
        contrast, values < 1 decrease contrast. The list should specify
        factors for the largest scales first; unspecified levels use a factor
        of 1.0.

    Returns
    -------
    `FloatImagePlane`
        Image with contrast adjusted at multiple spatial scales.
    """
    maxLevel = int(np.min(np.log2(image.shape)))
    support = 1 << (maxLevel - 1)
    padY_amounts = levelPadder(image.shape[0] + support, maxLevel)
    padX_amounts = levelPadder(image.shape[1] + support, maxLevel)
    imagePadded = cv2.copyMakeBorder(
        image, *(0, support), *(0, support), cv2.BORDER_REPLICATE, None, None
    ).astype(image.dtype)
    lap = makeLapPyramid(imagePadded, padY_amounts, padX_amounts, None, None)
    for i, factor in enumerate(contrast_factors):
        i = i + 2
        if i > len(lap):
            break
        lap[-1 * i] *= factor
    output = lap[-1]
    for i in range(-2, -1 * len(lap) - 1, -1):
        upsampled = cv2.pyrUp(output)
        upsampled = upsampled[
            : upsampled.shape[0] - 2 * padY_amounts[i + 1], : upsampled.shape[1] - 2 * padX_amounts[i + 1]
        ]
        output = lap[i] + upsampled
    return output[: image.shape[0], : image.shape[1]]
