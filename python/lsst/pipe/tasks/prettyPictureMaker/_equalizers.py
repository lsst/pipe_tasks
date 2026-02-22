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
from scipy.ndimage import gaussian_filter


from ._localContrast import levelPadder, makeLapPyramid


def eigf_variance_analysis_no_mask(guide, sigma):
    """
    Computes average and variance of guide using Gaussian filtering.

    Parameters:
        guide (numpy.ndarray): 2D array representing the guide image.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        numpy.ndarray: Array where each pixel has [average, variance].
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


def eigf_blending_no_mask(image, av, feathering, filter_type):
    """
    Applies blending without a mask using averages and variances.

    Parameters:
        image (numpy.ndarray): 2D input image array.
        av (numpy.ndarray): Array with shape (height*width, 2) containing averages and variances.
        feathering (float): Feathering parameter for blending.
        filter_type (int): Blending type: 0 for linear, 1 for geometric mean.
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


def fast_eigf_surface_blur(image, sigma, feathering, iterations=1, filter_type=1):
    """
    Applies exposure-independent guided blur with down-scaling and up-sampling.

    Parameters:
        image (numpy.ndarray): Input image array of shape (height, width).
        sigma (float): Standard deviation for Gaussian kernel.
        feathering (float): Feathering parameter.
        iterations (int): Number of iterations to model diffusion.
        filter_type (int): Blending type: 0 for linear, 1 for geometric mean.
    """
    scaling = np.maximum(np.minimum(sigma, 4.0), 1.0)
    ds_sigma = np.maximum(sigma / scaling, 1.0)

    # Down-sampling dimensions

    for _ in range(iterations):
        av = eigf_variance_analysis_no_mask(image, ds_sigma)
        eigf_blending_no_mask(image, av.reshape(-1, 2), feathering, filter_type)


def tone_equalizer(image, tone_factors, weight, sigma, feathering, iterations=1, filter_type=1):
    # need to blur the input luminance image
    luminance = np.copy(image)
    fast_eigf_surface_blur(luminance, sigma, feathering, iterations, filter_type)
    exposure = luminance
    corrections = np.zeros_like(luminance)
    EXPOSURE_CENTERS = np.linspace(0, 1, 10)
    for eq_val, factor in zip(EXPOSURE_CENTERS, tone_factors):
        corrections += np.exp(-1 * (exposure - eq_val) ** 2 / (2 * weight**2)) * factor
    return image + corrections


def contrast_equalizer(image, contrast_factors):
    maxLevel = int(np.min(np.log2(image.shape)))
    support = 1 << (maxLevel - 1)
    padY_amounts = levelPadder(image.shape[0] + support, maxLevel)
    padX_amounts = levelPadder(image.shape[1] + support, maxLevel)
    imagePadded = cv2.copyMakeBorder(
        image, *(0, support), *(0, support), cv2.BORDER_REPLICATE, None, None
    ).astype(image.dtype)
    lap = makeLapPyramid(imagePadded, padY_amounts, padX_amounts, None, None)
    for i, factor in enumerate(contrast_factors):
        # negative indexing so +1, +1 to skip the highest pyramid
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
