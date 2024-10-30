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

from __future__ import annotations

__all__ = ("localContrast",)

import numpy as np
from numpy.typing import NDArray
import cv2
from numba import njit, prange
from numba.typed.typedlist import List
from collections.abc import Sequence
from itertools import cycle


@njit(fastmath=True, parallel=True, error_model="numpy", nogil=True)
def r(
    img: NDArray, out: NDArray, g: float, sigma: float, shadows: float, highlights: float, clarity: float
) -> NDArray:
    """
    Apply a post-processing effect to an image using the specified parameters.

    Parameters:
        img : `NDArray`
            The input image array of shape (n_images, height, width).
        out : `NDArray`
            The output image array where the result will be stored. Should have the same shape as `img`.
        g : `float`
            A parameter for gamma correction.
        sigma : `float`
            Parameter that defines the scale at which a change should be considered an edge.
        shadows : `float`
            Shadow adjustment factor.
        highlights `float`
            Highlight adjustment factor. Negative values INCREASE highlights.
        clarity : `float`
            Clarity adjustment factor.

    Returns:
        result : `NDArray`
            The processed image array with the same shape as `out`.
    """

    h_s = (highlights, shadows)

    # Iterate over each pixel in the image
    for i in prange(out.shape[0]):
        # Get the current image slice
        imgI = img[i]
        # Get the corresponding output slice
        outI = out[i]

        # Iterate over each pixel in the image
        for j in prange(out.shape[1]):
            # Calculate the contrast adjusted by gamma correction
            c = imgI[j] - g
            # Determine the sign of the contrast adjustment
            s = np.sign(c)

            # Compute the transformation term t based on the signed contrast
            t = s * c / (2.0 * sigma)
            # Clamp t to be within [0, 1]
            t = max(0, min(t, 1))

            t2 = t * t
            # Complement of t
            mt = 1.0 - t

            # Determine the index based on the sign of c (either 0 or 1)
            index = np.uint8(np.bool_(1 + s))

            # Compute the final pixel value using the transformation and
            # additional terms for shadows/highlights and clarity
            val = g + s * sigma * 2 * mt * t + t2 * (s * sigma + s * sigma * h_s[index])
            val = val + clarity * c * np.exp(-(c * c) / (2.0 * sigma * sigma / 3.0))

            # Assign the computed value to the output image
            outI[j] = val

    return out


def makeGaussianPyramid(
    img: NDArray, padY: list[int], padX: list[int], out: List[NDArray] | None
) -> Sequence[NDArray]:
    """
    Create a Gaussian Pyramid from an input image.

    Parameters:
        img : `NDArray`
            The input image, which will be processed to create the pyramid.
        padY : `list` of `int`
            List containing padding sizes along the Y-axis for each level of the pyramid.
        padX : `list` of `int`
            List containing padding sizes along the X-axis for each level of the pyramid.
        out  `numba.typed.typedlist.List` of `NDarray` or `None`
            Optional list to store the output images of the pyramid levels.
            If None, a new list is created.

    Returns:
        pyramid : `Sequence` of `NDArray`
            A sequence of images representing the Gaussian Pyramid.

    Notes:
        - The function creates a padded version of the input image and then
          reduces its size using `cv2.pyrDown` to generate each level of the
          pyramid.
        - If 'out' is provided, it will be used to store the pyramid levels;
          otherwise, a new list is dynamically created.
        - Padding is applied only if specified by non-zero values in `padY` and
         `padX`.
    """
    # Initialize the output pyramid list if not provided
    if out is None:
        pyramid = List()
    else:
        pyramid = out

    # Apply padding only if needed, ensuring the type matches the input image
    if padY[0] or padX[0]:
        paddedImage = cv2.copyMakeBorder(
            img, *(0, padY[0]), *(0, padX[0]), cv2.BORDER_REPLICATE, None if out is None else pyramid[0], None
        ).astype(img.dtype)
    else:
        paddedImage = img

    # Store the first level of the pyramid (padded image)
    if out is None:
        pyramid.append(paddedImage)
    else:
        # This might not be sound all the time, copy might be needed!
        # Update the first level in the provided list
        pyramid[0] = paddedImage

    # Generate each subsequent level of the Gaussian Pyramid
    for i in range(1, len(padY)):
        if padY[i] or padX[i]:
            paddedImage = cv2.copyMakeBorder(
                paddedImage, *(0, padY[i]), *(0, padX[i]), cv2.BORDER_REPLICATE, None, None
            ).astype(img.dtype)
        # Downsample the image
        paddedImage = cv2.pyrDown(paddedImage, None if out is None else pyramid[i])

        # Append to the list if not provided externally
        if out is None:
            pyramid.append(paddedImage)
    return pyramid


def makeLapPyramid(
    img: NDArray,
    padY: list[int],
    padX: list[int],
    gaussOut: List[NDArray] | None,
    lapOut: List[NDArray] | None,
    upscratch: List[NDArray] | None = None,
) -> Sequence[NDArray]:
    """
    Create a Laplacian pyramid from the input image.

    This function constructs a Laplacian pyramid from the input image. It first
    generates a Gaussian pyramid and then, for each level (except the last),
    subtracts the upsampled version of the next lower level from the current
    level to obtain the Laplacian levels. If `lapOut` is None, it creates a
    new list to store the Laplacian pyramid; otherwise, it uses the provided
    `lapOut`.

    Parameters
    ----------
    img : `NDArray`
        The input image as a numpy array.
    padY : `list` of `int`
        List of padding sizes for rows (vertical padding).
    padX : `list` of `int`
        List of padding sizes for columns (horizontal padding).
    gaussOut : `numba.typed.typedlist.List` of `NDArray` or None
        Preallocated storage for the output of the Gaussian pyramid function.
        If `None` new storage is allocated.
    lapOut : `numba.typed.typedlist.List` of `NDArray` or None
        Preallocated for the output Laplacian pyramid. If None, a new
        `numba.typed.typedlist.List` is created.
    upscratch : `numba.typed.typedlist.List` of `NDarray`, optional
        List to store intermediate results of pyramids (default is None).

    Returns
    -------
    results : `Sequence` of `NDArray`
        The Laplacian pyramid as a sequence of numpy arrays.

    """
    pyramid = makeGaussianPyramid(img, padY, padX, gaussOut)
    if lapOut is None:
        lapPyramid = List()
    else:
        lapPyramid = lapOut
    for i in range(len(pyramid) - 1):
        upsampled = cv2.pyrUp(pyramid[i + 1], None if upscratch is None else upscratch[i + 1])
        if padY[i + 1] or padX[i + 1]:
            upsampled = upsampled[
                : upsampled.shape[0] - 2 * padY[i + 1], : upsampled.shape[1] - 2 * padX[i + 1]
            ]
        if lapOut is None:
            lapPyramid.append(pyramid[i] - upsampled)
        else:
            cv2.subtract(pyramid[i], upsampled, dst=lapPyramid[i])
    if lapOut is None:
        lapPyramid.append(pyramid[-1])
    else:
        lapPyramid[-1][:, :] = pyramid[-1]
    return lapPyramid


@njit(fastmath=True, parallel=True, error_model="numpy", nogil=True)
def _calculateOutput(
    out: List[NDArray],
    pyramid: List[NDArray],
    gamma: NDArray,
    pyramidVectorsBottom: List[NDArray],
    pyramidVectorsTop: List[NDArray],
):
    """
    Computes the output by interpolating between basis vectors at each pixel in
    a Gaussian pyramid.

    The function iterates over each pixel in the Gaussian pyramids
    and interpolates between the corresponding basis vectors  from
    `pyramidVectorsBottom` and `pyramidVectorsTop`. If a pixel value is outside
    the range defined by gamma, it skips interpolation.

    Parameters:
    -----------
    out : `numba.typed.typedlist.List` of `np.ndarray`
        A list of numpy arrays representing the output image pyramids.
    pyramid : `numba.typed.typedlist.List` of `np.ndarray`
        A list of numpy arrays representing the Gaussian pyramids.
    gamma : `np.ndarray`
        A numpy array containing the range for pixel values to be considered in
        the interpolation.
    pyramidVectorsBottom : `numba.typed.typedlist.List` of `np.ndarray`
        A list of numpy arrays representing the basis vectors at the bottom
        level of each pyramid layer.
    pyramidVectorsTop : `numba.typed.typedlist.List` of `np.ndarray`
        A list of numpy arrays representing the basis vectors at the top level
        of each pyramid layer.

    """
    # loop over each pixel in the gaussian pyramid
    # gammaDiff = gamma[1] - gamma[0]
    for level in prange(0, len(pyramid) - 1):
        yshape = pyramid[level].shape[0]
        xshape = pyramid[level].shape[1]
        plevel = pyramid[level]
        outlevel = out[level]
        basisBottom = pyramidVectorsBottom[level]
        basisTop = pyramidVectorsTop[level]
        for y in prange(yshape):
            plevelY = plevel[y]
            outLevelY = outlevel[y]
            basisBottomY = basisBottom[y]
            basisTopY = basisTop[y]
            for x in prange(xshape):
                val = plevelY[x]
                if not (val >= gamma[0] and val <= gamma[1]):
                    continue
                a = (plevelY[x] - gamma[0]) / (gamma[1] - gamma[0])
                outLevelY[x] = (1 - a) * basisBottomY[x] + a * basisTopY[x]


def levelPadder(numb: int, levels: int) -> list[int]:
    """Determine if each level of transform will need to be padded by
    one to make the level divisible by two.

    Parameters
    ----------
    numb : int
        The size of the input dimension
    levels : int
        The number of times the dimensions will be reduced by a factor of two

    Returns
    -------
    padds : list of int
        A list where the entries are either zero or one depending on if the
        size will need padded to be a power of two.

    """
    pads = []
    if numb % 2 != 0:
        pads.append(1)
        numb += 1
    else:
        pads.append(0)
    for _ in range(levels):
        numb /= 2
        if numb % 2 != 0:
            pads.append(1)
            numb += 1
        else:
            pads.append(0)
    return pads


def localContrast(
    image: NDArray,
    sigma: float,
    highlights: float = -0.9,
    shadows: float = 0.4,
    clarity: float = 0.15,
    maxLevel: int | None = None,
    numGamma: int = 20,
) -> NDArray:
    """Enhance the local contrast of an input image.

    Parameters
    ----------
    image : `NDArray`
        Two dimensional numpy array representing the image to have contrast
        increased.
    sigma : `float`
        The scale over which edges are considered real and not noise.
    highlights : `float`
        A parameter that controls how highlights are enhansed or reduced,
        contrary to intuition, negative values increase highlights.
    shadows : `float`
        A parameter that controls how shadows are deepened.
    clarity : `float`
        A parameter that relates to the contrast between highlights and
        shadow.
    maxLevel : `int` or `None`
        The maximum number of image pyramid levels to enhanse the contrast over.
        Each level has a spatial scale of roughly 2^(level) pixles.
    numGamma : `int`
        This is an optimization parameter. This algorithm divides up contrast
        space into a certain numbers over which the expensive computation
        is done. Contrast values in the image which fall between two of these
        values are interpolated to get the outcome. The higher the numGamma,
        the smoother the image is post contrast enhancement, though above
        some number there is no decerable difference.

    Returns
    -------
    image : `NDArray`
        Two dimensional numpy array of the input image with increased local
        contrast.

    Raises
    ------
    ValueError
        Raised if the max level to enhance to is greater than the image
        supports.

    Notes
    -----
    This function, and it's supporting functions, spiritually implement the
    algorithm outlined at
    https://people.csail.mit.edu/sparis/publi/2011/siggraph/
    titled "Local Laplacian Filters: Edge-aware Image Processing with Laplacian
    Pyramid". This is not a 1:1 implementation, it's optimized for the
    python language and runtime performance. Most notably it transforms only
    certain levels and linearly interpolates to find other values. This
    implementation is inspired by the ony done in the darktable image editor:
    https://www.darktable.org/2017/11/local-laplacian-pyramids/. None of the
    code is in common, nor is the implementation 1:1, but reading the original
    paper and the darktable implementation gives more info about this function.
    Specifically some variable names follow the paper/other implementation,
    and may be confusing when viewed without that context.

    """
    # ensure the supplied values are floats, and not ints
    highlights = float(highlights)
    shadows = float(shadows)
    clarity = float(clarity)

    # Determine the maximum level over which the image will be inhanced
    # and the amount of padding that will be needed to be added to the
    # image.
    maxImageLevel = int(np.min(np.log2(image.shape)))
    if maxLevel is None:
        maxLevel = maxImageLevel
    if maxImageLevel < maxLevel:
        raise ValueError(
            f"The supplied max level {maxLevel} is is greater than the max of the image: {maxImageLevel}"
        )
    support = 1 << (maxLevel - 1)
    padY_amounts = levelPadder(image.shape[0] + support, maxLevel)
    padX_amounts = levelPadder(image.shape[1] + support, maxLevel)
    imagePadded = cv2.copyMakeBorder(
        image, *(0, support), *(0, support), cv2.BORDER_REPLICATE, None, None
    ).astype(image.dtype)

    # build a list of intensities
    gamma = np.linspace(image.min(), image.max(), numGamma)

    # make gaussian pyramid
    pyramid = makeGaussianPyramid(imagePadded, padY_amounts, padX_amounts, None)

    finalPyramid = List()
    for sample in pyramid[:-1]:
        finalPyramid.append(np.zeros_like(sample))
    finalPyramid.append(pyramid[-1])

    # make a working array for gaussian pyramid in Lap
    # make two working arrays for laplace as the true value is interpolated
    # between the endpoints.
    # This prevents needing re-allocations which can be time consuming.
    tmpGauss = List()
    tmpLap1 = List()
    tmpLap2 = List()
    upscratch = List()
    for i, sample in enumerate(pyramid):
        tmpGauss.append(np.empty_like(sample))
        tmpLap1.append(np.empty_like(sample))
        tmpLap2.append(np.empty_like(sample))
        if i == 0:
            upscratch.append(np.empty((0, 0), dtype=image.dtype))
            continue
        upscratch.append(np.empty((sample.shape[0] * 2, sample.shape[1] * 2), dtype=image.dtype))
    # cycle between the endpoints, because there is no reason to recalculate both
    # endpoints as only one changes for each bin.
    cycler = iter(cycle((tmpLap1, tmpLap2)))
    # allocate temporary arrays to use for each bin
    outCycle = iter(cycle((np.copy(imagePadded), np.copy(imagePadded))))
    prevImg = r(
        imagePadded, next(outCycle), gamma[0], sigma, shadows=shadows, highlights=highlights, clarity=clarity
    )
    prevLapPyr = makeLapPyramid(
        prevImg, padY_amounts, padX_amounts, tmpGauss, next(cycler), upscratch=upscratch
    )

    for value in range(1, len(gamma) - 1):
        pyramidVectors = List()
        pyramidVectors.append(prevLapPyr)
        newImg = r(
            imagePadded,
            next(outCycle),
            gamma[value],
            sigma,
            shadows=shadows,
            highlights=highlights,
            clarity=clarity,
        )
        prevLapPyr = makeLapPyramid(
            newImg, padY_amounts, padX_amounts, tmpGauss, next(cycler), upscratch=upscratch
        )
        pyramidVectors.append(prevLapPyr)

        _calculateOutput(
            finalPyramid,
            pyramid,
            np.array((gamma[value - 1], gamma[value])),
            pyramidVectors[0],
            pyramidVectors[1],
        )
        del pyramidVectors

    # time to reconstruct
    output = finalPyramid[-1]
    for i in range(-2, -1 * len(finalPyramid) - 1, -1):
        upsampled = cv2.pyrUp(output)
        upsampled = upsampled[
            : upsampled.shape[0] - 2 * padY_amounts[i + 1], : upsampled.shape[1] - 2 * padX_amounts[i + 1]
        ]
        output = finalPyramid[i] + upsampled
    return output[:-support, :-support]
