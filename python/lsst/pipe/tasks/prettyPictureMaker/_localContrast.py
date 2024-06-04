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


@njit(fastmath=True, parallel=True)
def r_old(img: NDArray, out: NDArray, g: float, sigma: float, beta: float, alpha: float) -> NDArray:
    diff = img - g
    # smallMask = np.abs(diff) < sigma
    # smallMask = smallMask.astype(np.bool_)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if np.abs(diff[i, j]) <= sigma:
                out[i, j] = g + np.sign(diff[i, j]) * sigma * (np.abs(diff[i, j]) / sigma) ** alpha
                # out[i, j] = img[i,j] + (np.abs(diff[i, j]) / sigma)*np.exp(-1*(np.abs(diff[i, j]) / sigma)**2/2*alpha**2)/alpha**2
            else:
                out[i, j] = beta * img[i, j]

    # out[smallMask] = g + np.sign(diff[smallMask])*sigma*(diff[smallMask]/sigma)**alpha
    # no need to do this if alpha is one because it is a no op
    # if beta != 1:
    #    out[~smallMask] = g + np.sign(diff[~smallMask])*(diff[~smallMask])
    return out


@njit(fastmath=True, parallel=True, error_model="numpy", nogil=True)
def r(
    img: NDArray, out: NDArray, g: float, sigma: float, shadows: float, highlights: float, clarity: float
) -> NDArray:
    h_s = (highlights, shadows)
    for i in prange(out.shape[0]):
        imgI = img[i]
        outI = out[i]
        for j in prange(out.shape[1]):
            c = imgI[j] - g
            s = np.sign(c)
            t = s * c / (2.0 * sigma)
            t = max(0, min(t, 1))
            t2 = t * t
            mt = 1.0 - t
            index = np.uint8(np.bool_(1 + s))
            val = g + s * sigma * 2 * mt * t + t2 * (s * sigma + s * sigma * h_s[index])
            val = val + clarity * c * np.exp(-(c * c) / (2.0 * sigma * sigma / 3.0))
            outI[j] = val

    return out


def makeGaussianPyramid(
    img: NDArray, padY: list[int], padX: list[int], out: List[NDArray] | None
) -> Sequence[NDArray]:
    if out is None:
        pyramid = List()
    else:
        pyramid = out
    if padY[0] or padX[0]:
        paddedImage = cv2.copyMakeBorder(
            img, *(0, padY[0]), *(0, padX[0]), cv2.BORDER_REPLICATE, None if out is None else pyramid[0], None
        ).astype(img.dtype)
    else:
        paddedImage = img
    if out is None:
        pyramid.append(paddedImage)
    else:
        # This might not be sound all the time, copy might be needed!
        pyramid[0] = paddedImage
    for i in range(1, len(padY)):
        if padY[i] or padX[i]:
            paddedImage = cv2.copyMakeBorder(
                paddedImage, *(0, padY[i]), *(0, padX[i]), cv2.BORDER_REPLICATE, None, None
            ).astype(img.dtype)
        paddedImage = cv2.pyrDown(paddedImage, None if out is None else pyramid[i])
        if out is None:
            pyramid.append(paddedImage)
    return pyramid


def makeLapPyramid(
    img: NDArray, padY: list[int], padX: list[int], gaussOut, lapOut, upscratch=None
) -> Sequence[NDArray]:
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


def padImage(image: NDArray) -> tuple[NDArray, list[tuple[int, int]]]:
    padding = []
    for size in image.shape:
        next2 = int(2 ** np.ceil(np.log2(size)))
        pad = next2 - size
        padHalf1 = pad // 2
        padHalf2 = pad - padHalf1
        padding.append((padHalf1, padHalf2))

    imagePadded = cv2.copyMakeBorder(image, *(padding[0]), *(padding[1]), cv2.BORDER_REPLICATE, None, None)
    return imagePadded, padding


@njit(fastmath=True, parallel=True, error_model="numpy", nogil=True)
def _calculateOutput(
    out: List[NDArray],
    pyramid: List[NDArray],
    gamma: NDArray,
    pyramidVectorsBottom: List[NDArray],
    pyramidVectorsTop: List[NDArray],
):
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
    one in order to make the level divisible by two.

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
        The scale over which edges are to be considered real and not noise.
    highlights : `float`
        A parameter that controls how highlights will be enhansed or reduced,
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
        the smoother the image will be post contrast enhancement, though above
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
