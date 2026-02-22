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

__all__ = ("ExposureBracketer",)

import numpy as np
import cv2

from lsst.pipe.tasks.prettyPictureMaker.types import FloatImagePlane
from lsst.pex.config.configurableActions import ConfigurableAction
from lsst.pex.config import ListField

from .._localContrast import levelPadder, makeGaussianPyramid, makeLapPyramid


def _fuseExposureLum(images, sigma=0.1, maxLevel=3):
    weights = np.zeros((len(images), *images[0].shape[:2]))
    for i, image in enumerate(images):
        exposure = np.exp(-((image[:, :] - 0.7) ** 2) / (2 * sigma))
        # dont weight at all values greater than 1
        exposure[image > 1] *= 0.5

        weights[i, :, :] = exposure
    norm = np.sum(weights, axis=0)
    np.divide(weights, norm, out=weights)

    # loop over each image again to build pyramids
    g_pyr = []
    l_pyr = []
    maxImageLevel = int(np.min(np.log2(images[0].shape[:2])))
    if maxLevel is None:
        maxLevel = maxImageLevel
    if maxImageLevel < maxLevel:
        raise ValueError(
            f"The supplied max level {maxLevel} is is greater than the max of the image: {maxImageLevel}"
        )
    support = 1 << (maxLevel - 1)
    padY_amounts = levelPadder(image.shape[0] + support, maxLevel)
    padX_amounts = levelPadder(image.shape[1] + support, maxLevel)
    for image, weight in zip(images, weights):
        imagePadded = cv2.copyMakeBorder(
            image, *(0, support), *(0, support), cv2.BORDER_REFLECT, None, None
        ).astype(image.dtype)
        weightPadded = cv2.copyMakeBorder(
            weight, *(0, support), *(0, support), cv2.BORDER_REFLECT, None, None
        ).astype(image.dtype)

        g_pyr.append(list(makeGaussianPyramid(weightPadded, padY_amounts, padX_amounts, None)))
        l_pyr.append(list(makeLapPyramid(imagePadded, padY_amounts, padX_amounts, None, None)))

    # time to blend
    blended = []
    for level in range(len(padY_amounts)):
        accumulate = np.zeros_like(l_pyr[0][level])
        for img in range(len(g_pyr)):
            accumulate[:, :] += l_pyr[img][level][:, :] * g_pyr[img][level]
        blended.append(accumulate)

    # time to reconstruct
    output = blended[-1]
    for i in range(-2, -1 * len(blended) - 1, -1):
        upsampled = cv2.pyrUp(output)
        upsampled = upsampled[
            : upsampled.shape[0] - 2 * padY_amounts[i + 1], : upsampled.shape[1] - 2 * padX_amounts[i + 1]
        ]
        output = blended[i] + upsampled
    return output[:-support, :-support]


class ExposureBracketer(ConfigurableAction):
    exposureBrackets = ListField[float](
        doc=(
            "Exposure scaling factors used in creating multiple exposures with different scalings which will "
            "then be fused into a final image"
        ),
        optional=True,
        default=[1.25, 1, 0.75],
    )

    def __call__(self, intensities: FloatImagePlane) -> FloatImagePlane:
        if self.exposureBrackets is None:
            return intensities
        stack = []
        for bracket in self.exposureBrackets:
            intensities = intensities / bracket
            stack.append(intensities)

        if len(stack) == 1:
            intensities = stack[0]
        else:
            intensities = _fuseExposureLum(stack)

        return intensities
