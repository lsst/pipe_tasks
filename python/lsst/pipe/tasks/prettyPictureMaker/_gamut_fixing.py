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

__all__ = ("fixOutOfGamutColors",)

import numpy as np
from numpy.typing import NDArray
from typing import Literal
import logging

from skimage.restoration import inpaint_biharmonic

from lsst.rubinoxide import rgb
from lsst.cpputils import fixGamutOK

from .types import LABImage, RGBImage


def fixOutOfGamutColors(
    Lab: LABImage,
    xyz_whitepoint: tuple[float, float],
    gamutMethod: Literal["mapping", "inpaint"] = "inpaint",
) -> RGBImage:
    """Remap colors that fall outside an RGB color gamut back into it.

    Parameters
    ----------
    Lab : `LABImage`
        Input image array in the Lab colorspace with shape (height, width, 3).
    xyz_whitepoint : `tuple` of `float`, `float`
        Sets the white point of the xyz colorspace in xy coordinates.
    gamutMethod : `str`, optional
        This determines what algorithm will be used to map out of gamut
        colors. Must be one of ``mapping`` or ``inpaint``.

    Returns
    -------
    result : `RGBImage`
        Image with out-of-gamut colors remapped in RGB colorspace.

    Raises
    ------
    ValueError
        Raised if gamutMethod is not one of the supported options (``mapping`` or ``inpaint``).
    """
    rgb_prime = rgb.Oklab_to_RGB(np.ascontiguousarray(Lab), xyz_whitepoint)

    # Determine if there are any out of bounds pixels
    outOfBounds = np.bitwise_or(
        np.bitwise_or(rgb_prime[:, :, 0] > 1, rgb_prime[:, :, 1] > 1), rgb_prime[:, :, 2] > 1
    )

    # If all pixels are in bounds, return immediately.
    if not np.any(outOfBounds):
        logging.info("There are no out of gamut pixels.")
        return rgb_prime

    logging.info("There are out of gamut pixels, remapping colors")
    match gamutMethod:
        case "inpaint":
            results = inpaint_biharmonic(rgb_prime, outOfBounds, channel_axis=-1)
        case "mapping":
            results = fixGamutOK(Lab[outOfBounds])  # type: ignore
            Lab[outOfBounds] = results
            results = rgb.Oklab_to_RGB(np.ascontiguousarray(Lab), xyz_whitepoint)
        case _:
            raise ValueError(f"gamut correction {gamutMethod} is not supported")

    logging.debug(f"The total number of remapped pixels is: {np.sum(outOfBounds)}")
    return results
