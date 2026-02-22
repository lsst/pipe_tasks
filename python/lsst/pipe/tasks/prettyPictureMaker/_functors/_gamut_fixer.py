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

__all__ = ("GamutFixer",)

import skimage
import numpy as np
import logging


from lsst.cpputils import fixGamutOK
from lsst.pipe.tasks.prettyPictureMaker.types import LABImage, RGBImage
from lsst.pex.config.configurableActions import ConfigurableAction
from lsst.pex.config import ChoiceField
from lsst.rubinoxide import rgb


class GamutFixer(ConfigurableAction):
    gamutMethod = ChoiceField[str](
        doc="If doRemapGamut is True this determines the method",
        default="inpaint",
        allowed={
            "mapping": "Use a mapping function",
            "inpaint": "Use surrounding pixels to determine likely value",
            "none": "Don't fix out of gamut colors",
        },
    )

    def __call__(self, Lab: LABImage, xyz_whitepoint: tuple[float, float]) -> RGBImage:
        """Remap colors that fall outside an RGB color gamut back into it.

        This function modifies the input Lab array in-place for memory reasons.

        Parameters
        ----------
        Lab : `NDArray`
            A NxMX3 array that contains data in the Lab colorspace.
        xyz_whitepoint : `tuple` of `float`, `float`
            Sets the white point of the xyz colorspace in xy coordinates.
        """
        rgb_prime = rgb.Oklab_to_RGB(np.ascontiguousarray(Lab), xyz_whitepoint)

        if self.gamut_method == "none":
            return rgb_prime

        # Determine if there are any out of bounds pixels
        outOfBounds = np.bitwise_or(
            np.bitwise_or(rgb_prime[:, :, 0] > 1, rgb_prime[:, :, 1] > 1), rgb_prime[:, :, 2] > 1
        )

        # If all pixels are in bounds, return immediately.
        if not np.any(outOfBounds):
            logging.info("There are no out of gamut pixels.")
            return rgb_prime

        logging.info("There are out of gamut pixels, remapping colors")
        match self.gamut_method:
            case "inpaint":
                results = skimage.restoration.inpaint_biharmonic(rgb_prime, outOfBounds, channel_axis=-1)
            case "mapping":
                results = fixGamutOK(Lab[outOfBounds])
                Lab[outOfBounds] = results
                results = rgb.Oklab_to_RGB(np.ascontiguousarray(Lab), xyz_whitepoint)
            case _:
                raise ValueError(f"gamut correction {self.gamut_method} is not supported")

        logging.debug(f"The total number of remapped pixels is: {np.sum(outOfBounds)}")
        return results
