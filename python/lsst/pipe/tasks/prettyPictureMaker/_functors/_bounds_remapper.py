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

__all__ = ("BoundsRemapper",)

import numpy as np

from lsst.pipe.tasks.prettyPictureMaker.types import RGBImage
from lsst.pex.config.configurableActions import ConfigurableAction
from lsst.pex.config import Field


class BoundsRemapper(ConfigurableAction):
    """Remaps input images to a known range of values.

    Often input images are not mapped to any defined range of values
    (for instance if they are in count units). This controls how the units of
    and image are mapped to a zero to one range by determining an upper
    bound.
    """

    quant = Field[float](
        doc=(
            "The maximum values of each of the three channels will be multiplied by this factor to "
            "determine the maximum flux of the image, values larger than this quantity will be clipped."
        ),
        default=0.8,
    )
    absMax = Field[float](
        doc="Instead of determining the maximum value from the image, use this fixed value instead",
        default=220,
        optional=True,
    )

    def __call__(self, img: RGBImage) -> RGBImage:
        """Bound images to a range between zero and one.

        Some images supplied aren't properly bounded with a maximum value of 1.
        Either the images exceed the bounds of 1, or that no value seems to close,
        implying indeterminate maximum value. This function determines an
        appropriate maximum either by taking the value supplied in the absMax
        argument or by scaling the maximum across all channels with the
        supplied quant variable.

        Parameters
        ----------
        img : `NDArray` like
            Must have dimensions of y,x,3 where the channels are in RGB order

        Returns
        -------
        image : `NDArray`
            The result of the remapping process
        """
        if np.max(img) == 1:
            return img

        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        if self.absMax is not None:
            scale = self.absMax
        else:
            r_quant = np.quantile(r, 0.95)
            g_quant = np.quantile(g, 0.95)
            b_quant = np.quantile(b, 0.95)
            turnover = np.max((r_quant, g_quant, b_quant))
            scale = turnover * self.quant

        image = np.copy(img)
        image /= scale

        # Clip values that exceed the bound to ensure all values are within [0, absMax]
        return np.clip(image, 0, 1)
