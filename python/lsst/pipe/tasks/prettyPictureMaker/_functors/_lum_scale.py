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

__all__ = ("LumCompressor",)

import skimage
import numpy as np
import logging


from lsst.pipe.tasks.prettyPictureMaker.types import FloatImagePlane
from lsst.pex.config.configurableActions import ConfigurableAction
from lsst.pex.config import Field, ListField
from lsst.rubinoxide import rgb

from .._equalizers import contrast_equalizer, tone_equalizer


class LumCompressor(ConfigurableAction):
    """Configurations to control how luminance is mapped in the rgb code"""

    stretch = Field[float](doc="The stretch of the luminance in asinh", default=400)
    max = Field[float](doc="The maximum allowed luminance on a 0 to 1 scale", default=0.85)
    floor = Field[float](doc="A value in nJy that is used to map luminances to a very dark gray", default=0.0)
    Q = Field[float](
        doc="softening parameter",
        default=0.7,
        deprecated="This field is no longer used and will be removed after v31.",
    )
    highlight = Field[float](
        doc="The value of highlights in scaling factor applied to post asinh streaching", default=1.0
    )
    shadow = Field[float](
        doc="The value of shadows in scaling factor applied to post asinh streaching", default=0.0
    )
    midtone = Field[float](
        doc="The value of midtone in scaling factor applied to post asinh streaching", default=0.5
    )
    equalizerLevels = ListField[float](
        doc=(
            "A list of factors to modify the constrast in a scale dependent way. "
            "One coefficient for each spatial scale, starting from the largest. "
            "Values large than 1 increase contrast, while less than 1 decreases. "
            "This adjustment is multaplicative. "
            "Only scales upto and including the largest to be modified need specified, "
            "IE [1.1,0.9] modifieds the first two [1.1,1,0.9] modifies the first three."
        ),
        optional=True,
    )
    toneAdjustment = ListField[float](
        doc=(
            "A list of length 10 that adjusts the brightness of the image ranging "
            "from dark regions to light. These 10 values represent control points along "
            "the lumanance interval 0-1, but the actual adjustments made are continuous "
            "and are calculated from these control points."
        ),
        length=10,
        optional=True,
    )
    toneWidth = Field[float](
        doc=(
            "This parameters controls how each tone control point affect the adjustment "
            "of the values in between. Increase the value to have a more continuous "
            "change between control points, decrease to make the control shaprer. Value "
            "must be greater than zero."
        ),
        default=0.07,
    )
    doDenoise = Field[bool](doc="Denoise the luminance image")

    def __call__(self, intensities: FloatImagePlane) -> FloatImagePlane:
        if self.doDenoise:
            intensities = skimage.restoration.denoise_wavelet(intensities)

        # Scale the values with linear manipulation for contrast
        intensities = abs(intensities)
        # intensities = rgb_diffusion.diffuse_gray_image(intensities)
        nj_to_lum = rgb.RGB_to_Oklab(
            np.array([[[self.floor, self.floor, self.floor]]], dtype=float), (0.28, 0.28)
        )[0, 0, 0]
        top = np.arcsinh(self.stretch)
        bottom = (np.arcsinh(nj_to_lum * self.stretch) - 0.2 * top) / 0.8
        intensities = (np.arcsinh(intensities * self.stretch) - bottom) / (top - bottom)
        logging.debug("arcsinh max %d", intensities.max())
        intensities = np.clip(intensities, 0, 1)
        intensities = (intensities - self.shadow) / ((self.highlight) - self.shadow)
        logging.debug("post lin streatch max %d", intensities.max())
        intensities = ((self.midtone - 1) * intensities) / (
            ((2 * self.midtone - 1) * intensities) - self.midtone
        )
        logging.debug("midtone adjustment max %d", intensities.max())
        intensities = np.clip(intensities, 0.0, self.max)

        if self.equalizerLevels is not None:
            intensities = contrast_equalizer(intensities, self.equalizerLevels)
            logging.debug("equalizer max %d", intensities.max())

        if self.toneAdjustment is not None:
            intensities = np.clip(intensities, 0, self.max)
            intensities = tone_equalizer(intensities, self.toneAdjustment, self.toneWidth, 10, 5)

        intensities = np.clip(intensities, 0, 1)

        return intensities
