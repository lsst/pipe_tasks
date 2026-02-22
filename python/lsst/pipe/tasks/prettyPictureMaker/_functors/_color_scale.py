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

__all__ = ("ColorScaler",)

import numpy as np

from lsst.pipe.tasks.prettyPictureMaker.types import FloatImagePlane
from lsst.pex.config.configurableActions import ConfigurableAction
from lsst.pex.config import Field, ListField

from .._equalizers import contrast_equalizer


class ColorScaler(ConfigurableAction):
    saturation = Field[float](
        doc=(
            "The overall saturation factor with the scaled luminance between zero and one. "
            "A value of one is not recommended as it makes bright pixels very saturated"
        ),
        default=0.5,
    )
    maxChroma = Field[float](
        doc=(
            "The maximum chromaticity in the OKLCh color space, large "
            "values will cause bright pixels to fall outside the RGB gamut."
        ),
        default=0.4,
    )
    equalizer_levels = ListField[float](
        doc=(
            "A list of factors to modify the color constrast in a scale dependent way. "
            "One coefficient for each spatial scale, starting from the largest. "
            "Values large than 1 increase contrast, while less than 1 decreases"
            "Only scales upto and including the largest to be modified need specified"
            "IE [1.1,0.9] modifieds the first two [1.1,1,0.9] modifies the first three."
        ),
        optional=True,
    )

    def __call__(
        self, old_lum: FloatImagePlane, new_lum: FloatImagePlane, a: FloatImagePlane, b: FloatImagePlane
    ) -> tuple[FloatImagePlane, FloatImagePlane]:
        """
        Adjusts the color saturation while keeping the hue constant.

        This function adjusts the chromaticity (a, b) of colors to maintain a
        consistent saturation level, based on their original luminance. It uses
        the CIELAB color space representation and the `luminance` is the new target
        luminance for all colors.

        Parameters
        ----------
        old_lum : `FloatImagePlane`
            Luminance values of the original colors.
        new_lum : `FloatImagePlane`
            Target luminance values for the transformed colors.
        a : `FloatImagePlane`
            Chromaticity parameter 'a' corresponding to green-red axis in CIELAB.
        b : `FloatImagePlane`
            Chromaticity parameter 'b' corresponding to blue-yellow axis in CIELAB.

        Returns
        -------
        new_a : `FloatImagePlane`
            New a values representing the adjusted chromaticity.
        new_b : `FloatImagePlane`
            New b values representing the adjusted chromaticity.
        """
        # Calculate the square of the chroma, which is the distance from origin in
        # the a-b plane.
        chroma1_2 = a**2 + b**2
        chroma1 = np.sqrt(chroma1_2)

        # Calculate the hue angle, taking the absolute value to ensure non-negative
        # angle representation.
        chromaMask = chroma1 == 0
        chroma1[chromaMask] = 1
        # I think these ratios are right, but I see them opposite in a different
        # version. Keep an eye on this in testing.
        sinHue = b / chroma1
        cosHue = a / chroma1
        sinHue[chromaMask] = 0
        cosHue[chromaMask] = 0

        # Compute a divisor for saturation calculation, adding 1 to avoid division
        # by zero.
        div = chroma1_2 + old_lum**2
        div[div <= 0] = 1

        # Calculate the square of the new chroma based on desired saturation
        sat_original_2 = chroma1_2 / div
        chroma2_2 = self.saturation * sat_original_2 * new_lum**2 / (1 - sat_original_2)

        # Compute new 'a' values using the square root of adjusted chroma and
        # considering hue direction.
        chroma2 = np.sqrt(chroma2_2)
        if self.equalizer_levels is not None:
            chroma2 = contrast_equalizer(chroma2, self.equalizer_levels)
        # Cap the chroma to avoid excessive values that are visually unrealistic
        chroma2[chroma2 > self.maxChroma] = self.maxChroma

        new_a = chroma2 * cosHue

        # Compute new 'b' values using the root of the adjusted chroma and hue
        # direction.
        new_b = chroma2 * sinHue

        return new_a, new_b
