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
import colour


from lsst.cpputils import fixGamutOK
from lsst.pipe.tasks.prettyPictureMaker.types import LABImage, RGBImage, WhitePoint
from lsst.pex.config.configurableActions import ConfigurableAction
from lsst.pex.config import ChoiceField, Field
from lsst.rubinoxide import rgb
from scipy.ndimage import label, find_objects, binary_dilation


# illuminant=input_whitepoint,
# target_illuminant=output_whitepoint,
class HDRRec2020Gammut(ConfigurableAction):
    def __call__(self, Lab: LABImage, input_whitepoint, output_whitepoint: WhitePoint) -> RGBImage:
        xyz_data = colour.Oklab_to_XYZ(Lab)
        xyz_to_rec2020_matrix = colour.models.RGB_COLOURSPACE_LIN_REC2020_SCENE.matrix_XYZ_to_RGB
        rgb_data = colour.XYZ_to_RGB(
            xyz_data,
            colourspace=colour.models.RGB_COLOURSPACE_LIN_REC2020_SCENE,
        )
        return rgb_data
