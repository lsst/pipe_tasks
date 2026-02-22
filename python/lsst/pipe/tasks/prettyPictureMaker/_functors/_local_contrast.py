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

__all__ = ("DiffusionFunction", "LocalContrastEnhansor")

from lsst.pipe.tasks.prettyPictureMaker.types import FloatImagePlane
from lsst.pex.config.configurableActions import ConfigurableAction, ConfigurableActionField
from lsst.pex.config import Field
from lsst.rubinoxide import rgb

from .._localContrast import localContrast


class DiffusionFunction(ConfigurableAction):
    """Configuration that controlls the rgb diffusion process. This can be
    used for things like sharpening, or filling in holes.
    """

    iterations = Field[int]("number of interations in the diffusion process", default=3)
    anisotropy_first = Field[float](
        "The diffusion direction of low-frequency wavelets relative to their own gradient orientation",
        default=1,
    )
    anisotropy_second = Field[float](
        "The diffusion direction of low-frequency wavelets relative to the high-frequency gradient.",
        default=1,
    )
    anisotropy_third = Field[float](
        "The diffusion direction of high-frequency wavelets relative to the low-frequency gradient.",
        default=1,
    )
    anisotropy_fourth = Field[float](
        "The diffusion direction of high-frequency wavelets relative to their own gradient orientation",
        default=1,
    )
    regularization = Field[float]("Regularization of coefficiencts used to detect edges", default=2.94)
    variance_threshold = Field[float](
        doc=(
            "The variance threshold modulates the filter's response to low-variance regions, with positive "
            " values enhancing local contrast and negative values suppressing noise and blur in those areas"
        ),
        default=0.0,
    )
    radius_center = Field[float](
        doc=(
            "The diffusion scale parameter: zero diffuses fine details (deblurring/denoising), "
            "while non-zero values selectively diffuse larger scales to enhance local contrast."
        ),
        default=0.0,
    )
    radius = Field[float](
        doc="The diffusion span defines a radial band (center ± span) for detail modification.", default=5.0
    )
    first = Field[float](doc="Anisotropic diffusion speed of low-frequency wavelets", default=0.0065)
    second = Field[float](
        doc="Low-frequency wavelet diffusion speed along the 2nd-order anisotropy axis", default=-0.25
    )
    third = Field[float](
        doc="High-frequency wavelet diffusion speed along the 3rd-order anisotropy axis", default=-0.25
    )
    fourth = Field[float](
        doc="High-frequency wavelet diffusion speed along the 4th-order anisotropy axis.", default=-0.2774
    )
    shaprness = Field[float](
        doc="Adjusts wavelet detail amplitude. Positive values sharpen, negative values blur.", default=0.0
    )

    def __call__(self, intensities: FloatImagePlane) -> FloatImagePlane:
        return rgb.diffuse_gray_image(
            intensities,
            iterations=self.iterations,
            radius_center=self.radius_center,
            radius=self.radius,
            regularization=self.regularization,
            anisotropy_first=self.anisotropy_first,
            anisotropy_second=self.anisotropy_second,
            anisotropy_third=self.anisotropy_third,
            anisotropy_fourth=self.anisotropy_fourth,
            first=self.first,
            second=self.second,
            third=self.third,
            fourth=self.fourth,
            variance_threshold=self.variance_threshold,
            sharpness=self.shaprness,
        )


class LocalContrastEnhansor(ConfigurableAction):
    doLocalContrast = Field[bool](
        "Do apply local contrast",
        default=True,
        deprecated="This will stop working in v31 and be removed in v32, please set doLocalContrast on PrettyPictureConfig",
    )
    highlights = Field[float](doc="Adjustment factor for the highlights", default=-0.9)
    shadows = Field[float](doc="Adjustment factor for the shadows", default=0.5)
    clarity = Field[float](doc="Amount of clarity to apply to contrast modification", default=0.1)
    sigma = Field[float](
        doc="The scale size of what is considered local in the contrast enhancement", default=30
    )
    maxLevel = Field[int](
        doc="The maximum number of scales the contrast should be enhanced over, if None then all",
        default=4,
        optional=True,
    )
    skipLevels = Field[int]("Skip this many lowest levels in laplace pyramid", default=0)
    doDiffusion = Field[bool]("Run the diffusion function or not", default=True)
    diffusionFunction = ConfigurableActionField[DiffusionFunction](
        doc="Diffusion function to enhance local contrast",
    )

    def setDefaults(self) -> None:
        self.diffusionFunction.iterations = 2
        self.diffusionFunction.radius_center = 280
        self.diffusionFunction.radius = 400
        self.diffusionFunction.regularization = 0.0
        self.diffusionFunction.first = -1.25
        self.diffusionFunction.third = 0.0
        self.diffusionFunction.fourth = 0.0

    def __call__(self, intensities: FloatImagePlane) -> FloatImagePlane:
        intensities = localContrast(
            intensities,
            sigma=self.sigma,
            highlights=self.highlights,
            shadows=self.shadows,
            clarity=self.clarity,
            maxLevel=self.maxLevel,
            skip_levels=self.skipLevels,
        )
        if self.doDiffusion:
            intensities = self.diffusionFunction(intensities)
        return intensities
