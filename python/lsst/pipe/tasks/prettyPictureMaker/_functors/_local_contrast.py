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

__all__ = ("DiffusionFunction", "LocalContrastEnhancer")

from lsst.pipe.tasks.prettyPictureMaker.types import FloatImagePlane
from lsst.pex.config.configurableActions import ConfigurableAction, ConfigurableActionField
from lsst.pex.config import Field
from lsst.rubinoxide import rgb

from .._localContrast import localContrast


class DiffusionFunction(ConfigurableAction):
    """Apply anisotropic diffusion processing to enhance image details.

    Anisotropic diffusion is a multi-scale image processing technique that
    selectively smooths regions while preserving edges by using spatially
    varying diffusion coefficients. This implementation uses wavelet-based
    anisotropic diffusion with configurable anisotropy parameters to control
    how different frequency components diffuse relative to their gradients.

    The diffusion process works by:
    - Applying multiple iterations of gradient-based diffusion
    - Using different diffusion speeds for low and high frequency wavelets
    - Controlling diffusion direction via anisotropy parameters
    - Regularizing coefficients to detect and preserve edges
    - Modulating response to low-variance regions via variance threshold
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
    regularization = Field[float]("Regularization of coefficients used to detect edges", default=2.94)
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
    sharpness = Field[float](
        doc="Adjusts wavelet detail amplitude. Positive values sharpen, negative values blur.", default=0.0
    )

    def __call__(self, intensities: FloatImagePlane) -> FloatImagePlane:
        """Apply anisotropic diffusion to the input intensity image.

        Parameters
        ----------
        intensities : `FloatImagePlane`
            The input intensity image to process.

        Returns
        -------
        result : `FloatImagePlane`
            The diffused intensity image with enhanced details.

        Notes
        -----
        This method implements wavelet-based anisotropic diffusion:

        1. Multi-scale decomposition: The image is analyzed across multiple
           frequency bands using wavelet decomposition.
        2. Directional diffusion: Low-frequency wavelets diffuse according to
           their own gradient orientation (anisotropy_first) and high-frequency
           gradients (anisotropy_second). High-frequency wavelets diffuse
           relative to low-frequency gradients (anisotropy_third) and their
           own gradients (anisotropy_fourth).
        3. Speed control: Diffusion speeds are configured via `first`, `second`,
           `third`, and `fourth` parameters for each anisotropy axis.
        4. Edge preservation: Regularization prevents diffusion across edges.
           Variance threshold modulates response to smooth regions.
        5. Scale selection: `radius_center` and `radius` define which scales
           are modified, enabling targeted enhancement or denoising.

        The diffusion equation is solved iteratively for `iterations` steps,
        with the `sharpness` parameter adjusting final detail amplitudes.
        """
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
            sharpness=self.sharpness,
        )


class LocalContrastEnhancer(ConfigurableAction):
    """Multi-stage local contrast enhancement processor.

    This class implements a two-stage approach for enhancing image contrast:

    1. **Local Contrast Enhancement**: Applies scale-space contrast enhancement
       using a Laplacian pyramid approach. This adjusts highlights, shadows,
       and clarity while operating on multiple resolution levels.

    2. **Anisotropic Diffusion**: Optionally applies wavelet-based anisotropic
       diffusion to further sharpen details and preserve edges. This stage
       selectively smooths regions based on local gradient information.

    The processing pipeline is configurable via parameters for both stages,
    allowing fine-tuned control over the enhancement behavior.
    """

    doLocalContrast = Field[bool](
        "Do apply local contrast",
        default=True,
        deprecated=(
            "This will stop working in v31 and be removed in v32, please set doLocalContrast on"
            " PrettyPictureConfig"
        ),
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
        """Apply multi-stage contrast enhancement to the input image.

        Parameters
        ----------
        intensities : `FloatImagePlane`
            The input intensity image to process.

        Returns
        -------
        result : `FloatImagePlane`
            The enhanced intensity image with improved local contrast.

        Notes
        -----
        This method implements a two-stage enhancement pipeline:

        1. **Local Contrast Enhancement** (via `localContrast`):
           - Builds a Laplacian pyramid of the input image
           - Applies scale-dependent contrast modifications
           - Adjusts highlights and shadows via `highlights` and `shadows`
           - Controls clarity and sharpness via `clarity` parameter
           - Operates over `maxLevel` scales, skipping `skipLevels` lowest
           - Uses `sigma` to define what is considered "local"

        2. **Anisotropic Diffusion** (optional, via `diffusionFunction`):
           - Applied only if `doDiffusion=True`
           - Performs wavelet-based anisotropic diffusion
           - Preserves edges while enhancing details
           - Configurable via diffusionFunction parameters

        The two stages are applied sequentially, with the diffusion stage
        operating on the locally enhanced image to further refine details.
        """
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
