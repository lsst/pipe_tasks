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

__all__ = ("MakePsfMatchedWarpConfig", "MakePsfMatchedWarpConnections", "MakePsfMatchedWarpTask",)

from typing import TYPE_CHECKING

import lsst.geom as geom
import numpy as np
import warnings

from lsst.afw.geom import Polygon, makeWcsPairTransform
from lsst.coadd.utils import copyGoodPixels
from lsst.ip.diffim import ModelPsfMatchTask
from lsst.meas.algorithms import GaussianPsfFactory, WarpedPsf
from lsst.pex.config import ConfigurableField
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.utils.timer import timeMethod

if TYPE_CHECKING:
    from lsst.afw.image import Exposure


class MakePsfMatchedWarpConnections(
    PipelineTaskConnections,
    dimensions=("tract", "patch", "skymap", "instrument", "visit"),
    defaultTemplates={
        "coaddName": "deep",
    },
):
    """Connections for MakePsfMatchedWarpTask"""

    direct_warp = Input(
        doc="Direct warped exposure produced by resampling calexps onto the skyMap patch geometry",
        name="{coaddName}Coadd_directWarp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "instrument", "visit"),
    )

    psf_matched_warp = Output(
        doc=(
            "Output PSF-Matched warped exposure, produced by resampling ",
            "calexps onto the skyMap patch geometry and PSF-matching to a model PSF.",
        ),
        name="{coaddName}Coadd_psfMatchedWarp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
    )


class MakePsfMatchedWarpConfig(
    PipelineTaskConfig,
    pipelineConnections=MakePsfMatchedWarpConnections,
):
    """Config for MakePsfMatchedWarpTask."""

    modelPsf = GaussianPsfFactory.makeField(doc="Model Psf factory")
    psfMatch = ConfigurableField(
        target=ModelPsfMatchTask,
        doc="Task to warp and PSF-match calexp",
    )

    def setDefaults(self):
        super().setDefaults()
        self.psfMatch.kernel["AL"].alardSigGauss = [1.0, 2.0, 4.5]
        self.modelPsf.defaultFwhm = 7.7


class MakePsfMatchedWarpTask(PipelineTask):
    ConfigClass = MakePsfMatchedWarpConfig
    _DefaultName = "makePsfMatchedWarp"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("psfMatch")

    @timeMethod
    def run(self, direct_warp: Exposure):
        """Make a PSF-matched warp from a direct warp.

        Each individual detector from the direct warp is isolated, one at a
        time, and PSF-matched to the same model PSF. The PSF-matched images are
        then added back together to form the final PSF-matched warp. The bulk
        of the work is done by the `psfMatchTask`.

        Notes
        -----
        Pixels that receive no inputs are set to NaN, for e.g, chip gaps. This
        violates LSST algorithms group policy.

        Parameters
        ----------
        direct_warp : `lsst.afw.image.Exposure`
            Direct warp to be PSF-matched.

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            Struct containing the PSF-matched warp under the attribute
            `psf_matched_warp`.
        """
        modelPsf = self.config.modelPsf.apply()

        # Prepare the output exposure. We clone the input image to keep the
        # metadata, but reset the image and variance plans.
        exposure_psf_matched = direct_warp.clone()
        exposure_psf_matched.image.array[:, :] = np.nan
        exposure_psf_matched.variance.array[:, :] = np.inf
        exposure_psf_matched.setPsf(modelPsf)

        bit_mask = direct_warp.mask.getPlaneBitMask("NO_DATA")
        total_good_pixels = 0  # Total number of pixels copied to output.

        for row in direct_warp.info.getCoaddInputs().ccds:
            transform = makeWcsPairTransform(row.wcs, direct_warp.wcs)
            warp_psf = WarpedPsf(row.getPsf(), transform)

            if (destination_polygon := row.validPolygon) is None:
                # Calculate the polygon for this detector.
                src_polygon = Polygon(
                    [geom.Point2D(corner) for corner in row.getBBox().getCorners()]
                )
                destination_polygon = src_polygon.transform(transform).intersectionSingle(
                    geom.Box2D(direct_warp.getBBox())
                )
                self.log.debug("Polygon for detector=%d is calculated as %s",
                               row["ccd"],
                               destination_polygon
                               )
            else:
                self.log.debug("Polygon for detector=%d is read from the input calexp as %s",
                               row["ccd"],
                               destination_polygon
                               )

            # Compute the minimum possible bounding box that overlaps the CCD.
            # First find the intersection polygon between the per-detector warp
            # and the warp bounding box.
            bbox = geom.Box2I()
            for corner in destination_polygon.getVertices():
                bbox.include(geom.Point2I(corner))
            bbox.clip(direct_warp.getBBox())  # Additional safeguard

            self.log.debug("PSF-matching CCD %d with bbox %s", row["ccd"], bbox)

            ccd_mask_array = ~(destination_polygon.createImage(bbox).array <= 0)

            # Clone the subimage, set the PSF to the model and reset the planes
            # outside the detector.
            temp_warp = direct_warp[bbox].clone()
            temp_warp.setPsf(warp_psf)
            temp_warp.image.array *= ccd_mask_array
            temp_warp.mask.array |= (~ccd_mask_array) * bit_mask
            # We intend to divide by zero outside the detector to set the
            # per-pixel variance values to infinity. Suppress the warning.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="divide by zero", category=RuntimeWarning)
                temp_warp.variance.array /= ccd_mask_array

            temp_psf_matched = self.psfMatch.run(temp_warp, modelPsf).psfMatchedExposure
            del temp_warp

            # Set pixels outside the intersection polygon to NO_DATA.
            temp_psf_matched.maskedImage[bbox].mask.array |= (~ccd_mask_array) * bit_mask

            num_good_pixels = copyGoodPixels(
                exposure_psf_matched.maskedImage[bbox],
                temp_psf_matched.maskedImage[bbox],
                bit_mask,
            )

            del temp_psf_matched

            self.log.info(
                "Copied %d pixels from CCD %d to exposure_psf_matched", num_good_pixels, row["ccd"],
            )
            total_good_pixels += num_good_pixels

        self.log.info("Total number of good pixels = %d", total_good_pixels)

        return Struct(psf_matched_warp=exposure_psf_matched)
