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

__all__ = ("RewriteCellCoaddConnections", "RewriteCellCoaddTask", "RewriteCellCoaddConfig")

from typing import Any, ClassVar

import astropy.io.fits
import astropy.time
import astropy.units

import lsst.pipe.base.connectionTypes as cT
from lsst.afw.image import Exposure as LegacyExposure
from lsst.afw.image import Mask as LegacyMask
from lsst.afw.math import BackgroundList
from lsst.cell_coadds import MultipleCellCoadd as LegacyMultipleCellCoadd
from lsst.images import Box, Mask, get_legacy_deep_coadd_mask_planes
from lsst.images.cells import CellCoadd
from lsst.images.fields import field_from_legacy_background
from lsst.images.fits import ExtensionKey, FitsOpaqueMetadata
from lsst.meas.algorithms import SubtractBackgroundTask
from lsst.pex.config import ConfigurableField, Field
from lsst.pipe.base import (
    AlgorithmError,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.skymap import BaseSkyMap


class RewriteCellCoaddConnections(
    PipelineTaskConnections,
    dimensions={"patch", "band"},
    defaultTemplates={"legacy_prefix": "legacy_", "future_prefix": ""},
):
    legacy_cell_coadd = cT.Input(
        "deep_coadd_cell_predetection",
        storageClass="MultipleCellCoadd",
        dimensions={"patch", "band"},
        doc="The pre-detection cell coadd to convert.",
    )
    object_background = cT.Input(
        "deep_coadd_background",
        storageClass="Background",
        dimensions={"patch", "band"},
        doc="The background model used to generate the object catalog, to be subtracted from the coadd.",
        minimum=0,
    )
    object_mask = cT.Input(
        "{legacy_prefix}deep_coadd.mask",
        storageClass="Mask",
        dimensions={"patch", "band"},
        doc="The mask to use for the coadd, including the final DETECTED mask plane.",
        minimum=0,
    )
    alternate_background_coadd = cT.Input(
        "pretty_coadd_extra_bg_subtracted",
        storageClass="ExposureF",
        dimensions={"patch", "band"},
        doc="A coadd to subtract from ``legacy_cell_coadd`` and fit a background to.",
        minimum=0,
    )
    sky_map = cT.Input(
        doc="Description of the skymap's tracts and patches.",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    future_cell_coadd = cT.Output(
        "{future_prefix}deep_coadd",
        storageClass="CellCoadd",
        dimensions={"patch", "band"},
        doc="The output CellCoadd.",
    )

    config: RewriteCellCoaddConfig

    def __init__(self, config: RewriteCellCoaddConfig | None):
        super().__init__(config=config)
        if not self.config.do_alternate_background_fit:
            del self.alternate_background_coadd


class RewriteCellCoaddConfig(
    PipelineTaskConfig,
    pipelineConnections=RewriteCellCoaddConnections,
):
    object_background_description = Field(
        "Description of the object background, to be stored with the image.",
        dtype=str,
        default=(
            "Background subtracted from the image when generating the Object catalog. "
            "This intentionally oversubtracts the background to reduce blending and ensure "
            "scattered light is subtracted. "
            "Restoring this background does not restore all original backgrounds, "
            "as the coadd was built from background-subtracted visit images; in most "
            "cases this background term is actually quite small."
        ),
    )
    do_alternate_background_fit = Field(
        "Whether to fit an alternate background to the difference between "
        "the alternate_background_coadd and the main coadd.",
        dtype=bool,
        default=True,
    )
    alternate_background_fit = ConfigurableField(
        target=SubtractBackgroundTask,
        doc=(
            "Configuration for fitting the alternate background. "
            "Ignored if do_alternate_background_fit is False"
        ),
    )
    alternate_background_name = Field(
        "Name for the alternate background attached to the image. "
        "ignored if do_alternate_background_fit is False.",
        dtype=str,
        default="pretty",
    )
    alternate_background_description = Field(
        "Description for the alternate background attached to the image. "
        "Ignored if do_alternate_background_fit is False.",
        dtype=str,
        default=(
            "An alternate background optimized for visually attractive RGB images. "
            "'Subtracting' this background will generally *add* flux, correcting "
            "for most oversubtraction problems, but leaving in scattered light and "
            "some instrumental backgrounds in some cases. "
            "Because this background is fit to a difference of two wholly different "
            "coadds that may have different input images, it can also be pulled "
            "up or down by bright variable or transient objects."
        ),
    )

    def setDefaults(self) -> None:
        super().setDefaults()
        self.alternate_background_fit.useApprox = False
        self.alternate_background_fit.binSize = 64
        self.alternate_background_fit.ignoredPixelMask = ["NO_DATA", "SAT"]


class RewriteCellCoaddTask(PipelineTask):
    ConfigClass: ClassVar[type[RewriteCellCoaddConfig]] = RewriteCellCoaddConfig
    config: RewriteCellCoaddConfig
    _DefaultName = "rewriteCellCoadd"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.makeSubtask("alternate_background_fit")

    def run(
        self,
        legacy_cell_coadd: LegacyMultipleCellCoadd,
        sky_map: BaseSkyMap,
        *,
        object_background: BackgroundList | None = None,
        object_mask: LegacyMask | None = None,
        alternate_background_coadd: LegacyExposure | None = None,
    ) -> Struct:
        tract_info = sky_map[legacy_cell_coadd.identifiers.tract]
        patch_bbox = tract_info[legacy_cell_coadd.identifiers.patch].getOuterBBox()
        future_cell_coadd = CellCoadd.from_legacy_cell_coadd(
            legacy_cell_coadd,
            tract_info=tract_info,
            bbox=Box.from_legacy(patch_bbox),
        )
        if getattr(future_cell_coadd, "_opaque_metadata", None) is None:
            future_cell_coadd._opaque_metadata = FitsOpaqueMetadata()
        primary_header = future_cell_coadd._opaque_metadata.headers.setdefault(
            ExtensionKey(), astropy.io.fits.Header()
        )
        primary_header.set("INSTRUME", "LSSTCam")
        primary_header.set("ORIGIN", "NSF-DOE Vera C. Rubin Observatory")
        primary_header.set("TELESCOP", "Rubin:Simonyi")
        primary_header.set("DATE", astropy.time.Time.now().fits, "UTC date this HDU was written.")
        if object_mask is not None:
            future_cell_coadd.mask.clear()
            future_cell_coadd.mask.update(
                Mask.from_legacy(object_mask, plane_map=get_legacy_deep_coadd_mask_planes())
            )
        else:
            # If we can't load the object DETECTED plane, clear the one that's
            # present, as it's (misleadingly, in this context) the union of the
            # calibrateImage detections.
            self.log.warning(
                "No object mask found; clearing DETECTED and using other predetection mask planes."
            )
            future_cell_coadd.mask.clear("DETECTED")

        if self.config.do_alternate_background_fit:
            if alternate_background_coadd is not None:
                alt_bg_diff = future_cell_coadd.to_legacy(copy=True)
                alt_bg_diff.maskedImage -= alternate_background_coadd.getMaskedImage()
                try:
                    alt_bg_result = self.alternate_background_fit.run(alt_bg_diff, stats=False)
                except AlgorithmError:
                    self.log.exception("Could not fit alternate background to diff; rewriting without it.")
                else:
                    alt_bg = field_from_legacy_background(alt_bg_result.background, unit=astropy.units.nJy)
                    future_cell_coadd.backgrounds.add(
                        self.config.alternate_background_name,
                        alt_bg,
                        self.config.alternate_background_description,
                    )
            else:
                self.log.warning("No alternate background coadd found; rewriting without it.")
        elif alternate_background_coadd is not None:
            raise TypeError(
                "alternate_background_coadd provided but config.do_alternate_background_fit=False"
            )
        if object_background is not None:
            # We subtract the deep_coadd_background using afw to avoid tiny
            # differences between the afw and lsst.images spline
            # implementations, in the definition of deep_coadd itself (though
            # that will still ultimately differ from the original due to lossy
            # compression).
            future_cell_coadd.image.array -= object_background.getImage().array
            future_cell_coadd.backgrounds.add(
                "object",
                field_from_legacy_background(object_background, unit=astropy.units.nJy),
                self.config.object_background_description,
                is_subtracted=True,
            )
        else:
            self.log.warning(
                "No object background model found; rewriting coadd without subtracting background."
            )
        return Struct(future_cell_coadd=future_cell_coadd)
