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

__all__ = ("RewriteVisitImageConnections", "RewriteVisitImageTask", "RewriteVisitImageConfig")

from typing import ClassVar

import astropy.units
import numpy as np

import lsst.pipe.base.connectionTypes as cT
from lsst.afw.image import PhotoCalib
from lsst.afw.math import BackgroundList, BackgroundMI
from lsst.images import VisitImage
from lsst.images.fields import field_from_legacy_background, field_from_legacy_photo_calib
from lsst.pex.config import ChoiceField, Field
from lsst.pipe.base import (
    InputQuantizedConnection,
    OutputQuantizedConnection,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    QuantumContext,
    Struct,
)


class RewriteVisitImageConnections(
    PipelineTaskConnections,
    dimensions={"visit", "detector"},
    defaultTemplates={"legacy_prefix": "legacy_", "future_prefix": ""},
):
    legacy_exposure = cT.Input(
        "{legacy_prefix}visit_image",
        # We expect the repository storage class to be ExposureF, but we set
        # the storage class we want to VisitImage so the butler can do most
        # of the conversion on read, and in doing so preserve the quantization
        # so we don't have doubly-lossless compression.
        storageClass="VisitImage",
        dimensions={"visit", "detector"},
        deferLoad=True,  # So we can pass preserve_quantization=True as a parameter.
        doc="The input image to convert.",
    )
    visit_summary = cT.Input(
        "visit_summary",
        storageClass="ExposureCatalog",
        dimensions={"visit"},
        doc="A visit summary catalog with the PhotoCalib that was already applied to the image's pixels.",
    )
    photo_calib = cT.Input(
        "initial_photo_calib_detector",
        storageClass="PhotoCalib",
        dimensions={"visit", "detector"},
        doc="The PhotoCalib that was already applied to the image's pixels.",
    )
    subtracted_background = cT.Input(
        "visit_image_background",
        storageClass="Background",
        dimensions={"visit", "detector"},
        doc="The background model that was subtracted from this image.",
    )
    alternate_background = cT.Input(
        "skyCorr",
        storageClass="Background",
        dimensions={"visit", "detector"},
        doc="A different background model that was not subtracted from the image.",
    )
    future_visit_image = cT.Output(
        "{future_prefix}visit_image",
        storageClass="VisitImage",
        dimensions={"visit", "detector"},
        doc="The output VisitImage.",
    )

    config: RewriteVisitImageConfig

    def __init__(self, config: RewriteVisitImageConfig | None):
        super().__init__(config=config)
        match self.config.photo_calib_source:
            case "attached":
                del self.visit_summary
                del self.photo_calib
            case "visit_summary":
                del self.photo_calib
            case "standalone":
                del self.visit_summary
        if self.config.alternate_background_type is None:
            del self.alternate_background


class RewriteVisitImageConfig(
    PipelineTaskConfig,
    pipelineConnections=RewriteVisitImageConnections,
):
    photo_calib_source = ChoiceField[str](
        "Which kind of PhotoCalib to load.",
        dtype=str,
        allowed={
            "attached": "The input image has uncalibrated pixels and a nontrivial PhotoCalib attached to it.",
            "visit_summary": "Load the visit_summary connection to find the PhotoCalib already applied.",
            "standalone": "Load the photo_calib connection to find the PhotoCalib already applied.",
        },
        default="visit_summary",
    )
    subtracted_background_description = Field(
        "Description of the subtracted background, to be stored with the image.",
        dtype=str,
        default="Background subtracted from the image when generating the Source catalog.",
    )
    alternate_background_name = Field(
        "Name for the alternate background attached to the image. "
        "ignored if alternate_background_type is None.",
        dtype=str,
        default="skyCorr",
    )
    alternate_background_description = Field(
        "Description for the alternate background attached to the image. "
        "Ignored if alternate_background_type is None.",
        dtype=str,
        default=(
            "An alternate large-scale visit-level background subtracted from the input images that go into "
            "pretty_coadd and RGB color images.  May preserve more low surface-brightness features, but may "
            "also be more affected by scattered light and other artifacts."
        ),
    )
    alternate_background_type = ChoiceField(
        "How the alternate background relates to the subtracted background.",
        dtype=str,
        allowed={
            "independent": (
                "The alternate background should be subtracted after restoring the subtracted background."
            ),
            "differential_composed": (
                "The alternate background starts with terms that invert the subtracted background."
            ),
            "differential_fit": (
                "The alternate background was fit to the already-subtracted image, and "
                "can only be subtracted from the already-subtracted image."
            ),
        },
        default="differential_composed",
        optional=True,
    )
    instrumental_unit = Field(
        "Unit for instrumental flux pixels (i.e. uncalibrated side of a PhotoCalib, "
        "and the units of all background inputs).",
        dtype=str,
        default="electron",
    )


class RewriteVisitImageTask(PipelineTask):
    ConfigClass: ClassVar[type[RewriteVisitImageConfig]] = RewriteVisitImageConfig
    config: RewriteVisitImageConfig
    _DefaultName = "rewriteVisitImage"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        inputs = butlerQC.get(inputRefs)
        visit_image = inputs.pop("legacy_exposure").get(parameters={"preserve_quantization": True})
        photo_calib: PhotoCalib | None
        match self.config.photo_calib_source:
            case "attached":
                photo_calib = None
            case "visit_summary":
                visit_summary = inputs.pop("visit_summary")
                photo_calib = visit_summary.find(butlerQC.quantum.dataId["detector"]).getPhotoCalib()
            case "standalone":
                photo_calib = inputs.pop("photo_calib")
        outputs = self.run(visit_image, photo_calib=photo_calib, **inputs)
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        visit_image: VisitImage,
        *,
        photo_calib: PhotoCalib | None = None,
        subtracted_background: BackgroundList,
        alternate_background: BackgroundList | None = None,
    ) -> Struct:
        instrumental_unit = astropy.units.Unit(self.config.instrumental_unit)
        if photo_calib is not None:
            visit_image.photometric_scaling = field_from_legacy_photo_calib(
                photo_calib, bounds=visit_image.bbox, instrumental_unit=instrumental_unit
            )
        visit_image.backgrounds.add(
            "subtracted",
            field_from_legacy_background(
                subtracted_background, bounds=visit_image.bbox, unit=instrumental_unit
            ),
            self.config.subtracted_background_description,
            is_subtracted=True,
        )
        if alternate_background is not None:
            assert self.config.alternate_background_name is not None, (
                "Configuration and arguments are inconsistent."
            )
            match self.config.alternate_background_type:
                case "independent":
                    pass
                case "differential_composed":
                    for (subtracted_term, *_), (alternate_term, *_) in zip(
                        subtracted_background, alternate_background
                    ):
                        if not self._do_backgrounds_cancel(subtracted_term, alternate_term):
                            raise RuntimeError(
                                "alternate_background_type='differential_composed', but the alternate "
                                "background does not start with the inverse of the subtracted background."
                            )
                    alternate_background = BackgroundList(
                        *[
                            alternate_background[n]
                            for n in range(len(subtracted_background), len(alternate_background))
                        ]
                    )
                case "differential_fit":
                    alternate_background = BackgroundList(*subtracted_background, *alternate_background)
            visit_image.backgrounds.add(
                self.config.alternate_background_name,
                field_from_legacy_background(
                    alternate_background, bounds=visit_image.bbox, unit=instrumental_unit
                ),
                self.config.alternate_background_description,
                is_subtracted=True,
            )
        return Struct(future_visit_image=visit_image)

    def _do_backgrounds_cancel(self, bg1: BackgroundMI, bg2: BackgroundMI) -> bool:
        ctrl1 = bg1.getBackgroundControl()
        ctrl2 = bg2.getBackgroundControl()
        if ctrl1.getInterpStyle() != ctrl2.getInterpStyle():
            return False
        if ctrl1.getApproximateControl().getStyle() != ctrl2.getApproximateControl().getStyle():
            return False
        bins1 = bg1.getStatsImage()
        bins2 = bg2.getStatsImage()
        if bins1.getBBox() != bins2.getBBox():
            return False
        return np.array_equal(bins1.image.array, -bins2.image.array, equal_nan=True)
