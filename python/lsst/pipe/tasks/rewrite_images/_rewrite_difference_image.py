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

__all__ = ("RewriteDifferenceImageConnections", "RewriteDifferenceImageTask", "RewriteDifferenceImageConfig")

from typing import ClassVar

import astropy.units

import lsst.pipe.base.connectionTypes as cT
from lsst.afw.image import PhotoCalib
from lsst.images import DifferenceImage
from lsst.images.fields import field_from_legacy_photo_calib
from lsst.pex.config import Field
from lsst.pipe.base import (
    InputQuantizedConnection,
    OutputQuantizedConnection,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    QuantumContext,
    Struct,
)


class RewriteDifferenceImageConnections(
    PipelineTaskConnections,
    dimensions={"visit", "detector"},
    defaultTemplates={"legacy_prefix": "legacy_", "future_prefix": ""},
):
    legacy_exposure = cT.Input(
        "{legacy_prefix}difference_image",
        # We expect the repository storage class to be ExposureF, but we set
        # the storage class we want to DifferenceImage so the butler can do
        # most of the conversion on read, and in doing so preserve the
        # quantization so we don't have doubly-lossless compression.
        storageClass="DifferenceImage",
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
    future_difference_image = cT.Output(
        "{future_prefix}difference_image",
        storageClass="DifferenceImage",
        dimensions={"visit", "detector"},
        doc="The output difference image.",
    )

    config: RewriteDifferenceImageConfig


class RewriteDifferenceImageConfig(
    PipelineTaskConfig,
    pipelineConnections=RewriteDifferenceImageConnections,
):
    instrumental_unit = Field(
        "Unit for instrumental flux pixels (i.e. uncalibrated side of a PhotoCalib).",
        dtype=str,
        default="electron",
    )


class RewriteDifferenceImageTask(PipelineTask):
    ConfigClass: ClassVar[type[RewriteDifferenceImageConfig]] = RewriteDifferenceImageConfig
    config: RewriteDifferenceImageConfig
    _DefaultName = "rewriteDifferenceImage"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        inputs = butlerQC.get(inputRefs)
        difference_image = inputs.pop("legacy_exposure").get(parameters={"preserve_quantization": True})
        visit_summary = inputs.pop("visit_summary")
        photo_calib: PhotoCalib = visit_summary.find(butlerQC.quantum.dataId["detector"]).getPhotoCalib()
        outputs = self.run(difference_image, photo_calib=photo_calib, **inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, difference_image: DifferenceImage, *, photo_calib: PhotoCalib) -> Struct:
        instrumental_unit = astropy.units.Unit(self.config.instrumental_unit)
        difference_image.photometric_scaling = field_from_legacy_photo_calib(
            photo_calib, bounds=difference_image.bbox, instrumental_unit=instrumental_unit
        )
        return Struct(future_difference_image=difference_image)
