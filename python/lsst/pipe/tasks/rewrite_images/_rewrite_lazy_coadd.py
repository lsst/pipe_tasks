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

__all__ = ("RewriteLazyCoaddConnections", "RewriteLazyCoaddTask", "RewriteLazyCoaddConfig")

from typing import ClassVar

import lsst.pipe.base.connectionTypes as cT
from lsst.afw.image import Exposure as LegacyExposure
from lsst.daf.butler import DataCoordinate
from lsst.images.cells import CellCoadd
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
from lsst.skymap import BaseSkyMap


class RewriteLazyCoaddConnections(
    PipelineTaskConnections,
    dimensions={"patch", "band"},
    defaultTemplates={"legacy_prefix": "legacy_", "future_prefix": ""},
):
    legacy_exposure = cT.Input(
        "template_coadd",
        storageClass="ExposureF",
        dimensions={"patch", "band"},
        doc="The lazy-PSF Exposure coadd to convert.",
    )
    sky_map = cT.Input(
        doc="Description of the skymap's tracts and patches.",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    future_cell_coadd = cT.Output(
        "{future_prefix}template_coadd",
        storageClass="CellCoadd",
        dimensions={"patch", "band"},
        doc="The output CellCoadd.",
    )

    config: RewriteLazyCoaddConfig


class RewriteLazyCoaddConfig(
    PipelineTaskConfig,
    pipelineConnections=RewriteLazyCoaddConnections,
):
    instrument = Field(
        "Name of the instrument assumed for all inputs to this coadd.",
        dtype=str,
        default="LSSTCam",
    )


class RewriteLazyCoaddTask(PipelineTask):
    ConfigClass: ClassVar[type[RewriteLazyCoaddConfig]] = RewriteLazyCoaddConfig
    config: RewriteLazyCoaddConfig
    _DefaultName = "rewriteLazyCoadd"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs, data_id=butlerQC.quantum.dataId)
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        legacy_exposure: LegacyExposure,
        sky_map: BaseSkyMap,
        data_id: DataCoordinate,
    ) -> Struct:
        future_cell_coadd = CellCoadd.from_legacy_lazy_coadd(
            legacy_exposure,
            instrument=self.config.instrument,
            tract_info=sky_map[data_id["tract"]],
            patch=data_id["patch"],
            band=data_id["band"],
        )
        return Struct(future_cell_coadd=future_cell_coadd)
