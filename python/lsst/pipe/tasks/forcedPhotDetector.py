# This file is part of meas_base.
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

import astropy.table

import lsst.pex.config
from lsst.meas.base.simple_forced_measurement import SimpleForcedMeasurementTask
from lsst.meas.base.applyApCorr import ApplyApCorrTask
import lsst.pipe.base as pipeBase
from lsst.pipe.base import PipelineTaskConnections, NoWorkFound
import lsst.pipe.base.connectionTypes as cT
from lsst.skymap import BaseSkyMap

__all__ = ("ForcedPhotDetectorConfig", "ForcedPhotDetectorTask")


class ForcedPhotDetectorConnections(PipelineTaskConnections,
                                    dimensions=("visit", "detector", "tract")):
    exposure = cT.Input(
        doc="Input exposure to perform photometry on.",
        name="calexp",
        storageClass="ExposureF",
        dimensions=["visit", "detector"],
    )
    refCat = cT.Input(
        doc="Catalog of shapes and positions at which to force photometry.",
        name="object_patch",
        storageClass="ArrowAstropy",
        dimensions=["tract", "patch"],
        multiple=True,
        deferLoad=True,
    )
    skyMap = cT.Input(
        doc="SkyMap dataset that defines the coordinate system of the reference catalog.",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=["skymap"],
    )
    measTable = cT.Output(
        doc="Output forced photometry catalog.",
        name="object_forced_source_direct_unstandardized",
        storageClass="ArrowAstropy",
        dimensions=["instrument", "visit", "detector", "skymap", "tract"],
    )


class ForcedPhotDetectorConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=ForcedPhotDetectorConnections):
    measurement = lsst.pex.config.ConfigurableField(
        target=SimpleForcedMeasurementTask,
        doc="subtask to do forced measurement"
    )
    doApCorr = lsst.pex.config.Field(
        dtype=bool,
        default=True,
        doc="Run subtask to apply aperture corrections"
    )
    applyApCorr = lsst.pex.config.ConfigurableField(
        target=ApplyApCorrTask,
        doc="Subtask to apply aperture corrections"
    )
    refCatIdColumn = lsst.pex.config.Field(
        dtype=str,
        default="diaObjectId",
        doc=(
            "Name of the column that provides the object ID from the refCat connection. "
            "measurement.copyColumns['id'] must be set to this value as well."
        )
    )
    refCatRaColumn = lsst.pex.config.Field(
        dtype=str,
        default="ra",
        doc=(
            "Name of the column that provides the right ascension (in floating-point degrees) from the "
            "refCat connection. "
        )
    )
    refCatDecColumn = lsst.pex.config.Field(
        dtype=str,
        default="dec",
        doc=(
            "Name of the column that provides the declination (in floating-point degrees) from the "
            "refCat connection. "
        )
    )


class ForcedPhotDetectorTask(pipeBase.PipelineTask):

    ConfigClass = ForcedPhotDetectorConfig
    _DefaultName = "forcedPhotDetector"

    def __init__(self, initInputs=None, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("measurement")
        if self.config.doApCorr:
            self.makeSubtask("applyApCorr", schema=self.measurement.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        tract = butlerQC.quantum.dataId['tract']
        skyMap = inputs.pop('skyMap')
        refWcs = skyMap[tract].getWcs()
        exposure = inputs['exposure']
        if inputs["exposure"].getWcs() is None:
            raise NoWorkFound("Exposure has no WCS.")
        self.log.info("Filtering ref cats: %s", ','.join([str(i.dataId) for i in inputs['refCat']]))
        table = self.filterRefCat(
            inputs['refCat'],
            inputs['exposure'].getBBox(),
            inputs['exposure'].getWcs(),
        )
        outputs = self.run(table, exposure, refWcs)
        butlerQC.put(outputs, outputRefs)

    def run(self, table, exposure, refWcs):
        outputs = self.measurement.run(table, exposure, refWcs)
        if self.config.doApCorr:
            apCorrMap = exposure.getInfo().getApCorrMap()
            if apCorrMap is None:
                self.log.warning("Forced exposure image does not have valid aperture correction; skipping.")
            else:
                self.applyApCorr.run(
                    catalog=outputs.measTable,
                    apCorrMap=apCorrMap,
                )
        return outputs

    def filterRefCat(self, refCatHandles, exposureBBox, exposureWcs):
        """Prepare a merged, filtered reference catalog from ArrowAstropy
        inputs.

        Parameters
        ----------
        refCatHandles : sequence of `lsst.daf.butler.DeferredDatasetHandle`
            Handles for catalogs of shapes and positions at which to force
            photometry.
        exposureBBox :   `lsst.geom.Box2I`
            Bounding box on which to select rows that overlap
        exposureWcs : `lsst.afw.geom.SkyWcs`
            World coordinate system to convert sky coords in ref cat to
            pixel coords with which to compare with exposureBBox

        Returns
        -------
        refCat : `lsst.afw.table.SourceTable`
            Source Catalog with minimal schema that overlaps exposureBBox
        """
        table_list = [
            i.get(
                parameters={
                    "columns": [
                        self.config.refCatIdColumn,
                        self.config.refCatRaColumn,
                        self.config.refCatDecColumn,
                    ]
                }
            )
            for i in refCatHandles
        ]
        full_table = astropy.table.vstack(table_list)
        # translate ra/dec coords in table to detector pixel coords
        # to down-select rows that overlap the detector bbox
        x, y = exposureWcs.skyToPixelArray(
            full_table[self.config.refCatRaColumn],
            full_table[self.config.refCatDecColumn],
            degrees=True,
        )
        inBBox = lsst.geom.Box2D(exposureBBox).contains(x, y)
        return full_table[inBBox]
