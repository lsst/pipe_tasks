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

import functools

import astropy.table
import numpy as np
import pandas as pd

import lsst.afw.geom
import lsst.afw.image
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
        name="visit_image",
        storageClass="ExposureF",
        dimensions=["visit", "detector"],
    )
    diaExposure = cT.Input(
        doc="Input difference image to perform photometry on.",
        name="difference_image",
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
    outputCatalog = cT.Output(
        doc="Output forced photometry catalog.",
        name="object_forced_source_unstandardized",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "detector", "skymap", "tract")
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        assert isinstance(config, ForcedPhotDetectorConfig)

        if not config.doDirectPhotometry:
            del self.exposure
        if not config.doDifferencePhotometry:
            del self.diaExposure


class ForcedPhotDetectorConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=ForcedPhotDetectorConnections):
    """Configuration for the ForcedPhotDetectorTask."""
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
    doDirectPhotometry = lsst.pex.config.Field(
        doc="Perform direct photometry on the input exposure.",
        dtype=bool,
        default=True,
    )
    doDifferencePhotometry = lsst.pex.config.Field(
        doc="Perform photometry on the difference image.",
        dtype=bool,
        default=True,
    )


class ForcedPhotDetectorTask(pipeBase.PipelineTask):
    """A pipeline task for performing forced photometry on CCD images."""
    ConfigClass = ForcedPhotDetectorConfig
    _DefaultName = "forcedPhotDetector"

    def __init__(self, initInputs=None, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("measurement")
        if self.config.doApCorr:
            self.makeSubtask("applyApCorr", schema=self.measurement.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        if "exposure" in inputs:
            exposure = inputs['exposure']
            bbox = exposure.getBBox()
            wcs = exposure.getWcs()
            if wcs is None:
                raise NoWorkFound("Exposure has no valid WCS.")
        else:
            exposure = None

        if "diaExposure" in inputs:
            diaExposure = inputs['diaExposure']
            if exposure is None:
                bbox = diaExposure.getBBox()
                wcs = diaExposure.getWcs()
                if wcs is None:
                    raise NoWorkFound("Difference exposure has no valid WCS.")
        else:
            if exposure is None:
                # If neither exposure nor diaExposure is provided, we cannot proceed.
                raise NoWorkFound("No valid exposure or difference exposure provided.")
            diaExposure = None

        tract = butlerQC.quantum.dataId['tract']
        skyMap = inputs.pop('skyMap')
        refWcs = skyMap[tract].getWcs()

        self.log.info("Filtering ref cats: %s", ','.join([str(i.dataId) for i in inputs['refCat']]))

        table = self._filterRefCat(
            inputs['refCat'],
            bbox,
            wcs,
        )

        outputs = self.run(
            table=table,
            exposure=exposure,
            diaExposure=diaExposure,
            visit=butlerQC.quantum.dataId['visit'],
            detector=butlerQC.quantum.dataId['detector'],
            refWcs=refWcs,
            band=butlerQC.quantum.dataId["band"],
        )

        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        table: astropy.table.Table,
        visit: int,
        detector: int,
        refWcs: lsst.afw.geom.SkyWcs,
        exposure: lsst.afw.image.Exposure | None = None,
        diaExposure: lsst.afw.image.Exposure | None = None,
        band: str | None = None,
    ) -> pipeBase.Struct:
        results: dict[str, astropy.table.Table] = {}
        if self.config.doDirectPhotometry:
            if exposure is None:
                raise ValueError("`exposure` must be provided for direct photometry.")
            self.log.info("Running forced measurement on %s objects", len(table))
            results['calexp'] = self._runForcedPhotometry(table, exposure, refWcs)

        if self.config.doDifferencePhotometry:
            if diaExposure is None:
                raise ValueError("`diaExposure` must be provided for difference photometry.")
            self.log.info("Running forced measurement on %s objects on difference image", len(table))
            results['diff'] = self._runForcedPhotometry(table, diaExposure, refWcs)

        # Convert the astropy tables to pandas DataFrames and reindex them
        dfs = []
        for dataset, table in results.items():
            if self.config.measurement.refCatRaColumn != "coord_ra":
                table.rename_column(self.config.measurement.refCatRaColumn, "coord_ra")
                table.rename_column(self.config.measurement.refCatDecColumn, "coord_dec")
            df = table.to_pandas().set_index(self.config.measurement.refCatIdColumn, drop=False)
            df = df.reindex(sorted(df.columns), axis=1)
            df["visit"] = visit
            # int16 instead of uint8 because databases don't like unsigned bytes.
            df["detector"] = np.int16(detector)
            df["band"] = band if band else pd.NA
            df.columns = pd.MultiIndex.from_tuples([(dataset, c) for c in df.columns],
                                                   names=("dataset", "column"))
            dfs.append(df)

        # Join the DataFrames on the index (which is the object ID)
        outputCatalog = functools.reduce(lambda d1, d2: d1.join(d2), dfs)
        return pipeBase.Struct(outputCatalog=outputCatalog)

    def _runForcedPhotometry(
        self,
        table: astropy.table.Table,
        exposure: lsst.afw.image.Exposure,
        refWcs: lsst.afw.geom.SkyWcs
    ) -> astropy.table.Table:
        """Perform forced measurement on a single exposure.

        Parameters
        ----------
        table : `astropy.table.Table`
            Astropy table containing the reference catalog data, with columns
            for the object ID, right ascension, and declination.
        exposure : `lsst.afw.image.exposure.Exposure`
            Input exposure to adjust calibrations.
        refWcs : `lsst.afw.geom.SkyWcs`
            Defines the X,Y coordinate system of ``refCat``.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            A struct containing the measurement results, including the
            measured table. The struct has the following attributes:
            - `measTable`: `astropy.table.Table`
                containing the forced photometry results
        """
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
        return outputs.measTable

    def _filterRefCat(self, refCatHandles, exposureBBox, exposureWcs):
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
                        self.config.measurement.refCatIdColumn,
                        self.config.measurement.refCatRaColumn,
                        self.config.measurement.refCatDecColumn,
                    ]
                }
            )
            for i in refCatHandles
        ]
        full_table = astropy.table.vstack(table_list)
        # translate ra/dec coords in table to detector pixel coords
        # to down-select rows that overlap the detector bbox
        x, y = exposureWcs.skyToPixelArray(
            full_table[self.config.measurement.refCatRaColumn],
            full_table[self.config.measurement.refCatDecColumn],
            degrees=True,
        )
        inBBox = lsst.geom.Box2D(exposureBBox).contains(x, y)
        return full_table[inBBox]
