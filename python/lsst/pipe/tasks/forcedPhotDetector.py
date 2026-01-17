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
import lsst.daf.butler
import lsst.pex.config
from lsst.meas.base import DetectorVisitIdGeneratorConfig
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
        dimensions=("visit", "detector", "skymap", "tract")
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
    refCatIdColumn = lsst.pex.config.Field(
        dtype=str,
        default="objectId",
        doc=(
            "Name of the column that provides the object ID from the refCat connection. "
            "measurement.copyColumns['id'] must be set to this value as well."
            "Ignored if refCatStorageClass='SourceCatalog'."
        )
    )
    refCatRaColumn = lsst.pex.config.Field(
        dtype=str,
        default="coord_ra",
        doc=(
            "Name of the column that provides the right ascension (in floating-point degrees) from the "
            "refCat connection. "
            "Ignored if refCatStorageClass='SourceCatalog'."
        )
    )
    refCatDecColumn = lsst.pex.config.Field(
        dtype=str,
        default="coord_dec",
        doc=(
            "Name of the column that provides the declination (in floating-point degrees) from the "
            "refCat connection. "
            "Ignored if refCatStorageClass='SourceCatalog'."
        )
    )
    idGenerator = DetectorVisitIdGeneratorConfig.make_field()


class ForcedPhotDetectorTask(pipeBase.PipelineTask):
    """A pipeline task for performing forced photometry on CCD images."""
    ConfigClass = ForcedPhotDetectorConfig
    _DefaultName = "forcedPhotDetector"

    def __init__(self, initInputs=None, **kwargs):
        super().__init__(**kwargs)

        refSchema = lsst.afw.table.SourceTable.makeMinimalSchema()

        self.makeSubtask("measurement", refSchema=refSchema)
        if self.config.doApCorr:
            self.makeSubtask("applyApCorr", schema=self.measurement.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        if "exposure" in inputs:
            exposure = inputs["exposure"]
            bbox = exposure.getBBox()
            wcs = exposure.getWcs()
            if wcs is None:
                raise NoWorkFound("Exposure has no valid WCS.")
        else:
            exposure = None

        if "diaExposure" in inputs:
            diaExposure = inputs["diaExposure"]
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

        tract = butlerQC.quantum.dataId["tract"]
        skyMap = inputs.pop("skyMap")
        refWcs = skyMap[tract].getWcs()

        self.log.info("Filtering ref cats: %s", ','.join([str(i.dataId) for i in inputs["refCat"]]))

        refTable = self._filterRefTable(
            inputs["refCat"],
            bbox,
            wcs,
        )
        # Convert the table into a SourceCatalog.
        # This is necessary because the forced measurement subtask expects a
        # SourceCatalog as input.
        refCat = self._makeMinimalSourceCatalogFromAstropy(refTable)

        # Create the catalogs to hold measurements
        if self.config.doDirectPhotometry:
            id_generator = self.config.idGenerator.apply(inputRefs.exposure.dataId)
            directCat = self._generateMeasCat(refCat, idFactory=id_generator.make_table_id_factory())
        else:
            directCat = None
        if self.config.doDifferencePhotometry:
            id_generator = self.config.idGenerator.apply(inputRefs.diaExposure.dataId)
            diffCat = self._generateMeasCat(refCat, idFactory=id_generator.make_table_id_factory())
        else:
            diffCat = None

        outputs = self.run(
            refCat=refCat,
            objectIds=refTable[self.config.refCatIdColumn],
            visit=butlerQC.quantum.dataId["visit"],
            detector=butlerQC.quantum.dataId["detector"],
            refWcs=refWcs,
            directCat=directCat,
            diffCat=diffCat,
            exposure=exposure,
            diaExposure=diaExposure,
            band=butlerQC.quantum.dataId["band"],
        )

        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        refCat: lsst.afw.table.SourceCatalog,
        objectIds: np.ndarray,
        visit: int,
        detector: int,
        refWcs: lsst.afw.geom.SkyWcs,
        directCat: lsst.afw.table.SourceCatalog | None = None,
        diffCat: lsst.afw.table.SourceCatalog | None = None,
        exposure: lsst.afw.image.Exposure | None = None,
        diaExposure: lsst.afw.image.Exposure | None = None,
        band: str | None = None,
    ) -> pipeBase.Struct:
        """Run forced photometry on a single detector.

        There is a lot of prep work in the `runQuantum` method and it is
        expected that this taks is usually run as a pipeline task, not
        executed as a stand alone function.

        Parameters
        ----------
        refCat :
            Reference catalog for forced photometry.
        objectIds :
            Array of object IDs corresponding to the reference catalog.
        visit :
            Visit ID for the observation.
        detector :
            Detector ID for the observation.
        refWcs :
            Reference WCS for the observation.
        directCat :
            Catalog for direct photometry.
            Only required when `config.doDirectPhotometry` is True.
        diffCat :
            Catalog for difference photometry.
            Only required when `config.doDifferencePhotometry` is True.
        exposure :
            Exposure for direct photometry.
            Only required when `config.doDirectPhotometry` is True.
        diaExposure :
            Exposure for difference photometry.
            Only required when `config.doDifferencePhotometry` is True.
        band :
            Band for the observation.
        """
        results: dict[str, lsst.afw.table.SourceCatalog] = {}
        if self.config.doDirectPhotometry:
            if exposure is None:
                raise ValueError("`exposure` must be provided for direct photometry.")
            if directCat is None:
                raise ValueError("`directCat` must be provided for direct photometry.")
            self.log.info("Running forced measurement on %s objects", len(refCat))
            self._runForcedPhotometry(refCat, directCat, exposure, refWcs)
            results["calexp"] = directCat

        if self.config.doDifferencePhotometry:
            if diaExposure is None:
                raise ValueError("`diaExposure` must be provided for difference photometry.")
            if diffCat is None:
                raise ValueError("`diffCat` must be provided for difference photometry.")
            self.log.info("Running forced measurement on %s objects on difference image", len(refCat))
            self._runForcedPhotometry(refCat, diffCat, diaExposure, refWcs)
            results["diff"] = diffCat

        # Convert the astropy tables to pandas DataFrames and reindex them
        dfs = []
        for dataset, catalog in results.items():
            measTbl = catalog.asAstropy()
            measTbl[self.config.refCatIdColumn] = objectIds
            df = measTbl.to_pandas().set_index(self.config.refCatIdColumn, drop=False)
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
        refCat: lsst.afw.table.SourceCatalog,
        measCat: lsst.afw.table.SourceCatalog,
        exposure: lsst.afw.image.Exposure,
        refWcs: lsst.afw.geom.SkyWcs,
    ) -> None:
        """Perform forced measurement on a single exposure.

        Parameters
        ----------
        refCat : `lsst.afw.table.SourceCatalog`
            Catalog containing the reference catalog data, with columns
            for the object ID, right ascension, and declination.
        measCat : `lsst.afw.table.SourceCatalog`
            Catalog containing the measurement data, with columns for the
            object ID and measured quantities.
            This catalog is updated in-place.
        exposure : `lsst.afw.image.exposure.Exposure`
            Input exposure to adjust calibrations.
        refWcs : `lsst.afw.geom.SkyWcs`
            Defines the X,Y coordinate system of ``refCat``.
        """
        self.measurement.run(
            refCat=refCat,
            measCat=measCat,
            exposure=exposure,
            refWcs=refWcs,
        )
        if self.config.doApCorr:
            apCorrMap = exposure.getInfo().getApCorrMap()
            if apCorrMap is None:
                self.log.warning("Forced exposure image does not have valid aperture correction; skipping.")
            else:
                self.applyApCorr.run(
                    catalog=measCat,
                    apCorrMap=apCorrMap,
                )

    def _makeMinimalSourceCatalogFromAstropy(self, table):
        """Create minimal schema SourceCatalog from an Astropy Table.

        The forced measurement subtask expects this as input.

        Parameters
        ----------
        table : `astropy.table.Table`
            Table with locations and ids.

        Returns
        -------
        outputCatalog : `lsst.afw.table.SourceTable`
            Output catalog with minimal schema.
        """
        schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        outputCatalog = lsst.afw.table.SourceCatalog(schema)
        outputCatalog.resize(len(table))
        outputCatalog["id"] = table[self.config.refCatIdColumn]
        outputCatalog[outputCatalog.getCoordKey().getRa()] = np.deg2rad(table[self.config.refCatRaColumn])
        outputCatalog[outputCatalog.getCoordKey().getDec()] = np.deg2rad(table[self.config.refCatDecColumn])
        return outputCatalog

    def _filterRefTable(self, refTableHandles, exposureBBox, exposureWcs):
        """Prepare a merged, filtered reference catalog from ArrowAstropy
        inputs.

        Parameters
        ----------
        refTableHandles : sequence of `lsst.daf.butler.DeferredDatasetHandle`
            Handles for catalogs of shapes and positions at which to force
            photometry.
        exposureBBox :   `lsst.geom.Box2I`
            Bounding box on which to select rows that overlap
        exposureWcs : `lsst.afw.geom.SkyWcs`
            World coordinate system to convert sky coords in ref cat to
            pixel coords with which to compare with exposureBBox

        Returns
        -------
        refTable : `astropy.table.Table`
            Astropy Table with only rows from the reference
            catalogs that overlap the exposure bounding box.
        """
        table_list = []
        for i in refTableHandles:
            try:
                table_list.append(i.get(
                    parameters={
                        "columns": [
                            self.config.refCatIdColumn,
                            self.config.refCatRaColumn,
                            self.config.refCatDecColumn,
                        ]
                    }
                ))
            except ValueError:
                self.log.info("Skipping %s due to empty object table.", i.dataId)
                continue
        if not table_list:
            raise NoWorkFound("All overlapping object catalogs are empty.")
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

    def _generateMeasCat(
        self,
        refCat: lsst.afw.table.SourceCatalog,
        idFactory: lsst.afw.table.IdFactory | None = None,
    ) -> lsst.afw.table.SourceCatalog:
        r"""Initialize an output catalog from the reference catalog.

        Parameters
        ----------
        refCat : `lsst.afw.table.SourceCatalog,`
            Catalog of reference sources.
        idFactory : `lsst.afw.table.IdFactory`, optional
            Factory for creating IDs for sources.

        Returns
        -------
        meascat : `lsst.afw.table.SourceCatalog`
            Source catalog ready for measurement.

        Notes
        -----
        This generates a new blank `~lsst.afw.table.SourceRecord` for each
        record in ``refCat``. Note that this method does not attach any
        `~lsst.afw.detection.Footprint`\ s, which is done in
        ``config.measure``.
        """
        if idFactory is None:
            idFactory = lsst.afw.table.IdFactory.makeSimple()
        table = lsst.afw.table.SourceTable.make(self.measurement.schema, idFactory)
        measCat = lsst.afw.table.SourceCatalog(table)
        table = measCat.table
        table.setMetadata(self.measurement.algMetadata)
        table.preallocate(len(refCat))
        for ref in refCat:
            newSource = measCat.addNew()
            newSource.assign(ref, self.measurement.mapper)
        return measCat
