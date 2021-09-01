# This will go in meas_base


import collections
import logging
import numpy as np
import pandas as pd
import lsst.pex.config
import lsst.pex.exceptions
import lsst.pipe.base
import lsst.geom
import lsst.afw.geom
import lsst.afw.image
import lsst.afw.table
import lsst.sphgeom
from lsst.meas.base.pluginRegistry import register
from lsst.pipe.base import PipelineTaskConnections
import lsst.pipe.base.connectionTypes as cT
import lsst.afw.table as afwTable
import lsst.pipe.base as pipeBase
from lsst.skymap import BaseSkyMap

from lsst.meas.base.references import MultiBandReferencesTask
from lsst.meas.base.forcedMeasurement import ForcedMeasurementTask
from lsst.meas.base.applyApCorr import ApplyApCorrTask
from lsst.meas.base.catalogCalculation import CatalogCalculationTask
from lsst.meas.base.forcedPhotCcd import ForcedPhotCcdConnections, ForcedPhotCcdConfig, ForcedPhotCcdTask
from lsst.meas.base import (
    ForcedMeasurementTask,
    ForcedTransformedCentroidConfig,
    ForcedTransformedCentroidPlugin)


class ForcedTransformedCentroidFromCoordConfig(ForcedTransformedCentroidConfig):
    """Configuration for the forced transformed coord algorithm.
    """
    pass


@register("base_TransformedCentroidFromCoord")
class ForcedTransformedCentroidFromCoordPlugin(ForcedTransformedCentroidPlugin):
    """Record the transformation of the reference catalog coord.
    The coord recorded in the reference catalog is tranformed to the
    measurement coordinate system and stored.
    Parameters
    ----------
    config : `ForcedTransformedCentroidFromCoordConfig`
        Plugin configuration
    name : `str`
        Plugin name
    schemaMapper : `lsst.afw.table.SchemaMapper`
        A mapping from reference catalog fields to output
        catalog fields. Output fields are added to the output schema.
    metadata : `lsst.daf.base.PropertySet`
        Plugin metadata that will be attached to the output catalog.
    Notes
    -----
    This can be used as the slot centroid in forced measurement when only a
    reference coord exits, allowing subsequent measurements to simply refer to
    the slot value just as they would in single-frame measurement.
    """

    ConfigClass = ForcedTransformedCentroidFromCoordConfig

    def measure(self, measRecord, exposure, refRecord, refWcs):
        targetWcs = exposure.getWcs()

        targetPos = targetWcs.skyToPixel(refRecord.getCoord())
        measRecord.set(self.centroidKey, targetPos)

        if self.flagKey is not None:
            measRecord.set(self.flagKey, refRecord.getCentroidFlag())


class ForcedPhotCcdOnParquetConnections(ForcedPhotCcdConnections,
                                        dimensions=("instrument", "visit", "detector", "skymap", "tract"),
                                        defaultTemplates={"inputCoaddName": "goodSeeing",
                                                          "inputName": "calexp"}):
    refCat = cT.Input(
        doc="Catalog of shapes and positions at which to force photometry.",
        name="{inputCoaddName}Diff_fullDiaObjTable",
        storageClass="DataFrame",
        dimensions=["skymap", "tract", "patch"],
        multiple=True,
        deferLoad=True,
    )
    measCat = cT.Output(
        doc="Output forced photometry catalog.",
        name="forced_src_new_name",
        storageClass="DataFrame",
        dimensions=["instrument", "visit", "detector", "skymap", "tract"],
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        self.initInputs.remove("inputSchema")


class ForcedPhotCcdOnParquetConfig(ForcedPhotCcdConfig,
                                   pipelineConnections=ForcedPhotCcdOnParquetConnections):
    def setDefaults(self):
        self.measurement.doReplaceWithNoise = False
        self.measurement.plugins = ["base_TransformedCentroidFromCoord", "base_PsfFlux"]
        self.measurement.copyColumns = {'id': 'diaObjectId', 'coord_ra': 'coord_ra', 'coord_dec': 'coord_dec'}
        self.measurement.slots.centroid = "base_TransformedCentroidFromCoord"
        self.measurement.slots.psfFlux = "base_PsfFlux"
        self.measurement.slots.shape = None
        self.catalogCalculation.plugins.names = []


class ForcedPhotCcdOnParquetTask(ForcedPhotCcdTask):
    _DefaultName = "forcedPhotCcdOnParquet"
    ConfigClass = ForcedPhotCcdOnParquetConfig

    def __init__(self, butler=None, refSchema=None, initInputs=None, **kwds):
        # refSchema = afwTable.SourceTable.makeMinimalSchema()
        # super().__init__(refSchema=refSchema, **kwds)
        # copied and pasted init from super:

        pipeBase.PipelineTask.__init__(self, **kwds)
        # why doesn't this work:
        #super().__init__(**kwds)
        refSchema = afwTable.SourceTable.makeMinimalSchema()

        self.makeSubtask("measurement", refSchema=refSchema)
        # It is necessary to get the schema internal to the forced measurement task until such a time
        # that the schema is not owned by the measurement task, but is passed in by an external caller
        if self.config.doApCorr:
            self.makeSubtask("applyApCorr", schema=self.measurement.schema)
        self.makeSubtask('catalogCalculation', schema=self.measurement.schema)
        self.outputSchema = lsst.afw.table.SourceCatalog(self.measurement.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):

        inputs = butlerQC.get(inputRefs)
        skyMap = inputs.pop("skyMap")
        refCat = self.df2refCat([i.get(parameters={"columns": ['diaObjectId', 'ra', 'decl']})
                                 for i in inputs['refCat']],
                                inputs['exposure'].getBBox(), inputs['exposure'].getWcs())
        inputs['refCat'] = refCat
        inputs['refWcs'] = inputs['exposure'].getWcs()
        inputs['measCat'], inputs['exposureId'] = self.generateMeasCat(inputRefs.exposure.dataId,
                                                                       inputs['exposure'], inputs['refCat'],
                                                                       inputs['refWcs'],
                                                                       "visit_detector")
        outputs = self.run(**inputs)
        outputs.measCat = outputs.measCat.asAstropy().to_pandas()
        butlerQC.put(outputs, outputRefs)

    def df2refCat(self, dfList, exposureBBox, exposureWcs):
        df = pd.concat(dfList)
        xy = exposureWcs.getTransform().getMapping().applyInverse(np.array(df[['ra', 'decl']].values*2*np.pi/360).T)
        inBBox = exposureBBox.contains(xy[0], xy[1])
        refCat = self.pandasToSourceCatalog(df[inBBox])
        return refCat

    def pandasToSourceCatalog(self, df):
        """Create minimal schema SourceCatalog from a pandas DataFrame.
        We need a catalog of this type to run within the forced measurement
        subtask.
        Parameters
        ----------
        df : `pandas.DataFrame`
            DiaObjects with locations and ids. ``
        Returns
        -------
        outputCatalog : `lsst.afw.table.SourceTable`
            Output catalog with minimal schema.
        """
        schema = afwTable.SourceTable.makeMinimalSchema()
        outputCatalog = afwTable.SourceCatalog(schema)
        outputCatalog.reserve(len(df))

        # Use diaObjectId instead of the row index:
        for id, diaObjectId, ra, decl in df[['diaObjectId', 'ra', 'decl']].itertuples():
            outputRecord = outputCatalog.addNew()
            outputRecord.setId(diaObjectId)
            outputRecord.setCoord(lsst.geom.SpherePoint(ra, decl, lsst.geom.degrees))
        return outputCatalog
