# This will go in meas_base


import collections
import logging
from lsst.meas.base.pluginRegistry import register
import lsst.pex.config
import lsst.pex.exceptions
import lsst.pipe.base
import lsst.geom
import lsst.afw.geom
import lsst.afw.image
import lsst.afw.table
import lsst.sphgeom

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


@register("ap_assoc_TransformedCentroid")
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


class ForcedPhotCcdOnParquetConnections(PipelineTaskConnections,
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
    exposure = cT.Input(
        doc="Input exposure to perform photometry on.",
        name="{inputName}",
        storageClass="ExposureF",
        dimensions=["instrument", "visit", "detector"],
    )
    measCat = cT.Output(
        doc="Output forced photometry catalog.",
        name="forced_src_new_name",
        storageClass="DataFrame",
        dimensions=["instrument", "visit", "detector", "skymap", "tract"],
    )


class ForcedPhotCcdOnParquetConfig(pipeBase.PipelineTaskConfig,
                                   pipelineConnections=ForcedPhotCcdOnParquetConnections):
    forcedMeasurement = lsst.pex.config.ConfigurableField(
        target=ForcedMeasurementTask,
        doc="Subtask to force photometer DiaObjects in the direct and "
            "difference images.",
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

    def setDefaults(self):
        self.forcedMeasurement.plugins = ["ap_assoc_TransformedCentroid",
                                          "base_PsfFlux"]
        self.forcedMeasurement.doReplaceWithNoise = False
        self.forcedMeasurement.copyColumns = {
            "id": "diaObjectId",
            "coord_ra": "coord_ra",
            "coord_dec": "coord_dec"}
        self.forcedMeasurement.slots.centroid = "ap_assoc_TransformedCentroid"
        self.forcedMeasurement.slots.psfFlux = "base_PsfFlux"
        self.forcedMeasurement.slots.shape = None
class ForcedPhotCcdOnParquetTask(pipeBase.PipelineTask):
    _DefaultName = "forcedPhotCcdOnParquet"
    ConfigClass = ForcedPhotCcdOnParquetConfig

    def __init__(self, butler=None, refSchema=None, initInputs=None, **kwds):
        import pdb; pdb.set_trace()
        # refSchema = afwTable.SourceTable.makeMinimalSchema()
        # super().__init__(refSchema=refSchema, **kwds)
        # copied and pasted init from super:
        pipeBase.PipelineTask.__init__(self, **kwds)
        # why doesn't this work:
        # super().__init__(**kwds)

        refSchema = afwTable.SourceTable.makeMinimalSchema()
        self.makeSubtask("forcedMeasurement", refSchema=refSchema)

        # self.makeSubtask("measurement", refSchema=refSchema)
        # It is necessary to get the schema internal to the forced measurement task until such a time
        # that the schema is not owned by the measurement task, but is passed in by an external caller
        if self.config.doApCorr:
            self.makeSubtask("applyApCorr", schema=self.forcedMeasurement.schema)
        #self.makeSubtask('catalogCalculation', schema=self.measurement.schema)
        self.outputSchema = lsst.afw.table.SourceCatalog(self.forcedMeasurement.schema)