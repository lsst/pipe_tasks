#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008-2015 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
import warnings

from lsst.pipe.base import (Struct, PipelineTask, PipelineTaskConfig, PipelineTaskConnections)
import lsst.pipe.base.connectionTypes as cT
from lsst.pex.config import Config, Field, ConfigurableField, ChoiceField
from lsst.meas.algorithms import DynamicDetectionTask, ReferenceObjectLoader, ScaleVarianceTask
from lsst.meas.base import SingleFrameMeasurementTask, ApplyApCorrTask, CatalogCalculationTask
from lsst.meas.deblender import SourceDeblendTask
from lsst.meas.extensions.scarlet import ScarletDeblendTask
from lsst.meas.astrom import DirectMatchTask, denormalizeMatches
from lsst.pipe.tasks.fakes import BaseFakeSourcesTask
from lsst.pipe.tasks.setPrimaryFlags import SetPrimaryFlagsTask
from lsst.pipe.tasks.propagateSourceFlags import PropagateSourceFlagsTask
import lsst.afw.table as afwTable
import lsst.afw.math as afwMath
from lsst.daf.base import PropertyList
from lsst.skymap import BaseSkyMap
from lsst.obs.base import ExposureIdInfo

# NOTE: these imports are a convenience so multiband users only have to import this file.
from .mergeDetections import MergeDetectionsConfig, MergeDetectionsTask  # noqa: F401
from .mergeMeasurements import MergeMeasurementsConfig, MergeMeasurementsTask  # noqa: F401
from .multiBandUtils import CullPeaksConfig, _makeGetSchemaCatalogs  # noqa: F401
from .deblendCoaddSourcesPipeline import DeblendCoaddSourcesSingleConfig  # noqa: F401
from .deblendCoaddSourcesPipeline import DeblendCoaddSourcesSingleTask  # noqa: F401
from .deblendCoaddSourcesPipeline import DeblendCoaddSourcesMultiConfig  # noqa: F401
from .deblendCoaddSourcesPipeline import DeblendCoaddSourcesMultiTask  # noqa: F401


"""
New set types:
* deepCoadd_det: detections from what used to be processCoadd (tract, patch, filter)
* deepCoadd_mergeDet: merged detections (tract, patch)
* deepCoadd_meas: measurements of merged detections (tract, patch, filter)
* deepCoadd_ref: reference sources (tract, patch)
All of these have associated *_schema catalogs that require no data ID and hold no records.

In addition, we have a schema-only dataset, which saves the schema for the PeakRecords in
the mergeDet, meas, and ref dataset Footprints:
* deepCoadd_peak_schema
"""


##############################################################################################################
class DetectCoaddSourcesConnections(PipelineTaskConnections,
                                    dimensions=("tract", "patch", "band", "skymap"),
                                    defaultTemplates={"inputCoaddName": "deep", "outputCoaddName": "deep"}):
    detectionSchema = cT.InitOutput(
        doc="Schema of the detection catalog",
        name="{outputCoaddName}Coadd_det_schema",
        storageClass="SourceCatalog",
    )
    exposure = cT.Input(
        doc="Exposure on which detections are to be performed",
        name="{inputCoaddName}Coadd",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "band", "skymap")
    )
    outputBackgrounds = cT.Output(
        doc="Output Backgrounds used in detection",
        name="{outputCoaddName}Coadd_calexp_background",
        storageClass="Background",
        dimensions=("tract", "patch", "band", "skymap")
    )
    outputSources = cT.Output(
        doc="Detected sources catalog",
        name="{outputCoaddName}Coadd_det",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "band", "skymap")
    )
    outputExposure = cT.Output(
        doc="Exposure post detection",
        name="{outputCoaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "band", "skymap")
    )


class DetectCoaddSourcesConfig(PipelineTaskConfig, pipelineConnections=DetectCoaddSourcesConnections):
    """!
    @anchor DetectCoaddSourcesConfig_

    @brief Configuration parameters for the DetectCoaddSourcesTask
    """
    doScaleVariance = Field(dtype=bool, default=True, doc="Scale variance plane using empirical noise?")
    scaleVariance = ConfigurableField(target=ScaleVarianceTask, doc="Variance rescaling")
    detection = ConfigurableField(target=DynamicDetectionTask, doc="Source detection")
    coaddName = Field(dtype=str, default="deep", doc="Name of coadd")
    doInsertFakes = Field(dtype=bool, default=False,
                          doc="Run fake sources injection task",
                          deprecated=("doInsertFakes is no longer supported. This config will be removed "
                                      "after v24."))
    insertFakes = ConfigurableField(target=BaseFakeSourcesTask,
                                    doc="Injection of fake sources for testing "
                                    "purposes (must be retargeted)",
                                    deprecated=("insertFakes is no longer supported. This config will "
                                                "be removed after v24."))
    hasFakes = Field(
        dtype=bool,
        default=False,
        doc="Should be set to True if fake sources have been inserted into the input data.",
    )

    def setDefaults(self):
        super().setDefaults()
        self.detection.thresholdType = "pixel_stdev"
        self.detection.isotropicGrow = True
        # Coadds are made from background-subtracted CCDs, so any background subtraction should be very basic
        self.detection.reEstimateBackground = False
        self.detection.background.useApprox = False
        self.detection.background.binSize = 4096
        self.detection.background.undersampleStyle = 'REDUCE_INTERP_ORDER'
        self.detection.doTempWideBackground = True  # Suppress large footprints that overwhelm the deblender

## @addtogroup LSST_task_documentation
## @{
## @page page_DetectCoaddSourcesTask DetectCoaddSourcesTask
## @ref DetectCoaddSourcesTask_ "DetectCoaddSourcesTask"
## @copybrief DetectCoaddSourcesTask
## @}


class DetectCoaddSourcesTask(PipelineTask):
    """Detect sources on a coadd."""
    _DefaultName = "detectCoaddSources"
    ConfigClass = DetectCoaddSourcesConfig
    getSchemaCatalogs = _makeGetSchemaCatalogs("det")

    def __init__(self, schema=None, **kwargs):
        """!
        @brief Initialize the task. Create the @ref SourceDetectionTask_ "detection" subtask.

        Keyword arguments (in addition to those forwarded to PipelineTask.__init__):

        @param[in] schema:   initial schema for the output catalog, modified-in place to include all
                             fields set by this task.  If None, the source minimal schema will be used.
        @param[in] **kwargs: keyword arguments to be passed to lsst.pipe.base.task.Task.__init__
        """
        # N.B. Super is used here to handle the multiple inheritance of PipelineTasks, the init tree
        # call structure has been reviewed carefully to be sure super will work as intended.
        super().__init__(**kwargs)
        if schema is None:
            schema = afwTable.SourceTable.makeMinimalSchema()
        self.schema = schema
        self.makeSubtask("detection", schema=self.schema)
        if self.config.doScaleVariance:
            self.makeSubtask("scaleVariance")

        self.detectionSchema = afwTable.SourceCatalog(self.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        exposureIdInfo = ExposureIdInfo.fromDataId(butlerQC.quantum.dataId, "tract_patch_band")
        inputs["idFactory"] = exposureIdInfo.makeSourceIdFactory()
        inputs["expId"] = exposureIdInfo.expId
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, exposure, idFactory, expId):
        """!
        @brief Run detection on an exposure.

        First scale the variance plane to match the observed variance
        using @ref ScaleVarianceTask. Then invoke the @ref SourceDetectionTask_ "detection" subtask to
        detect sources.

        @param[in,out] exposure: Exposure on which to detect (may be backround-subtracted and scaled,
                                 depending on configuration).
        @param[in] idFactory: IdFactory to set source identifiers
        @param[in] expId: Exposure identifier (integer) for RNG seed

        @return a pipe.base.Struct with fields
        - sources: catalog of detections
        - backgrounds: list of backgrounds
        """
        if self.config.doScaleVariance:
            varScale = self.scaleVariance.run(exposure.maskedImage)
            exposure.getMetadata().add("VARIANCE_SCALE", varScale)
        backgrounds = afwMath.BackgroundList()
        table = afwTable.SourceTable.make(self.schema, idFactory)
        detections = self.detection.run(table, exposure, expId=expId)
        sources = detections.sources
        fpSets = detections.fpSets
        if hasattr(fpSets, "background") and fpSets.background:
            for bg in fpSets.background:
                backgrounds.append(bg)
        return Struct(outputSources=sources, outputBackgrounds=backgrounds, outputExposure=exposure)


##############################################################################################################


class DeblendCoaddSourcesConfig(Config):
    """DeblendCoaddSourcesConfig

    Configuration parameters for the `DeblendCoaddSourcesTask`.
    """
    singleBandDeblend = ConfigurableField(target=SourceDeblendTask,
                                          doc="Deblend sources separately in each band")
    multiBandDeblend = ConfigurableField(target=ScarletDeblendTask,
                                         doc="Deblend sources simultaneously across bands")
    simultaneous = Field(dtype=bool,
                         default=True,
                         doc="Simultaneously deblend all bands? "
                             "True uses `multibandDeblend` while False uses `singleBandDeblend`")
    coaddName = Field(dtype=str, default="deep", doc="Name of coadd")
    hasFakes = Field(dtype=bool,
                     default=False,
                     doc="Should be set to True if fake sources have been inserted into the input data.")

    def setDefaults(self):
        Config.setDefaults(self)
        self.singleBandDeblend.propagateAllPeaks = True


class MeasureMergedCoaddSourcesConnections(PipelineTaskConnections,
                                           dimensions=("tract", "patch", "band", "skymap"),
                                           defaultTemplates={"inputCoaddName": "deep",
                                                             "outputCoaddName": "deep",
                                                             "deblendedCatalog": "deblendedFlux"}):
    warnings.warn("MeasureMergedCoaddSourcesConnections.defaultTemplates is deprecated and no longer used. "
                  "Use MeasureMergedCoaddSourcesConfig.inputCatalog.")
    inputSchema = cT.InitInput(
        doc="Input schema for measure merged task produced by a deblender or detection task",
        name="{inputCoaddName}Coadd_deblendedFlux_schema",
        storageClass="SourceCatalog"
    )
    outputSchema = cT.InitOutput(
        doc="Output schema after all new fields are added by task",
        name="{inputCoaddName}Coadd_meas_schema",
        storageClass="SourceCatalog"
    )
    refCat = cT.PrerequisiteInput(
        doc="Reference catalog used to match measured sources against known sources",
        name="ref_cat",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        deferLoad=True,
        multiple=True
    )
    exposure = cT.Input(
        doc="Input coadd image",
        name="{inputCoaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "band", "skymap")
    )
    skyMap = cT.Input(
        doc="SkyMap to use in processing",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    visitCatalogs = cT.Input(
        doc="Source catalogs for visits which overlap input tract, patch, band. Will be "
            "further filtered in the task for the purpose of propagating flags from image calibration "
            "and characterization to coadd objects. Only used in legacy PropagateVisitFlagsTask.",
        name="src",
        dimensions=("instrument", "visit", "detector"),
        storageClass="SourceCatalog",
        multiple=True
    )
    sourceTableHandles = cT.Input(
        doc=("Source tables that are derived from the ``CalibrateTask`` sources. "
             "These tables contain astrometry and photometry flags, and optionally "
             "PSF flags."),
        name="sourceTable_visit",
        storageClass="DataFrame",
        dimensions=("instrument", "visit"),
        multiple=True,
        deferLoad=True,
    )
    finalizedSourceTableHandles = cT.Input(
        doc=("Finalized source tables from ``FinalizeCalibrationTask``. These "
             "tables contain PSF flags from the finalized PSF estimation."),
        name="finalized_src_table",
        storageClass="DataFrame",
        dimensions=("instrument", "visit"),
        multiple=True,
        deferLoad=True,
    )
    inputCatalog = cT.Input(
        doc=("Name of the input catalog to use."
             "If the single band deblender was used this should be 'deblendedFlux."
             "If the multi-band deblender was used this should be 'deblendedModel, "
             "or deblendedFlux if the multiband deblender was configured to output "
             "deblended flux catalogs. If no deblending was performed this should "
             "be 'mergeDet'"),
        name="{inputCoaddName}Coadd_{deblendedCatalog}",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "band", "skymap"),
    )
    scarletCatalog = cT.Input(
        doc="Catalogs produced by multiband deblending",
        name="{inputCoaddName}Coadd_deblendedCatalog",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap"),
    )
    scarletModels = cT.Input(
        doc="Multiband scarlet models produced by the deblender",
        name="{inputCoaddName}Coadd_scarletModelData",
        storageClass="ScarletModelData",
        dimensions=("tract", "patch", "skymap"),
    )
    outputSources = cT.Output(
        doc="Source catalog containing all the measurement information generated in this task",
        name="{outputCoaddName}Coadd_meas",
        dimensions=("tract", "patch", "band", "skymap"),
        storageClass="SourceCatalog",
    )
    matchResult = cT.Output(
        doc="Match catalog produced by configured matcher, optional on doMatchSources",
        name="{outputCoaddName}Coadd_measMatch",
        dimensions=("tract", "patch", "band", "skymap"),
        storageClass="Catalog",
    )
    denormMatches = cT.Output(
        doc="Denormalized Match catalog produced by configured matcher, optional on "
            "doWriteMatchesDenormalized",
        name="{outputCoaddName}Coadd_measMatchFull",
        dimensions=("tract", "patch", "band", "skymap"),
        storageClass="Catalog",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if config.doPropagateFlags is False:
            self.inputs -= set(("visitCatalogs",))
            self.inputs -= set(("sourceTableHandles",))
            self.inputs -= set(("finalizedSourceTableHandles",))
        elif config.propagateFlags.target == PropagateSourceFlagsTask:
            # New PropagateSourceFlagsTask does not use visitCatalogs.
            self.inputs -= set(("visitCatalogs",))
            # Check for types of flags required.
            if not config.propagateFlags.source_flags:
                self.inputs -= set(("sourceTableHandles",))
            if not config.propagateFlags.finalized_source_flags:
                self.inputs -= set(("finalizedSourceTableHandles",))
        else:
            # Deprecated PropagateVisitFlagsTask uses visitCatalogs.
            self.inputs -= set(("sourceTableHandles",))
            self.inputs -= set(("finalizedSourceTableHandles",))

        if config.inputCatalog == "deblendedCatalog":
            self.inputs -= set(("inputCatalog",))

            if not config.doAddFootprints:
                self.inputs -= set(("scarletModels",))
        else:
            self.inputs -= set(("deblendedCatalog"))
            self.inputs -= set(("scarletModels",))

        if config.doMatchSources is False:
            self.outputs -= set(("matchResult",))

        if config.doWriteMatchesDenormalized is False:
            self.outputs -= set(("denormMatches",))


class MeasureMergedCoaddSourcesConfig(PipelineTaskConfig,
                                      pipelineConnections=MeasureMergedCoaddSourcesConnections):
    """!
    @anchor MeasureMergedCoaddSourcesConfig_

    @brief Configuration parameters for the MeasureMergedCoaddSourcesTask
    """
    inputCatalog = ChoiceField(
        dtype=str,
        default="deblendedCatalog",
        allowed={
            "deblendedCatalog": "Output catalog from ScarletDeblendTask",
            "deblendedFlux": "Output catalog from SourceDeblendTask",
            "mergeDet": "The merged detections before deblending."
        },
        doc="The name of the input catalog.",
    )
    doAddFootprints = Field(dtype=bool,
                            default=True,
                            doc="Whether or not to add footprints to the input catalog from scarlet models. "
                                "This should be true whenever using the multi-band deblender, "
                                "otherwise this should be False.")
    doConserveFlux = Field(dtype=bool, default=True,
                           doc="Whether to use the deblender models as templates to re-distribute the flux "
                               "from the 'exposure' (True), or to perform measurements on the deblender "
                               "model footprints.")
    doStripFootprints = Field(dtype=bool, default=True,
                              doc="Whether to strip footprints from the output catalog before "
                                  "saving to disk. "
                                  "This is usually done when using scarlet models to save disk space.")
    measurement = ConfigurableField(target=SingleFrameMeasurementTask, doc="Source measurement")
    setPrimaryFlags = ConfigurableField(target=SetPrimaryFlagsTask, doc="Set flags for primary tract/patch")
    doPropagateFlags = Field(
        dtype=bool, default=True,
        doc="Whether to match sources to CCD catalogs to propagate flags (to e.g. identify PSF stars)"
    )
    propagateFlags = ConfigurableField(target=PropagateSourceFlagsTask, doc="Propagate source flags to coadd")
    doMatchSources = Field(dtype=bool, default=True, doc="Match sources to reference catalog?")
    match = ConfigurableField(target=DirectMatchTask, doc="Matching to reference catalog")
    doWriteMatchesDenormalized = Field(
        dtype=bool,
        default=False,
        doc=("Write reference matches in denormalized format? "
             "This format uses more disk space, but is more convenient to read."),
    )
    coaddName = Field(dtype=str, default="deep", doc="Name of coadd")
    psfCache = Field(dtype=int, default=100, doc="Size of psfCache")
    checkUnitsParseStrict = Field(
        doc="Strictness of Astropy unit compatibility check, can be 'raise', 'warn' or 'silent'",
        dtype=str,
        default="raise",
    )
    doApCorr = Field(
        dtype=bool,
        default=True,
        doc="Apply aperture corrections"
    )
    applyApCorr = ConfigurableField(
        target=ApplyApCorrTask,
        doc="Subtask to apply aperture corrections"
    )
    doRunCatalogCalculation = Field(
        dtype=bool,
        default=True,
        doc='Run catalogCalculation task'
    )
    catalogCalculation = ConfigurableField(
        target=CatalogCalculationTask,
        doc="Subtask to run catalogCalculation plugins on catalog"
    )

    hasFakes = Field(
        dtype=bool,
        default=False,
        doc="Should be set to True if fake sources have been inserted into the input data."
    )

    @property
    def refObjLoader(self):
        return self.match.refObjLoader

    def setDefaults(self):
        super().setDefaults()
        self.measurement.plugins.names |= ['base_InputCount',
                                           'base_Variance',
                                           'base_LocalPhotoCalib',
                                           'base_LocalWcs']
        self.measurement.plugins['base_PixelFlags'].masksFpAnywhere = ['CLIPPED', 'SENSOR_EDGE',
                                                                       'INEXACT_PSF']
        self.measurement.plugins['base_PixelFlags'].masksFpCenter = ['CLIPPED', 'SENSOR_EDGE',
                                                                     'INEXACT_PSF']


## @addtogroup LSST_task_documentation
## @{
## @page page_MeasureMergedCoaddSourcesTask MeasureMergedCoaddSourcesTask
## @ref MeasureMergedCoaddSourcesTask_ "MeasureMergedCoaddSourcesTask"
## @copybrief MeasureMergedCoaddSourcesTask
## @}

class MeasureMergedCoaddSourcesTask(PipelineTask):
    """Deblend sources from main catalog in each coadd seperately and measure."""
    _DefaultName = "measureCoaddSources"
    ConfigClass = MeasureMergedCoaddSourcesConfig
    getSchemaCatalogs = _makeGetSchemaCatalogs("meas")

    def __init__(self, butler=None, schema=None, peakSchema=None, refObjLoader=None, initInputs=None,
                 **kwargs):
        """!
        @brief Initialize the task.

        Keyword arguments (in addition to those forwarded to PipelineTask.__init__):
        @param[in] schema: the schema of the merged detection catalog used as input to this one
        @param[in] peakSchema: the schema of the PeakRecords in the Footprints in the merged detection catalog
        @param[in] refObjLoader: an instance of LoadReferenceObjectsTasks that supplies an external reference
            catalog. May be None if the loader can be constructed from the butler argument or all steps
            requiring a reference catalog are disabled.
        @param[in] butler: a butler used to read the input schemas from disk or construct the reference
            catalog loader, if schema or peakSchema or refObjLoader is None

        The task will set its own self.schema attribute to the schema of the output measurement catalog.
        This will include all fields from the input schema, as well as additional fields for all the
        measurements.
        """
        super().__init__(**kwargs)
        self.deblended = self.config.inputCatalog.startswith("deblended")
        self.inputCatalog = "Coadd_" + self.config.inputCatalog
        if initInputs is not None:
            schema = initInputs['inputSchema'].schema
        if schema is None:
            assert butler is not None, "Neither butler nor schema is defined"
            schema = butler.get(self.config.coaddName + self.inputCatalog + "_schema", immediate=True).schema
        self.schemaMapper = afwTable.SchemaMapper(schema)
        self.schemaMapper.addMinimalSchema(schema)
        self.schema = self.schemaMapper.getOutputSchema()
        self.algMetadata = PropertyList()
        self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)
        self.makeSubtask("setPrimaryFlags", schema=self.schema)
        if self.config.doMatchSources:
            self.makeSubtask("match", butler=butler, refObjLoader=refObjLoader)
        if self.config.doPropagateFlags:
            self.makeSubtask("propagateFlags", schema=self.schema)
        self.schema.checkUnits(parse_strict=self.config.checkUnitsParseStrict)
        if self.config.doApCorr:
            self.makeSubtask("applyApCorr", schema=self.schema)
        if self.config.doRunCatalogCalculation:
            self.makeSubtask("catalogCalculation", schema=self.schema)

        self.outputSchema = afwTable.SourceCatalog(self.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        refObjLoader = ReferenceObjectLoader([ref.datasetRef.dataId for ref in inputRefs.refCat],
                                             inputs.pop('refCat'),
                                             name=self.config.connections.refCat,
                                             config=self.config.refObjLoader,
                                             log=self.log)
        self.match.setRefObjLoader(refObjLoader)

        # Set psfcache
        # move this to run after gen2 deprecation
        inputs['exposure'].getPsf().setCacheCapacity(self.config.psfCache)

        # Get unique integer ID for IdFactory and RNG seeds
        exposureIdInfo = ExposureIdInfo.fromDataId(butlerQC.quantum.dataId, "tract_patch")
        inputs['exposureId'] = exposureIdInfo.expId
        idFactory = exposureIdInfo.makeSourceIdFactory()
        # Transform inputCatalog
        table = afwTable.SourceTable.make(self.schema, idFactory)
        sources = afwTable.SourceCatalog(table)
        # Load the correct input catalog
        if "scarletCatalog" in inputs:
            inputCatalog = inputs.pop("scarletCatalog")
            catalogRef = inputRefs.scarletCatalog
        else:
            inputCatalog = inputs.pop("inputCatalog")
            catalogRef = inputRefs.inputCatalog
        sources.extend(inputCatalog, self.schemaMapper)
        del inputCatalog
        # Add the HeavyFootprints to the deblended sources
        if self.config.doAddFootprints:
            modelData = inputs.pop('scarletModels')
            if self.config.doConserveFlux:
                redistributeImage = inputs['exposure'].image
            else:
                redistributeImage = None
            modelData.updateCatalogFootprints(
                catalog=sources,
                band=inputRefs.exposure.dataId["band"],
                psfModel=inputs['exposure'].getPsf(),
                redistributeImage=redistributeImage,
                removeScarletData=True,
            )
        table = sources.getTable()
        table.setMetadata(self.algMetadata)  # Capture algorithm metadata to write out to the source catalog.
        inputs['sources'] = sources

        skyMap = inputs.pop('skyMap')
        tractNumber = catalogRef.dataId['tract']
        tractInfo = skyMap[tractNumber]
        patchInfo = tractInfo.getPatchInfo(catalogRef.dataId['patch'])
        skyInfo = Struct(
            skyMap=skyMap,
            tractInfo=tractInfo,
            patchInfo=patchInfo,
            wcs=tractInfo.getWcs(),
            bbox=patchInfo.getOuterBBox()
        )
        inputs['skyInfo'] = skyInfo

        if self.config.doPropagateFlags:
            if self.config.propagateFlags.target == PropagateSourceFlagsTask:
                # New version
                ccdInputs = inputs["exposure"].getInfo().getCoaddInputs().ccds
                inputs["ccdInputs"] = ccdInputs

                if "sourceTableHandles" in inputs:
                    sourceTableHandles = inputs.pop("sourceTableHandles")
                    sourceTableHandleDict = {handle.dataId["visit"]: handle
                                             for handle in sourceTableHandles}
                    inputs["sourceTableHandleDict"] = sourceTableHandleDict
                if "finalizedSourceTableHandles" in inputs:
                    finalizedSourceTableHandles = inputs.pop("finalizedSourceTableHandles")
                    finalizedSourceTableHandleDict = {handle.dataId["visit"]: handle
                                                      for handle in finalizedSourceTableHandles}
                    inputs["finalizedSourceTableHandleDict"] = finalizedSourceTableHandleDict
            else:
                # Deprecated legacy version
                # Filter out any visit catalog that is not coadd inputs
                ccdInputs = inputs['exposure'].getInfo().getCoaddInputs().ccds
                visitKey = ccdInputs.schema.find("visit").key
                ccdKey = ccdInputs.schema.find("ccd").key
                inputVisitIds = set()
                ccdRecordsWcs = {}
                for ccdRecord in ccdInputs:
                    visit = ccdRecord.get(visitKey)
                    ccd = ccdRecord.get(ccdKey)
                    inputVisitIds.add((visit, ccd))
                    ccdRecordsWcs[(visit, ccd)] = ccdRecord.getWcs()

                inputCatalogsToKeep = []
                inputCatalogWcsUpdate = []
                for i, dataRef in enumerate(inputRefs.visitCatalogs):
                    key = (dataRef.dataId['visit'], dataRef.dataId['detector'])
                    if key in inputVisitIds:
                        inputCatalogsToKeep.append(inputs['visitCatalogs'][i])
                        inputCatalogWcsUpdate.append(ccdRecordsWcs[key])
                inputs['visitCatalogs'] = inputCatalogsToKeep
                inputs['wcsUpdates'] = inputCatalogWcsUpdate
                inputs['ccdInputs'] = ccdInputs

        outputs = self.run(**inputs)
        # Strip HeavyFootprints to save space on disk
        sources = outputs.outputSources
        butlerQC.put(outputs, outputRefs)

    def run(self, exposure, sources, skyInfo, exposureId, ccdInputs=None, visitCatalogs=None, wcsUpdates=None,
            butler=None, sourceTableHandleDict=None, finalizedSourceTableHandleDict=None):
        """Run measurement algorithms on the input exposure, and optionally populate the
        resulting catalog with extra information.

        Parameters
        ----------
        exposure : `lsst.afw.exposure.Exposure`
            The input exposure on which measurements are to be performed
        sources :  `lsst.afw.table.SourceCatalog`
            A catalog built from the results of merged detections, or
            deblender outputs.
        skyInfo : `lsst.pipe.base.Struct`
            A struct containing information about the position of the input exposure within
            a `SkyMap`, the `SkyMap`, its `Wcs`, and its bounding box
        exposureId : `int` or `bytes`
            packed unique number or bytes unique to the input exposure
        ccdInputs : `lsst.afw.table.ExposureCatalog`
            Catalog containing information on the individual visits which went into making
            the coadd.
        sourceTableHandleDict : `dict` [`int`: `lsst.daf.butler.DeferredDatasetHandle`]
            Dict for sourceTable_visit handles (key is visit) for propagating flags.
            These tables are derived from the ``CalibrateTask`` sources, and contain
            astrometry and photometry flags, and optionally PSF flags.
        finalizedSourceTableHandleDict : `dict` [`int`: `lsst.daf.butler.DeferredDatasetHandle`], optional
            Dict for finalized_src_table handles (key is visit) for propagating flags.
            These tables are derived from ``FinalizeCalibrationTask`` and contain
            PSF flags from the finalized PSF estimation.
        visitCatalogs : list of `lsst.afw.table.SourceCatalogs`
            A list of source catalogs corresponding to measurements made on the individual
            visits which went into the input exposure. If None and butler is `None` then
            the task cannot propagate visit flags to the output catalog.
            Deprecated, to be removed with PropagateVisitFlagsTask.
        wcsUpdates : list of `lsst.afw.geom.SkyWcs`
            If visitCatalogs is not `None` this should be a list of wcs objects which correspond
            to the input visits. Used to put all coordinates to common system. If `None` and
            butler is `None` then the task cannot propagate visit flags to the output catalog.
            Deprecated, to be removed with PropagateVisitFlagsTask.
        butler : `None`
            This was a Gen2 butler used to load visit catalogs.
            No longer used and should not be set. Will be removed in the
            future.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Results of running measurement task. Will contain the catalog in the
            sources attribute. Optionally will have results of matching to a
            reference catalog in the matchResults attribute, and denormalized
            matches in the denormMatches attribute.
        """
        if butler is not None:
            warnings.warn("The 'butler' parameter is no longer used and can be safely removed.",
                          category=FutureWarning, stacklevel=2)
            butler = None

        self.measurement.run(sources, exposure, exposureId=exposureId)

        if self.config.doApCorr:
            self.applyApCorr.run(
                catalog=sources,
                apCorrMap=exposure.getInfo().getApCorrMap()
            )

        # TODO DM-11568: this contiguous check-and-copy could go away if we
        # reserve enough space during SourceDetection and/or SourceDeblend.
        # NOTE: sourceSelectors require contiguous catalogs, so ensure
        # contiguity now, so views are preserved from here on.
        if not sources.isContiguous():
            sources = sources.copy(deep=True)

        if self.config.doRunCatalogCalculation:
            self.catalogCalculation.run(sources)

        self.setPrimaryFlags.run(sources, skyMap=skyInfo.skyMap, tractInfo=skyInfo.tractInfo,
                                 patchInfo=skyInfo.patchInfo)
        if self.config.doPropagateFlags:
            if self.config.propagateFlags.target == PropagateSourceFlagsTask:
                # New version
                self.propagateFlags.run(
                    sources,
                    ccdInputs,
                    sourceTableHandleDict,
                    finalizedSourceTableHandleDict
                )
            else:
                # Legacy deprecated version
                self.propagateFlags.run(
                    butler,
                    sources,
                    ccdInputs,
                    exposure.getWcs(),
                    visitCatalogs,
                    wcsUpdates
                )

        results = Struct()

        if self.config.doMatchSources:
            matchResult = self.match.run(sources, exposure.getInfo().getFilter().bandLabel)
            matches = afwTable.packMatches(matchResult.matches)
            matches.table.setMetadata(matchResult.matchMeta)
            results.matchResult = matches
            if self.config.doWriteMatchesDenormalized:
                if matchResult.matches:
                    denormMatches = denormalizeMatches(matchResult.matches, matchResult.matchMeta)
                else:
                    self.log.warning("No matches, so generating dummy denormalized matches file")
                    denormMatches = afwTable.BaseCatalog(afwTable.Schema())
                    denormMatches.setMetadata(PropertyList())
                    denormMatches.getMetadata().add("COMMENT",
                                                    "This catalog is empty because no matches were found.")
                    results.denormMatches = denormMatches
                results.denormMatches = denormMatches

        results.outputSources = sources
        return results
