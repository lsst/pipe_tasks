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
import numpy as np

from lsst.coadd.utils.coaddDataIdContainer import ExistingCoaddDataIdContainer
from lsst.coadd.utils.getGen3CoaddExposureId import getGen3CoaddExposureId
from lsst.pipe.base import (CmdLineTask, Struct, ArgumentParser, ButlerInitializedTaskRunner,
                            PipelineTask, PipelineTaskConfig, PipelineTaskConnections)
import lsst.pipe.base.connectionTypes as cT
from lsst.pex.config import Config, Field, ConfigurableField
from lsst.meas.algorithms import DynamicDetectionTask, ReferenceObjectLoader
from lsst.meas.base import SingleFrameMeasurementTask, ApplyApCorrTask, CatalogCalculationTask
from lsst.meas.deblender import SourceDeblendTask
from lsst.meas.extensions.scarlet import ScarletDeblendTask
from lsst.pipe.tasks.coaddBase import getSkyInfo
from lsst.pipe.tasks.scaleVariance import ScaleVarianceTask
from lsst.meas.astrom import DirectMatchTask, denormalizeMatches
from lsst.pipe.tasks.fakes import BaseFakeSourcesTask
from lsst.pipe.tasks.setPrimaryFlags import SetPrimaryFlagsTask
from lsst.pipe.tasks.propagateVisitFlags import PropagateVisitFlagsTask
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.afw.math as afwMath
from lsst.daf.base import PropertyList
from lsst.skymap import BaseSkyMap
from lsst.obs.base import ExposureIdInfo

# NOTE: these imports are a convenience so multiband users only have to import this file.
from .mergeDetections import MergeDetectionsConfig, MergeDetectionsTask  # noqa: F401
from .mergeMeasurements import MergeMeasurementsConfig, MergeMeasurementsTask  # noqa: F401
from .multiBandUtils import MergeSourcesRunner, CullPeaksConfig, _makeGetSchemaCatalogs  # noqa: F401
from .multiBandUtils import getInputSchema, readCatalog, _makeMakeIdFactory  # noqa: F401
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
                          doc="Run fake sources injection task")
    insertFakes = ConfigurableField(target=BaseFakeSourcesTask,
                                    doc="Injection of fake sources for testing "
                                    "purposes (must be retargeted)")
    hasFakes = Field(
        dtype=bool,
        default=False,
        doc="Should be set to True if fake sources have been inserted into the input data."
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
## @page DetectCoaddSourcesTask
## @ref DetectCoaddSourcesTask_ "DetectCoaddSourcesTask"
## @copybrief DetectCoaddSourcesTask
## @}


class DetectCoaddSourcesTask(PipelineTask, CmdLineTask):
    r"""!
    @anchor DetectCoaddSourcesTask_

    @brief Detect sources on a coadd

    @section pipe_tasks_multiBand_Contents Contents

      - @ref pipe_tasks_multiBand_DetectCoaddSourcesTask_Purpose
      - @ref pipe_tasks_multiBand_DetectCoaddSourcesTask_Initialize
      - @ref pipe_tasks_multiBand_DetectCoaddSourcesTask_Run
      - @ref pipe_tasks_multiBand_DetectCoaddSourcesTask_Config
      - @ref pipe_tasks_multiBand_DetectCoaddSourcesTask_Debug
      - @ref pipe_tasks_multiband_DetectCoaddSourcesTask_Example

    @section pipe_tasks_multiBand_DetectCoaddSourcesTask_Purpose	Description

    Command-line task that detects sources on a coadd of exposures obtained with a single filter.

    Coadding individual visits requires each exposure to be warped. This introduces covariance in the noise
    properties across pixels. Before detection, we correct the coadd variance by scaling the variance plane
    in the coadd to match the observed variance. This is an approximate approach -- strictly, we should
    propagate the full covariance matrix -- but it is simple and works well in practice.

    After scaling the variance plane, we detect sources and generate footprints by delegating to the @ref
    SourceDetectionTask_ "detection" subtask.

      @par Inputs:
        deepCoadd{tract,patch,filter}: ExposureF
      @par Outputs:
        deepCoadd_det{tract,patch,filter}: SourceCatalog (only parent Footprints)
        @n deepCoadd_calexp{tract,patch,filter}: Variance scaled, background-subtracted input
                                                 exposure (ExposureF)
        @n deepCoadd_calexp_background{tract,patch,filter}: BackgroundList
      @par Data Unit:
        tract, patch, filter

    DetectCoaddSourcesTask delegates most of its work to the @ref SourceDetectionTask_ "detection" subtask.
    You can retarget this subtask if you wish.

    @section pipe_tasks_multiBand_DetectCoaddSourcesTask_Initialize       Task initialization

    @copydoc \_\_init\_\_

    @section pipe_tasks_multiBand_DetectCoaddSourcesTask_Run       Invoking the Task

    @copydoc run

    @section pipe_tasks_multiBand_DetectCoaddSourcesTask_Config       Configuration parameters

    See @ref DetectCoaddSourcesConfig_ "DetectSourcesConfig"

    @section pipe_tasks_multiBand_DetectCoaddSourcesTask_Debug		Debug variables

    The @link lsst.pipe.base.cmdLineTask.CmdLineTask command line task@endlink interface supports a
    flag @c -d to import @b debug.py from your @c PYTHONPATH; see @ref baseDebug for more about @b debug.py
    files.

    DetectCoaddSourcesTask has no debug variables of its own because it relegates all the work to
    @ref SourceDetectionTask_ "SourceDetectionTask"; see the documetation for
    @ref SourceDetectionTask_ "SourceDetectionTask" for further information.

    @section pipe_tasks_multiband_DetectCoaddSourcesTask_Example A complete example
    of using DetectCoaddSourcesTask

    DetectCoaddSourcesTask is meant to be run after assembling a coadded image in a given band. The purpose of
    the task is to update the background, detect all sources in a single band and generate a set of parent
    footprints. Subsequent tasks in the multi-band processing procedure will merge sources across bands and,
    eventually, perform forced photometry. Command-line usage of DetectCoaddSourcesTask expects a data
    reference to the coadd to be processed. A list of the available optional arguments can be obtained by
    calling detectCoaddSources.py with the `--help` command line argument:
    @code
    detectCoaddSources.py --help
    @endcode

    To demonstrate usage of the DetectCoaddSourcesTask in the larger context of multi-band processing, we
    will process HSC data in the [ci_hsc](https://github.com/lsst/ci_hsc) package. Assuming one has followed
    steps 1 - 4 at @ref pipeTasks_multiBand, one may detect all the sources in each coadd as follows:
    @code
    detectCoaddSources.py $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-I
    @endcode
    that will process the HSC-I band data. The results are written to
    `$CI_HSC_DIR/DATA/deepCoadd-results/HSC-I`.

    It is also necessary to run:
    @code
    detectCoaddSources.py $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-R
    @endcode
    to generate the sources catalogs for the HSC-R band required by the next step in the multi-band
    processing procedure: @ref MergeDetectionsTask_ "MergeDetectionsTask".
    """
    _DefaultName = "detectCoaddSources"
    ConfigClass = DetectCoaddSourcesConfig
    getSchemaCatalogs = _makeGetSchemaCatalogs("det")
    makeIdFactory = _makeMakeIdFactory("CoaddId")

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd", help="data ID, e.g. --id tract=12345 patch=1,2 filter=r",
                               ContainerClass=ExistingCoaddDataIdContainer)
        return parser

    def __init__(self, schema=None, **kwargs):
        """!
        @brief Initialize the task. Create the @ref SourceDetectionTask_ "detection" subtask.

        Keyword arguments (in addition to those forwarded to CmdLineTask.__init__):

        @param[in] schema:   initial schema for the output catalog, modified-in place to include all
                             fields set by this task.  If None, the source minimal schema will be used.
        @param[in] **kwargs: keyword arguments to be passed to lsst.pipe.base.task.Task.__init__
        """
        # N.B. Super is used here to handle the multiple inheritance of PipelineTasks, the init tree
        # call structure has been reviewed carefully to be sure super will work as intended.
        super().__init__(**kwargs)
        if schema is None:
            schema = afwTable.SourceTable.makeMinimalSchema()
        if self.config.doInsertFakes:
            self.makeSubtask("insertFakes")
        self.schema = schema
        self.makeSubtask("detection", schema=self.schema)
        if self.config.doScaleVariance:
            self.makeSubtask("scaleVariance")

        self.detectionSchema = afwTable.SourceCatalog(self.schema)

    def runDataRef(self, patchRef):
        """!
        @brief Run detection on a coadd.

        Invokes @ref run and then uses @ref write to output the
        results.

        @param[in] patchRef: data reference for patch
        """
        if self.config.hasFakes:
            exposure = patchRef.get("fakes_" + self.config.coaddName + "Coadd", immediate=True)
        else:
            exposure = patchRef.get(self.config.coaddName + "Coadd", immediate=True)
        expId = getGen3CoaddExposureId(patchRef, coaddName=self.config.coaddName, log=self.log)
        results = self.run(exposure, self.makeIdFactory(patchRef), expId=expId)
        self.write(results, patchRef)
        return results

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
        if self.config.doInsertFakes:
            self.insertFakes.run(exposure, background=backgrounds)
        table = afwTable.SourceTable.make(self.schema, idFactory)
        detections = self.detection.run(table, exposure, expId=expId)
        sources = detections.sources
        fpSets = detections.fpSets
        if hasattr(fpSets, "background") and fpSets.background:
            for bg in fpSets.background:
                backgrounds.append(bg)
        return Struct(outputSources=sources, outputBackgrounds=backgrounds, outputExposure=exposure)

    def write(self, results, patchRef):
        """!
        @brief Write out results from runDetection.

        @param[in] exposure: Exposure to write out
        @param[in] results: Struct returned from runDetection
        @param[in] patchRef: data reference for patch
        """
        coaddName = self.config.coaddName + "Coadd"
        patchRef.put(results.outputBackgrounds, coaddName + "_calexp_background")
        patchRef.put(results.outputSources, coaddName + "_det")
        if self.config.hasFakes:
            patchRef.put(results.outputExposure, "fakes_" + coaddName + "_calexp")
        else:
            patchRef.put(results.outputExposure, coaddName + "_calexp")

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


class DeblendCoaddSourcesRunner(MergeSourcesRunner):
    """Task runner for the `MergeSourcesTask`

    Required because the run method requires a list of
    dataRefs rather than a single dataRef.
    """
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Provide a list of patch references for each patch, tract, filter combo.

        Parameters
        ----------
        parsedCmd:
            The parsed command
        kwargs:
            Keyword arguments passed to the task

        Returns
        -------
        targetList: list
            List of tuples, where each tuple is a (dataRef, kwargs) pair.
        """
        refDict = MergeSourcesRunner.buildRefDict(parsedCmd)
        kwargs["psfCache"] = parsedCmd.psfCache
        return [(list(p.values()), kwargs) for t in refDict.values() for p in t.values()]


class DeblendCoaddSourcesTask(CmdLineTask):
    """Deblend the sources in a merged catalog

    Deblend sources from master catalog in each coadd.
    This can either be done separately in each band using the HSC-SDSS deblender
    (`DeblendCoaddSourcesTask.config.simultaneous==False`)
    or use SCARLET to simultaneously fit the blend in all bands
    (`DeblendCoaddSourcesTask.config.simultaneous==True`).
    The task will set its own `self.schema` atribute to the `Schema` of the
    output deblended catalog.
    This will include all fields from the input `Schema`, as well as additional fields
    from the deblender.

    `pipe.tasks.multiband.DeblendCoaddSourcesTask Description
    ---------------------------------------------------------
    `

    Parameters
    ----------
    butler: `Butler`
        Butler used to read the input schemas from disk or
        construct the reference catalog loader, if `schema` or `peakSchema` or
    schema: `Schema`
        The schema of the merged detection catalog as an input to this task.
    peakSchema: `Schema`
        The schema of the `PeakRecord`s in the `Footprint`s in the merged detection catalog
    """
    ConfigClass = DeblendCoaddSourcesConfig
    RunnerClass = DeblendCoaddSourcesRunner
    _DefaultName = "deblendCoaddSources"
    makeIdFactory = _makeMakeIdFactory("MergedCoaddId", includeBand=False)

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd_calexp",
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=g^r^i",
                               ContainerClass=ExistingCoaddDataIdContainer)
        parser.add_argument("--psfCache", type=int, default=100, help="Size of CoaddPsf cache")
        return parser

    def __init__(self, butler=None, schema=None, peakSchema=None, **kwargs):
        CmdLineTask.__init__(self, **kwargs)
        if schema is None:
            assert butler is not None, "Neither butler nor schema is defined"
            schema = butler.get(self.config.coaddName + "Coadd_mergeDet_schema", immediate=True).schema
        self.schemaMapper = afwTable.SchemaMapper(schema)
        self.schemaMapper.addMinimalSchema(schema)
        self.schema = self.schemaMapper.getOutputSchema()
        if peakSchema is None:
            assert butler is not None, "Neither butler nor peakSchema is defined"
            peakSchema = butler.get(self.config.coaddName + "Coadd_peak_schema", immediate=True).schema

        if self.config.simultaneous:
            self.makeSubtask("multiBandDeblend", schema=self.schema, peakSchema=peakSchema)
        else:
            self.makeSubtask("singleBandDeblend", schema=self.schema, peakSchema=peakSchema)

    def getSchemaCatalogs(self):
        """Return a dict of empty catalogs for each catalog dataset produced by this task.

        Returns
        -------
        result: dict
            Dictionary of empty catalogs, with catalog names as keys.
        """
        catalog = afwTable.SourceCatalog(self.schema)
        return {self.config.coaddName + "Coadd_deblendedFlux": catalog,
                self.config.coaddName + "Coadd_deblendedModel": catalog}

    def runDataRef(self, patchRefList, psfCache=100):
        """Deblend the patch

        Deblend each source simultaneously or separately
        (depending on `DeblendCoaddSourcesTask.config.simultaneous`).
        Set `is-primary` and related flags.
        Propagate flags from individual visits.
        Write the deblended sources out.

        Parameters
        ----------
        patchRefList: list
            List of data references for each filter
        """

        if self.config.hasFakes:
            coaddType = "fakes_" + self.config.coaddName
        else:
            coaddType = self.config.coaddName

        if self.config.simultaneous:
            # Use SCARLET to simultaneously deblend across filters
            filters = []
            exposures = []
            for patchRef in patchRefList:
                exposure = patchRef.get(coaddType + "Coadd_calexp", immediate=True)
                filter = patchRef.get(coaddType + "Coadd_filterLabel", immediate=True)
                filters.append(filter.bandLabel)
                exposures.append(exposure)
            # Sort inputs by band to match Gen3 order of inputs
            exposures = [exposure for _, exposure in sorted(zip(filters, exposures))]
            patchRefList = [patchRef for _, patchRef in sorted(zip(filters, patchRefList))]
            filters.sort()
            # The input sources are the same for all bands, since it is a merged catalog
            sources = self.readSources(patchRef)
            exposure = afwImage.MultibandExposure.fromExposures(filters, exposures)
            templateCatalogs, fluxCatalogs = self.multiBandDeblend.run(exposure, sources)
            for n in range(len(patchRefList)):
                self.write(patchRefList[n], templateCatalogs[filters[n]], "Model")
                if filters[n] in fluxCatalogs:
                    self.write(patchRefList[n], fluxCatalogs[filters[n]], "Flux")
        else:
            # Use the singeband deblender to deblend each band separately
            for patchRef in patchRefList:
                exposure = patchRef.get(coaddType + "Coadd_calexp", immediate=True)
                exposure.getPsf().setCacheCapacity(psfCache)
                sources = self.readSources(patchRef)
                self.singleBandDeblend.run(exposure, sources)
                self.write(patchRef, sources)

    def readSources(self, dataRef):
        """Read merged catalog

        Read the catalog of merged detections and create a catalog
        in a single band.

        Parameters
        ----------
        dataRef: data reference
            Data reference for catalog of merged detections

        Returns
        -------
        sources: `SourceCatalog`
            List of sources in merged catalog

        We also need to add columns to hold the measurements we're about to make
        so we can measure in-place.
        """
        merged = dataRef.get(self.config.coaddName + "Coadd_mergeDet", immediate=True)
        self.log.info("Read %d detections: %s", len(merged), dataRef.dataId)
        idFactory = self.makeIdFactory(dataRef)
        # There may be gaps in the mergeDet catalog, which will cause the
        # source ids to be inconsistent. So we update the id factory
        # with the largest id already in the catalog.
        maxId = np.max(merged["id"])
        idFactory.notify(maxId)
        table = afwTable.SourceTable.make(self.schema, idFactory)
        sources = afwTable.SourceCatalog(table)
        sources.extend(merged, self.schemaMapper)
        return sources

    def write(self, dataRef, sources, catalogType):
        """Write the source catalog(s)

        Parameters
        ----------
        dataRef: Data Reference
            Reference to the output catalog.
        sources: `SourceCatalog`
            Flux conserved sources to write to file.
            If using the single band deblender, this is the catalog
            generated.
        template_sources: `SourceCatalog`
            Source catalog using the multiband template models
            as footprints.
        """
        dataRef.put(sources, self.config.coaddName + f"Coadd_deblended{catalogType}")
        self.log.info("Wrote %d sources: %s", len(sources), dataRef.dataId)

    def writeMetadata(self, dataRefList):
        """Write the metadata produced from processing the data.
        Parameters
        ----------
        dataRefList
            List of Butler data references used to write the metadata.
            The metadata is written to dataset type `CmdLineTask._getMetadataName`.
        """
        for dataRef in dataRefList:
            try:
                metadataName = self._getMetadataName()
                if metadataName is not None:
                    dataRef.put(self.getFullMetadata(), metadataName)
            except Exception as e:
                self.log.warning("Could not persist metadata for dataId=%s: %s", dataRef.dataId, e)


class MeasureMergedCoaddSourcesConnections(PipelineTaskConnections,
                                           dimensions=("tract", "patch", "band", "skymap"),
                                           defaultTemplates={"inputCoaddName": "deep",
                                                             "outputCoaddName": "deep"}):
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
            "and characterization to codd objects",
        name="src",
        dimensions=("instrument", "visit", "detector"),
        storageClass="SourceCatalog",
        multiple=True
    )
    inputCatalog = cT.Input(
        doc=("Name of the input catalog to use."
             "If the single band deblender was used this should be 'deblendedFlux."
             "If the multi-band deblender was used this should be 'deblendedModel, "
             "or deblendedFlux if the multiband deblender was configured to output "
             "deblended flux catalogs. If no deblending was performed this should "
             "be 'mergeDet'"),
        name="{inputCoaddName}Coadd_deblendedFlux",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "band", "skymap"),
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
    inputCatalog = Field(dtype=str, default="deblendedFlux",
                         doc=("Name of the input catalog to use."
                              "If the single band deblender was used this should be 'deblendedFlux."
                              "If the multi-band deblender was used this should be 'deblendedModel."
                              "If no deblending was performed this should be 'mergeDet'"))
    measurement = ConfigurableField(target=SingleFrameMeasurementTask, doc="Source measurement")
    setPrimaryFlags = ConfigurableField(target=SetPrimaryFlagsTask, doc="Set flags for primary tract/patch")
    doPropagateFlags = Field(
        dtype=bool, default=True,
        doc="Whether to match sources to CCD catalogs to propagate flags (to e.g. identify PSF stars)"
    )
    propagateFlags = ConfigurableField(target=PropagateVisitFlagsTask, doc="Propagate visit flags to coadd")
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

    def validate(self):
        super().validate()
        refCatGen2 = getattr(self.refObjLoader, "ref_dataset_name", None)
        if refCatGen2 is not None and refCatGen2 != self.connections.refCat:
            raise ValueError(
                f"Gen2 ({refCatGen2}) and Gen3 ({self.connections.refCat}) reference catalogs "
                f"are different.  These options must be kept in sync until Gen2 is retired."
            )


## @addtogroup LSST_task_documentation
## @{
## @page MeasureMergedCoaddSourcesTask
## @ref MeasureMergedCoaddSourcesTask_ "MeasureMergedCoaddSourcesTask"
## @copybrief MeasureMergedCoaddSourcesTask
## @}


class MeasureMergedCoaddSourcesRunner(ButlerInitializedTaskRunner):
    """Get the psfCache setting into MeasureMergedCoaddSourcesTask"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return ButlerInitializedTaskRunner.getTargetList(parsedCmd, psfCache=parsedCmd.psfCache)


class MeasureMergedCoaddSourcesTask(PipelineTask, CmdLineTask):
    r"""!
    @anchor MeasureMergedCoaddSourcesTask_

    @brief Deblend sources from master catalog in each coadd seperately and measure.

    @section pipe_tasks_multiBand_Contents Contents

      - @ref pipe_tasks_multiBand_MeasureMergedCoaddSourcesTask_Purpose
      - @ref pipe_tasks_multiBand_MeasureMergedCoaddSourcesTask_Initialize
      - @ref pipe_tasks_multiBand_MeasureMergedCoaddSourcesTask_Run
      - @ref pipe_tasks_multiBand_MeasureMergedCoaddSourcesTask_Config
      - @ref pipe_tasks_multiBand_MeasureMergedCoaddSourcesTask_Debug
      - @ref pipe_tasks_multiband_MeasureMergedCoaddSourcesTask_Example

    @section pipe_tasks_multiBand_MeasureMergedCoaddSourcesTask_Purpose	Description

    Command-line task that uses peaks and footprints from a master catalog to perform deblending and
    measurement in each coadd.

    Given a master input catalog of sources (peaks and footprints) or deblender outputs
    (including a HeavyFootprint in each band), measure each source on the
    coadd. Repeating this procedure with the same master catalog across multiple coadds will generate a
    consistent set of child sources.

    The deblender retains all peaks and deblends any missing peaks (dropouts in that band) as PSFs. Source
    properties are measured and the @c is-primary flag (indicating sources with no children) is set. Visit
    flags are propagated to the coadd sources.

    Optionally, we can match the coadd sources to an external reference catalog.

      @par Inputs:
        deepCoadd_mergeDet{tract,patch} or deepCoadd_deblend{tract,patch}: SourceCatalog
        @n deepCoadd_calexp{tract,patch,filter}: ExposureF
      @par Outputs:
        deepCoadd_meas{tract,patch,filter}: SourceCatalog
      @par Data Unit:
        tract, patch, filter

    MeasureMergedCoaddSourcesTask delegates most of its work to a set of sub-tasks:

    <DL>
      <DT> @ref SingleFrameMeasurementTask_ "measurement"
      <DD> Measure source properties of deblended sources.</DD>
      <DT> @ref SetPrimaryFlagsTask_ "setPrimaryFlags"
      <DD> Set flag 'is-primary' as well as related flags on sources. 'is-primary' is set for sources that are
      not at the edge of the field and that have either not been deblended or are the children of deblended
      sources</DD>
      <DT> @ref PropagateVisitFlagsTask_ "propagateFlags"
      <DD> Propagate flags set in individual visits to the coadd.</DD>
      <DT> @ref DirectMatchTask_ "match"
      <DD> Match input sources to a reference catalog (optional).
      </DD>
    </DL>
    These subtasks may be retargeted as required.

    @section pipe_tasks_multiBand_MeasureMergedCoaddSourcesTask_Initialize       Task initialization

    @copydoc \_\_init\_\_

    @section pipe_tasks_multiBand_MeasureMergedCoaddSourcesTask_Run       Invoking the Task

    @copydoc run

    @section pipe_tasks_multiBand_MeasureMergedCoaddSourcesTask_Config       Configuration parameters

    See @ref MeasureMergedCoaddSourcesConfig_

    @section pipe_tasks_multiBand_MeasureMergedCoaddSourcesTask_Debug		Debug variables

    The @link lsst.pipe.base.cmdLineTask.CmdLineTask command line task@endlink interface supports a
    flag @c -d to import @b debug.py from your @c PYTHONPATH; see @ref baseDebug for more about @b debug.py
    files.

    MeasureMergedCoaddSourcesTask has no debug variables of its own because it delegates all the work to
    the various sub-tasks. See the documetation for individual sub-tasks for more information.

    @section pipe_tasks_multiband_MeasureMergedCoaddSourcesTask_Example	A complete example of using
    MeasureMergedCoaddSourcesTask

    After MeasureMergedCoaddSourcesTask has been run on multiple coadds, we have a set of per-band catalogs.
    The next stage in the multi-band processing procedure will merge these measurements into a suitable
    catalog for driving forced photometry.

    Command-line usage of MeasureMergedCoaddSourcesTask expects a data reference to the coadds
    to be processed.
    A list of the available optional arguments can be obtained by calling measureCoaddSources.py with the
    `--help` command line argument:
    @code
    measureCoaddSources.py --help
    @endcode

    To demonstrate usage of the DetectCoaddSourcesTask in the larger context of multi-band processing, we
    will process HSC data in the [ci_hsc](https://github.com/lsst/ci_hsc) package. Assuming one has finished
    step 6 at @ref pipeTasks_multiBand, one may perform deblending and measure sources in the HSC-I band
    coadd as follows:
    @code
    measureCoaddSources.py $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-I
    @endcode
    This will process the HSC-I band data. The results are written in
    `$CI_HSC_DIR/DATA/deepCoadd-results/HSC-I/0/5,4/meas-HSC-I-0-5,4.fits

    It is also necessary to run
    @code
    measureCoaddSources.py $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-R
    @endcode
    to generate the sources catalogs for the HSC-R band required by the next step in the multi-band
    procedure: @ref MergeMeasurementsTask_ "MergeMeasurementsTask".
    """
    _DefaultName = "measureCoaddSources"
    ConfigClass = MeasureMergedCoaddSourcesConfig
    RunnerClass = MeasureMergedCoaddSourcesRunner
    getSchemaCatalogs = _makeGetSchemaCatalogs("meas")
    # The IDs we already have are of this type
    makeIdFactory = _makeMakeIdFactory("MergedCoaddId", includeBand=False)

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd_calexp",
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=r",
                               ContainerClass=ExistingCoaddDataIdContainer)
        parser.add_argument("--psfCache", type=int, default=100, help="Size of CoaddPsf cache")
        return parser

    def __init__(self, butler=None, schema=None, peakSchema=None, refObjLoader=None, initInputs=None,
                 **kwargs):
        """!
        @brief Initialize the task.

        Keyword arguments (in addition to those forwarded to CmdLineTask.__init__):
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
                                             inputs.pop('refCat'), config=self.config.refObjLoader,
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
        sources.extend(inputs.pop('inputCatalog'), self.schemaMapper)
        table = sources.getTable()
        table.setMetadata(self.algMetadata)  # Capture algorithm metadata to write out to the source catalog.
        inputs['sources'] = sources

        skyMap = inputs.pop('skyMap')
        tractNumber = inputRefs.inputCatalog.dataId['tract']
        tractInfo = skyMap[tractNumber]
        patchInfo = tractInfo.getPatchInfo(inputRefs.inputCatalog.dataId['patch'])
        skyInfo = Struct(
            skyMap=skyMap,
            tractInfo=tractInfo,
            patchInfo=patchInfo,
            wcs=tractInfo.getWcs(),
            bbox=patchInfo.getOuterBBox()
        )
        inputs['skyInfo'] = skyInfo

        if self.config.doPropagateFlags:
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
        butlerQC.put(outputs, outputRefs)

    def runDataRef(self, patchRef, psfCache=100):
        """!
        @brief Deblend and measure.

        @param[in] patchRef: Patch reference.

        Set 'is-primary' and related flags. Propagate flags
        from individual visits. Optionally match the sources to a reference catalog and write the matches.
        Finally, write the deblended sources and measurements out.
        """
        if self.config.hasFakes:
            coaddType = "fakes_" + self.config.coaddName
        else:
            coaddType = self.config.coaddName
        exposure = patchRef.get(coaddType + "Coadd_calexp", immediate=True)
        exposure.getPsf().setCacheCapacity(psfCache)
        sources = self.readSources(patchRef)
        table = sources.getTable()
        table.setMetadata(self.algMetadata)  # Capture algorithm metadata to write out to the source catalog.
        skyInfo = getSkyInfo(coaddName=self.config.coaddName, patchRef=patchRef)

        if self.config.doPropagateFlags:
            ccdInputs = self.propagateFlags.getCcdInputs(exposure)
        else:
            ccdInputs = None

        expId = getGen3CoaddExposureId(patchRef, coaddName=self.config.coaddName, includeBand=False,
                                       log=self.log)
        results = self.run(exposure=exposure, sources=sources, skyInfo=skyInfo, exposureId=expId,
                           ccdInputs=ccdInputs, butler=patchRef.getButler())

        if self.config.doMatchSources:
            self.writeMatches(patchRef, results)
        self.write(patchRef, results.outputSources)

    def run(self, exposure, sources, skyInfo, exposureId, ccdInputs=None, visitCatalogs=None, wcsUpdates=None,
            butler=None):
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
            the exposure
        visitCatalogs : list of `lsst.afw.table.SourceCatalogs` or `None`
            A list of source catalogs corresponding to measurements made on the individual
            visits which went into the input exposure. If None and butler is `None` then
            the task cannot propagate visit flags to the output catalog.
        wcsUpdates : list of `lsst.afw.geom.SkyWcs` or `None`
            If visitCatalogs is not `None` this should be a list of wcs objects which correspond
            to the input visits. Used to put all coordinates to common system. If `None` and
            butler is `None` then the task cannot propagate visit flags to the output catalog.
        butler : `lsst.daf.butler.Butler` or `lsst.daf.persistence.Butler`
            Either a gen2 or gen3 butler used to load visit catalogs

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Results of running measurement task. Will contain the catalog in the
            sources attribute. Optionally will have results of matching to a
            reference catalog in the matchResults attribute, and denormalized
            matches in the denormMatches attribute.
        """
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
            self.propagateFlags.run(butler, sources, ccdInputs, exposure.getWcs(), visitCatalogs, wcsUpdates)

        results = Struct()

        if self.config.doMatchSources:
            matchResult = self.match.run(sources, exposure.getInfo().getFilterLabel().bandLabel)
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

    def readSources(self, dataRef):
        """!
        @brief Read input sources.

        @param[in] dataRef: Data reference for catalog of merged detections
        @return List of sources in merged catalog

        We also need to add columns to hold the measurements we're about to make
        so we can measure in-place.
        """
        merged = dataRef.get(self.config.coaddName + self.inputCatalog, immediate=True)
        self.log.info("Read %d detections: %s", len(merged), dataRef.dataId)
        idFactory = self.makeIdFactory(dataRef)
        for s in merged:
            idFactory.notify(s.getId())
        table = afwTable.SourceTable.make(self.schema, idFactory)
        sources = afwTable.SourceCatalog(table)
        sources.extend(merged, self.schemaMapper)
        return sources

    def writeMatches(self, dataRef, results):
        """!
        @brief Write matches of the sources to the astrometric reference catalog.

        @param[in] dataRef: data reference
        @param[in] results: results struct from run method
        """
        if hasattr(results, "matchResult"):
            dataRef.put(results.matchResult, self.config.coaddName + "Coadd_measMatch")
        if hasattr(results, "denormMatches"):
            dataRef.put(results.denormMatches, self.config.coaddName + "Coadd_measMatchFull")

    def write(self, dataRef, sources):
        """!
        @brief Write the source catalog.

        @param[in] dataRef: data reference
        @param[in] sources: source catalog
        """
        dataRef.put(sources, self.config.coaddName + "Coadd_meas")
        self.log.info("Wrote %d sources: %s", len(sources), dataRef.dataId)
