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
"""
New dataset types:
- deepCoadd_det: detections from what used to be processCoadd (tract, patch, filter)
- deepCoadd_mergeDet: merged detections (tract, patch)
- deepCoadd_meas: measurements of merged detections (tract, patch, filter)
- deepCoadd_ref: reference sources (tract, patch)
All of these have associated schema catalogs that require no data ID and hold no records.

In addition, we have a schema-only dataset, which saves the schema for the PeakRecords in
the mergeDet, meas, and ref dataset Footprints:
- deepCoadd_peak_schema
"""
import numpy

from lsst.coadd.utils.coaddDataIdContainer import ExistingCoaddDataIdContainer
from lsst.pipe.base import (CmdLineTask, Struct, ArgumentParser, ButlerInitializedTaskRunner,
                            PipelineTask, InitOutputDatasetField, InputDatasetField, OutputDatasetField,
                            QuantumConfig)
from lsst.pex.config import Config, Field, ConfigurableField
from lsst.meas.algorithms import DynamicDetectionTask
from lsst.meas.base import SingleFrameMeasurementTask, ApplyApCorrTask, CatalogCalculationTask
from lsst.meas.deblender import SourceDeblendTask, MultibandDeblendTask
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

from .mergeDetections import MergeDetectionsConfig, MergeDetectionsTask  # noqa: F401
from .mergeMeasurements import MergeMeasurementsConfig, MergeMeasurementsTask  # noqa: F401
from .multiBandUtils import MergeSourcesRunner, CullPeaksConfig, _makeGetSchemaCatalogs  # noqa: F401
from .multiBandUtils import getInputSchema, getShortFilterName, readCatalog, _makeMakeIdFactory  # noqa: F401





##############################################################################################################

class DetectCoaddSourcesConfig(Config):
    """Configuration parameters for the DetectCoaddSourcesTask
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
    detectionSchema = InitOutputDatasetField(
        doc="Schema of the detection catalog",
        name="{}Coadd_det_schema",
        storageClass="SourceCatalog",
    )
    exposure = InputDatasetField(
        doc="Exposure on which detections are to be performed",
        name="deepCoadd",
        scalar=True,
        storageClass="Exposure",
        dimensions=("Tract", "Patch", "AbstractFilter", "SkyMap")
    )
    outputBackgrounds = OutputDatasetField(
        doc="Output Backgrounds used in detection",
        name="{}Coadd_calexp_background",
        scalar=True,
        storageClass="Background",
        dimensions=("Tract", "Patch", "AbstractFilter", "SkyMap")
    )
    outputSources = OutputDatasetField(
        doc="Detected sources catalog",
        name="{}Coadd_det",
        scalar=True,
        storageClass="SourceCatalog",
        dimensions=("Tract", "Patch", "AbstractFilter", "SkyMap")
    )
    outputExposure = OutputDatasetField(
        doc="Exposure post detection",
        name="{}Coadd_calexp",
        scalar=True,
        storageClass="Exposure",
        dimensions=("Tract", "Patch", "AbstractFilter", "SkyMap")
    )
    quantum = QuantumConfig(
        dimensions=("Tract", "Patch", "AbstractFilter", "SkyMap")
    )

    def setDefaults(self):
        Config.setDefaults(self)
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


class DetectCoaddSourcesTask(CmdLineTask):
    """Command-line task that detects sources on a coadd of exposures obtained with a single filter.

    Coadding individual visits requires each exposure to be warped. This introduces covariance in the noise
    properties across pixels. Before detection, we correct the coadd variance by scaling the variance plane
    in the coadd to match the observed variance. This is an approximate approach -- strictly, we should
    propagate the full covariance matrix -- but it is simple and works well in practice.

    After scaling the variance plane, we detect sources and generate footprints by delegating to the @ref
    SourceDetectionTask_ "detection" subtask.

    Parameters
    -----
    deepCoadd :
        {tract,patch,filter}. ExposureF
    deepCoadd_det :
        {tract,patch,filter}. SourceCatalog (only parent Footprints)
    deepCoadd_calexp :
        {tract,patch,filter}. Variance scaled, background-subtracted input
        exposure (ExposureF)
    deepCoadd_calexp_background :
        {tract,patch,filter}. BackgroundList
    Data Unit :
        tract, patch, filter
    schema :
        initial schema for the output catalog, modified-in place to include all
        fields set by this task.  If None, the source minimal schema will be used.
    kwargs :
        keyword arguments to be passed to lsst.pipe.base.task.Task.__init__
    Notes
    -----

    Examples
    --------

    DetectCoaddSourcesTask is meant to be run after assembling a coadded image in a given band. The purpose of
    the task is to update the background, detect all sources in a single band and generate a set of parent
    footprints. Subsequent tasks in the multi-band processing procedure will merge sources across bands and,
    eventually, perform forced photometry. Command-line usage of DetectCoaddSourcesTask expects a data
    reference to the coadd to be processed. A list of the available optional arguments can be obtained by
    calling detectCoaddSources.py with the `--help` command line argument:

    >>> detectCoaddSources.py --help

    To demonstrate usage of the DetectCoaddSourcesTask in the larger context of multi-band processing, we
    will process HSC data in the [ci_hsc](https://github.com/lsst/ci_hsc) package. Assuming one has followed
    steps 1 - 4 at @ref pipeTasks_multiBand, one may detect all the sources in each coadd as follows:


    >>> detectCoaddSources.py $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-I

    that will process the HSC-I band data. The results are written to
    `$CI_HSC_DIR/DATA/deepCoadd-results/HSC-I`.

    It is also necessary to run:

    >>> detectCoaddSources.py $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-R

    to generate the sources catalogs for the HSC-R band required by the next step in the multi-band
    processing procedure: MergeDetectionsTask_ "MergeDetectionsTask".
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

    @classmethod
    def getOutputDatasetTypes(cls, config):
        coaddName = config.coaddName
        for name in ("outputBackgrounds", "outputSources", "outputExposure"):
            attr = getattr(config, name)
            setattr(attr, "name", attr.name.format(coaddName))
        outputTypeDict = super().getOutputDatasetTypes(config)
        return outputTypeDict

    @classmethod
    def getInitOutputDatasetTypes(cls, config):
        coaddName = config.coaddName
        attr = config.detectionSchema
        setattr(attr, "name", attr.name.format(coaddName))
        output = super().getInitOutputDatasetTypes(config)
        return output

    def __init__(self, schema=None, **kwargs):
        """Initialize the task. Create the @ref SourceDetectionTask_ "detection" subtask.

        Keyword arguments (in addition to those forwarded to CmdLineTask.__init__):

        Parameters
        ----------
        schema :
            initial schema for the output catalog, modified-in place to include all
            fields set by this task.  If None, the source minimal schema will be used.
        kwargs :
            keyword arguments to be passed to lsst.pipe.base.task.Task.__init__
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

    def getInitOutputDatasets(self):
        return {"detectionSchema": afwTable.SourceCatalog(self.schema)}

    def runDataRef(self, patchRef):
        """Run detection on a coadd.

        Invokes run and then uses write to output the
        results.

        Parameters
        ----------
        patchRef :
            data reference for patch
        """
        exposure = patchRef.get(self.config.coaddName + "Coadd", immediate=True)
        expId = int(patchRef.get(self.config.coaddName + "CoaddId"))
        results = self.run(exposure, self.makeIdFactory(patchRef), expId=expId)
        self.write(results, patchRef)
        return results

    def adaptArgsAndRun(self, inputData, inputDataIds, outputDataIds, butler):
        # FINDME: DM-15843 needs to come back and address these next two lines with a final solution
        inputData["idFactory"] = afwTable.IdFactory.makeSimple()
        inputData["expId"] = 0
        return self.run(**inputData)

    def run(self, exposure, idFactory, expId):
        """Run detection on an exposure.

        First scale the variance plane to match the observed variance
        using ScaleVarianceTask. Then invoke the SourceDetectionTask_ "detection" subtask to
        detect sources.

        Parameters
        ----------
        exposure :
            Exposure on which to detect (may be backround-subtracted and scaled,
            depending on configuration).
        idFactory :
            IdFactory to set source identifiers
        expId :
            Exposure identifier (integer) for RNG seed

        Returns
        -------
        result : `pipe.base.Struct`
            a pipe.base.Struct with fields:
            - ``sources`` : catalog of detections
            - ``backgrounds`` : list of backgrounds
        """
        if self.config.doScaleVariance:
            varScale = self.scaleVariance.run(exposure.maskedImage)
            exposure.getMetadata().add("variance_scale", varScale)
        backgrounds = afwMath.BackgroundList()
        if self.config.doInsertFakes:
            self.insertFakes.run(exposure, background=backgrounds)
        table = afwTable.SourceTable.make(self.schema, idFactory)
        detections = self.detection.makeSourceCatalog(table, exposure, expId=expId)
        sources = detections.sources
        fpSets = detections.fpSets
        if hasattr(fpSets, "background") and fpSets.background:
            for bg in fpSets.background:
                backgrounds.append(bg)
        return Struct(outputSources=sources, outputBackgrounds=backgrounds, outputExposure=exposure)

    def write(self, results, patchRef):
        """!
        @brief Write out results from runDetection.

        exposure:
            Exposure to write out
        results:
            Struct returned from runDetection
        patchRef:
            data reference for patch
        """
        coaddName = self.config.coaddName + "Coadd"
        patchRef.put(results.outputBackgrounds, coaddName + "_calexp_background")
        patchRef.put(results.outputSources, coaddName + "_det")
        patchRef.put(results.outputExposure, coaddName + "_calexp")

##############################################################################################################


class DeblendCoaddSourcesConfig(Config):
    """DeblendCoaddSourcesConfig

    Configuration parameters for the `DeblendCoaddSourcesTask`.
    """
    singleBandDeblend = ConfigurableField(target=SourceDeblendTask,
                                          doc="Deblend sources separately in each band")
    multiBandDeblend = ConfigurableField(target=MultibandDeblendTask,
                                         doc="Deblend sources simultaneously across bands")
    simultaneous = Field(dtype=bool, default=False, doc="Simultaneously deblend all bands?")
    coaddName = Field(dtype=str, default="deep", doc="Name of coadd")

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
    ( `DeblendCoaddSourcesTask.config.simultaneous==False` )
    or use SCARLET to simultaneously fit the blend in all bands
    ( `DeblendCoaddSourcesTask.config.simultaneous==True` ).
    The task will set its own `self.schema` atribute to the `Schema` of the
    output deblended catalog.
    This will include all fields from the input `Schema`, as well as additional fields
    from the deblender.

    Parameters
    ----------
    butler: `Butler`
        Butler used to read the input schemas from disk or
        construct the reference catalog loader, if `schema` or `peakSchema`
    schema: `Schema`
        The schema of the merged detection catalog as an input to this task.
    peakSchema: `Schema`
        The schema of the `PeakRecord` in the `Footprint` in the merged detection catalog
    """
    ConfigClass = DeblendCoaddSourcesConfig
    RunnerClass = DeblendCoaddSourcesRunner
    _DefaultName = "deblendCoaddSources"
    makeIdFactory = _makeMakeIdFactory("MergedCoaddId")

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
        result: `dict`
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
        patchRefList: `list`
            List of data references for each filter
        """
        if self.config.simultaneous:
            # Use SCARLET to simultaneously deblend across filters
            filters = []
            exposures = []
            for patchRef in patchRefList:
                exposure = patchRef.get(self.config.coaddName + "Coadd_calexp", immediate=True)
                filters.append(patchRef.dataId["filter"])
                exposures.append(exposure)
            # The input sources are the same for all bands, since it is a merged catalog
            sources = self.readSources(patchRef)
            exposure = afwImage.MultibandExposure.fromExposures(filters, exposures)
            fluxCatalogs, templateCatalogs = self.multiBandDeblend.run(exposure, sources)
            for n in range(len(patchRefList)):
                self.write(patchRefList[n], fluxCatalogs[filters[n]], templateCatalogs[filters[n]])
        else:
            # Use the singeband deblender to deblend each band separately
            for patchRef in patchRefList:
                exposure = patchRef.get(self.config.coaddName + "Coadd_calexp", immediate=True)
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
        self.log.info("Read %d detections: %s" % (len(merged), dataRef.dataId))
        idFactory = self.makeIdFactory(dataRef)
        for s in merged:
            idFactory.notify(s.getId())
        table = afwTable.SourceTable.make(self.schema, idFactory)
        sources = afwTable.SourceCatalog(table)
        sources.extend(merged, self.schemaMapper)
        return sources

    def write(self, dataRef, flux_sources, template_sources=None):
        """Write the source catalog(s)

        Parameters
        ----------
        dataRef: Data Reference
            Reference to the output catalog.
        flux_sources: `SourceCatalog`
            Flux conserved sources to write to file.
            If using the single band deblender, this is the catalog
            generated.
        template_sources: `SourceCatalog`
            Source catalog using the multiband template models
            as footprints.
        """
        # The multiband deblender does not have to conserve flux,
        # so only write the flux conserved catalog if it exists
        if flux_sources is not None:
            assert not self.config.simultaneous or self.config.multiBandDeblend.conserveFlux
            dataRef.put(flux_sources, self.config.coaddName + "Coadd_deblendedFlux")
        # Only the multiband deblender has the option to output the
        # template model catalog, which can optionally be used
        # in MeasureMergedCoaddSources
        if template_sources is not None:
            assert self.config.multiBandDeblend.saveTemplates
            dataRef.put(template_sources, self.config.coaddName + "Coadd_deblendedModel")
        self.log.info("Wrote %d sources: %s" % (len(flux_sources), dataRef.dataId))

    def writeMetadata(self, dataRefList):
        """Write the metadata produced from processing the data.

        Parameters
        ----------
        dataRefList : `list`
            List of Butler data references used to write the metadata.
            The metadata is written to dataset type `CmdLineTask._getMetadataName`.
        """
        for dataRef in dataRefList:
            try:
                metadataName = self._getMetadataName()
                if metadataName is not None:
                    dataRef.put(self.getFullMetadata(), metadataName)
            except Exception as e:
                self.log.warn("Could not persist metadata for dataId=%s: %s", dataRef.dataId, e)

    def getExposureId(self, dataRef):
        """Get the ExposureId from a data reference
        """
        return int(dataRef.get(self.config.coaddName + "CoaddId"))


class MeasureMergedCoaddSourcesConfig(Config):
    """Configuration parameters for the MeasureMergedCoaddSourcesTask
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

    def setDefaults(self):
        Config.setDefaults(self)
        self.measurement.plugins.names |= ['base_InputCount', 'base_Variance']
        self.measurement.plugins['base_PixelFlags'].masksFpAnywhere = ['CLIPPED', 'SENSOR_EDGE',
                                                                       'INEXACT_PSF']
        self.measurement.plugins['base_PixelFlags'].masksFpCenter = ['CLIPPED', 'SENSOR_EDGE',
                                                                     'INEXACT_PSF']

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


class MeasureMergedCoaddSourcesTask(CmdLineTask):
    """Command-line task that uses peaks and footprints from a master catalog to perform deblending and
    measurement in each coadd.

    Given a master input catalog of sources (peaks and footprints) or deblender outputs
    (including a HeavyFootprint in each band), measure each source on the
    coadd. Repeating this procedure with the same master catalog across multiple coadds will generate a
    consistent set of child sources.

    The deblender retains all peaks and deblends any missing peaks (dropouts in that band) as PSFs. Source
    properties are measured and the is-primary flag (indicating sources with no children) is set. Visit
    flags are propagated to the coadd sources.

    Optionally, we can match the coadd sources to an external reference catalog.

    Parameters
    ----------
        deepCoadd_mergeDet{tract,patch} :
        deepCoadd_deblend{tract,patch} :
            SourceCatalog
        deepCoadd_calexp{tract,patch,filter} :
            ExposureF
        deepCoadd_meas{tract,patch,filter} :
            SourceCatalog
        Data Unit :
            tract, patch, filter

    Notes
    -----
    MeasureMergedCoaddSourcesTask delegates most of its work to a set of sub-tasks:

    The ``lsst.pipe.base.cmdLineTask.CmdLineTask`` command line task interface supports a
    flag -d to import debug.py from your PYTHONPATH; see baseDebug for more about debug.py
    files.

    MeasureMergedCoaddSourcesTask has no debug variables of its own because it delegates all the work to
    the various sub-tasks. See the documetation for individual sub-tasks for more information.

    Examples
    --------
    After MeasureMergedCoaddSourcesTask has been run on multiple coadds, we have a set of per-band catalogs.
    The next stage in the multi-band processing procedure will merge these measurements into a suitable
    catalog for driving forced photometry.

    Command-line usage of MeasureMergedCoaddSourcesTask expects a data reference to the coadds
    to be processed.
    A list of the available optional arguments can be obtained by calling measureCoaddSources.py with the
    `--help` command line argument:

    >>> measureCoaddSources.py --help


    To demonstrate usage of the DetectCoaddSourcesTask in the larger context of multi-band processing, we
    will process HSC data in the [ci_hsc](https://github.com/lsst/ci_hsc) package. Assuming one has finished
    step 6 at pipeTasks_multiBand, one may perform deblending and measure sources in the HSC-I band
    coadd as follows:

    >>> measureCoaddSources.py $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-I

    This will process the HSC-I band data. The results are written in
    `$CI_HSC_DIR/DATA/deepCoadd-results/HSC-I/0/5,4/meas-HSC-I-0-5,4.fits`

    It is also necessary to run

    >>> measureCoaddSources.py $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-R

    to generate the sources catalogs for the HSC-R band required by the next step in the multi-band
    procedure: "MergeMeasurementsTask" .
    """
    _DefaultName = "measureCoaddSources"
    ConfigClass = MeasureMergedCoaddSourcesConfig
    RunnerClass = MeasureMergedCoaddSourcesRunner
    getSchemaCatalogs = _makeGetSchemaCatalogs("meas")
    makeIdFactory = _makeMakeIdFactory("MergedCoaddId")  # The IDs we already have are of this type

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd_calexp",
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=r",
                               ContainerClass=ExistingCoaddDataIdContainer)
        parser.add_argument("--psfCache", type=int, default=100, help="Size of CoaddPsf cache")
        return parser

    def __init__(self, butler=None, schema=None, peakSchema=None, refObjLoader=None, **kwargs):
        """Initialize the task.

        Parameters
        ----------
        Keyword arguments (in addition to those forwarded to CmdLineTask.__init__):
        schema :
            the schema of the merged detection catalog used as input to this one
        peakSchema :
            the schema of the PeakRecords in the Footprints in the merged detection catalog
        refObjLoader :
            an instance of LoadReferenceObjectsTasks that supplies an external reference
            catalog. May be None if the loader can be constructed from the butler argument or all steps
            requiring a reference catalog are disabled.
        butler :
            a butler used to read the input schemas from disk or construct the reference
            catalog loader, if schema or peakSchema or refObjLoader is None

        Notes
        -----
        The task will set its own self.schema attribute to the schema of the output measurement catalog.
        This will include all fields from the input schema, as well as additional fields for all the
        measurements.
        """
        CmdLineTask.__init__(self, **kwargs)
        self.deblended = self.config.inputCatalog.startswith("deblended")
        self.inputCatalog = "Coadd_" + self.config.inputCatalog
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
            if refObjLoader is None:
                assert butler is not None, "Neither butler nor refObjLoader is defined"
            self.makeSubtask("match", butler=butler, refObjLoader=refObjLoader)
        if self.config.doPropagateFlags:
            self.makeSubtask("propagateFlags", schema=self.schema)
        self.schema.checkUnits(parse_strict=self.config.checkUnitsParseStrict)
        if self.config.doApCorr:
            self.makeSubtask("applyApCorr", schema=self.schema)
        if self.config.doRunCatalogCalculation:
            self.makeSubtask("catalogCalculation", schema=self.schema)

    def runDataRef(self, patchRef, psfCache=100):
        """Deblend and measure.

        Parameters
        ----------
        patchRef :
            Patch reference.

        Notes
        -----
        Set 'is-primary' and related flags. Propagate flags
        from individual visits. Optionally match the sources to a reference catalog and write the matches.
        Finally, write the deblended sources and measurements out.
        """
        exposure = patchRef.get(self.config.coaddName + "Coadd_calexp", immediate=True)
        exposure.getPsf().setCacheCapacity(psfCache)
        sources = self.readSources(patchRef)
        table = sources.getTable()
        table.setMetadata(self.algMetadata)  # Capture algorithm metadata to write out to the source catalog.

        self.measurement.run(sources, exposure, exposureId=self.getExposureId(patchRef))

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

        skyInfo = getSkyInfo(coaddName=self.config.coaddName, patchRef=patchRef)
        self.setPrimaryFlags.run(sources, skyInfo.skyMap, skyInfo.tractInfo, skyInfo.patchInfo,
                                 includeDeblend=self.deblended)
        if self.config.doPropagateFlags:
            self.propagateFlags.run(patchRef.getButler(), sources, self.propagateFlags.getCcdInputs(exposure),
                                    exposure.getWcs())
        if self.config.doMatchSources:
            self.writeMatches(patchRef, exposure, sources)
        self.write(patchRef, sources)

    def readSources(self, dataRef):
        """Read input sources.

        Parameters
        ----------
        dataRef :
            Data reference for catalog of merged detections

        Returns
        -------
        sources : `list`
            List of sources in merged catalog

        Notes
        -----
        We also need to add columns to hold the measurements we're about to make
        so we can measure in-place.
        """
        merged = dataRef.get(self.config.coaddName + self.inputCatalog, immediate=True)
        self.log.info("Read %d detections: %s" % (len(merged), dataRef.dataId))
        idFactory = self.makeIdFactory(dataRef)
        for s in merged:
            idFactory.notify(s.getId())
        table = afwTable.SourceTable.make(self.schema, idFactory)
        sources = afwTable.SourceCatalog(table)
        sources.extend(merged, self.schemaMapper)
        return sources

    def writeMatches(self, dataRef, exposure, sources):
        """Write matches of the sources to the astrometric reference catalog.

        We use the Wcs in the exposure to match sources.

        Parameters
        ----------
        dataRef :
            data reference
        exposure :
            exposure with Wcs
        sources :
            source catalog
        """
        result = self.match.run(sources, exposure.getInfo().getFilter().getName())
        if result.matches:
            matches = afwTable.packMatches(result.matches)
            matches.table.setMetadata(result.matchMeta)
            dataRef.put(matches, self.config.coaddName + "Coadd_measMatch")
            if self.config.doWriteMatchesDenormalized:
                denormMatches = denormalizeMatches(result.matches, result.matchMeta)
                dataRef.put(denormMatches, self.config.coaddName + "Coadd_measMatchFull")

    def write(self, dataRef, sources):
        """Write the source catalog.

        Parameters
        ----------
        dataRef :
            data reference
        sources :
            source catalog
        """
        dataRef.put(sources, self.config.coaddName + "Coadd_meas")
        self.log.info("Wrote %d sources: %s" % (len(sources), dataRef.dataId))

    def getExposureId(self, dataRef):
        return int(dataRef.get(self.config.coaddName + "CoaddId"))
