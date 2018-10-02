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
from lsst.pipe.base import CmdLineTask, Struct, TaskRunner, ArgumentParser, ButlerInitializedTaskRunner
from lsst.pex.config import Config, Field, ListField, ConfigurableField, RangeField, ConfigField
from lsst.meas.algorithms import DynamicDetectionTask, SkyObjectsTask
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
import lsst.afw.detection as afwDetect
from lsst.daf.base import PropertyList

__all__ = ('DetectCoaddSourcesConfig', 'MergeSourcesRunner', 'MergeSourcesConfig', 'MergeSourcesTask',
           'CullPeaksConfig', 'MergeDetectionsConfig', 'MergeDetectionsTask', 'DeblendCoaddSourcesConfig',
           'DeblendCoaddSourcesRunner', 'DeblendCoaddSourcesTask', 'MeasureMergedCoaddSourcesConfig',
           'MeasureMergedCoaddSourcesRunner', 'MeasureMergedCoaddSourcesTask', 'MergeMeasurementsConfig',
           'MergeMeasurementsTask')


def _makeGetSchemaCatalogs(datasetSuffix):
    """Construct a getSchemaCatalogs instance method

    These are identical for most of the classes here, so we'll consolidate
    the code.

    Parameters
    ----------
    datasetSuffix :
        Suffix of dataset name, e.g., "src" for "deepCoadd_src"
    """

    def getSchemaCatalogs(self):
        """Return a dict of empty catalogs for each catalog dataset produced by this task."""
        src = afwTable.SourceCatalog(self.schema)
        if hasattr(self, "algMetadata"):
            src.getTable().setMetadata(self.algMetadata)
        return {self.config.coaddName + "Coadd_" + datasetSuffix: src}
    return getSchemaCatalogs


def _makeMakeIdFactory(datasetName):
    """Construct a makeIdFactory instance method

    These are identical for all the classes here, so this consolidates
    the code.

    Parameters
    ----------
    datasetName :
        Dataset name without the coadd name prefix, e.g., "CoaddId" for "deepCoaddId"
    """

    def makeIdFactory(self, dataRef):
        """Return an IdFactory for setting the detection identifiers

        The actual parameters used in the IdFactory are provided by
        the butler (through the provided data reference.
        """
        expBits = dataRef.get(self.config.coaddName + datasetName + "_bits")
        expId = int(dataRef.get(self.config.coaddName + datasetName))
        return afwTable.IdFactory.makeSource(expId, 64 - expBits)
    return makeIdFactory


def getShortFilterName(name):
    """Given a longer, camera-specific filter name (e.g. "HSC-I") return its shorthand name ("i").
    """
    # I'm not sure if this is the way this is supposed to be implemented, but it seems to work,
    # and its the only way I could get it to work.
    return afwImage.Filter(name).getFilterProperty().getName()


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

    def __init__(self, schema=None, **kwargs):
        CmdLineTask.__init__(self, **kwargs)
        if schema is None:
            schema = afwTable.SourceTable.makeMinimalSchema()
        if self.config.doInsertFakes:
            self.makeSubtask("insertFakes")
        self.schema = schema
        self.makeSubtask("detection", schema=self.schema)
        if self.config.doScaleVariance:
            self.makeSubtask("scaleVariance")

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
        self.write(exposure, results, patchRef)
        return results

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
        return Struct(sources=sources, backgrounds=backgrounds)

    def write(self, exposure, results, patchRef):
        """Write out results from runDetection.

        exposure:
            Exposure to write out
        results:
            Struct returned from runDetection
        patchRef:
            data reference for patch
        """
        coaddName = self.config.coaddName + "Coadd"
        patchRef.put(results.backgrounds, coaddName + "_calexp_background")
        patchRef.put(results.sources, coaddName + "_det")
        patchRef.put(exposure, coaddName + "_calexp")

##############################################################################################################


class MergeSourcesRunner(TaskRunner):
    """Task runner for the `MergeSourcesTask`

    Required because the run method requires a list of
    dataRefs rather than a single dataRef.
    """
    def makeTask(self, parsedCmd=None, args=None):
        """Provide a butler to the Task constructor.

        Parameters
        ----------
        parsedCmd:
            The parsed command
        args: tuple
            Tuple of a list of data references and kwargs (un-used)

        Raises
        ------
        RuntimeError
            Thrown if both `parsedCmd` & `args` are `None`
        """
        if parsedCmd is not None:
            butler = parsedCmd.butler
        elif args is not None:
            dataRefList, kwargs = args
            butler = dataRefList[0].getButler()
        else:
            raise RuntimeError("Neither parsedCmd or args specified")
        return self.TaskClass(config=self.config, log=self.log, butler=butler)

    @staticmethod
    def buildRefDict(parsedCmd):
        """Build a hierarchical dictionary of patch references

        Parameters
        ----------
        parsedCmd:
            The parsed command

        Returns
        -------
        refDict: dict
            A reference dictionary of the form {patch: {tract: {filter: dataRef}}}

        Raises
        ------
        RuntimeError
            Thrown when multiple references are provided for the same
            combination of tract, patch and filter
        """
        refDict = {}  # Will index this as refDict[tract][patch][filter] = ref
        for ref in parsedCmd.id.refList:
            tract = ref.dataId["tract"]
            patch = ref.dataId["patch"]
            filter = ref.dataId["filter"]
            if tract not in refDict:
                refDict[tract] = {}
            if patch not in refDict[tract]:
                refDict[tract][patch] = {}
            if filter in refDict[tract][patch]:
                raise RuntimeError("Multiple versions of %s" % (ref.dataId,))
            refDict[tract][patch][filter] = ref
        return refDict

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
        return [(list(p.values()), kwargs) for t in refDict.values() for p in t.values()]


class MergeSourcesConfig(Config):
    """Configuration for merging sources.
    """
    priorityList = ListField(dtype=str, default=[],
                             doc="Priority-ordered list of bands for the merge.")
    coaddName = Field(dtype=str, default="deep", doc="Name of coadd")

    def validate(self):
        Config.validate(self)
        if len(self.priorityList) == 0:
            raise RuntimeError("No priority list provided")


class MergeSourcesTask(CmdLineTask):
    """A base class for merging source catalogs.

    Merging detections (MergeDetectionsTask) and merging measurements (MergeMeasurementsTask) are
    so similar that it makes sense to re-use the code, in the form of this abstract base class.

    Parameters
    ----------
    Keyword arguments (in addition to those forwarded to CmdLineTask.__init__) :
    schema :
        the schema of the detection catalogs used as input to this one
    butler :
        a butler used to read the input schema from disk, if schema is None

    Notes
    -----
    Derived classes should use the getInputSchema() method to handle the additional
    arguments and retreive the actual input schema.

    NB: Do not use this class directly. Instead use one of the child classes that inherit from
    MergeSourcesTask such as MergeDetectionsTask "MergeDetectionsTask" or MergeMeasurementsTask
    "MergeMeasurementsTask"

    Sub-classes should set the following class variables:

    - `_DefaultName` : name of Task
    - `inputDataset` : name of dataset to read
    - `outputDataset` : name of dataset to write
    - `getSchemaCatalogs` : to the result of `_makeGetSchemaCatalogs(outputDataset)`

    In addition, sub-classes must implement the run method.
    """
    _DefaultName = None
    ConfigClass = MergeSourcesConfig
    RunnerClass = MergeSourcesRunner
    inputDataset = None
    outputDataset = None
    getSchemaCatalogs = None

    @classmethod
    def _makeArgumentParser(cls):
        """Create a suitable ArgumentParser.

        We will use the ArgumentParser to get a provide a list of data
        references for patches; the RunnerClass will sort them into lists
        of data references for the same patch
        """
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd_" + cls.inputDataset,
                               ContainerClass=ExistingCoaddDataIdContainer,
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=g^r^i")
        return parser

    def getInputSchema(self, butler=None, schema=None):
        """ Obtain the input schema either directly or froma  butler reference.

        Parameters
        ----------
        butler :
            butler reference to obtain the input schema from
        schema :
            the input schema
        """
        if schema is None:
            assert butler is not None, "Neither butler nor schema specified"
            schema = butler.get(self.config.coaddName + "Coadd_" + self.inputDataset + "_schema",
                                immediate=True).schema
        return schema

    def __init__(self, butler=None, schema=None, **kwargs):
        CmdLineTask.__init__(self, **kwargs)

    def runDataRef(self, patchRefList):
        """Merge coadd sources from multiple bands. Calls`run` which must be defined in
        subclasses that inherit from MergeSourcesTask.

        Parameters
        ----------
        patchRefList :
            list of data references for each filter
        """
        catalogs = dict(self.readCatalog(patchRef) for patchRef in patchRefList)
        mergedCatalog = self.run(catalogs, patchRefList[0])
        self.write(patchRefList[0], mergedCatalog)

    def readCatalog(self, patchRef):
        """Read input catalog.

        We read the input dataset provided by the 'inputDataset'
        class variable.

        Parameters
        ----------
        patchRef :
            data reference for patch

        Returns
        -------
        filterName :
        catalog :
        """
        filterName = patchRef.dataId["filter"]
        catalog = patchRef.get(self.config.coaddName + "Coadd_" + self.inputDataset, immediate=True)
        self.log.info("Read %d sources for filter %s: %s" % (len(catalog), filterName, patchRef.dataId))
        return filterName, catalog

    def run(self, catalogs, patchRef):
        """Merge multiple catalogs. This function must be defined in all subclasses that inherit from
        MergeSourcesTask.

        Parameters
        ----------
        catalogs : dict mapping filter name to source catalog

        Returns
        -------
        result :
            merged catalog
        """
        raise NotImplementedError()

    def write(self, patchRef, catalog):
        """Write the output.

        Parameters
        ----------
        patchRef :
            data reference for patch
        catalog :
            catalog

        Notes
        -----
        We write as the dataset provided by the 'outputDataset'
        class variable.
        """
        patchRef.put(catalog, self.config.coaddName + "Coadd_" + self.outputDataset)
        # since the filter isn't actually part of the data ID for the dataset we're saving,
        # it's confusing to see it in the log message, even if the butler simply ignores it.
        mergeDataId = patchRef.dataId.copy()
        del mergeDataId["filter"]
        self.log.info("Wrote merged catalog: %s" % (mergeDataId,))

    def writeMetadata(self, dataRefList):
        """No metadata to write, and not sure how to write it for a list of dataRefs.
        """
        pass


class CullPeaksConfig(Config):
    """Configuration for culling garbage peaks after merging footprints.

    Peaks may also be culled after detection or during deblending; this configuration object
    only deals with culling after merging Footprints.

    Notes
    -----
    These cuts are based on three quantities:

    - nBands: the number of bands in which the peak was detected
    - peakRank: the position of the peak within its family, sorted from brightest to faintest.
    - peakRankNormalized: the peak rank divided by the total number of peaks in the family.

    The formula that identifie peaks to cull is:

    nBands < nBandsSufficient
    AND (rank >= rankSufficient)
    AND (rank >= rankConsider OR rank >= rankNormalizedConsider)

    To disable peak culling, simply set nBandsSufficient=1.
    """

    nBandsSufficient = RangeField(dtype=int, default=2, min=1,
                                  doc="Always keep peaks detected in this many bands")
    rankSufficient = RangeField(dtype=int, default=20, min=1,
                                doc="Always keep this many peaks in each family")
    rankConsidered = RangeField(dtype=int, default=30, min=1,
                                doc=("Keep peaks with less than this rank that also match the "
                                     "rankNormalizedConsidered condition."))
    rankNormalizedConsidered = RangeField(dtype=float, default=0.7, min=0.0,
                                          doc=("Keep peaks with less than this normalized rank that"
                                               " also match the rankConsidered condition."))


class MergeDetectionsConfig(MergeSourcesConfig):
    """Configuration parameters for the MergeDetectionsTask.
    """
    minNewPeak = Field(dtype=float, default=1,
                       doc="Minimum distance from closest peak to create a new one (in arcsec).")

    maxSamePeak = Field(dtype=float, default=0.3,
                        doc="When adding new catalogs to the merge, all peaks less than this distance "
                        " (in arcsec) to an existing peak will be flagged as detected in that catalog.")
    cullPeaks = ConfigField(dtype=CullPeaksConfig, doc="Configuration for how to cull peaks.")

    skyFilterName = Field(dtype=str, default="sky",
                          doc="Name of `filter' used to label sky objects (e.g. flag merge_peak_sky is set)\n"
                          "(N.b. should be in MergeMeasurementsConfig.pseudoFilterList)")
    skyObjects = ConfigurableField(target=SkyObjectsTask, doc="Generate sky objects")

    def setDefaults(self):
        MergeSourcesConfig.setDefaults(self)
        self.skyObjects.avoidMask = ["DETECTED"]  # Nothing else is available in our custom mask


## @addtogroup LSST_task_documentation
## @{
## @page MergeDetectionsTask
## @ref MergeDetectionsTask_ "MergeDetectionsTask"
## @copybrief MergeDetectionsTask
## @}


class MergeDetectionsTask(MergeSourcesTask):
    """Command-line task that merges sources detected in coadds of exposures obtained with different filters.

    To perform photometry consistently across coadds in multiple filter bands, we create a master catalog of
    sources from all bands by merging the sources (peaks & footprints) detected in each coadd, while keeping
    track of which band each source originates in.

    The catalog merge is performed by @ref getMergedSourceCatalog. Spurious peaks detected around bright
    objects are culled as described in @ref CullPeaksConfig.

    Parameters
    ----------

    deepCoadd_det :
        {tract,patch,filter}. SourceCatalog (only parent Footprints)
    deepCoadd_mergeDet :
        {tract,patch}. SourceCatalog (only parent Footprints)
    Data Unit:
        tract, patch
    schema :
        the schema of the detection catalogs used as input to this one
    butler :
        a butler used to read the input schema from disk, if schema is None
    kwargs :
        keyword arguments to be passed to MergeSourcesTask.__init__
        the task will set its own self.schema attribute to the schema of the output merged catalog.

    Notes
    -----
    A FootprintMergeList "FootprintMergeList" will be used to
    merge the source catalogs.

    Examples
    --------
    A complete example of using MergeDetectionsTask

    MergeDetectionsTask is meant to be run after detecting sources in coadds generated for the chosen subset
    of the available bands.
    The purpose of the task is to merge sources (peaks & footprints) detected in the coadds generated from the
    chosen subset of filters.
    Subsequent tasks in the multi-band processing procedure will deblend the generated master list of sources
    and, eventually, perform forced photometry.
    Command-line usage of MergeDetectionsTask expects data references for all the coadds to be processed.
    A list of the available optional arguments can be obtained by calling mergeCoaddDetections.py with the
    `--help` command line argument:

    >>> mergeCoaddDetections.py --help

    To demonstrate usage of the DetectCoaddSourcesTask in the larger context of multi-band processing, we
    will process HSC data in the [ci_hsc](https://github.com/lsst/ci_hsc) package. Assuming one has finished
    step 5 at @ref pipeTasks_multiBand, one may merge the catalogs of sources from each coadd as follows:

    >>> mergeCoaddDetections.py $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-I^HSC-R

    This will merge the HSC-I & -R band parent source catalogs and write the results to
    `$CI_HSC_DIR/DATA/deepCoadd-results/merged/0/5,4/mergeDet-0-5,4.fits`.

    The next step in the multi-band processing procedure is "MeasureMergedCoaddSourcesTask"
    """
    ConfigClass = MergeDetectionsConfig
    _DefaultName = "mergeCoaddDetections"
    inputDataset = "det"
    outputDataset = "mergeDet"
    makeIdFactory = _makeMakeIdFactory("MergedCoaddId")

    def __init__(self, butler=None, schema=None, **kwargs):
        MergeSourcesTask.__init__(self, butler=butler, schema=schema, **kwargs)
        self.makeSubtask("skyObjects")
        self.schema = self.getInputSchema(butler=butler, schema=schema)

        filterNames = [getShortFilterName(name) for name in self.config.priorityList]
        filterNames += [self.config.skyFilterName]
        self.merged = afwDetect.FootprintMergeList(self.schema, filterNames)

    def run(self, catalogs, patchRef):
        """Merge multiple catalogs.


        After ordering the catalogs and filters in priority order,
        getMergedSourceCatalog of the FootprintMergeList "FootprintMergeList" created by
        \_\_init\_\_ is used to perform the actual merging. Finally, cullPeaks is used to remove
        garbage peaks detected around bright objects.

        Parameters
        ----------
        catalogs :
        patchRef :
        mergedList :
        """

        # Convert distance to tract coordinate
        skyInfo = getSkyInfo(coaddName=self.config.coaddName, patchRef=patchRef)
        tractWcs = skyInfo.wcs
        peakDistance = self.config.minNewPeak / tractWcs.getPixelScale().asArcseconds()
        samePeakDistance = self.config.maxSamePeak / tractWcs.getPixelScale().asArcseconds()

        # Put catalogs, filters in priority order
        orderedCatalogs = [catalogs[band] for band in self.config.priorityList if band in catalogs.keys()]
        orderedBands = [getShortFilterName(band) for band in self.config.priorityList
                        if band in catalogs.keys()]

        mergedList = self.merged.getMergedSourceCatalog(orderedCatalogs, orderedBands, peakDistance,
                                                        self.schema, self.makeIdFactory(patchRef),
                                                        samePeakDistance)

        #
        # Add extra sources that correspond to blank sky
        #
        skySeed = patchRef.get(self.config.coaddName + "MergedCoaddId")
        skySourceFootprints = self.getSkySourceFootprints(mergedList, skyInfo, skySeed)
        if skySourceFootprints:
            key = mergedList.schema.find("merge_footprint_%s" % self.config.skyFilterName).key
            for foot in skySourceFootprints:
                s = mergedList.addNew()
                s.setFootprint(foot)
                s.set(key, True)

        # Sort Peaks from brightest to faintest
        for record in mergedList:
            record.getFootprint().sortPeaks()
        self.log.info("Merged to %d sources" % len(mergedList))
        # Attempt to remove garbage peaks
        self.cullPeaks(mergedList)
        return mergedList

    def cullPeaks(self, catalog):
        """Attempt to remove garbage peaks (mostly on the outskirts of large blends).

        Parameters
        ----------
        catalog Source catalog
        """
        keys = [item.key for item in self.merged.getPeakSchema().extract("merge_peak_*").values()]
        assert len(keys) > 0, "Error finding flags that associate peaks with their detection bands."
        totalPeaks = 0
        culledPeaks = 0
        for parentSource in catalog:
            # Make a list copy so we can clear the attached PeakCatalog and append the ones we're keeping
            # to it (which is easier than deleting as we iterate).
            keptPeaks = parentSource.getFootprint().getPeaks()
            oldPeaks = list(keptPeaks)
            keptPeaks.clear()
            familySize = len(oldPeaks)
            totalPeaks += familySize
            for rank, peak in enumerate(oldPeaks):
                if ((rank < self.config.cullPeaks.rankSufficient) or
                    (sum([peak.get(k) for k in keys]) >= self.config.cullPeaks.nBandsSufficient) or
                    (rank < self.config.cullPeaks.rankConsidered and
                     rank < self.config.cullPeaks.rankNormalizedConsidered * familySize)):
                    keptPeaks.append(peak)
                else:
                    culledPeaks += 1
        self.log.info("Culled %d of %d peaks" % (culledPeaks, totalPeaks))

    def getSchemaCatalogs(self):
        """
        Return a dict of empty catalogs for each catalog dataset produced by this task.

        Parameters
        ----------
        result : `dict`
            dictionary of empty catalogs
        """
        mergeDet = afwTable.SourceCatalog(self.schema)
        peak = afwDetect.PeakCatalog(self.merged.getPeakSchema())
        return {self.config.coaddName + "Coadd_mergeDet": mergeDet,
                self.config.coaddName + "Coadd_peak": peak}

    def getSkySourceFootprints(self, mergedList, skyInfo, seed):
        """Return a list of Footprints of sky objects which don't overlap with anything in mergedList

        Parameters
        ----------
        mergedList :
            The merged Footprints from all the input bands
        skyInfo :
            A description of the patch
        seed :
            Seed for the random number generator
        """
        mask = afwImage.Mask(skyInfo.patchInfo.getOuterBBox())
        detected = mask.getPlaneBitMask("DETECTED")
        for s in mergedList:
            s.getFootprint().spans.setMask(mask, detected)

        footprints = self.skyObjects.run(mask, seed)
        if not footprints:
            return footprints

        # Need to convert the peak catalog's schema so we can set the "merge_peak_<skyFilterName>" flags
        schema = self.merged.getPeakSchema()
        mergeKey = schema.find("merge_peak_%s" % self.config.skyFilterName).key
        converted = []
        for oldFoot in footprints:
            assert len(oldFoot.getPeaks()) == 1, "Should be a single peak only"
            peak = oldFoot.getPeaks()[0]
            newFoot = afwDetect.Footprint(oldFoot.spans, schema)
            newFoot.addPeak(peak.getFx(), peak.getFy(), peak.getPeakValue())
            newFoot.getPeaks()[0].set(mergeKey, True)
            converted.append(newFoot)

        return converted


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


class MergeMeasurementsConfig(MergeSourcesConfig):
    """Configuration parameters for the MergeMeasurementsTask
    """
    pseudoFilterList = ListField(dtype=str, default=["sky"],
                                 doc="Names of filters which may have no associated detection\n"
                                     "(N.b. should include MergeDetectionsConfig.skyFilterName)")
    snName = Field(dtype=str, default="base_PsfFlux",
                   doc="Name of flux measurement for calculating the S/N when choosing the reference band.")
    minSN = Field(dtype=float, default=10.,
                  doc="If the S/N from the priority band is below this value (and the S/N "
                      "is larger than minSNDiff compared to the priority band), use the band with "
                      "the largest S/N as the reference band.")
    minSNDiff = Field(dtype=float, default=3.,
                      doc="If the difference in S/N between another band and the priority band is larger "
                      "than this value (and the S/N in the priority band is less than minSN) "
                      "use the band with the largest S/N as the reference band")
    flags = ListField(dtype=str, doc="Require that these flags, if available, are not set",
                      default=["base_PixelFlags_flag_interpolatedCenter", "base_PsfFlux_flag",
                               "ext_photometryKron_KronFlux_flag", "modelfit_CModel_flag", ])

## @addtogroup LSST_task_documentation
## @{
## @page MergeMeasurementsTask
## @ref MergeMeasurementsTask_ "MergeMeasurementsTask"
## @copybrief MergeMeasurementsTask
## @}


class MergeMeasurementsTask(MergeSourcesTask):
    """Command-line task that merges measurements from multiple bands.

    Combines consistent (i.e. with the same peaks and footprints) catalogs of sources from multiple filter
    bands to construct a unified catalog that is suitable for driving forced photometry. Every source is
    required to have centroid, shape and flux measurements in each band.

    Parameters
    ----------
    deepCoadd_meas{tract,patch,filter} :
        SourceCatalog
    deepCoadd_ref{tract,patch} :
        SourceCatalog
    Data Unit:
        tract, patch
    kwargs :
        Additional keyword arguments (forwarded to MergeSourcesTask.__init__)
    schema :
        the schema of the detection catalogs used as input to this one
    butler :
        a butler used to read the input schema from disk, if schema is None

    Notes
    -----
    The task will set its own self.schema attribute to the schema of the output merged catalog.

    MergeMeasurementsTask is meant to be run after deblending & measuring sources in every band.
    The purpose of the task is to generate a catalog of sources suitable for driving forced photometry in
    coadds and individual exposures.
    Command-line usage of MergeMeasurementsTask expects a data reference to the coadds to be processed. A list
    of the available optional arguments can be obtained by calling mergeCoaddMeasurements.py with the `--help`
    command line argument:

    >>> mergeCoaddMeasurements.py --help


    To demonstrate usage of the DetectCoaddSourcesTask in the larger context of multi-band processing, we
    will process HSC data in the [ci_hsc](https://github.com/lsst/ci_hsc) package. Assuming one has finished
    step 7 at @ref pipeTasks_multiBand, one may merge the catalogs generated after deblending and measuring
    as follows:

    >>> mergeCoaddMeasurements.py $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-I^HSC-R

    This will merge the HSC-I & HSC-R band catalogs. The results are written in
    `$CI_HSC_DIR/DATA/deepCoadd-results/`.
    """
    _DefaultName = "mergeCoaddMeasurements"
    ConfigClass = MergeMeasurementsConfig
    inputDataset = "meas"
    outputDataset = "ref"
    getSchemaCatalogs = _makeGetSchemaCatalogs("ref")

    def __init__(self, butler=None, schema=None, **kwargs):
        MergeSourcesTask.__init__(self, butler=butler, schema=schema, **kwargs)
        inputSchema = self.getInputSchema(butler=butler, schema=schema)
        self.schemaMapper = afwTable.SchemaMapper(inputSchema, True)
        self.schemaMapper.addMinimalSchema(inputSchema, True)
        self.instFluxKey = inputSchema.find(self.config.snName + "_instFlux").getKey()
        self.instFluxErrKey = inputSchema.find(self.config.snName + "_instFluxErr").getKey()
        self.fluxFlagKey = inputSchema.find(self.config.snName + "_flag").getKey()

        self.flagKeys = {}
        for band in self.config.priorityList:
            short = getShortFilterName(band)
            outputKey = self.schemaMapper.editOutputSchema().addField(
                "merge_measurement_%s" % short,
                type="Flag",
                doc="Flag field set if the measurements here are from the %s filter" % band
            )
            peakKey = inputSchema.find("merge_peak_%s" % short).key
            footprintKey = inputSchema.find("merge_footprint_%s" % short).key
            self.flagKeys[band] = Struct(peak=peakKey, footprint=footprintKey, output=outputKey)
        self.schema = self.schemaMapper.getOutputSchema()

        self.pseudoFilterKeys = []
        for filt in self.config.pseudoFilterList:
            try:
                self.pseudoFilterKeys.append(self.schema.find("merge_peak_%s" % filt).getKey())
            except Exception as e:
                self.log.warn("merge_peak is not set for pseudo-filter %s: %s" % (filt, e))

        self.badFlags = {}
        for flag in self.config.flags:
            try:
                self.badFlags[flag] = self.schema.find(flag).getKey()
            except KeyError as exc:
                self.log.warn("Can't find flag %s in schema: %s" % (flag, exc,))

    def run(self, catalogs, patchRef):
        """Merge measurement catalogs to create a single reference catalog for forced photometry

        Parameters
        ----------
        catalogs :
            the catalogs to be merged
        patchRef :
            patch reference for data

        Notes
        -----
        For parent sources, we choose the first band in config.priorityList for which the
        merge_footprint flag for that band is is True.

        For child sources, the logic is the same, except that we use the merge_peak flags.
        """
        # Put catalogs, filters in priority order
        orderedCatalogs = [catalogs[band] for band in self.config.priorityList if band in catalogs.keys()]
        orderedKeys = [self.flagKeys[band] for band in self.config.priorityList if band in catalogs.keys()]

        mergedCatalog = afwTable.SourceCatalog(self.schema)
        mergedCatalog.reserve(len(orderedCatalogs[0]))

        idKey = orderedCatalogs[0].table.getIdKey()
        for catalog in orderedCatalogs[1:]:
            if numpy.any(orderedCatalogs[0].get(idKey) != catalog.get(idKey)):
                raise ValueError("Error in inputs to MergeCoaddMeasurements: source IDs do not match")

        # This first zip iterates over all the catalogs simultaneously, yielding a sequence of one
        # record for each band, in priority order.
        for orderedRecords in zip(*orderedCatalogs):

            maxSNRecord = None
            maxSNFlagKeys = None
            maxSN = 0.
            priorityRecord = None
            priorityFlagKeys = None
            prioritySN = 0.
            hasPseudoFilter = False

            # Now we iterate over those record-band pairs, keeping track of the priority and the
            # largest S/N band.
            for inputRecord, flagKeys in zip(orderedRecords, orderedKeys):
                parent = (inputRecord.getParent() == 0 and inputRecord.get(flagKeys.footprint))
                child = (inputRecord.getParent() != 0 and inputRecord.get(flagKeys.peak))

                if not (parent or child):
                    for pseudoFilterKey in self.pseudoFilterKeys:
                        if inputRecord.get(pseudoFilterKey):
                            hasPseudoFilter = True
                            priorityRecord = inputRecord
                            priorityFlagKeys = flagKeys
                            break
                    if hasPseudoFilter:
                        break

                isBad = any(inputRecord.get(flag) for flag in self.badFlags)
                if isBad or inputRecord.get(self.fluxFlagKey) or inputRecord.get(self.instFluxErrKey) == 0:
                    sn = 0.
                else:
                    sn = inputRecord.get(self.instFluxKey)/inputRecord.get(self.instFluxErrKey)
                if numpy.isnan(sn) or sn < 0.:
                    sn = 0.
                if (parent or child) and priorityRecord is None:
                    priorityRecord = inputRecord
                    priorityFlagKeys = flagKeys
                    prioritySN = sn
                if sn > maxSN:
                    maxSNRecord = inputRecord
                    maxSNFlagKeys = flagKeys
                    maxSN = sn

            # If the priority band has a low S/N we would like to choose the band with the highest S/N as
            # the reference band instead.  However, we only want to choose the highest S/N band if it is
            # significantly better than the priority band.  Therefore, to choose a band other than the
            # priority, we require that the priority S/N is below the minimum threshold and that the
            # difference between the priority and highest S/N is larger than the difference threshold.
            #
            # For pseudo code objects we always choose the first band in the priority list.
            bestRecord = None
            bestFlagKeys = None
            if hasPseudoFilter:
                bestRecord = priorityRecord
                bestFlagKeys = priorityFlagKeys
            elif (prioritySN < self.config.minSN and (maxSN - prioritySN) > self.config.minSNDiff and
                  maxSNRecord is not None):
                bestRecord = maxSNRecord
                bestFlagKeys = maxSNFlagKeys
            elif priorityRecord is not None:
                bestRecord = priorityRecord
                bestFlagKeys = priorityFlagKeys

            if bestRecord is not None and bestFlagKeys is not None:
                outputRecord = mergedCatalog.addNew()
                outputRecord.assign(bestRecord, self.schemaMapper)
                outputRecord.set(bestFlagKeys.output, True)
            else:  # if we didn't find any records
                raise ValueError("Error in inputs to MergeCoaddMeasurements: no valid reference for %s" %
                                 inputRecord.getId())

        # more checking for sane inputs, since zip silently iterates over the smallest sequence
        for inputCatalog in orderedCatalogs:
            if len(mergedCatalog) != len(inputCatalog):
                raise ValueError("Mismatch between catalog sizes: %s != %s" %
                                 (len(mergedCatalog), len(orderedCatalogs)))

        return mergedCatalog
