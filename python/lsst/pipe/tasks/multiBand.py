import numpy

from lsst.pipe.base import CmdLineTask, Struct, TaskRunner, ArgumentParser, ButlerInitializedTaskRunner
from lsst.pex.config import Config, Field, ListField, ConfigurableField, RangeField, ConfigField
from lsst.meas.algorithms import SourceDetectionTask, SourceMeasurementTask
from lsst.meas.deblender import SourceDeblendTask
from lsst.pipe.tasks.coaddBase import getSkyInfo, ExistingCoaddDataIdContainer
from lsst.pipe.tasks.astrometry import AstrometryTask
from lsst.pipe.tasks.setPrimaryFlags import SetPrimaryFlagsTask
from lsst.pipe.tasks.propagateVisitFlags import PropagateVisitFlagsTask
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.afw.detection as afwDetect
from lsst.daf.base import PropertyList

"""
New dataset types:
* deepCoadd_det: detections from what used to be processCoadd (tract, patch, filter)
* deepCoadd_mergeDet: merged detections (tract, patch)
* deepCoadd_meas: measurements of merged detections (tract, patch, filter)
* deepCoadd_ref: reference sources (tract, patch)
All of these have associated *_schema catalogs that require no data ID and hold no records.

In addition, we have a schema-only dataset, which saves the schema for the PeakRecords in
the mergeDet, meas, and ref dataset Footprints:
* deepCoadd_peak_schema
"""


def _makeGetSchemaCatalogs(datasetSuffix):
    """Construct a getSchemaCatalogs instance method

    These are identical for most of the classes here, so we'll consolidate
    the code.

    datasetSuffix:  Suffix of dataset name, e.g., "src" for "deepCoadd_src"
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

    datasetName:  Dataset name without the coadd name prefix, e.g., "CoaddId" for "deepCoaddId"
    """
    def makeIdFactory(self, dataRef):
        """Return an IdFactory for setting the detection identifiers

        The actual parameters used in the IdFactory are provided by
        the butler (through the provided data reference.
        """
        expBits = dataRef.get(self.config.coaddName + datasetName + "_bits")
        expId = long(dataRef.get(self.config.coaddName + datasetName))
        return afwTable.IdFactory.makeSource(expId, 64 - expBits)
    return makeIdFactory


def copySlots(oldCat, newCat):
    """Copy table slots definitions from one catalog to another"""
    for name in ("Centroid", "Shape", "ApFlux", "ModelFlux", "PsfFlux", "InstFlux", "CalibFlux"):
        meas = getattr(oldCat.table, "get" + name + "Key")()
        err = getattr(oldCat.table, "get" + name + "ErrKey")()
        flag = getattr(oldCat.table, "get" + name + "FlagKey")()
        getattr(newCat.table, "define" + name)(meas, err, flag)


def getShortFilterName(name):
    """Given a longer, camera-specific filter name (e.g. "HSC-I") return its shorthand name ("i").
    """
    # I'm not sure if this is the way this is supposed to be implemented, but it seems to work,
    # and its the only way I could get it to work.
    return afwImage.Filter(name).getFilterProperty().getName()


##############################################################################################################

class DetectCoaddSourcesConfig(Config):
    doScaleVariance = Field(dtype=bool, default=True, doc="Scale variance plane using empirical noise?")
    detection = ConfigurableField(target=SourceDetectionTask, doc="Source detection")
    coaddName = Field(dtype=str, default="deep", doc="Name of coadd")

    def setDefaults(self):
        Config.setDefaults(self)
        self.detection.thresholdType = "pixel_stdev"
        self.detection.isotropicGrow = True
        # Coadds are made from background-subtracted CCDs, so background subtraction should be very basic
        self.detection.background.useApprox = False
        self.detection.background.binSize = 4096
        self.detection.background.undersampleStyle = 'REDUCE_INTERP_ORDER'


class DetectCoaddSourcesTask(CmdLineTask):
    """Detect sources on a coadd

    This operation is performed separately in each band.  The detections from
    each band will be merged before performing the measurement stage.
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
        """Initialize the task.

        Keyword arguments (in addition to those forwarded to CmdLineTask.__init__):
         - schema: the initial schema for the output catalog, modified-in place to include all
                   fields set by this task.  If None, the source minimal schema will be used.
        """
        CmdLineTask.__init__(self, **kwargs)
        if schema is None:
            schema = afwTable.SourceTable.makeMinimalSchema()
        self.schema = schema
        self.makeSubtask("detection", schema=self.schema)

    def run(self, patchRef):
        exposure = patchRef.get(self.config.coaddName + "Coadd", immediate=True)
        results = self.runDetection(exposure, self.makeIdFactory(patchRef))
        self.write(results, patchRef)
        patchRef.put(exposure, self.config.coaddName + "Coadd_calexp")
        return results

    def scaleVariance(self, maskedImage):
        """Scale the variance in a maskedImage

        Scales the variance plane to match the measured variance.
        """
        ctrl = afwMath.StatisticsControl()
        ctrl.setAndMask(~0x0)
        var    = maskedImage.getVariance()
        mask   = maskedImage.getMask()
        dstats = afwMath.makeStatistics(maskedImage, afwMath.VARIANCECLIP, ctrl).getValue()
        vstats = afwMath.makeStatistics(var, mask, afwMath.MEANCLIP, ctrl).getValue()
        vrat   = dstats / vstats
        self.log.info("Renormalising variance by %f" % (vrat))
        var   *= vrat

    def runDetection(self, exposure, idFactory):
        """Run detection on an exposure

        exposure: Exposure on which to detect
        idFactory: IdFactory to set source identifiers

        Returns: Struct(sources: catalog of detections,
                        backgrounds: list of backgrounds
                        )
        """
        if self.config.doScaleVariance:
            self.scaleVariance(exposure.getMaskedImage())
        backgrounds = afwMath.BackgroundList()
        table = afwTable.SourceTable.make(self.schema, idFactory)
        detections = self.detection.makeSourceCatalog(table, exposure)
        sources = detections.sources
        fpSets = detections.fpSets
        if fpSets.background:
            backgrounds.append(fpSets.background)
        return Struct(sources=sources, backgrounds=backgrounds)

    def write(self, results, patchRef):
        """Write out results from runDetection

        results: Struct returned from runDetection
        patchRef: data reference for patch
        """
        coaddName = self.config.coaddName + "Coadd"
        patchRef.put(results.backgrounds, coaddName + "_calexpBackground")
        patchRef.put(results.sources, coaddName + "_det")


##############################################################################################################

class MergeSourcesRunner(TaskRunner):
    def makeTask(self, parsedCmd=None, args=None):
        """Provide a butler to the Task constructor"""
        if parsedCmd is not None:
            butler = parsedCmd.butler
        elif args is not None:
            dataRefList, kwargs = args
            butler = dataRefList[0].getButler()
        else:
            raise RuntimeError("Neither parsedCmd or args specified")
        return self.TaskClass(config=self.config, log=self.log, butler=butler)

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Provide a list of patch references for each patch

        The patch references within the list will have different filters.
        """
        refList = {} # Will index this as refList[tract][patch][filter] = ref
        for ref in parsedCmd.id.refList:
            tract = ref.dataId["tract"]
            patch = ref.dataId["patch"]
            filter = ref.dataId["filter"]
            if not tract in refList:
                refList[tract] = {}
            if not patch in refList[tract]:
                refList[tract][patch] = {}
            if filter in refList[tract][patch]:
                raise RuntimeError("Multiple versions of %s" % (ref.dataId,))
            refList[tract][patch][filter] = ref
        return [(p.values(), kwargs) for t in refList.itervalues() for p in t.itervalues()]


class MergeSourcesConfig(Config):
    priorityList = ListField(dtype=str, default=[],
                             doc="Priority-ordered list of bands for the merge.")
    coaddName = Field(dtype=str, default="deep", doc="Name of coadd")

    def validate(self):
        Config.validate(self)
        if len(self.priorityList) == 0:
            raise RuntimeError("No priority list provided")


class MergeSourcesTask(CmdLineTask):
    """A base class for merging source catalogs

    Merging detections (MergeDetectionsTask) and merging measurements
    (MergeMeasurementsTask) are currently so similar that it makes
    sense to re-use the code, in the form of this abstract base class.

    Sub-classes should set the following class variables:
    * _DefaultName: name of Task
    * inputDataset: name of dataset to read
    * outputDataset: name of dataset to write
    * getSchemaCatalogs to the output of _makeGetSchemaCatalogs(outputDataset)

    In addition, sub-classes must implement the mergeCatalogs method.
    """
    _DefaultName = None
    ConfigClass = MergeSourcesConfig
    RunnerClass = MergeSourcesRunner
    inputDataset = None
    outputDataset = None
    getSchemaCatalogs = None

    @classmethod
    def _makeArgumentParser(cls):
        """Create a suitable ArgumentParser

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
        if schema is None:
            assert butler is not None, "Neither butler nor schema specified"
            schema = butler.get(self.config.coaddName + "Coadd_" + self.inputDataset + "_schema",
                                immediate=True).schema
        return schema

    def __init__(self, butler=None, schema=None, **kwargs):
        """Initialize the task.

        Keyword arguments (in addition to those forwarded to CmdLineTask.__init__):
         - schema: the schema of the detection catalogs used as input to this one
         - butler: a butler used to read the input schema from disk, if schema is None

        Derived classes should use the getInputSchema() method to handle the additional
        arguments and retreive the actual input schema.
        """
        CmdLineTask.__init__(self, **kwargs)

    def run(self, patchRefList):
        """Merge coadd sources from multiple bands

        patchRefList: list of patch data reference for each filter
        """
        catalogs = dict(self.readCatalog(patchRef) for patchRef in patchRefList)
        mergedCatalog = self.mergeCatalogs(catalogs, patchRefList[0])
        self.write(patchRefList[0], mergedCatalog)

    def readCatalog(self, patchRef):
        """Read input catalog

        We read the input dataset provided by the 'inputDataset'
        class variable.
        """
        filterName = patchRef.dataId["filter"]
        catalog = patchRef.get(self.config.coaddName + "Coadd_" + self.inputDataset, immediate=True)
        self.log.info("Read %d sources for filter %s: %s" % (len(catalog), filterName, patchRef.dataId))
        return filterName, catalog

    def mergeCatalogs(self, catalogs, patchRef):
        """Merge multiple catalogs

        catalogs: dict mapping filter name to source catalog

        Returns: merged catalog
        """
        raise NotImplementedError()

    def write(self, patchRef, catalog):
        """Write the output

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
        """No metadata to write, and not sure how to write it for a list of dataRefs"""
        pass


class CullPeaksConfig(Config):
    """Configuration for culling garbage peaks after merging Footprints.

    Peaks may also be culled after detection or during deblending; this configuration object
    only deals with culling after merging Footprints.

    These cuts are based on three quantities:
     - nBands: the number of bands in which the peak was detected
     - peakRank: the position of the peak within its family, sorted from brightest to faintest.
     - peakRankNormalized: the peak rank divided by the total number of peaks in the family.

    The formula that identifie peaks to cull is:

      nBands < nBandsSufficient
        AND (rank >= rankSufficient)
        AND (rank >= rankConsider OR rank >= rankNormalizedConsider)

    To disable peak culling, simply set nBandsSafe=1.
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
    minNewPeak = Field(dtype=float, default=1,
                       doc="Minimum distance from closest peak to create a new one (in arcsec).")

    maxSamePeak = Field(dtype=float, default=0.3,
                        doc="When adding new catalogs to the merge, all peaks less than this distance "
                        " (in arcsec) to an existing peak will be flagged as detected in that catalog.")
    cullPeaks = ConfigField(dtype=CullPeaksConfig, doc="Configuration for how to cull peaks.")


class MergeDetectionsTask(MergeSourcesTask):
    """Merge detections from multiple bands"""
    ConfigClass = MergeDetectionsConfig
    _DefaultName = "mergeCoaddDetections"
    inputDataset = "det"
    outputDataset = "mergeDet"
    makeIdFactory = _makeMakeIdFactory("MergedCoaddId")

    def __init__(self, butler=None, schema=None, **kwargs):
        """Initialize the task.

        Additional keyword arguments (forwarded to MergeSourcesTask.__init__):
         - schema: the schema of the detection catalogs used as input to this one
         - butler: a butler used to read the input schema from disk, if schema is None

        The task will set its own self.schema attribute to the schema of the output merged catalog.
        """
        MergeSourcesTask.__init__(self, butler=butler, schema=schema, **kwargs)
        self.schema = self.getInputSchema(butler=butler, schema=schema)
        self.merged = afwDetect.FootprintMergeList(
            self.schema,
            [getShortFilterName(name) for name in self.config.priorityList]
        )

    def mergeCatalogs(self, catalogs, patchRef):
        """Merge multiple catalogs
        """

        # Convert distance to tract coordiante
        skyInfo = getSkyInfo(coaddName=self.config.coaddName, patchRef=patchRef)
        tractWcs = skyInfo.wcs
        peakDistance = self.config.minNewPeak / tractWcs.pixelScale().asArcseconds()
        samePeakDistance = self.config.maxSamePeak / tractWcs.pixelScale().asArcseconds()

        # Put catalogs, filters in priority order
        orderedCatalogs = [catalogs[band] for band in self.config.priorityList if band in catalogs.keys()]
        orderedBands = [getShortFilterName(band) for band in self.config.priorityList
                        if band in catalogs.keys()]

        mergedList = self.merged.getMergedSourceCatalog(orderedCatalogs, orderedBands, peakDistance,
                                                        self.schema, self.makeIdFactory(patchRef),
                                                        samePeakDistance)
        copySlots(orderedCatalogs[0], mergedList)
        # Sort Peaks from brightest to faintest
        for record in mergedList:
            record.getFootprint().sortPeaks()
        self.log.info("Merged to %d sources" % len(mergedList))
        # Attempt to remove garbage peaks
        self.cullPeaks(mergedList)
        return mergedList

    def cullPeaks(self, catalog):
        """Attempt to remove garbage peaks (mostly on the outskirts of large blends)"""
        keys = [item.key for item in self.merged.getPeakSchema().extract("merge.peak.*").itervalues()]
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
                    (self.config.cullPeaks.nBandsSufficient > 1 and
                     sum([peak.get(k) for k in keys]) >= self.config.cullPeaks.nBandsSufficient) or
                    (rank < self.config.cullPeaks.rankConsidered and
                     rank < self.config.cullPeaks.rankNormalizedConsidered * familySize)):
                    keptPeaks.append(peak)
                else:
                    culledPeaks += 1
        self.log.info("Culled %d of %d peaks" % (culledPeaks, totalPeaks))

    def getSchemaCatalogs(self):
        """Return a dict of empty catalogs for each catalog dataset produced by this task."""
        mergeDet = afwTable.SourceCatalog(self.schema)
        peak = afwDetect.PeakCatalog(self.merged.getPeakSchema())
        return {self.config.coaddName + "Coadd_mergeDet": mergeDet,
                self.config.coaddName + "Coadd_peak": peak}

##############################################################################################################

class MeasureMergedCoaddSourcesConfig(Config):
    doDeblend = Field(dtype=bool, default=True, doc="Deblend sources?")
    deblend = ConfigurableField(target=SourceDeblendTask, doc="Deblend sources")
    measurement = ConfigurableField(target=SourceMeasurementTask, doc="Source measurement")
    setPrimaryFlags = ConfigurableField(target=SetPrimaryFlagsTask, doc="Set flags for primary tract/patch")
    doPropagateFlags = Field(
        dtype=bool, default=True,
        doc="Whether to match sources to CCD catalogs to propagate flags (to e.g. identify PSF stars)"
    )
    propagateFlags = ConfigurableField(target=PropagateVisitFlagsTask, doc="Propagate visit flags to coadd")
    doMatchSources = Field(dtype=bool, default=True, doc="Match sources to reference catalog?")
    astrometry = ConfigurableField(target=AstrometryTask, doc="Astrometric matching")
    coaddName = Field(dtype=str, default="deep", doc="Name of coadd")

    def setDefaults(self):
        Config.setDefaults(self)
        self.deblend.propagateAllPeaks = True
        self.astrometry.forceKnownWcs=True
        self.astrometry.solver.calculateSip=False

class MeasureMergedCoaddSourcesTask(CmdLineTask):
    """Measure sources using the merged catalog of detections

    This operation is performed separately on each band.  We deblend and measure on
    the list of merge detections.  The results from each band will subsequently
    be merged to create a final reference catalog for forced measurement.
    """
    _DefaultName = "measureCoaddSources"
    ConfigClass = MeasureMergedCoaddSourcesConfig
    RunnerClass = ButlerInitializedTaskRunner
    getSchemaCatalogs = _makeGetSchemaCatalogs("meas")
    makeIdFactory = _makeMakeIdFactory("MergedCoaddId") # The IDs we already have are of this type

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd_calexp",
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=r",
                               ContainerClass=ExistingCoaddDataIdContainer)
        return parser

    def __init__(self, butler=None, schema=None, peakSchema=None, **kwargs):
        """Initialize the task.

        Keyword arguments (in addition to those forwarded to CmdLineTask.__init__):
         - schema: the schema of the merged detection catalog used as input to this one
         - peakSchema: the schema of the PeakRecords in the Footprints in the merged detection catalog
         - butler: a butler used to read the input schemas from disk, if schema or peakSchema is None

        The task will set its own self.schema attribute to the schema of the output measurement catalog.
        This will include all fields from the input schema, as well as additional fields for all the
        measurements.
        """
        CmdLineTask.__init__(self, **kwargs)
        if schema is None:
            assert butler is not None, "Neither butler nor schema is defined"
            schema = butler.get(self.config.coaddName + "Coadd_mergeDet_schema", immediate=True).schema
        self.schemaMapper = afwTable.SchemaMapper(schema)
        self.schemaMapper.addMinimalSchema(schema)
        self.schema = self.schemaMapper.getOutputSchema()
        self.algMetadata = PropertyList()
        if self.config.doDeblend:
            if peakSchema is None:
                assert butler is not None, "Neither butler nor peakSchema is defined"
                peakSchema = butler.get(self.config.coaddName + "Coadd_peak_schema", immediate=True).schema
            self.makeSubtask("deblend", schema=self.schema, peakSchema=peakSchema)
        self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)
        self.makeSubtask("setPrimaryFlags", schema=self.schema)
        if self.config.doPropagateFlags:
            self.makeSubtask("propagateFlags", schema=self.schema)
        if self.config.doMatchSources:
            self.makeSubtask("astrometry", schema=self.schema)

    def run(self, patchRef):
        """Measure and deblend"""
        exposure = patchRef.get(self.config.coaddName + "Coadd_calexp", immediate=True)
        sources = self.readSources(patchRef)
        if self.config.doDeblend:
            self.deblend.run(exposure, sources, exposure.getPsf())

            bigKey = sources.schema["deblend.parent-too-big"].asKey()
            numBig = sum((s.get(bigKey) for s in sources)) # catalog is non-contiguous so can't extract column
            if numBig > 0:
                self.log.warn("Patch %s contains %d large footprints that were not deblended" %
                              (patchRef.dataId, numBig))
        self.measurement.run(exposure, sources)
        skyInfo = getSkyInfo(coaddName=self.config.coaddName, patchRef=patchRef)
        self.setPrimaryFlags.run(sources, skyInfo.skyMap, skyInfo.tractInfo, skyInfo.patchInfo,
                                 includeDeblend=self.config.doDeblend)
        if self.config.doPropagateFlags:
            self.propagateFlags.run(patchRef.getButler(), sources, self.propagateFlags.getCcdInputs(exposure),
                                    exposure.getWcs())
        if self.config.doMatchSources:
            self.writeMatches(patchRef, exposure, sources)
        self.write(patchRef, sources)

    def readSources(self, dataRef):
        """Read input sources

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

    def writeMatches(self, dataRef, exposure, sources):
        """Write matches of the sources to the astrometric reference catalog

        We use the Wcs in the exposure to match sources.

        dataRef: data reference
        exposure: exposure with Wcs
        sources: source catalog
        """
        result = self.astrometry.astrometer.useKnownWcs(sources, exposure=exposure)
        if result.matches:
            matches = afwTable.packMatches(result.matches)
            matches.table.setMetadata(result.matchMetadata)
            dataRef.put(matches, self.config.coaddName + "Coadd_srcMatch")

    def write(self, dataRef, sources):
        """Write the source catalog"""
        dataRef.put(sources, self.config.coaddName + "Coadd_meas")
        self.log.info("Wrote %d sources: %s" % (len(sources), dataRef.dataId))


##############################################################################################################

class MergeMeasurementsTask(MergeSourcesTask):
    """Measure measurements from multiple bands"""
    _DefaultName = "mergeCoaddMeasurements"
    inputDataset = "meas"
    outputDataset = "ref"
    getSchemaCatalogs = _makeGetSchemaCatalogs("ref")

    def __init__(self, butler=None, schema=None, **kwargs):
        """Initialize the task.

        Additional keyword arguments (forwarded to MergeSourcesTask.__init__):
         - schema: the schema of the detection catalogs used as input to this one
         - butler: a butler used to read the input schema from disk, if schema is None

        The task will set its own self.schema attribute to the schema of the output merged catalog.
        """
        MergeSourcesTask.__init__(self, butler=butler, schema=schema, **kwargs)
        inputSchema = self.getInputSchema(butler=butler, schema=schema)
        self.schemaMapper = afwTable.SchemaMapper(inputSchema)
        self.schemaMapper.addMinimalSchema(inputSchema, True)
        self.flagKeys = {}
        for band in self.config.priorityList:
            short = getShortFilterName(band)
            outputKey = self.schemaMapper.editOutputSchema().addField(
                "merge.measurement.%s" % short,
                type="Flag",
                doc="Flag field set if the measurements here are from the %s filter" % band
            )
            peakKey = inputSchema.find("merge.peak.%s" % short).key
            footprintKey = inputSchema.find("merge.footprint.%s" % short).key
            self.flagKeys[band] = Struct(peak=peakKey, footprint=footprintKey, output=outputKey)
        self.schema = self.schemaMapper.getOutputSchema()

    def mergeCatalogs(self, catalogs, patchRef):
        """Merge measurement catalogs to create a single reference catalog for forced photometry

        For parent sources, we choose the first band in config.priorityList for which the
        merge.footprint flag for that band is is True.

        For child sources, the logic is the same, except that we use the merge.peak flags.
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
        # record for each band, in order.
        for n, orderedRecords in enumerate(zip(*orderedCatalogs)):
            # Now we iterate over those record-band pairs, until we find the one with the right flag set.
            for inputRecord, flagKeys in zip(orderedRecords, orderedKeys):
                bestParent = (inputRecord.getParent() == 0 and inputRecord.get(flagKeys.footprint))
                bestChild = (inputRecord.getParent() != 0 and inputRecord.get(flagKeys.peak))
                if bestParent or bestChild:
                    outputRecord = mergedCatalog.addNew()
                    outputRecord.assign(inputRecord, self.schemaMapper)
                    outputRecord.set(flagKeys.output, True)
                    break
            else: # if we didn't break (i.e. didn't find any record with right flag set)
                raise ValueError("Error in inputs to MergeCoaddMeasurements: no valid reference for %s" %
                                 inputRecord.getId())

        copySlots(orderedCatalogs[0], mergedCatalog)

        # more checking for sane inputs, since zip silently iterates over the smallest sequence
        for inputCatalog in orderedCatalogs:
            if len(mergedCatalog) != len(inputCatalog):
                raise ValueError("Mismatch between catalog sizes: %s != %s" %
                                 (len(mergedCatalog), len(orderedCatalogs)))

        return mergedCatalog
