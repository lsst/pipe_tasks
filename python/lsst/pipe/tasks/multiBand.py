from lsst.pipe.base import CmdLineTask, Struct, TaskRunner, ArgumentParser, ButlerInitializedTaskRunner
from lsst.pex.config import Config, Field, ListField, ConfigurableField
from lsst.meas.algorithms import SourceDetectionTask, SourceMeasurementTask
from lsst.meas.deblender import SourceDeblendTask
from lsst.pipe.tasks.coaddBase import getSkyInfo, ExistingCoaddDataIdContainer
from lsst.pipe.tasks.astrometry import AstrometryTask
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
"""


def _makeGetSchemaCatalogs(datasetSuffix):
    """Construct a getSchemaCatalogs instance method

    These are identical for all the classes here, so we'll consolidate
    the code.

    datasetSuffix:  Suffix of dataset name, e.g., "src" for "deepCoadd_src"
    """
    def getSchemaCatalogs(self):
        """Return a dict of empty catalogs for each catalog dataset produced by this task."""
        src = afwTable.SourceCatalog(self.schema)
        src.getTable().setMetadata(self.algMetadata)
        return {self.config.coaddName + "Coadd_" + datasetSuffix: src}
    return getSchemaCatalogs


##############################################################################################################

class DetectCoaddSourcesConfig(Config):
    doScaleVariance = Field(dtype=bool, default=True, doc="Scale variance plane using empirical noise?")
    detection = ConfigurableField(target=SourceDetectionTask, doc="Source detection")
    coaddName = Field(dtype=str, default="deep", doc="Name of coadd")

    def setDefaults(self):
        Config.setDefaults(self)
        self.detection.thresholdType = "pixel_stdev"
        self.detection.isotropicGrow = True
        self.detection.background.undersampleStyle = 'REDUCE_INTERP_ORDER'


class DetectCoaddSourcesTask(CmdLineTask):
    """Detect sources on a coadd

    This operation is performed separately in each band.  The detections from
    each band will be merged before performing the measurement stage.
    """
    _DefaultName = "detect"
    ConfigClass = DetectCoaddSourcesConfig
    getSchemaCatalogs = _makeGetSchemaCatalogs("det")

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd", help="data ID, e.g. --id tract=12345 patch=1,2 filter=r",
                               ContainerClass=ExistingCoaddDataIdContainer)
        return parser

    def __init__(self, **kwargs):
        CmdLineTask.__init__(self, **kwargs)
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = PropertyList()
        self.makeSubtask("detection", schema=self.schema)

    def makeIdFactory(self, dataRef):
        """Return an IdFactory for setting the detection identifiers

        The actual parameters used in the IdFactory are provided by
        the butler (through the provided data reference.
        """
        expBits = dataRef.get(self.config.coaddName + "CoaddId_bits")
        expId = long(dataRef.get(self.config.coaddName + "CoaddId"))
        return afwTable.IdFactory.makeSource(expId, 64 - expBits)

    def run(self, patchRef):
        """Run detection on a coadd"""
        exposure = patchRef.get(self.config.coaddName + "Coadd", immediate=True)
        if self.config.doScaleVariance:
            self.scaleVariance(exposure.getMaskedImage())
        results = self.runDetection(exposure, self.makeIdFactory(patchRef))
        self.write(results, patchRef)

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
        backgrounds = afwMath.BackgroundList()
        table = afwTable.SourceTable.make(self.schema, idFactory)
        table.setMetadata(self.algMetadata)
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
    sense to re-use the code, in the form of this base class.
    This base class uses a drop-dead simple implementation for the
    merge: it simply takes the catalog with the highest priority.

    Sub-classes should set the following class variables:
    * inputDataset: name of dataset to read
    * outputDataset: name of dataset to write
    * refColumn: name of column to add
    * getSchemaCatalogs to the output of _makeGetSchemaCatalogs(outputDataset)
    """
    _DefaultName = "merge"
    ConfigClass = MergeSourcesConfig
    RunnerClass = MergeSourcesRunner
    inputDataset = None
    outputDataset = None
    refColumn = None
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

    def __init__(self, butler, **kwargs):
        CmdLineTask.__init__(self, **kwargs)
        inSchema = butler.get(self.config.coaddName + "Coadd_" + self.inputDataset + "_schema",
                              immediate=True).schema
        self.schemaMapper = afwTable.SchemaMapper(inSchema)
        self.schemaMapper.addMinimalSchema(inSchema)
        self.schema = self.schemaMapper.getOutputSchema()
        self.algMetadata = PropertyList()
        self.refKey = self.schema.addField(self.refColumn, type=str,
                                           size=max(len(s) for s in self.config.priorityList),
                                           doc="Band used for reference")


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

        This implementation is a merge placeholder for something hard.
        It simply takes the catalog for the band with the highest priority.

        We add a field (specified by the 'refColumn' class variable) to this
        catalog with the band that each individual source came from.

        catalogs: dict mapping filter name to source catalog

        Returns: merged catalog
        """
        best = None
        for f in self.config.priorityList:
            if f in catalogs:
                best = f
                break
        else:
            raise RuntimeError("No overlap between catalogs (%s) and priority list (%s)" %
                               (catalogs.keys(), self.config.priorityList))

        # Tell people where things came from
        merged = afwTable.SourceCatalog(self.schema)
        merged.extend(catalogs[best], self.schemaMapper)
        # Can't set a string column; do it row by row
        for s in catalogs[best]:
            s[self.refKey] = best
        self.log.info("Merged to %d sources" % len(merged))
        return merged

    def write(self, patchRef, catalog):
        """Write the output

        We write as the dataset provided by the 'outputDataset'
        class variable.
        """
        patchRef.put(catalog, self.config.coaddName + "Coadd_" + self.outputDataset)
        self.log.info("Wrote merged catalog: %s" % (patchRef.dataId,))

    def writeMetadata(self, dataRefList):
        """No metadata to write, and not sure how to write it for a list of dataRefs"""
        pass


class MergeDetectionsConfig(MergeSourcesConfig):
    minNewPeak = Field(dtype=float, default=1,
                       doc="Minimum distance from closest peak to create a new one (in arcsec).")


class MergeDetectionsTask(MergeSourcesTask):
    """Merge detections from multiple bands"""
    ConfigClass = MergeDetectionsConfig
    _DefaultName = "mergeDet"
    inputDataset = "det"
    outputDataset = "mergeDet"
    refColumn = "detection.ref"
    getSchemaCatalogs = _makeGetSchemaCatalogs("mergeDet")

    def __init__(self, butler, **kwargs):
        MergeSourcesTask.__init__(self, butler, **kwargs)
        self.schemaMerge = afwTable.SourceTable.makeMinimalSchema()
        self.merged = afwDetect.FootprintMergeList(self.schemaMerge, self.config.priorityList)

    def makeIdFactory(self, dataRef):
        """Return an IdFactory for setting the detection identifiers

        The actual parameters used in the IdFactory are provided by
        the butler (through the provided data reference).
        """
        expBits = dataRef.get(self.config.coaddName + "mergedCoaddId_bits")
        expId = long(dataRef.get(self.config.coaddName + "mergedCoaddId"))
        return afwTable.IdFactory.makeSource(expId, 64 - expBits)

    def mergeCatalogs(self, catalogs, patchRef):
        """Merge multiple catalogs
        """

        # Convert distance to tract coordiante
        skyInfo = getSkyInfo(coaddName=self.config.coaddName, patchRef=patchRef)
        tractWcs = skyInfo.wcs
        peakDistance = self.config.minNewPeak / tractWcs.pixelScale().asArcseconds()

        # Put catalogs, filters in priority order
        orderedCatalogs = [catalogs[band] for band in self.config.priorityList if band in catalogs.keys()]
        orderedBands = [band for band in self.config.priorityList if band in catalogs.keys()]

        mergedList = self.merged.getMergedSourceCatalog(orderedCatalogs, orderedBands, peakDistance,
                                                        self.schemaMerge, self.makeIdFactory(patchRef))
        self.log.info("Merged to %d sources" % len(mergedList))
        return mergedList

##############################################################################################################

class MeasureMergedCoaddSourcesConfig(Config):
    doDeblend = Field(dtype=bool, default=True, doc="Deblend sources?")
    deblend = ConfigurableField(target=SourceDeblendTask, doc="Deblend sources")
    measurement = ConfigurableField(target=SourceMeasurementTask, doc="Source measurement")
    doMatchSources = Field(dtype=bool, default=True, doc="Match sources to reference catalog?")
    astrometry = ConfigurableField(target=AstrometryTask, doc="Astrometric matching")
    coaddName = Field(dtype=str, default="deep", doc="Name of coadd")

class MeasureMergedCoaddSourcesTask(CmdLineTask):
    """Measure sources using the merged catalog of detections

    This operation is performed separately on each band.  We deblend and measure on
    the list of merge detections.  The results from each band will subsequently
    be merged to create a final reference catalog for forced measurement.
    """
    _DefaultName = "measure"
    ConfigClass = MeasureMergedCoaddSourcesConfig
    RunnerClass = ButlerInitializedTaskRunner
    getSchemaCatalogs = _makeGetSchemaCatalogs("meas")

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd", help="data ID, e.g. --id tract=12345 patch=1,2 filter=r",
                               ContainerClass=ExistingCoaddDataIdContainer)
        return parser

    def __init__(self, butler, **kwargs):
        CmdLineTask.__init__(self, **kwargs)
        detSchema = butler.get(self.config.coaddName + "Coadd_mergeDet_schema", immediate=True).schema
        self.schemaMapper = afwTable.SchemaMapper(detSchema)
        self.schemaMapper.addMinimalSchema(detSchema)
        self.schema = self.schemaMapper.getOutputSchema()
        self.algMetadata = PropertyList()
        if self.config.doDeblend:
            self.makeSubtask("deblend", schema=self.schema)
        self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)
        self.makeSubtask("astrometry", schema=self.schema)

        self.isPatchInnerKey = self.schema.addField(
            "detect.is-patch-inner", type="Flag",
            doc="true if source is in the inner region of a coadd patch",
        )
        self.isTractInnerKey = self.schema.addField(
            "detect.is-tract-inner", type="Flag",
            doc="true if source is in the inner region of a coadd tract",
        )
        self.isPrimaryKey = self.schema.addField(
            "detect.is-primary", type="Flag",
            doc="true if source has no children and is in the inner region of a coadd patch " \
                + "and is in the inner region of a coadd tract",
        )

    def run(self, patchRef):
        """Measure and deblend"""
        exposure = patchRef.get(self.config.coaddName + "Coadd", immediate=True)
        sources = self.readSources(patchRef)
        if self.config.doDeblend:
            self.deblend.run(exposure, sources, exposure.getPsf())
        self.measurement.run(exposure, sources)
        self.setIsPrimaryFlag(sources, patchRef)
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
        sources = afwTable.SourceCatalog(self.schema)
        sources.extend(merged, self.schemaMapper)
        return sources

    def setIsPrimaryFlag(self, sources, patchRef):
        """Set is-primary and related flags on sources

        sources: a SourceTable
            - reads centroid fields and an nChild field
            - writes is-patch-inner, is-tract-inner and is-primary flags
        patchRef: a patch reference (for retrieving sky info)

        Will raise RuntimeError if self.config.doDeblend and the nChild key is not found in the table
        """
        skyInfo = getSkyInfo(coaddName=self.config.coaddName, patchRef=patchRef)

        # Test for the presence of the nchild key instead of checking config.doDeblend because sources
        # might be unpersisted with deblend info, even if deblending is not run again.
        nChildKeyName = "deblend.nchild"
        try:
            nChildKey = self.schema.find(nChildKeyName).key
        except Exception:
            nChildKey = None

        if self.config.doDeblend and nChildKey is None:
            # deblending was run but the nChildKey was not found; this suggests a variant deblender
            # was used that we cannot use the output from, or some obscure error.
            raise RuntimeError("Ran the deblender but cannot find %r in the source table" % (nChildKeyName,))

        # set inner flags for each source and set primary flags for sources with no children
        # (or all sources if deblend info not available)
        innerFloatBBox = afwGeom.Box2D(skyInfo.patchInfo.getInnerBBox())
        tractId = skyInfo.tractInfo.getId()
        for source in sources:
            if source.getCentroidFlag():
                # centroid unknown, so leave the inner and primary flags False
                continue

            centroidPos = source.getCentroid()
            # Skip source whose centroidPos is nan
            # I do not know why this can happen (NY)
            if centroidPos[0] != centroidPos[0] or centroidPos[1] != centroidPos[1]:
                continue
            isPatchInner = innerFloatBBox.contains(centroidPos)
            source.setFlag(self.isPatchInnerKey, isPatchInner)

            skyPos = source.getCoord()
            sourceInnerTractId = skyInfo.skyMap.findTract(skyPos).getId()
            isTractInner = sourceInnerTractId == tractId
            source.setFlag(self.isTractInnerKey, isTractInner)

            if nChildKey is None or source.get(nChildKey) == 0:
                source.setFlag(self.isPrimaryKey, isPatchInner and isTractInner)

    def writeMatches(self, dataRef, exposure, sources):
        """Write matches of the sources to the astrometric reference catalog

        We use the Wcs in the exposure to match sources.

        dataRef: data reference
        exposure: exposure with Wcs
        sources: source catalog
        """
        result = self.astrometry.astrometer.useKnownWcs(sources, exposure=exposure)
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
    _DefaultName = "mergeMeas"
    inputDataset = "meas"
    outputDataset = "ref"
    refColumn = "measurement.ref"
    getSchemaCatalogs = _makeGetSchemaCatalogs("ref")
