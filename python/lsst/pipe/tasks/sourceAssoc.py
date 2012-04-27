#
# LSST Data Management System
# Copyright 2012 LSST Corporation.
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
# see <http://www.lsstcorp.org/LegalNotices/>.
#

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.skypix as skypix
import lsst.ap.match as apMatch
import lsst.ap.utils as apUtils
import lsst.ap.cluster as apCluster

from .sourceAssocArgumentParser import SourceAssocArgumentParser

__all__ = ["SourceAssocConfig", "SourceAssocTask"]


class SourceAssocConfig(pexConfig.Config):
    """Config for SourceAssocTask.
    """
    sourceProcessing = pexConfig.ConfigField(
        dtype=apCluster.SourceProcessingConfig,
        doc="source processing parameters")

    clustering = pexConfig.ConfigField(
        dtype=apCluster.ClusteringConfig,
        doc="source clustering parameters")
    doCluster = pexConfig.Field(
        dtype=bool, default=True,
        doc="Cluster sources?")
    doDiscardNoiseClusters = pexConfig.Field(
        dtype=bool, default=True,
        doc="Discard single source clusters?")
    doWriteClusters = pexConfig.Field(
        dtype=bool, default=True,
        doc="Write source clusters?")

    algorithmFlags = pexConfig.DictField(
        keytype=str, itemtype=str,
        doc="A dictionary mapping from algorithm names to strings "
            "containing comma separated lists of flag field names. "
            "If any flag is set for a source, then that source is "
            "ignored when computing the measurement mean of the "
            "corresponding algorithm.")

    doWriteSources = pexConfig.Field(
        dtype=bool, default=True,
        doc="Write sources?")
    doWriteBadSources = pexConfig.Field(
        dtype=bool, default=True,
        doc="Write bad sources?")
    doWriteInvalidSources = pexConfig.Field(
        dtype=bool, default=True,
        doc="Write invalid sources?")

    sourceHistogramResolution = pexConfig.RangeField(
        dtype=int, default=2000, min=1,
        doc="X and Y resolution of source position histograms")
    doMakeSourceHistogram = pexConfig.Field(
        dtype=bool, default=True,
        doc="Make 2D histogram of source positions?")
    doMakeBadSourceHistogram = pexConfig.Field(
        dtype=bool, default=True,
        doc="Make 2D histogram of bad source positions?")
    doWriteSourceHistogram = pexConfig.Field(
        dtype=bool, default=True,
        doc="Write source histogram?")
    doWriteBadSourceHistogram = pexConfig.Field(
        dtype=bool, default=True,
        doc="Write bad source histogram?")

    def setDefaults(self):
        self.sourceProcessing.badFlagFields = ["flags.negative",
                                               "flags.pixel.edge",
                                               "shape.sdss.flags.unweightedbad",
                                              ]
        flags = ",".join(["flags.negative",
                          "flags.badcentroid",
                          "flags.pixel.edge",
                          "flags.pixel.interpolated.center",
                          "flags.pixel.saturated.center",
                         ])
        self.algorithmFlags = {
            "flux.gaussian": "flux.gaussian.flags,flux.gaussian.flags.badapcorr," + flags,
            "flux.naive":    "flux.naive.flags," + flags,
            "flux.psf":      "flux.psf.flags," + flags,
            "flux.sinc":     "flux.sinc.flags," + flags,
            "shape.sdss":    "shape.sdss.flags.unweightedbad," + flags,
        }


def _flagKeys(schema, config, alg):
    """Create an lsst.afw.table.FlagKeyVector identifying sources to
       ignore when computing measurement means for the given algorithm.
    """
    vec = afwTable.FlagKeyVector()
    if alg in config.algorithmFlags:
        flags = config.algorithmFlags[alg]
        for f in flags.split(","):
            f = f.strip()
            if len(f) == 0:
                continue
            si = schema.find(f)
            if si.field.getTypeString() != "Flag":
                raise TypeError(f + " field is not a Flag field")
            vec.append(si.key) 
    return vec


class SourceAssocTask(pipeBase.CmdLineTask):
    """Cluster the sources belonging to a sky-tile, and compute
       cluster attributes.
    """
    ConfigClass = SourceAssocConfig
    _DefaultName = "sourceAssoc"

    def __init__(self, *args, **kwds):
        pipeBase.Task.__init__(self, *args, **kwds)

    @pipeBase.timeMethod
    def cluster(self, skyTileId, butler):
        """Cluster sources falling inside the given sky-tile.

           @param skyTileId: Integer sky-tile ID
           @param butler:    Butler responsible for retrieving calibrated
                             exposure metadata and associated sources

           @return A lsst.pipe.base.Struct with the following fields:

                   - sources:            Sources inside the sky-tile with valid
                                         positions and no "bad" flags set.
                   - badSources:         Sources inside the sky-tile with valid
                                         positions and at least one "bad" flag set.
                   - invalidSources:     Sources with invalid positions/centroids.
                   - clusters:           A list of lsst.afw.table.SourceCatalog objects,
                                         one per cluster generated. 
                   - exposures:          An lsst.ap.match.ExposureInfoMap,
                                         mapping calibrated exposure IDs to
                                         lsst.ap.match.ExposureInfo objects.
                   - sourceHistogram:    A 2D histogram of source positions.
                   - badSourceHistogram: A 2D histogram of bad source positions.

                   Note that any of these return values can legitimately be None
                   due to lack of inputs, problems reading them in, or the task
                   configuration.
        """
        # sky-tile setup
        qsp = skypix.createQuadSpherePixelization(butler.mapper.skypolicy)
        root, x, y = qsp.coords(skyTileId);
        skyTile = apUtils.PT1SkyTile(qsp.resolution, root, x, y, skyTileId)
        del root, x, y
        spControl = self.config.sourceProcessing.makeControl()
        sourceTable = None
        keys = butler.getKeys(datasetType="src").keys()
        results = pipeBase.Struct(
            sources = None,
            badSources = None,
            invalidSources = None,
            clusters = None,
            exposures = apMatch.ExposureInfoMap(),
            sourceHistogram = None,
            badSourceHistogram = None
        )

        for values in butler.queryMetadata("raw", "sensor", keys, skyTile=skyTileId):
            dataId = dict(zip(keys, values))
            if not butler.datasetExists("src", dataId):
                continue
            try:
                expMd = butler.get("calexp_md", dataId, immediate=True)
                expSources = butler.get("src", dataId, immediate=True)
            except:
                self.log.warn(str.format(
                    "skipping {} : failed to unpersist src or calexp_md dataset",
                    str(dataId)))
                continue
            if sourceTable == None:
                # create output source table
                sourceTable, schemaMapper = apCluster.makeOutputSourceTable(
                    expSources.getTable(), spControl)
                # create output source catalogs
                results.sources = afwTable.SourceCatalog(sourceTable)
                results.badSources = afwTable.SourceCatalog(sourceTable)
                results.invalidSources = afwTable.SourceCatalog(sourceTable)
            # process sources: segregate into "good", "bad", and "invalid"
            # sets, discard sources outside sky-tile, and denormalize the
            # source schema.
            expInfo = apMatch.ExposureInfo(expMd)
            results.exposures.insert(expInfo)
            apCluster.processSources(
                expSources,
                expInfo,
                skyTile,
                spControl,
                schemaMapper,
                results.sources,
                results.badSources,
                results.invalidSources)
            # in hopes of freeing memory occupied by input sources
            del expSources, expMd

        if (sourceTable == None):
            return results # nothing to do
        # create clusters
        if self.config.doCluster and len(results.sources) > 0:
            results.clusters = apCluster.cluster(
                results.sources, self.config.clustering.makeControl())
        # create good/bad source histograms
        if self.config.doMakeSourceHistogram and len(results.sources) > 0:
            results.sourceHistogram, wcs = apUtils.createImageCoveringSkyTile(
                qsp, skyTileId, self.config.sourceHistogramResolution)
            apUtils.makeSourceHistogram(
                results.sourceHistogram.getImage(), results.sources, wcs, False)
        if self.config.doMakeBadSourceHistogram and len(results.badSources) > 0:
            results.badSourceHistogram, wcs = apUtils.createImageCoveringSkyTile(
                qsp, skyTileId, self.config.sourceHistogramResolution)
            apUtils.makeSourceHistogram(
                results.badSourceHistogram.getImage(), results.badSources, wcs, False)
        return results

    @pipeBase.timeMethod
    def attributes(self, skyTileId, clusters, exposures):
        """Compute source cluster attributes for a sky-tile.

           @param skyTileId: Integer sky-tile ID
           @param clusters:  List of lsst.afw.table.SourceCatalog objects,
                             each containing the sources for one cluster
           @param exposures: A lsst.ap.match.ExposureInfoMap object, mapping
                             calibrated exposure IDs to lsst.ap.match.ExposureInfo
                             objects.

           @return An lsst.ap.cluster.SourceClusterCatalog containing measurement
                   means for each cluster.
        """
        if len(clusters) == 0:
            return None
        self.log.info(str.format("Computing attributes for {} clusters", len(clusters)))
        spControl = self.config.sourceProcessing.makeControl()
        minNeighbors = self.config.clustering.makeControl().minNeighbors
        scTable = apCluster.makeSourceClusterTable(
            clusters[0].getTable(),
            apCluster.SourceClusterIdFactory(skyTileId),
            spControl)
        flagsNoiseKey = scTable.getSchema().find("flags.noise").key
        scCat = apCluster.SourceClusterCatalog(scTable)
        algorithmFlags = dict()
        for alg in spControl.fluxFields:
            algorithmFlags[alg] = _flagKeys(clusters[0].getSchema(), self.config, alg)
        for alg in spControl.shapeFields:
            algorithmFlags[alg] = _flagKeys(clusters[0].getSchema(), self.config, alg)
        numNoise = 0
        for sources in clusters:
            if len(sources) == 1 and minNeighbors > 0:
                numNoise += 1
                if self.config.doDiscardNoiseClusters:
                    continue
                else:
                    sc = scCat.addNew()
                    sc.set(flagsNoiseKey, True)
            else:
                sc = scCat.addNew()
            sev = apCluster.computeBasicAttributes(
                sc, sources, exposures, spControl.exposurePrefix)
            for alg in spControl.fluxFields:
                apCluster.computeFluxMean(sc, sev, alg, algorithmFlags[alg],
                                          spControl.fluxScale)
            for alg in spControl.shapeFields:
                apCluster.computeShapeMean(sc, sev, alg, algorithmFlags[alg]) 
            apCluster.setClusterFields(sources, sc, spControl)
        msg = "Computed attributes for {} clusters"
        if self.config.doDiscardNoiseClusters:
            msg += ", discarded {} noise clusters"
        else:
            msg += ", including {} noise clusters"
        self.log.info(str.format(msg, len(scCat), numNoise))
        return scCat

    @pipeBase.timeMethod
    def run(self, skyTileId, butler):
        """Run source association on a single sky-tile; return None.
        """
        self.log.info(str.format("Processing sky-tile {}", skyTileId))
        res = self.cluster(skyTileId, butler)
        if (self.config.doCluster and res.clusters != None and
            len(res.clusters) > 0):
            clusters = self.attributes(skyTileId, res.clusters, res.exposures)
            # persist clusters
            if self.config.doWriteClusters and len(clusters) > 0:
                butler.put(clusters, "object", skyTile=skyTileId)
        # persist sources
        if (self.config.doWriteSources and res.sources != None and
            len(res.sources) > 0):
            butler.put(res.sources, "source", skyTile=skyTileId)
        if (self.config.doWriteBadSources and res.badSources != None and
            len(res.badSources) > 0):
            butler.put(res.badSources, "badSource", skyTile=skyTileId)
        if (self.config.doWriteInvalidSources and res.invalidSources != None and
            len(res.invalidSources) > 0):
            butler.put(res.invalidSources, "invalidSource", skyTile=skyTileId)
        # persist source histograms
        if self.config.doWriteSourceHistogram and res.sourceHistogram != None:
            butler.put(res.sourceHistogram, "sourceHist", skyTile=skyTileId)
        if (self.config.doWriteBadSourceHistogram and
            res.badSourceHistogram != None):
            butler.put(res.badSourceHistogram, "badSourceHist", skyTile=skyTileId)

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        return SourceAssocArgumentParser(name=cls._DefaultName)

    @classmethod
    def parseAndRun(cls, args=None, config=None, log=None):
        """Parse argument list and run command.

            @param args:   list of command-line arguments;
                           if None use sys.argv
            @param config: config for task (instance of lsst.pex.config.Config);
                           if None use cls.ConfigClass()
            @param log:    log (instance of lsst.pex.logging.Log);
                           if None use the default log
        """
        argumentParser = cls._makeArgumentParser()
        if config is None:
            config = cls.ConfigClass()
        parsedCmd = argumentParser.parse_args(config=config, args=args, log=log)
        task = cls(name = cls._DefaultName, config = parsedCmd.config, log = parsedCmd.log)
        task.config.save(str.format("{}_config.py", task.getName()))
        if not hasattr(parsedCmd, "skyTileIds") or len(parsedCmd.skyTileIds) == 0:
            parsedCmd.skyTileIds = parsedCmd.butler.queryMetadata("raw", "skyTile")
        for skyTileId in parsedCmd.skyTileIds:
            task.run(skyTileId, parsedCmd.butler)
 
