#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import sys
import traceback

import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.skymap import skyMapRegistry

class MakeSkyMapConfig(pexConfig.Config):
    """Config for MakeSkyMapTask
    """
    coaddName = pexConfig.Field(
        doc = "coadd name, e.g. deep, goodSeeing, chiSquared",
        dtype = str,
        default = "deep",
    )
    skyMap = skyMapRegistry.makeField(
        doc = "type of skyMap",
        default = "dodeca",
    )
    doWrite = pexConfig.Field(
        doc = "persist the skyMap?",
        dtype = bool,
        default = True,
    )


class MakeSkyMapTask(pipeBase.CmdLineTask):
    """Make a SkyMap in a repository, setting it up for coaddition
    """
    ConfigClass = MakeSkyMapConfig
    _DefaultName = "makeSkyMap"

    def __init__(self, **kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
    
    @pipeBase.timeMethod
    def run(self, dataRef):
        """Make a skymap
        
        @param dataRef: data reference for sky map; purely used to get hold of a butler
        @return a pipeBase Struct containing:
        - skyMap: the constructed SkyMap
        """
        skyMap = self.config.skyMap.apply()
        self.log.log(self.log.INFO, "sky map has %s tracts" % (len(skyMap),))
        for tractInfo in skyMap:
            wcs = tractInfo.getWcs()
            posBox = afwGeom.Box2D(tractInfo.getBBox())
            pixelPosList = (
                posBox.getMin(),
                afwGeom.Point2D(posBox.getMaxX(), posBox.getMinY()),
                posBox.getMax(),
                afwGeom.Point2D(posBox.getMinX(), posBox.getMaxY()),
            )
            skyPosList = [wcs.pixelToSky(pos).getPosition(afwGeom.degrees) for pos in pixelPosList]
            posStrList = ["(%0.3f, %0.3f)" % tuple(skyPos) for skyPos in skyPosList]
            self.log.log(self.log.INFO, "tract %s has corners %s (RA, Dec deg) and %s x %s patches" % \
                (tractInfo.getId(), ", ".join(posStrList), \
                tractInfo.getNumPatches()[0], tractInfo.getNumPatches()[1]))
        if self.config.doWrite:
            dataRef.put(skyMap)
        return pipeBase.Struct(
            skyMap = skyMap
        )

    @classmethod
    def parseAndRun(cls, args=None, config=None, log=None):
        """Parse an argument list and run the command. This variant does not persist config or metadata.

        @param args: list of command-line arguments; if None use sys.argv
        @param config: config for task (instance of pex_config Config); if None use cls.ConfigClass()
        @param log: log (instance of pex_logging Log); if None use the default log
        
        @return a Struct containing:
        - argumentParser: the argument parser
        - parsedCmd: the parsed command returned by argumentParser.parse_args
        - task: the instantiated task
        The return values are primarily for testing and debugging
        """
        name = cls._DefaultName
        argumentParser = cls._makeArgumentParser()
        if config is None:
            config = cls.ConfigClass()
        parsedCmd = argumentParser.parse_args(config=config, args=args, log=log)
        task = cls(name = name, config = parsedCmd.config, log = parsedCmd.log)
        for dataRef in parsedCmd.dataRefList:
            if parsedCmd.doraise:
                task.run(dataRef)
            else:
                try:
                    task.run(dataRef)
                except Exception, e:
                    task.log.log(task.log.FATAL, "Failed on dataId=%s: %s" % (dataRef.dataId, e))
                    traceback.print_exc(file=sys.stderr)
        return pipeBase.Struct(
            argumentParser = argumentParser,
            parsedCmd = parsedCmd,
            task = task,
        )

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        return SkyMapParser(name=cls._DefaultName, datasetType="deepCoadd_skyMap")

class SkyMapParser(pipeBase.ArgumentParser):
    """A version of lsst.pipe.base.ArgumentParser specialized for making sky maps.
    """
    def _makeDataRefList(self, namespace):
        """Make namespace.dataRefList from namespace.dataIdList
        """
        datasetType = namespace.config.coaddName + "Coadd_skyMap"
        namespace.dataRefList = [
            namespace.butler.dataRef(
                datasetType = datasetType,
                dataId = dict(),
            )
        ]
