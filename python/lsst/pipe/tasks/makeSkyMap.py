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
        self.log.info("sky map has %s tracts" % (len(skyMap),))
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
            self.log.info("tract %s has corners %s (RA, Dec deg) and %s x %s patches" % \
                (tractInfo.getId(), ", ".join(posStrList), \
                tractInfo.getNumPatches()[0], tractInfo.getNumPatches()[1]))
        if self.config.doWrite:
            dataRef.put(skyMap)
        return pipeBase.Struct(
            skyMap = skyMap
        )

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        return SkyMapParser(name=cls._DefaultName, datasetType="deepCoadd_skyMap")
    
    def _getConfigName(self):
        """Return the name of the config dataset
        """
        return "%s_makeSkyMap_config" % (self.config.coaddName,)
    
    def _getMetadataName(self):
        """Return the name of the metadata dataset
        """
        return "%s_makeSkyMap_metadata" % (self.config.coaddName,)


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
