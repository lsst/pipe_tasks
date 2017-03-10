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
from __future__ import absolute_import, division, print_function
import sys
import traceback

import lsst.afw.geom as afwGeom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.skymap import skyMapRegistry


class MakeSkyMapConfig(pexConfig.Config):
    """Config for MakeSkyMapTask
    """
    coaddName = pexConfig.Field(
        doc="coadd name, e.g. deep, goodSeeing, chiSquared",
        dtype=str,
        default="deep",
    )
    skyMap = skyMapRegistry.makeField(
        doc="type of skyMap",
        default="dodeca",
    )
    doWrite = pexConfig.Field(
        doc="persist the skyMap? If False then run generates the sky map and returns it, "
        + "but does not save it to the data repository",
        dtype=bool,
        default=True,
    )


class MakeSkyMapRunner(pipeBase.TaskRunner):
    """Only need a single butler instance to run on."""
    @staticmethod
    def getTargetList(parsedCmd):
        return [parsedCmd.butler]

    def precall(self, parsedCmd):
        # We overload to disable writing/checking of schemas and configs.
        # There's only one SkyMap per rerun anyway, so the config is redundant,
        # and checking it means we can't overwrite or append to one once we've
        # written it.
        return True

    def __call__(self, butler):
        task = self.TaskClass(config=self.config, log=self.log)
        if self.doRaise:
            results = task.run(butler)
        else:
            try:
                results = task.run(butler)
            except Exception as e:
                task.log.fatal("Failed: %s" % e)
                if not isinstance(e, pipeBase.TaskError):
                    traceback.print_exc(file=sys.stderr)
        task.writeMetadata(butler)
        if self.doReturnResults:
            return results


class MakeSkyMapTask(pipeBase.CmdLineTask):
    """!Make a sky map in a repository

    Making a sky map in a repository is a prerequisite for making a coadd,
    since the sky map is used as the pixelization for the coadd.
    """
    ConfigClass = MakeSkyMapConfig
    _DefaultName = "makeSkyMap"
    RunnerClass = MakeSkyMapRunner

    def __init__(self, **kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)

    @pipeBase.timeMethod
    def run(self, butler):
        """!Make a skymap, persist it (optionally) and log some information about it

        @param[in]   butler  data butler
        @return      a pipeBase Struct containing:
                     - skyMap: the constructed SkyMap
        """
        skyMap = self.config.skyMap.apply()
        self.logSkyMapInfo(skyMap)
        if self.config.doWrite:
            butler.put(skyMap, self.config.coaddName + "Coadd_skyMap")
        return pipeBase.Struct(
            skyMap=skyMap
        )

    def logSkyMapInfo(self, skyMap):
        """!Log information about a sky map

        @param[in] skyMap  sky map (an lsst.skyMap.SkyMap)
        """
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
            self.log.info("tract %s has corners %s (RA, Dec deg) and %s x %s patches" %
                          (tractInfo.getId(), ", ".join(posStrList),
                           tractInfo.getNumPatches()[0], tractInfo.getNumPatches()[1]))

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser

        No identifiers are added because none are used.
        """
        return pipeBase.ArgumentParser(name=cls._DefaultName)

    def _getConfigName(self):
        """Return the name of the config dataset
        """
        return "%s_makeSkyMap_config" % (self.config.coaddName,)

    def _getMetadataName(self):
        """Return the name of the metadata dataset
        """
        return "%s_makeSkyMap_metadata" % (self.config.coaddName,)
