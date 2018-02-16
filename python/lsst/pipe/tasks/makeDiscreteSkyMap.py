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
import lsst.geom

import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.skymap import DiscreteSkyMap, BaseSkyMap
from lsst.pipe.base import ArgumentParser


class MakeDiscreteSkyMapConfig(pexConfig.Config):
    """Config for MakeDiscreteSkyMapTask
    """
    coaddName = pexConfig.Field(
        doc="coadd name, e.g. deep, goodSeeing, chiSquared",
        dtype=str,
        default="deep",
    )
    skyMap = pexConfig.ConfigField(
        dtype=BaseSkyMap.ConfigClass,
        doc="SkyMap configuration parameters, excluding position and radius"
    )
    borderSize = pexConfig.Field(
        doc="additional border added to the bounding box of the calexps, in degrees",
        dtype=float,
        default=0.0
    )
    doAppend = pexConfig.Field(
        doc="append another tract to an existing DiscreteSkyMap on disk, if present?",
        dtype=bool,
        default=False
    )
    doWrite = pexConfig.Field(
        doc="persist the skyMap?",
        dtype=bool,
        default=True,
    )

    def setDefaults(self):
        self.skyMap.tractOverlap = 0.0


class MakeDiscreteSkyMapRunner(pipeBase.TaskRunner):
    """Run a task with all dataRefs at once, rather than one dataRef at a time.

    Call the run method of the task using two positional arguments:
    - butler: data butler
    - dataRefList: list of all dataRefs,
    """
    @staticmethod
    def getTargetList(parsedCmd):
        return [(parsedCmd.butler, parsedCmd.id.refList)]

    def __call__(self, args):
        """
        @param args     Arguments for Task.run()

        @return:
        - None if self.doReturnResults false
        - A pipe_base Struct containing these fields if self.doReturnResults true:
            - dataRef: the provided data reference
            - metadata: task metadata after execution of run
            - result: result returned by task run, or None if the task fails
        """
        butler, dataRefList = args
        task = self.TaskClass(config=self.config, log=self.log)
        result = None  # in case the task fails
        if self.doRaise:
            result = task.run(butler, dataRefList)
        else:
            try:
                result = task.run(butler, dataRefList)
            except Exception as e:
                task.log.fatal("Failed: %s" % e)
                if not isinstance(e, pipeBase.TaskError):
                    traceback.print_exc(file=sys.stderr)
        for dataRef in dataRefList:
            task.writeMetadata(dataRef)

        if self.doReturnResults:
            return pipeBase.Struct(
                dataRefList=dataRefList,
                metadata=task.metadata,
                result=result,
            )


class MakeDiscreteSkyMapTask(pipeBase.CmdLineTask):
    """!Make a DiscreteSkyMap in a repository, using the bounding box of a set of calexps.

    The command-line and run signatures and config are sufficiently different from MakeSkyMapTask
    that we don't inherit from it, but it is a replacement, so we use the same config/metadata names.
    """
    ConfigClass = MakeDiscreteSkyMapConfig
    _DefaultName = "makeDiscreteSkyMap"
    RunnerClass = MakeDiscreteSkyMapRunner

    def __init__(self, **kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)

    @pipeBase.timeMethod
    def run(self, butler, dataRefList):
        """!Make a skymap from the bounds of the given set of calexps.

        @param[in]  butler        data butler used to save the SkyMap
        @param[in]  dataRefList   dataRefs of calexps used to determine the size and pointing of the SkyMap
        @return     a pipeBase Struct containing:
                    - skyMap: the constructed SkyMap
        """
        self.log.info("Extracting bounding boxes of %d images" % len(dataRefList))
        points = []
        for dataRef in dataRefList:
            if not dataRef.datasetExists("calexp"):
                self.log.warn("CalExp for %s does not exist: ignoring" % (dataRef.dataId,))
                continue
            md = dataRef.get("calexp_md", immediate=True)
            wcs = afwGeom.makeSkyWcs(md)
            # nb: don't need to worry about xy0 because Exposure saves Wcs with CRPIX shifted by (-x0, -y0).
            boxI = afwImage.bboxFromMetadata(md)
            boxD = afwGeom.Box2D(boxI)
            points.extend(tuple(wcs.pixelToSky(corner).getVector()) for corner in boxD.getCorners())
        if len(points) == 0:
            raise RuntimeError("No data found from which to compute convex hull")
        self.log.info("Computing spherical convex hull")
        polygon = lsst.geom.convexHull(points)
        if polygon is None:
            raise RuntimeError(
                "Failed to compute convex hull of the vertices of all calexp bounding boxes; "
                "they may not be hemispherical."
            )
        circle = polygon.getBoundingCircle()

        datasetName = self.config.coaddName + "Coadd_skyMap"

        skyMapConfig = DiscreteSkyMap.ConfigClass()
        if self.config.doAppend and butler.datasetExists(datasetName):
            oldSkyMap = butler.get(datasetName, immediate=True)
            if not isinstance(oldSkyMap.config, DiscreteSkyMap.ConfigClass):
                raise TypeError("Cannot append to existing non-discrete skymap")
            compareLog = []
            if not self.config.skyMap.compare(oldSkyMap.config, output=compareLog.append):
                raise ValueError("Cannot append to existing skymap - configurations differ:", *compareLog)
            skyMapConfig.raList.extend(oldSkyMap.config.raList)
            skyMapConfig.decList.extend(oldSkyMap.config.decList)
            skyMapConfig.radiusList.extend(oldSkyMap.config.radiusList)
        skyMapConfig.update(**self.config.skyMap.toDict())
        skyMapConfig.raList.append(circle.center[0])
        skyMapConfig.decList.append(circle.center[1])
        skyMapConfig.radiusList.append(circle.radius + self.config.borderSize)
        skyMap = DiscreteSkyMap(skyMapConfig)

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
        if self.config.doWrite:
            butler.put(skyMap, datasetName)
        return pipeBase.Struct(
            skyMap=skyMap
        )

    def _getConfigName(self):
        """Return None to disable saving config

        There's only one SkyMap per repository, so the config is redundant, and checking it means we can't
        easily overwrite or append to an existing repository.
        """
        return None

    def _getMetadataName(self):
        """Return None to disable saving metadata

        The metadata is not interesting, and by not saving it we can eliminate a dataset type.
        """
        return None

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id", datasetType="calexp", help="data ID, e.g. --id visit=123 ccd=1,2")
        return parser
