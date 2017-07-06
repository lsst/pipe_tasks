#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011, 2012 LSST Corporation.
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
"""Select images for a given coadd patch
"""
from __future__ import print_function
import numpy

import lsst.pex.config as pexConfig
import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom
import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.makeSkyMap import MakeSkyMapTask
from lsst.pipe.tasks.selectImages import WcsSelectImagesTask

__all__ = ["ReportImagesInPatchTask", ]


class ReportImagesInPatchConfig(pexConfig.Config):
    """Config for ReportImagesInPatchTask
    """
    coaddName = pexConfig.Field(
        doc="coadd name: one of deep or goodSeeing",
        dtype=str,
        default="deep",
    )
    select = pexConfig.ConfigurableField(
        doc="image selection subtask",
        target=WcsSelectImagesTask,
    )


class ReportImagesInPatchTask(pipeBase.CmdLineTask):
    """Report which tracts and patches are needed for coaddition
    """
    ConfigClass = ReportImagesInPatchConfig
    _DefaultName = "reportImagesInPatch"

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)
        self.makeSubtask("select")

    @pipeBase.timeMethod
    def run(self, patchRef):
        """Select images for a region and report how many are in each tract and patch

        Also report quartiles of FWHM

        @param patchRef: data reference for coadd patch.
        @return: a pipeBase.Struct with fields:
        - exposureInfoList: a list of exposure info objects, as returned by the select subtask
        """
        skyMap = patchRef.get(self.config.coaddName + "Coadd_skyMap")

        tractId = patchRef.dataId["tract"]
        patchIndex = [int(val) for val in patchRef.dataId["patch"].split("x")]
        tractInfo = skyMap[tractId]
        wcs = tractInfo.getWcs()
        patchInfo = tractInfo.getPatchInfo(patchIndex)
        posBox = afwGeom.Box2D(patchInfo.getOuterBBox())
        coordList = [wcs.pixelToSky(pos) for pos in _getBox2DCorners(posBox)]

        skyPosList = [coord.getPosition(afwGeom.degrees) for coord in coordList]
        skyPosStrList = ["(%0.3f, %0.3f)" % tuple(skyPos) for skyPos in skyPosList]
        skyPosStr = ", ".join(skyPosStrList)
        print("PatchId=%s; corner RA/Dec (deg)=%s" % (patchRef.dataId, skyPosStr))

        exposureInfoList = self.select.runDataRef(
            dataRef=patchRef,
            coordList=coordList,
            makeDataRefList=False,
        ).exposureInfoList

        print("Found %d suitable exposures" % (len(exposureInfoList),))
        if len(exposureInfoList) < 1:
            return

        fwhmList = [exposureInfo.fwhm for exposureInfo in exposureInfoList]
        fwhmList = numpy.array(fwhmList, dtype=float)
        print("FWHM Q1=%0.2f Q2=%0.2f Q3=%0.2f" % (
            numpy.percentile(fwhmList, 25.0),
            numpy.percentile(fwhmList, 50.0),
            numpy.percentile(fwhmList, 75.0),
        ))

        print("Image IDs:")
        if len(exposureInfoList) > 0:
            idKeys = sorted(exposureInfoList[0].dataId.keys())
            for exposureInfo in exposureInfoList:
                idStr = " ".join("%s=%s" % (key, exposureInfo.dataId[key]) for key in idKeys)
                skyPosList = [coord.getPosition(afwGeom.degrees) for coord in exposureInfo.coordList]
                skyPosStrList = ["(%0.3f, %0.3f)" % tuple(skyPos) for skyPos in skyPosList]
                skyPosStr = ", ".join(skyPosStrList)
                print("dataId=%s; corner RA/Dec (deg)=%s" % (idStr, skyPosStr))

        return pipeBase.Struct(exposureInfoList=exposureInfoList)

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser

        Use datasetType="deepCoadd" to get the right keys (even chi-squared coadds
        need filter information for this particular task).
        """
        parser = pipeBase.InputOnlyArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd", help="data ID, e.g. --id tract=12345 patch=1,2",
                               ContainerClass=ReportImagesInPatchDataIdContainer)
        return parser

    def _getConfigName(self):
        """Don't persist config, so return None
        """
        return None

    def _getMetadataName(self):
        """Don't persist metadata, so return None
        """
        return None


class ReportImagesInPatchDataIdContainer(pipeBase.DataIdContainer):
    """A version of lsst.pipe.base.DataIdContainer specialized for reporting images.

    Required because butler.subset cannot handle this dataset type.
    """

    def makeDataRefList(self, namespace):
        """Make self.refList from self.idList"""
        datasetType = namespace.config.coaddName + "Coadd"

        for dataId in self.idList:
            dataRef = namespace.butler.dataRef(
                datasetType=datasetType,
                dataId=dataId,
            )
            self.refList.append(dataRef)


def _getBox2DCorners(bbox):
    """Return the four corners of a bounding box (Box2I or Box2D) as four afwGeom Point2D
    """
    bbox = afwGeom.Box2D(bbox)  # mak
    return (
        bbox.getMin(),
        afwGeom.Point2D(bbox.getMaxX(), bbox.getMinY()),
        bbox.getMax(),
        afwGeom.Point2D(bbox.getMinX(), bbox.getMaxY()),
    )


if __name__ == "__main__":
    ReportImagesInPatchTask.parseAndRun()
