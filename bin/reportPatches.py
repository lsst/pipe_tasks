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
"""Select images and report which tracts and patches they are in

@warning: this is a very basic start. Misfeatures include:
- Only reports the best tract and patch containing the center of each image;
  a proper implementation will report all tracts and patches that overlap each image
- One must specify a patch and tract even though those arguments are ignored.
"""
import numpy

import lsst.pex.config as pexConfig
import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom
import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.makeSkyMap import MakeSkyMapTask
from lsst.pipe.tasks.coadd import NullSelectTask

__all__ = ["ReportPatchesTask", "ReportPatchesArgumentParser"]

class ReportPatchesConfig(pexConfig.Config):
    """Config for ReportPatchesTask
    """
    coaddName = pexConfig.Field(
        doc = "coadd name: one of deep or goodSeeing",
        dtype = str,
        default = "deep",
    )
    raDecRange = pexConfig.ListField(
        doc = "min RA, min Dec, max RA, max Dec (ICRS, deg)",
        dtype = float,
        length = 4,
    )


class ReportPatchesTask(pipeBase.CmdLineTask):
    """Report which tracts and patches are needed for coaddition
    """
    ConfigClass = ReportPatchesConfig
    _DefaultName = "reportPatches"
    
    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

    @pipeBase.timeMethod
    def run(self, dataRef):
        """Report tracts and patches that are within a given region of a skymap

        @param dataRef: data reference for sky map.
        @return: a pipeBase.Struct with fields:
        - ccdInfoSetDict: a dict of (tractId, patchIndex): set of CcdExposureInfo
        """
        skyMap = dataRef.get(self.config.coaddName + "Coadd_skyMap")

        # make coords in the correct order to form an enclosed space
        raRange  = (self.config.raDecRange[0], self.config.raDecRange[2])
        decRange = (self.config.raDecRange[1], self.config.raDecRange[3])
        raDecList = [
            (raRange[0], decRange[0]),
            (raRange[1], decRange[0]),
            (raRange[1], decRange[1]),
            (raRange[0], decRange[1]),
        ]
        coordList = [
            afwCoord.IcrsCoord(afwGeom.Angle(ra, afwGeom.degrees), afwGeom.Angle(dec, afwGeom.degrees))
            for ra, dec in raDecList]
        tractPatchList = skyMap.findTractPatchList(coordList)
        for tractInfo, patchInfoList in tractPatchList:
            for patchInfo in patchInfoList:
                patchIndex = patchInfo.getIndex()
                print "tract=%d patch=%d,%d" % (tractInfo.getId(), patchIndex[0], patchIndex[1])

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        
        Use datasetType="deepCoadd" to get the right keys (even chi-squared coadds
        need filter information for this particular task).
        """
        return ReportPatchesArgumentParser(name=cls._DefaultName, datasetType="deepCoadd")


class ReportPatchesArgumentParser(pipeBase.ArgumentParser):
    """A version of lsst.pipe.base.ArgumentParser specialized for reporting images.
    
    Required because there is no dataset type that is has exactly the right keys for this task.
    datasetType = namespace.config.coaddName + "Coadd" comes closest, but includes "patch" and "tract",
    which are irrelevant to the task, but required to make a data reference of this dataset type.
    Also required because butler.subset cannot handle this dataset type.
    """
    def _makeDataRefList(self, namespace):
        """Make namespace.dataRefList from namespace.dataIdList
        """
        datasetType = namespace.config.coaddName + "Coadd"

        namespace.dataRefList = []
        for dataId in namespace.dataIdList:
            expandedDataId = dict(patch=0, tract=(0,0))
            expandedDataId.update(dataId)
            dataRef = namespace.butler.dataRef(
                datasetType = datasetType,
                dataId = expandedDataId,
            )
            namespace.dataRefList.append(dataRef)


if __name__ == "__main__":
    ReportPatchesTask.parseAndRun()
