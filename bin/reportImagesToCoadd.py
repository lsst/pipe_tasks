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
from lsst.pipe.tasks.coaddArgumentParser import CoaddArgumentParser

class ReportImagesToCoaddConfig(pexConfig.Config):
    """Config for ReportImagesToCoaddTask
    """
    coaddName = pexConfig.Field(
        doc = "coadd name: one of deep or goodSeeing",
        dtype = str,
        default = "deep",
    )
    select = pexConfig.ConfigurableField(
        doc = "image selection subtask",
        target = NullSelectTask, # must be retargeted
    )


class ReportImagesToCoaddTask(pipeBase.CmdLineTask):
    """Report which tracts and patches are needed for coaddition
    
    @warning: this is very crude at the moment. It only selects tracts and patches
    that are best for the center of a given image, so it misses all overlaps.
    """
    ConfigClass = ReportImagesToCoaddConfig
    _DefaultName = "reportImagesToCoadd"
    
    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)
        self.makeSubtask("select")

    @pipeBase.timeMethod
    def run(self, dataRef):
        """Select images across the sky and report how many are in each tract and patch
        
        Also report quartiles of FWHM 
    
        @param dataRef: data reference for sky map.
        @return: a pipeBase.Struct with fields:
        - ccdInfoSetDict: a dict of (tractId, patchIndex): set of CcdExposureInfo
        """
        skyMap = dataRef.get(self.config.coaddName + "Coadd_skyMap")

        # determine which images meet coaddition selection criteria
        ccdInfoList = self.select.searchWholeSky(dataRef).ccdInfoList
        
        numExp = len(ccdInfoList)
        self.log.log(self.log.INFO, "Found %s exposures that match your selection criteria" % (numExp,))
        if numExp < 1:
            return
        
        ccdInfoSetDict = dict()
        
        fwhmList = []
        numOffCoadd = 0
        for ccdInfo in ccdInfoList:
            fwhmList.append(ccdInfo.fwhm)

            ctrCoord = afwCoord.IcrsCoord(
                afwGeom.Angle(ccdInfo.ctrRaDec[0], afwGeom.degrees),
                afwGeom.Angle(ccdInfo.ctrRaDec[1], afwGeom.degrees),
            )
            tractInfo = skyMap.findTract(ctrCoord)
            try:
                patchInfo = tractInfo.findPatch(ctrCoord)
            except LookupError:
                numOffCoadd += 1
            
            key = (tractInfo.getId(), patchInfo.getIndex())
            ccdInfoSet = ccdInfoSetDict.get(key)
            if ccdInfoSet is None:
                ccdInfoSetDict[key] = set([ccdInfo])
            else:
                ccdInfoSet.add(ccdInfo)
        
        fwhmList = numpy.array(fwhmList, dtype=float)
        print "FWHM Q1=%0.2f Q2=%0.2f Q3=%0.2f" % (
            numpy.percentile(fwhmList, 25.0),
            numpy.percentile(fwhmList, 50.0),
            numpy.percentile(fwhmList, 75.0),
        )
        
        print "Tract  patchX  patchY  numExp"
        for key in sorted(ccdInfoSetDict.keys()):
            ccdInfoSet = ccdInfoSetDict[key]
            print "%5d   %5d   %5d  %5d" % (key[0], key[1][0], key[1][1], len(ccdInfoSet))
        
        return pipeBase.Struct(
            ccdInfoSetDict = ccdInfoSetDict,
        )

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        return CoaddArgumentParser(name=cls._DefaultName, datasetType="deepCoadd")

ReportImagesToCoaddTask.parseAndRun()
