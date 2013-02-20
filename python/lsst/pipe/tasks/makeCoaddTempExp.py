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
import math

import numpy

import lsst.pex.config as pexConfig
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
from .coaddBase import CoaddBaseTask
from .warpAndPsfMatch import WarpAndPsfMatchTask
from .coaddHelpers import groupExposures, getTempExpRef

__all__ = ["MakeCoaddTempExpTask"]

class MakeCoaddTempExpConfig(CoaddBaseTask.ConfigClass):
    """Config for MakeCoaddTempExpTask
    """
    coaddKernelSizeFactor = pexConfig.Field(
        dtype = float,
        doc = "coadd kernel size = coadd FWHM converted to pixels * coaddKernelSizeFactor",
        default = 3.0,
    )
    warpAndPsfMatch = pexConfig.ConfigurableField(
        target = WarpAndPsfMatchTask,
        doc = "Task to warp, PSF-match and zero-point-match calexp",
    )
    doWrite = pexConfig.Field(
        doc = "persist <coaddName>Coadd_tempExp and (if desiredFwhm not None) <coaddName>Coadd_initPsf?",
        dtype = bool,
        default = True,
    )
    doOverwrite = pexConfig.Field(
        doc = "overwrite <coaddName>Coadd_tempExp and (if desiredFwhm not None) <coaddName>Coadd_initPsf?  If False, continue if the file exists on disk",
        dtype = bool,
        default = True,
    )
    bgSubtracted = pexConfig.Field(
        doc = "Work with a background subtracted calexp?",
        dtype = bool,
        default = False,
    )


class MakeCoaddTempExpTask(CoaddBaseTask):
    """Task to produce <coaddName>Coadd_tempExp images and (optional) <coaddName>Coadd_initPsf
    """
    ConfigClass = MakeCoaddTempExpConfig
    _DefaultName = "makeCoaddTempExp"
    
    def __init__(self, *args, **kwargs):
        CoaddBaseTask.__init__(self, *args, **kwargs)
        self.makeSubtask("warpAndPsfMatch")

    @pipeBase.timeMethod
    def run(self, patchRef):
        """Produce <coaddName>Coadd_tempExp images and (optional) <coaddName>Coadd_initPsf
        
        <coaddName>Coadd_tempExp are produced by PSF-matching (optional) and warping.
        If PSF-matching is used then <coaddName>Coadd_initPsf is also computed.
        
        PSF matching is to a double gaussian model with core FWHM = self.config.warpAndPsfMatch.desiredFwhm
        and wings of amplitude 1/10 of core and FWHM = 2.5 * core.
        
        PSF-matching is performed before warping so the code can use the PSF models
        associated with the calibrated science exposures (without having to warp those models).
        
        @param[in] patchRef: data reference for sky map patch. Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter" or "band")
        @return: a pipeBase.Struct with fields:
        - dataRefList: a list of data references for the new <coaddName>Coadd_tempExp

        @warning: this task assumes that all exposures in a coaddTempExp have the same filter.
        
        @warning: this task sets the Calib of the coaddTempExp to the Calib of the first calexp
        with any good pixels in the patch. For a mosaic camera the resulting Calib should be ignored
        (assembleCoadd should determine zeropoint scaling without referring to it).
        """
        skyInfo = self.getSkyInfo(patchRef)

        calExpRefList = self.selectExposures(patchRef, skyInfo)
        if len(calExpRefList) == 0:
            self.log.warn("No exposures to coadd for patch %s" % patchRef.dataId)
            return None
        self.log.info("Processing %d calexps for patch %s" % (len(calExpRefList), patchRef.dataId))

        groupData = groupExposures(patchRef, calExpRefList)
        self.log.info("Processing %d tempExps for patch %s" % (len(groupData.groups), patchRef.dataId))

        dataRefList = []
        for i, (tempExpTuple, calexpRefList) in enumerate(groupData.groups.itervalues()):
            tempExpRef = getTempExpRef(patchRef.getButler(), self.getTempExpName(),
                                       tempExpTuple, groupData.keys)
            if not self.config.doOverwrite and tempExpRef.datasetExists(datasetType=tempExpName):
                self.log.info("tempCoaddExp %s exists; skipping" % (tempExpRef.dataId,))
                dataRefList.append(tempExpRef)
                continue
            self.log.info("Processing tempExp %d/%d: id=%s" % (i, len(tempExpIdDict), tempExpId))
            exp = self.createTempExp(calexpRefList)
            if exp is not None:
                dataRefList.append(tempExpRef)
                if self.config.doWrite:
                    self.writeTempExp(tempExpRef, exp)
            else:
                self.log.warn("tempExp %s could not be created" % (tempExpRef.dataId,))
        return dataRefList

    def getTempExpName(self):
        return self.config.coaddName + "Coadd_tempExp"

    def createTempExp(self, calexpRefList):
        coaddTempExp = afwImage.ExposureF(patchBBox, tractWcs)
        edgeMask = afwImage.MaskU.getPlaneBitMask("EDGE")
        coaddTempExp.getMaskedImage().set(numpy.nan, edgeMask, numpy.inf) # XXX these are the wrong values!
        totGoodPix = 0
        didSetMetadata = False
        for calExpInd, calExpRef in enumerate(calExpSubsetRefList):
            self.log.info("Processing calexp %d of %d for this tempExp: id=%s" % \
                (calExpInd+1, len(calExpSubsetRefList), calExpRef.dataId))
            try:
                exposure = self.getCalExp(calExpRef, getPsf=doPsfMatch, bgSubtracted=self.config.bgSubtracted)
                exposure = self.warpAndPsfMatch.run(exposure, wcs=tractWcs, maxBBox=patchBBox).exposure
                numGoodPix = coaddUtils.copyGoodPixels(
                    coaddTempExp.getMaskedImage(), exposure.getMaskedImage(), self._badPixelMask)
                totGoodPix += numGoodPix
                self.log.logdebug("Calexp %s has %d good pixels in this patch" %
                                  (calExpRef.dataId, numGoodPix))
                if numGoodPix > 0 and not didSetMetadata:
                    coaddTempExp.setCalib(exposure.getCalib())
                    coaddTempExp.setFilter(exposure.getFilter())
                    didSetMetadata = True
            except Exception, e:
                self.log.warn("Error processing calexp %s; skipping it: %s" % (calExpRef.dataId, e))
                continue

        self.log.info("coaddTempExp has %d good pixels" % (totGoodPix))
        return coaddTempExp if totGoodPix > 0 and didSetMetadata else None

    def writeTempExp(self, tempExpRef, coaddTempExp):
        tempExpName = self.getTempExpName()
        self.log.info("Persisting %s %s" % (tempExpName, tempExpRef.dataId))
        tempExpRef.put(coaddTempExp, tempExpName)
        if self.config.desiredFwhm is not None:
            psf = self.makeModelPsf(fwhmPixels=self.config.desiredFwhm, wcs=wcs,
                                    sizeFactor=self.config.coaddKernelSizeFactor)
            self.writeCoaddOutput(patchRef, psf, "initPsf")
