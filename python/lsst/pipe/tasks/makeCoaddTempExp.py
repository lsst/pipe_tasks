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
    bgSubtracted = pexConfig.Field(
        doc = "Work with a background subtracted calexp?",
        dtype = bool,
        default = False,
    )



class MakeCoaddTempExpTask(CoaddBaseTask):
    """Coadd temporary images by PSF-matching (optional), warping and computing a weighted sum
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
        The size of the PSF matching kernel is the same as the size of the kernel
        found in the first calibrated science exposure, since there is no benefit
        to making it any other size.
        
        PSF-matching is performed before warping so the code can use the PSF models
        associated with the calibrated science exposures (without having to warp those models).
    
        @param[in] patchRef: data reference for sky map patch. Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter" or "band")
        @return: a pipeBase.Struct with fields:
        - dataRefList: a list of data references for the new <coaddName>Coadd_tempExp
        """
        skyInfo = self.getSkyInfo(patchRef)
        
        tractWcs = skyInfo.wcs
        patchBBox = skyInfo.bbox
        
        calExpRefList = self.selectExposures(patchRef=patchRef, wcs=tractWcs, bbox=patchBBox)
        
        # initialize outputs
        dataRefList = []
        
        numExp = len(calExpRefList)
        if numExp < 1:
            raise pipeBase.TaskError("No exposures to coadd")
        self.log.info("Coadd %s calexp" % (numExp,))
    
        doPsfMatch = self.config.warpAndPsfMatch.desiredFwhm is not None
        if not doPsfMatch:
            self.log.info("No PSF matching will be done (desiredFwhm is None)")

        tempExpName = self.config.coaddName + "Coadd_tempExp"

        # compute tempKeyList: a tuple of ID key names in a calExpId that identify a coaddTempExp.
        # You must also specify tract and patch to make a complete coaddTempExp ID.
        butler = patchRef.butlerSubset.butler
        tempExpKeySet = set(butler.getKeys(datasetType=tempExpName, level="Ccd")) - set(("patch", "tract"))
        tempExpKeyList = tuple(sorted(tempExpKeySet))

        # compute tempExpIdDict, a dict whose:
        # - keys are tuples of coaddTempExp ID values in tempKeyList order
        # - values are a list of calExp data references for calExp that belong in this coaddTempExp
        tempExpIdDict = dict()
        for calExpRef in calExpRefList:
            calExpId = calExpRef.dataId
            if not calExpRef.datasetExists("calexp"):
                self.log.warn("Could not find calexp %s; skipping it" % (calExpId,))
                continue
            
            tempExpIdTuple = tuple(calExpId[key] for key in tempExpKeyList)
            calExpSubsetRefList = tempExpIdDict.get(tempExpIdTuple)
            if calExpSubsetRefList:
                calExpSubsetRefList.append(calExpRef)
            else:
                tempExpIdDict[tempExpIdTuple] = [calExpRef]

        numTempExp = len(tempExpIdDict)
        coaddTempExp = None
        for tempExpInd, calExpSubsetRefList in enumerate(tempExpIdDict.itervalues()):
            # derive tempExpId from the first calExpId
            tempExpId = dict((key, calExpSubsetRefList[0].dataId[key]) for key in tempExpKeyList)
            tempExpId.update(patchRef.dataId)
            tempExpRef = calExpRef.butlerSubset.butler.dataRef(
                datasetType = tempExpName,
                dataId = tempExpId,
            )
            self.log.info("Computing coaddTempExp %d of %d: id=%s" % (tempExpInd+1, numTempExp, tempExpId))

            totGoodPix = 0
            coaddTempExp = afwImage.ExposureF(patchBBox, tractWcs)
            edgeMask = afwImage.MaskU.getPlaneBitMask("EDGE")
            coaddTempExp.getMaskedImage().set(numpy.nan, edgeMask, numpy.inf)
            for calExpInd, calExpRef in enumerate(calExpSubsetRefList):
                self.log.info("Processing calexp %d of %d for this tempExp: id=%s" % \
                    (calExpInd+1, len(calExpSubsetRefList), calExpRef.dataId))
                try:
                    exposure = self.warpAndPsfMatch.getCalExp(calExpRef, getPsf=doPsfMatch, bgSubtracted=self.config.bgSubtracted) 
                    exposure = self.warpAndPsfMatch.run(exposure, wcs=tractWcs, maxBBox=patchBBox).exposure
                    numGoodPix = coaddUtils.copyGoodPixels(
                        coaddTempExp.getMaskedImage(), exposure.getMaskedImage(), self._badPixelMask)
                    totGoodPix += numGoodPix
                    if numGoodPix == 0:
                        self.log.warn("Calexp %s has no good pixels in this patch" % (calExpRef.dataId,))
                    else:
                        self.log.info("Calexp %s has %s good pixels in this patch" % \
                            (calExpRef.dataId, numGoodPix))
                except Exception, e:
                    self.log.warn("Error processing calexp %s; skipping it: %s" % \
                        (calExpRef.dataId, e))
                    continue

            if totGoodPix == 0:
                raise RuntimeError("Could not compute coaddTempExp: no usable input data")
            self.log.info("coaddTempExp %s has %s good pixels" % (tempExpRef.dataId, totGoodPix))
                
            if self.config.doWrite and coaddTempExp is not None:
                self.log.info("Persisting %s %s" % (tempExpName, tempExpRef.dataId))
                tempExpRef.put(coaddTempExp, tempExpName)
                if self.config.warpAndPsfMatch.desiredFwhm is not None:
                    psfName = self.config.coaddName + "Coadd_initPsf"
                    self.log.info("Persisting %s %s" % (psfName, tempExpRef.dataId))
                    wcs = coaddTempExp.getWcs()
                    fwhmPixels = self.config.warpAndPsfMatch.desiredFwhm / wcs.pixelScale().asArcseconds()
                    kernelSize = int(round(fwhmPixels * self.config.coaddKernelSizeFactor))
                    kernelDim = afwGeom.Point2I(kernelSize, kernelSize)
                    coaddPsf = self.makeModelPsf(fwhmPixels=fwhmPixels, kernelDim=kernelDim)
                    patchRef.put(coaddPsf, psfName)

            if coaddTempExp:
                dataRefList.append(tempExpRef)
            else:
                self.log.warn("This %s temp coadd exposure could not be created"%(tempExpRef.dataId,))
        
        return pipeBase.Struct(
            dataRefList = dataRefList,
        )
