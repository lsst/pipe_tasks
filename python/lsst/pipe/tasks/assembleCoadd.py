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
import lsst.pex.config as pexConfig
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
from .coaddBase import CoaddBaseTask

__all__ = ["AssembleCoaddTask"]

class AssembleCoaddConfig(CoaddBaseTask.ConfigClass):
    consolidateKeys = pexConfig.ListField(
        dtype = str,
        doc = "data ID keys to consolidate on a single temporary exposure. " \
            "This is intended for mosaic cameras where there is sure to be no overlap; " \
            "for example LSST consolidates raft and sensor data" \
            "Warning: if you specify the wrong value the coadd temp exposure cannot be persisted",
        optional = True,
    )
    subregionSize = pexConfig.ListField(
        dtype = int,
        doc = """width, height of stack subregion size;
                make small enough that a full stack of images will fit into memory at once""",
        length = 2,
        default = (2000, 2000),
    )
    doSigmaClip = pexConfig.Field(
        dtype = bool,
        doc = "perform sigma clipping (if False then compute simple mean)",
        default = True,
    )
    sigmaClip = pexConfig.Field(
        dtype = float,
        doc = "sigma for outlier rejection; ignored if doSigmaClip false",
        default = 3.0,
    )
    clipIter = pexConfig.Field(
        dtype = int,
        doc = "number of iterations of outlier rejection; ignored if doSigmaClip false",
        default = 2,
    )
    doWrite = pexConfig.Field(
        doc = "persist coadd?",
        dtype = bool,
        default = True,
    )
    

class AssembleCoaddTask(CoaddBaseTask):
    """Assemble a coadd from a set of coaddTempExp
    
    @todo: compute weight of each coaddTempExp
    """
    ConfigClass = AssembleCoaddConfig
    _DefaultName = "assembleCoadd"

    @pipeBase.timeMethod
    def run(self, patchRef):
        """Assemble coaddTempExp
        
        @param patchRef: data reference for sky map. Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter" or "band")

        @return: a pipeBase.Struct with fields:
        - coaddExposure: coadd exposure
        """
        skyInfo = self.getSkyInfo(patchRef)
        datasetType = self.config.coaddName + "Coadd"
        
        wcs = skyInfo.wcs
        bbox = skyInfo.bbox
        
        calExpRefList = self.selectExposures(patchRef=patchRef, wcs=wcs, bbox=bbox)
        
        numExp = len(calExpRefList)
        if numExp < 1:
            raise pipeBase.TaskError("No exposures to coadd")
        self.log.info("Selected %s calexp" % (numExp,))

        tempExpName = self.config.coaddName + "Coadd_tempExp"
        tempExpSubName = tempExpName + "_sub"

        # compute tempKeyList: a tuple of ID key names in a calExpId that identify a coaddTempExp.
        # You must also specify tract and patch to make a complete coaddTempExp ID.
        butler = patchRef.butlerSubset.butler
        tempExpKeySet = set(butler.getKeys(datasetType=tempExpName, level="Ccd")) - set(("patch", "tract"))
        tempExpKeyList = tuple(sorted(tempExpKeySet))

        # compute tempExpIdDict, a dict whose:
        # - keys are tuples of coaddTempExp ID values in tempKeyList order
        # - values are tempExpRef
        # Do not check for existence yet (to avoid checking one coaddTempExp multiple times);
        # wait until all desired coaddTempExp have been identified
        tempExpIdDict = dict()
        for calExpRef in calExpRefList:
            calExpId = calExpRef.dataId
            tempExpIdTuple = tuple(calExpId[key] for key in tempExpKeyList)
            if tempExpIdTuple not in tempExpIdDict:
                tempExpId = dict((key, calExpId[key]) for key in tempExpKeyList)
                tempExpId.update(patchRef.dataId)
                tempExpRef = calExpRef.butlerSubset.butler.dataRef(
                    datasetType = tempExpName,
                    dataId = tempExpId,
                )
                tempExpIdDict[tempExpIdTuple] = tempExpRef
        
        # compute tempExpRefList: a list of tempExpRef that actually exist
        tempExpRefList = []
        for tempExpRef in tempExpIdDict.itervalues():
            if not tempExpRef.datasetExists(tempExpName):
                self.log.warn("Could not find %s %s; skipping it" % (tempExpName, calExpId))
                continue
            tempExpRefList.append(tempExpRef)
        del tempExpIdDict

        if not tempExpRefList:
            raise pipeBase.TaskError("No coadd temporary exposures found")
        self.log.info("Assembling %s %s" % (len(tempExpRefList), tempExpName))

        edgeMask = afwImage.MaskU.getPlaneBitMask("EDGE")
        
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(self.config.sigmaClip)
        statsCtrl.setNumIter(self.config.clipIter)
        statsCtrl.setAndMask(self.getBadPixelMask())
        statsCtrl.setNanSafe(True)
        statsCtrl.setCalcErrorFromInputVariance(True)

        if self.config.doSigmaClip:
            statsFlags = afwMath.MEANCLIP
        else:
            statsFlags = afwMath.MEAN
    
        coaddExposure = afwImage.ExposureF(bbox, wcs)
        coaddMaskedImage = coaddExposure.getMaskedImage()
        subregionSizeArr = self.config.subregionSize
        subregionSize = afwGeom.Extent2I(subregionSizeArr[0], subregionSizeArr[1])
        didSetMetadata = False
        for subBBox in _subBBoxIter(bbox, subregionSize):
            self.log.info("Computing coadd %s" % (subBBox,))
            coaddView = afwImage.MaskedImageF(coaddMaskedImage, subBBox, afwImage.PARENT, False)
            maskedImageList = afwImage.vectorMaskedImageF() # [] is rejected by afwMath.statisticsStack
            weightList = []
            for tempExpRef in tempExpRefList:
                exposure = tempExpRef.get(tempExpSubName, bbox=subBBox, imageOrigin="PARENT")
                maskedImage = exposure.getMaskedImage()
                if not didSetMetadata:
                    coaddExposure.setFilter(exposure.getFilter())
                    coaddExposure.setCalib(exposure.getCalib())
                    didSetMetadata = True

                maskedImageList.append(maskedImage)
                weightList.append(1.0)

            try:
                with self.timer("stack"):
                    coaddSubregion = afwMath.statisticsStack(
                        maskedImageList, statsFlags, statsCtrl, weightList)
    
                coaddView <<= coaddSubregion
            except Exception, e:
                self.log.log(self.log.ERR, "Cannot compute this subregion: %s" % (e,))
    
        coaddUtils.setCoaddEdgeBits(coaddMaskedImage.getMask(), coaddMaskedImage.getVariance())
        self.postprocessCoadd(coaddExposure)

        if self.config.doWrite:
            coaddName = self.config.coaddName + "Coadd"
            self.log.info("Persisting %s" % (coaddName,))
            patchRef.put(coaddExposure, coaddName)

        return pipeBase.Struct(
            coaddExposure = coaddExposure,
        )


def _subBBoxIter(bbox, subregionSize):
    """Iterate over subregions of a bbox
    
    @param[in] bbox: bounding box over which to iterate: afwGeom.Box2I
    @param[in] subregionSize: size of sub-bboxes

    @return subBBox: next sub-bounding box of size subregionSize or smaller;
        each subBBox is contained within bbox, so it may be smaller than subregionSize at the edges of bbox,
        but it will never be empty
    """
    if bbox.isEmpty():
        raise RuntimeError("bbox %s is empty" % (bbox,))
    if subregionSize[0] < 1 or subregionSize[1] < 1:
        raise RuntimeError("subregionSize %s must be nonzero" % (subregionSize,))

    for rowShift in range(0, bbox.getHeight(), subregionSize[1]):
        for colShift in range(0, bbox.getWidth(), subregionSize[0]):
            subBBox = afwGeom.Box2I(bbox.getMin() + afwGeom.Extent2I(colShift, rowShift), subregionSize)
            subBBox.clip(bbox)
            if subBBox.isEmpty():
                raise RuntimeError("Bug: empty bbox! bbox=%s, subregionSize=%s, colShift=%s, rowShift=%s" % \
                    (bbox, subregionSize, colShift, rowShift))
            yield subBBox
