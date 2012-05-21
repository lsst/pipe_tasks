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
from .coadd import CoaddTask

class OutlierRejectedCoaddConfig(CoaddTask.ConfigClass):
    subregionSize = pexConfig.ListField(
        dtype = int,
        doc = """width, height of stack subregion size;
                make small enough that a full stack of images will fit into memory at once""",
        length = 2,
        default = (2000, 2000),
        optional = None,
    )
    sigmaClip = pexConfig.Field(
        dtype = float,
        doc = "sigma for outlier rejection",
        default = 3.0,
        optional = None,
    )
    clipIter = pexConfig.Field(
        dtype = int,
        doc = "number of iterations of outlier rejection",
        default = 2,
        optional = False,
    )
    

class OutlierRejectedCoaddTask(CoaddTask):
    """Construct an outlier-rejected (robust mean) coadd
    """
    ConfigClass = OutlierRejectedCoaddConfig
    _DefaultName = "outlierRejectedCoadd"

    def __init__(self, *args, **kwargs):
        CoaddTask.__init__(self, *args, **kwargs)
        self._badPixelMask = afwImage.MaskU.getPlaneBitMask(self.config.badMaskPlanes)

    def getBadPixelMask(self):
        return self._badPixelMask
    
    @pipeBase.timeMethod
    def run(self, patchRef):
        """PSF-match, warp and coadd images, using outlier rejection
        
        PSF matching is to a double gaussian model with core FWHM = desFwhm
        and wings of amplitude 1/10 of core and FWHM = 2.5 * core.
        The size of the PSF matching kernel is the same as the size of the kernel
        found in the first calibrated science exposure, since there is no benefit
        to making it any other size.
        
        PSF-matching is performed before warping so the code can use the PSF models
        associated with the calibrated science exposures (without having to warp those models).
        
        @param patchRef: data reference for sky map. Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter" or "band")

        @return: a pipeBase.Struct with fields:
        - coaddExposure: coadd exposure
        """
        skyInfo = self.getSkyInfo(patchRef)
        datasetType = self.config.coaddName + "Coadd"
        
        wcs = skyInfo.wcs
        bbox = skyInfo.bbox
        
        imageRefList = self.selectExposures(patchRef=patchRef, wcs=wcs, bbox=bbox)
        
        numExp = len(imageRefList)
        if numExp < 1:
            raise RuntimeError("No exposures to coadd")
        self.log.log(self.log.INFO, "Coadd %s calexp" % (numExp,))
    
        doPsfMatch = self.config.desFwhm > 0
        if not doPsfMatch:
            self.log.log(self.log.INFO, "No PSF matching will be done (desFwhm <= 0)")

        exposureMetadataList = []
        for ind, dataRef in enumerate(imageRefList):
            self.log.log(self.log.INFO, "Processing exposure %d of %d: id=%s" % \
                (ind+1, numExp, dataRef.dataId))
            exposure = self.getCalexp(dataRef, getPsf=doPsfMatch)
            exposure = self.preprocessExposure(exposure, wcs=wcs, destBBox=bbox)
            tempDataId = dataRef.dataId.copy()
            tempDataId.update(patchRef.dataId)
            tempDataRef = dataRef.butlerSubset.butler.dataRef(
                datasetType = "coaddTempExp",
                dataId = tempDataId,
            )
            tempDataRef.put(exposure)
            expMetadata = ExposureMetadata(
                    dataRef = tempDataRef,
                    exposure = exposure,
                    badPixelMask = self.getBadPixelMask(),
                )
            exposureMetadataList.append(expMetadata)
        del exposure

        edgeMask = afwImage.MaskU.getPlaneBitMask("EDGE")
        
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(self.config.sigmaClip)
        statsCtrl.setNumIter(self.config.clipIter)
        statsCtrl.setAndMask(self.getBadPixelMask())
    
        coaddExposure = afwImage.ExposureF(bbox, wcs)
        coaddExposure.setCalib(self.zeroPointScaler.getCalib())
    
        filterDict = {} # dict of name: Filter
        for expMeta in exposureMetadataList:
            filterDict.setdefault(expMeta.filter.getName(), expMeta.filter)
        if len(filterDict) == 1:
            coaddExposure.setFilter(filterDict.values()[0])
        self.log.log(self.log.INFO, "Filter=%s" % (coaddExposure.getFilter().getName(),))
    
        coaddMaskedImage = coaddExposure.getMaskedImage()
        subregionSizeArr = self.config.subregionSize
        subregionSize = afwGeom.Extent2I(subregionSizeArr[0], subregionSizeArr[1])
        for subBBox in _subBBoxIter(bbox, subregionSize):
            self.log.log(self.log.INFO, "Computing coadd %s" % (subBBox,))
            coaddView = afwImage.MaskedImageF(coaddMaskedImage, subBBox, afwImage.PARENT, False)
            maskedImageList = afwImage.vectorMaskedImageF() # [] is rejected by afwMath.statisticsStack
            weightList = []
            for expMeta in exposureMetadataList:
                if not subBBox.overlaps(expMeta.bbox):
                    # there is no overlap between this temporary exposure and this coadd subregion
                    self.log.log(self.log.INFO, "Skipping %s; no overlap" % (expMeta.path,))
                    continue
                
                if expMeta.bbox.contains(subBBox):
                    # this temporary image fully overlaps this coadd subregion
                    exposure = expMeta.dataRef.get("coaddTempExp_sub", bbox=subBBox, imageOrigin="PARENT")
                    maskedImage = exposure.getMaskedImage()
                else:
                    # this temporary image partially overlaps this coadd subregion;
                    # make a new image of EDGE pixels using the coadd subregion
                    # and set the overlapping pixels from the temporary exposure
                    overlapBBox = afwGeom.Box2I(expMeta.bbox)
                    overlapBBox.clip(subBBox)
                    self.log.log(self.log.INFO,
                        "Processing %s; grow from %s to %s" % (expMeta.path, overlapBBox, subBBox))
                    maskedImage = afwImage.MaskedImageF(subBBox)
                    maskedImage.getMask().set(edgeMask)
                    tempExposure = expMeta.dataRef.get("coaddTempExp_sub",
                        bbox=overlapBBox, imageOrigin="PARENT")
                    tempMaskedImage = tempExposure.getMaskedImage()
                    maskedImageView = afwImage.MaskedImageF(maskedImage, overlapBBox, afwImage.PARENT, False)
                    maskedImageView <<= tempMaskedImage
                maskedImageList.append(maskedImage)
                weightList.append(expMeta.weight)
            try:
                coaddSubregion = afwMath.statisticsStack(
                    maskedImageList, afwMath.MEANCLIP, statsCtrl, weightList)
    
                coaddView <<= coaddSubregion
            except Exception, e:
                self.log.log(self.log.ERR, "Outlier rejection failed; setting EDGE mask: %s" % (e,))
                # re-raise the exception so setCoaddEdgeBits will set the whole coadd mask to EDGE
                raise
    
        coaddUtils.setCoaddEdgeBits(coaddMaskedImage.getMask(), coaddMaskedImage.getVariance())

        if self.config.doWrite:
            patchRef.put(coaddExposure, self.config.coaddName + "Coadd")
    
        return pipeBase.Struct(
            coaddExposure = coaddExposure,
        )



class ExposureMetadata(object):
    """Metadata for an exposure
        
    Attributes:
    - dataRef: data reference for exposure
    - wcs: WCS of exposure
    - bbox: parent bounding box of exposure
    - weight = weightFactor / clipped mean variance
    """
    def __init__(self, dataRef, exposure, badPixelMask, weightFactor = 1.0):
        """Create an ExposureMetadata
        
        @param[in] dataRef: data reference for exposure
        @param[in] exposure: Exposure
        @param[in] badPixelMask: bad pixel mask for pixels to ignore
        @param[in] weightFactor: additional scaling factor for weight:
        """
        self.dataRef = dataRef
        self.wcs = exposure.getWcs()
        self.bbox = exposure.getBBox(afwImage.PARENT)
        self.filter = exposure.getFilter()
        
        maskedImage = exposure.getMaskedImage()

        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(3.0)
        statsCtrl.setNumIter(2)
        statsCtrl.setAndMask(badPixelMask)
        statObj = afwMath.makeStatistics(maskedImage.getVariance(), maskedImage.getMask(),
            afwMath.MEANCLIP, statsCtrl)
        meanVar, meanVarErr = statObj.getResult(afwMath.MEANCLIP);
        weight = weightFactor / float(meanVar)
        self.weight = weight


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
