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
"""@todo: 
- use butler to save and retrieve intermediate images
- modify debug mode that allow retrieving existing data so that it
    is controlled by debug or config, and perhaps implement it as "reuse if it exists"
    (but this is dangerous unless the file name contains enough info to tell if it's the right image)
"""
import math
import os
import sys

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.coadd.utils as coaddUtils
import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from .coadd import CoaddTask

class OutlierRejectedCoaddConfig(CoaddTask.ConfigClass):
    subregionSize = pexConfig.ListField(
        dtype = int,
        doc = """width, height of stack subregion size;
                make small enough that a full stack of images will fit into memory at once""",
        length = 2,
        default = (200, 200),
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

        coaddConfig = self.config.coadd
        self._badPixelMask = afwImage.MaskU.getPlaneBitMask(coaddConfig.badMaskPlanes)
        self._coaddCalib = coaddUtils.makeCalib(coaddConfig.coaddZeroPoint)

    @classmethod
    def parseAndRun(cls, args=None, config=None, log=None):
        """Parse an argument list and run the command

        @param args: list of command-line arguments; if None use sys.arv
        @param config: config for task (instance of pex_config Config); if None use cls.ConfigClass()
        @param log: log (instance of pex_logging Log); if None use the default log
        """
        argumentParser = cls._makeArgumentParser()
        if config is None:
            config = cls.ConfigClass()
        parsedCmd = argumentParser.parse_args(config=config, args=args, log=log)
        task = cls(name = cls._DefaultName, config = parsedCmd.config, log = parsedCmd.log)

        # normally the butler would do this, but it doesn't have support for coadds yet
        task.config.save("%s_config.py" % (task.getName(),))

        taskRes = task.run(
            dataRefList = parsedCmd.dataRefList,
            bbox = parsedCmd.bbox,
            wcs = parsedCmd.wcs,
            desFwhm = parsedCmd.fwhm,
        )
        
        coaddExposure = taskRes.coaddExposure
    
        # normally the butler would do this, but it doesn't have support for coadds yet
        filterName = coaddExposure.getFilter().getName()
        if filterName == "_unknown_":
            filterStr = "unk"
        coaddBaseName = "%s_filter_%s_fwhm_%s" % (task.getName(), filterName, parsedCmd.fwhm)
        coaddPath = coaddBaseName + ".fits"
        print "Saving coadd as %s" % (coaddPath,)
        coaddExposure.writeFits(coaddPath)
        
        # normally the butler would do this, but it doesn't have support for coadds yet
        fullMetadata = task.getFullMetadata()
        mdStr = fullMetadata.toString()
        with file("%s_metadata.txt" % (task.getName(),), "w") as mdfile:
            mdfile.write(mdStr)
    
    def getBadPixelMask(self):
        return self._badPixelMask

    def getCoaddCalib(self):
        return self._coaddCalib
    
    @pipeBase.timeMethod
    def run(self, dataRefList, bbox, wcs, desFwhm):
        """PSF-match, warp and coadd images, using outlier rejection
        
        PSF matching is to a double gaussian model with core FWHM = desFwhm
        and wings of amplitude 1/10 of core and FWHM = 2.5 * core.
        The size of the PSF matching kernel is the same as the size of the kernel
        found in the first calibrated science exposure, since there is no benefit
        to making it any other size.
        
        PSF-matching is performed before warping so the code can use the PSF models
        associated with the calibrated science exposures (without having to warp those models).
        
        @param dataRefList: list of sensor-level data references
        @param bbox: bounding box of coadd
        @param wcs: WCS of coadd
        @param desFwhm: desired FWHM of PSF, in science exposure pixels
                (note that the coadd often has a different scale than the science images);
                if 0 then no PSF matching is performed.

        @return: a pipeBase.Struct with fields:
        - coaddExposure: coadd exposure
        """
        numExp = len(dataRefList)
        if numExp < 1:
            raise RuntimeError("No exposures to coadd")
        self.log.log(self.log.INFO, "Coadd %s calexp" % (len(dataRefList),))
    
        exposureMetadataList = self.psfMatchAndWarp(
            dataRefList = dataRefList,
            desFwhm = desFwhm,
            wcs = wcs,
            bbox = bbox,
        )
        
        edgeMask = afwImage.MaskU.getPlaneBitMask("EDGE")
        
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(self.config.sigmaClip)
        statsCtrl.setNumIter(self.config.clipIter)
        statsCtrl.setAndMask(self.getBadPixelMask())
    
        coaddExposure = afwImage.ExposureF(bbox, wcs)
        coaddExposure.setCalib(self.getCoaddCalib())
    
        filterDict = {} # dict of name: Filter
        for expMeta in exposureMetadataList:
            filterDict.setdefault(expMeta.filter.getName(), expMeta.filter)
        if len(filterDict) == 1:
            coaddExposure.setFilter(filterDict.values()[0])
        self.log.log(self.log.INFO, "Filter=%s" % (coaddExposure.getFilter().getName(),))
        
        coaddExposure.writeFits("blankCoadd.fits")
    
        coaddMaskedImage = coaddExposure.getMaskedImage()
        subregionSizeArr = self.config.subregionSize
        subregionSize = afwGeom.Extent2I(subregionSizeArr[0], subregionSizeArr[1])
        dumPS = dafBase.PropertySet()
        for subBBox in _subBBoxIter(bbox, subregionSize):
            self.log.log(self.log.INFO, "Computing coadd %s" % (subBBox,))
            coaddView = afwImage.MaskedImageF(coaddMaskedImage, subBBox, afwImage.PARENT, False)
            maskedImageList = afwImage.vectorMaskedImageF() # [] is rejected by afwMath.statisticsStack
            weightList = []
            for expMeta in exposureMetadataList:
                if expMeta.bbox.contains(subBBox):
                    maskedImage = afwImage.MaskedImageF(expMeta.path, 0, dumPS, subBBox, afwImage.PARENT)
                elif not subBBox.overlaps(expMeta.bbox):
                    self.log.log(self.log.INFO, "Skipping %s; no overlap" % (expMeta.path,))
                    continue
                else:
                    overlapBBox = afwGeom.Box2I(expMeta.bbox)
                    overlapBBox.clip(subBBox)
                    self.log.log(self.log.INFO,
                        "Processing %s; grow from %s to %s" % (expMeta.path, overlapBBox, subBBox))
                    maskedImage = afwImage.MaskedImageF(subBBox)
                    maskedImage.getMask().set(edgeMask)
                    maskedImageView = afwImage.MaskedImageF(maskedImage, overlapBBox, afwImage.PARENT, False)
                    maskedImageView <<= afwImage.MaskedImageF(expMeta.path, 0, dumPS, overlapBBox,
                        afwImage.PARENT)
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
    
        return pipeBase.Struct(
            coaddExposure = coaddExposure,
        )

    def psfMatchAndWarp(self, dataRefList, bbox, wcs, desFwhm):
        """Normalize, PSF-match (if desFWhm > 0) and warp exposures; save resulting exposures as FITS files
        
        @param dataRefList: list of sensor-level data references
        @param bbox: bounding box for coadd
        @param wcs: desired WCS of coadd
        @param desFwhm: desired FWHM (pixels)

        @return
        - exposureMetadataList: a list of ExposureMetadata objects
            describing the saved psf-matched and warped exposures
        """
        numExp = len(dataRefList)

        doPsfMatch = desFwhm > 0
    
        if not doPsfMatch:
            self.log.log(self.log.INFO, "No PSF matching will be done (desFwhm <= 0)")

        exposureMetadataList = []
        for ind, dataRef in enumerate(dataRefList):
            dataId = dataRef.dataId
            outPath = "_".join(["%s_%s" % (k, dataId[k]) for k in sorted(dataId.keys())])
            outPath = outPath.replace(",", "_")
            outPath = outPath + ".fits"
            if True:        
                self.log.log(self.log.INFO, "Processing exposure %d of %d: dataId=%s" % \
                    (ind+1, numExp, dataId))
                exposure = self.getCalexp(dataRef, getPsf=doPsfMatch)
        
                srcCalib = exposure.getCalib()
                scaleFac = 1.0 / srcCalib.getFlux(self.config.coadd.coaddZeroPoint)
                maskedImage = exposure.getMaskedImage()
                maskedImage *= scaleFac
                self.log.log(self.log.INFO, "Normalized using scaleFac=%0.3g" % (scaleFac,))
    
                if desFwhm > 0:
                    modelPsf = self.makeModelPsf(exposure, desFwhm)
                    self.log.log(self.log.INFO, "PSF-match exposure")
                    psfRes = self.psfMatch.run(exposure, modelPsf)
                    exposure = psfRes.psfMatchedExposure
                
                self.log.log(self.log.INFO, "Warp exposure")
                with self.timer("warp"):
                    exposure = self.warper.warpExposure(wcs, exposure, maxBBox = bbox)
                exposure.setCalib(self.getCoaddCalib())
    
                self.log.log(self.log.INFO, "Saving intermediate exposure as %s" % (outPath,))
                exposure.writeFits(outPath)
            else:
                # debug mode; exposures already exist
                self.log.log(self.log.WARN, "DEBUG MODE; Processing dataId=%s; retrieving from %s" % \
                    (dataId, outPath))
                exposure = afwImage.ExposureF(outPath)
    
            expMetadata = ExposureMetadata(
                    path = outPath,
                    exposure = exposure,
                    badPixelMask = self.getBadPixelMask(),
                )
            exposureMetadataList.append(expMetadata)
            
        return exposureMetadataList


class ExposureMetadata(object):
    """Metadata for an exposure
        
    Attributes:
    - path: path to exposure FITS file
    - wcs: WCS of exposure
    - bbox: parent bounding box of exposure
    - weight = weightFactor / clipped mean variance
    """
    def __init__(self, path, exposure, badPixelMask, weightFactor = 1.0):
        """Create an ExposureMetadata
        
        @param[in] path: path to Exposure FITS file
        @param[in] exposure: Exposure
        @param[in] badPixelMask: bad pixel mask for pixels to ignore
        @param[in] weightFactor: additional scaling factor for weight:
        """
        self.path = path
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
