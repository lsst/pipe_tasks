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

class AssembleCoaddConfig(CoaddTask.ConfigClass):
    coaddName = pexConfig.Field(
        doc = "coadd name: typically one of deep or goodSeeing",
        dtype = str,
        default = "deep",
    )
    select = pexConfig.ConfigurableField(
        doc = "image selection subtask",
        target = BadSelectImagesTask,
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
    badMaskPlanes = pexConfig.ListField(
        dtype = str,
        doc = "mask planes that, if set, the associated pixel should not be included in the coadd",
        default = ("EDGE", "SAT"),
    )
    coaddKernelSizeFactor = pexConfig.Field(
        dtype = float,
        doc = "coadd kernel size = coadd FWHM converted to pixels * coaddKernelSizeFactor",
        default = 3.0,
    )
    doInterp = pexConfig.Field(
        doc = "interpolate over EDGE pixels?",
        dtype = bool,
        default = True,
    )
    interpKernelFallbackFwhm = pexConfig.Field(
        dtype = float,
        doc = """normally desiredFwhm is used as the FWHM of PSF kernel for interpolating NaNs,
            but if desiredFwhm is None then interpKernelFallbackFwhm is used (arc seconds)""",
        default = 1.5,
    )
    interpKernelSizeFactor = pexConfig.Field(
        dtype = float,
        doc = "interpolation kernel size = interpFwhm converted to pixels * interpKernelSizeFactor",
        default = 3.0,
    )
    doWrite = pexConfig.Field(
        doc = "persist coadd?",
        dtype = bool,
        default = True,
    )
    

class AssembleCoaddTask(CoaddTask):
    """Assemble a coadd from a set of coaddTempExp
    """
    ConfigClass = AssembleCoaddConfig
    _DefaultName = "outlierRejectedCoadd"

    def __init__(self, *args, **kwargs):
        CoaddTask.__init__(self, *args, **kwargs)
        self._badPixelMask = afwImage.MaskU.getPlaneBitMask(self.config.badMaskPlanes)

    def getBadPixelMask(self):
        return self._badPixelMask
    
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
        
        imageRefList = self.selectExposures(patchRef=patchRef, wcs=wcs, bbox=bbox)
        
        numExp = len(imageRefList)
        if numExp < 1:
            raise pipeBase.TaskError("No exposures to coadd")
        self.log.info("Coadd %s calexp" % (numExp,))
    
        tempExpRefList = []
        tempExpName = self.config.coaddName + "CoaddTempExp"
        for ind, calExpRef in enumerate(imageRefList):
            tempExpId = calExpRef.dataId.copy()
            tempExpId.update(patchRef.dataId)
            tempExpRef = calExpRef.butlerSubset.butler.dataRef(
                datasetType = tempExpName,
                dataId = tempExpId,
            )

            if not tempExpRef.datasetExists(tempExpName):
                self.log.log(self.log.WARN, "Could not find %s %s; skipping it" % \
                    (tempExpName, calExpRef.dataId,))
                continue
            
            tempExpRefList.append(tempExpRef)

        if not tempExpRefList:
            raise pipeBase.TaskError("No images to coadd")

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
                exposure = expMeta.dataRef.get("coaddTempExp_sub", bbox=subBBox, imageOrigin="PARENT")
                maskedImage = exposure.getMaskedImage()
                if not didSetMetadata:
                    coadd.setFilter(exposure.getFilter())
                    coadd.setCalib(exposure.getCalib())
                    didSetMetadata = True

                maskedImageList.append(maskedImage)
                weightList.append(expMeta.weight)

            try:
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

    def selectExposures(self, patchRef, wcs, bbox):
        """Select exposures to coadd
        
        @param patchRef: data reference for sky map patch. Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter" or "band")
        @param[in] wcs: WCS of coadd patch
        @param[in] bbox: bbox of coadd patch
        @return a list of science exposures to coadd, as butler data references
        """
        cornerPosList = afwGeom.Box2D(bbox).getCorners()
        coordList = [wcs.pixelToSky(pos) for pos in cornerPosList]
        return self.select.runDataRef(patchRef, coordList).dataRefList
    
    def getSkyInfo(self, patchRef):
        """Return SkyMap, tract and patch

        @param patchRef: data reference for sky map. Must include keys "tract" and "patch"
        
        @return pipe_base Struct containing:
        - skyMap: sky map
        - tractInfo: information for chosen tract of sky map
        - patchInfo: information about chosen patch of tract
        - wcs: WCS of tract
        - bbox: outer bbox of patch, as an afwGeom Box2I
        """
        skyMap = patchRef.get(self.config.coaddName + "Coadd_skyMap")
        tractId = patchRef.dataId["tract"]
        tractInfo = skyMap[tractId]

        # patch format is "xIndex,yIndex"
        patchIndex = tuple(int(i) for i in patchRef.dataId["patch"].split(","))
        patchInfo = tractInfo.getPatchInfo(patchIndex)
        
        return pipeBase.Struct(
            skyMap = skyMap,
            tractInfo = tractInfo,
            patchInfo = patchInfo,
            wcs = tractInfo.getWcs(),
            bbox = patchInfo.getOuterBBox(),
        )

    @pipeBase.timeMethod
    def interpolateEdgePixels(self, exposure):
        """Interpolate over edge pixels
        
        This interpolates over things like saturated pixels and replaces edge pixels with 0.
        
        @param[in,out] exposure: exposure over which to interpolate over edge pixels
        @param[in] PSF to use to detect NaNs
        """
        self.log.info("Interpolate over EDGE pixels")
        wcs = exposure.getWcs()
        fwhm = self.config.desiredFwhm if self.config.desiredFwhm is not None \
            else self.config.interpKernelFallbackFwhm
        fwhmPixels = fwhm / wcs.pixelScale().asArcseconds()
        kernelSize = int(round(fwhmPixels * self.config.interpKernelSizeFactor))
        kernelDim = afwGeom.Point2I(kernelSize, kernelSize)
        psfModel = self.makeModelPsf(fwhmPixels=fwhmPixels, kernelDim=kernelDim)

        maskedImage = exposure.getMaskedImage()
        nanDefectList = ipIsr.getDefectListFromMask(maskedImage, "EDGE", growFootprints=0)
        measAlg.interpolateOverDefects(exposure.getMaskedImage(), psfModel, nanDefectList, 0.0)
    
    def postprocessCoadd(self, coaddExposure):
        """Postprocess the exposure coadd, e.g. interpolate over edge pixels
        """
        if self.config.doInterp:
            self.interpolateEdgePixels(exposure=coaddExposure)
        else:
            self.log.info("config.doInterp is None; do not interpolate over EDGE pixels")

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        return CoaddArgumentParser(name=cls._DefaultName, datasetType="deepCoadd")

    def _getConfigName(self):
        """Return the name of the config dataset
        """
        return "%s_coadd_config" % (self.config.coaddName,)
    
    def _getMetadataName(self):
        """Return the name of the metadata dataset
        """
        return "%s_coadd_metadata" % (self.config.coaddName,)


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


class CoaddArgumentParser(pipeBase.ArgumentParser):
    """A version of lsst.pipe.base.ArgumentParser specialized for coaddition.
    
    Required because butler.subset does not support patch and tract
    """
    def _makeDataRefList(self, namespace):
        """Make namespace.dataRefList from namespace.dataIdList
        """
        datasetType = namespace.config.coaddName + "Coadd"
        validKeys = namespace.butler.getKeys(datasetType=datasetType, level=self._dataRefLevel)

        namespace.dataRefList = []
        for dataId in namespace.dataIdList:
            # tract and patch are required
            for key in validKeys:
                if key not in dataId:
                    self.error("--id must include " + key)
            dataRef = namespace.butler.dataRef(
                datasetType = datasetType,
                dataId = dataId,
            )
            namespace.dataRefList.append(dataRef)
