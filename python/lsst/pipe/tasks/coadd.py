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

import lsst.pex.config as pexConfig
import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
import lsst.ip.isr as ipIsr
from lsst.ip.diffim import ModelPsfMatchTask
from lsst.pipe.tasks.selectImages import BadSelectImagesTask

__all__ = ["CoaddTask", "CoaddArgumentParser"]

FWHMPerSigma = 2 * math.sqrt(2 * math.log(2))

class CoaddConfig(pexConfig.Config):
    """Config for CoaddTask
    """
    coaddName = pexConfig.Field(
        doc = "coadd name: typically one of deep or goodSeeing",
        dtype = str,
        default = "deep",
    )
    select = pexConfig.ConfigurableField(
        doc = "image selection subtask",
        target = BadSelectImagesTask,
    )
    desiredFwhm = pexConfig.Field(
        doc = "desired FWHM of coadd (arc seconds); None for no FWHM matching",
        dtype = float,
        optional = True,
    )
    badMaskPlanes = pexConfig.ListField(
        dtype = str,
        doc = "mask planes that, if set, the associated pixel should not be included in the coadd",
        default = ("EDGE", "SAT"),
    )
    coaddZeroPoint = pexConfig.Field(
        dtype = float,
        doc = "photometric zero point of coadd (mag)",
        default = 27.0,
    )
    psfMatch = pexConfig.ConfigurableField(
        target = ModelPsfMatchTask,
        doc = "PSF matching model to model task",
    )
    warp = pexConfig.ConfigField(
        dtype = afwMath.Warper.ConfigClass,
        doc = "warper configuration",
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


class CoaddTask(pipeBase.CmdLineTask):
    """Coadd images by PSF-matching (optional), warping and computing a weighted sum
    """
    ConfigClass = CoaddConfig
    _DefaultName = "coadd"
    
    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("select")
        self.makeSubtask("psfMatch")
        self.warper = afwMath.Warper.fromConfig(self.config.warp)
        self.zeroPointScaler = coaddUtils.ZeroPointScaler(self.config.coaddZeroPoint)

    @pipeBase.timeMethod
    def run(self, patchRef):
        """Coadd images by PSF-matching (optional), warping and computing a weighted sum
        
        PSF matching is to a double gaussian model with core FWHM = self.config.desiredFwhm
        and wings of amplitude 1/10 of core and FWHM = 2.5 * core.
        The size of the PSF matching kernel is the same as the size of the kernel
        found in the first calibrated science exposure, since there is no benefit
        to making it any other size.
        
        PSF-matching is performed before warping so the code can use the PSF models
        associated with the calibrated science exposures (without having to warp those models).
        
        Coaddition is performed as a weighted sum. See lsst.coadd.utils.Coadd for details.
    
        @param patchRef: data reference for sky map patch. Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter" or "band")
        @return: a pipeBase.Struct with fields:
        - coadd: a coaddUtils.Coadd object
        - coaddExposure: coadd exposure, as returned by coadd.getCoadd()
        """
        skyInfo = self.getSkyInfo(patchRef)
        
        wcs = skyInfo.wcs
        bbox = skyInfo.bbox
        
        imageRefList = self.selectExposures(patchRef=patchRef, wcs=wcs, bbox=bbox)
        
        numExp = len(imageRefList)
        if numExp < 1:
            raise pipeBase.TaskError("No exposures to coadd")
        self.log.info("Coadd %s calexp" % (numExp,))
    
        doPsfMatch = self.config.desiredFwhm is not None
        if not doPsfMatch:
            self.log.info("No PSF matching will be done (desiredFwhm is None)")
    
        coadd = self.makeCoadd(bbox, wcs)
        for ind, dataRef in enumerate(imageRefList):
            if not dataRef.datasetExists("calexp"):
                self.log.warn("Could not find calexp %s; skipping it" % (dataRef.dataId,))
                continue

            self.log.info("Processing exposure %d of %d: id=%s" % \
                (ind+1, numExp, dataRef.dataId))
            exposure = self.getCalExp(dataRef, getPsf=doPsfMatch)
            try:
                exposure = self.preprocessExposure(exposure, wcs=wcs, destBBox=bbox)
            except Exception, e:
                self.log.warn("Error preprocessing exposure %s; skipping it: %s" % \
                    (dataRef.dataId, e))
                continue
            try:
                coadd.addExposure(exposure)
            except RuntimeError, e:
                self.log.warn("Could not add exposure to coadd: %s" % (e,))
        
        coaddExposure = coadd.getCoadd()
        self.postprocessCoadd(coaddExposure)

        self.persistCoadd(patchRef, coaddExposure)
        
        return pipeBase.Struct(
            coaddExposure = coaddExposure,
            coadd = coadd,
        )
    
    def persistCoadd(self, patchRef, coaddExposure):
        """Persist coadd and PSF, as appropriate
        
        If self.config.doWrite is False then do nothing.
        If self.config.desiredFwhm is not None, then compute the model PSF and persist it.
        
        @param[in] patchRef: data reference to coadd patch
        @param[in] coaddExposure: coadd exposure
        """
        if self.config.doWrite:
            coaddName = self.config.coaddName + "Coadd"
            self.log.info("Persisting %s" % (coaddName,))
            patchRef.put(coaddExposure, coaddName)
            if self.config.desiredFwhm is not None:
                psfName = self.config.coaddName + "Coadd_initPsf"
                self.log.info("Persisting %s" % (psfName,))
                wcs = coaddExposure.getWcs()
                fwhmPixels = self.config.desiredFwhm / wcs.pixelScale().asArcseconds()
                kernelSize = int(round(fwhmPixels * self.config.coaddKernelSizeFactor))
                kernelDim = afwGeom.Point2I(kernelSize, kernelSize)
                coaddPsf = self.makeModelPsf(fwhmPixels=fwhmPixels, kernelDim=kernelDim)
                patchRef.put(coaddPsf, psfName)
    
    def selectExposures(self, patchRef, wcs, bbox):
        """Select exposures to coadd
        
        @param patchRef: data reference for sky map patch. Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter" or "band")
        @param[in] wcs: WCS of coadd patch
        @param[in] bbox: bbox of coadd patch
        @return a list of science exposures to coadd, as butler data references
        """
        cornerPosList = _getBox2DCorners(bbox)
        coordList = [wcs.pixelToSky(pos) for pos in cornerPosList]
        return self.select.runDataRef(patchRef, coordList).dataRefList
    
    def getCalExp(self, dataRef, getPsf=True):
        """Return one "calexp" calibrated exposure, perhaps with psf
        
        @param dataRef: a sensor-level data reference
        @param getPsf: include the PSF?
        @return calibrated exposure with psf
        """
        exposure = dataRef.get("calexp")
        if getPsf:
            psf = dataRef.get("psf")
            exposure.setPsf(psf)
        return exposure

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
    
    def makeCoadd(self, bbox, wcs):
        """Make a coadd object, e.g. lsst.coadd.utils.Coadd
        
        @param[in] bbox: bounding box for coadd
        @param[in] wcs: WCS for coadd
        
        This exists to allow subclasses to return a different kind of coadd
        """
        return coaddUtils.Coadd(bbox=bbox, wcs=wcs, badMaskPlanes=self.config.badMaskPlanes)
    
    def makeModelPsf(self, fwhmPixels, kernelDim):
        """Construct a model PSF, or reuse the prior model, if possible
        
        The model PSF is a double Gaussian with core FWHM = fwhmPixels
        and wings of amplitude 1/10 of core and FWHM = 2.5 * core.
        
        @param fwhmPixels: desired FWHM of core Gaussian, in pixels
        @param kernelDim: desired dimensions of PSF kernel, in pixels
        @return model PSF
        """
        self.log.log(self.log.DEBUG,
            "Create double Gaussian PSF model with core fwhm %0.1f pixels and size %dx%d" % \
            (fwhmPixels, kernelDim[0], kernelDim[1]))
        coreSigma = fwhmPixels / FWHMPerSigma
        return afwDetection.createPsf("DoubleGaussian", kernelDim[0], kernelDim[1],
            coreSigma, coreSigma * 2.5, 0.1)
    
    def preprocessExposure(self, exposure, wcs, maxBBox=None, destBBox=None):
        """PSF-match exposure (if self.config.desiredFwhm is not None), warp and scale
        
        @param[in,out] exposure: exposure to preprocess; FWHM fitting is done in place
        @param[in] wcs: desired WCS of temporary images
        @param maxBBox: maximum allowed parent bbox of warped exposure (an afwGeom.Box2I or None);
            if None then the warped exposure will be just big enough to contain all warped pixels;
            if provided then the warped exposure may be smaller, and so missing some warped pixels;
            ignored if destBBox is not None
        @param destBBox: exact parent bbox of warped exposure (an afwGeom.Box2I or None);
            if None then maxBBox is used to determine the bbox, otherwise maxBBox is ignored
        
        @return preprocessed exposure
        """
        if self.config.desiredFwhm is not None:
            self.log.info("PSF-match exposure")
            fwhmPixels = self.config.desiredFwhm / wcs.pixelScale().asArcseconds()
            kernelDim = exposure.getPsf().getKernel().getDimensions()
            modelPsf = self.makeModelPsf(fwhmPixels=fwhmPixels, kernelDim=kernelDim)
            exposure = self.psfMatch.run(exposure, modelPsf).psfMatchedExposure
        self.log.info("Warp exposure")
        with self.timer("warp"):
            exposure = self.warper.warpExposure(wcs, exposure, maxBBox=maxBBox, destBBox=destBBox)
        
        self.zeroPointScaler.scaleExposure(exposure)

        return exposure
    
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


def _getBox2DCorners(bbox):
    """Return the four corners of a bounding box (Box2I or Box2D) as four afwGeom Point2D
    """
    bbox = afwGeom.Box2D(bbox) # mak
    return (
        bbox.getMin(),
        afwGeom.Point2D(bbox.getMaxX(), bbox.getMinY()),
        bbox.getMax(),
        afwGeom.Point2D(bbox.getMinX(), bbox.getMaxY()),
    )
