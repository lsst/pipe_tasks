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
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
from lsst.ip.diffim import ModelPsfMatchTask
from .coaddArgumentParser import CoaddArgumentParser

FWHMPerSigma = 2 * math.sqrt(2 * math.log(2))

class NullSelectTask(pipeBase.Task):
    ConfigClass = pexConfig.Config
    def runDataRef(self, dataRef, coordList):
        raise RuntimeError("No select task specified")
    
    def searchWholeSky(self, dataRef):
        raise RuntimeError("No select task specified")

class CoaddConfig(pexConfig.Config):
    """Config for CoaddTask
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
    desFwhm = pexConfig.Field(
        doc = "desired FWHM of coadd; 0 for no FWHM matching",
        dtype = float,
        default = 0,
    )
    badMaskPlanes = pexConfig.ListField(
        dtype = str,
        doc = "mask planes that, if set, the associated pixel should not be included in the coadd",
        default = ("EDGE", "SAT"),
    )
    coaddZeroPoint = pexConfig.Field(
        dtype = float,
        doc = "Photometric zero point of coadd (mag).",
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
        self._prevKernelDim = afwGeom.Extent2I(0, 0)
        self._modelPsf = None

    @pipeBase.timeMethod
    def run(self, patchRef):
        """Coadd images by PSF-matching (optional), warping and computing a weighted sum
        
        PSF matching is to a double gaussian model with core FWHM = self.config.desFwhm
        and wings of amplitude 1/10 of core and FWHM = 2.5 * core.
        The size of the PSF matching kernel is the same as the size of the kernel
        found in the first calibrated science exposure, since there is no benefit
        to making it any other size.
        
        PSF-matching is performed before warping so the code can use the PSF models
        associated with the calibrated science exposures (without having to warp those models).
        
        Coaddition is performed as a weighted sum. See lsst.coadd.utils.Coadd for details.
    
        @param patchRef: data reference for sky map. Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter" or "band")
        @return: a pipeBase.Struct with fields:
        - coadd: a coaddUtils.Coadd object
        - coaddExposure: coadd exposure, as returned by coadd.getCoadd()
        """
        skyInfo = self.getSkyInfo(patchRef)
        
        wcs = skyInfo.wcs
        bbox = skyInfo.bbox
        cornerPosList = _getBox2DCorners(bbox)
        coordList = [wcs.pixelToSky(pos) for pos in cornerPosList]
        
        # determine which images to coadd
        imageRefList = self.select.runDataRef(patchRef, coordList).dataRefList
        
        numExp = len(imageRefList)
        if numExp < 1:
            raise RuntimeError("No exposures to coadd")
        self.log.log(self.log.INFO, "Coadd %s calexp" % (numExp,))
    
        doPsfMatch = self.config.desFwhm > 0
        if not doPsfMatch:
            self.log.log(self.log.INFO, "No PSF matching will be done (desFwhm <= 0)")
    
        coadd = self.makeCoadd(bbox, wcs)
        for ind, dataRef in enumerate(imageRefList):
            if not dataRef.datasetExists("calexp"):
                self.log.log(self.log.WARN, "Could not find calexp %s; skipping it" % (dataRef.dataId,))
                continue

            self.log.log(self.log.INFO, "Processing exposure %d of %d: id=%s" % \
                (ind+1, numExp, dataRef.dataId))
            exposure = self.getCalexp(dataRef, getPsf=doPsfMatch)
            exposure = self.preprocessExposure(exposure, wcs=wcs, destBBox=bbox)
            exposure.writeFits("temp%s.fits" % (ind,))
            try:
                coadd.addExposure(exposure)
            except RuntimeError, e:
                self.log.log(self.log.WARN, "Could not add exposure to coadd: %s" % (e,))
        
        coaddExposure = coadd.getCoadd()

        if self.config.doWrite:
            patchRef.put(coaddExposure, self.config.coaddName + "Coadd")
        
        return pipeBase.Struct(
            coaddExposure = coaddExposure,
            coadd = coadd,
        )
    
    def getCalexp(self, dataRef, getPsf=True):
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
    
    def makeCoadd(self, bbox, wcs):
        """Make a coadd object, e.g. lsst.coadd.utils.Coadd
        
        @param[in] bbox: bounding box for coadd
        @param[in] wcs: WCS for coadd
        
        This exists to allow subclasses to return a different kind of coadd
        """
        return coaddUtils.Coadd(bbox=bbox, wcs=wcs, badMaskPlanes=self.config.badMaskPlanes)
    
    def makeModelPsf(self, kernelDim):
        """Construct a model PSF, or reuse the prior model, if possible
        
        The model PSF is a double Gaussian with core FWHM = self.config.desFwhm
        and wings of amplitude 1/10 of core and FWHM = 2.5 * core.
        
        @param kernelDim: desired dimensions of PSF kernel
        @return model PSF
        
        @raise RuntimeError if self.config.desFwhm <= 0
        """
        desFwhm = self.config.desFwhm
        if desFwhm <= 0:
            raise RuntimeError("desFwhm = %s; must be positive" % (desFwhm,))
        kernelDim = exposure.getPsf().getKernel().getDimensions()
        if self._modelPsf is None or kernelDim != self._prevKernelDim:
            self._prevKernelDim = kernelDim
            self.log.log(self.log.INFO,
                "Create double Gaussian PSF model with core fwhm %0.1f and size %dx%d" % \
                (desFwhm, kernelDim[0], kernelDim[1]))
            coreSigma = desFwhm / FWHMPerSigma
            self._modelPsf = afwDetection.createPsf("DoubleGaussian", kernelDim[0], kernelDim[1],
                coreSigma, coreSigma * 2.5, 0.1)
        return self._modelPsf
    
    def preprocessExposure(self, exposure, wcs, maxBBox=None, destBBox=None):
        """PSF-match exposure (if self.config.desFwhm > 0), warp and scale
        
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
        if self.config.desFwhm > 0:
            kernelDim = exposure.getPsf().getKernel().getDimensions()
            modelPsf = self.makeModelPsf(kernelDim=kernelDim)
            self.log.log(self.log.INFO, "PSF-match exposure")
            exposure = self.psfMatch.run(exposure, modelPsf).psfMatchedExposure
        
        self.log.log(self.log.INFO, "Warp exposure")
        with self.timer("warp"):
            exposure = self.warper.warpExposure(wcs, exposure, maxBBox=maxBBox, destBBox=destBBox)
        
        self.zeroPointScaler.scaleExposure(exposure)

        return exposure

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        return CoaddArgumentParser(name=cls._DefaultName, datasetType="deepCoadd")


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
