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
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
import lsst.ip.isr as ipIsr
from lsst.ip.diffim import ModelPsfMatchTask
from .selectImages import BadSelectImagesTask

__all__ = ["CoaddBaseTask", "CoaddCalexpBaseTask", "CoaddArgumentParser"]

FWHMPerSigma = 2 * math.sqrt(2 * math.log(2))

class CoaddBaseConfig(pexConfig.Config):
    """Config for CoaddBaseTask
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
    badMaskPlanes = pexConfig.ListField(
        dtype = str,
        doc = "mask planes that, if set, the associated pixel should not be included in the coaddTempExp",
        default = ("EDGE", "SAT"),
    )


class CoaddCalexpBaseConfig(CoaddBaseConfig):
    """Config for CoaddCalexpBaseTask
    """
    desiredFwhm = pexConfig.Field(
        doc = "desired FWHM of coadd (arc seconds); None for no FWHM matching",
        dtype = float,
        optional = True,
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
    warp = pexConfig.ConfigField(
        dtype = afwMath.Warper.ConfigClass,
        doc = "warper configuration",
    )


class CoaddBaseTask(pipeBase.CmdLineTask):
    """Base class for coaddition.
    
    Subclasses must specify _DefaultName
    """
    ConfigClass = CoaddBaseConfig
    
    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("select")
        self._badPixelMask = afwImage.MaskU.getPlaneBitMask(self.config.badMaskPlanes)

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
            self.log.log(self.log.INFO, "config.doInterp is None; do not interpolate over EDGE pixels")

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        return CoaddArgumentParser(name=cls._DefaultName, datasetType="deepCoadd")

    def _getConfigName(self):
        """Return the name of the config dataset
        """
        return "%s_%s_config" % (self.config.coaddName, self._DefaultName)
    
    def _getMetadataName(self):
        """Return the name of the metadata dataset
        """
        return "%s_%s_metadata" % (self.config.coaddName, self._DefaultName)


class CoaddCalexpBaseTask(CoaddBaseTask):
    """Base class for coaddition that adds the ability to preprocess calexp
    """
    ConfigClass = CoaddCalexpBaseConfig

    def __init__(self, *args, **kwargs):
        CoaddBaseTask.__init__(self, *args, **kwargs)
        self.makeSubtask("psfMatch")
        self.warper = afwMath.Warper.fromConfig(self.config.warp)
        self.zeroPointScaler = coaddUtils.ZeroPointScaler(self.config.coaddZeroPoint)

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
    
    def processCalexp(self, exposure, wcs, maxBBox=None, destBBox=None):
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
            self.log.log(self.log.INFO, "PSF-match exposure")
            fwhmPixels = self.config.desiredFwhm / wcs.pixelScale().asArcseconds()
            kernelDim = exposure.getPsf().getKernel().getDimensions()
            modelPsf = self.makeModelPsf(fwhmPixels=fwhmPixels, kernelDim=kernelDim)
            exposure = self.psfMatch.run(exposure, modelPsf).psfMatchedExposure
        self.log.log(self.log.INFO, "Warp exposure")
        with self.timer("warp"):
            exposure = self.warper.warpExposure(wcs, exposure, maxBBox=maxBBox, destBBox=destBBox)
        
        self.zeroPointScaler.scaleExposure(exposure)

        return exposure


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
