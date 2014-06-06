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
import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
from lsst.ip.diffim import ModelPsfMatchTask
from .coaddBase import CoaddBaseTask, DoubleGaussianPsfConfig
from .interpImage import InterpImageTask
from .warpAndPsfMatch import WarpAndPsfMatchTask

__all__ = ["CoaddTask"]

class CoaddConfig(CoaddBaseTask.ConfigClass):
    """Config for CoaddTask
    """
    doPsfMatch = pexConfig.Field(dtype=bool, doc="Match to modelPsf?", default=False)
    modelPsf = pexConfig.ConfigField(dtype=DoubleGaussianPsfConfig, doc="Model Psf specification")
    warpAndPsfMatch = pexConfig.ConfigurableField(
        target = WarpAndPsfMatchTask,
        doc = "Task to warp, PSF-match and zero-point-match calexp",
    )
    scaleZeroPoint = pexConfig.ConfigurableField(
        target = coaddUtils.ScaleZeroPointTask,
        doc = "Task to compute zero point scale",
    )
    doInterp = pexConfig.Field(
        doc = "Interpolate over NO_DATA pixels?",
        dtype = bool,
        default = True,
    )
    interpImage = pexConfig.ConfigurableField(
        target = InterpImageTask,
        doc = "Task to interpolate over NO_DATA pixels",
    )
    interpFwhm = pexConfig.Field(
        dtype = float,
        doc = "FWHM of PSf used for interpolation (arcsec)",
        default = 1.5,
        check = lambda x: x > 0,
        )
    doWrite = pexConfig.Field(
        doc = "Persist coadd and associated products?",
        dtype = bool,
        default = True,
    )


class CoaddTask(CoaddBaseTask):
    """Coadd images by PSF-matching (optional), warping and computing a weighted sum
    """
    ConfigClass = CoaddConfig
    _DefaultName = "coadd"

    def __init__(self, *args, **kwargs):
        CoaddBaseTask.__init__(self, *args, **kwargs)
        self.makeSubtask("interpImage")
        self.makeSubtask("warpAndPsfMatch")
        self.makeSubtask("scaleZeroPoint")

    @pipeBase.timeMethod
    def run(self, patchRef, selectDataList=[]):
        """Coadd images by PSF-matching (optional), warping and computing a weighted sum
        
        This task is deprecated: the preferred technique is to use makeCoaddTempExp followed by assembleCoadd,
        configuring the latter to disable outlier rejection.
        However, this task computes a weight map and assembleCoadd cannot.
        Once the afw statistics stacker can compute a weight map this task will go away.
        
        PSF-matching is performed before warping so the code can use the PSF models
        associated with the calibrated science exposures (without having to warp those models).
        This is clearly incorrect (as the warping will change the PSF as a function of position).
        
        Coaddition is performed as a weighted sum. See lsst.coadd.utils.Coadd for details.
    
        @param patchRef: data reference for sky map patch. Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter" or "band")
        @return: a pipeBase.Struct with fields:
        - coadd: a coaddUtils.Coadd object
        - coaddExposure: coadd exposure, as returned by coadd.getCoadd()
        """
        skyInfo = self.getSkyInfo(patchRef)
        tractWcs = skyInfo.wcs
        patchBBox = skyInfo.bbox

        imageRefList = self.selectExposures(patchRef, skyInfo, selectDataList=selectDataList)
        if len(imageRefList) == 0:
            raise pipeBase.TaskError("No exposures to coadd")
        self.log.info("Coadding %d exposures" % len(imageRefList))

        modelPsf = self.makeModelPsf(self.config.modelPsf, tractWcs) if self.doPsfMatch else None
        coaddData = self.warpAndCoadd(imageRefList, patchBBox, tractWcs, modelPsf=modelPsf)
        if self.config.doInterp:
            self.interpolateExposure(coaddData.coaddExposure)

        if self.config.doWrite:
            self.writeCoaddOutput(patchRef, coaddData.coaddExposure)
            self.writeCoaddOutput(patchRef, coaddData.weightMap, "depth")

        return coaddData
    
    def makeCoadd(self, bbox, wcs):
        """Make a coadd object, e.g. lsst.coadd.utils.Coadd
        
        @param[in] bbox: bounding box for coadd
        @param[in] wcs: WCS for coadd
        
        This exists to allow subclasses to return a different kind of coadd
        """
        return coaddUtils.Coadd(bbox=bbox, wcs=wcs, badMaskPlanes=self.config.badMaskPlanes)

    def warpAndCoadd(self, imageRefList, bbox, wcs, modelPsf=None):
        """Warp and coadd each input image

        Individual CCDs within an exposure are treated separately.

        @param imageRefList: List of input image data references
        @param bbox: bounding box for coadd
        @param wcs: Wcs for coadd
        @param modelPsf: Target model PSF (or None for no PSF matching)
        @return Struct with:
        - coaddExposure: the coadded exposure
        - weightMap: the weight map of the coadded exposure
        - coadd: coaddUtils.Coadd object with results
        """
        coadd = self.makeCoadd(bbox, wcs)
        for ind, calExpRef in enumerate(imageRefList):
            if not calExpRef.datasetExists("calexp"):
                self.log.warn("Could not find calexp %s; skipping it" % (calExpRef.dataId,))
                continue

            self.log.info("Processing exposure %d of %d: %s" % (ind+1, len(imageRefList), calExpRef.dataId))
            exposure = self.getCalExp(calExpRef, bgSubtracted=True)
            try:
                exposure = self.warpAndPsfMatch.run(exposure, wcs=wcs, modelPsf=modelPsf,
                                                    maxBBox=bbox).exposure
                self.scaleExposure(exposure)
                coadd.addExposure(exposure)
            except Exception, e:
                self.log.warn("Error processing exposure %s; skipping it: %s" % (calExpRef.dataId, e))
                continue
        
        coaddExposure = coadd.getCoadd()
        if modelPsf is not None:
            coaddExposure.setPsf(modelPsf)
        coaddExposure.setCalib(self.scaleZeroPoint.getCalib())
        return pipeBase.Struct(coaddExposure=coaddExposure, weightMap=coadd.getWeightMap(), coadd=coadd)

    def scaleExposure(self, exposure):
        """Apply photometric scaling to an exposure

        The exposure is scaled in-place.
        """
        scale = self.scaleZeroPoint.scaleFromCalib(exposure.getCalib()).scale
        maskedImage = exposure.getMaskedImage()
        maskedImage *= scale

    def interpolateExposure(self, exp):
        """Interpolate an exposure to remove bad pixels

        In this case, we're interested in interpolating over "NO_DATA" pixels,
        which are pixels that have no contributing (good) input pixels.

        Presently, the interpolation code in meas_algorithms doesn't use
        the input PSF, so we're not too careful what we give it.
        """
        fwhmArcSec = self.config.interpFwhm
        fwhmPixels = fwhmArcSec / exp.getWcs().pixelScale().asArcseconds()
        self.interpImage.interpolateOnePlane(
            maskedImage = exp.getMaskedImage(),
            planeName = "NO_DATA",
            fwhmPixels = fwhmPixels,
            )
