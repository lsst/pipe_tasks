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
from .coaddBase import CoaddBaseTask, CoaddArgumentParser
from .warpAndPsfMatch import InterpTask
from .warpAndPsfMatch import WarpAndPsfMatchTask

# export CoaddArgumentParser for backward compatibility; new code should get it from coaddBase

__all__ = ["CoaddTask", "CoaddArgumentParser"]

FWHMPerSigma = 2 * math.sqrt(2 * math.log(2))

class CoaddConfig(CoaddBaseTask.ConfigClass):
    """Config for CoaddTask
    """
    warpAndPsfMatch = pexConfig.ConfigurableField(
        target = WarpAndPsfMatchTask,
        doc = "Task to warp, PSF-match and zero-point-match calexp",
    )
    coaddKernelSizeFactor = pexConfig.Field(
        dtype = float,
        doc = "coadd kernel size = coadd FWHM converted to pixels * coaddKernelSizeFactor",
        default = 3.0,
    )
    doInterp = pexConfig.Field(
        doc = "Interpolate over EDGE pixels?",
        dtype = bool,
        default = True,
    )
    interp = pexConfig.ConfigurableField(
        target = InterpTask,
        doc = "Task to interpolate over EDGE pixels",
    )
    doWrite = pexConfig.Field(
        doc = "persist coadd?",
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
            exposure = self.warpAndPsfMatch.getCalExp(calExpRef, getPsf=doPsfMatch)
            try:
                exposure = self.warpAndPsfMatch.run(calexp, wcs=tractWcs, maxBBox=patchBBox).exposure            
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
    
    def makeCoadd(self, bbox, wcs):
        """Make a coadd object, e.g. lsst.coadd.utils.Coadd
        
        @param[in] bbox: bounding box for coadd
        @param[in] wcs: WCS for coadd
        
        This exists to allow subclasses to return a different kind of coadd
        """
        return coaddUtils.Coadd(bbox=bbox, wcs=wcs, badMaskPlanes=self.config.badMaskPlanes)
