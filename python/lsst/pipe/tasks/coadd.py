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
import math
import os
import sys

import lsst.pex.config as pexConfig
import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.coadd.utils as coaddUtils
import lsst.ip.diffim as ipDiffIm
import lsst.pipe.base as pipeBase

FWHMPerSigma = 2 * math.sqrt(2 * math.log(2))

class CoaddConfig(pexConfig.Config):
    """Config for CoaddTask
    """
    coadd    = pexConfig.ConfigField(dtype = coaddUtils.Coadd.ConfigClass, doc = "")
    warp     = pexConfig.ConfigField(dtype = afwMath.Warper.ConfigClass, doc = "")
    psfMatch = pexConfig.ConfigField(dtype = ipDiffIm.ModelPsfMatchTask.ConfigClass, doc = "a hack!")


class CoaddTask(pipeBase.Task):
    """Coadd images by PSF-matching (optional), warping and computing a weighted sum
    """
    ConfigClass = CoaddConfig
    
    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("psfMatch", ipDiffIm.ModelPsfMatch)
        self.warper = afwMath.Warper.fromConfig(self.config.warp)
        self._prevKernelDim = afwGeom.Extent2I(0, 0)
        self._modelPsf = None
    
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
    
    def makeCoadd(self, bbox, wcs):
        """Make a coadd object, e.g. lsst.coadd.utils.Coadd
        
        @param[in] bbox: bounding box for coadd
        @param[in] wcs: WCS for coadd
        """
        return coaddUtils.Coadd.fromConfig(bbox=bbox, wcs=wcs, config=self.config.coadd)
    
    def makeModelPsf(self, exposure, desFwhm):
        """Construct a model PSF, or reuse the prior model, if possible
        
        The model PSF is a double Gaussian with core FWHM = desFwhm
        and wings of amplitude 1/10 of core and FWHM = 2.5 * core.
        
        @param exposure: exposure containing a psf; the model PSF will have the same dimensions
        @param desFwhm: desired FWHM of PSF, in pixels
        @return model PSF
        
        @raise RuntimeError if desFwhm <= 0
        """
        if desFwhm <= 0:
            raise RuntimeError("desFwhm = %s; must be positive" % (desFwhm,))
        psfKernel = exposure.getPsf().getKernel()
        kernelDim = psfKernel.getDimensions()
        if self._modelPsf is None or kernelDim != self._prevKernelDim:
            self._prevKernelDim = kernelDim
            self.log.log(self.log.INFO,
                "Create double Gaussian PSF model with core fwhm %0.1f and size %dx%d" % \
                (desFwhm, kernelDim[0], kernelDim[1]))
            coreSigma = desFwhm / FWHMPerSigma
            self._modelPsf = afwDetection.createPsf("DoubleGaussian", kernelDim[0], kernelDim[1],
                coreSigma, coreSigma * 2.5, 0.1)
        return self._modelPsf

    def run(self, dataRefList, bbox, wcs, desFwhm):
        """Coadd images by PSF-matching (optional), warping and computing a weighted sum
        
        PSF matching is to a double gaussian model with core FWHM = desFwhm
        and wings of amplitude 1/10 of core and FWHM = 2.5 * core.
        The size of the PSF matching kernel is the same as the size of the kernel
        found in the first calibrated science exposure, since there is no benefit
        to making it any other size.
        
        PSF-matching is performed before warping so the code can use the PSF models
        associated with the calibrated science exposures (without having to warp those models).
        
        Coaddition is performed as a weighted sum. See lsst.coadd.utils.Coadd for details.
    
        @param dataRefList: list of data identity dictionaries
        @param bbox: bounding box of coadd
        @param wcs: WCS of coadd
        @param desFwhm: desired FWHM of PSF, in science exposure pixels
                (note that the coadd often has a different scale than the science images);
                if 0 then no PSF matching is performed.
        @return: a pipeBase.Struct with fields:
        - coadd: a coaddUtils.Coadd object; call coadd.getCoadd() to get the coadd exposure
        """
        numExp = len(dataRefList)
        if numExp < 1:
            raise RuntimeError("No exposures to coadd")
        self.log.log(self.log.INFO, "Coadd %s calexp" % (numExp,))

        doPsfMatch = desFwhm > 0
    
        if not doPsfMatch:
            self.log.log(self.log.INFO, "No PSF matching will be done (desFwhm <= 0)")
    
        coadd = self.makeCoadd(bbox, wcs)
        for ind, dataRef in enumerate(dataRefList):
            self.log.log(self.log.INFO, "Processing exposure %d of %d: id=%s" % (ind+1, numExp, dataRef.dataId))
            exposure = self.getCalexp(dataRef, getPsf=doPsfMatch)
            if desFwhm > 0:
                modelPsf = self.makeModelPsf(exposure, desFwhm)
                self.log.log(self.log.INFO, "PSF-match exposure")
                psfRes = self.psfMatch.run(exposure, modelPsf)
                exposure = psfRes.psfMatchedExposure
            self.log.log(self.log.INFO, "Warp exposure")
            exposure = self.warper.warpExposure(wcs, exposure, maxBBox = bbox)
            coadd.addExposure(exposure)
        
        return pipeBase.Struct(
            coadd = coadd,
        )
