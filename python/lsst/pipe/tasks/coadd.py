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

# experimental use of Martin's new Config
# move this to afwMath once Config is close enough to start modifying afw
class TempWarpConfig(pexConfig.Config):
    warpingKernelName = pexConfig.ChoiceField(
        doc = "Warping kernel",
        dtype = str,
        default = "lanczos4",
        allowed = {
            "nearest":  "nearest-neighbor interpolation; kernel size is 2x2, with only one nonzero pixel",
            "bilinear": "bilinear interpolation; kernel size is 2x2",
            "lanczos2": "Lanczos kernel of order 2; kernel size is 4x4",
            "lanczos3": "Lanczos kernel of order 3; kernel size is 6x6",
            "lanczos4": "Lanczos kernel of order 4; kernel size is 8x8",
            "lanczos5": "Lanczos kernel of order 5; kernel size is 10x10",
            "lanczos6": "Lanczos kernel of order 5; kernel size is 12x12",
        }
        optional = False,
    )
    interpLength = pexConfig.Field(
        doc = "interpLength argument to lsst.afw.math.warpExposure",
        dtype = int,
        default = 10,
        optional = False,
    )
    cacheSize = pexConfig.Field(
        doc = "cacheSize argument to lsst.afw.math.SeparableKernel.computeCache",
        dtype = int,
        default = 0,
        optional = False,
    )


class TempPsfMatchConfig(Config):
    pass # too complex to put here and it belongs in ip_diffim anyway; it makes a great test of pexConfig!


class CoaddConfig(Config):
    """Config for CoaddTask
    """
    coadd = pexConfig.ConfigField(coaddUtils.Coadd.ConfigClass, doc="", optional=False)
    warp = pexConfig.ConfigField(TempWarpConfig, doc="", optional=False)
    psfMatch = pexConfig.ConfigField(TempPsfMatchConfig, doc="", optional=False)


class CoaddTask(pipeBase.Task):
    """Coadd images by PSF-matching (optional), warping and computing a weighted sum
    """
    ConfigClass = CoaddConfig
    
    def __init__(self, *args, **kwargs):
        self.pipeBase.Task.__init__(self, *args, **kwargs)
        self.psfMatcher = ipDiffIm.ModelPsfMatch(self.config.psfMatchConfig)
        self.warper = afwMath.Warper.fromConfig(self.config.warpConfig)
        self._prevKernelDim = afwGeom.Extent2I(0, 0)
        self._modelPsf = None
    
    def getCalexp(self, butler, id):
        """Return one "calexp" calibrated exposure, with psf
        
        @param butler: data butler
        @param dataId: data identity of exposure
        @return calibrated exposure with psf
        """
        exposure = butler.get("calexp", id)
        psf = butler.get("psf", id)
        exposure.setPsf(psf)
        return exposure
    
    def makeCoadd(self, coaddBBox, coaddWcs):
        """Make a coadd object, e.g. lsst.coadd.utils.Coadd
        """
        return coaddUtils.Coadd.fromConfig(coaddBBox, coaddWcs, self.config.coaddConfig)
    
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
            self.log.log(pexLog.INFO,
                "Create double Gaussian PSF model with core fwhm %0.1f and size %dx%d" % \
                (desFwhm, kernelDim[0], kernelDim[1]))
            coreSigma = desFwhm / FWHMPerSigma
            self._modelPsf = afwDetection.createPsf("DoubleGaussian", kernelDim[0], kernelDim[1],
                coreSigma, coreSigma * 2.5, 0.1)
        return self._modelPsf

    def run(self, butler, idList, coaddBBox, coaddWcs, desFwhm):
        """Coadd images by PSF-matching (optional), warping and computing a weighted sum
        
        PSF matching is to a double gaussian model with core FWHM = desFwhm
        and wings of amplitude 1/10 of core and FWHM = 2.5 * core.
        The size of the PSF matching kernel is the same as the size of the kernel
        found in the first calibrated science exposure, since there is no benefit
        to making it any other size.
        
        PSF-matching is performed before warping so the code can use the PSF models
        associated with the calibrated science exposures (without having to warp those models).
        
        Coaddition is performed as a weighted sum. See lsst.coadd.utils.Coadd for details.
    
        @param butler: data butler
        @param idList: list of data identity dictionaries
        @param coaddBBox: bounding box for coadd
        @param coaddWcs: WCS for coadd
        @param desFwhm: desired FWHM of PSF, in science exposure pixels
                (note that the coadd often has a different scale than the science images);
                if 0 then no PSF matching is performed.
        @return: a pipeBase.Struct with fields:
        - coadd: a coaddUtils.Coadd object; call coadd.getExposure() to get the coadd exposure
        """
        numExp = len(idList)
        if numExp < 1:
            raise RuntimeError("No exposures to coadd")
        self.log.log(pexLog.INFO, "Coadd %s calexp" % (numExp,))
    
        if desFwhm <= 0:
            self.log.log(pexLog.INFO, "No PSF matching will be done (desFwhm <= 0)")
    
        coadd = self.makeCoadd(coaddBBox, coaddWcs)
        for ind, id in enumerate(idList):
            self.log.log(pexLog.INFO, "Processing exposure %d of %d: id=%s" % (ind+1, numExp, id))
            exposure = self.getCalexp(butler, id)
            if desFwhm > 0:
                modelPsf = self.makeModelPsf(exposure, desFwhm)
                self.log.log(pexLog.INFO, "PSF-match exposure")
                exposure, psfMatchingKernel, kernelCellSet = self.psfMatcher.matchExposure(exposure, modelPsf)
            self.log.log(pexLog.INFO, "Warp exposure")
            exposure = self.warper.warpExposure(coaddWcs, exposure, maxBBox = coaddBBox)
            coadd.addExposure(exposure)
        
        return pipeBase.Struct(
            coadd = coadd,
        )
