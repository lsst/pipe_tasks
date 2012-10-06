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

__all__ = ["WarpAndPsfMatchTask"]

FwhmPerSigma = 2 * math.sqrt(2 * math.log(2))

class WarpAndPsfMatchConfig(pexConfig.Config):
    """Config for WarpAndPsfMatchTask
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
    warp = pexConfig.ConfigField(
        dtype = afwMath.Warper.ConfigClass,
        doc = "warper configuration",
    )


class WarpAndPsfMatchTask(pipeBase.Task):
    """A task to warp, PSF-match and zeropoint scale an exposure
    """
    ConfigClass = WarpAndPsfMatchConfig

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("psfMatch")
        self.warper = afwMath.Warper.fromConfig(self.config.warp)
        self.zeroPointScaler = coaddUtils.ZeroPointScaler(self.config.coaddZeroPoint)

    def getCalExp(self, dataRef, getPsf=True, bgSubtracted=False):
        """Return one "calexp" calibrated exposure, optionally with psf
        
        @param dataRef: a sensor-level data reference
        @param getPsf: include the PSF?
        @param bgSubtracted: return background subtracted calexp?
        @return calibrated exposure with psf
        """
        exposure = dataRef.get("calexp") #We assume calexps are background subtracted
        if not bgSubtracted:
            background = dataRef.get("calexpBackground")
            try:
                mi = exposure.getMaskedImage()
                mi += background
                del mi
            except Exception, e:
                self.log.warn("There was a problem adding the background: %s.  Continuing without adding a background."%(e))
        if getPsf:
            psf = dataRef.get("psf")
            exposure.setPsf(psf)
        return exposure
    
    def run(self, exposure, wcs, maxBBox=None, destBBox=None):
        """PSF-match exposure (if self.config.desiredFwhm is not None), warp and scale
        
        @param[in,out] exposure: exposure to preprocess; PSF matching is done in place
        @param[in] wcs: desired WCS of temporary images
        @param maxBBox: maximum allowed parent bbox of warped exposure (an afwGeom.Box2I or None);
            if None then the warped exposure will be just big enough to contain all warped pixels;
            if provided then the warped exposure may be smaller, and so missing some warped pixels;
            ignored if destBBox is not None
        @param destBBox: exact parent bbox of warped exposure (an afwGeom.Box2I or None);
            if None then maxBBox is used to determine the bbox, otherwise maxBBox is ignored
        
        @return a pipe_base Struct containing:
        - exposure: processed exposure
        """
        if self.config.desiredFwhm is not None:
            self.log.info("PSF-match exposure")
            fwhmPixels = self.config.desiredFwhm / wcs.pixelScale().asArcseconds()
            kernelDim = exposure.getPsf().getKernel().getDimensions()
            coreSigma = fwhmPixels / FwhmPerSigma
            modelPsf = afwDetection.createPsf("DoubleGaussian", kernelDim[0], kernelDim[1],
                coreSigma, coreSigma * 2.5, 0.1)
            exposure = self.psfMatch.run(exposure, modelPsf).psfMatchedExposure
        self.log.info("Warp exposure")
        with self.timer("warp"):
            exposure = self.warper.warpExposure(wcs, exposure, maxBBox=maxBBox, destBBox=destBBox)
        
        self.zeroPointScaler.scaleExposure(exposure)

        return pipeBase.Struct(
            exposure = exposure,
        )
