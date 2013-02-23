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

    def getCalExp(self, dataRef, getPsf=True, bgSubtracted=False):
        """Return one "calexp" calibrated exposure, optionally with psf
        
        @param dataRef: a sensor-level data reference
        @param getPsf: include the PSF?
        @param bgSubtracted: return calexp with background subtracted? If False then
            get the calexp's background background model and add it to the calexp.
        @return calibrated exposure with psf
        """
        exposure = dataRef.get("calexp", immediate=True)
        if not bgSubtracted:
            background = dataRef.get("calexpBackground", immediate=True)
            mi = exposure.getMaskedImage()
            mi += background
            del mi
        if getPsf:
            psf = dataRef.get("psf", immediate=True)
            exposure.setPsf(psf)
        return exposure
    
    def run(self, exposure, wcs, maxBBox=None, destBBox=None):
        """PSF-match exposure (if self.config.desiredFwhm is not None) and warp
        
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
        
        return pipeBase.Struct(
            exposure = exposure,
        )
