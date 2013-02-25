# 
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011 LSST Corporation.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the LSST License Statement and 
# the GNU General Public License along with this program.  If not, 
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import math
import lsst.pex.config as pexConfig
import lsst.afw.cameraGeom  as cameraGeom
import lsst.afw.display.ds9 as ds9
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.detection as afwDet
import lsst.meas.algorithms as measAlg
import lsst.pipe.base as pipeBase

import lsstDebug

class RepairConfig(pexConfig.Config):
    doInterpolate = pexConfig.Field(
        dtype = bool,
        doc = "Interpolate over defects? (ignored unless you provide a list of defects)",
        default = True,
    )

    doCosmicRay = pexConfig.Field(
        dtype = bool,
        doc = "Find and mask out cosmic rays?",
        default = True,
    )
    cosmicray = pexConfig.ConfigField(
        dtype = measAlg.FindCosmicRaysConfig,
        doc = "Options for finding and masking cosmic rays",
    )

class RepairTask(pipeBase.Task):
    """Conversion notes:
    
    Display code should be updated once we settle on a standard way of controlling what is displayed.
    """
    ConfigClass = RepairConfig

    @pipeBase.timeMethod
    def run(self, exposure, defects=None, keepCRs=None):
        """Repair exposure's instrumental problems

        @param[in,out] exposure Exposure to process
        @param defects Defect list
        @param keepCRs  Don't interpolate over the CR pixels (defer to pex_config if None)
        """
        assert exposure, "No exposure provided"
        psf = exposure.getPsf()
        assert psf, "No PSF provided"

        if defects is not None and self.config.doInterpolate:
            self.interpolate(exposure, defects)

        if self.config.doCosmicRay:
            self.cosmicRay(exposure, keepCRs=keepCRs)

        self.display('after', exposure=exposure)
        
    def interpolate(self, exposure, defects):
        """Interpolate over defects

        @param[in,out] exposure Exposure to process
        @param defects Defect list
        """
        assert exposure, "No exposure provided"
        assert defects is not None, "No defects provided"
        psf = exposure.getPsf()
        assert psf, "No psf provided"

        mi = exposure.getMaskedImage()
        fallbackValue = afwMath.makeStatistics(mi, afwMath.MEANCLIP).getValue()
        measAlg.interpolateOverDefects(mi, psf, defects, fallbackValue)
        self.log.log(self.log.INFO, "Interpolated over %d defects." % len(defects))

    def cosmicRay(self, exposure, keepCRs=None):
        """Mask cosmic rays

        @param[in,out] exposure Exposure to process
        @param keepCRs  Don't interpolate over the CR pixels (defer to pex_config if None)
        """
        import lsstDebug
        display = lsstDebug.Info(__name__).display
        displayCR = lsstDebug.Info(__name__).displayCR

        assert exposure, "No exposure provided"
        psf = exposure.getPsf()
        assert psf, "No psf provided"

        # Blow away old mask
        try:
            mask = exposure.getMaskedImage().getMask()
            crBit = mask.getMaskPlane("CR")
            mask.clearMaskPlane(crBit)
        except Exception:
            pass

        exposure0 = exposure            # initial value of exposure
        binSize = self.config.cosmicray.background.binSize
        nx, ny = exposure.getWidth()/binSize, exposure.getHeight()/binSize
        if nx*ny <= 1:
            bg = afwMath.makeStatistics(exposure.getMaskedImage(), afwMath.MEDIAN).getValue()
        else:
            exposure = exposure.Factory(exposure, True)
            bg, exposure = measAlg.estimateBackground(exposure, self.config.cosmicray.background,
                                                      subtract=True)
            bg = 0.0

        if keepCRs is None:
            keepCRs = self.config.cosmicray.keepCRs
        try:
            crs = measAlg.findCosmicRays(exposure.getMaskedImage(),
                                         psf, bg, pexConfig.makePolicy(self.config.cosmicray), keepCRs)
        except Exception, e:
            if display:
                ds9.mtv(exposure0, title="Failed CR")
            raise
            
        num = 0
        if crs is not None:
            mask = exposure0.getMaskedImage().getMask()
            crBit = mask.getPlaneBitMask("CR")
            afwDet.setMaskFromFootprintList(mask, crs, crBit)
            num = len(crs)

            if display and displayCR:
                import lsst.afw.display.utils as displayUtils

                ds9.incrDefaultFrame()
                ds9.mtv(exposure0, title="Post-CR")
                
                with ds9.Buffering():
                    for cr in crs:
                        displayUtils.drawBBox(cr.getBBox(), borderWidth=0.55)

        self.log.log(self.log.INFO, "Identified %s cosmic rays." % (num,))

