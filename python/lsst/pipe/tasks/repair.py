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
import lsst.pex.config as pexConfig
import lsst.afw.math as afwMath
import lsst.afw.detection as afwDet
import lsst.meas.algorithms as measAlg
import lsst.pipe.base as pipeBase

import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

class RepairConfig(pexConfig.Config):
    doInterpolate = pexConfig.Field(
        dtype = bool,
        doc = "Interpolate over defects? (ignored unless you provide a list of defects)",
        default = True,
    )
    doCosmicRay = pexConfig.Field(
        dtype = bool,
        doc = "Find and mask out cosmic rays?",
        default = False,
    )

class RepairTask(pipeBase.Task):
    """Conversion notes:
    
    Display code should be updated once we settle on a standard way of controlling what is displayed.
    """
    ConfigClass = RepairConfig

    def __init__(self, keepCRs=False, *args, **kwargs):
        # why is this not part of config (or even config for measAlg.findCosmicRays)?
        self._keepCRs = keepCRs
        pipeBase.Task.__init__(self, *args, **kwargs)
    
    @pipeBase.timeMethod
    def run(self, exposure, psf, defects=None):
        """Repair exposure's instrumental problems

        @param[in,out] exposure Exposure to process
        @param psf Point spread function
        @param defects Defect list
        """
        assert exposure, "No exposure provided"

        display = lsstDebug.Info(__name__).display
        if display:
            self.display('prerepair', exposure=exposure)

        if defects is not None and self.config.doInterpolate:
            self.interpolate(exposure, psf, defects)

        if self.config.doCosmicray:
            self.cosmicRay(exposure, psf)

        if display:
            self.display('repair', exposure=exposure)

    def interpolate(self, exposure, psf, defects):
        """Interpolate over defects

        @param[in,out] exposure Exposure to process
        @param psf PSF for interpolation
        @param defects Defect list
        """
        assert exposure, "No exposure provided"
        assert defects is not None, "No defects provided"
        assert psf, "No psf provided"

        mi = exposure.getMaskedImage()
        fallbackValue = afwMath.makeStatistics(mi, afwMath.MEANCLIP).getValue()
        measAlg.interpolateOverDefects(mi, psf, defects, fallbackValue)
        self.log.log(self.log.INFO, "Interpolated over %d defects." % len(defects))

    def cosmicRay(self, exposure, psf):
        """Mask cosmic rays

        @param[in,out] exposure Exposure to process
        @param psf PSF
        """
        import lsstDebug
        display = lsstDebug.Info(__name__).display
        displayCR = lsstDebug.Info(__name__).displayCR

        assert exposure, "No exposure provided"
        assert psf, "No psf provided"

        # Blow away old mask
        try:
            mask = exposure.getMaskedImage().getMask()
            crBit = mask.getMaskPlane("CR")
            mask.clearMaskPlane(crBit)
        except Exception:
            pass
        
        bg = afwMath.makeStatistics(mi, afwMath.MEDIAN).getValue()
        crs = measAlg.findCosmicRays(mi, psf, bg, self.config.doCosmicRay, self._keepCRs)
        num = 0
        if crs is not None:
            mask = mi.getMask()
            crBit = mask.getPlaneBitMask("CR")
            afwDet.setMaskFromFootprintList(mask, crs, crBit)
            num = len(crs)

            if display and displayCR:
                ds9.incrDefaultFrame()
                ds9.mtv(exposure, title="Post-CR")
                
                ds9.cmdBuffer.pushSize()

                for cr in crs:
                    displayUtils.drawBBox(cr.getBBox(), borderWidth=0.55)

                ds9.cmdBuffer.popSize()

        self.log.log(self.log.INFO, "Identified %s cosmic rays." % (num,))

