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
import numpy as num
import lsst.pex.config as pexConfig
import lsst.afw.detection as afwDet
import lsst.afw.image as afwImage
import lsst.meas.utils.sourceDetection as muDetection
import lsst.pipe.base as pipeBase
from lsst.coadd.utils import Coadd, addToCoadd, setCoaddEdgeBits
from lsst.ip.diffim import SnapPsfMatchTask

from .photometry import PhotometryDiffTask
from .calibrate import CalibrateTask
from .repair import RepairTask

class SnapCombineConfig(pexConfig.Config):
    doRepair = pexConfig.Field(
        dtype = bool,
        doc = "Repair images (CR reject and interpolate) before combining",
        default = True,
    )
    repairPsfFwhm = pexConfig.Field(
        dtype = float,
        doc = "Psf FWHM (pixels) used to detect CRs", 
        default = 2.5 # pixels
    )

    doDiffim = pexConfig.Field(
        dtype = bool,
        doc = "Perform difference imaging before combining",
        default = True,
    )

    repair     = pexConfig.ConfigField(dtype = RepairTask.ConfigClass, doc = "")
    diffim     = pexConfig.ConfigField(dtype = SnapPsfMatchTask.ConfigClass, doc = "")
    coadd      = pexConfig.ConfigField(dtype = Coadd.ConfigClass, doc="")
    calibrate  = pexConfig.ConfigField(dtype = CalibrateTask.ConfigClass,  doc = "")
    photometry = pexConfig.ConfigField(dtype = PhotometryDiffTask.ConfigClass,  doc = "")

class SnapCombineTask(pipeBase.Task):
    ConfigClass = SnapCombineConfig

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("repair", RepairTask)
        self.makeSubtask("diffim", SnapPsfMatchTask)
        self.makeSubtask("photometry", PhotometryDiffTask)
 
    @pipeBase.timeMethod
    def run(self, snap0, snap1, defects=None):

        if self.config.doRepair:
            fakePsf, wcs = self.makeFakePsf(snap0, fwhmPix=self.config.repairPsfFwhm)

            self.repair.run(snap0, fakePsf, defects=defects, keepCRs=False)
            self.repair.run(snap1, fakePsf, defects=defects, keepCRs=False)
            self.display('repair0', exposure=snap0)
            self.display('repair1', exposure=snap1)
            
        if self.config.doDiffim:
            diffRet = self.diffim.run(snap0, snap1, "subtractExposures")
            diffExp = diffRet.subtractedImage
            diffExp.writeFits("/tmp/diff.fits")

            fakePsf, wcs = self.makeFakePsf(snap0)
            photRet = self.photometry.run(diffExp, fakePsf, wcs=wcs)
            sources = photRet.sources
            footprints = photRet.footprintSets

            coaddMi   = snap0.getMaskedImage().Factory(snap0.getBBox(afwImage.PARENT))
            weightMap = coaddMi.getImage().Factory(coaddMi.getBBox(afwImage.PARENT))
            weight    = 1.0

            badMaskPlanes  = []
            for bmp in self.config.coadd.badMaskPlanes:
                badMaskPlanes.append(bmp)
            badMaskPlanes.append("BAD")
            badMaskPlanes.append("CR")
            badPixelMask   = afwImage.MaskU.getPlaneBitMask(badMaskPlanes)
            addToCoadd(coaddMi, weightMap, snap0.getMaskedImage(), badPixelMask, weight)
            addToCoadd(coaddMi, weightMap, snap1.getMaskedImage(), badPixelMask, weight)
            coaddMi /= weightMap
            setCoaddEdgeBits(coaddMi.getMask(), weightMap)

            # Need copy of Filter, Detector, Wcs, Calib in new Exposure
            coaddExp = afwImage.ExposureF(snap0, True)
            coaddExp.setMaskedImage(coaddMi)
        
        return pipeBase.Struct(visitExposure = coaddExp,
                               metadata = self.metadata)

    def makeFakePsf(self, exposure, fwhmPix=None):
        """Initialise the detection procedure by setting the PSF and WCS

        @param exposure Exposure to process
        @return PSF, WCS
        """
        assert exposure, "No exposure provided"
        wcs = exposure.getWcs()
        assert wcs, "No wcs in exposure"
        
        if fwhmPix is None:
            fwhmPix = self.config.calibrate.fwhm / wcs.pixelScale().asArcseconds()
            
        size = self.config.calibrate.size
        model = self.config.calibrate.model
        self.log.log(self.log.INFO, "makeFakePsf fwhm=%s pixels; size=%s pixels" % (fwhmPix, size))
        psf = afwDet.createPsf(model, size, size, fwhmPix/(2.0*num.sqrt(2*num.log(2.0))))
        return psf, wcs
