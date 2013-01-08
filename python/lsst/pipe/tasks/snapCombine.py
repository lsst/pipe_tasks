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
import lsst.daf.base as dafBase
import lsst.afw.detection as afwDet
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pipe.base as pipeBase
from lsst.coadd.utils import Coadd, addToCoadd, setCoaddEdgeBits
from lsst.ip.diffim import SnapPsfMatchTask
from lsst.meas.algorithms import SourceDetectionTask, SourceMeasurementTask

from .repair import RepairTask
from .calibrate import InitialPsfConfig

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
    doSimpleAverage = pexConfig.Field(
        dtype = bool,
        doc = "The combined snap is a straight average of the data",
        default = True,
    )
    doPsfMatch = pexConfig.Field(
        dtype = bool,
        doc = "Perform difference imaging before combining",
        default = True,
    )
    doMeasurement = pexConfig.Field(
        dtype = bool,
        doc = "Measure difference sources",
        default = True
    )

    repair      = pexConfig.ConfigurableField(target = RepairTask, doc = "")
    diffim      = pexConfig.ConfigurableField(target = SnapPsfMatchTask, doc = "")
    coadd       = pexConfig.ConfigField(dtype = Coadd.ConfigClass, doc="")
    detection   = pexConfig.ConfigurableField(target = SourceDetectionTask, doc = "")
    initialPsf  = pexConfig.ConfigField(dtype = InitialPsfConfig, doc = "")
    measurement = pexConfig.ConfigurableField(target = SourceMeasurementTask, doc = "")

    def setDefaults(self):
        self.detection.thresholdPolarity = "both"

    def validate(self):
        if self.detection.thresholdPolarity != "both":
            raise ValueError("detection.thresholdPolarity must be 'both' for SnapCombineTask")

class SnapCombineTask(pipeBase.Task):
    ConfigClass = SnapCombineConfig

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("repair")
        self.makeSubtask("diffim")
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask("detection", schema=self.schema)
        if self.config.doMeasurement:
            self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)
 
    @pipeBase.timeMethod
    def run(self, snap0, snap1, defects=None):
        """Combine two snaps
        
        @param[in] snap0: snapshot exposure 0
        @param[in] snap1: snapshot exposure 1
        @defects[in] defect list (for repair task)
        @return a pipe_base Struct with fields:
        - exposure: snap-combined exposure
        - sources: detected sources, or None if detection not performed
        """
        if self.config.doSimpleAverage:
            self.log.info("snapCombine by straight average")
            coaddExp  = afwImage.ExposureF(snap0, True)
            coaddMi   = coaddExp.getMaskedImage()
            coaddMi  += snap1.getMaskedImage()
            sources = None
            return pipeBase.Struct(
                exposure = coaddExp,
                sources = None,
            )

        if self.config.doRepair:
            self.log.info("snapCombine repair")
            psf = self.makeInitialPsf(snap0, fwhmPix=self.config.repairPsfFwhm)
            snap0.setPsf(psf)
            snap1.setPsf(psf)
            self.repair.run(snap0, defects=defects, keepCRs=False)
            self.repair.run(snap1, defects=defects, keepCRs=False)
            self.display('repair0', exposure=snap0)
            self.display('repair1', exposure=snap1)

        if self.config.doPsfMatch:
            self.log.info("snapCombine psfMatch")
            diffRet  = self.diffim.run(snap0, snap1, "subtractExposures")
            diffExp  = diffRet.subtractedImage

            # Measure centroid and width of kernel; dependent on ticket #1980
            # Useful diagnostic for the degree of astrometric shift between snaps.
            diffKern = diffRet.psfMatchingKernel
            width, height = diffKern.getDimensions()
            # TBD...
            #psfAttr = measAlg.PsfAttributes(diffKern, width//2, height//2)

        else:
            diffExp  = afwImage.ExposureF(snap0, True)
            diffMi   = diffExp.getMaskedImage()
            diffMi  -= snap1.getMaskedImage()

        psf = self.makeInitialPsf(snap0)
        diffExp.setPsf(psf)
        table = afwTable.SourceTable.make(self.schema)
        table.setMetadata(self.algMetadata)
        detRet = self.detection.makeSourceCatalog(table, diffExp)
        sources = detRet.sources
        fpSets = detRet.fpSets
        if self.config.doMeasurement:
            self.measurement.measure(diffExp, sources)
        
        mask0 = snap0.getMaskedImage().getMask()
        mask1 = snap1.getMaskedImage().getMask()
        fpSets.positive.setMask(mask0, "DETECTED")
        fpSets.negative.setMask(mask1, "DETECTED")
        
        maskD = diffExp.getMaskedImage().getMask()
        fpSets.positive.setMask(maskD, "DETECTED")
        fpSets.negative.setMask(maskD, "DETECTED_NEGATIVE")

        coaddMi   = snap0.getMaskedImage().Factory(snap0.getBBox(afwImage.PARENT))
        weightMap = coaddMi.getImage().Factory(coaddMi.getBBox(afwImage.PARENT))
        weight    = 1.0
        
        badMaskPlanes  = []
        for bmp in self.config.coadd.badMaskPlanes:
            badMaskPlanes.append(bmp)
        badMaskPlanes.append("DETECTED")
        badPixelMask   = afwImage.MaskU.getPlaneBitMask(badMaskPlanes)
        self.log.info("snapCombine coaddition")
        addToCoadd(coaddMi, weightMap, snap0.getMaskedImage(), badPixelMask, weight)
        addToCoadd(coaddMi, weightMap, snap1.getMaskedImage(), badPixelMask, weight)
        coaddMi /= weightMap
        coaddMi *= 2.0
        setCoaddEdgeBits(coaddMi.getMask(), weightMap)

        # Need copy of Filter, Detector, Wcs, Calib in new Exposure
        coaddExp = afwImage.ExposureF(snap0, True)
        coaddExp.setMaskedImage(coaddMi)

        return pipeBase.Struct(
            exposure = coaddExp,
            sources = sources,
        )

    def makeInitialPsf(self, exposure, fwhmPix=None):
        """Initialise the detection procedure by setting the PSF and WCS

        @param exposure Exposure to process
        @return PSF, WCS
        """
        assert exposure, "No exposure provided"
        wcs = exposure.getWcs()
        assert wcs, "No wcs in exposure"
        
        if fwhmPix is None:
            fwhmPix = self.config.initialPsf.fwhm / wcs.pixelScale().asArcseconds()
            
        size = self.config.initialPsf.size
        model = self.config.initialPsf.model
        self.log.info("installInitialPsf fwhm=%s pixels; size=%s pixels" % (fwhmPix, size))
        psf = afwDet.createPsf(model, size, size, fwhmPix/(2.0*num.sqrt(2*num.log(2.0))))
        return psf
