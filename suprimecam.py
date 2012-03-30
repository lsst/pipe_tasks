#!/usr/bin/env python

import lsst.afw.cameraGeom as afwCG
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.ip.isr as ipIsr
import lsst.pipe.tasks.processCcd as ptProcessCcd
import hsc.pipe.tasks.calibrate as hscCalibrate
import hsc.pipe.tasks.isr as hscIsr

class SuprimeCamIsrTask(hscIsr.HscIsrTask):
    def run(self, *args, **kwargs):
        result = super(SuprimeCamIsrTask, self).run(*args, **kwargs)
        self.guider(result.postIsrExposure)
        return result
    
    def guider(self, exposure):
        """Mask defects and trim guider shadow

        @param exposure Exposure to process
        @return Defect list
        """
        assert exposure, "No exposure provided"
        
        ccd = afwCG.cast_Ccd(exposure.getDetector()) # This is Suprime-Cam so we know the Detector is a Ccd
        ccdNum = ccd.getId().getSerial()
        if ccdNum not in [0, 1, 2, 6, 7]:
            # No need to mask
            return

        md = exposure.getMetadata()
        if not md.exists("S_AG-X"):
            self.log.log(self.log.WARN, "No autoguider position in exposure metadata.")
            return

        xGuider = md.get("S_AG-X")
        if ccdNum in [1, 2, 7]:
            maskLimit = int(60.0 * xGuider - 2300.0) # From SDFRED
        elif ccdNum in [0, 6]:
            maskLimit = int(60.0 * xGuider - 2000.0) # From SDFRED

        
        mi = exposure.getMaskedImage()
        height = mi.getHeight()
        if height < maskLimit:
            # Nothing to mask!
            return

        if False:
            # XXX This mask plane isn't respected by background subtraction or source detection or measurement
            self.log.log(self.log.INFO, "Masking autoguider shadow at y > %d" % maskLimit)
            mask = mi.getMask()
            bbox = afwGeom.Box2I(afwGeom.Point2I(0, maskLimit - 1),
                                 afwGeom.Point2I(mask.getWidth() - 1, height - 1))
            badMask = mask.Factory(mask, bbox, afwImage.LOCAL)
            
            mask.addMaskPlane("GUIDER")
            badBitmask = mask.getPlaneBitMask("GUIDER")
            
            badMask |= badBitmask
        else:
            # XXX Temporary solution until a mask plane is respected by downstream processes
            self.log.log(self.log.INFO, "Removing pixels affected by autoguider shadow at y > %d" % maskLimit)
            bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(mi.getWidth(), maskLimit))
            good = mi.Factory(mi, bbox, afwImage.LOCAL)
            exposure.setMaskedImage(good)
