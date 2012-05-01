#!/usr/bin/env python

import gc

from lsst.pex.config import Config, Field, ConfigField
from lsst.pipe.base import Task, Struct

import lsst.daf.base as dafBase
import lsst.daf.persistence as dafPersist
import lsst.afw.detection as afwDet
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.ip.isr as ipIsr
import lsst.meas.algorithms as measAlg
import lsst.pipe.tasks.calibrate as ptCal


class ForcedPhotConfig(Config):
    calibrate = ConfigField(dtype=ptCal.CalibrateConfig, doc="Configuration for calibration of stack")
    detection = ConfigField(dtype=measAlg.SourceDetectionConfig, doc="Configuration for detection on stack")
    measurement = ConfigField(dtype=measAlg.SourceMeasurementConfig,
                              doc="Configuration for measurement on CCD")

    def setDefaults(self):
        self.calibrate.doAstrometry = False
        self.calibrate.doPhotoCal = False


class ForcedPhotTask(Task):
    ConfigClass = ForcedPhotConfig
    _DefaultName = "forcedPhot"

    def __init__(self, *args, **kwargs):
        super(ForcedPhotTask, self).__init__(*args, **kwargs)
        self.makeSubtask("calibrate", ptCal.CalibrateTask)
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask("detection", measAlg.SourceDetectionTask, schema=self.schema)
        self.makeSubtask("measurement", measAlg.SourceMeasurementTask,
                         schema=self.schema, algMetadata=self.algMetadata)

    def run(self, butler, stackId, expIdList):

        stackMeas = self.measureStack(butler, stackId)
        for expId in expIdList:
            self.measureExp(butler, expId, stackMeas)


    def measureStack(self, butler, dataId):
        try:
            sources = foilReadProxy(butler.get("stacksources", dataId))
            wcs = afwImage.makeWcs(butler.get("stack_md", dataId))
        except Exception, e:
            self.log.log(self.log.INFO, "Stack products not available (%s); attempting to create" % e)
            exposure = butler.get("stack", dataId)
            self.interpolateNans(exposure, afwDet.createPsf("DoubleGaussian", 15, 15, 1.0))       
            calib = self.calibrate.run(exposure)
            exposure = calib.exposure
            wcs = exposure.getWcs()

            det = self.detect(exposure)
            sources = det.sources
            self.measurement.run(stack, sources, apCorr=calib.apCorr)
            butler.put(sources, "stacksources", dataId)
            butler.put(stack.getPsf(), "stackpsf", dataId)

        return Struct(sources=sources, wcs=wcs)


    def interpolateNans(self, exposure, psf):
        exposure.getMaskedImage().getMask().addMaskPlane("UNMASKEDNAN")
        nanMasker = ipIsr.UnmaskedNanCounterF()
        nanMasker.apply(exposure.getMaskedImage())
        nans = ipIsr.Isr().getDefectListFromMask(exposure.getMaskedImage(), maskName='UNMASKEDNAN')
        self.log.log(self.log.INFO, "Interpolating over %d NANs" % len(nans))
        measAlg.interpolateOverDefects(exposure.getMaskedImage(), psf, nans, 0.0)
        

    def detect(self, exposure):
        table = afwTable.SourceTable.make(self.schema)
        table.setMetadata(self.algMetadata)
        return self.detection.makeSourceCatalog(table, exposure)

    def measureExp(self, butler, dataId, stackMeas):
        exposure = butler.get("calexp", dataId)
        psf = butler.get("psf", dataId)
        exposure.setPsf(psf)
        wcs = exposure.getWcs()
        apCorr = butler.get("apCorr", dataId)
        box = afwGeom.Box2D(exposure.getBBox())

        # Only bother with sources that are definitely on the image
        stackSubset = afwTable.SourceCatalog(stackMeas.sources.getTable())
        sources = afwTable.SourceCatalog(stackMeas.sources.getSchema())
        table = sources.getTable()
        for s in stackMeas.sources:
            coord = s.getCoord()
            if box.contains(wcs.skyToPixel(coord)):
                new = table.makeRecord()
                new.setCoord(coord)
                sources.append(new)
                stackSubset.append(s)
        self.log.log(self.log.INFO, "Subset of %d/%d sources for %s" % 
                     (len(sources), len(stackMeas.sources), dataId))

        self.measurement.run(exposure, sources, apCorr=apCorr,
                             references=stackSubset, refWcs=stackMeas.wcs)

        butler.put(sources, "src", dataId)


def foilReadProxy(obj):
    if isinstance(obj, dafPersist.ReadProxy):
        obj = obj.__subject__
    return obj
