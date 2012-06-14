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
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.afw.cameraGeom as afwCameraGeom
from lsst.meas.algorithms import SourceDetectionTask, SourceMeasurementTask
from lsst.pipe.tasks.calibrate import CalibrateTask

class ProcessCcdSdssConfig(pexConfig.Config):
    """Config for ProcessCcdSdss"""
    removePedestal = pexConfig.Field(dtype=bool, default=True, doc = "Remove SDSS pedestal from fpC file")
    pedestalVal = pexConfig.Field(dtype=int, default=1000, doc = "Number of counts in the SDSS pedestal")

    doCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Perform calibration?")
    doDetection = pexConfig.Field(dtype=bool, default=True, doc = "Detect sources?")
    doMeasurement = pexConfig.Field(dtype=bool, default=True, doc = "Measure sources?")
    doWriteCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Write calibration results?")
    doWriteSources = pexConfig.Field(dtype=bool, default=True, doc = "Write sources?")
    calibrate = pexConfig.ConfigurableField(
        target = CalibrateTask,
        doc = "Calibration (inc. high-threshold detection and measurement)",
    )
    detection = pexConfig.ConfigurableField(
        target = SourceDetectionTask,
        doc = "Low-threshold detection for final measurement",
    )
    measurement = pexConfig.ConfigurableField(
        target = SourceMeasurementTask,
        doc = "Final source measurement on low-threshold detections",
    )

    def validate(self):
        pexConfig.Config.validate(self)
        if self.doMeasurement and not self.doDetection:
            raise ValueError("Cannot run source measurement without source detection.")

    def setDefaults(self):
        # OPTIMIZE FOR SDSS
        self.calibrate.repair.doInterpolate = False
        self.calibrate.repair.doCosmicRay = False

        self.calibrate.background.binSize = 512 
        self.calibrate.detection.background.binSize = 512
        self.detection.background.binSize = 512
        
class ProcessCcdSdssTask(pipeBase.CmdLineTask):
    """Process a CCD for SDSS
    """
    ConfigClass = ProcessCcdSdssConfig
    _DefaultName = "processCcd"

    def __init__(self, **kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.makeSubtask("calibrate")
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        if self.config.doDetection:
            self.makeSubtask("detection", schema=self.schema)
        if self.config.doMeasurement:
            self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)

    @classmethod
    def _makeArgumentParser(cls):
        return pipeBase.ArgumentParser(name=cls._DefaultName, datasetType="fpC")        

    @pipeBase.timeMethod
    def makeExp(self, frameRef):
        image = frameRef.get("fpC").convertF()
        if self.config.removePedestal:
            image -= self.config.pedestalVal
        mask  = frameRef.get("fpM")
        wcs   = frameRef.get("asTrans")
        calib, gain = frameRef.get("tsField")
        var   = afwImage.ImageF(image, True)
        var  /= gain

        mi    = afwImage.MaskedImageF(image, mask, var)
        exp   = afwImage.ExposureF(mi, wcs)
        exp.setCalib(calib)
        det = afwCameraGeom.Detector(afwCameraGeom.Id("%s%d" %
                                                      (frameRef.dataId["filter"], frameRef.dataId["camcol"])))
        exp.setDetector(det)
        exp.setFilter(afwImage.Filter(frameRef.dataId['filter']))

        return exp


    @pipeBase.timeMethod
    def run(self, frameRef):
        """Process a CCD: including source detection, photometry and WCS determination
        
        @param frameRef: frame-level butler data reference
        @return pipe_base Struct containing these fields:
        - exposure: calibrated exposure (calexp): as computed if config.doCalibrate,
            else as upersisted if config.doDetection, else None
        - calib: object returned by calibration process if config.doCalibrate, else None
        - apCorr: aperture correction: as computed config.doCalibrate, else as unpersisted
            if config.doMeasure, else None
        - sources: detected source if config.doPhotometry, else None
        """
        self.log.log(self.log.INFO, "Processing %s" % (frameRef.dataId))

        # We make one IdFactory that will be used by both icSrc and src datasets;
        # I don't know if this is the way we ultimately want to do things, but at least
        # this ensures the source IDs are fully unique.
        expBits = frameRef.get("ccdExposureId_bits")
        expId = long(frameRef.get("ccdExposureId"))
        idFactory = afwTable.IdFactory.makeSource(expId, 64 - expBits)

        # initialize outputs
        calExposure = None
        calib = None
        apCorr = None
        sources = None

        if self.config.doCalibrate:
            self.log.log(self.log.INFO, "Performing Calibrate on fpC %s" % (frameRef.dataId))
            exp = self.makeExp(frameRef)
            calib = self.calibrate.run(exp, idFactory=idFactory)
            calExposure = calib.exposure
            apCorr = calib.apCorr
            if self.config.doWriteCalibrate:
                frameRef.put(calExposure, 'calexp')
                frameRef.put(calib.sources, 'icSrc')
                if calib.psf is not None:
                    frameRef.put(calib.psf, 'psf')
                if calib.apCorr is not None:
                    frameRef.put(calib.apCorr, 'apCorr')
                if calib.matches is not None:
                    normalizedMatches = afwTable.packMatches(calib.matches)
                    normalizedMatches.table.setMetadata(calib.matchMeta)
                    frameRef.put(normalizedMatches, 'icMatch')

        if self.config.doDetection:
            if calExposure is None:
                calExposure = self.makeExp(frameRef)
            if calib is None or calib.psf is None:
                psf = frameRef.get('psField')
                calExposure.setPsf(psf)
                frameRef.put(psf, 'psf')
            table = afwTable.SourceTable.make(self.schema, idFactory)
            table.setMetadata(self.algMetadata)
            sources = self.detection.makeSourceCatalog(table, calExposure).sources

        if self.config.doMeasurement:
            if apCorr is None:
                apCorr = frameRef.get("apCorr")
            self.measurement.run(calExposure, sources, apCorr)

        if sources is not None and self.config.doWriteSources:
            frameRef.put(sources, 'src')

        return pipeBase.Struct(
            exposure = calExposure,
            calib = calib,
            apCorr = apCorr,
            sources = sources,
        )
