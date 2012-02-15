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
import lsst.afw.detection as afwDet

from lsst.ip.isr import IsrTask
from lsst.pipe.tasks.calibrate import CalibrateTask
from lsst.pipe.tasks.photometry import PhotometryTask

class ProcessCcdConfig(pexConfig.Config):
    """Config for ProcessCcd"""
    doIsr = pexConfig.Field(dtype=bool, default=True, doc = "Perform ISR?")
    doCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Perform calibration?")
    doPhotometry = pexConfig.Field(dtype=bool, default=True, doc = "Perform photometry?")
    doWriteIsr = pexConfig.Field(dtype=bool, default=True, doc = "Write ISR results?")
    doWriteCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Write calibration results?")
    doWritePhotometry = pexConfig.Field(dtype=bool, default=True, doc = "Write photometry results?")
    isr = pexConfig.ConfigField(dtype=IsrTask.ConfigClass, doc="Instrumental Signature Removal")
    calibrate = pexConfig.ConfigField(dtype=CalibrateTask.ConfigClass, doc="Calibration")
    photometry = pexConfig.ConfigField(dtype=PhotometryTask.ConfigClass, doc="Photometry")

    def __init__(self, *args, **kwargs):
        pexConfig.Config.__init__(self, *args, **kwargs)
        self.doWriteIsr = False
        self.isr.methodList=["doConversionForIsr", "doSaturationDetection",
                             "doOverscanCorrection", "doVariance", "doFlatCorrection"]
        self.isr.doWrite = False
        
        self.calibrate.repair.doCosmicRay = True
        self.calibrate.repair.cosmicray.nCrPixelMax = 100000
        self.calibrate.background.binSize = 1024
        
        # PSF determination
        self.calibrate.measurePsf.starSelector.name = "secondMoment"
        self.calibrate.measurePsf.psfDeterminer.name = "pca"
        self.calibrate.measurePsf.starSelector["secondMoment"].clumpNSigma = 2.0
        self.calibrate.measurePsf.psfDeterminer["pca"].nEigenComponents = 4
        self.calibrate.measurePsf.psfDeterminer["pca"].kernelSize = 7.0
        self.calibrate.measurePsf.psfDeterminer["pca"].spatialOrder = 2
        self.calibrate.measurePsf.psfDeterminer["pca"].kernelSizeMin = 25
        
        # Final photometry
        self.photometry.detect.thresholdValue = 5.0
        self.photometry.detect.includeThresholdMultiplier = 1.0
        self.photometry.measure.source.astrom = "NAIVE"
        self.photometry.measure.source.apFlux = "NAIVE"
        self.photometry.measure.source.modelFlux = "GAUSSIAN"
        self.photometry.measure.source.psfFlux = "PSF"
        self.photometry.measure.source.shape = "SDSS"
        self.photometry.measure.astrometry.names = ["GAUSSIAN", "NAIVE", "SDSS"]
        self.photometry.measure.shape.names = ["SDSS"]
        self.photometry.measure.photometry.names = ["NAIVE", "GAUSSIAN", "PSF", "SINC"]
        self.photometry.measure.photometry["NAIVE"].radius = 7.0
        self.photometry.measure.photometry["GAUSSIAN"].shiftmax = 10
        self.photometry.measure.photometry["SINC"].radius = 7.0
        
        # Initial photometry
        self.calibrate.photometry.detect.thresholdValue = 5.0
        self.calibrate.photometry.detect.includeThresholdMultiplier = 10.0
        self.calibrate.photometry.measure = self.photometry.measure
        
        # Aperture correction
        self.calibrate.apCorr.alg1.name = "PSF"
        self.calibrate.apCorr.alg2.name = "SINC"
        self.calibrate.apCorr.alg1[self.calibrate.apCorr.alg1.name] = self.photometry.measure.photometry[self.calibrate.apCorr.alg1.name]
        self.calibrate.apCorr.alg2[self.calibrate.apCorr.alg2.name] = self.photometry.measure.photometry[self.calibrate.apCorr.alg2.name]
        
        # Astrometry
        self.calibrate.astrometry.distortion.name = "radial"
        self.calibrate.astrometry.distortion["radial"].coefficients = [0.0, 1.0, 7.16417e-08, 3.03146e-10, 5.69338e-14, -6.61572e-18]
        self.calibrate.astrometry.distortion["radial"].observedToCorrected = True


class ProcessCcdTask(pipeBase.Task):
    """Process a CCD"""
    ConfigClass = ProcessCcdConfig

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("isr", IsrTask)
        self.makeSubtask("calibrate", CalibrateTask)
        self.makeSubtask("photometry", PhotometryTask)

    @pipeBase.timeMethod
    def run(self, sensorRef):
        self.log.log(self.log.INFO, "Processing %s" % (sensorRef.dataId))
        if self.config.doIsr:
            butler = sensorRef.butlerSubset.butler
            calibSet = self.isr.makeCalibDict(butler, sensorRef.dataId)
            rawExposure = sensorRef.get("raw")
            isrRes = self.isr.run(rawExposure, calibSet)
            self.display("isr", exposure=isrRes.postIsrExposure, pause=True)
            ccdExposure = self.isr.doCcdAssembly([isrRes.postIsrExposure])
            self.display("ccdAssembly", exposure=ccdExposure)
            if self.config.doWriteIsr:
                sensorRef.put(ccdExposure, 'postISRCCD')
        else:
            ccdExposure = None

        if self.config.doCalibrate:
            if ccdExposure is None:
                ccdExposure = sensorRef.get('postISRCCD')
            calib = self.calibrate.run(ccdExposure)
            ccdExposure = calib.exposure
            if self.config.doWriteCalibrate:
                sensorRef.put(ccdExposure, 'calexp')
                sensorRef.put(afwDet.PersistableSourceVector(calib.sources), 'icSrc')
                if calib.psf is not None:
                    sensorRef.put(calib.psf, 'psf')
                if calib.apCorr is not None:
                    #sensorRef.put(calib.apCorr, 'apcorr')
                    pass
                if calib.matches is not None:
                    sensorRef.put(afwDet.PersistableSourceMatchVector(calib.matches, calib.matchMeta),
                               'icMatch')
        else:
            calib = None

        if self.config.doPhotometry:
            if ccdExposure is None:
                ccdExposure = sensorRef.get('calexp')
            if calib is None:
                psf = sensorRef.get('psf')
                apCorr = None # sensorRef.get('apcorr')
            else:
                psf = calib.psf
                apCorr = calib.apCorr
            phot = self.photometry.run(ccdExposure, psf, apcorr=apCorr)
            if self.config.doWritePhotometry:
                sensorRef.put(afwDet.PersistableSourceVector(phot.sources), 'src')
        else:
            phot = None

        return pipeBase.Struct(
            ccdExposure = isrRes.postIsrExposure if self.config.doIsr else None,
            exposure = ccdExposure,
            psf = psf,
            apCorr = apCorr,
            sources = phot.sources if phot else None,
            matches = calib.matches if calib else None,
            matchMeta = calib.matchMeta if calib else None,
        )
