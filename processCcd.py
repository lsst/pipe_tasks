#!/usr/bin/env python

import lsst.pex.config as pexConfig
import lsst.afw.detection as afwDet
import lsst.afw.table as afwTable
import lsst.daf.base as dafBase
import lsst.meas.algorithms as measAlg
import lsst.pipe.base as pipeBase
import lsst.pipe.tasks.processCcd as ptProcessCcd
import hsc.pipe.tasks.astrometry as hscAstrom
import hsc.pipe.tasks.suprimecam as hscSuprimeCam
import hsc.pipe.tasks.calibrate as hscCalibrate
import hsc.pipe.tasks.isr as hscIsr
import hsc.pipe.tasks.hscDc2 as hscDc2


class SubaruProcessCcdConfig(ptProcessCcd.ProcessCcdConfig):
    calibrate = pexConfig.ConfigField(dtype=hscCalibrate.HscCalibrateConfig, doc="Calibration")
    isr = pexConfig.ConfigField(dtype=hscIsr.HscIsrConfig, doc="Instrument signature removal")

class SubaruProcessCcdTask(ptProcessCcd.ProcessCcdTask):
    """Subaru version of ProcessCcdTask, with method to write outputs
    after producing a new multi-frame WCS.
    """
    ConfigClass = SubaruProcessCcdConfig

    def __init__(self, *args, **kwargs):
        super(SubaruProcessCcdTask, self).__init__(*args, **kwargs)
        self.makeSubtask("isr", hscIsr.HscIsrTask)
        self.makeSubtask("calibrate", hscCalibrate.HscCalibrateTask)
        self.makeSubtask("photometry", PhotometryTask)

    # The 'run' method is copied wholesale from lsst.pipe.tasks.processCcd.ProcessCcdTask.run, with minor
    # modifications to change when the CCD assembly is performed.
    @pipeBase.timeMethod
    def run(self, sensorRef):
        self.log.log(self.log.INFO, "Processing %s" % (sensorRef.dataId))
        if self.config.doIsr:
            butler = sensorRef.butlerSubset.butler
            calibSet = self.isr.makeCalibDict(butler, sensorRef.dataId)
            rawExposure = sensorRef.get("raw")
            isrRes = self.isr.run(rawExposure, calibSet)
            self.display("isr", exposure=isrRes.postIsrExposure, pause=True)
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


    def write(self, butler, dataId, struct, wcs=None):
        if wcs is None:
            wcs = struct.exposure.getWcs()
            self.log.log(self.log.WARN, "WARNING: No new WCS provided")

        # Apply WCS to sources
        # No longer handling matchSources explicitly - these should all be in calib.sources,
        # or there's a bug in the calibrate task.
        struct.exposure.setWcs(wcs)
        for sources in (struct.sources, struct.calib.sources):
            if sources is None:
                continue
            for s in sources:
                s.updateCoord(wcs)
                
        normalizedMatches = afwTable.packMatches(struct.calib.matches)
        normalizedMatches.table.setMetadata(struct.calib.matchMeta)

        butler.put(struct.exposure, 'calexp', dataId)
        butler.put(struct.sources, 'src', dataId)
        butler.put(normalizedMatches, 'icMatch', dataId)
        butler.put(struct.calib.psf, 'psf', dataId)
        butler.put(struct.calib.apCorr, 'apCorr', dataId)
        butler.put(struct.calib.sources, 'icSrc', dataId)

class SuprimeCamProcessCcdTask(SubaruProcessCcdTask):
    
    def __init__(self, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)
        self.makeSubtask("isr", hscSuprimeCam.SuprimeCamIsrTask)
        self.makeSubtask("calibrate", hscCalibrate.HscCalibrateTask)
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        if self.config.doDetection:
            self.makeSubtask("detection", measAlg.SourceDetectionTask, schema=self.schema)
        if self.config.doMeasurement:
            self.makeSubtask("measurement", measAlg.SourceMeasurementTask,
                             schema=self.schema, algMetadata=self.algMetadata)

class HscProcessCcdTask(SubaruProcessCcdTask):

    def __init__(self, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)
        self.makeSubtask("isr", HscIsrTask)
        self.makeSubtask("calibrate", hscDc2.HscDc2CalibrateTask)
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        if self.config.doDetection:
            self.makeSubtask("detection", measAlg.SourceDetectionTask, schema=self.schema)
        if self.config.doMeasurement:
            self.makeSubtask("measurement", measAlg.SourceMeasurementTask,
                             schema=self.schema, algMetadata=self.algMetadata)
