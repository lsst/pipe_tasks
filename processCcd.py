#!/usr/bin/env python

import lsst.pex.config as pexConfig
import lsst.afw.detection as afwDet
from lsst.ip.isr import IsrTask
import lsst.pipe.base as pipeBase
import lsst.pipe.tasks.processCcd as ptProcessCcd
from lsst.pipe.tasks.photometry import PhotometryTask
import hsc.pipe.tasks.astrometry as hscAstrom
import hsc.pipe.tasks.suprimecam as hscSuprimeCam
import hsc.pipe.tasks.calibrate as hscCalibrate
import hsc.pipe.tasks.hscDc2 as hscDc2

class HscProcessCcdConfig(ptProcessCcd.ProcessCcdConfig):
    calibrate = pexConfig.ConfigField(dtype=hscCalibrate.HscCalibrateConfig, doc="Calibration")

class HscProcessCcdTask(ptProcessCcd.ProcessCcdTask):
    """HSC version of ProcessCcdTask, with method to write outputs
    after producing a new WCS.
    """
    ConfigClass = HscProcessCcdConfig
    def write(self, butler, dataId, struct, wcs=None):
        exposure = struct.exposure
        psf = struct.psf
        apCorr = struct.apCorr
        brightSources = struct.brightSources if hasattr(struct, 'brightSources') else None
        faintSources = struct.sources
        matches = struct.matches
        matchMeta = struct.matchMeta

        if brightSources is None:
            brightSources = afwDet.SourceSet()
            for match in matches:
                brightSources.push_back(match.second)

        if wcs is None:
            wcs = exposure.getWcs()
            self.log.log(self.log.WARN, "WARNING: No new WCS provided")

        # Apply WCS to sources
        # In the matchlist, only convert the matches.second source, which is our measured source.
        exposure.setWcs(wcs)
        matchSources = [m.second for m in matches] if matches is not None else None
        for sources in (faintSources, brightSources, matchSources):
            if sources is None:
                continue
            for s in sources:
                s.setRaDec(wcs.pixelToSky(s.getXAstrom(), s.getYAstrom()))

        butler.put(exposure, 'calexp', dataId)
        butler.put(afwDet.PersistableSourceVector(afwDet.SourceSet(faintSources)), 'src', dataId)
        butler.put(afwDet.PersistableSourceMatchVector(matches, matchMeta), 'icMatch', dataId)
        butler.put(psf, 'psf', dataId)
        if brightSources is not None:
            butler.put(afwDet.PersistableSourceVector(afwDet.SourceSet(brightSources)), 'icSrc', dataId)

class SuprimeCamProcessCcdTask(HscProcessCcdTask):
    def __init__(self, config=HscProcessCcdConfig(), *args, **kwargs):
        pipeBase.Task.__init__(self, config=config, *args, **kwargs)
        self.makeSubtask("isr", hscSuprimeCam.SuprimeCamIsrTask, config=config.isr)
        self.makeSubtask("calibrate", hscCalibrate.HscCalibrateTask, config=config.calibrate)
        self.makeSubtask("photometry", PhotometryTask, config=config.photometry)


class HscDc2ProcessCcdConfig(HscProcessCcdConfig):
    def __init__(self, *args, **kwargs):
        super(HscDc2ProcessCcdConfig, self).__init__(*args, **kwargs)

class HscDc2ProcessCcdTask(HscProcessCcdTask):
    ConfigClass = HscProcessCcdConfig
    def __init__(self, config=HscProcessCcdConfig(), *args, **kwargs):
        pipeBase.Task.__init__(self, config=config, *args, **kwargs)
        self.makeSubtask("isr", IsrTask, config=config.isr)
        self.makeSubtask("calibrate", hscDc2.HscDc2CalibrateTask, config=config.calibrate)
        self.makeSubtask("photometry", PhotometryTask, config=config.photometry)

