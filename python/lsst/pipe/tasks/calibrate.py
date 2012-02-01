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

import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.afw.detection as afwDet
import lsst.meas.algorithms as measAlg
import lsst.meas.algorithms.apertureCorrection as maApCorr
import lsst.meas.utils.sourceDetection as muDetection
#import lsst.meas.photocal as photocal
#from .astrometry import AstrometryTask
import lsst.pipe.base as pipeBase
from .repair import RepairTask
from .measurePsf import MeasurePsfTask
from .photometry import PhotometryTask, RephotometryTask

def propagateFlag(flag, old, new):
    """Propagate a flag from one source to another"""
    if old.getFlagForDetection() & flag:
        new.setFlagForDetection(new.getFlagForDetection() | flag)


class CalibrateConfig(pexConfig.Config):
    model = pexConfig.ChoiceField(
        dtype = str,
        doc = "PSF model type",
        default = "SingleGaussian",
        allowed = {
            "SingleGaussian": "Single Gaussian model",
            "DoubleGaussian": "Double Gaussian model",
        },
    )
    fwhm = pexConfig.Field(
        dtype = float,
        doc = "FWHM of PSF model (arcsec)",
        default = 1.0,
    )
    size = pexConfig.Field(
        dtype = int,
        doc = "Size of PSF model (pixels)",
        default = 15,
    )
    thresholdValue = pexConfig.Field(
        dtype = float,
        doc = "Threshold for PSF stars (relative to regular detection limit)",
        default = 10.0,
    )
    magnitudeLimitForCalibration = pexConfig.Field(
        dtype = float,
        doc = "The faintest star to consider for photometric calibration",
        default = 22.0,
    )
    doPsf = pexConfig.Field(
        dtype = bool,
        doc = "Perform PSF fitting?",
        default = True,
    )
    doAstrometry = pexConfig.Field(
        dtype = bool,
        doc = "Compute astrometric solution?",
        default = True,
    )
    doZeropoint = pexConfig.Field(
        dtype = bool,
        doc = "Compute photometric zeropoint?",
        default = True,
    )
    doApCorr = pexConfig.Field(
        dtype = bool,
        doc = "Calculate the aperture correction?",
        default = True,
    )
    doBackground = pexConfig.Field(
        dtype = bool,
        doc = "Subtract background (after computing it, if not supplied)?",
        default = True,
    )
    background = pexConfig.ConfigField(
        dtype = muDetection.estimateBackground.ConfigClass,
        doc = "Background estimation configuration"
        )
    apCorr       = pexConfig.ConfigField(dtype = maApCorr.ApertureCorrectionConfig, doc = "")
    repair       = pexConfig.ConfigField(dtype = RepairTask.ConfigClass,            doc = "")
    photometry   = pexConfig.ConfigField(dtype = PhotometryTask.ConfigClass,        doc = "")
    measurePsf   = pexConfig.ConfigField(dtype = MeasurePsfTask.ConfigClass,        doc = "")
#    astrometry   = pexConfig.ConfigField(dtype = AstrometryTask.ConfigClass,        doc = "")


class CalibrateTask(pipeBase.Task):
    """Conversion notes:
    
    Disabled display until we figure out how to turn it off or on
    
    Warning: I'm not sure I'm using metadata correctly (to replace old sdqa code)
    
    Made new subtasks for measuring PSF and astrometry    
    
    Eliminated the background subtask because it was such a thin layer around muDetection.estimateBackground
    
    Modified to NOT estimate a new background model if the user supplies one. The old code first applied
    the user-supplied background (if any) then fit and subtracted a new background.
    """
    ConfigClass = CalibrateConfig

    def __init__(self, config=CalibrateConfig(), *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("repair", RepairTask, config=config.repair)
        self.makeSubtask("photometry", PhotometryTask, config=config.photometry)
        self.makeSubtask("measurePsf", MeasurePsfTask, config=config.measurePsf)
        self.makeSubtask("rephotometry", RephotometryTask, config=config.photometry)
#        self.makeSubtask("astrometry", AstrometryTask, config=config.astrometry)

    @pipeBase.timeMethod
    def run(self, exposure, defects=None):
        """Calibrate an exposure: measure PSF, subtract background, measure astrometry and photometry

        @param exposure Exposure to calibrate
        @param defects List of defects on exposure
        @return a pipeBase.Struct with fields:
        - psf: Point spread function
        - apCorr: Aperture correction
        - sources: Sources used in calibration
        - matches: Astrometric matches
        - matchMeta: Metadata for astrometric matches
        """
        assert exposure is not None, "No exposure provided"

        fakePsf, wcs = self.makeFakePsf(exposure)

        self.repair.run(exposure, fakePsf, defects=defects, keepCRs=True)
        self.display('repair', exposure=exposure)

        if self.config.doBackground:
            with self.timer("background"):
                bg, exposure = muDetection.estimateBackground(exposure, self.config.background, subtract=True)
                del bg
            self.display('background', exposure=exposure)
        
        if self.config.doPsf or self.config.doAstrometry or self.config.doZeropoint:
            with self.timer("photometry"):
                photRet = self.photometry.run(exposure, fakePsf)
                sources = photRet.sources
                footprints = photRet.footprintSet
        else:
            sources, footprints = None, None

        if self.config.doPsf:
            psfRet = self.measurePsf.run(exposure, sources)
            psf = psfRet.psf
            cellSet = psfRet.cellSet
        else:
            psf, cellSet = None, None

        if self.config.doPsf and self.config.doApCorr:
            apCorr = self.apCorr(exposure, cellSet) # calculate the aperture correction; we may use it later
        else:
            apCorr = None

        # Wash, rinse, repeat with proper PSF

        if self.config.doPsf:
            self.repair.run(exposure, psf, defects=defects, keepCRs=False)
            self.display('repair', exposure=exposure)

        if self.config.doBackground:
            with self.timer("background"):
                # Subtract background
                background, exposure = muDetection.estimateBackground(
                    exposure, self.config.background, subtract=True)
                self.log.log(self.log.INFO, "Fit and subtracted background")
            self.display('background', exposure=exposure)

        if self.config.doPsf and (self.config.doAstrometry or self.config.doZeropoint):
            rephotRet = self.rephotometry.run(exposure, footprints, psf, apCorr)
            for old, new in zip(sources, rephotRet.sources):
                for flag in (measAlg.Flags.STAR, measAlg.Flags.PSFSTAR):
                    propagateFlag(flag, old, new)
            sources = rephotRet.sources
            del rephotRet
        
        if False and (self.config.doAstrometry or self.config.doZeropoint):
            astromRet = self.astrometry.run(exposure, sources)
            matches = astromRet.matches
            matchMeta = astromRet.matchMeta
        else:
            matches, matchMeta = None, None

        if self.config.doZeropoint and matches is not None:
            self.zeropoint(exposure, matches)

#        self.display('calibrate', exposure=exposure, sources=sources, matches=matches)

        return pipeBase.Struct(
            exposure = exposure,
            psf = psf,
            apCorr = apCorr,
            sources = sources,
            matches = matches,
            matchMeta = matchMeta,
        )

    def makeFakePsf(self, exposure):
        """Initialise the calibration procedure by setting the PSF and WCS

        @param exposure Exposure to process
        @return PSF, WCS
        """
        assert exposure, "No exposure provided"
        
        wcs = exposure.getWcs()
        assert wcs, "No wcs in exposure"

        model = self.config.model
        fwhm = self.config.fwhm / wcs.pixelScale().asArcseconds()
        size = self.config.size
        psf = afwDet.createPsf(model, size, size, fwhm/(2*math.sqrt(2*math.log(2))))
        return psf, wcs

    @pipeBase.timeMethod
    def apCorr(self, exposure, cellSet):
        """Measure aperture correction

        @param exposure Exposure to process
        @param cellSet Set of cells of PSF stars
        """
        assert exposure, "No exposure provided"
        assert cellSet, "No cellSet provided"
        metadata = dafBase.PropertyList()
        corr = maApCorr.ApertureCorrection(exposure, cellSet, metadata, self.config.apCorr, self.log)
        x, y = exposure.getWidth() / 2.0, exposure.getHeight() / 2.0
        value, error = corr.computeAt(x, y)
        self.log.log(self.log.INFO, "Aperture correction using %d/%d stars: %f +/- %f" %
                     (metadata.get("numAvailStars"), metadata.get("numGoodStars"), value, error))
        for key in metadata.names():
            self.metadata.add("apCorr.%s" % key, metadata.get(key))
        # XXX metadata?
        return corr

    @pipeBase.timeMethod
    def zeropoint(self, exposure, matches):
        """Photometric calibration

        @param exposure Exposure to process
        @param matches Matched sources
        """
        assert exposure, "No exposure provided"
        assert matches, "No matches provided"

        zp = photocal.calcPhotoCal(matches, log=self.log, goodFlagValue=0)
        self.log.log(self.log.INFO, "Photometric zero-point: %f" % zp.getMag(1.0))
        exposure.getCalib().setFluxMag0(zp.getFlux(0))
        return


class CalibratePsfTask(CalibrateTask):
    """Calibrate only the PSF for an image.
    
    Explicitly turns off other functions.
    
    Conversion notes:
    - Is it really necessary to restore the old config?
    - Surely there is a cleaner way to do this, such as creating a config that
      has these flags explicitly turned off?
    """
    def run(self, *args, **kwargs):
        oldConfig = self.config.copy()
        self.config.doBackground = False
        self.config.doDistortion = False
        self.config.doAstrometry = False
        self.config.doZeropoint = False

        retVal = CalibrateTask.run(self, *args, **kwargs)

        self.config = oldConfig
        
        return retVal
