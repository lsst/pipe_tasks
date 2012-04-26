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
import lsst.afw.table as afwTable
import lsst.meas.algorithms as measAlg
import lsst.pipe.base as pipeBase
from lsst.meas.photocal import PhotoCalTask
from .astrometry import AstrometryTask
from .repair import RepairTask
from .measurePsf import MeasurePsfTask

class InitialPsfConfig(pexConfig.Config):
    """Describes the initial PSF used for detection and measurement before we do PSF determination."""

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

class CalibrateConfig(pexConfig.Config):
    initialPsf = pexConfig.ConfigField(dtype=InitialPsfConfig, doc=InitialPsfConfig.__doc__)
    doBackground = pexConfig.Field(
        dtype = bool,
        doc = "Subtract background (after computing it, if not supplied)?",
        default = True,
    )
    doPsf = pexConfig.Field(
        dtype = bool,
        doc = "Perform PSF fitting?",
        default = True,
    )
    doComputeApCorr = pexConfig.Field(
        dtype = bool,
        doc = "Calculate the aperture correction?",
        default = True,
    )
    doAstrometry = pexConfig.Field(
        dtype = bool,
        doc = "Compute astrometric solution?",
        default = True,
    )
    doPhotoCal = pexConfig.Field(
        dtype = bool,
        doc = "Compute photometric zeropoint?",
        default = True,
    )
    background = pexConfig.ConfigField(
        dtype = measAlg.estimateBackground.ConfigClass,
        doc = "Background estimation configuration"
        )
    repair       = pexConfig.ConfigurableField(target = RepairTask, doc = "")
    detection    = pexConfig.ConfigurableField(
        target = measAlg.SourceDetectionTask,
        doc = "Initial (high-threshold) detection phase for calibration",
    )
    initialMeasurement = pexConfig.ConfigurableField(
        target = measAlg.SourceMeasurementTask,
        doc = "Initial measurements used to feed PSF determination and aperture correction determination",
    )
    measurePsf   = pexConfig.ConfigurableField(target = MeasurePsfTask, doc = "")
    measurement = pexConfig.ConfigurableField(
        target = measAlg.SourceMeasurementTask,
        doc = "Post-PSF-determination measurements used to feed other calibrations",
    )
    computeApCorr = pexConfig.ConfigField(dtype = measAlg.ApertureCorrectionConfig,
                                          doc = measAlg.ApertureCorrectionConfig.__doc__)
    astrometry    = pexConfig.ConfigurableField(target = AstrometryTask, doc = "")
    photocal      = pexConfig.ConfigurableField(target = PhotoCalTask, doc="")

    def validate(self):
        pexConfig.Config.validate(self)
        if self.doPsf and (self.doPhotoCal or self.doComputeApCorr or self.doAstrometry):
            if self.initialMeasurement.prefix == self.measurement.prefix:
                raise ValueError("CalibrateConfig.initialMeasurement and CalibrateConfig.measurement "\
                                     "have the same prefix; field names may clash.")
        if self.doComputeApCorr and not self.doPsf:
            raise ValueError("Cannot compute aperture correction without doing PSF determination")
        if self.measurement.doApplyApCorr and not self.doComputeApCorr:
            raise ValueError("Cannot apply aperture correction without computing it")
        if self.doPhotoCal and not self.doAstrometry:
            raise ValueError("Cannot do photometric calibration without doing astrometric matching")

    def setDefaults(self):
        self.detection.includeThresholdMultiplier = 10.0
        self.initialMeasurement.prefix = "initial."
        self.initialMeasurement.doApplyApCorr = False
        self.background.binSize = 1024
        self.computeApCorr.alg1.name = "flux.psf"
        self.computeApCorr.alg2.name = "flux.sinc"
        

class CalibrateTask(pipeBase.Task):
    """Calibrate an exposure: measure PSF, subtract background, etc.
    """
    ConfigClass = CalibrateConfig

    def __init__(self, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask("repair")
        self.makeSubtask("detection", schema=self.schema)
        self.makeSubtask("initialMeasurement", schema=self.schema, algMetadata=self.algMetadata)
        self.makeSubtask("measurePsf", schema=self.schema)
        self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)
        self.makeSubtask("astrometry", schema=self.schema)
        self.makeSubtask("photocal", schema=self.schema)

    @pipeBase.timeMethod
    def run(self, exposure, defects=None):
        """Calibrate an exposure: measure PSF, subtract background, measure astrometry and photometry

        @param[in,out]  exposure   Exposure to calibrate; measured PSF will be installed there as well
        @param[in]      defects    List of defects on exposure
        @return a pipeBase.Struct with fields:
        - psf: Point spread function
        - apCorr: Aperture correction
        - sources: Sources used in calibration
        - matches: Astrometric matches
        - matchMeta: Metadata for astrometric matches
        """
        assert exposure is not None, "No exposure provided"

        self.installInitialPsf(exposure)

        keepCRs = True                  # At least until we know the PSF
        self.repair.run(exposure, defects=defects, keepCRs=keepCRs)
        self.display('repair', exposure=exposure)

        if self.config.doBackground:
            with self.timer("background"):
                bg, exposure = measAlg.estimateBackground(exposure, self.config.background, subtract=True)
                del bg

            self.display('background', exposure=exposure)
        
        table = afwTable.SourceTable.make(self.schema) # TODO: custom IdFactory for globally unique IDs
        table.setMetadata(self.algMetadata)
        detRet = self.detection.makeSourceCatalog(table, exposure)
        sources = detRet.sources

        if self.config.doPsf:
            self.initialMeasurement.measure(exposure, sources)
            psfRet = self.measurePsf.run(exposure, sources)
            cellSet = psfRet.cellSet
            psf = psfRet.psf
        else:
            psf, cellSet = None, None

        # Wash, rinse, repeat with proper PSF

        if self.config.doPsf:
            self.repair.run(exposure, defects=defects, keepCRs=None)
            self.display('repair', exposure=exposure)

        if self.config.doBackground:   # is repeating this necessary?  (does background depend on PSF model?)
            with self.timer("background"):
                # Subtract background
                background, exposure = measAlg.estimateBackground(
                    exposure, self.config.background, subtract=True,
                    statsKeys=('BGMEAN2', 'BGVAR2'))
                self.log.log(self.log.INFO, "Fit and subtracted background")

            self.display('background', exposure=exposure)

        if self.config.doComputeApCorr or self.config.doAstrometry or self.config.doPhotoCal:
            self.measurement.measure(exposure, sources)   # don't use run, because we don't have apCorr yet

        if self.config.doComputeApCorr:
            assert(self.config.doPsf)
            apCorr = self.computeApCorr(exposure, cellSet)
        else:
            apCorr = None

        if self.measurement.config.doApplyApCorr:
            assert(apCorr is not None)
            self.measurement.applyApCorr(sources, apCorr)

        if self.config.doAstrometry:
            astromRet = self.astrometry.run(exposure, sources)
            matches = astromRet.matches
            matchMeta = astromRet.matchMeta
        else:
            matches, matchMeta = None, None

        if self.config.doPhotoCal:
            assert(matches is not None)
            photocalRet = self.photocal.run(matches)
            zp = photocalRet.photocal
            self.log.log(self.log.INFO, "Photometric zero-point: %f" % zp.getMag(1.0))
            exposure.getCalib().setFluxMag0(zp.getFlux(0))

        self.display('calibrate', exposure=exposure, sources=sources, matches=matches)

        return pipeBase.Struct(
            exposure = exposure,
            psf = psf,
            apCorr = apCorr,
            sources = sources,
            matches = matches,
            matchMeta = matchMeta,
        )

    def installInitialPsf(self, exposure):
        """Initialise the calibration procedure by setting the PSF to a configuration-defined guess.

        @param[in,out] exposure Exposure to process; fake PSF will be installed here.
        """
        assert exposure, "No exposure provided"
        
        wcs = exposure.getWcs()
        assert wcs, "No wcs in exposure"

        model = self.config.initialPsf.model
        fwhm = self.config.initialPsf.fwhm / wcs.pixelScale().asArcseconds()
        size = self.config.initialPsf.size
        self.log.log(self.log.INFO, "installInitialPsf fwhm=%s pixels; size=%s pixels" % (fwhm, size))
        psf = afwDet.createPsf(model, size, size, fwhm/(2*math.sqrt(2*math.log(2))))
        exposure.setPsf(psf)

    @pipeBase.timeMethod
    def computeApCorr(self, exposure, cellSet):
        """Measure aperture correction

        @param exposure Exposure to process
        @param cellSet Set of cells of PSF stars
        """
        assert exposure, "No exposure provided"
        assert cellSet, "No cellSet provided"
        metadata = dafBase.PropertyList()
        apCorr = measAlg.ApertureCorrection(exposure, cellSet, metadata, self.config.computeApCorr, self.log)
        x, y = exposure.getWidth() / 2.0, exposure.getHeight() / 2.0
        value, error = apCorr.computeAt(x, y)
        self.log.log(self.log.INFO, "Aperture correction using %d/%d stars: %f +/- %f" %
                     (metadata.get("numAvailStars"), metadata.get("numGoodStars"), value, error))
        for key in metadata.names():
            self.metadata.add("apCorr.%s" % key, metadata.get(key))
        # XXX metadata?
        return apCorr
