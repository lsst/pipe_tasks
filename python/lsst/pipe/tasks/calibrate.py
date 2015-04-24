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
import numpy

import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
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
    doCurveOfGrowth = pexConfig.Field(
        dtype = bool,
        doc = "Measure and apply curve of growth?",
        default = True,
    )
    doMeasureApCorr = pexConfig.Field(
        dtype = bool,
        doc = "Compute aperture corrections?",
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
    requireAstrometry = pexConfig.Field(
        dtype = bool,
        doc = "Require astrometry to succeed, if activated?",
        default = False,
        )
    requirePhotoCal = pexConfig.Field(
        dtype = bool,
        doc = "Require photometric calibration to succeed?",
        default = False,
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
    measureCurveOfGrowth = pexConfig.ConfigurableField(
        target = measAlg.CurveOfGrowthMeasurementTask,
        doc = "Curve of growth, for correcting apertures to infinity",
    )
    measureApCorr   = pexConfig.ConfigurableField(
        target = measAlg.MeasureApCorrTask,
        doc = "subtask to measure aperture corrections"
    )
    astrometry    = pexConfig.ConfigurableField(target = AstrometryTask, doc = "")
    photocal      = pexConfig.ConfigurableField(target = PhotoCalTask, doc="")

    def validate(self):
        pexConfig.Config.validate(self)
        if self.doPsf and (self.doPhotoCal or self.doAstrometry):
            if self.initialMeasurement.prefix == self.measurement.prefix:
                raise ValueError("CalibrateConfig.initialMeasurement and CalibrateConfig.measurement "\
                                     "have the same prefix; field names may clash.")
        if self.doPhotoCal and not self.doAstrometry:
            raise ValueError("Cannot do photometric calibration without doing astrometric matching")
        if self.requireAstrometry and not self.doAstrometry:
            raise ValueError("Astrometric solution required, but not activated")
        if self.requirePhotoCal and not self.doPhotoCal:
            raise ValueError("Photometric calibration required, but not activated")

    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
        self.detection.includeThresholdMultiplier = 10.0
        # Because we don't deblend in CalibrateTask, we want to minimize merge detections caused by
        # growing the footprints.
        self.detection.returnOriginalFootprints = True
        self.initialMeasurement.prefix = "initial."
        self.initialMeasurement.algorithms.names -= ["correctfluxes"]
        initflags = [self.initialMeasurement.prefix+x
                     for x in self.measurePsf.starSelector["catalog"].badStarPixelFlags]
        self.measurePsf.starSelector["catalog"].badStarPixelFlags.extend(initflags)
        self.background.binSize = 1024
        #
        # Don't measure the elliptical aperture fluxes when calibrating
        #
        aperture_elliptical = "flux.aperture.elliptical"
        if aperture_elliptical in self.measurement.value.algorithms.names:
            self.measurement.value.algorithms.names -= (aperture_elliptical,)
        #
        # Stop flux.gaussian recomputing the Gaussian's weights (as shape.sdss already did that)
        #
        try:
            self.initialMeasurement.algorithms['flux.gaussian'].fixed = True
            self.measurement.algorithms['flux.gaussian'].fixed = True
            self.initialMeasurement.algorithms['flux.gaussian'].centroid = \
                'initial.shape.sdss.centroid'
            self.initialMeasurement.algorithms['flux.gaussian'].shape = 'initial.shape.sdss'
        except pexConfig.FieldValidationError: # "flux.gaussian" isn't there
            pass
        
class CalibrateTask(pipeBase.Task):
    """Calibrate an exposure: measure PSF, subtract background, etc.
    """
    ConfigClass = CalibrateConfig

    def __init__(self, schema=None, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)
        if schema is None:
            schema = afwTable.SourceTable.makeMinimalSchema()
        self.schema = schema
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask("repair")
        self.makeSubtask("detection", schema=self.schema)
        self.makeSubtask("initialMeasurement", schema=self.schema, algMetadata=self.algMetadata)
        self.makeSubtask("measurePsf", schema=self.schema)
        self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)
        self.makeSubtask("measureCurveOfGrowth", schema=self.schema)
        self.makeSubtask("measureApCorr", schema=self.schema)
        self.makeSubtask("astrometry", schema=self.schema)
        self.makeSubtask("photocal", schema=self.schema)

    def getCalibKeys(self):
        """
        Return a sequence of schema keys that represent fields that should be propagated from
        icSrc to src by ProcessCcdTask.
        """
        if self.config.doPsf:
            if self.measurePsf.config.reserveFraction > 0:
                return (self.measurePsf.candidateKey, self.measurePsf.usedKey, self.measurePsf.reservedKey)
            else:
                return (self.measurePsf.candidateKey, self.measurePsf.usedKey)
        else:
            return ()

    @pipeBase.timeMethod
    def run(self, exposure, defects=None, idFactory=None, expId=0):
        """Calibrate an exposure: measure PSF, subtract background, measure astrometry and photometry

        @param[in,out]  exposure   Exposure to calibrate; measured Psf, Wcs, ApCorr, Calib, etc. will
                                   be installed there as well
        @param[in]      defects    List of defects on exposure
        @param[in]      idFactory  afw.table.IdFactory to use for source catalog.
        @param[in]      expId      Exposure id used for random number generation.
        @return a pipeBase.Struct with fields:
        - backgrounds: A list of background models applied in the calibration phase
        - psf: Point spread function
        - sources: Sources used in calibration
        - matches: Astrometric matches
        - matchMeta: Metadata for astrometric matches
        - apCorrMap: Map of aperture corrections
        - photocal: Output of photocal subtask
        """
        assert exposure is not None, "No exposure provided"

        if not exposure.hasPsf():
            self.installInitialPsf(exposure)
        if idFactory is None:
            idFactory = afwTable.IdFactory.makeSimple()
        backgrounds = afwMath.BackgroundList()
        keepCRs = True                  # At least until we know the PSF
        self.repair.run(exposure, defects=defects, keepCRs=keepCRs)
        self.display('repair', exposure=exposure)
        if self.config.doBackground:
            with self.timer("background"):
                bg, exposure = measAlg.estimateBackground(exposure, self.config.background, subtract=True)
                backgrounds.append(bg)

            self.display('background', exposure=exposure)
        table = afwTable.SourceTable.make(self.schema, idFactory)
        table.setMetadata(self.algMetadata)
        detRet = self.detection.makeSourceCatalog(table, exposure)
        sources = detRet.sources
        if detRet.fpSets.background:
            backgrounds.append(detRet.fpSets.background)

        if self.config.doPsf:
            self.initialMeasurement.measure(exposure, sources)

            matches = None
            if self.config.doAstrometry:
                # If doAstrometry is False, we force the Star Selector to either make them itself
                # or hope it doesn't need them.
                origWcs = exposure.getWcs()
                try:
                    astromRet = self.astrometry.run(exposure, sources)
                    matches = astromRet.matches
                except RuntimeError as e:
                    if self.config.requireAstrometry:
                        raise
                    self.log.warn("Unable to perform astrometry (%s): attempting to proceed" % e)
                finally:
                    # Restore original Wcs: we're going to repeat the astrometry later, and if it succeeded
                    # this time, running it again with the same basic setup means it should succeed again.
                    exposure.setWcs(origWcs)

            # This is an initial, throw-away run of photocal, since we need a valid Calib to run CModel,
            # and we need to run CModel to compute aperture corrections from it.
            if self.config.doPhotoCal:
                try:
                    if not matches:
                        raise RuntimeError("No matches available")
                    photocalRet = self.photocal.run(exposure, matches,
                                                    prefix=self.config.initialMeasurement.prefix,
                                                    doSelectUnresolved=False)# don't trust s/g without good PSF
                    self.log.info("Initial photometric zero-point: %f" % photocalRet.calib.getMagnitude(1.0))
                    exposure.getCalib().setFluxMag0(photocalRet.calib.getFluxMag0())
                except Exception, e:
                    self.log.warn("Failed to determine initial photometric zero-point: %s" % e)

            psfRet = self.measurePsf.run(exposure, sources, expId=expId, matches=matches)
            psf = psfRet.psf
        elif exposure.hasPsf():
            psf = exposure.getPsf()
        else:
            psf = None

        # Wash, rinse, repeat with proper PSF

        if self.config.doPsf:
            self.repair.run(exposure, defects=defects, keepCRs=None)
            self.display('repair', exposure=exposure)

        if self.config.doBackground:
            # Background estimation ignores (by default) pixels with the
            # DETECTED bit set, so now we re-estimate the background,
            # ignoring sources.  (see BackgroundConfig.ignoredPixelMask)
            with self.timer("background"):
                # Subtract background
                bg, exposure = measAlg.estimateBackground(
                    exposure, self.config.background, subtract=True,
                    statsKeys=('BGMEAN2', 'BGVAR2'))
                self.log.info("Fit and subtracted background")
                backgrounds.append(bg)

            self.display('background', exposure=exposure)

        if self.config.doMeasureApCorr:
            # Because we want to both compute the aperture corrections and apply them - and we do the latter
            # as a source measurement plugin ("CorrectFluxes"), we have to sandwich the aperture correction
            # measurement in between two source measurement passes, using the priority range arguments added
            # just for this purpose.
            apCorrApplyPriority = self.config.measurement.algorithms["correctfluxes"].priority
            self.measurement.run(exposure, sources, endPriority=apCorrApplyPriority)
            if self.config.doCurveOfGrowth:
                curveOfGrowth = self.applyCurveOfGrowth(sources)
            else:
                curveOfGrowth = None
            apCorrMap = self.measureApCorr.run(exposure.getBBox(afwImage.PARENT), sources)
            exposure.getInfo().setApCorrMap(apCorrMap)
            self.measurement.run(exposure, sources, beginPriority=apCorrApplyPriority)
        else:
            curveOfGrowth = None
            apCorrMap = None
            self.measurement.run(exposure, sources)

        matches, matchMeta = None, None
        if self.config.doAstrometry:
            try:
                astromRet = self.astrometry.run(exposure, sources)
                matches = astromRet.matches
                matchMeta = astromRet.matchMeta
            except RuntimeError as e:
                if self.config.requireAstrometry:
                    raise
                self.log.warn("Unable to perform astrometry (%s): attempting to proceed" % e)

        if self.config.doPhotoCal:
            try:
                if not matches:
                    raise RuntimeError("No matches available")
                photocalRet = self.photocal.run(exposure, matches)
            except Exception, e:
                if self.config.requirePhotoCal:
                    raise
                self.log.warn("Failed to determine updated photometric zero-point: %s" % e)
                photocalRet = None
                self.metadata.set('MAGZERO', float("NaN"))

            if photocalRet:
                self.log.info("Updated photometric zero-point: %f" % photocalRet.calib.getMagnitude(1.0))
                exposure.getCalib().setFluxMag0(photocalRet.calib.getFluxMag0())
                metadata = exposure.getMetadata()
                # convert to (mag/sec/adu) for metadata
                try:
                    magZero = photocalRet.zp - 2.5 * math.log10(exposure.getCalib().getExptime() )
                    metadata.set('MAGZERO', magZero)
                except:
                    self.log.warn("Could not set normalized MAGZERO in header: no exposure time")
                metadata.set('MAGZERO_RMS', photocalRet.sigma)
                metadata.set('MAGZERO_NOBJ', photocalRet.ngood)
                metadata.set('COLORTERM1', 0.0)
                metadata.set('COLORTERM2', 0.0)
                metadata.set('COLORTERM3', 0.0)    
        else:
            photocalRet = None

        self.display('calibrate', exposure=exposure, sources=sources, matches=matches)

        return pipeBase.Struct(
            exposure = exposure,
            backgrounds = backgrounds,
            psf = psf,
            sources = sources,
            matches = matches,
            matchMeta = matchMeta,
            curveOfGrowth = curveOfGrowth,
            apCorrMap = apCorrMap,
            photocal = photocalRet,
        )

    def installInitialPsf(self, exposure):
        """Initialise the calibration procedure by setting the PSF to a configuration-defined guess.

        @param[in,out] exposure Exposure to process; fake PSF will be installed here.
        """
        assert exposure, "No exposure provided"
        
        wcs = exposure.getWcs()
        assert wcs, "No wcs in exposure"

        cls = getattr(measAlg, self.config.initialPsf.model + "Psf")

        fwhm = self.config.initialPsf.fwhm / wcs.pixelScale().asArcseconds()
        size = self.config.initialPsf.size
        self.log.info("installInitialPsf fwhm=%.2f pixels; size=%d pixels" % (fwhm, size))
        psf = cls(size, size, fwhm/(2*math.sqrt(2*math.log(2))))
        exposure.setPsf(psf)

    def applyCurveOfGrowth(self, sources):
        """Measure and apply the curve of growth to the source catalog

        Returns the results from running measureCurveOfGrowth.
        """
        cogResults = self.measureCurveOfGrowth.run(sources)
        cog = cogResults.curveOfGrowth

        # Figure out which aperture corresponds to our calibration aperture
        # This requires assuming a parameter name for the aperture;
        # "radius" is used for the algorithms flux.sinc and flux.naive
        calibAlg = self.config.measurement.slots.calibFlux
        radius = self.config.measurement.algorithms[calibAlg].radius
        apertures = numpy.array(self.config.measurement.algorithms["flux.aperture"].radii)
        if len(numpy.where(apertures == radius)[0]) == 0:
            raise RuntimeError(
                "Calibration aperture (algorithm %s, radius %f) is not measured by flux.aperture (radii %s)" %
                (calibAlg, radius, apertures))
        calibIndex = numpy.where(apertures == radius)[0][0]
        corrIndex = numpy.where(numpy.isfinite(cog.apertureFlux))[0][-1] # Biggest aperture with good correctn

        ratio, ratioErr = cog.getRatio(calibIndex, corrIndex)
        self.log.info("Applying curve of growth to calibration flux %s (radius %.1f --> %.1f): %f +/- %f" %
                      (calibAlg, radius, apertures[corrIndex], ratio, ratioErr))

        # Apply the correction to the calibration flux. The aperture correction will take care of everything
        # that gets aperture-corrected.  The other apertures don't need to be corrected - as long as we're
        # using something corrected to a large radius to set the zero points, they'll automatically be
        # correct.
        sources[calibAlg][:] *= ratio
        sources[calibAlg + ".err"][:] = numpy.sqrt(sources[calibAlg + ".err"]**2 + ratioErr**2)

        return cogResults
