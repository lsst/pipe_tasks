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
import lsst.afw.math as afwMath
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
    astrometry    = pexConfig.ConfigurableField(target = AstrometryTask, doc = "")
    photocal      = pexConfig.ConfigurableField(target = PhotoCalTask, doc="")

    def validate(self):
        pexConfig.Config.validate(self)

        if self.doPhotoCal and not self.doAstrometry:
            raise ValueError("Cannot do photometric calibration without doing astrometric matching")

    def setDefaults(self):
        self.detection.includeThresholdMultiplier = 10.0
        self.initialMeasurement.algorithms.names -= ["correctfluxes", "classification.extendedness"]
        self.measurement.algorithms.names -= ["correctfluxes", "classification.extendedness"]
        initflags = [x for x in self.measurePsf.starSelector["catalog"].badStarPixelFlags]
        self.measurePsf.starSelector["catalog"].badStarPixelFlags.extend(initflags)
        self.background.binSize = 1024        

class CalibrateTask(pipeBase.Task):
    """Calibrate an exposure: measure PSF, subtract background, etc.
    """
    ConfigClass = CalibrateConfig

    def __init__(self, tableVersion=0, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)
        self.schema1 = afwTable.SourceTable.makeMinimalSchema()
        minimalCount = self.schema1.getFieldCount()
        self.algMetadata = dafBase.PropertyList()
        self.tableVersion = tableVersion
        self.makeSubtask("repair")
        self.makeSubtask("detection", schema=self.schema1, tableVersion=tableVersion)
        beginInitial = self.schema1.getFieldCount()
        self.makeSubtask("initialMeasurement", schema=self.schema1, algMetadata=self.algMetadata)
        endInitial = self.schema1.getFieldCount()
        self.makeSubtask("measurePsf", schema=self.schema1, tableVersion=tableVersion)

        self.schema2 = afwTable.SourceTable.makeMinimalSchema()
        beginMeasurement = self.schema2.getFieldCount()
        self.makeSubtask("measurement", schema=self.schema2, algMetadata=self.algMetadata)
        endMeasurement = self.schema2.getFieldCount()
        self.makeSubtask("astrometry", schema=self.schema2, tableVersion=tableVersion)
        self.makeSubtask("photocal", schema=self.schema2, tableVersion=tableVersion)
        self.schema = afwTable.Schema()
        self.schemaMapper1 = afwTable.SchemaMapper(self.schema1)
        self.schemaMapper2 = afwTable.SchemaMapper(self.schema2)
        if self.tableVersion == 0: separator = "."
        else: separator =  "_"
        count = 0
        for item in self.schema1:
            count = count + 1
            field = item.getField()
            name = field.getName()
            if count > beginInitial and count <= endInitial: 
                name = "initial" + separator + name 
            self.schemaMapper1.addMapping(item.key, name)
            self.schemaMapper2.addOutputField(field.copyRenamed(name))
        count = 0
        for item in self.schema2:
            count = count + 1
            if count <= minimalCount: continue
            field = item.getField()
            name = field.getName()
            if count > beginMeasurement and count <= endMeasurement: name = "measurement" + separator + name 
            self.schemaMapper2.addMapping(item.key, name)
            self.schemaMapper1.addOutputField(field.copyRenamed(name))
        self.schema = self.schemaMapper2.getOutputSchema()

    def getCalibKeys(self):
        """
        Return a sequence of schema keys that represent fields that should be propagated from
        icSrc to src by ProcessCcdTask.
        """
        if self.config.doPsf:
            return (self.measurePsf.candidateKey, self.measurePsf.usedKey)
        else:
            return ()

    @pipeBase.timeMethod
    def run(self, exposure, defects=None, idFactory=None):
        """Calibrate an exposure: measure PSF, subtract background, measure astrometry and photometry

        @param[in,out]  exposure   Exposure to calibrate; measured PSF will be installed there as well
        @param[in]      defects    List of defects on exposure
        @param[in]      idFactory  afw.table.IdFactory to use for source catalog.
        @return a pipeBase.Struct with fields:
        - backgrounds: A list of background models applied in the calibration phase
        - psf: Point spread function
        - sources: Sources used in calibration
        - matches: Astrometric matches
        - matchMeta: Metadata for astrometric matches
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
        table1 = afwTable.SourceTable.make(self.schema1, idFactory)
        table1.setMetadata(self.algMetadata)
        table1.setVersion(self.tableVersion)
        detRet = self.detection.makeSourceCatalog(table1, exposure)
        sources1 = detRet.sources


        if self.config.doAstrometry or self.config.doPhotoCal:
            # make a second table with which to do the second measurement
            # the schemaMapper will copy the footprints and ids, which is all we need.
            table2 = afwTable.SourceTable.make(self.schema2, idFactory)
            table2.setMetadata(self.algMetadata)
            table2.setVersion(self.tableVersion)
            sources2 = afwTable.SourceCatalog(table2)
            # transfer to a second table
            schemaMapper = afwTable.SchemaMapper(self.schema1)
            count = 0
            for item in self.schema1:
                field = item.getField()
                name = field.getName()
                schemaMapper.addMapping(item.key, name)
                break
            sources2.extend(sources1, schemaMapper)

        if detRet.fpSets.background:
            backgrounds.append(detRet.fpSets.background)

        if self.config.doPsf:
            self.initialMeasurement.measure(exposure, sources1)

            if self.config.doAstrometry:
                astromRet = self.astrometry.run(exposure, sources1)
                matches = astromRet.matches
            else:
                # If doAstrometry is False, we force the Star Selector to either make them itself
                # or hope it doesn't need them.
                matches = None
            psfRet = self.measurePsf.run(exposure, sources1, matches=matches)
            cellSet = psfRet.cellSet
            psf = psfRet.psf
        elif exposure.hasPsf():
            psf = exposure.getPsf()
            cellSet = None
        else:
            psf, cellSet = None, None

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

        if self.config.doAstrometry or self.config.doPhotoCal:
            self.measurement.run(exposure, sources2)

        if self.config.doAstrometry:
            astromRet = self.astrometry.run(exposure, sources2)
            matches = astromRet.matches
            matchMeta = astromRet.matchMeta
        else:
            matches, matchMeta = None, None

        if self.config.doPhotoCal:
            assert(matches is not None)
            try:
                photocalRet = self.photocal.run(exposure, matches)
            except Exception, e:
                self.log.warn("Failed to determine photometric zero-point: %s" % e)
                photocalRet = None
                self.metadata.set('MAGZERO', float("NaN"))
                
            if photocalRet:
                self.log.info("Photometric zero-point: %f" % photocalRet.calib.getMagnitude(1.0))
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

        self.display('calibrate', exposure=exposure, sources=sources2, matches=matches)
        assert(len(sources1) == len(sources2))
        for i in range(len(sources1)):
            foot2 = sources2[i].getFootprint()
            foot = sources1[i].getFootprint()
            assert(foot.getCentroid() == foot2.getCentroid())
            spans = foot.getSpans()
            spans2 = foot2.getSpans()
            assert(len(spans) == len(spans2))
            for j in range(len(spans)):
                assert(spans[j] == spans2[j])
        # now make the final sources catalog
        table = afwTable.SourceTable.make(self.schemaMapper1.getOutputSchema(), idFactory)
        table.setMetadata(self.algMetadata)
        table.setVersion(self.tableVersion)
        table.preallocate(len(sources1)) # not required, but nice
        sources = afwTable.SourceCatalog(table)
        sources.extend(sources1, self.schemaMapper1)
        assert(len(sources) == len(sources2))
        for i in range(len(sources)):
            foot2 = sources2[i].getFootprint()
            foot = sources[i].getFootprint()
            assert(foot.getCentroid() == foot2.getCentroid())
            spans = foot.getSpans()
            spans2 = foot2.getSpans()
            assert(len(spans) == len(spans2))
            for j in range(len(spans)):
                assert(spans[j] == spans2[j])

        for i in range(len(sources)):
            sources[i].assign(sources2[i], self.schemaMapper2)
      
        return pipeBase.Struct(
            exposure = exposure,
            backgrounds = backgrounds,
            psf = psf,
            sources = sources,
            matches = matches,
            matchMeta = matchMeta,
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
        self.log.info("installInitialPsf fwhm=%s pixels; size=%s pixels" % (fwhm, size))
        psf = cls(size, size, fwhm/(2*math.sqrt(2*math.log(2))))
        exposure.setPsf(psf)
