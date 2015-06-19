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
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.meas.algorithms as measAlg
import lsst.meas.base
import lsst.pipe.base as pipeBase
from lsst.meas.astrom import ANetAstrometryTask
from .photoCal import PhotoCalTask
from .repair import RepairTask
from .measurePsf import MeasurePsfTask

class InitialPsfConfig(pexConfig.Config):
    """!Describes the initial PSF used for detection and measurement before we do PSF determination."""

    model = pexConfig.ChoiceField(
        dtype = str,
        doc = "PSF model type",
        default = "SingleGaussian",
        allowed = {
            "SingleGaussian": "Single Gaussian model",
            "DoubleGaussian": "Double Gaussian model",
        },
    )
    pixelScale = pexConfig.Field(
        dtype = float,
        doc = "Pixel size (arcsec).  Only needed if no Wcs is provided",
        default = 0.25,
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
    doMeasureApCorr = pexConfig.Field(
        dtype = bool,
        doc = "Compute aperture corrections?",
        default = True,
    )
    doApplyApCorr = pexConfig.Field(
        dtype = bool,
        doc = "Apply aperture corrections? Ignored if doMeasureApCorr is False",
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
        target = lsst.meas.base.SingleFrameMeasurementTask,
        doc = "Initial measurements used to feed PSF determination and aperture correction determination",
    )
    measurePsf   = pexConfig.ConfigurableField(target = MeasurePsfTask, doc = "")
    measurement = pexConfig.ConfigurableField(
        target = lsst.meas.base.SingleFrameMeasurementTask,
        doc = "Post-PSF-determination measurements used to feed other calibrations",
    )
    measureApCorr   = pexConfig.ConfigurableField(
        target = lsst.meas.base.MeasureApCorrTask,
        doc = "subtask to measure aperture corrections"
    )
    applyApCorr   = pexConfig.ConfigurableField(
        target = lsst.meas.base.ApplyApCorrTask,
        doc = "subtask to apply aperture corrections"
    )
    astrometry    = pexConfig.ConfigurableField(
        target = ANetAstrometryTask,
        doc = "fit WCS of exposure",
    )
    photocal      = pexConfig.ConfigurableField(
        target = PhotoCalTask,
        doc = "peform photometric calibration",
    )

    def validate(self):
        pexConfig.Config.validate(self)
        if self.doPhotoCal and not self.doAstrometry:
            raise ValueError("Cannot do photometric calibration without doing astrometric matching")

    def setDefaults(self):
        self.detection.includeThresholdMultiplier = 10.0
        self.initialMeasurement.algorithms.names -= ["base_ClassificationExtendedness"]
        initflags = [x for x in self.measurePsf.starSelector["catalog"].badStarPixelFlags]
        self.measurePsf.starSelector["catalog"].badStarPixelFlags.extend(initflags)
        self.background.binSize = 1024

## \addtogroup LSST_task_documentation
## \{
## \page CalibrateTask
## \ref CalibrateTask_ "CalibrateTask"
## \copybrief CalibrateTask
## \}

class CalibrateTask(pipeBase.Task):
    """!
\anchor CalibrateTask_

\brief Calibrate an exposure: measure PSF, subtract background, measure astrometry and photometry

\section pipe_tasks_calibrate_Contents Contents

 - \ref pipe_tasks_calibrate_Purpose
 - \ref pipe_tasks_calibrate_Initialize
 - \ref pipe_tasks_calibrate_IO
 - \ref pipe_tasks_calibrate_Config
 - \ref pipe_tasks_calibrate_Metadata
 - \ref pipe_tasks_calibrate_Debug
 - \ref pipe_tasks_calibrate_Example

\section pipe_tasks_calibrate_Purpose	Description

\copybrief CalibrateTask

Calculate an Exposure's zero-point given a set of flux measurements of stars matched to an input catalogue.
The type of flux to use is specified by CalibrateConfig.fluxField.

The algorithm clips outliers iteratively, with parameters set in the configuration.

\note This task can adds fields to the schema, so any code calling this task must ensure that
these columns are indeed present in the input match list; see \ref pipe_tasks_calibrate_Example

\section pipe_tasks_calibrate_Initialize	Task initialisation

\copydoc \_\_init\_\_

CalibrateTask delegates most of its work to a set of sub-Tasks:
<DL>
<DT> repair \ref RepairTask_ "RepairTask"
<DD> Interpolate over defects such as bad columns and cosmic rays.  This task is called twice;  once
before the %measurePsf step and again after the PSF has been measured.
<DT> detection \ref SourceDetectionTask_ "SourceDetectionTask"
<DD> Initial (high-threshold) detection phase for calibration
<DT> initialMeasurement \ref SingleFrameMeasurementTask_ "SingleFrameMeasurementTask"
<DD> Make the initial measurements used to feed PSF determination and aperture correction determination
<DT> astrometry \ref AstrometryTask_ "AstrometryTask"
<DD> Solve the astrometry.  May be disabled by setting CalibrateTaskConfig.doAstrometry to be False.
This task is called twice;  once before the %measurePsf step and again after the PSF has been measured.
<DT> %measurePsf \ref MeasurePsfTask_ "MeasurePsfTask"
<DD> Estimate the PSF.  May be disabled by setting CalibrateTaskConfig.doPsf to be False.  If requested
the astrometry is solved before this is called, so if you disable the astrometry the %measurePsf
task won't have access to objects positions.
<DT> measurement \ref SingleFrameMeasurementTask_ "SingleFrameMeasurementTask"
<DD> Post-PSF-determination measurements used to feed other calibrations
<DT> photocal \ref PhotoCalTask_ "PhotoCalTask"
<DD> Solve for the photometric zeropoint.
May be disabled by setting CalibrateTaskConfig.doPhotoCal to be False.
\em N.b.  Requires that \c astrometry was successfully run.
</DL>

You can replace any of these subtasks if you wish, see \ref calibrate_MyAstrometryTask.
\note These task APIs are not well controlled, so replacing a task is a matter of matching
a poorly specified interface.  We will be working on this over the first year of construction.

\section pipe_tasks_calibrate_IO		Invoking the Task

\copydoc run

\section pipe_tasks_calibrate_Config       Configuration parameters

See \ref CalibrateConfig

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

\section pipe_tasks_calibrate_Metadata   Quantities set in Metadata

<DL>
<DT>Task metadata
<DD>
<DL>
<DT> MAGZERO <DD> Measured zeropoint (DN per second)
</DL>

<DT> Exposure metadata
<DD>
<DL>
<DT> MAGZERO_RMS <DD> MAGZERO's RMS == return.sigma
<DT> MAGZERO_NOBJ <DD> Number of stars used == return.ngood
<DT> COLORTERM1 <DD> ?? (always 0.0)
<DT> COLORTERM2 <DD> ?? (always 0.0)
<DT> COLORTERM3 <DD> ?? (always 0.0)
</DL>
</DL>

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

\section pipe_tasks_calibrate_Debug		Debug variables

The \link lsst.pipe.base.cmdLineTask.CmdLineTask command line task\endlink interface supports a
flag \c -d to import \b debug.py from your \c PYTHONPATH; see \ref baseDebug for more about \b debug.py files.

The calibrate task has a debug dictionary with keys which correspond to stages of the CalibrationTask
processing:
<DL>
<DT>repair
<DD> Fixed defects and masked cosmic rays with a guessed PSF.  Action: show the exposure.
<DT>background
<DD> Subtracted background (no sources masked).  Action: show the exposure
<DT>PSF_repair
<DD> Fixed defects and removed cosmic rays with an estimated PSF.  Action: show the exposure
<DT>PSF_background
<DD> Subtracted background (calibration sources masked).  Action: show the exposure
<DT>calibrate
<DD> Just before astro/photo calibration.  Action: show the exposure, and
 - sources as smallish green o
 - matches (if exposure has a Wcs).
  - catalog position as a largish yellow +
  - source position as a largish red x
</DL>
The values are the \c ds9 frame to display in (if >= 1); if <= 0, nothing's displayed.
There's an example \ref pipe_tasks_calibrate_Debug_example "here".

Some subtasks may also have their own debug information; see individual Task documentation.

\section pipe_tasks_calibrate_Example	A complete example of using CalibrateTask

This code is in \link calibrateTask.py\endlink in the examples directory, and can be run as \em e.g.
\code
examples/calibrateTask.py --ds9
\endcode
\dontinclude calibrateTask.py

Import the task (there are some other standard imports; read the file if you're curious)
\skipline CalibrateTask

Create the detection task
\skip CalibrateTask.ConfigClass
\until config=config
Note that we're using a custom AstrometryTask (because we don't have a valid astrometric catalogue handy);
see \ref calibrate_MyAstrometryTask.

We're now ready to process the data (we could loop over multiple exposures/catalogues using the same
task objects) and unpack the results
\skip loadData
\until sources

We then might plot the results (\em e.g. if you set \c --ds9 on the command line)
\skip display
\until dot

\subsection calibrate_MyAstrometryTask Using a Custom Astrometry Task

The first thing to do is define my own task:
\dontinclude calibrateTask.py
\skip MyAstrometryTask
\skip MyAstrometryTask
\until super

Then we need our own \c run method.  First unpack the filtername and wcs
\skip run
\until wcs
Then build a "reference catalog" by shamelessly copying the catalog of detected sources
\skip schema
\until get("photometric")
(you need to set "flux" as well as \c filterName due to a bug in the photometric calibration code;
<A HREF=https://jira.lsstcorp.org/browse/DM-933>DM-933</A>).

Then "match" by zipping up the two catalogs,
\skip matches
\until append
and finally return the desired results.
\skip return
\until )

<HR>
\anchor pipe_tasks_calibrate_Debug_example
To investigate the \ref pipe_tasks_calibrate_Debug, put something like
\code{.py}
    import lsstDebug
    def DebugInfo(name):
        di = lsstDebug.getInfo(name)        # N.b. lsstDebug.Info(name) would call us recursively
        if name == "lsst.pipe.tasks.calibrate":
            di.display = dict(
                repair = 1,
                calibrate = 2,
            )

        return di

    lsstDebug.Info = DebugInfo
\endcode
into your debug.py file and run calibrateTask.py with the \c --debug flag.
    """
    ConfigClass = CalibrateConfig
    _DefaultName = "calibrate"

    def __init__(self, **kwargs):
        """!
        Create the calibration task

        \param **kwargs keyword arguments to be passed to lsst.pipe.base.task.Task.__init__
        """
        pipeBase.Task.__init__(self, **kwargs)

        # the calibrate Source Catalog is divided into two catalogs to allow measurement to be run twice
        # schema1 contains everything except what is added by the second measurement task.
        # Before the second measurement task is run, self.schemaMapper transforms the sources into
        # the final output schema, at the same time renaming the measurement fields to "initial_" 
        self.schema1 = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask("repair")
        self.makeSubtask("detection", schema=self.schema1)
        beginInitial = self.schema1.getFieldCount()
        self.makeSubtask("initialMeasurement", schema=self.schema1, algMetadata=self.algMetadata)
        endInitial = self.schema1.getFieldCount()
        self.makeSubtask("measurePsf", schema=self.schema1)
        self.makeSubtask("measureApCorr", schema=self.schema1)
        self.makeSubtask("applyApCorr", schema=self.schema1)
        self.makeSubtask("astrometry", schema=self.schema1)
        self.makeSubtask("photocal", schema=self.schema1)

        # create a schemaMapper to map schema1 into schema2
        self.schemaMapper = afwTable.SchemaMapper(self.schema1)
        separator =  "_"
        count = 0
        for item in self.schema1:
            count = count + 1
            field = item.getField()
            name = field.getName()
            if count > beginInitial and count <= endInitial: 
                name = "initial" + separator + name 
            self.schemaMapper.addMapping(item.key, name)

        # measurements fo the second measurement step done with a second schema
        schema = self.schemaMapper.editOutputSchema()
        self.makeSubtask("measurement", schema=schema, algMetadata=self.algMetadata)

        # the final schema is the same as the schemaMapper output
        self.schema = self.schemaMapper.getOutputSchema()

    def getCalibKeys(self):
        """!
        Return a sequence of schema keys that represent fields that should be propagated from
        icSrc to src by ProcessCcdTask.
        """
        if self.config.doPsf:
            return (self.measurePsf.candidateKey, self.measurePsf.usedKey)
        else:
            return ()

    @pipeBase.timeMethod
    def run(self, exposure, defects=None, idFactory=None):
        """!Run the calibration task on an exposure

        \param[in,out]  exposure   Exposure to calibrate; measured PSF will be installed there as well
        \param[in]      defects    List of defects on exposure
        \param[in]      idFactory  afw.table.IdFactory to use for source catalog.
        \return a pipeBase.Struct with fields:
        - exposure: Repaired exposure
        - backgrounds: A list of background models applied in the calibration phase
        - psf: Point spread function
        - sources: Sources used in calibration
        - matches: Astrometric matches
        - matchMeta: Metadata for astrometric matches
        - photocal: Output of photocal subtask

        It is moderately important to provide a decent initial guess for the seeing if you want to
        deal with cosmic rays.  If there's a PSF in the exposure it'll be used; failing that the
        CalibrateConfig.initialPsf is consulted (although the pixel scale will be taken from the
        WCS if available).

        If the exposure contains an lsst.afw.image.Calib object with the exposure time set, MAGZERO
        will be set in the task metadata.
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

        # Make both tables from the same detRet, since detRet can only be run once
        table1 = afwTable.SourceTable.make(self.schema1, idFactory)
        table1.setMetadata(self.algMetadata)
        detRet = self.detection.makeSourceCatalog(table1, exposure)
        sources1 = detRet.sources


        if detRet.fpSets.background:
            backgrounds.append(detRet.fpSets.background)

        # do the initial measurement.  This is normally done for star selection, but do it 
        # even if the psf is not going to be calculated for consistency
        self.initialMeasurement.measure(exposure, sources1)

        if self.config.doPsf:
            if self.config.doAstrometry:
                astromRet = self.astrometry.run(exposure, sources1)
                matches = astromRet.matches
            else:
                # If doAstrometry is False, we force the Star Selector to either make them itself
                # or hope it doesn't need them.
                matches = None
            psfRet = self.measurePsf.run(exposure, sources1, matches=matches)
            psf = psfRet.psf
        elif exposure.hasPsf():
            psf = exposure.getPsf()
        else:
            psf = None

        # Wash, rinse, repeat with proper PSF

        if self.config.doPsf:
            self.repair.run(exposure, defects=defects, keepCRs=None)
            self.display('PSF_repair', exposure=exposure)

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

            self.display('PSF_background', exposure=exposure)

        # make a second table with which to do the second measurement
        # the schemaMapper will copy the footprints and ids, which is all we need.
        table2 = afwTable.SourceTable.make(self.schema, idFactory)
        table2.setMetadata(self.algMetadata)
        sources = afwTable.SourceCatalog(table2)
        # transfer to a second table -- note that the slots do not have to be reset here
        # as long as measurement.run follows immediately
        sources.extend(sources1, self.schemaMapper)

        if self.config.doMeasureApCorr:
            # Run measurement through all flux measurements (all have the same execution order),
            # then apply aperture corrections, then run the rest of the measurements
            apCorrOrder = lsst.meas.base.BasePlugin.APCORR_ORDER
            self.measurement.run(exposure, sources, endOrder=apCorrOrder)
            apCorrMap = self.measureApCorr.run(bbox=exposure.getBBox(), catalog=sources).apCorrMap
            exposure.getInfo().setApCorrMap(apCorrMap)
            if self.config.doApplyApCorr:
                self.applyApCorr.run(catalog=sources, apCorrMap=apCorrMap)
            self.measurement.run(exposure, sources, beginOrder=apCorrOrder)
        else:
            apCorrMap = None
            self.measurement.run(exposure, sources)

        if self.config.doAstrometry:
            astromRet = self.astrometry.run(exposure, sources)
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
        self.display('calibrate', exposure=exposure, sources=sources, matches=matches)
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
        """!Initialise the calibration procedure by setting the PSF to a configuration-defined guess.

        \param[in,out] exposure Exposure to process; fake PSF will be installed here.
        \throws AssertionError If exposure or exposure.getWcs() are None
        """
        assert exposure, "No exposure provided"

        wcs = exposure.getWcs()
        if wcs:
            pixelScale = wcs.pixelScale().asArcseconds()
        else:
            pixelScale = self.config.initialPsf.pixelScale

        cls = getattr(measAlg, self.config.initialPsf.model + "Psf")

        fwhm = self.config.initialPsf.fwhm/pixelScale
        size = self.config.initialPsf.size
        self.log.info("installInitialPsf fwhm=%s pixels; size=%s pixels" % (fwhm, size))
        psf = cls(size, size, fwhm/(2*math.sqrt(2*math.log(2))))
        exposure.setPsf(psf)
