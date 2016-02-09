#
# LSST Data Management System
# Copyright 2008-2016 AURA/LSST.
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
# see <https://www.lsstcorp.org/LegalNotices/>.
#
import math

from lsstDebug import getDebugFrame
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.daf.butlerUtils import ExposureIdInfo
import lsst.afw.table as afwTable
from lsst.meas.astrom import AstrometryTask, displayAstrometry
from .detectAndMeasure import DetectAndMeasureTask
from .photoCal import PhotoCalTask

class CalibrateConfig(pexConfig.Config):
    """Config for CalibrateTask"""
    doWrite = pexConfig.Field(
        dtype = bool,
        default = True,
        doc = "Save calibration results?",
    )
    doWriteMatches = pexConfig.Field(
        dtype = bool,
        default = True,
        doc = "Write reference matches (ignored if doWrite false)?",
    )
    detectAndMeasure = pexConfig.ConfigurableField(
        target = DetectAndMeasureTask,
        doc = "Detect sources to high sigma, deblend and peform single-frame measurement",
    )
    doAstrometry = pexConfig.Field(
        dtype = bool,
        default = True,
        doc = "Perform astrometric calibration?",
    )
    astrometry = pexConfig.ConfigurableField(
        target = AstrometryTask,
        doc = "Perform astrometric calibration to refine the WCS",
    )
    requireAstrometry = pexConfig.Field(
        dtype = bool,
        default = True,
        doc = "Raise an exception if astrometry fails? Ignored if doAstrometry false.",
    )
    doPhotoCal = pexConfig.Field(
        dtype = bool,
        default = True,
        doc = "Perform phometric calibration?",
    )
    requirePhotoCal = pexConfig.Field(
        dtype = bool,
        default = True,
        doc = "Raise an exception if photoCal fails? Ignored if doPhotoCal false.",
    )
    photoCal = pexConfig.ConfigurableField(
        target = PhotoCalTask,
        doc = "Perform photometric calibration",
    )
    icSourceFieldsToCopy = pexConfig.ListField(
        dtype = str,
        default = ("calib_psfCandidate", "calib_psfUsed", "calib_psfReserved"),
        doc = "Fields to copy from the icSource catalog to the output catalog for matching sources. "
        "Missing fields are logged (at debug level) and ignored. "
        "If detectAndMeasure.doMeasureApCorr is True and detectAndMeasure cannot determine its own "
        "suitable candidates, then this list must include "
        "config.detectAndMeasure.measureApCorr.inputFilterFlag."
    )

    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
        # measure and apply aperture correction
        self.detectAndMeasure.doMeasureApCorr = True
        self.detectAndMeasure.measurement.doApplyApCorr = "yes"


## \addtogroup LSST_task_documentation
## \{
## \page CalibrateTask
## \ref CalibrateTask_ "CalibrateTask"
## \copybrief CalibrateTask
## \}

class CalibrateTask(pipeBase.CmdLineTask):
    """!Calibrate an exposure: measure sources and perform astrometric and photometric calibration

    @anchor CalibrateTask_
    
    @section pipe_tasks_calibrate_Contents  Contents

     - @ref pipe_tasks_calibrate_Purpose
     - @ref pipe_tasks_calibrate_Initialize
     - @ref pipe_tasks_calibrate_IO
     - @ref pipe_tasks_calibrate_Config
     - @ref pipe_tasks_calibrate_Metadata
     - @ref pipe_tasks_calibrate_Debug
     - @ref pipe_tasks_calibrate_Example

    @section pipe_tasks_calibrate_Purpose  Description

    Given an exposure with a good PSF model (e.g. as provided by @ref CharacterizeImageTask),
    perform the following operations:
    - Detect sources, typically to low S/N
    - Deblend sources
    - Perform single frame measurement, typically measuring and applying aperture correction
    - Astrometric calibration:
        - match sources to objects found in a reference catalog
        - fit an improved WCS
    - Photometric calibration
        - using the matches already found, calculate the exposure's zero-point

    @section pipe_tasks_calibrate_Initialize  Task initialisation

    @copydoc \_\_init\_\_

    @section pipe_tasks_calibrate_IO  Invoking the Task

    If you want this task to unpersist inputs or persist outputs, then call
    the `run` method (a thin wrapper around the `calibrate` method).

    If you already have the inputs unpersisted and do not want to persist the output
    then it is more direct to call the `calibrate` method:

    @section pipe_tasks_calibrate_Config  Configuration parameters

    See @ref CalibrateConfig

    @section pipe_tasks_calibrate_Metadata  Quantities set in Metadata

    Task metadata
    <dl>
        <dt>MAGZERO <dd>Measured zeropoint (DN per second)
    </dl>

    Exposure metadata
    <dl>
        <dt>MAGZERO_RMS  <dd>MAGZERO's RMS == return.sigma
        <dt>MAGZERO_NOBJ <dd>Number of stars used == return.ngood
        <dt>COLORTERM1   <dd>?? (always 0.0)
        <dt>COLORTERM2   <dd>?? (always 0.0)
        <dt>COLORTERM3   <dd>?? (always 0.0)
    </dl>

    @section pipe_tasks_calibrate_Debug  Debug variables

    The @link lsst.pipe.base.cmdLineTask.CmdLineTask command line task@endlink interface supports a flag
    `--debug` to import `debug.py` from your `$PYTHONPATH`; see @ref baseDebug for more about `debug.py`.

    CalibrateTask has a debug dictionary containing one key:
    <dl>
    <dt>calibrate
    <dd>frame (an int; <= 0 to not display) in which to display the exposure, sources and matches.
        See @ref lsst.meas.astrom.displayAstrometry for the meaning of the various symbols.
    </dl>

    For example, put something like:
    @code{.py}
        import lsstDebug
        def DebugInfo(name):
            di = lsstDebug.getInfo(name)  # N.b. lsstDebug.Info(name) would call us recursively
            if name == "lsst.pipe.tasks.calibrate":
                di.display = dict(
                    calibrate = 1,
                )

            return di

        lsstDebug.Info = DebugInfo
    @endcode
    into your `debug.py` file and run `calibrateTask.py` with the `--debug` flag.

    Some subtasks may have their own debug variables; see individual Task documentation.

    @section pipe_tasks_calibrate_Example   A complete example of using CalibrateTask

    This code is in @link calibrateTask.py@endlink in the examples directory, and can be run as, e.g.:
    @code
    python examples/calibrateTask.py --display
    @endcode
    @dontinclude calibrateTask.py

    Import the task (there are some other standard imports; read the file if you're curious)
    @skipline CalibrateTask

    Create the task. Note that we're using a custom AstrometryTask (because we don't have a valid
    astrometric catalogue handy); see \ref calibrate_MyAstrometryTask.
    @skip CalibrateTask.ConfigClass
    @until config=config

    We're now ready to process the data. This occurs in two steps:
    - Characterize the image: measure bright sources, fit a background and PSF, and repairs cosmic rays
    - Calibrate the exposure: measure faint sources, fit an improved WCS and photometric zero-point

    we could loop over multiple exposures/catalogues using the same
    task objects) and optionally display the results:
    @skip loadData
    @until dot
    """
    ConfigClass = CalibrateConfig
    _DefaultName = "calibrate"

    def __init__(self, dataPrefix="", exposureIdName=None, icSourceSchema=None, **kwargs):
        """!Construct a CalibrateTask

        @param[in] dataPrefix  prefix for persisted products:
            - for calexp use the default of ""
            - for coadds use, e.g., "deepCoadd"
        @param[in] exposureIdName  name of exposure ID dataset, e.g. "ccdExposureId";
            the value None works for calexp and coadds; for other kinds of exposures you'll have to specify
        @param[in] icSourceSchema  schema for icSource catalog, or None.
            If measuring aperture correction and the task detectAndMeasure cannot determine
            its own suitable candidates, then this argument must be specified.
            See also config field `icSourceFieldsToCopy`.
        @param[in,out] kwargs  other keyword arguments for lsst.pipe.base.CmdLineTask
        """
        self.dataPrefix = str(dataPrefix)
        if exposureIdName is None:
            self.exposureIdName = dataPrefix + "Id" if dataPrefix else "ccdExposureId"
        else:
            self.exposureIdName = exposureIdName
        pipeBase.Task.__init__(self, **kwargs)

        self.haveFieldsToCopy = False
        if icSourceSchema is not None:
            # use a schema mapper to avoid copying each field separately
            self.schemaMapper = afwTable.SchemaMapper(icSourceSchema)
            self.schemaMapper.addMinimalSchema(afwTable.SourceTable.makeMinimalSchema())

            # Add fields to copy from an icSource catalog
            # and a field to indicate that the source matched a source in that catalog
            self.calibSourceKey = self.schemaMapper.addOutputField(
                afwTable.Field["Flag"]("calib_detected", "Source was detected as an icSource"))
            foundFieldNames = []
            missingFieldNames = []
            for fieldName in self.config.icSourceFieldsToCopy:
                try:
                    schemaItem = icSourceSchema.find(fieldName)
                except Exception:
                    missingFieldNames.append(fieldName)
                else: # search successful
                    self.schemaMapper.addMapping(schemaItem.getKey())
                    foundFieldNames.append(fieldName)

            if len(foundFieldNames) > 0:
                self.haveFieldsToCopy = True
            self.log.logdebug("fields found: %s and not found: %s in icSourceSchema" % \
                (foundFieldNames, missingFieldNames))

            # produce a temporary schema to pass to the subtasks; finalize it later
            self.schema=self.schemaMapper.editOutputSchema()
        else:
            self.schemaMapper = None
            self.schema = afwTable.SourceTable.makeMinimalSchema()

        self.makeSubtask("detectAndMeasure", schema=self.schema)
        if self.config.doAstrometry or self.config.doPhotoCal:
            self.makeSubtask("astrometry", schema=self.schema)
            self.loadAndMatch = self.astrometry.loadAndMatch
        else:
            self.loadAndMatch = None
        if self.config.doPhotoCal:
            self.makeSubtask("photoCal", schema=self.schema)

        if self.schemaMapper is not None:
            # finalize the schema
            self.schema = self.schemaMapper.getOutputSchema()

    @pipeBase.timeMethod
    def run(self, dataRef, exposure=None, background=None, icSourceCat=None, doUnpersist=True):
        """!Calibrate an exposure, optionally unpersisting inputs and persisting outputs.

        This is a wrapper around the `calibrate` method that unpersists inputs
        (if `doUnpersist` true) and persists outputs (if `config.doWrite` true)

        @param[in] dataRef  butler data reference corresponding to a science image
        @param[in,out] exposure  characterized exposure (an lsst.afw.image.ExposureF or similar),
            or None to unpersist existing icExp and icBackground.
            See calibrate method for details of what is read and written.
        @param[in,out] background  model of background model already subtracted from exposure
            (an lsst.afw.math.BackgroundList). May be None if no background has been subtracted,
            though that is unusual for calibration.
            A refined background model is output.
            Ignored if exposure is None.
        @param[in] icSourceCat  catalog from which to copy the fields specified by icSourceKeys, or None;
        @param[in] doUnpersist  unpersist data:
            - if True, exposure, background and icSourceCat are read from dataRef and those three arguments
                must all be None;
            - if False the exposure must be provided; background and icSourceCat are optional.
            True is intended for running as a command-line task, False for running as a subtask

        @warning until detectAndMeasure can determine suitable sources for measuring aperture correction
        by itself, you must provide icSourceCat and a suitable entry in config.icSourceFieldsToCopy

        @return same data as the calibrate method
        """
        self.log.info("Processing %s" % (dataRef.dataId))

        if doUnpersist:
            if None in (exposure, background, icSourceCat):
                raise pipeBase.TaskError("doUnpersist true; exposure, background and icSourceCat "
                    "must all be None")
            exposure = dataRef.get("icExp", immediate=True)
            background = dataRef.get("icExpBackground", immediate=True)
            icSourceCat = dataRef.get("icSourceCat", immediate=True)
        elif exposure is None:
            raise pipeBase.TaskError("doUnpersist false; exposure must be provided")

        exposureIdInfo = ExposureIdInfo.fromDataRef(dataRef, datasetName=self.exposureIdName)

        calRes = self.calibrate(
            exposure = exposure,
            exposureIdInfo = exposureIdInfo,
            background = background,
            icSourceCat = icSourceCat,
        )

        if self.config.doWrite:
            dataRef.put(calRes.sourceCat, self.dataPrefix + "src")
            if self.config.doWriteMatches:
                matches = calRes.photoMatches if calRes.photoMatches is not None else calRes.astromMatches
                if matches is not None:
                    normalizedMatches = afwTable.packMatches(matches)
                    normalizedMatches.table.setMetadata(calRes.matchMeta)
                    dataRef.put(normalizedMatches, self.dataPrefix + "srcMatch")
            dataRef.put(exposure, self.dataPrefix + "calexp")
            dataRef.put(calRes.background, self.dataPrefix + "calexpBackground")

        return calRes

    def calibrate(self, exposure, exposureIdInfo, background=None, icSourceCat=None):
        """!Calibrate an exposure (science image or coadd)

        @param[in,out] exposure  exposure to calibrate (an lsst.afw.image.ExposureF or similar);
            in:
            - MaskedImage
            - Psf
            out:
            - MaskedImage has background subtracted
            - Wcs is replaced
            - Calib is set
        @param[in,out] background  model of background model already subtracted from exposure
            (an lsst.afw.math.BackgroundList). May be None if no background has been subtracted,
            though that is unusual for calibration.
            A refined background model is output.

        @return pipe_base Struct containing these fields:
        - exposure  calibrate science exposure with refined WCS and Calib
        - background  model of background subtracted from exposure (an lsst.afw.math.BackgroundList)
        - sourceCat  table of measured sources
        - astromMatches  list of source/refObj matches from the astrometry solver
        - photoMatches  list of source/refObj matches from the photometry solver
        - matchMeta  metadata about the field (an lsst.daf.base.PropertyList);
            used to persist and unpersist match lists
        """
        # detect, deblend and measure sources
        if icSourceCat is not None and self.haveFieldsToCopy:
            def copyFields(sourceCat):
                self.copyIcSourceFields(icSourceCat=icSourceCat, sourceCat=sourceCat)
        else:
            copyFields = None

        procRes = self.detectAndMeasure.run(
            exposure = exposure,
            exposureIdInfo = exposureIdInfo,
            background = background,
            preMeasApCorrFunc = copyFields,
        )

        # perform astrometry calibration:
        # fit an improved WCS and update the exposure's WCS in place
        astromRes = pipeBase.Struct(
            matches = None,
            matchMeta = None,
        )
        if self.config.doAstrometry:
            try:
                astromRes = self.astrometry.run(
                    exposure = exposure,
                    sourceCat = procRes.sourceCat,
                )
            except Exception as e:
                if self.config.requireAstrometry:
                    raise
                self.log.warn("Unable to perform astrometric calibration (%s): attempting to proceed" % e)

        # compute photometric calibration
        photoRes = pipeBase.Struct(
            matches = None,
            matchMeta = None,
        )
        if self.config.doPhotoCal:
            try:
                if astromRes.matches is None:
                    astromRes = self.loadAndMatch(exposure=exposure, sourceCat=procRes.sourceCat)
                photoRes = self.photoCal.run(exposure, astromRes.matches)
                exposure.getCalib().setFluxMag0(photoRes.calib.getFluxMag0())
                self.log.info("Photometric zero-point: %f" % photoRes.calib.getMagnitude(1.0))
                self.setMetadata(exposure=exposure, photoRes=photoRes)
            except Exception as e:
                if self.config.requirePhotoCal:
                    raise
                self.log.warn("Unable to perform photometric calibration (%s): attempting to proceed" % e)
                self.setMetadata(exposure=exposure, photoRes=None)

        frame = getDebugFrame(self._display, "calibrate")
        if frame:
            displayAstrometry(
                sourceCat = procRes.sourceCat,
                exposure = exposure,
                matches = photoRes.matches,
                frame = frame,
                pause = False,
            )

        return pipeBase.Struct(
            exposure = exposure,
            background = procRes.background,
            sourceCat = procRes.sourceCat,
            astromMatches = astromRes.matches,
            photoMatches = photoRes.matches,
            matchMeta = astromRes.matchMeta if astromRes.matchMeta is not None else photoRes.matchMeta,
        )

    def getSchemaCatalogs(self):
        """Return a dict of empty catalogs for each catalog dataset produced by this task.
        """
        sourceCat = afwTable.SourceCatalog(self.schema)
        sourceCat.getTable().setMetadata(self.detectAndMeasure.algMetadata)
        return {self.dataPrefix + "src": sourceCat}

    def setMetadata(self, exposure, photoRes=None):
        """!Set task and exposure metadata

        Logs a warning and continues if needed data is missing.

        @param[in,out] exposure  exposure whose metadata is to be set
        @param[in]  photoRes  results of running photoCal; if None then it was not run
        """
        if photoRes is None:
            self.metadata.set('MAGZERO', float("NaN"))
            return

        # convert zero-point to (mag/sec/adu) for task MAGZERO metadata
        try:
            magZero = photoRes.zp - 2.5*math.log10(exposure.getCalib().getExptime())
            self.metadata.set('MAGZERO', magZero)
        except Exception:
            self.log.warn("Could not set normalized MAGZERO in header: no exposure time")

        try:
            metadata = exposure.getMetadata()
            metadata.set('MAGZERO_RMS', photoRes.sigma)
            metadata.set('MAGZERO_NOBJ', photoRes.ngood)
            metadata.set('COLORTERM1', 0.0)
            metadata.set('COLORTERM2', 0.0)
            metadata.set('COLORTERM3', 0.0)
        except Exception as e:
            self.log.warn("Could not set exposure metadata: %s" % (e,))

    def copyIcSourceFields(self, icSourceCat, sourceCat, matchRadius=3):
        """!Match sources in icSourceCat and sourceCat and copy the specified fields

        @param[in] icSourceCat  catalog from which to copy fields
        @param[in,out] sourceCat  catalog to which to copy fields
        @param[in] matchRadius  match radius for sources (pixels)

        The fields to be copied are a subset of `config.icSourceFieldsToCopy`
        """
        self.log.info("Matching icSource and Source catalogs to propagate flags.")
        if self.schemaMapper is None:
            raise RuntimeError("To copy icSource fields you must specify icSourceSchema "
                "and icSourceKeys when constructing this task")
        if icSourceCat is None or sourceCat is None:
            raise RuntimeError("icSourceCat and sourceCat must both be specified")
        if not self.haveFieldsToCopy:
            self.log.logdebug("copyIcSourceFields doing nothing because there are no fields to copy")
            return

        closest = False  # return all matched objects
        matched = afwTable.matchXy(icSourceCat, sourceCat, matchRadius, closest)
        if self.config.detectAndMeasure.doDeblend:
            matched = [m for m in matched if m[1].get("deblend_nChild") == 0] # if deblended, keep children

        # Because we had to allow multiple matches to handle parents, we now need to
        # prune to the best matches
        bestMatches = {}
        for m0, m1, d in matched:
            id0 = m0.getId()
            if bestMatches.has_key(id0):
                if d > bestMatches[id0][2]:
                    continue

            bestMatches[id0] = (m0, m1, d)

        matched = bestMatches.values()

        # Check that we got it right
        if len(set(m[0].getId() for m in matched)) != len(matched):
            self.log.warn("At least one icSourceCat source is matched to more than one sourceCat source")

        self.log.info("Matched %s sources" % (len(matched),))

        # Copy over the desired fields
        for ics, s, d in matched:
            s.setFlag(self.calibSourceKey, True)
            # We don't want to overwrite s's footprint with ics's; DM-407
            icsFootprint = ics.getFootprint()
            try:
                ics.setFootprint(s.getFootprint())
                s.assign(ics, self.schemaMapper)
            finally:
                ics.setFootprint(icsFootprint)
