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
import lsst.pex.exceptions as pexExceptions
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
from lsst.meas.algorithms import SourceDetectionTask, SourceMeasurementTask
from lsst.meas.deblender import SourceDeblendTask
from lsst.ip.isr import IsrTask
from lsst.pipe.tasks.calibrate import CalibrateTask

class ProcessImageConfig(pexConfig.Config):
    """Config for ProcessImage"""
    doCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Perform calibration?")
    doDetection = pexConfig.Field(dtype=bool, default=True, doc = "Detect sources?")
    ## NOTE, default this to False until it is fully vetted; #2138
    doDeblend = pexConfig.Field(dtype=bool, default=False, doc = "Deblend sources?")
    doMeasurement = pexConfig.Field(dtype=bool, default=True, doc = "Measure sources?")
    doWriteCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Write calibration results?")
    persistBackgroundModel = pexConfig.Field(dtype=bool, default=True, doc = "If True persist background model with background subtracted calexp.  \
        If False persist calexp with the background included.")
    doWriteCalibrateMatches = pexConfig.Field(dtype=bool, default=True,
                                              doc = "Write icSrc to reference matches?")
    doWriteSources = pexConfig.Field(dtype=bool, default=True, doc = "Write sources?")
    doWriteSourceMatches = pexConfig.Field(dtype=bool, default=False,
                                           doc = "Compute and write src to reference matches?")
    doWriteHeavyFootprintsInSources = pexConfig.Field(dtype=bool, default=False,
                                                      doc = "Include HeavyFootprint data in source table?")
    calibrate = pexConfig.ConfigurableField(
        target = CalibrateTask,
        doc = "Calibration (inc. high-threshold detection and measurement)",
    )
    detection = pexConfig.ConfigurableField(
        target = SourceDetectionTask,
        doc = "Low-threshold detection for final measurement",
    )
    deblend = pexConfig.ConfigurableField(
        target = SourceDeblendTask,
        doc = "Split blended sources into their components",
    )
    measurement = pexConfig.ConfigurableField(
        target = SourceMeasurementTask,
        doc = "Final source measurement on low-threshold detections",
    )

    def validate(self):
        pexConfig.Config.validate(self)
        if self.doMeasurement:
            if not self.doDetection:
                raise ValueError("Cannot run source measurement without source detection.")
            if "skycoord" not in self.measurement.algorithms.names:
                raise ValueError("If you run source measurement you must let it run the skycoord algorithm.")
        if self.doDeblend and not self.doDetection:
            raise ValueError("Cannot run source deblending without source detection.")
        if self.doWriteHeavyFootprintsInSources and not self.doWriteSources:
            raise ValueError("Cannot write HeavyFootprints (doWriteHeavyFootprintsInSources) without doWriteSources")

    def setDefaults(self):
        #
        # Stop flux.gaussian recomputing the Gaussian's weights (as shape.sdss already did that)
        #
        try:
            self.measurement.algorithms['flux.gaussian'].fixed = True
        except pexConfig.FieldValidationError: # "flux.gaussian" isn't there
            pass

class ProcessImageTask(pipeBase.CmdLineTask):
    """An abstract base class for tasks do simple calibration, detection, deblending, and measurement
    on individual images.

    Other command-line Process***Tasks (such as ProcessCcdTask and ProcessCoaddTask) rely on
    ProcessImageTask for their main algorithmic code, and only need to add pre- and post- processing
    and serialization.

    Subclasses are responsible for meeting the requirements of CmdLineTask.
    """
    ConfigClass = ProcessImageConfig

    dataPrefix = ""  # Name to prepend to all input and output datasets (e.g. 'goodSeeingCoadd_')

    def __init__(self, **kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.makeSubtask("calibrate")

        # Setup our schema by starting with fields we want to propagate from icSrc.
        calibSchema = self.calibrate.schema
        self.schemaMapper = afwTable.SchemaMapper(calibSchema)
        self.schemaMapper.addMinimalSchema(afwTable.SourceTable.makeMinimalSchema(), False)

        # Add fields needed to identify stars used in the calibration step
        self.calibSourceKey = self.schemaMapper.addOutputField(
            afwTable.Field["Flag"]("calib.detected", "Source was detected as an icSrc")
            )
        for key in self.calibrate.getCalibKeys():
            self.schemaMapper.addMapping(key)
        self.schema = self.schemaMapper.getOutputSchema()
        
        self.algMetadata = dafBase.PropertyList()
        if self.config.doDetection:
            self.makeSubtask("detection", schema=self.schema)
        if self.config.doDeblend:
            self.makeSubtask("deblend", schema=self.schema)
        if self.config.doMeasurement:
            self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)

    def makeIdFactory(self, sensorRef):
        raise NotImplementedError()

    @pipeBase.timeMethod
    def process(self, dataRef, inputExposure, enableWriteSources=True):
        """Process an Image
        
        @param dataRef: data reference that corresponds to the input image
        @param inputExposure:  exposure to process
        @param enableWriteSources: if True then writing sources is allowed.
            Set False if you need to postprocess sources before writing them.

        @return pipe_base Struct containing these fields:
        - postIsrExposure: exposure after ISR performed if calib.doIsr or config.doCalibrate, else None
        - exposure: calibrated exposure (calexp): as computed if config.doCalibrate,
            else as upersisted and updated if config.doDetection, else None
        - calib: object returned by calibration process if config.doCalibrate, else None
        - sources: detected source if config.doPhotometry, else None
        """
        idFactory = self.makeIdFactory(dataRef)
        expId = self.getExpId(dataRef)

        # initialize outputs
        calExposure = inputExposure
        calib = None
        sources = None
        backgrounds = afwMath.BackgroundList()
        if self.config.doCalibrate:
            calib = self.calibrate.run(inputExposure, idFactory=idFactory, expId=expId)
            calExposure = calib.exposure
            if self.config.doWriteCalibrate:
                dataRef.put(calib.sources, self.dataPrefix + "icSrc")
                if calib.matches is not None and self.config.doWriteCalibrateMatches:
                    normalizedMatches = afwTable.packMatches(calib.matches)
                    normalizedMatches.table.setMetadata(calib.matchMeta)
                    dataRef.put(normalizedMatches, self.dataPrefix + "icMatch")
            try:
                for bg in calib.backgrounds:
                    backgrounds.append(bg)
            except TypeError:     
                backgrounds.append(calib.backgrounds)
            except AttributeError:
                self.log.warn("The calibration task did not return any backgrounds. " +
                    "Any background subtracted in the calibration process cannot be persisted.")
        else:
            calib = None

        if self.config.doDetection:
            table = afwTable.SourceTable.make(self.schema, idFactory)
            table.setMetadata(self.algMetadata)
            detections = self.detection.makeSourceCatalog(table, calExposure)
            sources = detections.sources
            fpSets = detections.fpSets
            if fpSets.background:           
                backgrounds.append(fpSets.background)

        if self.config.doDeblend:
            self.deblend.run(calExposure, sources, calExposure.getPsf())

        if self.config.doMeasurement:
            self.measurement.run(calExposure, sources)

        if self.config.doWriteCalibrate:
            # wait until after detection and measurement, since detection sets detected mask bits
            # and both require a background subtracted exposure;
            # note that this overwrites an existing calexp if doCalibrate false
               
            if calExposure is None:
                self.log.warn("calibrated exposure is None; cannot save it")
            else:
                if self.config.persistBackgroundModel:
                    self.writeBackgrounds(dataRef, backgrounds)
                else:
                    self.restoreBackgrounds(calExposure, backgrounds)
                dataRef.put(calExposure, self.dataPrefix + "calexp")

        if calib is not None:
            self.propagateCalibFlags(calib.sources, sources)

        if sources is not None and self.config.doWriteSources:
            sourceWriteFlags = (0 if self.config.doWriteHeavyFootprintsInSources
                                else afwTable.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
            if enableWriteSources:
                dataRef.put(sources, self.dataPrefix + 'src', flags=sourceWriteFlags)
            
        if self.config.doMeasurement and self.config.doWriteSourceMatches:
            self.log.info("Matching src to reference catalogue" % (dataRef.dataId))
            srcMatches, srcMatchMeta = self.matchSources(calExposure, sources)

            if srcMatches is not None:
                normalizedSrcMatches = afwTable.packMatches(srcMatches)
                normalizedSrcMatches.table.setMetadata(srcMatchMeta)
                dataRef.put(normalizedSrcMatches, self.dataPrefix + "srcMatch")
        else:
            srcMatches = None; srcMatchMeta = None

        return pipeBase.Struct(
            inputExposure = inputExposure,
            exposure = calExposure,
            calib = calib,
            sources = sources,
            matches = srcMatches,
            matchMeta = srcMatchMeta,
            backgrounds = backgrounds,
        )

    def matchSources(self, exposure, sources):
        """Match the sources to the reference object loaded by the calibrate task"""
        try:
            astrometer = self.calibrate.astrometry.astrometer
            if astrometer is None:
                raise AttributeError("No astrometer")
        except AttributeError:
            self.log.warn("Failed to find an astrometer in calibrate's astronomy task")
            return None, None

        astromRet = astrometer.useKnownWcs(sources, exposure=exposure)
        # N.b. yes, this is what useKnownWcs calls the returned values
        return astromRet.matches, astromRet.matchMetadata

    def propagateCalibFlags(self, icSources, sources, matchRadius=1):
        """Match the icSources and sources, and propagate Interesting Flags (e.g. PSF star) to the sources
        """
        self.log.info("Matching icSource and Source catalogs to propagate flags.")
        if icSources is None or sources is None:
            return

        closest = False                 # return all matched objects
        matched = afwTable.matchRaDec(icSources, sources, matchRadius*afwGeom.arcseconds, closest)
        if self.config.doDeblend:
            matched = [m for m in matched if m[1].get("deblend.nchild") == 0] # if deblended, keep children
        #
        # Because we had to allow multiple matches to handle parents, we now need to
        # prune to the best matches
        #
        bestMatches = {}
        for m0, m1, d in matched:
            id0 = m0.getId()
            if bestMatches.has_key(id0):
                if d > bestMatches[id0][2]:
                    continue

            bestMatches[id0] = (m0, m1, d)

        matched = bestMatches.values()
        #
        # Check that we got it right
        #
        if len(set(m[0].getId() for m in matched)) != len(matched):
            self.log.warn("At least one icSource is matched to more than one Source")
        #
        # Copy over the desired flags
        #
        for ics, s, d in matched:
            s.setFlag(self.calibSourceKey, True)
            # We don't want to overwrite s's footprint with ics's; DM-407
            icsFootprint = ics.getFootprint()
            try:
                ics.setFootprint(s.getFootprint())
                s.assign(ics, self.schemaMapper)
            finally:
                ics.setFootprint(icsFootprint)

        return

    def getSchemaCatalogs(self):
        """Return a dict of empty catalogs for each catalog dataset produced by this task."""
        src = afwTable.SourceCatalog(self.schema)
        src.getTable().setMetadata(self.algMetadata)
        d = {self.dataPrefix + "src": src}
        icSrc = None
        try:
            icSrc = afwTable.SourceCatalog(self.calibrate.schema)
            icSrc.getTable().setMetadata(self.calibrate.algMetadata)
        except AttributeError:
            pass
        if icSrc is not None:
            d[self.dataPrefix + "icSrc"] = icSrc
        return d

    def writeBackgrounds(self, dataRef, backgrounds):
        """Backgrounds are persisted via the butler

        @param dataRef: Data reference
        @param backgrounds: List of background models
        """
        self.log.warn("Persisting background models")
        
        dataRef.put(backgrounds, self.dataPrefix+"calexpBackground")

    def restoreBackgrounds(self, exp, backgrounds):
        """Add backgrounds back in to an exposure

        @param exp: Exposure to which to add backgrounds
        @param backgrounds: List of background models
        """
        mi = exp.getMaskedImage()
        mi += backgrounds.getImage()

    def getExpId(self, dataRef):
        raise NotImplementedError()

