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
import lsst.daf.base as dafBase
import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom
from lsst.meas.algorithms import SourceDetectionTask, SourceMeasurementTask, SourceDeblendTask
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
        if self.doMeasurement and not self.doDetection:
            raise ValueError("Cannot run source measurement without source detection.")
        if self.doDeblend and not self.doDetection:
            raise ValueError("Cannot run source deblending without source detection.")
        if self.doWriteHeavyFootprintsInSources and not self.doWriteSources:
            raise ValueError("Cannot write HeavyFootprints (doWriteHeavyFootprintsInSources) without doWriteSources")

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
        self.schema = kwargs.pop("schema", None)
        if self.schema is None:
            self.schema = afwTable.SourceTable.makeMinimalSchema()
        # add fields needed to identify stars used in the calibration step
        self.calibSourceKey = self.schema.addField("calib.referenceSource",
                                                   type="Flag", doc="Source was detected as an icSrc")
        self.psfStarKey = self.schema.addField("calib.psfStar",
                                               type="Flag", doc="Source was used to determine the PSF")

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
    def process(self, dataRef, inputExposure):
        """Process an Image
        
        @param dataRef: data reference that corresponds to the input image
        @param inputExposure:  exposure to process

        @return pipe_base Struct containing these fields:
        - postIsrExposure: exposure after ISR performed if calib.doIsr or config.doCalibrate, else None
        - exposure: calibrated exposure (calexp): as computed if config.doCalibrate,
            else as upersisted and updated if config.doDetection, else None
        - calib: object returned by calibration process if config.doCalibrate, else None
        - apCorr: aperture correction: as computed config.doCalibrate, else as unpersisted
            if config.doMeasure, else None
        - sources: detected source if config.doPhotometry, else None
        """
        idFactory = self.makeIdFactory(dataRef)

        # initialize outputs
        calExposure = None
        calib = None
        apCorr = None
        sources = None
        psf = None
        
        if self.config.doCalibrate:
            calib = self.calibrate.run(inputExposure, idFactory=idFactory)
            psf = calib.psf
            calExposure = calib.exposure
            apCorr = calib.apCorr
            if self.config.doWriteCalibrate:
                dataRef.put(calib.sources, self.dataPrefix + "icSrc")
                if calib.psf is not None:
                    dataRef.put(calib.psf, self.dataPrefix + "psf")
                if calib.apCorr is not None:
                    dataRef.put(calib.apCorr, self.dataPrefix + "apCorr")
                if calib.matches is not None and self.config.doWriteCalibrateMatches:
                    normalizedMatches = afwTable.packMatches(calib.matches)
                    normalizedMatches.table.setMetadata(calib.matchMeta)
                    dataRef.put(normalizedMatches, self.dataPrefix + "icMatch")
        else:
            calib = None

        if self.config.doDetection:
            if calExposure is None:
                if not dataRef.datasetExists(self.dataPrefix + "calexp"):
                    raise pipeBase.TaskError("doCalibrate false, doDetection true and calexp does not exist")
                calExposure = dataRef.get(self.dataPrefix + "calexp")
            if calib is None or calib.psf is None:
                psf = dataRef.get(self.dataPrefix + "psf")
                calExposure.setPsf(psf)
            table = afwTable.SourceTable.make(self.schema, idFactory)
            table.setMetadata(self.algMetadata)
            sources = self.detection.makeSourceCatalog(table, calExposure).sources

        if self.config.doWriteCalibrate:
            # wait until after detection, since that sets detected mask bits and may tweak the background;
            # note that this overwrites an existing calexp if doCalibrate false
            if calExposure is None:
                self.log.log(self.log.WARN, "calibrated exposure is None; cannot save it")
            else:
                dataRef.put(calExposure, self.dataPrefix + "calexp")

        if self.config.doDeblend:
            if calExposure is None:
                calExposure = dataRef.get(self.dataPrefix + 'calexp')
            if psf is None:
                psf = dataRef.get(self.dataPrefix + 'psf')

            self.deblend.run(calExposure, sources, psf)

        if self.config.doMeasurement:
            if apCorr is None:
                apCorr = dataRef.get(self.dataPrefix + "apCorr")
            self.measurement.run(calExposure, sources, apCorr)

        if calib is not None:
            self.propagateIcFlags(calib.sources, sources)

        if sources is not None and self.config.doWriteSources:
            if self.config.doWriteHeavyFootprintsInSources:
                sources.setWriteHeavyFootprints(True)
            dataRef.put(sources, self.dataPrefix + 'src')
            
        if self.config.doWriteSourceMatches:
            self.log.log(self.log.INFO, "Matching src to reference catalogue" % (dataRef.dataId))
            srcMatches, srcMatchMeta = self.matchSources(calExposure, sources)

            normalizedSrcMatches = afwTable.packMatches(srcMatches)
            normalizedSrcMatches.table.setMetadata(srcMatchMeta)
            dataRef.put(normalizedSrcMatches, self.dataPrefix + "srcMatch")
        else:
            srcMatches = None; srcMatchMeta = None

        return pipeBase.Struct(
            inputExposure = inputExposure,
            exposure = calExposure,
            calib = calib,
            apCorr = apCorr,
            sources = sources,
            matches = srcMatches,
            matchMeta = srcMatchMeta,
        )

    def matchSources(self, exposure, sources):
        """Match the sources to the reference object loaded by the calibrate task"""
        try:
            astrometer = self.calibrate.astrometry.astrometer
        except AttributeError:
            self.log.log(self.log.WARN, "Failed to find an astrometer in calibrate's astronomy task")
            return None, None

        astromRet = astrometer.useKnownWcs(sources, exposure=exposure)
        # N.b. yes, this is what useKnownWcs calls the returned values
        return astromRet.matches, astromRet.matchMetadata

    def propagateIcFlags(self, icSources, sources, matchRadius=1):
        """Match the icSources and sources, and propagate Interesting Flags (e.g. PSF star) to the sources
        """
        if icSources is None or sources is None:
            return

        closest = False                 # return all matched objects
        matched = afwTable.matchRaDec(icSources, sources, matchRadius*afwGeom.arcseconds, closest)
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
            self.log.log(self.log.WARN, "At least one icSource is matched to more than one Source")
        #
        # Copy over the desired flags
        #
        psfStarKey_ic = icSources.getSchema().find("classification.psfstar").getKey()
        #
        # Actually set flags in sources
        #
        for ics, s, d in matched:
            s.set(self.calibSourceKey, True)
            s.set(self.psfStarKey, ics.get(psfStarKey_ic))
        return
    
