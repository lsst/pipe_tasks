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

class ProcessCcdConfig(pexConfig.Config):
    """Config for ProcessCcd"""
    doIsr = pexConfig.Field(dtype=bool, default=True, doc = "Perform ISR?")
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
    isr = pexConfig.ConfigurableField(
        target = IsrTask,
        doc = "Instrumental Signature Removal",
    )
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

class ProcessCcdTask(pipeBase.CmdLineTask):
    """Process a CCD
    
    Available steps include:
    - instrument signature removal (ISR)
    - calibrate
    - detect sources
    - measure sources
    """
    ConfigClass = ProcessCcdConfig
    _DefaultName = "processCcd"

    def __init__(self, **kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.makeSubtask("isr")
        self.makeSubtask("calibrate")
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        # add fields needed to identify stars used in the calibration step
        self.calibSourceKey = self.schema.addField("calib.referenceSource", type="Flag",
                                                   doc="Source was detected as an icSrc")
        self.psfStarCandidateKey = self.schema.addField("calib.psfStarCandidate", type="Flag",
                                                        doc="Source was a candidate to determine the PSF")
        self.psfStarKey = self.schema.addField("calib.psfStar", type="Flag",
                                               doc="Source was used to determine the PSF")

        self.algMetadata = dafBase.PropertyList()
        if self.config.doDetection:
            self.makeSubtask("detection", schema=self.schema)
        if self.config.doDeblend:
            self.makeSubtask("deblend", schema=self.schema)
        if self.config.doMeasurement:
            self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)

    @pipeBase.timeMethod
    def run(self, sensorRef, sources=None):
        """Process one CCD
        
        @param sensorRef: sensor-level butler data reference
        @return pipe_base Struct containing these fields:
        - postIsrExposure: exposure after ISR performed if calib.doIsr or config.doCalibrate, else None
        - exposure: calibrated exposure (calexp): as computed if config.doCalibrate,
            else as upersisted and updated if config.doDetection, else None
        - calib: object returned by calibration process if config.doCalibrate, else None
        - apCorr: aperture correction: as computed config.doCalibrate, else as unpersisted
            if config.doMeasure, else None
        - sources: detected source if config.doPhotometry, else None
        """
        self.log.log(self.log.INFO, "Processing %s" % (sensorRef.dataId))

        # We make one IdFactory that will be used by both icSrc and src datasets;
        # I don't know if this is the way we ultimately want to do things, but at least
        # this ensures the source IDs are fully unique.
        expBits = sensorRef.get("ccdExposureId_bits")
        expId = long(sensorRef.get("ccdExposureId"))
        idFactory = afwTable.IdFactory.makeSource(expId, 64 - expBits)

        # initialize outputs
        postIsrExposure = None
        calExposure = None
        calib = None
        apCorr = None
        sources = None
        psf = None
        
        if self.config.doIsr:
            postIsrExposure = self.isr.run(sensorRef).exposure

        if self.config.doCalibrate:
            if postIsrExposure is None:
                postIsrExposure = sensorRef.get("postISRCCD")
            calib = self.calibrate.run(postIsrExposure, idFactory=idFactory)
            psf = calib.psf
            calExposure = calib.exposure
            apCorr = calib.apCorr
            if self.config.doWriteCalibrate:
                sensorRef.put(calib.sources, "icSrc")
                if calib.psf is not None:
                    sensorRef.put(calib.psf, "psf")
                if calib.apCorr is not None:
                    sensorRef.put(calib.apCorr, "apCorr")
                if calib.matches is not None and self.config.doWriteCalibrateMatches:
                    normalizedMatches = afwTable.packMatches(calib.matches)
                    normalizedMatches.table.setMetadata(calib.matchMeta)
                    sensorRef.put(normalizedMatches, "icMatch")
        else:
            calib = None

        if self.config.doDetection:
            if calExposure is None:
                if not sensorRef.datasetExists("calexp"):
                    raise pipeBase.TaskError("doCalibrate false, doDetection true and calexp does not exist")
                calExposure = sensorRef.get("calexp")
            if calib is None or calib.psf is None:
                psf = sensorRef.get("psf")
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
                sensorRef.put(calExposure, "calexp")

        if self.config.doDeblend:
            if calExposure is None:
                calExposure = sensorRef.get('calexp')
            if psf is None:
                psf = sensorRef.get('psf')

            self.deblend.run(calExposure, sources, psf)

        if self.config.doMeasurement:
            if apCorr is None:
                apCorr = sensorRef.get("apCorr")
            self.measurement.run(calExposure, sources, apCorr)

        if calib is not None:
            self.propagateIcFlags(calib.sources, sources)

        if sources is not None and self.config.doWriteSources:
            if self.config.doWriteHeavyFootprintsInSources:
                sources.setWriteHeavyFootprints(True)
            sensorRef.put(sources, 'src')
            
        if self.config.doWriteSourceMatches:
            self.log.log(self.log.INFO, "Matching src to reference catalogue" % (sensorRef.dataId))
            srcMatches, srcMatchMeta = self.matchSources(calExposure, sources)

            normalizedSrcMatches = afwTable.packMatches(srcMatches)
            normalizedSrcMatches.table.setMetadata(srcMatchMeta)
            sensorRef.put(normalizedSrcMatches, "srcMatch")
        else:
            srcMatches = None; srcMatchMeta = None

        return pipeBase.Struct(
            postIsrExposure = postIsrExposure,
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
        psfStarCandidate_ic = icSources.getSchema().find("psfStarCandidate").getKey()
        #
        # Actually set flags in sources
        #
        for ics, s, d in matched:
            s.set(self.calibSourceKey, True)
            s.set(self.psfStarCandidateKey, ics.get(psfStarCandidate_ic))
            s.set(self.psfStarKey, ics.get(psfStarKey_ic))
        return
    
