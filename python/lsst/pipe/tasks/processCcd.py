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
from .processImage import ProcessImageTask
from lsst.ip.isr import IsrTask
from lsst.pipe.tasks.calibrate import CalibrateTask
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

class ProcessCcdConfig(ProcessImageTask.ConfigClass):
    """Config for ProcessCcd"""
    doIsr = pexConfig.Field(dtype=bool, default=True, doc = "Perform ISR?")
    isr = pexConfig.ConfigurableField(
        target = IsrTask,
        doc = "Instrumental Signature Removal",
    )
    doCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Perform calibration?")
    doWriteCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Write calibration results?")
    doWriteCalibrateMatches = pexConfig.Field(dtype=bool, default=True,
                                              doc = "Write icSrc to reference matches?")
    calibrate = pexConfig.ConfigurableField(
        target = CalibrateTask,
        doc = "Calibration (inc. high-threshold detection and measurement)",
    )

class ProcessCcdTask(ProcessImageTask):
    """Process a CCD
    
    Available steps include:
    - instrument signature removal (ISR)
    - calibrate
    - detect sources
    - measure sources
    """
    ConfigClass = ProcessCcdConfig
    _DefaultName = "processCcd"
    dataPrefix = ""

    def __init__(self, **kwargs):
        ProcessImageTask.__init__(self, **kwargs)
        self.makeSubtask("isr")
        self.makeSubtask("calibrate")

        # Setup our schema by starting with fields we want to propagate from icSrc.
        calibSchema = self.calibrate.schema
        self.schemaMapper = afwTable.SchemaMapper(calibSchema, self.schema)

        # Add fields needed to identify stars used in the calibration step
        self.calibSourceKey = self.schemaMapper.addOutputField(
            afwTable.Field["Flag"]("calib.detected", "Source was detected as an icSrc")
            )
        for key in self.calibrate.getCalibKeys():
            self.schemaMapper.addMapping(key)

    def makeIdFactory(self, sensorRef):
        expBits = sensorRef.get("ccdExposureId_bits")
        expId = self.getExpId(sensorRef)
        return afwTable.IdFactory.makeSource(expId, 64 - expBits)        

    def getExpId(self, dataRef):
        return long(dataRef.get("ccdExposureId", immediate=True))

    def getAstrometer(self):
        return self.calibrate.astrometry.astrometer

    @pipeBase.timeMethod
    def run(self, sensorRef):
        """Process one CCD
        
        @param sensorRef: sensor-level butler data reference
        @return pipe_base Struct containing these fields:
        - postIsrExposure: exposure after ISR performed if calib.doIsr or config.doCalibrate, else None
        - exposure: calibrated exposure (calexp): as computed if config.doCalibrate,
            else as upersisted and updated if config.doDetection, else None
        - calib: object returned by calibration process if config.doCalibrate, else None
        - sources: detected source if config.doPhotometry, else None
        """
        self.log.info("Processing %s" % (sensorRef.dataId))

        # initialize outputs
        postIsrExposure = None
        
        if self.config.doIsr:
            postIsrExposure = self.isr.run(sensorRef).exposure
        elif self.config.doCalibrate:
            postIsrExposure = sensorRef.get(self.dataPrefix + "postISRCCD")
        
        # initialize outputs
        calib = None
        sources = None
        backgrounds = afwMath.BackgroundList()
        if self.config.doCalibrate:
            idFactory = self.makeIdFactory(sensorRef)
            calib = self.calibrate.run(postIsrExposure, idFactory=idFactory, expId=self.getExpId(sensorRef))
            calExposure = calib.exposure
            if self.config.doWriteCalibrate:
                sensorRef.put(calib.sources, self.dataPrefix + "icSrc")
                if calib.matches is not None and self.config.doWriteCalibrateMatches:
                    normalizedMatches = afwTable.packMatches(calib.matches)
                    normalizedMatches.table.setMetadata(calib.matchMeta)
                    sensorRef.put(normalizedMatches, self.dataPrefix + "icMatch")
            try:
                for bg in calib.backgrounds:
                    backgrounds.append(bg)
            except TypeError:
                backgrounds.append(calib.backgrounds)
            except AttributeError:
                self.log.warn("The calibration task did not return any backgrounds. " +
                    "Any background subtracted in the calibration process cannot be persisted.")
        elif sensorRef.datasetExists("calexp"):
            calExposure = sensorRef.get("calexp", immediate=True)
        else:
            raise RuntimeError("No calibrated exposure available for processing")

        # delegate most of the work to ProcessImageTask
        result = self.process(sensorRef, calExposure)

        # combine the differential background we estimated while detecting the main src catalog
        # with the background estimated in the calibrate step
        for bg in result.backgrounds:
            backgrounds.append(bg)
        result.backgrounds = backgrounds

        if self.config.doCalibrate and self.config.doWriteCalibrate:
            # wait until after detection and measurement, since detection sets detected mask bits
            # and both require a background subtracted exposure;
            if self.config.persistBackgroundModel:
                self.writeBackgrounds(sensorRef, backgrounds)
            else:
                self.restoreBackgrounds(calExposure, backgrounds)
            sensorRef.put(calExposure, self.dataPrefix + "calexp")

        if calib is not None:
            self.propagateCalibFlags(calib.sources, sources)

        return pipeBase.Struct(
            postIsrExposure = postIsrExposure,
            calib = calib,
            **result.getDict()
        )


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

    def getSchemaCatalogs(self):
        """Return a dict of empty catalogs for each catalog dataset produced by this task."""
        d = ProcessImageTask.getSchemaCatalogs(self)
        icSrc = None
        icSrc = afwTable.SourceCatalog(self.calibrate.schema)
        icSrc.getTable().setMetadata(self.calibrate.algMetadata)
        d[self.dataPrefix + "icSrc"] = icSrc
        return d
