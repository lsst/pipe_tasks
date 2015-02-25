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
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
from lsst.meas.algorithms import SourceDetectionTask, SourceMeasurementTask
from lsst.meas.deblender import SourceDeblendTask

class ProcessImageConfig(pexConfig.Config):
    """Config for ProcessImage"""
    doDetection = pexConfig.Field(dtype=bool, default=True, doc = "Detect sources?")
    ## NOTE, default this to False until it is fully vetted; #2138
    doDeblend = pexConfig.Field(dtype=bool, default=False, doc = "Deblend sources?")
    doMeasurement = pexConfig.Field(dtype=bool, default=True, doc = "Measure sources?")
    persistBackgroundModel = pexConfig.Field(dtype=bool, default=True, doc = "If True persist background model with background subtracted calexp.  \
        If False persist calexp with the background included.")
    doWriteSources = pexConfig.Field(dtype=bool, default=True, doc = "Write sources?")
    doWriteSourceMatches = pexConfig.Field(dtype=bool, default=False,
                                           doc = "Compute and write src to reference matches?")
    doWriteHeavyFootprintsInSources = pexConfig.Field(dtype=bool, default=False,
                                                      doc = "Include HeavyFootprint data in source table?")
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
    """An abstract base class for tasks to do detection, deblending, and measurement
    on individual images.

    Other command-line Process***Tasks (such as ProcessCcdTask and ProcessCoaddTask) rely on
    ProcessImageTask for their main algorithmic code, and only need to add pre- and post- processing
    and serialization.

    Subclasses are responsible for meeting the requirements of CmdLineTask.
    """
    ConfigClass = ProcessImageConfig

    dataPrefix = ""  # Name to prepend to all input and output datasets (e.g. 'goodSeeingCoadd_')

    def __init__(self, schema=None, **kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.algMetadata = dafBase.PropertyList()
        if schema is None:
            schema = afwTable.SourceTable.makeMinimalSchema()
        self.schema = schema
        if self.config.doDetection:
            self.makeSubtask("detection", schema=self.schema)
        if self.config.doDeblend:
            self.makeSubtask("deblend", schema=self.schema)
        if self.config.doMeasurement:
            self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)

    def makeIdFactory(self, sensorRef):
        raise NotImplementedError()

    def getAstrometer(self):
        """Return an astrometer for matching sources to the reference catalog"""
        raise NotImplementedError("Not implemented in base class")

    @pipeBase.timeMethod
    def process(self, dataRef, inputExposure, backgrounds=None, enableWriteSources=True):
        """Process an Image
        
        @param dataRef: data reference that corresponds to the input image
        @param inputExposure:  exposure to process
        @param enableWriteSources: if True then writing sources is allowed.
            Set False if you need to postprocess sources before writing them.

        @return pipe_base Struct containing these fields:
        - exposure: input exposure (as modified in the course of processing)
        - sources: detected source if config.doPhotometry, else None
        - matches: matches between detected sources and astrometric reference catalog
        - matchMeta: metadata for matches
        - backgrounds: background list (input as modified)
        """
        if backgrounds is None:
            backgrounds = afwMath.BackgroundList()

        if self.config.doDetection:
            idFactory = self.makeIdFactory(dataRef)
            table = afwTable.SourceTable.make(self.schema, idFactory)
            table.setMetadata(self.algMetadata)
            detections = self.detection.makeSourceCatalog(table, inputExposure)
            sources = detections.sources
            fpSets = detections.fpSets
            if fpSets.background:           
                backgrounds.append(fpSets.background)

        if self.config.doDeblend:
            self.deblend.run(inputExposure, sources, inputExposure.getPsf())

        if self.config.doMeasurement:
            self.measurement.run(inputExposure, sources)

        if sources is not None and self.config.doWriteSources:
            sourceWriteFlags = (0 if self.config.doWriteHeavyFootprintsInSources
                                else afwTable.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
            if enableWriteSources:
                dataRef.put(sources, self.dataPrefix + 'src', flags=sourceWriteFlags)

        srcMatches = None; srcMatchMeta = None
        if self.config.doMeasurement and self.config.doWriteSourceMatches:
            self.log.info("Matching src to reference catalogue")
            try:
                srcMatches, srcMatchMeta = self.matchSources(inputExposure, sources)
                if srcMatches is not None:
                    normalizedSrcMatches = afwTable.packMatches(srcMatches)
                    normalizedSrcMatches.table.setMetadata(srcMatchMeta)
                    dataRef.put(normalizedSrcMatches, self.dataPrefix + "srcMatch")
            except Exception as e:
                self.log.warn("Unable to match to reference catalog: %s" % e)

        return pipeBase.Struct(
            exposure = inputExposure,
            sources = sources,
            matches = srcMatches,
            matchMeta = srcMatchMeta,
            backgrounds = backgrounds,
        )

    def matchSources(self, exposure, sources):
        """Match the sources to the astrometric reference catalog"""
        astromRet = self.getAstrometer().useKnownWcs(sources, exposure=exposure)
        # N.b. yes, this is what useKnownWcs calls the returned values
        return astromRet.matches, astromRet.matchMetadata

    def getSchemaCatalogs(self):
        """Return a dict of empty catalogs for each catalog dataset produced by this task."""
        src = afwTable.SourceCatalog(self.schema)
        src.getTable().setMetadata(self.algMetadata)
        return {self.dataPrefix + "src": src}

    def writeBackgrounds(self, dataRef, backgrounds):
        """Backgrounds are persisted via the butler

        @param dataRef: Data reference
        @param backgrounds: List of background models
        """
        self.log.info("Persisting background models")
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

