#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2012 LSST Corporation.
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
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
from lsst.meas.algorithms import SourceDetectionTask, SourceMeasurementTask, SourceDeblendTask
from lsst.ip.diffim import ImagePsfMatchTask

class ImageDifferenceConfig(pexConfig.Config):
    """Config for ImageDifferenceTask"""
    doSubtract = pexConfig.Field(dtype=bool, default=True, doc = "Compute subtracted exposure?")
    doDetection = pexConfig.Field(dtype=bool, default=True, doc = "Detect sources?")
    doDeblend = pexConfig.Field(dtype=bool, default=False,
        doc = "Deblend sources? Off by default because it may not be useful")
    doMeasurement = pexConfig.Field(dtype=bool, default=True, doc = "Measure sources?")
    doWriteSubtractedExp = pexConfig.Field(dtype=bool, default=True, doc = "Write difference image?")
    doWriteSources = pexConfig.Field(dtype=bool, default=True, doc = "Write sources?")
    doWriteHeavyFootprintsInSources = pexConfig.Field(dtype=bool, default=False,
        doc = "Include HeavyFootprint data in source table?")
                                                      
    coaddName = pexConfig.Field(
        doc = "coadd name: typically one of deep or goodSeeing",
        dtype = str,
        default = "deep",
    )
    subtract = pexConfig.ConfigurableField(
        target = ImagePsfMatchTask,
        doc = "Warp and PSF match template to exposure, then subtract",
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


class ImageDifferenceTask(pipeBase.CmdLineTask):
    """Subtract an image from a template coadd and measure the result
    
    Steps include:
    - warp template coadd to match WCS of image
    - PSF match image to warped template
    - subtract image from PSF-matched, warped template
    - persist difference image
    - detect sources
    - deblend sources (disabled by default)
    - measure sources
    """
    ConfigClass = ImageDifferenceConfig
    _DefaultName = "differenceImage"

    def __init__(self, **kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.makeSubtask("subtract")
        self.schema = afwTable.SourceTable.makeMinimalSchema()
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
        - subtractedExposure: exposure after subtracting template;
            the unpersisted version if subtraction not run but detection run
            None if neither subtraction nor detection run (i.e. nothing useful done)
        - subtractedRes: results of subtraction task; None if subtraction not run
        - sources: detected and possibly measured and deblended sources; None if detection not run
        """
        self.log.log(self.log.INFO, "Processing %s" % (sensorRef.dataId))

        # initialize outputs
        subtractedExposure = None
        subtractRes = None
        sources = None

        # We make one IdFactory that will be used by both icSrc and src datasets;
        # I don't know if this is the way we ultimately want to do things, but at least
        # this ensures the source IDs are fully unique.
        expBits = sensorRef.get("ccdExposureId_bits")
        expId = long(sensorRef.get("ccdExposureId"))
        idFactory = afwTable.IdFactory.makeSource(expId, 64 - expBits)
        
        exposure = sensorRef.get("calexp")
        subtractedExposureName = self.config.coaddName + "SubtractedExp"
        
        if self.config.doSubtract:
            templateExposure = self.getTemplate(exposure, sensorRef)
            
            # warp template exposure to match exposure,
            # PSF match template exposure to exposure,
            # then return the difference
            subtractRes = self.subtract.run(
                imageToConvolve = templateExposure,
                imageToNotConvolve = exposure,
                mode = "subtractExposures",
            )
            subtractedExposure = subtractRes.subtractedImage
            
            if self.config.doWriteSubtractedExp:
                 sensorRef.put(subtractedExposure, subtractedExposureName)
        
        if self.config.doDetection:
            if subtractedExposure is None:
                subtractedExposure = sensorRef.get(subtractedExposureName)
                psf = sensorRef.get("psf")
                subtractedExposure.setPsf(psf)

            table = afwTable.SourceTable.make(self.schema, idFactory)
            table.setMetadata(self.algMetadata)
            sources = self.detection.makeSourceCatalog(table, subtractedExposure).sources

            if self.config.doDeblend:
                self.deblend.run(subtractedExposure, sources, psf)
    
            if self.config.doMeasurement:
                apCorr = sensorRef.get("apCorr")
                self.measurement.run(subtractedExposure, sources, apCorr)
    
            if sources is not None and self.config.doWriteSources:
                if self.config.doWriteHeavyFootprintsInSources:
                    sources.setWriteHeavyFootprints(True)
                sensorRef.put(sources, self.config.coaddName + "SubtractedExposure_src")
            
        return pipeBase.Struct(
            subtractedExposure = subtractedExposure,
            subtractRes = subtractRes,
            sources = sources,
        )
    
    def getTemplate(self, exposure, sensorRef):
        """Return a template coadd exposure that overlaps the exposure
        
        @param[in] exposure: exposure
        @param[in] sensorRef: a Butler data reference that can be used to obtain coadd data

        @return coaddExposure: a template coadd exposure assembled out of patches
        
        @note: the coadd consists of whole patches stitched together, so it may be larger than necessary
        """
        skyMap = sensorRef.get(datasetType=self.config.coaddName + "Coadd_skyMap")
        expWcs = exposure.getWcs()
        expBoxD = afwGeom.Box2D(exposure.getBBox(afwImage.PARENT))
        ctrSkyPos = expWcs.pixelToSky(expBoxD.getCenter())
        tractInfo = skyMap.findTract(ctrSkyPos)
        self.log.info("Using skyMap tract %s" % (tractInfo.getId(),))
        skyCorners = [expWcs.pixelToSky(pixPos) for pixPos in expBoxD.getCorners()]
        patchList = tractInfo.findPatchList(skyCorners)
        if not patchList:
            raise RuntimeError("No suitable tract found")
        self.log.info("Assembling %s coadd patches" % (len(patchList),))
        # compute inclusive bounding box
        coaddBBox = afwGeom.Box2I()
        for patchInfo in patchList:
            outerBBox = patchInfo.getOuterBBox()
            for corner in outerBBox.getCorners():
                coaddBBox.include(corner)
        self.log.info("exposure dimensions=%s; coadd dimensions=%s" % \
            (exposure.getDimensions(), coaddBBox.getDimensions()))
        
        coaddExposure = afwImage.ExposureF(coaddBBox, tractInfo.getWcs())
        for patchInfo in patchList:
            coaddPatch = sensorRef.get(
                datasetType = self.config.coaddName + "Coadd",
                tract = tractInfo.getId(),
                patch = "%s,%s" % (patchInfo.getIndex()[0], patchInfo.getIndex()[1]),
            )
            coaddView = afwImage.MaskedImageF(coaddExposure.getMaskedImage(),
                coaddPatch.getBBox(afwImage.PARENT))
            coaddView <<= coaddPatch.getMaskedImage()
        
        return coaddExposure

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        return pipeBase.ArgumentParser(name=cls._DefaultName, datasetType="calexp")
