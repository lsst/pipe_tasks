#!/usr/bin/env python

from lsst.pex.config import Config, ConfigField
from lsst.pipe.base import Task, Struct

import lsst.daf.base as dafBase
import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom
import lsst.meas.algorithms as measAlg


class ForcedPhotConfig(Config):
    """Configuration for forced photometry.

    This is quite bare, but it may be extended by subclasses
    to support getting the list of reference sources.
    """
    measurement = ConfigField(dtype=measAlg.SourceMeasurementConfig,
                              doc="Configuration for forced measurement")

class ForcedPhotTask(Task):
    """Task to perform forced photometry.

    This is a base class; it will need sub-classing to implement
    the getReferences() method.

    "Forced photometry" is measurement on an image using the
    position from another source as the centroid, and without
    recentering.
    """

    ConfigClass = ForcedPhotConfig
    _DefaultName = "forcedPhot"

    def __init__(self, *args, **kwargs):
        super(ForcedPhotTask, self).__init__(*args, **kwargs)
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask("measurement", measAlg.SourceMeasurementTask,
                         schema=self.schema, algMetadata=self.algMetadata)

    def run(self, dataRef):
        inputs = self.readInputs(dataRef)
        exposure = inputs.exposure
        exposure.setPsf(inputs.psf)
        references = self.getReferences(dataRef, exposure)
        references = self.subsetReferences(references, exposure)
        sources = self.generateSources(len(references))
        self.measure(sources, exposure, references, apCorr=inputs.apCorr)
        return Struct(sources=sources, references=references)

    def readInputs(self, dataRef, exposureName="calexp", psfName="psf", apCorrName="apCorr"):
        """Read inputs for exposure

        @param dataRef         Data reference from butler
        @param exposureName    Name for exposure in butler
        @param psfName         Name for PSF in butler
        @param apCorrName      Name for aperture correction, or None
        """
        return Struct(exposure=dataRef.get(exposureName),
                      psf=dataRef.get(psfName),
                      apCorr=dataRef.get(apCorrName) if apCorrName is not None else None,
                      )

    def getReferences(self, dataRef, exposure):
        """Get reference sources on (or close to) exposure"""
        raise NotImplementedError("Don't know how to get reference sources in the generic case")

    def subsetReferences(self, references, exposure):
        """Generate a subset of reference sources to ensure all are in the exposure

        @param references  Reference sources
        @param exposure    Exposure of interest
        @return Subset of references
        """
        box = afwGeom.Box2D(exposure.getBBox())
        wcs = exposure.getWcs()
        subset = afwTable.SourceCatalog(references.table)
        for ref in references:
            coord = ref.getCoord()
            if box.contains(wcs.skyToPixel(coord)):
                subset.append(ref)
        return subset

    def generateSources(self, number):
        """Generate sources to be measured

        @param number  Number of sources to generate
        @return Sources ready for measurement
        """
        sources = afwTable.SourceCatalog(self.schema)
        table = sources.table
        table.setMetadata(self.algMetadata)
        sources.preallocate(number)
        for i in range(number):
            src = table.makeRecord()
            sources.append(src)
        return sources

    def measure(self, sources, exposure, references, apCorr=None):
        """Measure sources on the exposure at the position of the references

        @param sources     Sources to receive measurements
        @param exposure    Exposure to measure
        @param references  Reference sources
        @param apCorr      Aperture correction to apply, or None
        """
        self.log.log(self.log.INFO, "Forced measurement of %d sources" % len(
        self.measurement.run(exposure, sources, apCorr=apCorr, references=references)
