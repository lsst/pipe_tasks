#!/usr/bin/env python

import math

from lsst.pex.config import Config, ConfigurableField, DictField, Field
from lsst.pipe.base import Task, CmdLineTask, Struct, ArgumentParser, timeMethod

import lsst.daf.base as dafBase
import lsst.afw.table as afwTable
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.meas.algorithms as measAlg

class ReferencesConfig(Config):
    """Configuration for reference source catalog retrieval

    This is bare, but will be extended by subclasses
    to support getting the list of reference sources.
    """
    correct = Field(dtype=bool, default=False, doc="Correct references for astrometric offsets?")
    minFlux = Field(dtype=float, default=3000, doc="Minimum flux for calculating offsets")
    radius = Field(dtype=float, default=0.5, doc="Association radius for matching, arcsec")

class ReferencesTask(Task):
    """Task to generate a reference source catalog for forced photometry

    This is a base class, as it is not clear how to generate the
    reference sources in the generic case (different projects will
    want to do this differently: perhaps from measuring a coadd, or
    perhaps from a database, or ...) and so this class MUST be
    overridden to properly define the getReferences() method.
    """

    ConfigClass = ReferencesConfig

    def run(self, dataRef, exposure):
        references = self.getReferences(dataRef, exposure)
        self.log.log(self.log.INFO, "Retrieved %d reference sources" % len(references))
        references = self.subsetReferences(references, exposure)
        self.log.log(self.log.INFO, "Subset to %d reference sources" % len(references))
        if self.config.correct:
            references = self.correctReferences(dataRef, references)
        return references

    def getReferences(self, dataRef, exposure):
        """Get reference sources on (or close to) exposure.

        This method must be overridden by subclasses to return
        a lsst.afw.table.SourceCatalog.

        @param dataRef     Data reference from butler
        @param exposure    Exposure that has been read
        @return Catalog (lsst.afw.table.SourceCatalog) of reference sources
        """
        # XXX put something in the Mapper???
        self.log.log(self.log.FATAL,
                     """Calling base class implementation of ReferencesTask.getReferences()!
            You need to configure a subclass of ReferencesTask.  Put in your configuration
            override file something like:
                from some.namespace import SubclassReferencesTask
                root.references.retarget(SubclassReferencesTask)
            """)
        raise NotImplementedError("Don't know how to get reference sources in the generic case")

    def subsetReferences(self, references, exposure):
        """Generate a subset of reference sources to ensure all are in the exposure

        @param references  Reference source catalog
        @param exposure    Exposure of interest
        @return Reference catalog with subset
        """
        box = afwGeom.Box2D(exposure.getBBox())
        wcs = exposure.getWcs()
        subset = afwTable.SourceCatalog(references.table)
        for ref in references:
            coord = ref.getCoord()
            if box.contains(wcs.skyToPixel(coord)):
                subset.append(ref)
        return subset

    def correctReferences(self, dataRef, references):
        self.log.info("Correcting reference positions...")
        sources = dataRef.get("src")
        matches = afwTable.matchRaDec(sources, references, self.config.radius * afwGeom.arcseconds)
        num = len(matches)
        self.log.info("%d matches between source and reference catalogs" % num)
        stats = afwMath.StatisticsControl()
        # XXX statistics parameters?
        dra, ddec = afwMath.vectorF(), afwMath.vectorF()
        dra.reserve(num)
        ddec.reserve(num)
        # XXX errors in positions?
        for m in matches:
            src = m.first
            if src.getPsfFlux() < self.config.minFlux:
                continue
            ref = m.second
            offset = ref.getCoord().getOffsetFrom(src.getCoord(), afwGeom.arcseconds)
            dra.push_back(offset[0])
            ddec.push_back(offset[1])
        num = len(dra)
        draStats = afwMath.makeStatistics(dra, afwMath.MEANCLIP | afwMath.STDEVCLIP, stats)
        ddecStats = afwMath.makeStatistics(ddec, afwMath.MEANCLIP | afwMath.STDEVCLIP, stats)
        offset = afwGeom.Point2D(draStats.getValue(afwMath.MEANCLIP), ddecStats.getValue(afwMath.MEANCLIP))
        self.log.info("Offset from %d sources is dRA = %f +/- %f arcsec, dDec = %f +/- %f arcsec" %
                      (num, offset.getX(), draStats.getValue(afwMath.STDEVCLIP), offset.getY(),
                       ddecStats.getValue(afwMath.STDEVCLIP)))
        for ref in references:
            coord = ref.getCoord()
            coord.offset(offset, afwGeom.arcseconds)
            ref.setCoord(coord)
        return references

class ForcedPhotConfig(Config):
    """Configuration for forced photometry.

    """
    references = ConfigurableField(target=ReferencesTask, doc="Retrieve reference source catalog")
    measurement = ConfigurableField(target=measAlg.SourceMeasurementTask, doc="Forced measurement")
    copyColumns = DictField(keytype=str, itemtype=str, doc="Mapping of reference columns to source columns",
                            default={"id": "objectId"})


class ForcedPhotTask(CmdLineTask):
    """Task to perform forced photometry.

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
        self.makeSubtask("references")
        self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)

    @classmethod
    def _makeArgumentParser(cls):
        """Overriding CmdLineTask._makeArgumentParser to set dataset type"""
        return ArgumentParser(name=cls._DefaultName, datasetType="calexp")

    @timeMethod
    def run(self, dataRef):
        inputs = self.readInputs(dataRef)
        exposure = inputs.exposure
        exposure.setPsf(inputs.psf)
        references = self.references.run(dataRef, exposure)
        self.log.log(self.log.INFO, "Performing forced measurement on %d sources" % len(references))
        sources = self.generateSources(references)
        self.measurement.run(exposure, sources, apCorr=inputs.apCorr, references=references)
        self.writeOutput(dataRef, sources)

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

    def generateSources(self, references):
        """Generate sources to be measured
        
        @param references  Reference source catalog 
        @return Source catalog ready for measurement
        """
        schema = afwTable.Schema(self.schema)

        copyKeys = []
        for fromCol, toCol in self.config.copyColumns.items():
            item = references.schema.find(fromCol)
            schema.addField(toCol, item.field.getTypeString(), item.field.getDoc(), item.field.getUnits())
            keys = (item.key, schema.find(toCol).key)
            copyKeys.append(keys)
        
        sources = afwTable.SourceCatalog(schema)
        table = sources.table
        table.setMetadata(self.algMetadata)
        sources.preallocate(len(references))
        for ref in references:
            src = table.makeRecord()
            for fromKey, toKey in copyKeys:
                src.set(toKey, ref.get(fromKey))
            sources.append(src)
        return sources

    def writeOutput(self, dataRef, sources, outName="forcedsources"):
        """Write sources out.

        @param outName     Name of forced sources in butler
        """
        dataRef.put(sources, outName)

