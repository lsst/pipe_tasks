#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008-2013 LSST Corporation.
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

import lsst.afw.table
from lsst.daf.base import PropertyList
from lsst.meas.algorithms import SourceMeasurementTask
from lsst.pex.config import Config, ConfigurableField, DictField, Field, FieldValidationError
from lsst.pipe.base import Task, CmdLineTask, Struct, timeMethod, ButlerInitializedTaskRunner
from .references import MultiBandReferencesTask

__all__ = ("ForcedPhotImageTask",)

class ForcedPhotImageConfig(Config):
    """Configuration for forced photometry.
    """
    references = ConfigurableField(target=MultiBandReferencesTask, doc="Retrieve reference source catalog")
    measurement = ConfigurableField(target=SourceMeasurementTask, doc="measurement subtask")
    copyColumns = DictField(
        keytype=str, itemtype=str, doc="Mapping of reference columns to source columns",
        default={"id": "object.id", "parent":"object.parent", "deblend.nchild": "object.deblend.nchild"}
        )

    def _getTweakCentroids(self):
        return self.measurement.centroider.name is not None

    def _setTweakCentroids(self, doTweak):
        if doTweak:
            self.measurement.centroider.name = "centroid.sdss"
            self.measurement.algorithms.names -= ["centroid.sdss"]
            self.measurement.algorithms.names |= ["skycoord", "centroid.record"]
            self.measurement.slots.centroid = "centroid.sdss"
        else:
            self.measurement.centroider.name = None
            self.measurement.algorithms.names |= ["centroid.sdss", "centroid.record"]
            self.measurement.algorithms.names -= ["skycoord"]
            self.measurement.slots.centroid = "centroid.record"

    doTweakCentroids = property(
        _getTweakCentroids, _setTweakCentroids,
        doc=("A meta-config option (just a property, really) that sets whether to tweak centroids during "
             "measurement by modifying several other config options")
    )

    def setDefaults(self):
        self.doTweakCentroids = False

class ForcedPhotImageTask(CmdLineTask):
    """Base class for performing forced measurement, in which the results (often just centroids) from
    regular measurement on another image are used to perform restricted measurement on a new image.

    This task is not directly usable as a CmdLineTask; subclasses must:
     - Set the _DefaultName class attribute
     - Implement makeIdFactory
     - Implement fetchReferences
     - (optional) Implement attachFootprints
    """

    RunnerClass = ButlerInitializedTaskRunner
    ConfigClass = ForcedPhotImageConfig
    dataPrefix = ""  # Name to prepend to all input and output datasets (e.g. 'goodSeeingCoadd_')

    def __init__(self, *args, **kwargs):
        """Initialize the task.

        ForcedPhotImageTask takes two keyword arguments beyond the usual CmdLineTask arguments:
         - schema: the Schema of the reference catalog, passed to the constructor of the references subtask
         - butler: a butler that will be passed to the references subtask to allow it to load its Schema
           from disk.
        At least one of these arguments must be present; if both are, schema takes precedence.
        """
        butler = kwargs.pop("butler", None)
        refSchema = kwargs.pop("schema", None)
        super(ForcedPhotImageTask, self).__init__(*args, **kwargs)
        self.algMetadata = PropertyList()
        self.makeSubtask("references", butler=butler, schema=refSchema)
        if refSchema is None:
            refSchema = self.references.schema
        # We make a SchemaMapper to transfer fields from the reference catalog
        self.schemaMapper = lsst.afw.table.SchemaMapper(refSchema)
        # First we have to include the minimal schema all SourceCatalogs must have, but we don't
        # want to transfer those fields from the refSchema (hence doMap=False)
        self.schemaMapper.addMinimalSchema(lsst.afw.table.SourceTable.makeMinimalSchema(), False)
        # Now we setup mappings from refSchema to the output schema, setting doReplace=True
        # so we can set minimal schema fields if so configured.
        for refName, targetName in self.config.copyColumns.items():
            refItem = refSchema.find(refName)
            self.schemaMapper.addMapping(refItem.key, targetName, True) # doReplace=True
        # Extract the output schema, and add the actual forced measurement fields to it.
        self.schema = self.schemaMapper.getOutputSchema()
        self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata, isForced=True)

    def getSchemaCatalogs(self):
        catalog = lsst.afw.table.SourceCatalog(self.schema)
        return {self.dataPrefix + "forced_src": catalog}

    def makeIdFactory(self, dataRef):
        """Hook for derived classes to define how to make an IdFactory for forced sources.

        Note that this is for forced source IDs, not object IDs, which are usually handled by
        the copyColumns config option.
        """
        raise NotImplementedError()

    def fetchReferences(self, dataRef, exposure):
        """Hook for derived classes to define how to get references objects.

        Derived classes should call one of the fetch* methods on the references subtask,
        but which one they call depends on whether the region to get references for is a
        easy to describe in patches (as it would be when doing forced measurements on a
        coadd), or is just an arbitrary box (as it would be for CCD forced measurements).
        """
        raise NotImplementedError()

    def attachFootprints(self, dataRef, sources, references, exposure, refWcs):
        """Hook for derived classes to define how to attach Footprints to blank sources prior to measurement

        Footprints for forced photometry must be in the pixel coordinate system of the image being
        measured, while the actual detections may start out in a different coordinate system.

        Subclasses for ForcedPhotImageTask must implement this method to define how those Footprints
        should be generated.

        The default implementation transforms the Footprints from the reference catalog from the refWcs
        to the exposure's Wcs, which downgrades HeavyFootprints into regular Footprints, destroying
        deblend information.
        """
        exposureWcs = exposure.getWcs()
        region = exposure.getBBox(lsst.afw.image.PARENT)
        for srcRecord, refRecord in zip(sources, references):
            srcRecord.setFootprint(refRecord.getFootprint().transform(refWcs, exposureWcs, region))

    def getExposure(self, dataRef):
        """Read input exposure on which to perform the measurements

        @param dataRef       Data reference from butler
        """
        if dataRef.datasetExists(self.dataPrefix + "calexp"):
            return dataRef.get(self.dataPrefix + "calexp", immediate=True)
        else:
            return None

    def writeOutput(self, dataRef, sources):
        """Write forced source table

        @param dataRef  Data reference from butler
        @param sources  SourceCatalog to save
        """
        dataRef.put(sources, self.dataPrefix + "forced_src")

    def generateSources(self, dataRef, references):
        """Generate sources to be measured, copying any fields in self.config.copyColumns

        @param dataRef     Data reference from butler
        @param references  Sequence (not necessarily a SourceCatalog) of reference sources
        @param idFactory   Factory to generate unique ids for forced sources
        @return Source catalog ready for measurement
        """
        if self.schema is None:
            self._buildSchema(dataRef.butlerSubset.butler)
        idFactory = self.makeIdFactory(dataRef)
        table = lsst.afw.table.SourceTable.make(self.schema, idFactory)
        sources = lsst.afw.table.SourceCatalog(table)
        table = sources.table
        table.setMetadata(self.algMetadata)
        table.preallocate(len(references))
        for ref in references:
            sources.addNew().assign(ref, self.schemaMapper)
        return sources

    @lsst.pipe.base.timeMethod
    def run(self, dataRef):
        """Perform forced measurement on the exposure defined by the given dataref.

        The dataRef must contain a 'tract' key, which is used to resolve the correct references
        in the presence of tract overlaps, and also defines the WCS of the reference sources.
        """
        refWcs = self.references.getWcs(dataRef)
        exposure = self.getExposure(dataRef)
        if exposure:
            references = list(record for record in self.fetchReferences(dataRef, exposure)
                              if record.getFootprint().getArea() > 0 and not record.getCentroidFlag())
            self.log.info("Performing forced measurement on %s" % dataRef.dataId)
            sources = self.generateSources(dataRef, references)
            self.attachFootprints(dataRef, sources, references=references, exposure=exposure, refWcs=refWcs)
            self.measurement.run(exposure, sources, references=references, refWcs=refWcs)
            self.writeOutput(dataRef, sources)
            return Struct(sources=sources)
        else:
            self.log.info("No image exists for %s" % (dataRef.dataId))
            return Struct(sources=None)
