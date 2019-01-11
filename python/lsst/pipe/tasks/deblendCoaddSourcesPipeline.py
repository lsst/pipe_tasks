# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from lsst.pipe.base import (Struct, PipelineTask, InitInputDatasetField, InitOutputDatasetField,
                            InputDatasetField, OutputDatasetField, PipelineTaskConfig)

from lsst.pex.config import ConfigurableField
from lsst.meas.deblender import SourceDeblendTask, MultibandDeblendTask

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable

__all__ = ("DeblendCoaddSourcesSingleConfig", "DeblendCoaddSourcesSingleTask",
           "DeblendCoaddSourcesMultiConfig", "DeblendCoaddSourcesMultiTask")


class DeblendCoaddSourcesBaseConfig(PipelineTaskConfig):
    inputSchema = InitInputDatasetField(
        doc="Input schema to use in the deblend catalog",
        nameTemplate="{inputCoaddName}Coadd_mergeDet_schema",
        storageClass="SourceCatalog"
    )
    peakSchema = InitInputDatasetField(
        doc="Schema of the footprint peak catalogs",
        nameTemplate="{inputCoaddName}Coadd_peak_schema",
        storageClass="PeakCatalog"
    )
    mergedDetections = InputDatasetField(
        doc="Detection catalog merged across bands",
        nameTemplate="{inputCoaddName}Coadd_mergeDet",
        storageClass="SourceCatalog",
        scalar=True,
        dimensions=("Tract", "Patch", "SkyMap")
    )

    def setDefaults(self):
        super().setDefaults()
        self.quantum.dimensions = ("Tract", "Patch", "AbstractFilter", "SkyMap")
        self.formatTemplateNames({"inputCoaddName": "deep", "outputCoaddName": "deep"})


class DeblendCoaddSourcesSingleConfig(DeblendCoaddSourcesBaseConfig):
    singleBandDeblend = ConfigurableField(
        target=SourceDeblendTask,
        doc="Task to deblend an image in one band"
    )
    coadd = InputDatasetField(
        doc="Exposure on which to run deblending",
        nameTemplate="{inputCoaddName}Coadd_calexp",
        storageClass="ExposureF",
        scalar=True,
        dimensions=("Tract", "Patch", "AbstractFilter", "SkyMap")
    )
    measureCatalog = OutputDatasetField(
        doc="The output measurement catalog of deblended sources",
        nameTemplate="{outputCoaddName}Coadd_deblendedFlux",
        scalar=True,
        storageClass="SourceCatalog",
        dimensions=("Tract", "Patch", "AbstractFilter", "SkyMap")
    )
    outputSchema = InitOutputDatasetField(
        doc="Output of the schema used in deblending task",
        nameTemplate="{outputCoaddName}Coadd_deblendedFlux_schema",
        storageClass="SourceCatalog"
    )


class DeblendCoaddSourcesMultiConfig(DeblendCoaddSourcesBaseConfig):
    multibandDeblend = ConfigurableField(
        target=MultibandDeblendTask,
        doc="Task to deblend an images in multiple bands"
    )
    coadds = InputDatasetField(
        doc="Exposure on which to run deblending",
        nameTemplate="{inputCoaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("Tract", "Patch", "AbstractFilter", "SkyMap")
    )
    outputSchema = InitOutputDatasetField(
        doc="Output of the schema used in deblending task",
        nameTemplate="{outputCoaddName}Coadd_deblendedModel_schema",
        storageClass="SourceCatalog"
    )
    fluxCatalogs = OutputDatasetField(
        doc="Flux catalogs produced by multiband deblending, not written "
            "if conserve flux is turned off",
        nameTemplate="{outputCoaddName}Coadd_deblendedFlux",
        storageClass="SourceCatalog",
        dimensions=("Tract", "Patch", "AbstractFilter", "SkyMap")
    )
    templateCatalogs = OutputDatasetField(
        doc="Template catalogs produced by multiband deblending",
        nameTemplate="{outputCoaddName}Coadd_deblendedModel",
        storageClass="SourceCatalog",
        dimensions=("Tract", "Patch", "AbstractFilter", "SkyMap")
    )

    def setDefaults(self):
        super().setDefaults()
        self.quantum.dimensions = ("Tract", "Patch", "SkyMap")


class DeblendCoaddSourcesBaseTask(PipelineTask):
    def __init__(self, initInputs, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)
        schema = initInputs["inputSchema"].schema
        self.peakSchema = initInputs["peakSchema"].schema
        self.schemaMapper = afwTable.SchemaMapper(schema)
        self.schemaMapper.addMinimalSchema(schema)
        self.schema = self.schemaMapper.getOutputSchema()

    def getInitOutputDatasets(self):
        return {"outputSchema": afwTable.SourceCatalog(self.schema)}

    def adaptArgsAndRun(self, inputData, inputDataIds, outputDataIds, butler):
        # FINDME: DM-15843 needs to come back and address final solution
        inputData["idFactory"] = afwTable.IdFactory.makeSimple()
        return self.run(**inputData)

    def _makeSourceCatalog(self, mergedDetections, idFactory):
        # Need to do something more clever once we have a real Idfactory here
        # see command line task version. FINDME DM-15843
        table = afwTable.SourceTable.make(self.schema, idFactory)
        sources = afwTable.SourceCatalog(table)
        sources.extend(mergedDetections, self.schemaMapper)
        return sources


class DeblendCoaddSourcesSingleTask(DeblendCoaddSourcesBaseTask):
    ConfigClass = DeblendCoaddSourcesSingleConfig
    _DefaultName = "deblendCoaddSourcesSingle"

    def __init__(self, initInputs, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)
        self.makeSubtask("singleBandDeblend", schema=self.schema, peakSchema=self.peakSchema)

    def run(self, coadd, mergedDetections, idFactory):
        sources = self._makeSourceCatalog(mergedDetections, idFactory)
        self.singleBandDeblend.run(coadd, sources)
        return Struct(measureCatalog=sources)


class DeblendCoaddSourcesMultiTask(DeblendCoaddSourcesBaseTask):
    ConfigClass = DeblendCoaddSourcesMultiConfig
    _DefaultName = "deblendCoaddSourcesMulti"

    def __init__(self, initInputs, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)
        self.makeSubtask("multibandDeblend", schema=self.schema, peakSchema=self.peakSchema)

    @classmethod
    def getOutputDatasetTypes(cls, config):
        outputTypeDict = super().getOutputDatasetTypes(config)
        # If Conserve flux is set to false, remove that catalog as a possible output
        if not config.multibandDeblend.conserveFlux:
            outputTypeDict.pop("fluxCatalogs", None)
        return outputTypeDict

    def adaptArgsAndRun(self, inputData, inputDataIds, outputDataIds, butler):
        inputData["filters"] = [dId["abstract_filter"] for dId in inputDataIds["coadds"]]
        return super().adaptArgsAndRun(inputData, inputDataIds, outputDataIds, butler)

    def run(self, coadds, filters, mergedDetections, idFactory):
        sources = self._makeSourceCatalog(mergedDetections, idFactory)
        multiExposure = afwImage.MultibandExposure.fromExposures(filters, coadds)
        fluxCatalogs, templateCatalogs = self.multibandDeblend.run(multiExposure, sources)
        retStruct = Struct(templateCatalogs)
        if self.config.multibandDeblend.conserveFlux:
            retStruct.fluxCatalogs = fluxCatalogs
        return retStruct
