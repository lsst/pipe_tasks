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

from lsst.pipe.base import (Struct, PipelineTask, PipelineTaskConfig, PipelineTaskConnections)
import lsst.pipe.base.connectionTypes as ct

from lsst.pex.config import ConfigurableField
from lsst.meas.deblender import SourceDeblendTask, MultibandDeblendTask

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable

__all__ = ("DeblendCoaddSourcesSingleConfig", "DeblendCoaddSourcesSingleTask",
           "DeblendCoaddSourcesMultiConfig", "DeblendCoaddSourcesMultiTask")


deblendBaseTemplates = {"inputCoaddName": "deep", "outputCoaddName": "deep"}


class DeblendCoaddSourceSingleConnections(PipelineTaskConnections,
                                          dimensions=("tract", "patch", "abstract_filter", "skymap"),
                                          defaultTemplates=deblendBaseTemplates):
    inputSchema = ct.InitInput(
        name="{inputCoaddName}Coadd_mergeDet_schema",
        storageClass="SourceCatalog"
    )
    peakScheam = ct.InitInput(
        name="{inputCoaddName}Coadd_peak_schema",
        storageClass="PeakCatalog"
    )
    mergedDetections = ct.Input(
        nameTemplate="{inputCoaddName}Coadd_mergeDet",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap")
    )
    coadd = ct.Input(
        name="{inputCoaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "abstract_filter", "skymap")
    )
    measureCatalog = ct.Output(
        name="{outputCoaddName}Coadd_deblendedFlux",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "abstract_filter", "skymap")
    )
    outputSchema = ct.InitOutput(
        name="{outputCoaddName}Coadd_deblendedFlux_schema",
        storageClass="SourceCatalog"
    )

    def setDefaults(self):
        super().setDefaults()
        self.singleBandDeblend.propagateAllPeaks = True


class DeblendCoaddSourcesSingleConfig(PipelineTaskConfig,
                                      pipelineConnections=DeblendCoaddSourceSingleConnections):
    singleBandDeblend = ConfigurableField(
        target=SourceDeblendTask,
        doc="Task to deblend an image in one band"
    )


class DeblendCoaddSourcesMultiConnections(PipelineTaskConnections,
                                          dimensions=("tract", "patch", "skymap"),
                                          defaultTemplates=deblendBaseTemplates):
    inputSchema = ct.InitInput(
        name="{inputCoaddName}Coadd_mergeDet_schema",
        storageClass="SourceCatalog"
    )
    peakScheam = ct.InitInput(
        name="{inputCoaddName}Coadd_peak_schema",
        storageClass="PeakCatalog"
    )
    mergedDetections = ct.Input(
        nameTemplate="{inputCoaddName}Coadd_mergeDet",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap")
    )
    coadds = ct.Input(
        nameTemplate="{inputCoaddName}Coadd_calexp",
        storageClass="ExposureF",
        multiple=True,
        dimensions=("tract", "patch", "abstract_filter", "skymap")
    )
    outputSchema = ct.InitOutput(
        nameTemplate="{outputCoaddName}Coadd_deblendedModel_schema",
        storageClass="SourceCatalog"
    )
    fluxCatalogs = ct.Output(
        nameTemplate="{outputCoaddName}Coadd_deblendedFlux",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "abstract_filter", "skymap")
    )
    templateCatalogs = ct.Output(
        nameTemplate="{outputCoaddName}Coadd_deblendedModel",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "abstract_filter", "skymap")
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.multibandDeblend.conserveFlux:
            self.outputs -= set(("fluxCatalogs",))


class DeblendCoaddSourcesMultiConfig(PipelineTaskConfig,
                                     pipelineConnections=DeblendCoaddSourcesMultiConnections):
    multibandDeblend = ConfigurableField(
        target=MultibandDeblendTask,
        doc="Task to deblend an images in multiple bands"
    )


class DeblendCoaddSourcesBaseTask(PipelineTask):
    def __init__(self, initInputs, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)
        schema = initInputs["inputSchema"].schema
        self.peakSchema = initInputs["peakSchema"].schema
        self.schemaMapper = afwTable.SchemaMapper(schema)
        self.schemaMapper.addMinimalSchema(schema)
        self.schema = self.schemaMapper.getOutputSchema()
        self.outputSchema = afwTable.SourceCatalog(self.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        packedId, maxBits = butlerQC.registry.packDataId("tract_patch", inputRefs.mergedDetections.dataId,
                                                         returnMaxBits=True)
        inputs["idFactory"] = afwTable.IdFactory.makeSource(packedId, 64 - maxBits)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def _makeSourceCatalog(self, mergedDetections, idFactory):
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

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        packedId, maxBits = butlerQC.registry.packDataId("tract_patch", inputRefs.mergedDetections.dataId,
                                                         returnMaxBits=True)
        inputs["idFactory"] = afwTable.IdFactory.makeSource(packedId, 64 - maxBits)
        inputs["filters"] = [dRef.dataId["abstract_filter"] for dRef in inputRefs.coadds]
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, coadds, filters, mergedDetections, idFactory):
        sources = self._makeSourceCatalog(mergedDetections, idFactory)
        multiExposure = afwImage.MultibandExposure.fromExposures(filters, coadds)
        fluxCatalogs, templateCatalogs = self.multibandDeblend.run(multiExposure, sources)
        retStruct = Struct(templateCatalogs)
        if self.config.multibandDeblend.conserveFlux:
            retStruct.fluxCatalogs = fluxCatalogs
        return retStruct
