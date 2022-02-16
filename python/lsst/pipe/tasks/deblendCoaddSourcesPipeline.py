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

import numpy as np

from lsst.pipe.base import (Struct, PipelineTask, PipelineTaskConfig, PipelineTaskConnections)
import lsst.pipe.base.connectionTypes as cT

from lsst.pex.config import ConfigurableField
from lsst.meas.deblender import SourceDeblendTask
from lsst.meas.extensions.scarlet import ScarletDeblendTask
from lsst.obs.base import ExposureIdInfo

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable

from .makeCoaddTempExp import reorderRefs

__all__ = ("DeblendCoaddSourcesSingleConfig", "DeblendCoaddSourcesSingleTask",
           "DeblendCoaddSourcesMultiConfig", "DeblendCoaddSourcesMultiTask")


deblendBaseTemplates = {"inputCoaddName": "deep", "outputCoaddName": "deep"}


class DeblendCoaddSourceSingleConnections(PipelineTaskConnections,
                                          dimensions=("tract", "patch", "band", "skymap"),
                                          defaultTemplates=deblendBaseTemplates):
    inputSchema = cT.InitInput(
        doc="Input schema to use in the deblend catalog",
        name="{inputCoaddName}Coadd_mergeDet_schema",
        storageClass="SourceCatalog"
    )
    peakSchema = cT.InitInput(
        doc="Schema of the footprint peak catalogs",
        name="{inputCoaddName}Coadd_peak_schema",
        storageClass="PeakCatalog"
    )
    mergedDetections = cT.Input(
        doc="Detection catalog merged across bands",
        name="{inputCoaddName}Coadd_mergeDet",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap")
    )
    coadd = cT.Input(
        doc="Exposure on which to run deblending",
        name="{inputCoaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "band", "skymap")
    )
    measureCatalog = cT.Output(
        doc="The output measurement catalog of deblended sources",
        name="{outputCoaddName}Coadd_deblendedFlux",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "band", "skymap")
    )
    outputSchema = cT.InitOutput(
        doc="Output of the schema used in deblending task",
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
    inputSchema = cT.InitInput(
        doc="Input schema to use in the deblend catalog",
        name="{inputCoaddName}Coadd_mergeDet_schema",
        storageClass="SourceCatalog"
    )
    peakSchema = cT.InitInput(
        doc="Schema of the footprint peak catalogs",
        name="{inputCoaddName}Coadd_peak_schema",
        storageClass="PeakCatalog"
    )
    mergedDetections = cT.Input(
        doc="Detection catalog merged across bands",
        name="{inputCoaddName}Coadd_mergeDet",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap")
    )
    coadds = cT.Input(
        doc="Exposure on which to run deblending",
        name="{inputCoaddName}Coadd_calexp",
        storageClass="ExposureF",
        multiple=True,
        dimensions=("tract", "patch", "band", "skymap")
    )
    outputSchema = cT.InitOutput(
        doc="Output of the schema used in deblending task",
        name="{outputCoaddName}Coadd_deblendedFlux_schema",
        storageClass="SourceCatalog"
    )
    fluxCatalogs = cT.Output(
        doc="Flux weighted catalogs produced by multiband deblending",
        name="{outputCoaddName}Coadd_deblendedFlux",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "band", "skymap"),
        multiple=True
    )
    templateCatalogs = cT.Output(
        doc="Template catalogs produced by multiband deblending",
        name="{outputCoaddName}Coadd_deblendedModel",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "band", "skymap"),
        multiple=True
    )


class DeblendCoaddSourcesMultiConfig(PipelineTaskConfig,
                                     pipelineConnections=DeblendCoaddSourcesMultiConnections):
    multibandDeblend = ConfigurableField(
        target=ScarletDeblendTask,
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

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        inputs["idFactory"] = ExposureIdInfo.fromDataId(
            butlerQC.quantum.dataId,
            "tract_patch"
        ).makeSourceIdFactory()
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def _makeSourceCatalog(self, mergedDetections, idFactory):
        # There may be gaps in the mergeDet catalog, which will cause the
        # source ids to be inconsistent. So we update the id factory
        # with the largest id already in the catalog.
        maxId = np.max(mergedDetections["id"])
        idFactory.notify(maxId)
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
        self.outputSchema = afwTable.SourceCatalog(self.schema)

    def run(self, coadd, mergedDetections, idFactory):
        sources = self._makeSourceCatalog(mergedDetections, idFactory)
        self.singleBandDeblend.run(coadd, sources)
        if not sources.isContiguous():
            sources = sources.copy(deep=True)
        return Struct(measureCatalog=sources)


class DeblendCoaddSourcesMultiTask(DeblendCoaddSourcesBaseTask):
    ConfigClass = DeblendCoaddSourcesMultiConfig
    _DefaultName = "deblendCoaddSourcesMulti"

    def __init__(self, initInputs, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)
        self.makeSubtask("multibandDeblend", schema=self.schema, peakSchema=self.peakSchema)
        self.outputSchema = afwTable.SourceCatalog(self.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Obtain the list of bands, sort them (alphabetically), then reorder
        # all input lists to match this band order.
        bandOrder = [dRef.dataId["band"] for dRef in inputRefs.coadds]
        bandOrder.sort()
        inputRefs = reorderRefs(inputRefs, bandOrder, dataIdKey="band")
        inputs = butlerQC.get(inputRefs)
        exposureIdInfo = ExposureIdInfo.fromDataId(butlerQC.quantum.dataId, "tract_patch")
        inputs["idFactory"] = exposureIdInfo.makeSourceIdFactory()
        inputs["filters"] = [dRef.dataId["band"] for dRef in inputRefs.coadds]
        outputs = self.run(**inputs)
        for outRef in outputRefs.templateCatalogs:
            band = outRef.dataId['band']
            if (catalog := outputs.templateCatalogs.get(band)) is not None:
                butlerQC.put(catalog, outRef)

        for outRef in outputRefs.fluxCatalogs:
            band = outRef.dataId['band']
            if (catalog := outputs.fluxCatalogs.get(band)) is not None:
                butlerQC.put(catalog, outRef)

    def run(self, coadds, filters, mergedDetections, idFactory):
        sources = self._makeSourceCatalog(mergedDetections, idFactory)
        multiExposure = afwImage.MultibandExposure.fromExposures(filters, coadds)
        templateCatalogs, fluxCatalogs = self.multibandDeblend.run(multiExposure, sources)
        retStruct = Struct(templateCatalogs=templateCatalogs, fluxCatalogs=fluxCatalogs)
        return retStruct
