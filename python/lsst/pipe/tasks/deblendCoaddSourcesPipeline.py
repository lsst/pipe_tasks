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

__all__ = ["DeblendCoaddSourcesMultiConfig", "DeblendCoaddSourcesMultiTask"]

import dataclasses

import numpy as np

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
import lsst.pipe.base.connectionTypes as cT

from lsst.pex.config import ChoiceField, ConfigurableField, Field
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.meas.extensions.scarlet import ScarletDeblendTask

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.images as imgs
from lsst.images.cells import CellCoadd

from .coaddBase import reorderRefs


deblendBaseTemplates = {"inputCoaddName": "deep", "outputCoaddName": "deep"}


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
    coadds_cell = cT.Input(
        doc="Exposure on which to run deblending",
        name="{inputCoaddName}CoaddCell",
        storageClass="MultipleCellCoadd",
        multiple=True,
        dimensions=("tract", "patch", "band", "skymap")
    )
    backgrounds = cT.Input(
        doc="Background model to subtract from the cell-based coadd",
        name="{inputCoaddName}Coadd_calexp_background",
        storageClass="Background",
        multiple=True,
        dimensions=("tract", "patch", "band", "skymap")
    )
    deconvolvedCoadds = cT.Input(
        doc="Deconvolved coadds",
        name="deconvolved_{inputCoaddName}_coadd",
        storageClass="ExposureF",
        multiple=True,
        dimensions=("tract", "patch", "band", "skymap")
    )
    outputSchema = cT.InitOutput(
        doc="Output of the schema used in deblending task",
        name="{outputCoaddName}Coadd_deblendedFlux_schema",
        storageClass="SourceCatalog"
    )
    deblendedCatalog = cT.Output(
        doc="Catalogs produced by multiband deblending",
        name="{outputCoaddName}Coadd_deblendedCatalog",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap"),
    )
    scarletModelData = cT.Output(
        doc="Multiband scarlet models produced by the deblender",
        name="{outputCoaddName}Coadd_scarletModelData",
        storageClass="LsstScarletModelData",
        dimensions=("tract", "patch", "skymap"),
    )
    objectParents = cT.Output(
        doc="Parents of the deblended objects",
        name="object_parent_patch",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if self.config.imageType == "future":
            self.coadds = dataclasses.replace(self.coadds, storageClass="CellCoadd")
            self.deconvolvedCoadds = dataclasses.replace(self.deconvolvedCoadds, storageClass="MaskedImageV2")
            del self.coadds_cell
            del self.backgrounds
        elif self.config.useCellCoadds:
            del self.coadds
        else:
            del self.coadds_cell
            del self.backgrounds


class DeblendCoaddSourcesMultiConfig(PipelineTaskConfig,
                                     pipelineConnections=DeblendCoaddSourcesMultiConnections):
    useCellCoadds = Field[bool](
        doc="Use cell-based coadds instead of regular coadds?",
        default=False,
    )
    multibandDeblend = ConfigurableField(
        target=ScarletDeblendTask,
        doc="Task to deblend an images in multiple bands"
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()
    imageType = ChoiceField(
        "Which image type to expect for the input coadds. "
        "This option only directly affects connection storage classes and hence 'runQuantum'; the 'run' "
        "method behavior is determined by which type is actually passed in.",
        allowed={
            "legacy": (
                "Read a lsst.cell_coadds.MultipleCellCoadd via 'coadds_cells` and restore 'background' "
                "(if useCellCoadd) or lsst.afw.image.Exposure via `coadds` (if not useCellCoadd), and read "
                "lsst.afw.image.Exposure via 'deconvolvedCoadds'."
            ),
            "future": (
                "Read lsst.images.cells.CellCoadd via 'coadds' and lsst.images.MaskedImage via "
                "'deconvolvedCoadds'.  The useCellCoadds options is ignored."
            ),
        },
        dtype=str,
        optional=False,
        default="legacy",
    )


class DeblendCoaddSourcesMultiTask(PipelineTask):
    ConfigClass = DeblendCoaddSourcesMultiConfig
    _DefaultName = "deblendCoaddSourcesMulti"

    def __init__(self, initInputs, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)
        schema = initInputs["inputSchema"].schema
        self.peakSchema = initInputs["peakSchema"].schema
        self.schemaMapper = afwTable.SchemaMapper(schema)
        self.schemaMapper.addMinimalSchema(schema)
        self.schema = self.schemaMapper.getOutputSchema()
        self.makeSubtask("multibandDeblend", schema=self.schema, peakSchema=self.peakSchema)
        self.outputSchema = afwTable.SourceCatalog(self.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Obtain the list of bands, sort them (alphabetically), then reorder
        # all input lists to match this band order.
        # Note: sometimes deconvolution fails. If this happens then
        # the dataIds missing from deconvolvedRefs will be removed
        # during the process.
        deconvolvedRefs = inputRefs.deconvolvedCoadds
        bandOrder = [dRef.dataId["band"] for dRef in deconvolvedRefs]
        bandOrder.sort()
        inputRefs = reorderRefs(inputRefs, bandOrder, dataIdKey="band")
        inputs = butlerQC.get(inputRefs)
        bands = [dRef.dataId["band"] for dRef in deconvolvedRefs]
        mergedDetections = inputs.pop("mergedDetections")
        match self.config.imageType:
            case "legacy":
                if self.config.useCellCoadds:
                    exposures = [mcc.stitch().asExposure() for mcc in inputs.pop("coadds_cell")]
                    backgrounds = inputs.pop("backgrounds")
                    for exposure, background in zip(exposures, backgrounds):
                        exposure.image -= background.getImage()
                    coadds = exposures
                    coaddRefs = inputRefs.coadds_cell
                else:
                    coadds = inputs.pop("coadds")
                    coaddRefs = inputRefs.coadds
            case "future":
                coadds = inputs.pop("coadds")  # conversion deferred to run().
                coaddRefs = inputRefs.coadds
            case _:
                raise AssertionError(f"Invalid choice {self.config.imageType!r} for imageType.")

        # Ensure that the coadd bands and deconvolved coadd bands match
        coaddBands = [dRef.dataId["band"] for dRef in coaddRefs]
        if bands != coaddBands:
            self.log.error("Coadd bands %s != deconvolved coadd bands %s", bands, coaddBands)
            raise RuntimeError(
                "Number of coadd bands and deconvolved coadd bands do not match. "
                "This should never happen and indicates a bug in reorderRefs."
            )

        deconvolvedCoadds = inputs.pop("deconvolvedCoadds")

        # Check that all inputs have been extracted correctly.
        assert not inputs, "runQuantum got extra inputs"

        outputs = self.run(
            coadds=coadds,
            bands=bands,
            mergedDetections=mergedDetections,
            idFactory=self.config.idGenerator.apply(butlerQC.quantum.dataId).make_table_id_factory(),
            deconvolvedCoadds=deconvolvedCoadds,
        )
        butlerQC.put(outputs, outputRefs)

    def run(self, coadds, bands, mergedDetections, deconvolvedCoadds, idFactory):
        """Deblend coadds from multiple bands together.

        Parameters
        ----------
        coadds : `list` [`lsst.afw.image.Exposure` | \
                `lsst.images.cells.CellCoadd`]
            Coadds to deblend.
        bands : `list` [`str`]
            Names or the bands for ``coadds`` (zip-iteration compatible).
        mergedDetections : `lsst.afw.table.SourceCatalog`
            Input catalog of detections, already merged across bands.
        deconvolvedCoadds : `list` [`lsst.afw.image.Exposure` | \
                `lsst.images.MaskedImage`]
            Deconvolved versions of ``coadds`` (zip-iteration compatible).
        idFactory : `lsst.afw.table.IdFactory`
            Factory used to generate output source IDs.

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            Unmodified outputs of the ``multibandDeblend`` subtask.
        """
        coadds = [c.to_legacy() if isinstance(c, CellCoadd) else c for c in coadds]
        deconvolvedCoadds = [self._coerceDeconvolvedInput(d, c) for d, c in zip(deconvolvedCoadds, coadds)]
        sources = self._makeSourceCatalog(mergedDetections, idFactory)
        multiExposure = afwImage.MultibandExposure.fromExposures(bands, coadds)
        mDeconvolved = afwImage.MultibandExposure.fromExposures(bands, deconvolvedCoadds)
        result = self.multibandDeblend.run(multiExposure, mDeconvolved, sources)
        return result

    def _coerceDeconvolvedInput(
        self, deconvolved: afwImage.Exposure | imgs.MaskedImage, coadd: afwImage.Exposure
    ) -> afwImage.Exposure:
        if isinstance(deconvolved, imgs.MaskedImage):
            deconvolved = afwImage.Exposure(
                maskedImage=deconvolved.to_legacy(plane_map=imgs.get_legacy_deep_coadd_mask_planes()),
                exposureInfo=coadd.getInfo(),
                dtype=deconvolved.image.array.dtype,
            )
        return deconvolved

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
