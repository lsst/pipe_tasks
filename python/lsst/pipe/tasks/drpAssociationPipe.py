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

"""Pipeline for running DiaSource association in a DRP context.
"""

__all__ = ["DrpAssociationPipeTask",
           "DrpAssociationPipeConfig",
           "DrpAssociationPipeConnections"]

import numpy as np
import pandas as pd

import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.skymap import BaseSkyMap

from .coaddBase import makeSkyInfo
from .simpleAssociation import SimpleAssociationTask


class DrpAssociationPipeConnections(pipeBase.PipelineTaskConnections,
                                    dimensions=("tract", "patch", "skymap"),
                                    defaultTemplates={"coaddName": "deep",
                                                      "warpTypeSuffix": "",
                                                      "fakesType": ""}):
    diaSourceTables = pipeBase.connectionTypes.Input(
        doc="Set of catalogs of calibrated DiaSources.",
        name="{fakesType}{coaddName}Diff_diaSrcTable",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "detector"),
        deferLoad=True,
        multiple=True
    )
    skyMap = pipeBase.connectionTypes.Input(
        doc="Input definition of geometry/bbox and projection/wcs for coadded "
        "exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap", ),
    )
    assocDiaSourceTable = pipeBase.connectionTypes.Output(
        doc="Catalog of DiaSources covering the patch and associated with a "
            "DiaObject.",
        name="{fakesType}{coaddName}Diff_assocDiaSrcTable",
        storageClass="DataFrame",
        dimensions=("tract", "patch"),
    )
    diaObjectTable = pipeBase.connectionTypes.Output(
        doc="Catalog of DiaObjects created from spatially associating "
            "DiaSources.",
        name="{fakesType}{coaddName}Diff_diaObjTable",
        storageClass="DataFrame",
        dimensions=("tract", "patch"),
    )


class DrpAssociationPipeConfig(
        pipeBase.PipelineTaskConfig,
        pipelineConnections=DrpAssociationPipeConnections):
    associator = pexConfig.ConfigurableField(
        target=SimpleAssociationTask,
        doc="Task used to associate DiaSources with DiaObjects.",
    )
    doAddDiaObjectCoords = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Do pull diaObject's average coordinate as coord_ra and coord_dec"
            "Duplicates information, but needed for bulk ingest into qserv."
    )
    doWriteEmptyTables = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="If True, construct and write out empty diaSource and diaObject "
            "tables. If False, raise NoWorkFound"
    )


class DrpAssociationPipeTask(pipeBase.PipelineTask):
    """Driver pipeline for loading DiaSource catalogs in a patch/tract
    region and associating them.
    """
    ConfigClass = DrpAssociationPipeConfig
    _DefaultName = "drpAssociation"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask('associator')

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        inputs["tractId"] = butlerQC.quantum.dataId["tract"]
        inputs["patchId"] = butlerQC.quantum.dataId["patch"]
        tractPatchId, skymapBits = butlerQC.quantum.dataId.pack(
            "tract_patch",
            returnMaxBits=True)
        inputs["tractPatchId"] = tractPatchId
        inputs["skymapBits"] = skymapBits

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self,
            diaSourceTables,
            skyMap,
            tractId,
            patchId,
            tractPatchId,
            skymapBits):
        """Trim DiaSources to the current Patch and run association.

        Takes in the set of DiaSource catalogs that covers the current patch,
        trims them to the dimensions of the patch, and [TODO: eventually]
        runs association on the concatenated DiaSource Catalog.

        Parameters
        ----------
        diaSourceTables : `list` of `lst.daf.butler.DeferredDatasetHandle`
            Set of DiaSource catalogs potentially covering this patch/tract.
        skyMap : `lsst.skymap.BaseSkyMap`
            SkyMap defining the patch/tract
        tractId : `int`
            Id of current tract being processed.
        patchId : `int`
            Id of current patch being processed

        Returns
        -------
        output : `lsst.pipe.base.Struct`
            Results struct with attributes:

            ``assocDiaSourceTable``
                Table of DiaSources with updated value for diaObjectId.
                (`pandas.DataFrame`)
            ``diaObjectTable``
                Table of DiaObjects from matching DiaSources
                (`pandas.DataFrame`).
        """
        self.log.info("Running DPR Association on patch %i, tract %i...",
                      patchId, tractId)

        skyInfo = makeSkyInfo(skyMap, tractId, patchId)

        # Get the patch bounding box.
        innerPatchBox = geom.Box2D(skyInfo.patchInfo.getInnerBBox())

        diaSourceHistory = []
        for catRef in diaSourceTables:
            cat = catRef.get()

            isInTractPatch = self._trimToPatch(cat,
                                               innerPatchBox,
                                               skyInfo.wcs)

            nDiaSrc = isInTractPatch.sum()
            self.log.info(
                "Read DiaSource catalog of length %i from visit %i, "
                "detector %i. Found %i sources within the patch/tract "
                "footprint.",
                len(cat), catRef.dataId["visit"],
                catRef.dataId["detector"], nDiaSrc)

            if nDiaSrc <= 0:
                continue

            cutCat = cat[isInTractPatch]
            diaSourceHistory.append(cutCat)

        if diaSourceHistory:
            diaSourceHistoryCat = pd.concat(diaSourceHistory)
        else:
            # No rows to associate
            if self.config.doWriteEmptyTables:
                self.log.info("Constructing empty table")
                # Construct empty table using last table and dropping all the rows
                diaSourceHistoryCat = cat.drop(cat.index)
            else:
                raise pipeBase.NoWorkFound("Found no overlapping DIASources to associate.")

        self.log.info("Found %i DiaSources overlapping patch %i, tract %i",
                      len(diaSourceHistoryCat), patchId, tractId)

        assocResult = self.associator.run(diaSourceHistoryCat,
                                          tractPatchId,
                                          skymapBits)

        self.log.info("Associated DiaSources into %i DiaObjects",
                      len(assocResult.diaObjects))

        if self.config.doAddDiaObjectCoords:
            assocResult.assocDiaSources = self._addDiaObjectCoords(assocResult.diaObjects,
                                                                   assocResult.assocDiaSources)

        return pipeBase.Struct(
            diaObjectTable=assocResult.diaObjects,
            assocDiaSourceTable=assocResult.assocDiaSources)

    def _addDiaObjectCoords(self, objects, sources):
        obj = objects[['ra', 'decl']].rename(columns={"ra": "coord_ra", "decl": "coord_dec"})
        df = pd.merge(sources.reset_index(), obj, left_on='diaObjectId', right_index=True,
                      how='inner').set_index('diaSourceId')
        return df

    def _trimToPatch(self, cat, innerPatchBox, wcs):
        """Create generator testing if a set of DiaSources are in the
        patch/tract.

        Parameters
        ----------
        cat : `pandas.DataFrame`
            Catalog of DiaSources to test within patch/tract.
        innerPatchBox : `lsst.geom.Box2D`
            Bounding box of the patch.
        wcs : `lsst.geom.SkyWcs`
            Wcs of the tract.

        Returns
        ------
        isInPatch : `numpy.ndarray`, (N,)
            Booleans representing if the DiaSources are contained within the
            current patch and tract.
        """
        isInPatch = np.array([
            innerPatchBox.contains(
                wcs.skyToPixel(
                    geom.SpherePoint(row["ra"], row["decl"], geom.degrees)))
            for idx, row in cat.iterrows()])
        return isInPatch
