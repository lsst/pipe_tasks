# This file is part of pipe_tasks.

# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

"""Pipeline for running DiaSource association in a DRP context.
"""

import numpy as np
import pandas as pd

import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from .coaddBase import makeSkyInfo
from lsst.skymap import BaseSkyMap

__all__ = ["DrpAssociationPipeTask",
           "DrpAssociationPipeConfig",
           "DrpAssociationPipeConnections"]


class DrpAssociationPipeConnections(pipeBase.PipelineTaskConnections,
                            dimensions=("tract", "patch", "skymap"),
                            defaultTemplates={"coaddName": "deep",
                                              "warpTypeSuffix": "",
                                              "fakesType": ""}):
    diaSourceTables = pipeBase.connectionTypes.Input(
        doc="Catalog of calibrated DiaSources.",
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
    diaDiaObjectTable = pipeBase.connectionTypes.Output(
        doc="Catalog of associated DiaSources into DiaObjects.",
        name="{fakesType}{coaddName}Diff_diaObjTable",
        storageClass="DataFrame",
        dimensions=("tract", "patch"),
    )


class DrpAssociationPipeConfig(
        pipeBase.PipelineTaskConfig,
        pipelineConnections=DrpAssociationPipeConnections
    ):
    maxFootprintArea = pexConfig.Field(
        dtype=int,
        default=5000,
        doc="Maximum area of footprints")
    # associator = pexConfig.ConfigurableField(
    #     target=SimpleAssociationTask,
    #     doc="Task used to associate DiaSources with DiaObjects.",
    # )


class DrpAssociationPipeTask(pipeBase.PipelineTask):
    """Driver pipeline for loading DiaSource catalogs in a patch/tract
    region and associating them.
    """
    ConfigClass = DrpDiaPipeConfig
    _DefaultName = "drpDiaPieTask"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        inputs["tractId"] = butlerQC.quantum.dataId["tract"]
        inputs["patchId"] = butlerQC.quantum.dataId["patch"]

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, diaSourceTables, skyMap, tractId, patchId):
        """Trim DiaSources to the current Patch and run association.

        Takes in the set of DiaSource catalogs that covers the current patch,
        trims to them to the dimensions of the patch, and [TODO: eventually]
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

            ``diaDiaObjectTable``
                Table of DiaObjects from matching DiaSources
                (`pandas.DataFrame`).
        """
        self.log.info("Running DPR Association on patch %i, tract %i..." %
                      (patchId, tractId))

        skyInfo = makeSkyInfo(skyMap, tractId, patchId)

        # Get the patch bounding box.
        innerPatchBox = geom.Box2D(skyInfo.patchInfo.getInnerBBox())

        diaSourceHistory = []
        for catRef in diaSourceTables:
            cat = catRef.get(
                datasetType=self.config.connections.diaSourceTables,
                immediate=True)

            patchGen = self._trimToPatchGen(cat,
                                            innerPatchBox,
                                            skyInfo.wcs,
                                            skyMap,
                                            tractId)
            isInTractPatch = np.array(list(patchGen))

            nDiaSrc = isInTractPatch.sum()
            self.log.info(
                "Read DiaSource catalog of length %i from visit %i, "
                "detector %i. Found %i sources within the patch/tract "
                "footprint." %
                (len(cat), catRef.dataId["visit"],
                 catRef.dataId["detector"], nDiaSrc))

            if nDiaSrc <= 0:
                outData.append(pd.DataFrame(columns=cat.columns))
                continue

            cutCat = cat[isInTractPatch]
            diaSourceHistory.append(cutCat)

            # self.associator.addCatalog()

        diaSourceHistoryCat = pd.concat(diaSourceHistory)
        self.log.info("Found %i DiaSources overlaping patch %i, tract %i"
                      % (len(diaSourceHistoryCat), patchId, tractId))

        return pipeBase.Struct(
            diaDiaObjectTable=pd.DataFrame(columns=["diaObjectId",
                                                    "nDiaSources"]),
            assocDiaSourceTable=diaSourceHistoryCat)

    def _trimToPatchGen(self, cat, innerPatchBox, wcs):
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
        skyMap : ``
            SkyMap containing the patch and tract.

        Yields
        ------
        isInTractPatch : `bool`
            Boolean representing if the DiaSource is contained within the
            current patch and tract.
        """

        for idx, row in cat.iterrows():
            sphPoint = geom.SpherePoint(row["ra"], row["decl"], geom.degrees)
            inPatch = innerPatchBox.contains(wcs.skyToPixel(sphPoint))
            yield inPatch
