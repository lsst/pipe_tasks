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

import astropy.table as tb
import numpy as np
import pandas as pd


from lsst.pipe.tasks.ssoAssociation import SolarSystemAssociationTask
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.meas.base import SkyMapIdGeneratorConfig
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
    ssObjectTableRefs = pipeBase.connectionTypes.Input(
        doc="Reference to catalogs of SolarSolarSystem objects expected to be "
            "observable in each (visit, detector).",
        name="preloaded_DRP_SsObjects",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit", "detector"),
        minimum=0,
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
    finalVisitSummaryRefs = pipeBase.connectionTypes.Input(
        doc="Reference to finalVisitSummary of each exposure, containing "
        "visitInfo, bbox, and wcs.",
        name="finalVisitSummary",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
        deferLoad=True,
        multiple=True
    )
    assocDiaSourceTable = pipeBase.connectionTypes.Output(
        doc="Catalog of DiaSources covering the patch and associated with a "
            "DiaObject.",
        name="{fakesType}{coaddName}Diff_assocDiaSrcTable",
        storageClass="DataFrame",
        dimensions=("tract", "patch"),
    )
    associatedSsSources = pipeBase.connectionTypes.Output(
        doc="Optional output storing ssSource data computed during association.",
        name="{fakesType}{coaddName}Diff_assocSsSrcTable",
        storageClass="ArrowAstropy",
        dimensions=("tract", "patch"),
    )
    unassociatedSsObjects = pipeBase.connectionTypes.Output(
        doc="Expected locations of ssObjects with no associated source.",
        name="{fakesType}{coaddName}Diff_unassocSsObjTable",
        storageClass="ArrowAstropy",
        dimensions=("tract", "patch"),
    )
    diaObjectTable = pipeBase.connectionTypes.Output(
        doc="Catalog of DiaObjects created from spatially associating "
            "DiaSources.",
        name="{fakesType}{coaddName}Diff_diaObjTable",
        storageClass="DataFrame",
        dimensions=("tract", "patch"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config.doSolarSystemAssociation:
            del self.ssObjectTableRefs
            del self.associatedSsSources
            del self.unassociatedSsObjects
            del self.finalVisitSummaryRefs


class DrpAssociationPipeConfig(
        pipeBase.PipelineTaskConfig,
        pipelineConnections=DrpAssociationPipeConnections):
    associator = pexConfig.ConfigurableField(
        target=SimpleAssociationTask,
        doc="Task used to associate DiaSources with DiaObjects.",
    )
    solarSystemAssociator = pexConfig.ConfigurableField(
        target=SolarSystemAssociationTask,
        doc="Task used to associate DiaSources with SolarSystemObjects.",
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
    doSolarSystemAssociation = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Process SolarSystem objects through the pipeline.",
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()


class DrpAssociationPipeTask(pipeBase.PipelineTask):
    """Driver pipeline for loading DiaSource catalogs in a patch/tract
    region and associating them.
    """
    ConfigClass = DrpAssociationPipeConfig
    _DefaultName = "drpAssociation"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask('associator')
        if self.config.doSolarSystemAssociation:
            self.makeSubtask("solarSystemAssociator")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        inputs["tractId"] = butlerQC.quantum.dataId["tract"]
        inputs["patchId"] = butlerQC.quantum.dataId["patch"]
        inputs["idGenerator"] = self.config.idGenerator.apply(butlerQC.quantum.dataId)
        if not self.config.doSolarSystemAssociation:
            inputs["ssObjectTableRefs"] = []
            inputs["finalVisitSummaryRefs"] = []
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self,
            diaSourceTables,
            ssObjectTableRefs,
            skyMap,
            finalVisitSummaryRefs,
            tractId,
            patchId,
            idGenerator=None):
        """Trim DiaSources to the current Patch and run association.

        Takes in the set of DiaSource catalogs that covers the current patch,
        trims them to the dimensions of the patch, and [TODO: eventually]
        runs association on the concatenated DiaSource Catalog.

        Parameters
        ----------
        diaSourceTables : `list` of `lsst.daf.butler.DeferredDatasetHandle`
            Set of DiaSource catalogs potentially covering this patch/tract.
        ssObjectTableRefs: `list` of `lsst.daf.butler.DeferredDatasetHandle`
            Set of known SSO ephemerides potentially covering this patch/tract.
        skyMap : `lsst.skymap.BaseSkyMap`
            SkyMap defining the patch/tract
        visitInfoRefs : `list` of `lsst.daf.butler.DeferredDatasetHandle`
            Reference to finalVisitSummary of each exposure potentially
            covering this patch/tract, which contain visitInfo, bbox, and wcs
        tractId : `int`
            Id of current tract being processed.
        patchId : `int`
            Id of current patch being processed.
        idGenerator : `lsst.meas.base.IdGenerator`, optional
            Object that generates Object IDs and random number generator seeds.

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
        innerTractSkyRegion = skyInfo.tractInfo.getInnerSkyRegion()

        def visitDetectorPair(dataRef):
            return (dataRef.dataId["visit"], dataRef.dataId["detector"])

        # Keep track of our diaCats, ssObject cats, and finalVisitSummaries by their (visit, detector) IDs
        diaIdDict, ssObjectIdDict, finalVisitSummaryIdDict = {}, {}, {}
        for diaCatRef in diaSourceTables:
            diaIdDict[visitDetectorPair(diaCatRef)] = diaCatRef
        if self.config.doSolarSystemAssociation:
            for ssCatRef in ssObjectTableRefs:
                ssObjectIdDict[visitDetectorPair(ssCatRef)] = ssCatRef
            for finalVisitSummaryRef in finalVisitSummaryRefs:
                finalVisitSummaryIdDict[finalVisitSummaryRef.dataId["visit"]] = finalVisitSummaryRef

        diaSourceHistory, ssSourceHistory, unassociatedSsObjectHistory = [], [], []
        for visit, detector in diaIdDict:
            diaCatRef = diaIdDict[(visit, detector)]
            diaCat = diaCatRef.get()
            associatedSsSources, unassociatedSsObjects = None, None
            nSsSrc, nSsObj = 0, 0
            # Always false if ! self.config.doSolarSystemAssociation
            if all([(visit, detector) in ssObjectIdDict and visit in finalVisitSummaryIdDict]):
                # Get the ssCat and finalVisitSummary
                ssCat = ssObjectIdDict[(visit, detector)].get()
                finalVisitSummary = finalVisitSummaryIdDict[visit].get()
                # Get the exposure metadata from the detector's row in the finalVisitSummary table.
                visitInfo = finalVisitSummary.find(detector).visitInfo
                bbox = finalVisitSummary.find(detector).getBBox()
                wcs = finalVisitSummary.find(detector).wcs
                ssoAssocResult = self.solarSystemAssociator.run(
                    tb.Table.from_pandas(diaCat),
                    ssCat,
                    visitInfo,
                    bbox,
                    wcs,
                )

                associatedSsSources = ssoAssocResult.associatedSsSources
                associatedSsDiaSources = ssoAssocResult.ssoAssocDiaSources
                ssInTractPatch = self._trimToPatch(associatedSsSources.to_pandas(),
                                                   innerPatchBox,
                                                   innerTractSkyRegion,
                                                   skyInfo.wcs)
                associatedSsSources = associatedSsSources[ssInTractPatch]
                assocDiaSrcIds = set(associatedSsSources['diaSourceId'])
                diaSrcMask = [diaId in assocDiaSrcIds for diaId in associatedSsDiaSources['diaSourceId']]
                associatedSsDiaSources = associatedSsDiaSources[np.array(diaSrcMask)]

                unassociatedSsObjects = ssoAssocResult.unassociatedSsObjects
                ssObjInTractPatch = self._trimToPatch(unassociatedSsObjects.to_pandas(),
                                                      innerPatchBox,
                                                      innerTractSkyRegion,
                                                      skyInfo.wcs)
                unassociatedSsObjects = unassociatedSsObjects[ssObjInTractPatch]
                nSsSrc = ssInTractPatch.sum()
                nSsObj = ssObjInTractPatch.sum()
                if len(ssoAssocResult.unAssocDiaSources) > 0:
                    diaCat = ssoAssocResult.unAssocDiaSources.to_pandas()
                else:
                    diaCat = pd.DataFrame(columns=ssoAssocResult.unAssocDiaSources.columns)

            diaInTractPatch = self._trimToPatch(diaCat,
                                                innerPatchBox,
                                                innerTractSkyRegion,
                                                skyInfo.wcs)
            diaCat = diaCat[diaInTractPatch]

            nDiaSrc = diaInTractPatch.sum()

            self.log.info(
                "Read DiaSource catalog of length %i from visit %i, "
                "detector %i. Found %i sources within the patch/tract "
                "footprint, including %i associated with SSOs.",
                len(diaCat), diaCatRef.dataId["visit"],
                diaCatRef.dataId["detector"], nDiaSrc + nSsSrc, nSsSrc)

            if nDiaSrc > 0:
                diaSourceHistory.append(diaCat)
            if nSsSrc > 0:
                ssSourceHistory.append(associatedSsSources)
                diaSourceHistory.append(associatedSsDiaSources.to_pandas())
            if nSsObj > 0:
                unassociatedSsObjects['visit'] = visit
                unassociatedSsObjects['detector'] = detector
                unassociatedSsObjectHistory.append(unassociatedSsObjects)

        if diaSourceHistory:
            diaSourceHistoryCat = pd.concat(diaSourceHistory)
        else:
            diaSourceHistoryCat = diaCat.drop(diaCat.index)
        if self.config.doSolarSystemAssociation:
            nSsSrc, nSsObj = 0, 0
            if ssSourceHistory:
                ssSourceHistoryCat = tb.vstack(ssSourceHistory)
                ssSourceHistoryCat.remove_columns(['ra', 'dec'])
                nSsSrc = len(ssSourceHistoryCat)
            else:
                ssSourceHistoryCat = associatedSsSources  # Empty table?
            if unassociatedSsObjectHistory:
                unassociatedSsObjectHistoryCat = tb.vstack(unassociatedSsObjectHistory)
                nSsObj = len(unassociatedSsObjectHistoryCat)
            else:
                unassociatedSsObjectHistoryCat = unassociatedSsObjects  # Empty table?
            self.log.info("Found %i ssSources and %i missing ssObjects in patch %i, tract %i",
                          nSsSrc, nSsObj, patchId, tractId)
        else:
            ssSourceHistoryCat = None
            unassociatedSsObjectHistoryCat = None

        if (not diaSourceHistory) and not (self.config.doSolarSystemAssociation and ssSourceHistory):
            if not self.config.doWriteEmptyTables:
                raise pipeBase.NoWorkFound("Found no overlapping DIASources to associate.")

        self.log.info("Found %i DiaSources overlapping patch %i, tract %i",
                      len(diaSourceHistoryCat), patchId, tractId)

        assocResult = self.associator.run(diaSourceHistoryCat, idGenerator=idGenerator)

        self.log.info("Associated DiaSources into %i DiaObjects",
                      len(assocResult.diaObjects))

        if self.config.doAddDiaObjectCoords:
            assocResult.assocDiaSources = self._addDiaObjectCoords(assocResult.diaObjects,
                                                                   assocResult.assocDiaSources)

        return pipeBase.Struct(
            diaObjectTable=assocResult.diaObjects,
            assocDiaSourceTable=assocResult.assocDiaSources,
            associatedSsSources=ssSourceHistoryCat,
            unassociatedSsObjects=unassociatedSsObjectHistoryCat,
        )

    def _addDiaObjectCoords(self, objects, sources):
        obj = objects[['ra', 'dec']].rename(columns={"ra": "coord_ra", "dec": "coord_dec"})
        df = pd.merge(sources.reset_index(), obj, left_on='diaObjectId', right_index=True,
                      how='inner').set_index('diaSourceId')
        return df

    def _trimToPatch(self, cat, innerPatchBox, innerTractSkyRegion, wcs):
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
        isInPatch = np.zeros(len(cat), dtype=bool)

        for idx, row in cat.iterrows():
            spPoint = geom.SpherePoint(row["ra"], row["dec"], geom.degrees)
            pxCoord = wcs.skyToPixel(spPoint)
            ra_rad = np.deg2rad(row["ra"])
            dec_rad = np.deg2rad(row["dec"])

            isInPatch[idx] = innerPatchBox.contains(pxCoord) and innerTractSkyRegion.contains(ra_rad, dec_rad)

        return isInPatch
