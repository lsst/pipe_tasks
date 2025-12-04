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

import os

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
from .schemaUtils import convertDataFrameToSdmSchema, readSdmSchemaFile
from .simpleAssociation import SimpleAssociationTask


class DrpAssociationPipeConnections(pipeBase.PipelineTaskConnections,
                                    dimensions=("tract", "patch", "skymap"),
                                    defaultTemplates={"coaddName": "deep",
                                                      "warpTypeSuffix": "",
                                                      "fakesType": ""}):
    diaSourceTables = pipeBase.connectionTypes.Input(
        doc="Set of catalogs of calibrated DiaSources.",
        name="{fakesType}{coaddName}Diff_diaSrcTable",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit", "detector"),
        deferLoad=True,
        multiple=True
    )
    ssObjectTableRefs = pipeBase.connectionTypes.Input(
        doc="Reference to catalogs of SolarSolarSystem objects expected to be "
            "observable in each (visit, detector).",
        name="preloaded_ss_object_visit",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
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
    doUseSchema = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Use an existing schema to coerce the data types of the output columns."
    )
    schemaDir = pexConfig.Field(
        dtype=str,
        doc="Path to the directory containing schema definitions.",
        default=os.path.join("${SDM_SCHEMAS_DIR}",
                             "yml"),
    )
    schemaFile = pexConfig.Field(
        dtype=str,
        doc="Yaml file specifying the schema of the output catalog.",
        default="lsstcam.yaml",
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

        if self.config.doUseSchema:
            schemaFile = os.path.join(self.config.schemaDir, self.config.schemaFile)
            self.schema = readSdmSchemaFile(schemaFile)
        else:
            self.schema = None
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
        ssObjectTableRefs : `list` of `lsst.daf.butler.DeferredDatasetHandle`
            Set of known SSO ephemerides potentially covering this patch/tract.
        skyMap : `lsst.skymap.BaseSkyMap`
            SkyMap defining the patch/tract
        finalVisitSummaryRefs : `list` of `lsst.daf.butler.DeferredDatasetHandle`
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
        outerPatchBox = geom.Box2D(skyInfo.patchInfo.getOuterBBox())
        innerTractSkyRegion = skyInfo.tractInfo.getInnerSkyRegion()

        # Keep track of our diaCats, ssObject cats, and finalVisitSummaries by their (visit, detector) IDs
        diaIdDict = prepareCatalogDict(diaSourceTables, useVisitDetector=True)
        ssObjectIdDict = prepareCatalogDict(ssObjectTableRefs, useVisitDetector=False)
        finalVisitSummaryIdDict = prepareCatalogDict(finalVisitSummaryRefs, useVisitDetector=False)

        diaSourceHistory, ssSourceHistory, unassociatedSsObjectHistory = [], [], []
        nSsSrc, nSsObj = 0, 0
        visits = set([v for v, _ in diaIdDict.keys()])
        for visit in visits:
            # visit summaries and Solar System catalogs are per-visit, so only
            # load them once for all detectors with that visit
            visitSummary = finalVisitSummaryIdDict[visit].get() if visit in finalVisitSummaryIdDict else None
            ssCat = ssObjectIdDict[visit].get() if visit in ssObjectIdDict else None
            detectors = [det for (v, det) in diaIdDict.keys() if v == visit]
            for detector in detectors:
                diaCat = diaIdDict[(visit, detector)].get()
                nDiaSrcIn = len(diaCat)
                if (ssCat is not None) and (visitSummary is not None):
                    ssoAssocResult = self.runSolarSystemAssociation(diaCat,
                                                                    ssCat.copy(),
                                                                    visitSummary=visitSummary,
                                                                    patchBbox=innerPatchBox,
                                                                    patchWcs=skyInfo.wcs,
                                                                    innerTractSkyRegion=innerTractSkyRegion,
                                                                    detector=detector,
                                                                    visit=visit,
                                                                    )

                    nSsSrc = len(ssoAssocResult.associatedSsSources)
                    nSsObj = len(ssoAssocResult.unassociatedSsObjects)
                    # If diaSources were associated with Solar System objects,
                    # remove them from the catalog so they won't create new
                    # diaObjects or be associated with other diaObjects.
                    diaCat = ssoAssocResult.unassociatedDiaSources
                else:
                    nSsSrc, nSsObj = 0, 0

                # Only trim diaSources to the outer bbox of the patch, so that
                # diaSources near the patch boundary can be associated.
                # DiaObjects will be trimmed to the inner patch bbox, and any
                # diaSources associated with dropped diaObjects will also be dropped
                diaInPatch = self._trimToPatch(diaCat.to_pandas(), outerPatchBox, skyInfo.wcs)

                nDiaSrc = diaInPatch.sum()

                self.log.info(
                    "Read DiaSource catalog of length %i from visit %i, "
                    "detector %i. Found %i sources within the patch/tract "
                    "footprint, including %i associated with SSOs.",
                    nDiaSrcIn, visit, detector, nDiaSrc + nSsSrc, nSsSrc)

                if nDiaSrc > 0:
                    diaSourceHistory.append(diaCat[diaInPatch])
                if nSsSrc > 0:
                    ssSourceHistory.append(ssoAssocResult.associatedSsSources)
                if nSsObj > 0:
                    unassociatedSsObjectHistory.append(ssoAssocResult.unassociatedSsObjects)

        # After looping over all of the detector-level catalogs that overlap the
        # patch, combine them into patch-level catalogs
        diaSourceHistoryCat = self._stackCatalogs(diaSourceHistory)
        if self.config.doSolarSystemAssociation:
            ssSourceHistoryCat = self._stackCatalogs(ssSourceHistory, remove_columns=['ra', 'dec'])
            nSsSrcTotal = len(ssSourceHistoryCat) if ssSourceHistoryCat else 0
            unassociatedSsObjectHistoryCat = self._stackCatalogs(unassociatedSsObjectHistory)
            nSsObjTotal = len(unassociatedSsObjectHistoryCat) if unassociatedSsObjectHistoryCat else 0
            self.log.info("Found %i ssSources and %i missing ssObjects in patch %i, tract %i",
                          nSsSrcTotal, nSsObjTotal, patchId, tractId)
        else:
            ssSourceHistoryCat = None
            unassociatedSsObjectHistoryCat = None

        if (not diaSourceHistory) and (not ssSourceHistory):
            if not self.config.doWriteEmptyTables:
                raise pipeBase.NoWorkFound("Found no overlapping DIASources to associate.")

        self.log.info("Found %i DiaSources overlapping patch %i, tract %i",
                      len(diaSourceHistoryCat), patchId, tractId)

        diaSourceTable = diaSourceHistoryCat.to_pandas()
        diaSourceTable.set_index("diaSourceId", drop=False)
        # Now run diaObject association on the stacked remaining diaSources
        assocResult = self.associator.run(diaSourceTable, idGenerator=idGenerator)

        # Drop any diaObjects that were created outside the inner region of the
        # patch. These will be associated in the overlapping patch instead.
        objectsInTractPatch = self._trimToPatch(assocResult.diaObjects,
                                                innerPatchBox,
                                                skyInfo.wcs,
                                                innerTractSkyRegion=innerTractSkyRegion)
        diaObjects = assocResult.diaObjects[objectsInTractPatch]
        # Instead of dropping diaSources based on their patch, assign them to a
        # patch based on whether their diaObject was inside. This means that
        # some diaSources in the patch catalog will actually have coordinates
        # just outside the patch.
        assocDiaSources = self.dropDiaSourceByDiaObjectId(assocResult.diaObjects[~objectsInTractPatch].index,
                                                          assocResult.assocDiaSources)

        self.log.info("Associated DiaSources into %i DiaObjects", len(diaObjects))

        if self.config.doAddDiaObjectCoords:
            assocDiaSources = self._addDiaObjectCoords(diaObjects, assocDiaSources)
        if self.config.doUseSchema:
            diaObjects = convertDataFrameToSdmSchema(self.schema, diaObjects, tableName="DiaObject")
            assocDiaSources = convertDataFrameToSdmSchema(self.schema, assocDiaSources, tableName="DiaSource")

        return pipeBase.Struct(
            diaObjectTable=diaObjects,
            assocDiaSourceTable=assocDiaSources,
            associatedSsSources=ssSourceHistoryCat,
            unassociatedSsObjects=unassociatedSsObjectHistoryCat,
        )

    def _stackCatalogs(self, catalogs, remove_columns=None, empty=None):
        """Stack a list of catalogs.

        Parameters
        ----------
        catalogs : `list` of `astropy.table.Table`
            Input catalogs with the same columns to be combined.
        remove_columns : `list` of `str` or None, optional
            List of column names to drop from the tables before stacking.

        Returns
        -------
        `astropy.table.Table`
            The combined catalog.
        """
        if catalogs:
            sourceHistory = tb.vstack(catalogs)
            if remove_columns is not None:
                sourceHistory.remove_columns(remove_columns)
            return sourceHistory
        else:
            return empty

    def runSolarSystemAssociation(self, diaCat, ssCat,
                                  visitSummary,
                                  patchBbox,
                                  patchWcs,
                                  innerTractSkyRegion,
                                  detector,
                                  visit,
                                  ):
        """Run Solar System object association and filter the results.

        Parameters
        ----------
        diaCat : `pandas.DataFrame`
            Catalog of detected diaSources on the image difference.
        ssCat : `astropy.table.Table`
            Catalog of predicted coordinates of known Solar System objects.
        visitSummary : `lsst.afw.table.ExposureCatalog`
            Table of calibration and metadata for all detectors in a visit.
        patchBbox : `lsst.geom.Box2D`
            Bounding box of the patch.
        patchWcs : `lsst.geom.SkyWcs`
            Wcs of the tract containing the patch.
        innerTractSkyRegion : `lsst.sphgeom.Box`
            Region defining the inner non-overlapping part of a tract.
        detector : `int`
            Detector number of the science exposure.
        visit : `int`
            Visit number of the science exposure.

        Returns
        -------
        ssoAssocResult : `lsst.pipe.base.Struct`
            Results struct with attributes:

            ``associatedSsSources``
                Table of DiaSources associated with Solar System objects.
                (`astropy.table.Table`)
            ``associatedSsDiaSources``
                Table of Solar System objects associated with DiaSources.
                (`astropy.table.Table`).
            ``unassociatedSsObjects``
                Table of Solar System objects in the patch not associated with
                any DiaSource (`astropy.table.Table`).
            ``unassociatedDiaSources``
                Table of DiaSources not associated with any Solar System object
                (`astropy.table.Table`).
        """
        # Get the exposure metadata from the detector's row in the visitSummary table.
        ssoAssocResult = self.solarSystemAssociator.run(
            diaCat,
            ssCat,
            visitInfo=visitSummary.find(detector).visitInfo,
            bbox=visitSummary.find(detector).getBBox(),
            wcs=visitSummary.find(detector).wcs,
        )

        ssInTractPatch = self._trimToPatch(ssoAssocResult.associatedSsSources.to_pandas(),
                                           patchBbox,
                                           patchWcs,
                                           innerTractSkyRegion=innerTractSkyRegion)
        associatedSsSources = ssoAssocResult.associatedSsSources[ssInTractPatch].copy()
        assocDiaSrcIds = set(associatedSsSources['diaSourceId'])
        diaSrcMask = [diaId in assocDiaSrcIds for diaId in ssoAssocResult.ssoAssocDiaSources['diaSourceId']]
        associatedSsDiaSources = ssoAssocResult.ssoAssocDiaSources[np.array(diaSrcMask)]

        ssObjInTractPatch = self._trimToPatch(ssoAssocResult.unassociatedSsObjects.to_pandas(),
                                              patchBbox,
                                              patchWcs,
                                              innerTractSkyRegion=innerTractSkyRegion)
        unassociatedSsObjects = ssoAssocResult.unassociatedSsObjects[ssObjInTractPatch].copy()
        # Update the table of Solar System objects that were not found with the
        # visit and detector where they were predicted.
        if len(unassociatedSsObjects) > 0:
            unassociatedSsObjects['visit'] = visit
            unassociatedSsObjects['detector'] = detector

        return pipeBase.Struct(
            associatedSsSources=associatedSsSources,
            associatedSsDiaSources=associatedSsDiaSources,
            unassociatedSsObjects=unassociatedSsObjects,
            unassociatedDiaSources=ssoAssocResult.unAssocDiaSources
        )

    def _addDiaObjectCoords(self, objects, sources):
        obj = objects[['ra', 'dec']].rename(columns={"ra": "coord_ra", "dec": "coord_dec"})
        df = pd.merge(sources.reset_index(), obj, left_on='diaObjectId', right_index=True,
                      how='inner').set_index('diaSourceId')
        return df

    def _trimToPatch(self, cat, patchBox, wcs, innerTractSkyRegion=None):
        """Create generator testing if a set of DiaSources are in the
        patch/tract.

        Parameters
        ----------
        cat : `pandas.DataFrame`
            Catalog of DiaSources to test within patch/tract.
        patchBox : `lsst.geom.Box2D`
            Bounding box of the patch.
        wcs : `lsst.geom.SkyWcs`
            Wcs of the tract.
        innerTractSkyRegion : `lsst.sphgeom.Box`, optional
            Region defining the inner non-overlapping part of a tract.

        Returns
        -------
        isInPatch : `numpy.ndarray`, (N,)
            Booleans representing if the DiaSources are contained within the
            current patch and tract.
        """
        isInPatch = np.zeros(len(cat), dtype=bool)

        for idx, row in cat.reset_index().iterrows():
            spPoint = geom.SpherePoint(row["ra"], row["dec"], geom.degrees)
            pxCoord = wcs.skyToPixel(spPoint)
            ra_rad = np.deg2rad(row["ra"])
            dec_rad = np.deg2rad(row["dec"])
            isInPatch[idx] = patchBox.contains(pxCoord)

            if innerTractSkyRegion is not None:
                isInPatch[idx] &= innerTractSkyRegion.contains(ra_rad, dec_rad)

        return isInPatch

    def dropDiaSourceByDiaObjectId(self, droppedDiaObjectIds, diaSources):
        """Drop diaSources with diaObject IDs in the supplied list.

        Parameters
        ----------
        droppedDiaObjectIds : `pandas.DataFrame`
            DiaObjectIds to match and drop from the list of diaSources.
        diaSources : `pandas.DataFrame`
            Catalog of diaSources to check and filter.

        Returns
        -------
        filteredDiaSources : `pandas.DataFrame`
            The input diaSources with any rows matching the listed diaObjectIds
            removed.
        """
        toDrop = diaSources['diaObjectId'].isin(droppedDiaObjectIds)

        # Keep only rows that do NOT match
        return diaSources.loc[~toDrop].copy()


def prepareCatalogDict(dataRefList, useVisitDetector=True):
    """Prepare lookup tables of the data references.

    Parameters
    ----------
    dataRefList : `list` of `lsst.daf.butler.DeferredDatasetHandle`
        The data references to make a lookup table for.
    useVisitDetector : `bool`, optional
        Use both visit and detector in the dict key? If False, use only visit.

    Returns
    -------
    `dict` of `lsst.daf.butler.DeferredDatasetHandle`
        Lookup table of the data references by visit (and optionally detector)
    """
    dataDict = {}

    if useVisitDetector:
        for dataRef in dataRefList:
            dataDict[(dataRef.dataId["visit"], dataRef.dataId["detector"])] = dataRef
    else:
        for dataRef in dataRefList:
            dataDict[dataRef.dataId["visit"]] = dataRef
    return dataDict
