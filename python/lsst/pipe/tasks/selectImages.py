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

__all__ = ["BaseSelectImagesTask", "BaseExposureInfo", "WcsSelectImagesTask", "PsfWcsSelectImagesTask",
           "DatabaseSelectImagesConfig", "BestSeeingSelectVisitsTask",
           "BestSeeingQuantileSelectVisitsTask"]

import numpy as np
import lsst.sphgeom
import lsst.pex.config as pexConfig
import lsst.pex.exceptions as pexExceptions
import lsst.geom as geom
import lsst.pipe.base as pipeBase
from lsst.skymap import BaseSkyMap
from lsst.daf.base import DateTime
from lsst.utils.timer import timeMethod


def _meetsVisitSummaryMinValues(visit, visitSummary, visitSummaryMinValues, logger=None):
    """Check if the visitSummary meets minimum values in visitSummaryMinValues.

    Parameters
    ----------
    visit : `int`
        Visit number.
    visitSummary : `lsst.afw.table.ExposureCatalog`
        Exposure catalog with per-detector summary information.
    visitSummaryMinValues : `dict`
        Dictionary with column names as keys and minimum allowed values as values.
    logger : `lsst.log.Logger`
        Logger to log debug and warning messages.

    Returns
    -------
    result : `bool`
        True if all columns in visitSummary meet the minimum values specified
        in visitSummaryMinValues, False otherwise.
    """
    for columnName, minThreshold in visitSummaryMinValues.items():
        # Get values for this column across all detectors with a valid WCS
        values = np.asarray([vs[columnName] for vs in visitSummary if vs.getWcs()])
        meanValue = np.nanmean(values)
        if meanValue < minThreshold:
            if logger is not None:
                logger.info(f'Rejecting visit {visit}: mean {columnName} ({meanValue:.3f}) '
                            f'is below the minimum threshold ({minThreshold:.3f}).')
            return False
    return True


class DatabaseSelectImagesConfig(pexConfig.Config):
    """Base configuration for subclasses of BaseSelectImagesTask that use a
    database.
    """

    host = pexConfig.Field(
        doc="Database server host name",
        dtype=str,
    )
    port = pexConfig.Field(
        doc="Database server port",
        dtype=int,
    )
    database = pexConfig.Field(
        doc="Name of database",
        dtype=str,
    )
    maxExposures = pexConfig.Field(
        doc="maximum exposures to select; intended for debugging; ignored if None",
        dtype=int,
        optional=True,
    )


class BaseExposureInfo(pipeBase.Struct):
    """Data about a selected exposure.

        Parameters
        ----------
        dataId : `dict`
            Data ID keys of exposure.
        coordList : `list` [`lsst.afw.geom.SpherePoint`]
            ICRS coordinates of the corners of the exposure
            plus any others items that are desired.
    """

    def __init__(self, dataId, coordList):
        super(BaseExposureInfo, self).__init__(dataId=dataId, coordList=coordList)


class BaseSelectImagesTask(pipeBase.Task):
    """Base task for selecting images suitable for coaddition.
    """

    ConfigClass = pexConfig.Config
    _DefaultName = "selectImages"

    @timeMethod
    def run(self, coordList):
        """Select images suitable for coaddition in a particular region.

        Parameters
        ----------
        coordList : `list` [`lsst.geom.SpherePoint`] or `None`
            List of coordinates defining region of interest; if `None`, then
            select all images subclasses may add additional keyword arguments,
            as required.

        Returns
        -------
        result : `pipeBase.Struct`
            Results as a struct with attributes:

            ``exposureInfoList``
                A list of exposure information objects (subclasses of
                BaseExposureInfo), which have at least the following fields:
                - dataId: Data ID dictionary (`dict`).
                - coordList: ICRS coordinates of the corners of the exposure.
                (`list` [`lsst.geom.SpherePoint`])
        """
        raise NotImplementedError()


def _extractKeyValue(dataList, keys=None):
    """Extract the keys and values from a list of dataIds.

    The input dataList is a list of objects that have 'dataId' members.
    This allows it to be used for both a list of data references and a
    list of ExposureInfo.

    Parameters
    ----------
    dataList : `Unknown`
    keys : `Unknown`

    Returns
    -------
    keys : `Unknown`
    values : `Unknown`

    Raises
    ------
    RuntimeError
        Raised if DataId keys are inconsistent.
    """
    assert len(dataList) > 0
    if keys is None:
        keys = sorted(dataList[0].dataId.keys())
    keySet = set(keys)
    values = list()
    for data in dataList:
        thisKeys = set(data.dataId.keys())
        if thisKeys != keySet:
            raise RuntimeError("DataId keys inconsistent: %s vs %s" % (keySet, thisKeys))
        values.append(tuple(data.dataId[k] for k in keys))
    return keys, values


class SelectStruct(pipeBase.Struct):
    """A container for data to be passed to the WcsSelectImagesTask.

    Parameters
    ----------
    dataRef : `Unknown`
        Data reference.
    wcs : `lsst.afw.geom.SkyWcs`
        Coordinate system definition (wcs).
    bbox : `lsst.geom.box.Box2I`
        Integer bounding box for image.
    """

    def __init__(self, dataRef, wcs, bbox):
        super(SelectStruct, self).__init__(dataRef=dataRef, wcs=wcs, bbox=bbox)


class WcsSelectImagesConfig(pexConfig.Config):
    excludeDetectors = pexConfig.ListField(
        dtype=int,
        default=[],
        doc="Detectors to exclude from selection.",
        optional=True,
    )


class WcsSelectImagesTask(BaseSelectImagesTask):
    """Select images using their Wcs.

    We use the "convexHull" method of lsst.sphgeom.ConvexPolygon to define
    polygons on the celestial sphere, and test the polygon of the
    patch for overlap with the polygon of the image.

    We use "convexHull" instead of generating a ConvexPolygon
    directly because the standard for the inputs to ConvexPolygon
    are pretty high and we don't want to be responsible for reaching them.
    """

    ConfigClass = WcsSelectImagesConfig
    _DefaultName = "WcsSelectImages"

    def run(self, wcsList, bboxList, coordList, dataIds=None, **kwargs):
        """Return indices of provided lists that meet the selection criteria.

        Parameters
        ----------
        wcsList : `list` [`lsst.afw.geom.SkyWcs`]
            Specifying the WCS's of the input ccds to be selected.
        bboxList : `list` [`lsst.geom.Box2I`]
            Specifying the bounding boxes of the input ccds to be selected.
        coordList : `list` [`lsst.geom.SpherePoint`]
            ICRS coordinates specifying boundary of the patch.
        dataIds : iterable [`lsst.daf.butler.dataId`] or `None`, optional
            An iterable object of dataIds which point to reference catalogs.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        result : `list` [`int`]
            The indices of selected ccds.
        """
        if dataIds is None:
            dataIds = [None] * len(wcsList)
        patchVertices = [coord.getVector() for coord in coordList]
        patchPoly = lsst.sphgeom.ConvexPolygon.convexHull(patchVertices)
        result = []
        for i, (imageWcs, imageBox, dataId) in enumerate(zip(wcsList, bboxList, dataIds)):
            if dataId and ("detector" in dataId) and (dataId["detector"] in self.config.excludeDetectors):
                self.log.info("De-selecting exposure %s because detector %s is exluded from processing",
                              dataId, dataId["detector"])
            elif imageWcs is None:
                self.log.info("De-selecting exposure %s:  Exposure has no WCS.", dataId)
            else:
                imageCorners = self.getValidImageCorners(imageWcs, imageBox, patchPoly, dataId)
                if imageCorners:
                    result.append(i)
        return result

    def getValidImageCorners(self, imageWcs, imageBox, patchPoly, dataId=None):
        """Return corners or `None` if bad.

        Parameters
        ----------
        imageWcs : `Unknown`
        imageBox : `Unknown`
        patchPoly : `Unknown`
        dataId : `Unknown`
        """
        try:
            imageCorners = [imageWcs.pixelToSky(pix) for pix in geom.Box2D(imageBox).getCorners()]
        except (pexExceptions.DomainError, pexExceptions.RuntimeError) as e:
            # Protecting ourselves from awful Wcs solutions in input images
            self.log.debug("WCS error in testing calexp %s (%s): deselecting", dataId, e)
            return None

        imagePoly = lsst.sphgeom.ConvexPolygon.convexHull([coord.getVector() for coord in imageCorners])
        if imagePoly is None:
            self.log.debug("Unable to create polygon from image %s: deselecting", dataId)
            return None

        if patchPoly.intersects(imagePoly):
            # "intersects" also covers "contains" or "is contained by"
            self.log.info("Selecting calexp %s", dataId)
            return imageCorners

        return None


class PsfWcsSelectImagesConnections(pipeBase.PipelineTaskConnections,
                                    dimensions=("tract", "patch", "skymap", "instrument", "visit"),
                                    defaultTemplates={"coaddName": "deep"}):
    pass


class PsfWcsSelectImagesConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=PsfWcsSelectImagesConnections):
    maxEllipResidual = pexConfig.Field(
        doc="Maximum median ellipticity residual",
        dtype=float,
        default=0.007,
        optional=True,
    )
    maxSizeScatter = pexConfig.Field(
        doc="Maximum scatter in the size residuals",
        dtype=float,
        optional=True,
    )
    maxScaledSizeScatter = pexConfig.Field(
        doc="Maximum scatter in the size residuals, scaled by the median size",
        dtype=float,
        default=0.019,
        optional=True,
    )
    maxPsfTraceRadiusDelta = pexConfig.Field(
        doc="Maximum delta (max - min) of model PSF trace radius values evaluated on a grid on "
        "the unmasked detector pixels (pixel).",
        dtype=float,
        default=0.7,
        optional=True,
    )
    maxPsfApFluxDelta = pexConfig.Field(
        doc="Maximum delta (max - min) of model PSF aperture flux (with aperture radius of "
        "max(2, 3*psfSigma)) values evaluated on a grid on the unmasked detector pixels (based "
        "on a normalized-to-one flux).",
        dtype=float,
        default=0.24,
        optional=True,
    )
    maxPsfApCorrSigmaScaledDelta = pexConfig.Field(
        doc="Maximum delta (max - min) of model PSF aperture correction values evaluated on a grid "
        "on the unmasked detector pixels scaled (divided) by the measured model psfSigma.",
        dtype=float,
        default=0.22,
        optional=True,
    )
    minNPsfStarPerBand = pexConfig.DictField(
        keytype=str,
        itemtype=int,
        default={
            "u": 6,
            "g": 6,
            "r": 6,
            "i": 6,
            "z": 6,
            "y": 6,
            "fallback": 6,
        },
        doc="Minimum number of PSF stars for the final PSF model to be considered "
        "well-constrained and suitible for inclusion in the coadd.  This number should "
        "take into consideration the spatial order used for the PSF model.  If the current "
        "band for the exposure is not included as a key in this dict, the value associated "
        "with the \"fallback\" key will be used.",
    )
    excludeDetectors = pexConfig.ListField(
        dtype=int,
        default=[],
        doc="Detectors to exclude from selection.",
        optional=True,
    )

    def validate(self):
        super().validate()
        if "fallback" not in self.minNPsfStarPerBand:
            msg = ("Must include a \"fallback\" key in the config.minNPsfStarPerBand config dict. "
                   f"It is currenly: {self.minNPsfStarPerBand}.")
            raise pexConfig.FieldValidationError(self.__class__.minNPsfStarPerBand, self, msg)


class PsfWcsSelectImagesTask(WcsSelectImagesTask):
    """Select images using their Wcs and cuts on the PSF properties.

        The PSF quality criteria are based on the size and ellipticity
        residuals from the adaptive second moments of the star and the PSF.

        The criteria are:
          - the median of the ellipticty residuals.
          - the robust scatter of the size residuals (using the median absolute
            deviation).
          - the robust scatter of the size residuals scaled by the square of
            the median size.
    """

    ConfigClass = PsfWcsSelectImagesConfig
    _DefaultName = "PsfWcsSelectImages"

    def run(self, wcsList, bboxList, coordList, visitSummary, dataIds=None, **kwargs):
        """Return indices of provided lists that meet the selection criteria.

        Parameters
        ----------
        wcsList : `list` [`lsst.afw.geom.SkyWcs`]
            Specifying the WCS's of the input ccds to be selected.
        bboxList : `list` [`lsst.geom.Box2I`]
            Specifying the bounding boxes of the input ccds to be selected.
        coordList : `list` [`lsst.geom.SpherePoint`]
            ICRS coordinates specifying boundary of the patch.
        visitSummary : `list` [`lsst.afw.table.ExposureCatalog`]
            containing the PSF shape information for the input ccds to be
            selected.
        dataIds : iterable [`lsst.daf.butler.dataId`] or `None`, optional
            An iterable object of dataIds which point to reference catalogs.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        goodPsf : `list` [`int`]
            The indices of selected ccds.
        """
        goodWcs = super(PsfWcsSelectImagesTask, self).run(wcsList=wcsList, bboxList=bboxList,
                                                          coordList=coordList, dataIds=dataIds)

        goodPsf = []

        for i, dataId in enumerate(dataIds):
            if i not in goodWcs:
                continue
            if self.isValid(visitSummary, dataId["detector"]):
                goodPsf.append(i)

        return goodPsf

    def isValid(self, visitSummary, detectorId):
        """Should this ccd be selected based on its PSF shape information.

        Parameters
        ----------
        visitSummary : `lsst.afw.table.ExposureCatalog`
            Exposure catalog with per-detector summary information.
        detectorId : `int`
            Detector identifier.

        Returns
        -------
        valid : `bool`
            True if selected.
        """
        row = visitSummary.find(detectorId)
        if row is None:
            # This is not listed, so it must be bad.
            self.log.warning("Removing detector %d because summary stats not available.", detectorId)
            return False

        band = row["band"]
        if band in self.config.minNPsfStarPerBand:
            minNPsfStar = self.config.minNPsfStarPerBand[band]
        else:
            minNPsfStar = self.config.minNPsfStarPerBand["fallback"]
        nPsfStar = row["nPsfStar"]
        medianE = np.sqrt(row["psfStarDeltaE1Median"]**2. + row["psfStarDeltaE2Median"]**2.)
        scatterSize = row["psfStarDeltaSizeScatter"]
        scaledScatterSize = row["psfStarScaledDeltaSizeScatter"]
        psfTraceRadiusDelta = row["psfTraceRadiusDelta"]
        psfApFluxDelta = row["psfApFluxDelta"]
        psfApCorrSigmaScaledDelta = row["psfApCorrSigmaScaledDelta"]

        valid = True
        if self.config.maxEllipResidual and not (medianE <= self.config.maxEllipResidual):
            self.log.info("Removing visit %d detector %d because median e residual too large: %f vs %f",
                          row["visit"], detectorId, medianE, self.config.maxEllipResidual)
            valid = False
        elif self.config.maxSizeScatter and not (scatterSize <= self.config.maxSizeScatter):
            self.log.info("Removing visit %d detector %d because size scatter too large: %f vs %f",
                          row["visit"], detectorId, scatterSize, self.config.maxSizeScatter)
            valid = False
        elif self.config.maxScaledSizeScatter and not (scaledScatterSize <= self.config.maxScaledSizeScatter):
            self.log.info("Removing visit %d detector %d because scaled size scatter too large: %f vs %f",
                          row["visit"], detectorId, scaledScatterSize, self.config.maxScaledSizeScatter)
            valid = False
        elif (
                self.config.maxPsfTraceRadiusDelta is not None
                and not (psfTraceRadiusDelta <= self.config.maxPsfTraceRadiusDelta)
        ):
            self.log.info(
                "Removing visit %d detector %d because max-min delta of model PSF trace radius values "
                "across the unmasked detector pixels is not finite or too large: %.3f vs %.3f (pixels)",
                row["visit"], detectorId, psfTraceRadiusDelta, self.config.maxPsfTraceRadiusDelta
            )
            valid = False
        elif (
                self.config.maxPsfApFluxDelta is not None
                and not (psfApFluxDelta <= self.config.maxPsfApFluxDelta)
        ):
            self.log.info(
                "Removing visit %d detector %d because max-min delta of model PSF aperture flux values "
                "across the unmasked detector pixels is not finite or too large: %.3f vs %.3f (pixels)",
                row["visit"], detectorId, psfApFluxDelta, self.config.maxPsfApFluxDelta
            )
            valid = False
        elif (
                self.config.maxPsfApCorrSigmaScaledDelta is not None
                and not (psfApCorrSigmaScaledDelta <= self.config.maxPsfApCorrSigmaScaledDelta)
        ):
            self.log.info(
                "Removing visit %d detector %d because max-min delta of the model PSF apterture correction "
                "values scaled (divided) by the psfSigma across the unmasked detector pixels is not "
                "finite or too large: %.3f vs %.3f (fatcor)",
                row["visit"], detectorId, psfApCorrSigmaScaledDelta, self.config.maxPsfApCorrSigmaScaledDelta
            )
            valid = False
        elif minNPsfStar and (nPsfStar < minNPsfStar):
            self.log.info(
                "Removing visit %d detector %d because the PSF model had too few stars: %d vs %d",
                row["visit"], detectorId, nPsfStar, minNPsfStar
            )
            valid = False

        return valid


class BestSeeingSelectVisitsConnections(pipeBase.PipelineTaskConnections,
                                        dimensions=("tract", "patch", "skymap", "band", "instrument"),
                                        defaultTemplates={"coaddName": "goodSeeing",
                                                          "calexpType": ""}):
    skyMap = pipeBase.connectionTypes.Input(
        doc="Input definition of geometry/bbox and projection/wcs for coadded exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    visitSummaries = pipeBase.connectionTypes.Input(
        doc="Per-visit consolidated exposure metadata",
        name="finalVisitSummary",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit",),
        multiple=True,
        deferLoad=True
    )
    goodVisits = pipeBase.connectionTypes.Output(
        doc="Selected visits to be coadded.",
        name="{coaddName}Visits",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "tract", "patch", "skymap", "band"),
    )


class BestSeeingSelectVisitsConfig(pipeBase.PipelineTaskConfig,
                                   pipelineConnections=BestSeeingSelectVisitsConnections):
    nVisitsMax = pexConfig.RangeField(
        dtype=int,
        doc="Maximum number of visits to select; use -1 to select all.",
        default=12,
        min=-1,
    )
    maxPsfFwhm = pexConfig.Field(
        dtype=float,
        doc="Maximum PSF FWHM (in arcseconds) to select",
        default=1.5,
        optional=True
    )
    minPsfFwhm = pexConfig.Field(
        dtype=float,
        doc="Minimum PSF FWHM (in arcseconds) to select",
        default=0.,
        optional=True
    )
    doConfirmOverlap = pexConfig.Field(
        dtype=bool,
        doc="Do remove visits that do not actually overlap the patch?",
        default=True,
    )
    minMJD = pexConfig.Field(
        dtype=float,
        doc="Minimum visit MJD to select",
        default=None,
        optional=True
    )
    maxMJD = pexConfig.Field(
        dtype=float,
        doc="Maximum visit MJD to select",
        default=None,
        optional=True
    )
    visitSummaryMinValues = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        doc=("Optional thresholds for visit selection. Keys are visit_summary column names"
             "(e.g. 'effTimeZeroPointScale'), and values are minimum allowed values."),
        default=None,
        optional=True
    )


class BestSeeingSelectVisitsTask(pipeBase.PipelineTask):
    """Select up to a maximum number of the best-seeing visits.

    Don't exceed the FWHM range specified by configs min(max)PsfFwhm.
    This Task is a port of the Gen2 image-selector used in the AP pipeline:
    BestSeeingSelectImagesTask. This Task selects full visits based on the
    average PSF of the entire visit.
    """

    ConfigClass = BestSeeingSelectVisitsConfig
    _DefaultName = 'bestSeeingSelectVisits'

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        quantumDataId = butlerQC.quantum.dataId
        outputs = self.run(**inputs, dataId=quantumDataId)
        butlerQC.put(outputs, outputRefs)

    def run(self, visitSummaries, skyMap, dataId):
        """Run task.

        Parameters
        ----------
        visitSummary : `list` [`lsst.pipe.base.connections.DeferredDatasetRef`]
            List of `lsst.pipe.base.connections.DeferredDatasetRef` of
            visitSummary tables of type `lsst.afw.table.ExposureCatalog`.
        skyMap : `lsst.skyMap.SkyMap`
           SkyMap for checking visits overlap patch.
        dataId : `dict` of dataId keys
            For retrieving patch info for checking visits overlap patch.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``goodVisits``
                A `dict` with selected visit ids as keys,
                so that it can be be saved as a StructuredDataDict.
                StructuredDataList's are currently limited.
        """
        if self.config.doConfirmOverlap:
            patchPolygon = self.makePatchPolygon(skyMap, dataId)

        inputVisits = [visitSummary.ref.dataId['visit'] for visitSummary in visitSummaries]
        fwhmSizes = []
        visits = []
        for visit, visitSummary in zip(inputVisits, visitSummaries):
            # read in one-by-one and only once. There may be hundreds
            visitSummary = visitSummary.get()

            # mjd is guaranteed to be the same for every detector in the
            # visitSummary.
            mjd = visitSummary[0].getVisitInfo().getDate().get(system=DateTime.MJD)

            pixelScales = np.array([vs['pixelScale'] for vs in visitSummary if vs.getWcs()])
            # psfSigma is PSF model determinant radius at chip center in pixels
            psfSigmas = np.array([vs['psfSigma'] for vs in visitSummary if vs.getWcs()])
            fwhm = np.nanmean(psfSigmas * pixelScales) * np.sqrt(8.*np.log(2.))

            if self.config.maxPsfFwhm and fwhm > self.config.maxPsfFwhm:
                continue
            if self.config.minPsfFwhm and fwhm < self.config.minPsfFwhm:
                continue
            if self.config.minMJD and mjd < self.config.minMJD:
                self.log.debug('MJD %f earlier than %.2f; rejecting', mjd, self.config.minMJD)
                continue
            if self.config.maxMJD and mjd > self.config.maxMJD:
                self.log.debug('MJD %f later than %.2f;  rejecting', mjd, self.config.maxMJD)
                continue
            if self.config.doConfirmOverlap and not self.doesIntersectPolygon(visitSummary, patchPolygon):
                continue
            if (self.config.visitSummaryMinValues
                and not _meetsVisitSummaryMinValues(visit, visitSummary, self.config.visitSummaryMinValues,
                                                    self.log)):
                continue

            fwhmSizes.append(fwhm)
            visits.append(visit)

        sortedVisits = [ind for (_, ind) in sorted(zip(fwhmSizes, visits))]
        if self.config.nVisitsMax < 0:
            output = sortedVisits
        else:
            output = sortedVisits[:self.config.nVisitsMax]

        if len(output) == 0:
            self.log.info("All images rejected in BestSeeingSelectVisitsTask.")
            raise pipeBase.NoWorkFound(f"No good images found for {dataId}")
        else:
            self.log.info(
                "%d images selected with FWHM range of %f--%f arcseconds",
                len(output),
                fwhmSizes[visits.index(output[0])],
                fwhmSizes[visits.index(output[-1])],
            )

        # In order to store as a StructuredDataDict, convert list to dict
        goodVisits = {key: True for key in output}
        return pipeBase.Struct(goodVisits=goodVisits)

    def makePatchPolygon(self, skyMap, dataId):
        """Return True if sky polygon overlaps visit.

        Parameters
        ----------
        skyMap : `lsst.afw.table.ExposureCatalog`
            Exposure catalog with per-detector geometry.
        dataId : `dict` of dataId keys
            For retrieving patch info.

        Returns
        -------
        result : `lsst.sphgeom.ConvexPolygon.convexHull`
            Polygon of patch's outer bbox.
        """
        wcs = skyMap[dataId['tract']].getWcs()
        bbox = skyMap[dataId['tract']][dataId['patch']].getOuterBBox()
        sphCorners = wcs.pixelToSky(lsst.geom.Box2D(bbox).getCorners())
        result = lsst.sphgeom.ConvexPolygon.convexHull([coord.getVector() for coord in sphCorners])
        return result

    def doesIntersectPolygon(self, visitSummary, polygon):
        """Return True if sky polygon overlaps visit.

        Parameters
        ----------
        visitSummary : `lsst.afw.table.ExposureCatalog`
            Exposure catalog with per-detector geometry.
        polygon :` lsst.sphgeom.ConvexPolygon.convexHull`
            Polygon to check overlap.

        Returns
        -------
        doesIntersect : `bool`
            True if the visit overlaps the polygon.
        """
        doesIntersect = False
        for detectorSummary in visitSummary:
            if (np.all(np.isfinite(detectorSummary['raCorners']))
                    and np.all(np.isfinite(detectorSummary['decCorners']))):
                corners = [lsst.geom.SpherePoint(ra, dec, units=lsst.geom.degrees).getVector()
                           for (ra, dec) in
                           zip(detectorSummary['raCorners'], detectorSummary['decCorners'])]
                detectorPolygon = lsst.sphgeom.ConvexPolygon.convexHull(corners)
                if detectorPolygon.intersects(polygon):
                    doesIntersect = True
                    break
        return doesIntersect


class BestSeeingQuantileSelectVisitsConfig(pipeBase.PipelineTaskConfig,
                                           pipelineConnections=BestSeeingSelectVisitsConnections):
    qMin = pexConfig.RangeField(
        doc="Lower bound of quantile range to select. Sorts visits by seeing from narrow to wide, "
            "and select those in the interquantile range (qMin, qMax). Set qMin to 0 for Best Seeing. "
            "This config should be changed from zero only for exploratory diffIm testing.",
        dtype=float,
        default=0,
        min=0,
        max=1,
    )
    qMax = pexConfig.RangeField(
        doc="Upper bound of quantile range to select. Sorts visits by seeing from narrow to wide, "
            "and select those in the interquantile range (qMin, qMax). Set qMax to 1 for Worst Seeing.",
        dtype=float,
        default=0.33,
        min=0,
        max=1,
    )
    nVisitsMin = pexConfig.Field(
        doc="The minimum number of visits to select, if qMin and qMax alone would have "
            "selected fewer. In regimes with many visits, at least this number of visits will be "
            "selected, superceding quantile when necessary. "
            "For example, if 10 visits cover this patch, qMin=0, qMax=0.33, and nVisitsMin=5, "
            "the best 5 visits will be selected, even though 5 > 0.33*10. "
            "In regimes with few visits, all available visits will be selected. "
            "For example, if 2 visits cover this patch and nVisitsMin=12, "
            "both visits will be selected regardless of qMin and qMax.",
        dtype=int,
        default=6,
    )
    doConfirmOverlap = pexConfig.Field(
        dtype=bool,
        doc="Do remove visits that do not actually overlap the patch?",
        default=True,
    )
    minMJD = pexConfig.Field(
        dtype=float,
        doc="Minimum visit MJD to select",
        default=None,
        optional=True
    )
    maxMJD = pexConfig.Field(
        dtype=float,
        doc="Maximum visit MJD to select",
        default=None,
        optional=True
    )
    visitSummaryMinValues = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        doc=("Optional thresholds for visit selection. Keys are visit_summary column names"
             "(e.g. 'effTimeZeroPointScale'), and values are minimum allowed values."),
        default=None,
        optional=True
    )


class BestSeeingQuantileSelectVisitsTask(BestSeeingSelectVisitsTask):
    """Select a quantile of the best-seeing visits.

    Selects the best (for example, third) full visits based on the average
    PSF width in the entire visit. It can also be used for difference imaging
    experiments that require templates with the worst seeing visits.
    For example, selecting the worst third can be acheived by
    changing the config parameters qMin to 0.66 and qMax to 1.
    """
    ConfigClass = BestSeeingQuantileSelectVisitsConfig
    _DefaultName = 'bestSeeingQuantileSelectVisits'

    def run(self, visitSummaries, skyMap, dataId):
        if self.config.doConfirmOverlap:
            patchPolygon = self.makePatchPolygon(skyMap, dataId)
        visits = np.array([visitSummary.ref.dataId['visit'] for visitSummary in visitSummaries])
        radius = np.empty(len(visits))
        intersects = np.full(len(visits), True)
        for i, visitSummary in enumerate(visitSummaries):
            # read in one-by-one and only once. There may be hundreds
            visitSummary = visitSummary.get()
            # psfSigma is PSF model determinant radius at chip center in pixels
            psfSigma = np.nanmedian([vs['psfSigma'] for vs in visitSummary])
            radius[i] = psfSigma
            if self.config.doConfirmOverlap:
                intersects[i] = self.doesIntersectPolygon(visitSummary, patchPolygon)
            if self.config.minMJD or self.config.maxMJD:
                # mjd is guaranteed to be the same for every detector in the
                # visitSummary.
                mjd = visitSummary[0].getVisitInfo().getDate().get(system=DateTime.MJD)
                aboveMin = mjd > self.config.minMJD if self.config.minMJD else True
                belowMax = mjd < self.config.maxMJD if self.config.maxMJD else True
                intersects[i] = intersects[i] and aboveMin and belowMax
            if (self.config.visitSummaryMinValues
                and not _meetsVisitSummaryMinValues(visits[i], visitSummary,
                                                    self.config.visitSummaryMinValues, self.log)):
                intersects[i] = False

        sortedVisits = [v for rad, v in sorted(zip(radius[intersects], visits[intersects]))]
        lowerBound = min(int(np.round(self.config.qMin*len(visits[intersects]))),
                         max(0, len(visits[intersects]) - self.config.nVisitsMin))
        upperBound = max(int(np.round(self.config.qMax*len(visits[intersects]))), self.config.nVisitsMin)

        # In order to store as a StructuredDataDict, convert list to dict
        goodVisits = {int(visit): True for visit in sortedVisits[lowerBound:upperBound]}
        return pipeBase.Struct(goodVisits=goodVisits)
