#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import numpy as np
import lsst.sphgeom
import lsst.utils as utils
import lsst.pex.config as pexConfig
import lsst.pex.exceptions as pexExceptions
import lsst.geom as geom
import lsst.pipe.base as pipeBase
from lsst.skymap import BaseSkyMap
from lsst.daf.base import DateTime
from lsst.utils.timer import timeMethod

__all__ = ["BaseSelectImagesTask", "BaseExposureInfo", "WcsSelectImagesTask", "PsfWcsSelectImagesTask",
           "DatabaseSelectImagesConfig", "BestSeeingSelectVisitsTask",
           "BestSeeingQuantileSelectVisitsTask"]


class DatabaseSelectImagesConfig(pexConfig.Config):
    """Base configuration for subclasses of BaseSelectImagesTask that use a database"""
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
    """Data about a selected exposure
    """

    def __init__(self, dataId, coordList):
        """Create exposure information that can be used to generate data references

        The object has the following fields:
        - dataId: data ID of exposure (a dict)
        - coordList: ICRS coordinates of the corners of the exposure (list of lsst.geom.SpherePoint)
        plus any others items that are desired
        """
        super(BaseExposureInfo, self).__init__(dataId=dataId, coordList=coordList)


class BaseSelectImagesTask(pipeBase.Task):
    """Base task for selecting images suitable for coaddition
    """
    ConfigClass = pexConfig.Config
    _DefaultName = "selectImages"

    @timeMethod
    def run(self, coordList):
        """Select images suitable for coaddition in a particular region

        @param[in] coordList: list of coordinates defining region of interest; if None then select all images
        subclasses may add additional keyword arguments, as required

        @return a pipeBase Struct containing:
        - exposureInfoList: a list of exposure information objects (subclasses of BaseExposureInfo),
            which have at least the following fields:
            - dataId: data ID dictionary
            - coordList: ICRS coordinates of the corners of the exposure (list of lsst.geom.SpherePoint)
        """
        raise NotImplementedError()

    def _runArgDictFromDataId(self, dataId):
        """Extract keyword arguments for run (other than coordList) from a data ID

        @return keyword arguments for run (other than coordList), as a dict
        """
        raise NotImplementedError()

    def runDataRef(self, dataRef, coordList, makeDataRefList=True, selectDataList=[]):
        """Run based on a data reference

        This delegates to run() and _runArgDictFromDataId() to do the actual
        selection. In the event that the selectDataList is non-empty, this will
        be used to further restrict the selection, providing the user with
        additional control over the selection.

        @param[in] dataRef: data reference; must contain any extra keys needed by the subclass
        @param[in] coordList: list of coordinates defining region of interest; if None, search the whole sky
        @param[in] makeDataRefList: if True, return dataRefList
        @param[in] selectDataList: List of SelectStruct with dataRefs to consider for selection
        @return a pipeBase Struct containing:
        - exposureInfoList: a list of objects derived from ExposureInfo
        - dataRefList: a list of data references (None if makeDataRefList False)
        """
        runArgDict = self._runArgDictFromDataId(dataRef.dataId)
        exposureInfoList = self.run(coordList, **runArgDict).exposureInfoList

        if len(selectDataList) > 0 and len(exposureInfoList) > 0:
            # Restrict the exposure selection further
            ccdKeys, ccdValues = _extractKeyValue(exposureInfoList)
            inKeys, inValues = _extractKeyValue([s.dataRef for s in selectDataList], keys=ccdKeys)
            inValues = set(inValues)
            newExposureInfoList = []
            for info, ccdVal in zip(exposureInfoList, ccdValues):
                if ccdVal in inValues:
                    newExposureInfoList.append(info)
                else:
                    self.log.info("De-selecting exposure %s: not in selectDataList", info.dataId)
            exposureInfoList = newExposureInfoList

        if makeDataRefList:
            butler = dataRef.butlerSubset.butler
            dataRefList = [butler.dataRef(datasetType="calexp",
                                          dataId=expInfo.dataId,
                                          ) for expInfo in exposureInfoList]
        else:
            dataRefList = None

        return pipeBase.Struct(
            dataRefList=dataRefList,
            exposureInfoList=exposureInfoList,
        )


def _extractKeyValue(dataList, keys=None):
    """Extract the keys and values from a list of dataIds

    The input dataList is a list of objects that have 'dataId' members.
    This allows it to be used for both a list of data references and a
    list of ExposureInfo
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
    """A container for data to be passed to the WcsSelectImagesTask"""

    def __init__(self, dataRef, wcs, bbox):
        super(SelectStruct, self).__init__(dataRef=dataRef, wcs=wcs, bbox=bbox)


class WcsSelectImagesTask(BaseSelectImagesTask):
    """Select images using their Wcs

        We use the "convexHull" method of lsst.sphgeom.ConvexPolygon to define
        polygons on the celestial sphere, and test the polygon of the
        patch for overlap with the polygon of the image.

        We use "convexHull" instead of generating a ConvexPolygon
        directly because the standard for the inputs to ConvexPolygon
        are pretty high and we don't want to be responsible for reaching them.
        """

    def runDataRef(self, dataRef, coordList, makeDataRefList=True, selectDataList=[]):
        """Select images in the selectDataList that overlap the patch

        This method is the old entry point for the Gen2 commandline tasks and drivers
        Will be deprecated in v22.

        @param dataRef: Data reference for coadd/tempExp (with tract, patch)
        @param coordList: List of ICRS coordinates (lsst.geom.SpherePoint) specifying boundary of patch
        @param makeDataRefList: Construct a list of data references?
        @param selectDataList: List of SelectStruct, to consider for selection
        """
        dataRefList = []
        exposureInfoList = []

        patchVertices = [coord.getVector() for coord in coordList]
        patchPoly = lsst.sphgeom.ConvexPolygon.convexHull(patchVertices)

        for data in selectDataList:
            dataRef = data.dataRef
            imageWcs = data.wcs
            imageBox = data.bbox

            imageCorners = self.getValidImageCorners(imageWcs, imageBox, patchPoly, dataId=None)
            if imageCorners:
                dataRefList.append(dataRef)
                exposureInfoList.append(BaseExposureInfo(dataRef.dataId, imageCorners))

        return pipeBase.Struct(
            dataRefList=dataRefList if makeDataRefList else None,
            exposureInfoList=exposureInfoList,
        )

    def run(self, wcsList, bboxList, coordList, dataIds=None, **kwargs):
        """Return indices of provided lists that meet the selection criteria

        Parameters:
        -----------
        wcsList : `list` of `lsst.afw.geom.SkyWcs`
            specifying the WCS's of the input ccds to be selected
        bboxList : `list` of `lsst.geom.Box2I`
            specifying the bounding boxes of the input ccds to be selected
        coordList : `list` of `lsst.geom.SpherePoint`
            ICRS coordinates specifying boundary of the patch.

        Returns:
        --------
        result: `list` of `int`
            of indices of selected ccds
        """
        if dataIds is None:
            dataIds = [None] * len(wcsList)
        patchVertices = [coord.getVector() for coord in coordList]
        patchPoly = lsst.sphgeom.ConvexPolygon.convexHull(patchVertices)
        result = []
        for i, (imageWcs, imageBox, dataId) in enumerate(zip(wcsList, bboxList, dataIds)):
            imageCorners = self.getValidImageCorners(imageWcs, imageBox, patchPoly, dataId)
            if imageCorners:
                result.append(i)
        return result

    def getValidImageCorners(self, imageWcs, imageBox, patchPoly, dataId=None):
        "Return corners or None if bad"
        try:
            imageCorners = [imageWcs.pixelToSky(pix) for pix in geom.Box2D(imageBox).getCorners()]
        except (pexExceptions.DomainError, pexExceptions.RuntimeError) as e:
            # Protecting ourselves from awful Wcs solutions in input images
            self.log.debug("WCS error in testing calexp %s (%s): deselecting", dataId, e)
            return

        imagePoly = lsst.sphgeom.ConvexPolygon.convexHull([coord.getVector() for coord in imageCorners])
        if imagePoly is None:
            self.log.debug("Unable to create polygon from image %s: deselecting", dataId)
            return

        if patchPoly.intersects(imagePoly):
            # "intersects" also covers "contains" or "is contained by"
            self.log.info("Selecting calexp %s", dataId)
            return imageCorners


def sigmaMad(array):
    "Return median absolute deviation scaled to normally distributed data"
    return 1.4826*np.median(np.abs(array - np.median(array)))


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
        default=0.009,
        optional=True,
    )
    starSelection = pexConfig.Field(
        doc="select star with this field",
        dtype=str,
        default='calib_psf_used',
        deprecated=('This field has been moved to ComputeExposureSummaryStatsTask and '
                    'will be removed after v24.')
    )
    starShape = pexConfig.Field(
        doc="name of star shape",
        dtype=str,
        default='base_SdssShape',
        deprecated=('This field has been moved to ComputeExposureSummaryStatsTask and '
                    'will be removed after v24.')
    )
    psfShape = pexConfig.Field(
        doc="name of psf shape",
        dtype=str,
        default='base_SdssShape_psf',
        deprecated=('This field has been moved to ComputeExposureSummaryStatsTask and '
                    'will be removed after v24.')
    )
    doLegacyStarSelectionComputation = pexConfig.Field(
        doc="Perform the legacy star selection computations (for backwards compatibility)",
        dtype=bool,
        default=False,
        deprecated=("This field is here for backwards compatibility and will be "
                    "removed after v24.")
    )
    maxPsfTraceRadiusDelta = pexConfig.Field(
        doc="Maximum delta (max - min) of model PSF trace radius values evaluated on a grid on "
        "the unmasked detector pixels (pixel).",
        dtype=float,
        default=0.7,
        optional=True,
    )


class PsfWcsSelectImagesTask(WcsSelectImagesTask):
    """Select images using their Wcs and cuts on the PSF properties

        The PSF quality criteria are based on the size and ellipticity residuals from the
        adaptive second moments of the star and the PSF.

        The criteria are:
          - the median of the ellipticty residuals
          - the robust scatter of the size residuals (using the median absolute deviation)
          - the robust scatter of the size residuals scaled by the square of
            the median size
    """

    ConfigClass = PsfWcsSelectImagesConfig
    _DefaultName = "PsfWcsSelectImages"

    def runDataRef(self, dataRef, coordList, makeDataRefList=True, selectDataList=[]):
        """Select images in the selectDataList that overlap the patch and satisfy PSF quality critera.

        This method is the old entry point for the Gen2 commandline tasks and drivers
        Will be deprecated in v22.

        @param dataRef: Data reference for coadd/tempExp (with tract, patch)
        @param coordList: List of ICRS coordinates (lsst.geom.SpherePoint) specifying boundary of patch
        @param makeDataRefList: Construct a list of data references?
        @param selectDataList: List of SelectStruct, to consider for selection
        """
        result = super(PsfWcsSelectImagesTask, self).runDataRef(dataRef, coordList, makeDataRefList,
                                                                selectDataList)

        dataRefList = []
        exposureInfoList = []
        for dataRef, exposureInfo in zip(result.dataRefList, result.exposureInfoList):
            butler = dataRef.butlerSubset.butler
            srcCatalog = butler.get('src', dataRef.dataId)
            valid = self.isValidLegacy(srcCatalog, dataRef.dataId)
            if valid is False:
                continue

            dataRefList.append(dataRef)
            exposureInfoList.append(exposureInfo)

        return pipeBase.Struct(
            dataRefList=dataRefList,
            exposureInfoList=exposureInfoList,
        )

    def run(self, wcsList, bboxList, coordList, visitSummary, dataIds=None, srcList=None, **kwargs):
        """Return indices of provided lists that meet the selection criteria

        Parameters:
        -----------
        wcsList : `list` of `lsst.afw.geom.SkyWcs`
            specifying the WCS's of the input ccds to be selected
        bboxList : `list` of `lsst.geom.Box2I`
            specifying the bounding boxes of the input ccds to be selected
        coordList : `list` of `lsst.geom.SpherePoint`
            ICRS coordinates specifying boundary of the patch.
        visitSummary : `list` of `lsst.afw.table.ExposureCatalog`
            containing the PSF shape information for the input ccds to be selected.
        srcList : `list` of `lsst.afw.table.SourceCatalog`, optional
            containing the PSF shape information for the input ccds to be selected.
            This is only used if ``config.doLegacyStarSelectionComputation`` is
            True.

        Returns:
        --------
        goodPsf: `list` of `int`
            of indices of selected ccds
        """
        goodWcs = super(PsfWcsSelectImagesTask, self).run(wcsList=wcsList, bboxList=bboxList,
                                                          coordList=coordList, dataIds=dataIds)

        goodPsf = []

        if not self.config.doLegacyStarSelectionComputation:
            # Check for old inputs, and give a helpful error message if so.
            if 'nPsfStar' not in visitSummary[0].schema.getNames():
                raise RuntimeError("Old calexps detected. "
                                   "Please set config.doLegacyStarSelectionComputation=True for "
                                   "backwards compatibility.")

            for i, dataId in enumerate(dataIds):
                if i not in goodWcs:
                    continue
                if self.isValid(visitSummary, dataId["detector"]):
                    goodPsf.append(i)
        else:
            if dataIds is None:
                dataIds = [None] * len(srcList)
            for i, (srcCatalog, dataId) in enumerate(zip(srcList, dataIds)):
                if i not in goodWcs:
                    continue
                if self.isValidLegacy(srcCatalog, dataId):
                    goodPsf.append(i)

        return goodPsf

    def isValid(self, visitSummary, detectorId):
        """Should this ccd be selected based on its PSF shape information.

        Parameters
        ----------
        visitSummary : `lsst.afw.table.ExposureCatalog`
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

        medianE = np.sqrt(row["psfStarDeltaE1Median"]**2. + row["psfStarDeltaE2Median"]**2.)
        scatterSize = row["psfStarDeltaSizeScatter"]
        scaledScatterSize = row["psfStarScaledDeltaSizeScatter"]
        psfTraceRadiusDelta = row["psfTraceRadiusDelta"]

        valid = True
        if self.config.maxEllipResidual and medianE > self.config.maxEllipResidual:
            self.log.info("Removing visit %d detector %d because median e residual too large: %f vs %f",
                          row["visit"], detectorId, medianE, self.config.maxEllipResidual)
            valid = False
        elif self.config.maxSizeScatter and scatterSize > self.config.maxSizeScatter:
            self.log.info("Removing visit %d detector %d because size scatter too large: %f vs %f",
                          row["visit"], detectorId, scatterSize, self.config.maxSizeScatter)
            valid = False
        elif self.config.maxScaledSizeScatter and scaledScatterSize > self.config.maxScaledSizeScatter:
            self.log.info("Removing visit %d detector %d because scaled size scatter too large: %f vs %f",
                          row["visit"], detectorId, scaledScatterSize, self.config.maxScaledSizeScatter)
            valid = False
        elif (
                self.config.maxPsfTraceRadiusDelta
                and (
                    psfTraceRadiusDelta > self.config.maxPsfTraceRadiusDelta
                    or ~np.isfinite(psfTraceRadiusDelta)
                )
        ):
            self.log.info(
                "Removing visit %d detector %d because max-min delta of model PSF trace radius values "
                "across the unmasked detector pixels is not finite or too large: %.3f vs %.3f (pixels)",
                row["visit"], detectorId, psfTraceRadiusDelta, self.config.maxPsfTraceRadiusDelta
            )
            valid = False

        return valid

    def isValidLegacy(self, srcCatalog, dataId=None):
        """Should this ccd be selected based on its PSF shape information.

        This routine is only used in legacy processing (gen2 and
        backwards compatible old calexps) and should be removed after v24.

        Parameters
        ----------
        srcCatalog : `lsst.afw.table.SourceCatalog`
        dataId : `dict` of dataId keys, optional.
            Used only for logging. Defaults to None.

        Returns
        -------
        valid : `bool`
            True if selected.
        """
        mask = srcCatalog[self.config.starSelection]

        starXX = srcCatalog[self.config.starShape+'_xx'][mask]
        starYY = srcCatalog[self.config.starShape+'_yy'][mask]
        starXY = srcCatalog[self.config.starShape+'_xy'][mask]
        psfXX = srcCatalog[self.config.psfShape+'_xx'][mask]
        psfYY = srcCatalog[self.config.psfShape+'_yy'][mask]
        psfXY = srcCatalog[self.config.psfShape+'_xy'][mask]

        starSize = np.power(starXX*starYY - starXY**2, 0.25)
        starE1 = (starXX - starYY)/(starXX + starYY)
        starE2 = 2*starXY/(starXX + starYY)
        medianSize = np.median(starSize)

        psfSize = np.power(psfXX*psfYY - psfXY**2, 0.25)
        psfE1 = (psfXX - psfYY)/(psfXX + psfYY)
        psfE2 = 2*psfXY/(psfXX + psfYY)

        medianE1 = np.abs(np.median(starE1 - psfE1))
        medianE2 = np.abs(np.median(starE2 - psfE2))
        medianE = np.sqrt(medianE1**2 + medianE2**2)

        scatterSize = sigmaMad(starSize - psfSize)
        scaledScatterSize = scatterSize/medianSize**2

        valid = True
        if self.config.maxEllipResidual and medianE > self.config.maxEllipResidual:
            self.log.info("Removing visit %s because median e residual too large: %f vs %f",
                          dataId, medianE, self.config.maxEllipResidual)
            valid = False
        elif self.config.maxSizeScatter and scatterSize > self.config.maxSizeScatter:
            self.log.info("Removing visit %s because size scatter is too large: %f vs %f",
                          dataId, scatterSize, self.config.maxSizeScatter)
            valid = False
        elif self.config.maxScaledSizeScatter and scaledScatterSize > self.config.maxScaledSizeScatter:
            self.log.info("Removing visit %s because scaled size scatter is too large: %f vs %f",
                          dataId, scaledScatterSize, self.config.maxScaledSizeScatter)
            valid = False

        return valid


class BestSeeingSelectVisitsConnections(pipeBase.PipelineTaskConnections,
                                        dimensions=("tract", "patch", "skymap", "band", "instrument"),
                                        defaultTemplates={"coaddName": "goodSeeing"}):
    skyMap = pipeBase.connectionTypes.Input(
        doc="Input definition of geometry/bbox and projection/wcs for coadded exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    visitSummaries = pipeBase.connectionTypes.Input(
        doc="Per-visit consolidated exposure metadata from ConsolidateVisitSummaryTask",
        name="visitSummary",
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
        doc="Maximum number of visits to select",
        default=12,
        min=0
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


class BestSeeingSelectVisitsTask(pipeBase.PipelineTask):
    """Select up to a maximum number of the best-seeing visits

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
        """Run task

        Parameters:
        -----------
        visitSummary : `list`
            List of `lsst.pipe.base.connections.DeferredDatasetRef` of
            visitSummary tables of type `lsst.afw.table.ExposureCatalog`
        skyMap : `lsst.skyMap.SkyMap`
           SkyMap for checking visits overlap patch
        dataId : `dict` of dataId keys
            For retrieving patch info for checking visits overlap patch

        Returns
        -------
        result : `lsst.pipe.base.Struct`
           Result struct with components:

           - `goodVisits`: `dict` with selected visit ids as keys,
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

            # mjd is guaranteed to be the same for every detector in the visitSummary.
            mjd = visitSummary[0].getVisitInfo().getDate().get(system=DateTime.MJD)

            pixToArcseconds = [vs.getWcs().getPixelScale(vs.getBBox().getCenter()).asArcseconds()
                               for vs in visitSummary]
            # psfSigma is PSF model determinant radius at chip center in pixels
            psfSigmas = np.array([vs['psfSigma'] for vs in visitSummary])
            fwhm = np.nanmean(psfSigmas * pixToArcseconds) * np.sqrt(8.*np.log(2.))

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

            fwhmSizes.append(fwhm)
            visits.append(visit)

        sortedVisits = [ind for (_, ind) in sorted(zip(fwhmSizes, visits))]
        output = sortedVisits[:self.config.nVisitsMax]
        self.log.info("%d images selected with FWHM range of %d--%d arcseconds",
                      len(output), fwhmSizes[visits.index(output[0])], fwhmSizes[visits.index(output[-1])])

        # In order to store as a StructuredDataDict, convert list to dict
        goodVisits = {key: True for key in output}
        return pipeBase.Struct(goodVisits=goodVisits)

    def makePatchPolygon(self, skyMap, dataId):
        """Return True if sky polygon overlaps visit

        Parameters:
        -----------
        skyMap : `lsst.afw.table.ExposureCatalog`
            Exposure catalog with per-detector geometry
        dataId : `dict` of dataId keys
            For retrieving patch info

        Returns:
        --------
        result  :` lsst.sphgeom.ConvexPolygon.convexHull`
            Polygon of patch's outer bbox
        """
        wcs = skyMap[dataId['tract']].getWcs()
        bbox = skyMap[dataId['tract']][dataId['patch']].getOuterBBox()
        sphCorners = wcs.pixelToSky(lsst.geom.Box2D(bbox).getCorners())
        result = lsst.sphgeom.ConvexPolygon.convexHull([coord.getVector() for coord in sphCorners])
        return result

    def doesIntersectPolygon(self, visitSummary, polygon):
        """Return True if sky polygon overlaps visit

        Parameters:
        -----------
        visitSummary : `lsst.afw.table.ExposureCatalog`
            Exposure catalog with per-detector geometry
        polygon :` lsst.sphgeom.ConvexPolygon.convexHull`
            Polygon to check overlap

        Returns:
        --------
        doesIntersect: `bool`
            Does the visit overlap the polygon
        """
        doesIntersect = False
        for detectorSummary in visitSummary:
            corners = [lsst.geom.SpherePoint(ra, decl, units=lsst.geom.degrees).getVector() for (ra, decl) in
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
        doc="At least this number of visits selected and supercedes quantile. For example, if 10 visits "
            "cover this patch, qMin=0.33, and nVisitsMin=5, the best 5 visits will be selected.",
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


class BestSeeingQuantileSelectVisitsTask(BestSeeingSelectVisitsTask):
    """Select a quantile of the best-seeing visits

    Selects the best (for example, third) full visits based on the average
    PSF width in the entire visit. It can also be used for difference imaging
    experiments that require templates with the worst seeing visits.
    For example, selecting the worst third can be acheived by
    changing the config parameters qMin to 0.66 and qMax to 1.
    """
    ConfigClass = BestSeeingQuantileSelectVisitsConfig
    _DefaultName = 'bestSeeingQuantileSelectVisits'

    @utils.inheritDoc(BestSeeingSelectVisitsTask)
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
                # mjd is guaranteed to be the same for every detector in the visitSummary.
                mjd = visitSummary[0].getVisitInfo().getDate().get(system=DateTime.MJD)
                aboveMin = mjd > self.config.minMJD if self.config.minMJD else True
                belowMax = mjd < self.config.maxMJD if self.config.maxMJD else True
                intersects[i] = intersects[i] and aboveMin and belowMax

        sortedVisits = [v for rad, v in sorted(zip(radius[intersects], visits[intersects]))]
        lowerBound = min(int(np.round(self.config.qMin*len(visits[intersects]))),
                         max(0, len(visits[intersects]) - self.config.nVisitsMin))
        upperBound = max(int(np.round(self.config.qMax*len(visits[intersects]))), self.config.nVisitsMin)

        # In order to store as a StructuredDataDict, convert list to dict
        goodVisits = {int(visit): True for visit in sortedVisits[lowerBound:upperBound]}
        return pipeBase.Struct(goodVisits=goodVisits)
