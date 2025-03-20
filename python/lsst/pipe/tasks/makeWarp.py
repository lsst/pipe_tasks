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

__all__ = ["MakeWarpTask", "MakeWarpConfig"]

import logging
import numpy

import lsst.pex.config as pexConfig
import lsst.afw.image as afwImage
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
import lsst.utils as utils
import lsst.geom
from deprecated.sphinx import deprecated
from lsst.daf.butler import DeferredDatasetHandle
from lsst.meas.base import DetectorVisitIdGeneratorConfig
from lsst.meas.algorithms import CoaddPsf, CoaddPsfConfig, GaussianPsfFactory
from lsst.skymap import BaseSkyMap
from lsst.utils.timer import timeMethod
from .coaddBase import CoaddBaseTask, growValidPolygons, makeSkyInfo, reorderAndPadList
from .warpAndPsfMatch import WarpAndPsfMatchTask
from collections.abc import Iterable

log = logging.getLogger(__name__)


class MakeWarpConnections(pipeBase.PipelineTaskConnections,
                          dimensions=("tract", "patch", "skymap", "instrument", "visit"),
                          defaultTemplates={"coaddName": "deep",
                                            "calexpType": ""}):
    calExpList = connectionTypes.Input(
        doc="Input exposures to be resampled and optionally PSF-matched onto a SkyMap projection/patch",
        name="{calexpType}calexp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
        deferLoad=True,
    )
    backgroundList = connectionTypes.Input(
        doc="Input backgrounds to be added back into the calexp if bgSubtracted=False",
        name="calexpBackground",
        storageClass="Background",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
    )
    skyCorrList = connectionTypes.Input(
        doc="Input Sky Correction to be subtracted from the calexp if doApplySkyCorr=True",
        name="skyCorr",
        storageClass="Background",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
    )
    skyMap = connectionTypes.Input(
        doc="Input definition of geometry/bbox and projection/wcs for warped exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    direct = connectionTypes.Output(
        doc=("Output direct warped exposure (previously called CoaddTempExp), produced by resampling "
             "calexps onto the skyMap patch geometry."),
        name="{coaddName}Coadd_directWarp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
    )
    psfMatched = connectionTypes.Output(
        doc=("Output PSF-Matched warped exposure (previously called CoaddTempExp), produced by resampling "
             "calexps onto the skyMap patch geometry and PSF-matching to a model PSF."),
        name="{coaddName}Coadd_psfMatchedWarp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
    )
    visitSummary = connectionTypes.Input(
        doc="Input visit-summary catalog with updated calibration objects.",
        name="finalVisitSummary",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit",),
    )

    def __init__(self, *, config=None):
        if config.bgSubtracted:
            del self.backgroundList
        if not config.doApplySkyCorr:
            del self.skyCorrList
        if not config.makeDirect:
            del self.direct
        if not config.makePsfMatched:
            del self.psfMatched


@deprecated(reason="The Task corresponding to this Config is no longer in use. Will be removed after v29.",
            version="v29.0", category=FutureWarning)
class MakeWarpConfig(pipeBase.PipelineTaskConfig, CoaddBaseTask.ConfigClass,
                     pipelineConnections=MakeWarpConnections):
    """Config for MakeWarpTask."""

    warpAndPsfMatch = pexConfig.ConfigurableField(
        target=WarpAndPsfMatchTask,
        doc="Task to warp and PSF-match calexp",
    )
    doWrite = pexConfig.Field(
        doc="persist <coaddName>Coadd_<warpType>Warp",
        dtype=bool,
        default=True,
    )
    bgSubtracted = pexConfig.Field(
        doc="Work with a background subtracted calexp?",
        dtype=bool,
        default=True,
    )
    coaddPsf = pexConfig.ConfigField(
        doc="Configuration for CoaddPsf",
        dtype=CoaddPsfConfig,
    )
    makeDirect = pexConfig.Field(
        doc="Make direct Warp/Coadds",
        dtype=bool,
        default=True,
    )
    makePsfMatched = pexConfig.Field(
        doc="Make Psf-Matched Warp/Coadd?",
        dtype=bool,
        default=False,
    )
    modelPsf = GaussianPsfFactory.makeField(doc="Model Psf factory")
    useVisitSummaryPsf = pexConfig.Field(
        doc=(
            "If True, use the PSF model and aperture corrections from the 'visitSummary' connection. "
            "If False, use the PSF model and aperture corrections from the 'exposure' connection. "
        ),
        dtype=bool,
        default=True,
    )
    doWriteEmptyWarps = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Write out warps even if they are empty"
    )
    hasFakes = pexConfig.Field(
        doc="Should be set to True if fake sources have been inserted into the input data.",
        dtype=bool,
        default=False,
    )
    doApplySkyCorr = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Apply sky correction?",
    )
    idGenerator = DetectorVisitIdGeneratorConfig.make_field()

    def validate(self):
        CoaddBaseTask.ConfigClass.validate(self)

        if not self.makePsfMatched and not self.makeDirect:
            raise ValueError("At least one of config.makePsfMatched and config.makeDirect must be True")
        if self.warpAndPsfMatch.warp.cacheSize != self.coaddPsf.cacheSize:
            # This is an incomplete check: usually the CoaddPsf cache size
            # configured here in MakeWarpTask is superseded by the one in
            # AssembleCoaddTask.  A pipeline contract in the drp_pipe is
            # present to check that.
            raise ValueError("Image warping cache size and CoaddPSf warping cache size do not agree.")

    def setDefaults(self):
        CoaddBaseTask.ConfigClass.setDefaults(self)
        self.warpAndPsfMatch.warp.cacheSize = 0
        self.coaddPsf.cacheSize = 0


@deprecated(reason="The MakeWarpTask is replaced by MakeDirectWarpTask and MakePsfMatchedWarpTask. "
                   "This Task will be removed after v29.",
            version="v29.0", category=FutureWarning)
class MakeWarpTask(CoaddBaseTask):
    """Warp and optionally PSF-Match calexps onto an a common projection.

    Warp and optionally PSF-Match calexps onto a common projection, by
    performing the following operations:
    - Group calexps by visit/run
    - For each visit, generate a Warp by calling method @ref run.
      `run` loops over the visit's calexps calling
      `~lsst.pipe.tasks.warpAndPsfMatch.WarpAndPsfMatchTask` on each visit

    """
    ConfigClass = MakeWarpConfig
    _DefaultName = "makeWarp"

    def __init__(self, **kwargs):
        CoaddBaseTask.__init__(self, **kwargs)
        self.makeSubtask("warpAndPsfMatch")
        if self.config.hasFakes:
            self.calexpType = "fakes_calexp"
        else:
            self.calexpType = "calexp"

    @utils.inheritDoc(pipeBase.PipelineTask)
    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docstring to be augmented with info from PipelineTask.runQuantum
        """Notes
        -----
        Obtain the list of input detectors from calExpList.  Sort them by
        detector order (to ensure reproducibility).  Then ensure all input
        lists are in the same sorted detector order.
        """
        detectorOrder = [handle.datasetRef.dataId['detector'] for handle in inputRefs.calExpList]
        detectorOrder.sort()
        inputRefs = reorderRefs(inputRefs, detectorOrder, dataIdKey='detector')

        # Read in all inputs.
        inputs = butlerQC.get(inputRefs)

        # Construct skyInfo expected by `run`.  We remove the SkyMap itself
        # from the dictionary so we can pass it as kwargs later.
        skyMap = inputs.pop("skyMap")
        quantumDataId = butlerQC.quantum.dataId
        skyInfo = makeSkyInfo(skyMap, tractId=quantumDataId['tract'], patchId=quantumDataId['patch'])

        # Construct list of input DataIds expected by `run`.
        dataIdList = [ref.datasetRef.dataId for ref in inputRefs.calExpList]
        # Construct list of packed integer IDs expected by `run`.
        ccdIdList = [
            self.config.idGenerator.apply(dataId).catalog_id
            for dataId in dataIdList
        ]

        # Check early that the visitSummary contains everything we need.
        visitSummary = inputs["visitSummary"]
        bboxList = []
        wcsList = []
        for dataId in dataIdList:
            row = visitSummary.find(dataId["detector"])
            if row is None:
                bboxList.append(None)
                wcsList.append(None)
            else:
                bboxList.append(row.getBBox())
                wcsList.append(row.getWcs())
        inputs["bboxList"] = bboxList
        inputs["wcsList"] = wcsList

        # Do an initial selection on inputs with complete wcs/photoCalib info.
        # Qualifying calexps will be read in the following call.
        completeIndices = self._prepareCalibratedExposures(**inputs)
        inputs = self.filterInputs(indices=completeIndices, inputs=inputs)

        # Do another selection based on the configured selection task
        # (using updated WCSs to determine patch overlap if an external
        # calibration was applied).
        cornerPosList = lsst.geom.Box2D(skyInfo.bbox).getCorners()
        coordList = [skyInfo.wcs.pixelToSky(pos) for pos in cornerPosList]
        goodIndices = self.select.run(**inputs, coordList=coordList, dataIds=dataIdList)
        inputs = self.filterInputs(indices=goodIndices, inputs=inputs)

        # Extract integer visitId requested by `run`.
        visitId = dataIdList[0]["visit"]

        results = self.run(**inputs,
                           visitId=visitId,
                           ccdIdList=[ccdIdList[i] for i in goodIndices],
                           dataIdList=[dataIdList[i] for i in goodIndices],
                           skyInfo=skyInfo)
        if self.config.makeDirect and results.exposures["direct"] is not None:
            butlerQC.put(results.exposures["direct"], outputRefs.direct)
        if self.config.makePsfMatched and results.exposures["psfMatched"] is not None:
            butlerQC.put(results.exposures["psfMatched"], outputRefs.psfMatched)

    @timeMethod
    def run(self, calExpList, ccdIdList, skyInfo, visitId=0, dataIdList=None, **kwargs):
        """Create a Warp from inputs.

        We iterate over the multiple calexps in a single exposure to construct
        the warp (previously called a coaddTempExp) of that exposure to the
        supplied tract/patch.

        Pixels that receive no pixels are set to NAN; this is not correct
        (violates LSST algorithms group policy), but will be fixed up by
        interpolating after the coaddition.

        calExpList : `list` [ `lsst.afw.image.Exposure` ]
            List of single-detector input images that (may) overlap the patch
            of interest.
        skyInfo : `lsst.pipe.base.Struct`
            Struct from `~lsst.pipe.base.coaddBase.makeSkyInfo()` with
            geometric information about the patch.
        visitId : `int`
            Integer identifier for visit, for the table that will
            produce the CoaddPsf.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``exposures``
                A dictionary containing the warps requested:
                "direct": direct warp if ``config.makeDirect``
                "psfMatched": PSF-matched warp if ``config.makePsfMatched``
                (`dict`).
        """
        warpTypeList = self.getWarpTypeList()

        totGoodPix = {warpType: 0 for warpType in warpTypeList}
        didSetMetadata = {warpType: False for warpType in warpTypeList}
        warps = {warpType: self._prepareEmptyExposure(skyInfo) for warpType in warpTypeList}
        inputRecorder = {warpType: self.inputRecorder.makeCoaddTempExpRecorder(visitId, len(calExpList))
                         for warpType in warpTypeList}

        modelPsf = self.config.modelPsf.apply() if self.config.makePsfMatched else None
        if dataIdList is None:
            dataIdList = ccdIdList

        for calExpInd, (calExp, ccdId, dataId) in enumerate(zip(calExpList, ccdIdList, dataIdList)):
            self.log.info("Processing calexp %d of %d for this Warp: id=%s",
                          calExpInd+1, len(calExpList), dataId)
            try:
                warpedAndMatched = self.warpAndPsfMatch.run(calExp, modelPsf=modelPsf,
                                                            wcs=skyInfo.wcs, maxBBox=skyInfo.bbox,
                                                            makeDirect=self.config.makeDirect,
                                                            makePsfMatched=self.config.makePsfMatched)
            except Exception as e:
                self.log.warning("WarpAndPsfMatch failed for calexp %s; skipping it: %s", dataId, e)
                continue
            try:
                numGoodPix = {warpType: 0 for warpType in warpTypeList}
                for warpType in warpTypeList:
                    exposure = warpedAndMatched.getDict()[warpType]
                    if exposure is None:
                        continue
                    warp = warps[warpType]
                    numGoodPix[warpType] = coaddUtils.copyGoodPixels(
                        warp.getMaskedImage(), exposure.getMaskedImage(), self.getBadPixelMask())
                    totGoodPix[warpType] += numGoodPix[warpType]
                    self.log.debug("Calexp %s has %d good pixels in this patch (%.1f%%) for %s",
                                   dataId, numGoodPix[warpType],
                                   100.0*numGoodPix[warpType]/skyInfo.bbox.getArea(), warpType)
                    if numGoodPix[warpType] > 0 and not didSetMetadata[warpType]:
                        warp.info.id = exposure.info.id
                        warp.setPhotoCalib(exposure.getPhotoCalib())
                        warp.setFilter(exposure.getFilter())
                        warp.getInfo().setVisitInfo(exposure.getInfo().getVisitInfo())
                        # PSF replaced with CoaddPsf after loop if and only if
                        # creating direct warp.
                        warp.setPsf(exposure.getPsf())
                        didSetMetadata[warpType] = True

                    # Need inputRecorder for CoaddApCorrMap for both direct and
                    # PSF-matched.
                    inputRecorder[warpType].addCalExp(calExp, ccdId, numGoodPix[warpType])

            except Exception as e:
                self.log.warning("Error processing calexp %s; skipping it: %s", dataId, e)
                continue

        for warpType in warpTypeList:
            self.log.info("%sWarp has %d good pixels (%.1f%%)",
                          warpType, totGoodPix[warpType], 100.0*totGoodPix[warpType]/skyInfo.bbox.getArea())

            if totGoodPix[warpType] > 0 and didSetMetadata[warpType]:
                inputRecorder[warpType].finish(warps[warpType], totGoodPix[warpType])
                if warpType == "direct":
                    warps[warpType].setPsf(
                        CoaddPsf(inputRecorder[warpType].coaddInputs.ccds, skyInfo.wcs,
                                 self.config.coaddPsf.makeControl()))
                else:  # warpType == "psfMached"
                    growValidPolygons(
                        inputRecorder[warpType].coaddInputs,
                        -self.config.warpAndPsfMatch.psfMatch.kernel.active.kernelSize // 2,
                    )
            else:
                if not self.config.doWriteEmptyWarps:
                    # No good pixels. Exposure still empty.
                    warps[warpType] = None
                    # NoWorkFound is unnecessary as the downstream tasks will
                    # adjust the quantum accordingly.

        result = pipeBase.Struct(exposures=warps)
        return result

    def filterInputs(self, indices, inputs):
        """Filter task inputs by their indices.

        Parameters
        ----------
        indices : `list` [`int`]
        inputs : `dict` [`list`]
            A dictionary of input connections to be passed to run.

        Returns
        -------
        inputs : `dict` [`list`]
            Task inputs with their lists filtered by indices.
        """
        for key in inputs.keys():
            # Only down-select on list inputs
            if isinstance(inputs[key], list):
                inputs[key] = [inputs[key][ind] for ind in indices]
        return inputs

    def _prepareCalibratedExposures(self, *, visitSummary, calExpList=[], wcsList=None,
                                    backgroundList=None, skyCorrList=None, **kwargs):
        """Calibrate and add backgrounds to input calExpList in place.

        Parameters
        ----------
        visitSummary : `lsst.afw.table.ExposureCatalog`
            Exposure catalog with potentially all calibrations.  Attributes set
            to `None` are ignored.
        calExpList : `list` [`lsst.afw.image.Exposure` or
                     `lsst.daf.butler.DeferredDatasetHandle`]
            Sequence of single-epoch images (or deferred load handles for
            images) to be modified in place.  On return this always has images,
            not handles.
        wcsList : `list` [`lsst.afw.geom.SkyWcs` or `None` ]
            The WCSs of the calexps in ``calExpList``. These will be used to
            determine if the calexp should be used in the warp. The list is
            dynamically updated with the WCSs from the visitSummary.
        backgroundList : `list` [`lsst.afw.math.BackgroundList`], optional
            Sequence of backgrounds to be added back in if bgSubtracted=False.
        skyCorrList : `list` [`lsst.afw.math.BackgroundList`], optional
            Sequence of background corrections to be subtracted if
            doApplySkyCorr=True.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        indices : `list` [`int`]
            Indices of ``calExpList`` and friends that have valid
            photoCalib/skyWcs.
        """
        wcsList = len(calExpList)*[None] if wcsList is None else wcsList
        backgroundList = len(calExpList)*[None] if backgroundList is None else backgroundList
        skyCorrList = len(calExpList)*[None] if skyCorrList is None else skyCorrList

        indices = []
        for index, (calexp, background, skyCorr) in enumerate(zip(calExpList,
                                                                  backgroundList,
                                                                  skyCorrList)):
            if isinstance(calexp, DeferredDatasetHandle):
                calexp = calexp.get()

            if not self.config.bgSubtracted:
                calexp.maskedImage += background.getImage()

            detectorId = calexp.info.getDetector().getId()

            # Load all calibrations from visitSummary.
            row = visitSummary.find(detectorId)
            if row is None:
                self.log.warning(
                    "Detector id %d has no row in the visitSummary and will "
                    "not be used in the warp", detectorId,
                )
                continue
            if (photoCalib := row.getPhotoCalib()) is not None:
                calexp.setPhotoCalib(photoCalib)
            else:
                self.log.warning(
                    "Detector id %d for visit %d has None for photoCalib in the visitSummary and will "
                    "not be used in the warp", detectorId, row["visit"],
                )
                continue
            if (skyWcs := row.getWcs()) is not None:
                calexp.setWcs(skyWcs)
                wcsList[index] = skyWcs
            else:
                self.log.warning(
                    "Detector id %d for visit %d has None for wcs in the visitSummary and will "
                    "not be used in the warp", detectorId, row["visit"],
                )
                continue
            if self.config.useVisitSummaryPsf:
                if (psf := row.getPsf()) is not None:
                    calexp.setPsf(psf)
                else:
                    self.log.warning(
                        "Detector id %d for visit %d has None for psf in the visitSummary and will "
                        "not be used in the warp", detectorId, row["visit"],
                    )
                    continue
                if (apCorrMap := row.getApCorrMap()) is not None:
                    calexp.info.setApCorrMap(apCorrMap)
                else:
                    self.log.warning(
                        "Detector id %d for visit %d has None for apCorrMap in the visitSummary and will "
                        "not be used in the warp", detectorId, row["visit"],
                    )
                    continue
            else:
                if calexp.getPsf() is None:
                    self.log.warning(
                        "Detector id %d for visit %d has None for psf for the calexp and will "
                        "not be used in the warp", detectorId, row["visit"],
                    )
                    continue
                if calexp.info.getApCorrMap() is None:
                    self.log.warning(
                        "Detector id %d for visit %d has None for apCorrMap in the calexp and will "
                        "not be used in the warp", detectorId, row["visit"],
                    )
                    continue

            # Apply skycorr
            if self.config.doApplySkyCorr:
                calexp.maskedImage -= skyCorr.getImage()

            # Calibrate the image.
            calexp.maskedImage = photoCalib.calibrateImage(calexp.maskedImage)
            # This new PhotoCalib shouldn't need to be used, but setting it
            # here to reflect the fact that the image now has calibrated pixels
            # might help avoid future bugs.
            calexp.setPhotoCalib(afwImage.PhotoCalib(1.0))

            indices.append(index)
            calExpList[index] = calexp

        return indices

    @staticmethod
    def _prepareEmptyExposure(skyInfo):
        """Produce an empty exposure for a given patch.

        Parameters
        ----------
        skyInfo : `lsst.pipe.base.Struct`
            Struct from `~lsst.pipe.base.coaddBase.makeSkyInfo()` with
            geometric information about the patch.

        Returns
        -------
        exp : `lsst.afw.image.exposure.ExposureF`
            An empty exposure for a given patch.
        """
        exp = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
        exp.getMaskedImage().set(numpy.nan, afwImage.Mask
                                 .getPlaneBitMask("NO_DATA"), numpy.inf)
        exp.setPhotoCalib(afwImage.PhotoCalib(1.0))
        exp.metadata["BUNIT"] = "nJy"
        return exp

    def getWarpTypeList(self):
        """Return list of requested warp types per the config.
        """
        warpTypeList = []
        if self.config.makeDirect:
            warpTypeList.append("direct")
        if self.config.makePsfMatched:
            warpTypeList.append("psfMatched")
        return warpTypeList


def reorderRefs(inputRefs, outputSortKeyOrder, dataIdKey):
    """Reorder inputRefs per outputSortKeyOrder.

    Any inputRefs which are lists will be resorted per specified key e.g.,
    'detector.' Only iterables will be reordered, and values can be of type
    `lsst.pipe.base.connections.DeferredDatasetRef` or
    `lsst.daf.butler.core.datasets.ref.DatasetRef`.

    Returned lists of refs have the same length as the outputSortKeyOrder.
    If an outputSortKey not in the inputRef, then it will be padded with None.
    If an inputRef contains an inputSortKey that is not in the
    outputSortKeyOrder it will be removed.

    Parameters
    ----------
    inputRefs : `lsst.pipe.base.connections.QuantizedConnection`
        Input references to be reordered and padded.
    outputSortKeyOrder : `iterable`
        Iterable of values to be compared with inputRef's dataId[dataIdKey].
    dataIdKey : `str`
        The data ID key in the dataRefs to compare with the outputSortKeyOrder.

    Returns
    -------
    inputRefs : `lsst.pipe.base.connections.QuantizedConnection`
        Quantized Connection with sorted DatasetRef values sorted if iterable.
    """
    for connectionName, refs in inputRefs:
        if isinstance(refs, Iterable):
            if hasattr(refs[0], "dataId"):
                inputSortKeyOrder = [ref.dataId[dataIdKey] for ref in refs]
            else:
                inputSortKeyOrder = [handle.datasetRef.dataId[dataIdKey] for handle in refs]
            if inputSortKeyOrder != outputSortKeyOrder:
                setattr(inputRefs, connectionName,
                        reorderAndPadList(refs, inputSortKeyOrder, outputSortKeyOrder))
    return inputRefs
