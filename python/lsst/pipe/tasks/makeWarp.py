#
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011, 2012 LSST Corporation.
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
import logging
import numpy

import lsst.pex.config as pexConfig
import lsst.afw.image as afwImage
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
import lsst.utils as utils
import lsst.geom
from lsst.meas.algorithms import CoaddPsf, CoaddPsfConfig
from lsst.skymap import BaseSkyMap
from lsst.utils.timer import timeMethod
from .coaddBase import CoaddBaseTask, makeSkyInfo, reorderAndPadList
from .warpAndPsfMatch import WarpAndPsfMatchTask
from collections.abc import Iterable

__all__ = ["MakeWarpTask", "MakeWarpConfig"]

log = logging.getLogger(__name__)


class MakeWarpConnections(pipeBase.PipelineTaskConnections,
                          dimensions=("tract", "patch", "skymap", "instrument", "visit"),
                          defaultTemplates={"coaddName": "deep",
                                            "skyWcsName": "jointcal",
                                            "photoCalibName": "fgcm",
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
    externalSkyWcsTractCatalog = connectionTypes.Input(
        doc=("Per-tract, per-visit wcs calibrations.  These catalogs use the detector "
             "id for the catalog id, sorted on id for fast lookup."),
        name="{skyWcsName}SkyWcsCatalog",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit", "tract"),
    )
    externalSkyWcsGlobalCatalog = connectionTypes.Input(
        doc=("Per-visit wcs calibrations computed globally (with no tract information). "
             "These catalogs use the detector id for the catalog id, sorted on id for "
             "fast lookup."),
        name="{skyWcsName}SkyWcsCatalog",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
    )
    externalPhotoCalibTractCatalog = connectionTypes.Input(
        doc=("Per-tract, per-visit photometric calibrations.  These catalogs use the "
             "detector id for the catalog id, sorted on id for fast lookup."),
        name="{photoCalibName}PhotoCalibCatalog",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit", "tract"),
    )
    externalPhotoCalibGlobalCatalog = connectionTypes.Input(
        doc=("Per-visit photometric calibrations computed globally (with no tract "
             "information).  These catalogs use the detector id for the catalog id, "
             "sorted on id for fast lookup."),
        name="{photoCalibName}PhotoCalibCatalog",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
    )
    finalizedPsfApCorrCatalog = connectionTypes.Input(
        doc=("Per-visit finalized psf models and aperture correction maps. "
             "These catalogs use the detector id for the catalog id, "
             "sorted on id for fast lookup."),
        name="finalized_psf_ap_corr_catalog",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
    )
    direct = connectionTypes.Output(
        doc=("Output direct warped exposure (previously called CoaddTempExp), produced by resampling ",
             "calexps onto the skyMap patch geometry."),
        name="{coaddName}Coadd_directWarp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
    )
    psfMatched = connectionTypes.Output(
        doc=("Output PSF-Matched warped exposure (previously called CoaddTempExp), produced by resampling ",
             "calexps onto the skyMap patch geometry and PSF-matching to a model PSF."),
        name="{coaddName}Coadd_psfMatchedWarp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
    )
    # TODO DM-28769, have selectImages subtask indicate which connections they need:
    wcsList = connectionTypes.Input(
        doc="WCSs of calexps used by SelectImages subtask to determine if the calexp overlaps the patch",
        name="{calexpType}calexp.wcs",
        storageClass="Wcs",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
    )
    bboxList = connectionTypes.Input(
        doc="BBoxes of calexps used by SelectImages subtask to determine if the calexp overlaps the patch",
        name="{calexpType}calexp.bbox",
        storageClass="Box2I",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
    )
    visitSummary = connectionTypes.Input(
        doc="Consolidated exposure metadata from ConsolidateVisitSummaryTask",
        name="{calexpType}visitSummary",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit",),
    )
    srcList = connectionTypes.Input(
        doc="Source catalogs used by PsfWcsSelectImages subtask to further select on PSF stability",
        name="src",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if config.bgSubtracted:
            self.inputs.remove("backgroundList")
        if not config.doApplySkyCorr:
            self.inputs.remove("skyCorrList")
        if config.doApplyExternalSkyWcs:
            if config.useGlobalExternalSkyWcs:
                self.inputs.remove("externalSkyWcsTractCatalog")
            else:
                self.inputs.remove("externalSkyWcsGlobalCatalog")
        else:
            self.inputs.remove("externalSkyWcsTractCatalog")
            self.inputs.remove("externalSkyWcsGlobalCatalog")
        if config.doApplyExternalPhotoCalib:
            if config.useGlobalExternalPhotoCalib:
                self.inputs.remove("externalPhotoCalibTractCatalog")
            else:
                self.inputs.remove("externalPhotoCalibGlobalCatalog")
        else:
            self.inputs.remove("externalPhotoCalibTractCatalog")
            self.inputs.remove("externalPhotoCalibGlobalCatalog")
        if not config.doApplyFinalizedPsf:
            self.inputs.remove("finalizedPsfApCorrCatalog")
        if not config.makeDirect:
            self.outputs.remove("direct")
        if not config.makePsfMatched:
            self.outputs.remove("psfMatched")
        # TODO DM-28769: add connection per selectImages connections
        if config.select.target != lsst.pipe.tasks.selectImages.PsfWcsSelectImagesTask:
            self.inputs.remove("visitSummary")
            self.inputs.remove("srcList")
        elif not config.select.doLegacyStarSelectionComputation:
            # Remove backwards-compatibility connections.
            self.inputs.remove("srcList")


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
    doApplyFinalizedPsf = pexConfig.Field(
        doc="Whether to apply finalized psf models and aperture correction map.",
        dtype=bool,
        default=True,
    )

    def validate(self):
        CoaddBaseTask.ConfigClass.validate(self)

        if not self.makePsfMatched and not self.makeDirect:
            raise RuntimeError("At least one of config.makePsfMatched and config.makeDirect must be True")
        if self.doPsfMatch:
            # Backwards compatibility.
            log.warning("Config doPsfMatch deprecated. Setting makePsfMatched=True and makeDirect=False")
            self.makePsfMatched = True
            self.makeDirect = False

    def setDefaults(self):
        CoaddBaseTask.ConfigClass.setDefaults(self)
        self.warpAndPsfMatch.psfMatch.kernel.active.kernelSize = self.matchingKernelSize


class MakeWarpTask(CoaddBaseTask):
    """Warp and optionally PSF-Match calexps onto an a common projection
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
        """
        Notes
        ----
        Construct warps for requested warp type for single epoch
        """
        # Obtain the list of input detectors from calExpList.  Sort them by
        # detector order (to ensure reproducibility).  Then ensure all input
        # lists are in the same sorted detector order.
        detectorOrder = [ref.datasetRef.dataId['detector'] for ref in inputRefs.calExpList]
        detectorOrder.sort()
        inputRefs = reorderRefs(inputRefs, detectorOrder, dataIdKey='detector')

        # Read in all inputs.
        inputs = butlerQC.get(inputRefs)

        # Construct skyInfo expected by `run`.  We remove the SkyMap itself
        # from the dictionary so we can pass it as kwargs later.
        skyMap = inputs.pop("skyMap")
        quantumDataId = butlerQC.quantum.dataId
        skyInfo = makeSkyInfo(skyMap, tractId=quantumDataId['tract'], patchId=quantumDataId['patch'])

        # Construct list of input DataIds expected by `run`
        dataIdList = [ref.datasetRef.dataId for ref in inputRefs.calExpList]
        # Construct list of packed integer IDs expected by `run`
        ccdIdList = [dataId.pack("visit_detector") for dataId in dataIdList]

        # Run the selector and filter out calexps that were not selected
        # primarily because they do not overlap the patch
        cornerPosList = lsst.geom.Box2D(skyInfo.bbox).getCorners()
        coordList = [skyInfo.wcs.pixelToSky(pos) for pos in cornerPosList]
        goodIndices = self.select.run(**inputs, coordList=coordList, dataIds=dataIdList)
        inputs = self.filterInputs(indices=goodIndices, inputs=inputs)

        # Read from disk only the selected calexps
        inputs['calExpList'] = [ref.get() for ref in inputs['calExpList']]

        # Extract integer visitId requested by `run`
        visits = [dataId['visit'] for dataId in dataIdList]
        visitId = visits[0]

        if self.config.doApplyExternalSkyWcs:
            if self.config.useGlobalExternalSkyWcs:
                externalSkyWcsCatalog = inputs.pop("externalSkyWcsGlobalCatalog")
            else:
                externalSkyWcsCatalog = inputs.pop("externalSkyWcsTractCatalog")
        else:
            externalSkyWcsCatalog = None

        if self.config.doApplyExternalPhotoCalib:
            if self.config.useGlobalExternalPhotoCalib:
                externalPhotoCalibCatalog = inputs.pop("externalPhotoCalibGlobalCatalog")
            else:
                externalPhotoCalibCatalog = inputs.pop("externalPhotoCalibTractCatalog")
        else:
            externalPhotoCalibCatalog = None

        if self.config.doApplyFinalizedPsf:
            finalizedPsfApCorrCatalog = inputs.pop("finalizedPsfApCorrCatalog")
        else:
            finalizedPsfApCorrCatalog = None

        completeIndices = self.prepareCalibratedExposures(**inputs,
                                                          externalSkyWcsCatalog=externalSkyWcsCatalog,
                                                          externalPhotoCalibCatalog=externalPhotoCalibCatalog,
                                                          finalizedPsfApCorrCatalog=finalizedPsfApCorrCatalog)
        # Redo the input selection with inputs with complete wcs/photocalib info.
        inputs = self.filterInputs(indices=completeIndices, inputs=inputs)

        results = self.run(**inputs, visitId=visitId,
                           ccdIdList=[ccdIdList[i] for i in goodIndices],
                           dataIdList=[dataIdList[i] for i in goodIndices],
                           skyInfo=skyInfo)
        if self.config.makeDirect and results.exposures["direct"] is not None:
            butlerQC.put(results.exposures["direct"], outputRefs.direct)
        if self.config.makePsfMatched and results.exposures["psfMatched"] is not None:
            butlerQC.put(results.exposures["psfMatched"], outputRefs.psfMatched)

    @timeMethod
    def run(self, calExpList, ccdIdList, skyInfo, visitId=0, dataIdList=None, **kwargs):
        """Create a Warp from inputs

        We iterate over the multiple calexps in a single exposure to construct
        the warp (previously called a coaddTempExp) of that exposure to the
        supplied tract/patch.

        Pixels that receive no pixels are set to NAN; this is not correct
        (violates LSST algorithms group policy), but will be fixed up by
        interpolating after the coaddition.

        @param calexpRefList: List of data references for calexps that (may)
            overlap the patch of interest
        @param skyInfo: Struct from CoaddBaseTask.getSkyInfo() with geometric
            information about the patch
        @param visitId: integer identifier for visit, for the table that will
            produce the CoaddPsf
        @return a pipeBase Struct containing:
          - exposures: a dictionary containing the warps requested:
                "direct": direct warp if config.makeDirect
                "psfMatched": PSF-matched warp if config.makePsfMatched
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
                    if didSetMetadata[warpType]:
                        mimg = exposure.getMaskedImage()
                        mimg *= (warp.getPhotoCalib().getInstFluxAtZeroMagnitude()
                                 / exposure.getPhotoCalib().getInstFluxAtZeroMagnitude())
                        del mimg
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
                        # PSF replaced with CoaddPsf after loop if and only if creating direct warp
                        warp.setPsf(exposure.getPsf())
                        didSetMetadata[warpType] = True

                    # Need inputRecorder for CoaddApCorrMap for both direct and PSF-matched
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
            else:
                if not self.config.doWriteEmptyWarps:
                    # No good pixels. Exposure still empty
                    warps[warpType] = None
                    # NoWorkFound is unnecessary as the downstream tasks will
                    # adjust the quantum accordingly.

        result = pipeBase.Struct(exposures=warps)
        return result

    def filterInputs(self, indices, inputs):
        """Return task inputs with their lists filtered by indices

        Parameters
        ----------
        indices : `list` of integers
        inputs : `dict` of `list` of input connections to be passed to run
        """
        for key in inputs.keys():
            # Only down-select on list inputs
            if isinstance(inputs[key], list):
                inputs[key] = [inputs[key][ind] for ind in indices]
        return inputs

    def prepareCalibratedExposures(self, calExpList, backgroundList=None, skyCorrList=None,
                                   externalSkyWcsCatalog=None, externalPhotoCalibCatalog=None,
                                   finalizedPsfApCorrCatalog=None,
                                   **kwargs):
        """Calibrate and add backgrounds to input calExpList in place

        Parameters
        ----------
        calExpList : `list` of `lsst.afw.image.Exposure`
            Sequence of calexps to be modified in place
        backgroundList : `list` of `lsst.afw.math.backgroundList`, optional
            Sequence of backgrounds to be added back in if bgSubtracted=False
        skyCorrList : `list` of `lsst.afw.math.backgroundList`, optional
            Sequence of background corrections to be subtracted if doApplySkyCorr=True
        externalSkyWcsCatalog : `lsst.afw.table.ExposureCatalog`, optional
            Exposure catalog with external skyWcs to be applied
            if config.doApplyExternalSkyWcs=True.  Catalog uses the detector id
            for the catalog id, sorted on id for fast lookup.
        externalPhotoCalibCatalog : `lsst.afw.table.ExposureCatalog`, optional
            Exposure catalog with external photoCalib to be applied
            if config.doApplyExternalPhotoCalib=True.  Catalog uses the detector
            id for the catalog id, sorted on id for fast lookup.
        finalizedPsfApCorrCatalog : `lsst.afw.table.ExposureCatalog`, optional
            Exposure catalog with finalized psf models and aperture correction
            maps to be applied if config.doApplyFinalizedPsf=True.  Catalog uses
            the detector id for the catalog id, sorted on id for fast lookup.

        Returns
        -------
        indices : `list` [`int`]
            Indices of calExpList and friends that have valid photoCalib/skyWcs
        """
        backgroundList = len(calExpList)*[None] if backgroundList is None else backgroundList
        skyCorrList = len(calExpList)*[None] if skyCorrList is None else skyCorrList

        includeCalibVar = self.config.includeCalibVar

        indices = []
        for index, (calexp, background, skyCorr) in enumerate(zip(calExpList,
                                                                  backgroundList,
                                                                  skyCorrList)):
            if not self.config.bgSubtracted:
                calexp.maskedImage += background.getImage()

            detectorId = calexp.getInfo().getDetector().getId()

            # Find the external photoCalib
            if externalPhotoCalibCatalog is not None:
                row = externalPhotoCalibCatalog.find(detectorId)
                if row is None:
                    self.log.warning("Detector id %s not found in externalPhotoCalibCatalog "
                                     "and will not be used in the warp.", detectorId)
                    continue
                photoCalib = row.getPhotoCalib()
                if photoCalib is None:
                    self.log.warning("Detector id %s has None for photoCalib in externalPhotoCalibCatalog "
                                     "and will not be used in the warp.", detectorId)
                    continue
                calexp.setPhotoCalib(photoCalib)
            else:
                photoCalib = calexp.getPhotoCalib()
                if photoCalib is None:
                    self.log.warning("Detector id %s has None for photoCalib in the calexp "
                                     "and will not be used in the warp.", detectorId)
                    continue

            # Find and apply external skyWcs
            if externalSkyWcsCatalog is not None:
                row = externalSkyWcsCatalog.find(detectorId)
                if row is None:
                    self.log.warning("Detector id %s not found in externalSkyWcsCatalog "
                                     "and will not be used in the warp.", detectorId)
                    continue
                skyWcs = row.getWcs()
                if skyWcs is None:
                    self.log.warning("Detector id %s has None for skyWcs in externalSkyWcsCatalog "
                                     "and will not be used in the warp.", detectorId)
                    continue
                calexp.setWcs(skyWcs)
            else:
                skyWcs = calexp.getWcs()
                if skyWcs is None:
                    self.log.warning("Detector id %s has None for skyWcs in the calexp "
                                     "and will not be used in the warp.", detectorId)
                    continue

            # Find and apply finalized psf and aperture correction
            if finalizedPsfApCorrCatalog is not None:
                row = finalizedPsfApCorrCatalog.find(detectorId)
                if row is None:
                    self.log.warning("Detector id %s not found in finalizedPsfApCorrCatalog "
                                     "and will not be used in the warp.", detectorId)
                    continue
                psf = row.getPsf()
                if psf is None:
                    self.log.warning("Detector id %s has None for psf in finalizedPsfApCorrCatalog "
                                     "and will not be used in the warp.", detectorId)
                    continue
                calexp.setPsf(psf)
                apCorrMap = row.getApCorrMap()
                if apCorrMap is None:
                    self.log.warning("Detector id %s has None for ApCorrMap in finalizedPsfApCorrCatalog "
                                     "and will not be used in the warp.", detectorId)
                    continue
                calexp.info.setApCorrMap(apCorrMap)

            # Calibrate the image
            calexp.maskedImage = photoCalib.calibrateImage(calexp.maskedImage,
                                                           includeScaleUncertainty=includeCalibVar)
            calexp.maskedImage /= photoCalib.getCalibrationMean()
            # TODO: The images will have a calibration of 1.0 everywhere once RFC-545 is implemented.
            # exposure.setCalib(afwImage.Calib(1.0))

            # Apply skycorr
            if self.config.doApplySkyCorr:
                calexp.maskedImage -= skyCorr.getImage()

            indices.append(index)

        return indices

    @staticmethod
    def _prepareEmptyExposure(skyInfo):
        """Produce an empty exposure for a given patch"""
        exp = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
        exp.getMaskedImage().set(numpy.nan, afwImage.Mask
                                 .getPlaneBitMask("NO_DATA"), numpy.inf)
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

    def applySkyCorr(self, dataRef, calexp):
        """Apply correction to the sky background level

        Sky corrections can be generated with the 'skyCorrection.py'
        executable in pipe_drivers. Because the sky model used by that
        code extends over the entire focal plane, this can produce
        better sky subtraction.

        The calexp is updated in-place.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for calexp.
        calexp : `lsst.afw.image.Exposure` or `lsst.afw.image.MaskedImage`
            Calibrated exposure.
        """
        bg = dataRef.get("skyCorr")
        self.log.debug("Applying sky correction to %s", dataRef.dataId)
        if isinstance(calexp, afwImage.Exposure):
            calexp = calexp.getMaskedImage()
        calexp -= bg.getImage()


def reorderRefs(inputRefs, outputSortKeyOrder, dataIdKey):
    """Reorder inputRefs per outputSortKeyOrder

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
    outputSortKeyOrder : iterable
        Iterable of values to be compared with inputRef's dataId[dataIdKey]
    dataIdKey :  `str`
        dataIdKey in the dataRefs to compare with the outputSortKeyOrder.

    Returns:
    --------
    inputRefs: `lsst.pipe.base.connections.QuantizedConnection`
        Quantized Connection with sorted DatasetRef values sorted if iterable.
    """
    for connectionName, refs in inputRefs:
        if isinstance(refs, Iterable):
            if hasattr(refs[0], "dataId"):
                inputSortKeyOrder = [ref.dataId[dataIdKey] for ref in refs]
            else:
                inputSortKeyOrder = [ref.datasetRef.dataId[dataIdKey] for ref in refs]
            if inputSortKeyOrder != outputSortKeyOrder:
                setattr(inputRefs, connectionName,
                        reorderAndPadList(refs, inputSortKeyOrder, outputSortKeyOrder))
    return inputRefs
