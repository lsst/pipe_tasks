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

from __future__ import absolute_import, division, print_function
import numpy

import lsst.pex.config as pexConfig
import lsst.afw.image as afwImage
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
from lsst.meas.algorithms import CoaddPsf
from .coaddBase import CoaddBaseTask, WarpType
from .warpAndPsfMatch import WarpAndPsfMatchTask
from .coaddHelpers import groupPatchExposures, getGroupDataRef

__all__ = ["MakeCoaddTempExpTask"]


class MakeCoaddTempExpConfig(CoaddBaseTask.ConfigClass):
    """Config for MakeCoaddTempExpTask
    """
    warpAndPsfMatch = pexConfig.ConfigurableField(
        target=WarpAndPsfMatchTask,
        doc="Task to warp and PSF-match calexp",
    )
    doWrite = pexConfig.Field(
        doc="persist <coaddName>Coadd_<warpType>Warp",
        dtype=bool,
        default=True,
    )
    doOverwrite = pexConfig.Field(
        doc="overwrite <coaddName>Coadd_<warpType>Warp; If False, continue if the file exists on disk",
        dtype=bool,
        default=True,
    )
    bgSubtracted = pexConfig.Field(
        doc="Work with a background subtracted calexp?",
        dtype=bool,
        default=True,
    )


class MakeCoaddTempExpTask(CoaddBaseTask):
    """Task to produce <coaddName>Coadd_<warpType>Warp images
    """
    ConfigClass = MakeCoaddTempExpConfig
    _DefaultName = "makeCoaddTempExp"

    def __init__(self, *args, **kwargs):
        CoaddBaseTask.__init__(self, *args, **kwargs)
        self.makeSubtask("warpAndPsfMatch")

    @pipeBase.timeMethod
    def run(self, patchRef, selectDataList=[]):
        """Produce <coaddName>Coadd_<warpType>Warp images

        <coaddName>Coadd_<warpType>Warp images are produced by warping and optionally PSF-matching.

        @param[in] patchRef: data reference for sky map patch. Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter" or "band")
        @return: dataRefList: a list of data references for the new <coaddName>Coadd_directWarps
            if direct or both warp types are requested and <coaddName>Coadd_psfMatchedWarps if only psfMatched
            warps are requested.

        @warning: this task assumes that all exposures in a warp (coaddTempExp) have the same filter.

        @warning: this task sets the Calib of the coaddTempExp to the Calib of the first calexp
        with any good pixels in the patch. For a mosaic camera the resulting Calib should be ignored
        (assembleCoadd should determine zeropoint scaling without referring to it).
        """
        skyInfo = self.getSkyInfo(patchRef)

        # DataRefs to return are of type *_directWarp unless only *_psfMatchedWarp requested
        if self.config.makePsfMatched and not self.config.makeDirect:
            primaryWarpDataset = self.getTempExpDatasetName(WarpType.PSF_MATCHED)
        else:
            primaryWarpDataset = self.getTempExpDatasetName(WarpType.DIRECT)

        calExpRefList = self.selectExposures(patchRef, skyInfo, selectDataList=selectDataList)
        if len(calExpRefList) == 0:
            self.log.warn("No exposures to coadd for patch %s", patchRef.dataId)
            return None
        self.log.info("Selected %d calexps for patch %s", len(calExpRefList), patchRef.dataId)
        calExpRefList = [calExpRef for calExpRef in calExpRefList if calExpRef.datasetExists("calexp")]
        self.log.info("Processing %d existing calexps for patch %s", len(calExpRefList), patchRef.dataId)

        groupData = groupPatchExposures(patchRef, calExpRefList, self.getCoaddDatasetName(),
                                        primaryWarpDataset)
        self.log.info("Processing %d warp exposures for patch %s", len(groupData.groups), patchRef.dataId)

        dataRefList = []
        for i, (tempExpTuple, calexpRefList) in enumerate(groupData.groups.items()):
            tempExpRef = getGroupDataRef(patchRef.getButler(), primaryWarpDataset,
                                         tempExpTuple, groupData.keys)
            if not self.config.doOverwrite and tempExpRef.datasetExists(datasetType=primaryWarpDataset):
                self.log.info("Warp %s exists; skipping", tempExpRef.dataId)
                dataRefList.append(tempExpRef)
                continue
            self.log.info("Processing Warp %d/%d: id=%s", i, len(groupData.groups), tempExpRef.dataId)

            # TODO: mappers should define a way to go from the "grouping keys" to a numeric ID (#2776).
            # For now, we try to get a long integer "visit" key, and if we can't, we just use the index
            # of the visit in the list.
            try:
                visitId = int(tempExpRef.dataId["visit"])
            except (KeyError, ValueError):
                visitId = i

            exps = self.createTempExp(calexpRefList, skyInfo, visitId).exposures

            if any(exps.values()):
                dataRefList.append(tempExpRef)
            else:
                self.log.warn("Warp %s could not be created", tempExpRef.dataId)

            if self.config.doWrite:
                for (warpType, exposure) in exps.items():  # compatible w/ Py3
                    if exposure is not None:
                        self.log.info("Persisting %s" % self.getTempExpDatasetName(warpType))
                        tempExpRef.put(exposure, self.getTempExpDatasetName(warpType))

        return dataRefList

    def createTempExp(self, calexpRefList, skyInfo, visitId=0):
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
        coaddTempExps = {warpType: self._prepareEmptyExposure(skyInfo) for warpType in warpTypeList}
        inputRecorder = {warpType: self.inputRecorder.makeCoaddTempExpRecorder(visitId, len(calexpRefList))
                         for warpType in warpTypeList}

        modelPsf = self.config.modelPsf.apply() if self.config.makePsfMatched else None
        for calExpInd, calExpRef in enumerate(calexpRefList):
            self.log.info("Processing calexp %d of %d for this Warp: id=%s",
                          calExpInd+1, len(calexpRefList), calExpRef.dataId)
            try:
                ccdId = calExpRef.get("ccdExposureId", immediate=True)
            except Exception:
                ccdId = calExpInd
            try:
                # We augment the dataRef here with the tract, which is harmless for loading things
                # like calexps that don't need the tract, and necessary for meas_mosaic outputs,
                # which do.
                calExpRef = calExpRef.butlerSubset.butler.dataRef("calexp", dataId=calExpRef.dataId,
                                                                  tract=skyInfo.tractInfo.getId())
                calExp = self.getCalExp(calExpRef, bgSubtracted=self.config.bgSubtracted)
            except Exception as e:
                self.log.warn("Calexp %s not found; skipping it: %s", calExpRef.dataId, e)
                continue
            try:
                warpedAndMatched = self.warpAndPsfMatch.run(calExp, modelPsf=modelPsf,
                                                            wcs=skyInfo.wcs, maxBBox=skyInfo.bbox,
                                                            makeDirect=self.config.makeDirect,
                                                            makePsfMatched=self.config.makePsfMatched)
            except Exception as e:
                self.log.warn("WarpAndPsfMatch failed for calexp %s; skipping it: %s", calExpRef.dataId, e)
                continue
            try:
                numGoodPix = {warpType: 0 for warpType in warpTypeList}
                for warpType in warpTypeList:
                    exposure = warpedAndMatched.getDict()[warpType.value]
                    if exposure is None:
                        continue
                    coaddTempExp = coaddTempExps[warpType]
                    if didSetMetadata[warpType]:
                        mimg = exposure.getMaskedImage()
                        mimg *= (coaddTempExp.getCalib().getFluxMag0()[0] /
                                 exposure.getCalib().getFluxMag0()[0])
                        del mimg
                    numGoodPix[warpType] = coaddUtils.copyGoodPixels(
                        coaddTempExp.getMaskedImage(), exposure.getMaskedImage(), self.getBadPixelMask())
                    totGoodPix[warpType] += numGoodPix[warpType]
                    self.log.debug("Calexp %s has %d good pixels in this patch (%.1f%%) for %s",
                                   calExpRef.dataId, numGoodPix[warpType],
                                   100.0*numGoodPix[warpType]/skyInfo.bbox.getArea(), warpType.value)
                    if numGoodPix[warpType] > 0 and not didSetMetadata[warpType]:
                        coaddTempExp.setCalib(exposure.getCalib())
                        coaddTempExp.setFilter(exposure.getFilter())
                        # PSF replaced with CoaddPsf after loop if and only if creating direct warp
                        coaddTempExp.setPsf(exposure.getPsf())
                        didSetMetadata[warpType] = True

                    # Need inputRecorder for CoaddApCorrMap for both direct and PSF-matched
                    inputRecorder[warpType].addCalExp(calExp, ccdId, numGoodPix[warpType])

            except Exception as e:
                self.log.warn("Error processing calexp %s; skipping it: %s", calExpRef.dataId, e)
                continue

        for warpType in warpTypeList:
            self.log.info("%sWarp has %d good pixels (%.1f%%)",
                          warpType.value, totGoodPix[warpType], 100.0*totGoodPix[warpType]/skyInfo.bbox.getArea())

            if totGoodPix[warpType] > 0 and didSetMetadata[warpType]:
                inputRecorder[warpType].finish(coaddTempExps[warpType], totGoodPix[warpType])
                if warpType == WarpType.DIRECT:
                    coaddTempExps[warpType].setPsf(
                        CoaddPsf(inputRecorder[warpType].coaddInputs.ccds, skyInfo.wcs))
            else:
                # No good pixels. Exposure still empty
                coaddTempExps[warpType] = None

        result = pipeBase.Struct(exposures=coaddTempExps)
        return result

    def _prepareEmptyExposure(cls, skyInfo):
        """Produce an empty exposure for a given patch"""
        exp = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
        exp.getMaskedImage().set(numpy.nan, afwImage.MaskU.getPlaneBitMask("NO_DATA"), numpy.inf)
        return exp
