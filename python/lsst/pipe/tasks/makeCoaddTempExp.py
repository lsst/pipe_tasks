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
import numpy
import logging

import lsst.pex.config as pexConfig
import lsst.daf.persistence as dafPersist
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
from .selectImages import PsfWcsSelectImagesTask
from .warpAndPsfMatch import WarpAndPsfMatchTask
from .coaddHelpers import groupPatchExposures, getGroupDataRef
from collections.abc import Iterable

__all__ = ["MakeCoaddTempExpTask", "MakeWarpTask", "MakeWarpConfig"]

log = logging.getLogger(__name__)


class MissingExposureError(Exception):
    """Raised when data cannot be retrieved for an exposure.
    When processing patches, sometimes one exposure is missing; this lets us
    distinguish bewteen that case, and other errors.
    """
    pass


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
    doApplySkyCorr = pexConfig.Field(dtype=bool, default=False, doc="Apply sky correction?")

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
        self.select.retarget(PsfWcsSelectImagesTask)

## \addtogroup LSST_task_documentation
## \{
## \page page_MakeCoaddTempExpTask MakeCoaddTempExpTask
## \ref MakeCoaddTempExpTask_ "MakeCoaddTempExpTask"
## \copybrief MakeCoaddTempExpTask
## \}


class MakeCoaddTempExpTask(CoaddBaseTask):
    r"""!
    Warp and optionally PSF-Match calexps onto an a common projection.

    @anchor MakeCoaddTempExpTask_

    @section pipe_tasks_makeCoaddTempExp_Contents  Contents

     - @ref pipe_tasks_makeCoaddTempExp_Purpose
     - @ref pipe_tasks_makeCoaddTempExp_Initialize
     - @ref pipe_tasks_makeCoaddTempExp_IO
     - @ref pipe_tasks_makeCoaddTempExp_Config
     - @ref pipe_tasks_makeCoaddTempExp_Debug
     - @ref pipe_tasks_makeCoaddTempExp_Example

    @section pipe_tasks_makeCoaddTempExp_Purpose  Description

    Warp and optionally PSF-Match calexps onto a common projection, by
    performing the following operations:
    - Group calexps by visit/run
    - For each visit, generate a Warp by calling method @ref run.
      `run` loops over the visit's calexps calling
      @ref warpAndPsfMatch::WarpAndPsfMatchTask "WarpAndPsfMatchTask"
      on each visit

    The result is a `directWarp` (and/or optionally a `psfMatchedWarp`).

    @section pipe_tasks_makeCoaddTempExp_Initialize  Task Initialization

    @copydoc \_\_init\_\_

    This task has one special keyword argument: passing reuse=True will cause
    the task to skip the creation of warps that are already present in the
    output repositories.

    @section pipe_tasks_makeCoaddTempExp_IO  Invoking the Task

    This task is primarily designed to be run from the command line.

    The main method is `runDataRef`, which takes a single butler data reference for the patch(es)
    to process.

    @copydoc run

    WarpType identifies the types of convolutions applied to Warps (previously CoaddTempExps).
    Only two types are available: direct (for regular Warps/Coadds) and psfMatched
    (for Warps/Coadds with homogenized PSFs). We expect to add a third type, likelihood,
    for generating likelihood Coadds with Warps that have been correlated with their own PSF.

    @section pipe_tasks_makeCoaddTempExp_Config  Configuration parameters

    See @ref MakeCoaddTempExpConfig and parameters inherited from
    @link coaddBase::CoaddBaseConfig CoaddBaseConfig @endlink

    @subsection pipe_tasks_MakeCoaddTempExp_psfMatching Guide to PSF-Matching Configs

    To make `psfMatchedWarps`, select `config.makePsfMatched=True`. The subtask
    @link ip::diffim::modelPsfMatch::ModelPsfMatchTask ModelPsfMatchTask @endlink
    is responsible for the PSF-Matching, and its config is accessed via `config.warpAndPsfMatch.psfMatch`.
    The optimal configuration depends on aspects of dataset: the pixel scale, average PSF FWHM and
    dimensions of the PSF kernel. These configs include the requested model PSF, the matching kernel size,
    padding of the science PSF thumbnail and spatial sampling frequency of the PSF.

    *Config Guidelines*: The user must specify the size of the model PSF to which to match by setting
    `config.modelPsf.defaultFwhm` in units of pixels. The appropriate values depends on science case.
    In general, for a set of input images, this config should equal the FWHM of the visit
    with the worst seeing. The smallest it should be set to is the median FWHM. The defaults
    of the other config options offer a reasonable starting point.
    The following list presents the most common problems that arise from a misconfigured
    @link ip::diffim::modelPsfMatch::ModelPsfMatchTask ModelPsfMatchTask @endlink
    and corresponding solutions. All assume the default Alard-Lupton kernel, with configs accessed via
    ```config.warpAndPsfMatch.psfMatch.kernel['AL']```. Each item in the list is formatted as:
    Problem: Explanation. *Solution*

    *Troublshooting PSF-Matching Configuration:*
    - Matched PSFs look boxy: The matching kernel is too small. _Increase the matching kernel size.
        For example:_

            config.warpAndPsfMatch.psfMatch.kernel['AL'].kernelSize=27  # default 21

        Note that increasing the kernel size also increases runtime.
    - Matched PSFs look ugly (dipoles, quadropoles, donuts): unable to find good solution
        for matching kernel. _Provide the matcher with more data by either increasing
        the spatial sampling by decreasing the spatial cell size,_

            config.warpAndPsfMatch.psfMatch.kernel['AL'].sizeCellX = 64  # default 128
            config.warpAndPsfMatch.psfMatch.kernel['AL'].sizeCellY = 64  # default 128

        _or increasing the padding around the Science PSF, for example:_

            config.warpAndPsfMatch.psfMatch.autoPadPsfTo=1.6  # default 1.4

        Increasing `autoPadPsfTo` increases the minimum ratio of input PSF dimensions to the
        matching kernel dimensions, thus increasing the number of pixels available to fit
        after convolving the PSF with the matching kernel.
        Optionally, for debugging the effects of padding, the level of padding may be manually
        controlled by setting turning off the automatic padding and setting the number
        of pixels by which to pad the PSF:

            config.warpAndPsfMatch.psfMatch.doAutoPadPsf = False  # default True
            config.warpAndPsfMatch.psfMatch.padPsfBy = 6  # pixels. default 0

    - Deconvolution: Matching a large PSF to a smaller PSF produces
        a telltale noise pattern which looks like ripples or a brain.
        _Increase the size of the requested model PSF. For example:_

            config.modelPsf.defaultFwhm = 11  # Gaussian sigma in units of pixels.

    - High frequency (sometimes checkered) noise: The matching basis functions are too small.
        _Increase the width of the Gaussian basis functions. For example:_

            config.warpAndPsfMatch.psfMatch.kernel['AL'].alardSigGauss=[1.5, 3.0, 6.0]
            # from default [0.7, 1.5, 3.0]


    @section pipe_tasks_makeCoaddTempExp_Debug  Debug variables

    MakeCoaddTempExpTask has no debug output, but its subtasks do.

    @section pipe_tasks_makeCoaddTempExp_Example   A complete example of using MakeCoaddTempExpTask

    This example uses the package ci_hsc to show how MakeCoaddTempExp fits
    into the larger Data Release Processing.
    Set up by running:

        setup ci_hsc
        cd $CI_HSC_DIR
        # if not built already:
        python $(which scons)  # this will take a while

    The following assumes that `processCcd.py` and `makeSkyMap.py` have previously been run
    (e.g. by building `ci_hsc` above) to generate a repository of calexps and an
    output respository with the desired SkyMap. The command,

        makeCoaddTempExp.py $CI_HSC_DIR/DATA --rerun ci_hsc \
         --id patch=5,4 tract=0 filter=HSC-I \
         --selectId visit=903988 ccd=16 --selectId visit=903988 ccd=17 \
         --selectId visit=903988 ccd=23 --selectId visit=903988 ccd=24 \
         --config doApplyExternalPhotoCalib=False doApplyExternalSkyWcs=False \
         makePsfMatched=True modelPsf.defaultFwhm=11

    writes a direct and PSF-Matched Warp to
    - `$CI_HSC_DIR/DATA/rerun/ci_hsc/deepCoadd/HSC-I/0/5,4/warp-HSC-I-0-5,4-903988.fits` and
    - `$CI_HSC_DIR/DATA/rerun/ci_hsc/deepCoadd/HSC-I/0/5,4/psfMatchedWarp-HSC-I-0-5,4-903988.fits`
        respectively.

    @note PSF-Matching in this particular dataset would benefit from adding
    `--configfile ./matchingConfig.py` to
    the command line arguments where `matchingConfig.py` is defined by:

        echo "
        config.warpAndPsfMatch.psfMatch.kernel['AL'].kernelSize=27
        config.warpAndPsfMatch.psfMatch.kernel['AL'].alardSigGauss=[1.5, 3.0, 6.0]" > matchingConfig.py


    Add the option `--help` to see more options.
    """
    ConfigClass = MakeCoaddTempExpConfig
    _DefaultName = "makeCoaddTempExp"

    def __init__(self, reuse=False, **kwargs):
        CoaddBaseTask.__init__(self, **kwargs)
        self.reuse = reuse
        self.makeSubtask("warpAndPsfMatch")
        if self.config.hasFakes:
            self.calexpType = "fakes_calexp"
        else:
            self.calexpType = "calexp"

    @timeMethod
    def runDataRef(self, patchRef, selectDataList=[]):
        """!
        Produce @<coaddName>Coadd_@<warpType>Warp images by warping and optionally PSF-matching.

        @param[in] patchRef: data reference for sky map patch. Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter" or "band")
        @param[in] selectDataList list of @ref selectImages::SelectStruct "SelectStruct"
            to consider for selection
        @return: dataRefList: a list of data references for the new @<coaddName>Coadd_directWarps
            if direct or both warp types are requested and @<coaddName>Coadd_psfMatchedWarps
            if only psfMatched
            warps are requested.

        @warning: this task assumes that all exposures in a warp (coaddTempExp) have the same filter.

        @warning: this task sets the PhotoCalib of the coaddTempExp to the PhotoCalib of the first calexp
        with any good pixels in the patch. For a mosaic camera the resulting PhotoCalib should be ignored
        (assembleCoadd should determine zeropoint scaling without referring to it).
        """
        skyInfo = self.getSkyInfo(patchRef)

        # DataRefs to return are of type *_directWarp unless only *_psfMatchedWarp requested
        if self.config.makePsfMatched and not self.config.makeDirect:
            primaryWarpDataset = self.getTempExpDatasetName("psfMatched")
        else:
            primaryWarpDataset = self.getTempExpDatasetName("direct")

        calExpRefList = self.selectExposures(patchRef, skyInfo, selectDataList=selectDataList)

        if len(calExpRefList) == 0:
            self.log.warning("No exposures to coadd for patch %s", patchRef.dataId)
            return None
        self.log.info("Selected %d calexps for patch %s", len(calExpRefList), patchRef.dataId)
        calExpRefList = [calExpRef for calExpRef in calExpRefList if calExpRef.datasetExists(self.calexpType)]
        self.log.info("Processing %d existing calexps for patch %s", len(calExpRefList), patchRef.dataId)

        groupData = groupPatchExposures(patchRef, calExpRefList, self.getCoaddDatasetName(),
                                        primaryWarpDataset)
        self.log.info("Processing %d warp exposures for patch %s", len(groupData.groups), patchRef.dataId)

        dataRefList = []
        for i, (tempExpTuple, calexpRefList) in enumerate(groupData.groups.items()):
            tempExpRef = getGroupDataRef(patchRef.getButler(), primaryWarpDataset,
                                         tempExpTuple, groupData.keys)
            if self.reuse and tempExpRef.datasetExists(datasetType=primaryWarpDataset, write=True):
                self.log.info("Skipping makeCoaddTempExp for %s; output already exists.", tempExpRef.dataId)
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

            calExpList = []
            ccdIdList = []
            dataIdList = []

            for calExpInd, calExpRef in enumerate(calexpRefList):
                self.log.info("Reading calexp %s of %s for Warp id=%s", calExpInd+1, len(calexpRefList),
                              calExpRef.dataId)
                try:
                    ccdId = calExpRef.get("ccdExposureId", immediate=True)
                except Exception:
                    ccdId = calExpInd
                try:
                    # We augment the dataRef here with the tract, which is harmless for loading things
                    # like calexps that don't need the tract, and necessary for meas_mosaic outputs,
                    # which do.
                    calExpRef = calExpRef.butlerSubset.butler.dataRef(self.calexpType,
                                                                      dataId=calExpRef.dataId,
                                                                      tract=skyInfo.tractInfo.getId())
                    calExp = self.getCalibratedExposure(calExpRef, bgSubtracted=self.config.bgSubtracted)
                except Exception as e:
                    self.log.warning("Calexp %s not found; skipping it: %s", calExpRef.dataId, e)
                    continue

                if self.config.doApplySkyCorr:
                    self.applySkyCorr(calExpRef, calExp)

                calExpList.append(calExp)
                ccdIdList.append(ccdId)
                dataIdList.append(calExpRef.dataId)

            exps = self.run(calExpList, ccdIdList, skyInfo, visitId, dataIdList).exposures

            if any(exps.values()):
                dataRefList.append(tempExpRef)
            else:
                self.log.warning("Warp %s could not be created", tempExpRef.dataId)

            if self.config.doWrite:
                for (warpType, exposure) in exps.items():  # compatible w/ Py3
                    if exposure is not None:
                        self.log.info("Persisting %s", self.getTempExpDatasetName(warpType))
                        tempExpRef.put(exposure, self.getTempExpDatasetName(warpType))

        return dataRefList

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
        coaddTempExps = {warpType: self._prepareEmptyExposure(skyInfo) for warpType in warpTypeList}
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
                    coaddTempExp = coaddTempExps[warpType]
                    if didSetMetadata[warpType]:
                        mimg = exposure.getMaskedImage()
                        mimg *= (coaddTempExp.getPhotoCalib().getInstFluxAtZeroMagnitude()
                                 / exposure.getPhotoCalib().getInstFluxAtZeroMagnitude())
                        del mimg
                    numGoodPix[warpType] = coaddUtils.copyGoodPixels(
                        coaddTempExp.getMaskedImage(), exposure.getMaskedImage(), self.getBadPixelMask())
                    totGoodPix[warpType] += numGoodPix[warpType]
                    self.log.debug("Calexp %s has %d good pixels in this patch (%.1f%%) for %s",
                                   dataId, numGoodPix[warpType],
                                   100.0*numGoodPix[warpType]/skyInfo.bbox.getArea(), warpType)
                    if numGoodPix[warpType] > 0 and not didSetMetadata[warpType]:
                        coaddTempExp.info.id = exposure.info.id
                        coaddTempExp.setPhotoCalib(exposure.getPhotoCalib())
                        coaddTempExp.setFilter(exposure.getFilter())
                        coaddTempExp.getInfo().setVisitInfo(exposure.getInfo().getVisitInfo())
                        # PSF replaced with CoaddPsf after loop if and only if creating direct warp
                        coaddTempExp.setPsf(exposure.getPsf())
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
                inputRecorder[warpType].finish(coaddTempExps[warpType], totGoodPix[warpType])
                if warpType == "direct":
                    coaddTempExps[warpType].setPsf(
                        CoaddPsf(inputRecorder[warpType].coaddInputs.ccds, skyInfo.wcs,
                                 self.config.coaddPsf.makeControl()))
            else:
                if not self.config.doWriteEmptyWarps:
                    # No good pixels. Exposure still empty
                    coaddTempExps[warpType] = None
                    # NoWorkFound is unnecessary as the downstream tasks will
                    # adjust the quantum accordingly, and it prevents gen2
                    # MakeCoaddTempExp from continuing to loop over visits.

        result = pipeBase.Struct(exposures=coaddTempExps)
        return result

    def getCalibratedExposure(self, dataRef, bgSubtracted):
        """Return one calibrated Exposure, possibly with an updated SkyWcs.

        @param[in] dataRef        a sensor-level data reference
        @param[in] bgSubtracted   return calexp with background subtracted? If False get the
                                  calexp's background background model and add it to the calexp.
        @return calibrated exposure

        @raises MissingExposureError If data for the exposure is not available.

        If config.doApplyExternalPhotoCalib is `True`, the photometric calibration
        (`photoCalib`) is taken from `config.externalPhotoCalibName` via the
        `name_photoCalib` dataset.  Otherwise, the photometric calibration is
        retrieved from the processed exposure.  When
        `config.doApplyExternalSkyWcs` is `True`, the astrometric calibration
        is taken from `config.externalSkyWcsName` with the `name_wcs` dataset.
        Otherwise, the astrometric calibration is taken from the processed
        exposure.
        """
        try:
            exposure = dataRef.get(self.calexpType, immediate=True)
        except dafPersist.NoResults as e:
            raise MissingExposureError('Exposure not found: %s ' % str(e)) from e

        if not bgSubtracted:
            background = dataRef.get("calexpBackground", immediate=True)
            mi = exposure.getMaskedImage()
            mi += background.getImage()
            del mi

        if self.config.doApplyExternalPhotoCalib:
            source = f"{self.config.externalPhotoCalibName}_photoCalib"
            self.log.debug("Applying external photoCalib to %s from %s", dataRef.dataId, source)
            photoCalib = dataRef.get(source)
            exposure.setPhotoCalib(photoCalib)
        else:
            photoCalib = exposure.getPhotoCalib()

        if self.config.doApplyExternalSkyWcs:
            source = f"{self.config.externalSkyWcsName}_wcs"
            self.log.debug("Applying external skyWcs to %s from %s", dataRef.dataId, source)
            skyWcs = dataRef.get(source)
            exposure.setWcs(skyWcs)

        exposure.maskedImage = photoCalib.calibrateImage(exposure.maskedImage,
                                                         includeScaleUncertainty=self.config.includeCalibVar)
        exposure.maskedImage /= photoCalib.getCalibrationMean()
        # TODO: The images will have a calibration of 1.0 everywhere once RFC-545 is implemented.
        # exposure.setCalib(afwImage.Calib(1.0))
        return exposure

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
        doc="Consolidated exposure metadata",
        name="finalVisitSummary",
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


class MakeWarpConfig(pipeBase.PipelineTaskConfig, MakeCoaddTempExpConfig,
                     pipelineConnections=MakeWarpConnections):

    def validate(self):
        super().validate()


class MakeWarpTask(MakeCoaddTempExpTask):
    """Warp and optionally PSF-Match calexps onto an a common projection
    """
    ConfigClass = MakeWarpConfig
    _DefaultName = "makeWarp"

    @utils.inheritDoc(pipeBase.PipelineTask)
    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """
        Notes
        ----
        Construct warps for requested warp type for single epoch

        PipelineTask (Gen3) entry point to warp and optionally PSF-match
        calexps. This method is analogous to `runDataRef`.
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
