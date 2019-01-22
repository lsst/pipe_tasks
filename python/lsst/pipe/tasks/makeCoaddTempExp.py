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

import lsst.pex.config as pexConfig
import lsst.daf.persistence as dafPersist
import lsst.afw.image as afwImage
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
import lsst.log as log
from lsst.meas.algorithms import CoaddPsf, CoaddPsfConfig
from .coaddBase import CoaddBaseTask, makeSkyInfo
from .warpAndPsfMatch import WarpAndPsfMatchTask
from .coaddHelpers import groupPatchExposures, getGroupDataRef

__all__ = ["MakeCoaddTempExpTask", "MakeWarpTask", "MakeWarpConfig"]


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
    doApplySkyCorr = pexConfig.Field(dtype=bool, default=False, doc="Apply sky correction?")

    def validate(self):
        CoaddBaseTask.ConfigClass.validate(self)
        if not self.makePsfMatched and not self.makeDirect:
            raise RuntimeError("At least one of config.makePsfMatched and config.makeDirect must be True")
        if self.doPsfMatch:
            # Backwards compatibility.
            log.warn("Config doPsfMatch deprecated. Setting makePsfMatched=True and makeDirect=False")
            self.makePsfMatched = True
            self.makeDirect = False

    def setDefaults(self):
        CoaddBaseTask.ConfigClass.setDefaults(self)
        self.warpAndPsfMatch.psfMatch.kernel.active.kernelSize = self.matchingKernelSize

## \addtogroup LSST_task_documentation
## \{
## \page MakeCoaddTempExpTask
## \ref MakeCoaddTempExpTask_ "MakeCoaddTempExpTask"
## \copybrief MakeCoaddTempExpTask
## \}


class MakeCoaddTempExpTask(CoaddBaseTask):
    r"""!Warp and optionally PSF-Match calexps onto an a common projection.

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
    - For each visit, generate a Warp by calling method @ref makeTempExp.
      makeTempExp loops over the visit's calexps calling @ref WarpAndPsfMatch
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
    @link lsst.pipe.tasks.coaddBase.CoaddBaseConfig CoaddBaseConfig @endlink

    @subsection pipe_tasks_MakeCoaddTempExp_psfMatching Guide to PSF-Matching Configs

    To make `psfMatchedWarps`, select `config.makePsfMatched=True`. The subtask
    @link lsst.ip.diffim.modelPsfMatch.ModelPsfMatchTask ModelPsfMatchTask @endlink
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
    @link lsst.ip.diffim.modelPsfMatch.ModelPsfMatchTask ModelPsfMatchTask @endlink
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
         --config doApplyUberCal=False makePsfMatched=True modelPsf.defaultFwhm=11

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

    @pipeBase.timeMethod
    def runDataRef(self, patchRef, selectDataList=[]):
        """!Produce <coaddName>Coadd_<warpType>Warp images by warping and optionally PSF-matching.

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
            primaryWarpDataset = self.getTempExpDatasetName("psfMatched")
        else:
            primaryWarpDataset = self.getTempExpDatasetName("direct")

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
                    calExpRef = calExpRef.butlerSubset.butler.dataRef("calexp", dataId=calExpRef.dataId,
                                                                      tract=skyInfo.tractInfo.getId())
                    calExp = self.getCalibratedExposure(calExpRef, bgSubtracted=self.config.bgSubtracted)
                except Exception as e:
                    self.log.warn("Calexp %s not found; skipping it: %s", calExpRef.dataId, e)
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
                self.log.warn("Warp %s could not be created", tempExpRef.dataId)

            if self.config.doWrite:
                for (warpType, exposure) in exps.items():  # compatible w/ Py3
                    if exposure is not None:
                        self.log.info("Persisting %s" % self.getTempExpDatasetName(warpType))
                        tempExpRef.put(exposure, self.getTempExpDatasetName(warpType))

        return dataRefList

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
                self.log.warn("WarpAndPsfMatch failed for calexp %s; skipping it: %s", dataId, e)
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
                        mimg *= (coaddTempExp.getCalib().getFluxMag0()[0] /
                                 exposure.getCalib().getFluxMag0()[0])
                        del mimg
                    numGoodPix[warpType] = coaddUtils.copyGoodPixels(
                        coaddTempExp.getMaskedImage(), exposure.getMaskedImage(), self.getBadPixelMask())
                    totGoodPix[warpType] += numGoodPix[warpType]
                    self.log.debug("Calexp %s has %d good pixels in this patch (%.1f%%) for %s",
                                   dataId, numGoodPix[warpType],
                                   100.0*numGoodPix[warpType]/skyInfo.bbox.getArea(), warpType)
                    if numGoodPix[warpType] > 0 and not didSetMetadata[warpType]:
                        coaddTempExp.setCalib(exposure.getCalib())
                        coaddTempExp.setFilter(exposure.getFilter())
                        coaddTempExp.getInfo().setVisitInfo(exposure.getInfo().getVisitInfo())
                        # PSF replaced with CoaddPsf after loop if and only if creating direct warp
                        coaddTempExp.setPsf(exposure.getPsf())
                        didSetMetadata[warpType] = True

                    # Need inputRecorder for CoaddApCorrMap for both direct and PSF-matched
                    inputRecorder[warpType].addCalExp(calExp, ccdId, numGoodPix[warpType])

            except Exception as e:
                self.log.warn("Error processing calexp %s; skipping it: %s", dataId, e)
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
                # No good pixels. Exposure still empty
                coaddTempExps[warpType] = None

        result = pipeBase.Struct(exposures=coaddTempExps)
        return result

    def getCalibratedExposure(self, dataRef, bgSubtracted):
        """Return one calibrated Exposure, possibly with an updated SkyWcs.

        @param[in] dataRef        a sensor-level data reference
        @param[in] bgSubtracted   return calexp with background subtracted? If False get the
                                  calexp's background background model and add it to the calexp.
        @return calibrated exposure

        @raises MissingExposureError If data for the exposure is not available.

        If config.doApplyUberCal, the exposure will be photometrically
        calibrated via the `jointcal_photoCalib` dataset and have its SkyWcs
        updated to the `jointcal_wcs`, otherwise it will be calibrated via the
        Exposure's own Calib and have the original SkyWcs.
        """
        try:
            exposure = dataRef.get("calexp", immediate=True)
        except dafPersist.NoResults as e:
            raise MissingExposureError('Exposure not found: %s ' % str(e)) from e

        if not bgSubtracted:
            background = dataRef.get("calexpBackground", immediate=True)
            mi = exposure.getMaskedImage()
            mi += background.getImage()
            del mi

        # TODO: this is needed until DM-10153 is done and Calib is gone
        referenceFlux = 1e23 * 10**(48.6 / -2.5) * 1e9
        if self.config.doApplyUberCal:
            if self.config.useMeasMosaic:
                from lsst.meas.mosaic import applyMosaicResultsExposure
                # NOTE: this changes exposure in-place, updating its Calib and Wcs.
                # Save the calibration error, as it gets overwritten with zero.
                fluxMag0Err = exposure.getCalib().getFluxMag0()[1]
                try:
                    applyMosaicResultsExposure(dataRef, calexp=exposure)
                except dafPersist.NoResults as e:
                    raise MissingExposureError('Mosaic calibration not found: %s ' % str(e)) from e
                fluxMag0 = exposure.getCalib().getFluxMag0()[0]
                photoCalib = afwImage.PhotoCalib(referenceFlux/fluxMag0,
                                                 referenceFlux*fluxMag0Err/fluxMag0**2,
                                                 exposure.getBBox())
            else:
                photoCalib = dataRef.get("jointcal_photoCalib")
                skyWcs = dataRef.get("jointcal_wcs")
                exposure.setWcs(skyWcs)
        else:
            fluxMag0 = exposure.getCalib().getFluxMag0()
            photoCalib = afwImage.PhotoCalib(referenceFlux/fluxMag0[0],
                                             referenceFlux*fluxMag0[1]/fluxMag0[0]**2,
                                             exposure.getBBox())

        exposure.maskedImage = photoCalib.calibrateImage(exposure.maskedImage,
                                                         includeScaleUncertainty=self.config.includeCalibVar)
        exposure.maskedImage /= photoCalib.getCalibrationMean()
        exposure.setCalib(afwImage.Calib(photoCalib.getInstFluxAtZeroMagnitude()))
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
        if isinstance(calexp, afwImage.Exposure):
            calexp = calexp.getMaskedImage()
        calexp -= bg.getImage()


class MakeWarpConfig(pipeBase.PipelineTaskConfig, MakeCoaddTempExpConfig):
    calExpList = pipeBase.InputDatasetField(
        doc="Exposures which are resampled and optionally PSF-Matched onto SkyMap projection and pixelation",
        name="calexp",
        storageClass="ExposureF",
        dimensions=("Visit", "Detector")
    )
    backgroundList = pipeBase.InputDatasetField(
        doc="Backgrounds to optionally add back into the calexp",
        name="calexpBackground",
        storageClass="Background",
        dimensions=("Visit", "Detector")
    )
    skyCorrList = pipeBase.InputDatasetField(
        doc="SkyCorr",
        name="skyCorr",
        storageClass="Background",
        dimensions=("Visit", "Detector")
    )
    jointcal_wcs = pipeBase.InputDatasetField(
        doc=("WCS from jointcal or meas_mosaic Placeholder. Pre-flight gets angry about the tract."
             "Not tested because ci_hsc does not run jointcal or meas_mosaic."),
        name="jointcal_wcs",
        storageClass="FitsCatalogStorage",
        dimensions=("Visit", "Detector, Tract")
    )
    jointcal_photoCalib = pipeBase.InputDatasetField(
        doc=("PhotoCalib from jointcal or meas_mosaic Placeholder.  Pre-flight gets angry about the tract."
             "Not tested because ci_hsc does not run jointcal or meas_mosaic."),
        name="jointcal_photoCalib",
        storageClass="FitsCatalogStorage",
        dimensions=("Visit", "Detector, Tract")
    )
    skyMap = pipeBase.InputDatasetField(
        doc="SkyMap to be used in merging",
        nameTemplate="{coaddName}Coadd_skyMap",
        storageClass="SkyMap",
        dimensions=("SkyMap",),
        scalar=True
    )
    direct = pipeBase.OutputDatasetField(
        doc="Warp",
        nameTemplate="{coaddName}Coadd_directWarp",
        storageClass="ExposureF",
        dimensions=("Tract", "Patch", "SkyMap", "Visit"),
        scalar=True
    )
    psfMatched = pipeBase.OutputDatasetField(
        doc="Warp",
        nameTemplate="{coaddName}Coadd_psfMatchedWarp",
        storageClass="ExposureF",
        dimensions=("Tract", "Patch", "SkyMap", "Visit"),
        scalar=True
    )

    def setDefaults(self):
        self.formatTemplateNames({"coaddName": "deep"})
        self.quantum.dimensions = ("Tract", "Patch", "SkyMap", "Visit")

    def validate(self):
        super().validate(self)
        if self.doApplyUbercal:
            log.warn("Gen3 MakeWarpTask cannot apply meas_mosaic or jointcal results."
                     "Please set doApplyUbercal=False.")


class MakeWarpTask(MakeCoaddTempExpTask, pipeBase.PipelineTask):
    """First Draft of a Gen3 compatible MakeWarp Task

    Currently doesn't not handle doApplyUbercal=True
    """
    ConfigClass = MakeWarpConfig
    _DefaultName = "makeWarp"

    @classmethod
    def getInputDatasetTypes(cls, config):
        inputTypeDict = super().getInputDatasetTypes(config)
        # remove all optional datasets from InputDatasetsTypes per configs
        if config.bgSubtracted:
            inputTypeDict.pop("backgroundList", None)
        if not config.doApplySkyCorr:
            inputTypeDict.pop("skyCorr", None)
        if not config.doApplyUberCal:
            inputTypeDict.pop("jointcal_wcs", None)
            inputTypeDict.pop("jointcal_photoCalib", None)
        return inputTypeDict

    @classmethod
    def getOutputDatasetTypes(cls, config):
        outputTypeDict = super().getOutputDatasetTypes(config)
        if not config.makeDirect:
            outputTypeDict.pop("direct", None)
        if not config.makePsfMatched:
            outputTypeDict.pop("psfMatched", None)
        return outputTypeDict

    def adaptArgsAndRun(self, inputData, inputDataIds, outputDataIds, butler):
        self.prepareCalibratedExposures(**inputData)
        dataIdList = inputDataIds['calExpList']
        inputData['dataIdList'] = dataIdList
        inputData['ccdIdList'] = [dataId['detector'] for dataId in dataIdList]
        visits = [dataId['visit'] for dataId in dataIdList]
        assert(all(visits[0] == visit for visit in visits))
        inputData["visitId"] = visits[0]
        skyMap = inputData.pop("skyMap")
        outputDataId = next(iter(outputDataIds.values()))
        inputData['skyInfo'] = makeSkyInfo(skyMap,
                                           tractId=outputDataId['tract'],
                                           patchId=outputDataId['patch'])
        results = self.run(**inputData)
        return pipeBase.Struct(**results.exposures)

    def prepareCalibratedExposures(self, calExpList, backgroundList=None, skyCorrList=None, **kwargs):
        """Calibrate and add backgrounds to input calExpList in place

        TODO DM-17062: apply jointcal/meas_mosaic here

        Parameters
        ----------
        calExpList : `list` of `lsst.afw.image.Exposure`
            Sequence of calexps to be modified in place
        backgroundList : `list` of `lsst.afw.math.backgroundList`
            Sequence of backgrounds to be added back in if bgSubtracted=False
        skyCorrList : `list` of `lsst.afw.math.backgroundList`
            Sequence of background corrections to be subtracted if doApplySkyCorr=True
        """
        backgroundList = len(calExpList)*[None] if backgroundList is None else backgroundList
        skyCorrList = len(calExpList)*[None] if skyCorrList is None else skyCorrList
        for calexp, background, skyCorr in zip(calExpList, backgroundList, skyCorrList):
            mi = calexp.maskedImage
            if not self.config.bgSubtracted:
                mi += background.getImage()
            if self.config.doApplySkyCorr:
                mi -= skyCorr.getImage()
            del mi
