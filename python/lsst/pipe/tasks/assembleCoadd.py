from __future__ import absolute_import, division, print_function
from builtins import zip
from builtins import range
#
# LSST Data Management System
# Copyright 2008-2016 AURA/LSST.
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
# see <https://www.lsstcorp.org/LegalNotices/>.
#
import os
import numpy
import lsst.pex.config as pexConfig
import lsst.pex.exceptions as pexExceptions
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.afw.detection as afwDet
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
import lsst.meas.algorithms as measAlg
import lsst.log as log
import lsstDebug
from .coaddBase import CoaddBaseTask, SelectDataIdContainer
from .interpImage import InterpImageTask
from .matchBackgrounds import MatchBackgroundsTask
from .scaleZeroPoint import ScaleZeroPointTask
from .coaddHelpers import groupPatchExposures, getGroupDataRef
from lsst.meas.algorithms import SourceDetectionTask

__all__ = ["AssembleCoaddTask", "SafeClipAssembleCoaddTask", "CompareWarpAssembleCoaddTask"]


class AssembleCoaddConfig(CoaddBaseTask.ConfigClass):
    """!
\anchor AssembleCoaddConfig_

\brief Configuration parameters for the \ref AssembleCoaddTask_ "AssembleCoaddTask"
    """
    warpType = pexConfig.Field(
        doc="Warp name: one of 'direct' or 'psfMatched'",
        dtype=str,
        default="direct",
    )
    subregionSize = pexConfig.ListField(
        dtype=int,
        doc="Width, height of stack subregion size; "
        "make small enough that a full stack of images will fit into memory at once.",
        length=2,
        default=(2000, 2000),
    )
    statistic = pexConfig.Field(
        dtype=str,
        doc="Main stacking statistic for aggregating over the epochs.",
        default="MEANCLIP",
    )
    doSigmaClip = pexConfig.Field(
        dtype=bool,
        doc="Perform sigma clipped outlier rejection with MEANCLIP statistic? (DEPRECATED)",
        default=False,
    )
    sigmaClip = pexConfig.Field(
        dtype=float,
        doc="Sigma for outlier rejection; ignored if non-clipping statistic selected.",
        default=3.0,
    )
    clipIter = pexConfig.Field(
        dtype=int,
        doc="Number of iterations of outlier rejection; ignored if non-clipping statistic selected.",
        default=2,
    )
    scaleZeroPoint = pexConfig.ConfigurableField(
        target=ScaleZeroPointTask,
        doc="Task to adjust the photometric zero point of the coadd temp exposures",
    )
    doInterp = pexConfig.Field(
        doc="Interpolate over NaN pixels? Also extrapolate, if necessary, but the results are ugly.",
        dtype=bool,
        default=True,
    )
    interpImage = pexConfig.ConfigurableField(
        target=InterpImageTask,
        doc="Task to interpolate (and extrapolate) over NaN pixels",
    )
    matchBackgrounds = pexConfig.ConfigurableField(
        target=MatchBackgroundsTask,
        doc="Task to match backgrounds",
    )
    maxMatchResidualRatio = pexConfig.Field(
        doc="Maximum ratio of the mean squared error of the background matching model to the variance "
        "of the difference in backgrounds",
        dtype=float,
        default=1.1
    )
    maxMatchResidualRMS = pexConfig.Field(
        doc="Maximum RMS of residuals of the background offset fit in matchBackgrounds.",
        dtype=float,
        default=1.0
    )
    doWrite = pexConfig.Field(
        doc="Persist coadd?",
        dtype=bool,
        default=True,
    )
    doNImage = pexConfig.Field(
        doc="Create image of number of contributing exposures for each pixel",
        dtype=bool,
        default=False,
    )
    doMatchBackgrounds = pexConfig.Field(
        doc="Match backgrounds of coadd temp exposures before coadding them? "
        "If False, the coadd temp expsosures must already have been background subtracted or matched",
        dtype=bool,
        default=False,
    )
    autoReference = pexConfig.Field(
        doc="Automatically select the coadd temp exposure to use as a reference for background matching? "
        "Ignored if doMatchBackgrounds false. "
        "If False you must specify the reference temp exposure as the data Id",
        dtype=bool,
        default=True,
    )
    maskPropagationThresholds = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        doc=("Threshold (in fractional weight) of rejection at which we propagate a mask plane to "
             "the coadd; that is, we set the mask bit on the coadd if the fraction the rejected frames "
             "would have contributed exceeds this value."),
        default={"SAT": 0.1},
    )
    removeMaskPlanes = pexConfig.ListField(dtype=str, default=["NOT_DEBLENDED"],
                                           doc="Mask planes to remove before coadding")
    #
    # N.b. These configuration options only set the bitplane config.brightObjectMaskName
    # To make this useful you *must* also configure the flags.pixel algorithm, for example
    # by adding
    #   config.measurement.plugins["base_PixelFlags"].masksFpCenter.append("BRIGHT_OBJECT")
    #   config.measurement.plugins["base_PixelFlags"].masksFpAnywhere.append("BRIGHT_OBJECT")
    # to your measureCoaddSources.py and forcedPhotCoadd.py config overrides
    #
    doMaskBrightObjects = pexConfig.Field(dtype=bool, default=False,
                                          doc="Set mask and flag bits for bright objects?")
    brightObjectMaskName = pexConfig.Field(dtype=str, default="BRIGHT_OBJECT",
                                           doc="Name of mask bit used for bright objects")

    coaddPsf = pexConfig.ConfigField(
        doc="Configuration for CoaddPsf",
        dtype=measAlg.CoaddPsfConfig,
    )

    def setDefaults(self):
        CoaddBaseTask.ConfigClass.setDefaults(self)
        self.badMaskPlanes = ["NO_DATA", "BAD", "CR", ]

    def validate(self):
        CoaddBaseTask.ConfigClass.validate(self)
        if self.doPsfMatch:
            # Backwards compatibility.
            # Configs do not have loggers
            log.warn("Config doPsfMatch deprecated. Setting warpType='psfMatched'")
            self.warpType = 'psfMatched'
        if self.doSigmaClip and self.statistic != "MEANCLIP":
            log.warn('doSigmaClip deprecated. To replicate behavior, setting statistic to "MEANCLIP"')
            self.statistic = "MEANCLIP"
        if self.doInterp and self.statistic not in ['MEAN', 'MEDIAN', 'MEANCLIP', 'VARIANCE', 'VARIANCECLIP']:
            raise ValueError("Must set doInterp=False for statistic=%s, which does not "
                             "compute and set a non-zero coadd variance estimate." % (self.statistic))

        unstackableStats = ['NOTHING', 'ERROR', 'ORMASK']
        if not hasattr(afwMath.Property, self.statistic) or self.statistic in unstackableStats:
            stackableStats = [str(k) for k in afwMath.Property.__members__.keys()
                              if str(k) not in unstackableStats]
            raise ValueError("statistic %s is not allowed. Please choose one of %s."
                             % (self.statistic, stackableStats))


## \addtogroup LSST_task_documentation
## \{
## \page AssembleCoaddTask
## \ref AssembleCoaddTask_ "AssembleCoaddTask"
## \copybrief AssembleCoaddTask
## \}
class AssembleCoaddTask(CoaddBaseTask):
    """!
\anchor AssembleCoaddTask_

\brief Assemble a coadded image from a set of warps (coadded temporary exposures).

\section pipe_tasks_assembleCoadd_Contents Contents
  - \ref pipe_tasks_assembleCoadd_AssembleCoaddTask_Purpose
  - \ref pipe_tasks_assembleCoadd_AssembleCoaddTask_Initialize
  - \ref pipe_tasks_assembleCoadd_AssembleCoaddTask_Run
  - \ref pipe_tasks_assembleCoadd_AssembleCoaddTask_Config
  - \ref pipe_tasks_assembleCoadd_AssembleCoaddTask_Debug
  - \ref pipe_tasks_assembleCoadd_AssembleCoaddTask_Example

\section pipe_tasks_assembleCoadd_AssembleCoaddTask_Purpose	Description

\copybrief AssembleCoaddTask_

We want to assemble a coadded image from a set of Warps (also called
coadded temporary exposures or coaddTempExps.
Each input Warp covers a patch on the sky and corresponds to a single run/visit/exposure of the
covered patch. We provide the task with a list of Warps (selectDataList) from which it selects
Warps that cover the specified patch (pointed at by dataRef).
Each Warp that goes into a coadd will typically have an independent photometric zero-point.
Therefore, we must scale each Warp to set it to a common photometric zeropoint. By default, each
Warp has backgrounds and hence will require config.doMatchBackgrounds=True.
When background matching is enabled, the task may be configured to automatically select a reference exposure
(config.autoReference=True). If this is not done, we require that the input dataRef provides access to a
Warp (dataset type coaddName + 'Coadd' + warpType + 'Warp') which is used as the reference exposure.
WarpType may be one of 'direct' or 'psfMatched', and the boolean configs config.makeDirect and
config.makePsfMatched set which of the warp types will be coadded.
The coadd is computed as a mean with optional outlier rejection.
Criteria for outlier rejection are set in \ref AssembleCoaddConfig. Finally, Warps can have bad 'NaN'
pixels which received no input from the source calExps. We interpolate over these bad (NaN) pixels.

AssembleCoaddTask uses several sub-tasks. These are
<DL>
  <DT>\ref ScaleZeroPointTask_ "ScaleZeroPointTask"</DT>
  <DD> create and use an imageScaler object to scale the photometric zeropoint for each Warp</DD>
  <DT>\ref MatchBackgroundsTask_ "MatchBackgroundsTask"</DT>
  <DD> match background in a Warp to a reference exposure (and select the reference exposure if one is
  not provided).</DD>
  <DT>\ref InterpImageTask_ "InterpImageTask"</DT>
  <DD>interpolate across bad pixels (NaN) in the final coadd</DD>
</DL>
You can retarget these subtasks if you wish.

\section pipe_tasks_assembleCoadd_AssembleCoaddTask_Initialize       Task initialization
\copydoc \_\_init\_\_

\section pipe_tasks_assembleCoadd_AssembleCoaddTask_Run       Invoking the Task
\copydoc run

\section pipe_tasks_assembleCoadd_AssembleCoaddTask_Config       Configuration parameters
See \ref AssembleCoaddConfig_

\section pipe_tasks_assembleCoadd_AssembleCoaddTask_Debug		Debug variables
The \link lsst.pipe.base.cmdLineTask.CmdLineTask command line task\endlink interface supports a
flag \c -d to import \b debug.py from your \c PYTHONPATH; see \ref baseDebug for more about \b debug.py files.
AssembleCoaddTask has no debug variables of its own. Some of the subtasks may support debug variables. See
the documetation for the subtasks for further information.

\section pipe_tasks_assembleCoadd_AssembleCoaddTask_Example	A complete example of using AssembleCoaddTask

AssembleCoaddTask assembles a set of warped images into a coadded image. The AssembleCoaddTask
can be invoked by running assembleCoadd.py with the flag '--legacyCoadd'. Usage of assembleCoadd.py expects
a data reference to the tract patch and filter to be coadded (specified using
'--id = [KEY=VALUE1[^VALUE2[^VALUE3...] [KEY=VALUE1[^VALUE2[^VALUE3...] ...]]') along with a list of
Warps to attempt to coadd (specified using
'--selectId [KEY=VALUE1[^VALUE2[^VALUE3...] [KEY=VALUE1[^VALUE2[^VALUE3...] ...]]'). Only the Warps
that cover the specified tract and patch will be coadded. A list of the available optional
arguments can be obtained by calling assembleCoadd.py with the --help command line argument:
\code
assembleCoadd.py --help
\endcode
To demonstrate usage of the AssembleCoaddTask in the larger context of multi-band processing, we will generate
the HSC-I & -R band coadds from HSC engineering test data provided in the ci_hsc package. To begin, assuming
that the lsst stack has been already set up, we must set up the obs_subaru and ci_hsc packages.
This defines the environment variable $CI_HSC_DIR and points at the location of the package. The raw HSC
data live in the $CI_HSC_DIR/raw directory. To begin assembling the coadds, we must first
<DL>
  <DT>processCcd</DT>
  <DD> process the individual ccds in $CI_HSC_RAW to produce calibrated exposures</DD>
  <DT>makeSkyMap</DT>
  <DD> create a skymap that covers the area of the sky present in the raw exposures</DD>
  <DT>makeCoaddTempExp</DT>
  <DD> warp the individual calibrated exposures to the tangent plane of the coadd</DD>
</DL>
We can perform all of these steps by running
\code
$CI_HSC_DIR scons warp-903986 warp-904014 warp-903990 warp-904010 warp-903988
\endcode
This will produce warped exposures for each visit. To coadd the warped data, we call assembleCoadd.py as
follows:
\code
assembleCoadd.py --legacyCoadd $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-I \
--selectId visit=903986 ccd=16 --selectId visit=903986 ccd=22 --selectId visit=903986 ccd=23 \
--selectId visit=903986 ccd=100 --selectId visit=904014 ccd=1 --selectId visit=904014 ccd=6 \
--selectId visit=904014 ccd=12 --selectId visit=903990 ccd=18 --selectId visit=903990 ccd=25 \
--selectId visit=904010 ccd=4 --selectId visit=904010 ccd=10 --selectId visit=904010 ccd=100 \
--selectId visit=903988 ccd=16 --selectId visit=903988 ccd=17 --selectId visit=903988 ccd=23 \
--selectId visit=903988 ccd=24
\endcode
that will process the HSC-I band data. The results are written in
`$CI_HSC_DIR/DATA/deepCoadd-results/HSC-I`.

You may also choose to run:
\code
scons warp-903334 warp-903336 warp-903338 warp-903342 warp-903344 warp-903346
assembleCoadd.py --legacyCoadd $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-R \
--selectId visit=903334 ccd=16 --selectId visit=903334 ccd=22 --selectId visit=903334 ccd=23 \
--selectId visit=903334 ccd=100 --selectId visit=903336 ccd=17 --selectId visit=903336 ccd=24 \
--selectId visit=903338 ccd=18 --selectId visit=903338 ccd=25 --selectId visit=903342 ccd=4 \
--selectId visit=903342 ccd=10 --selectId visit=903342 ccd=100 --selectId visit=903344 ccd=0 \
--selectId visit=903344 ccd=5 --selectId visit=903344 ccd=11 --selectId visit=903346 ccd=1 \
--selectId visit=903346 ccd=6 --selectId visit=903346 ccd=12
\endcode
to generate the coadd for the HSC-R band if you are interested in following multiBand Coadd processing as
discussed in \ref pipeTasks_multiBand (but note that normally, one would use the
\ref SafeClipAssembleCoaddTask_ "SafeClipAssembleCoaddTask" rather than AssembleCoaddTask to make the coadd.
    """
    ConfigClass = AssembleCoaddConfig
    _DefaultName = "assembleCoadd"

    def __init__(self, *args, **kwargs):
        """!
        \brief Initialize the task. Create the \ref InterpImageTask "interpImage",
        \ref MatchBackgroundsTask "matchBackgrounds", & \ref ScaleZeroPointTask "scaleZeroPoint" subtasks.
        """
        CoaddBaseTask.__init__(self, *args, **kwargs)
        self.makeSubtask("interpImage")
        self.makeSubtask("matchBackgrounds")
        self.makeSubtask("scaleZeroPoint")

        if self.config.doMaskBrightObjects:
            mask = afwImage.Mask()
            try:
                self.brightObjectBitmask = 1 << mask.addMaskPlane(self.config.brightObjectMaskName)
            except pexExceptions.LsstCppException:
                raise RuntimeError("Unable to define mask plane for bright objects; planes used are %s" %
                                   mask.getMaskPlaneDict().keys())
            del mask

        self.warpType = self.config.warpType

    @pipeBase.timeMethod
    def run(self, dataRef, selectDataList=[]):
        """!
        \brief Assemble a coadd from a set of Warps

        Coadd a set of Warps. Compute weights to be applied to each Warp and find scalings to
        match the photometric zeropoint to a reference Warp. Optionally, match backgrounds across
        Warps if the background has not already been removed. Assemble the Warps using
        \ref assemble. Interpolate over NaNs and optionally write the coadd to disk. Return the coadded
        exposure.

        \anchor runParams
        \param[in] dataRef: Data reference defining the patch for coaddition and the reference Warp
                        (if config.autoReference=False). Used to access the following data products:
                        - [in] self.config.coaddName + "Coadd_skyMap"
                        - [in] self.config.coaddName + "Coadd_ + <warpType> + "Warp" (optionally)
                        - [out] self.config.coaddName + "Coadd"
        \param[in] selectDataList[in]: List of data references to Warps. Data to be coadded will be
                                   selected from this list based on overlap with the patch defined by dataRef.

        \return a pipeBase.Struct with fields:
                 - coaddExposure: coadded exposure
                 - nImage: exposure count image
        """
        skyInfo = self.getSkyInfo(dataRef)
        calExpRefList = self.selectExposures(dataRef, skyInfo, selectDataList=selectDataList)
        if len(calExpRefList) == 0:
            self.log.warn("No exposures to coadd")
            return
        self.log.info("Coadding %d exposures", len(calExpRefList))

        tempExpRefList = self.getTempExpRefList(dataRef, calExpRefList)
        inputData = self.prepareInputs(tempExpRefList)
        self.log.info("Found %d %s", len(inputData.tempExpRefList),
                      self.getTempExpDatasetName(self.warpType))
        if len(inputData.tempExpRefList) == 0:
            self.log.warn("No coadd temporary exposures found")
            return
        if self.config.doMatchBackgrounds:
            refImageScaler = self.getBackgroundReferenceScaler(dataRef)
            inputData = self.backgroundMatching(inputData, dataRef, refImageScaler)
            if len(inputData.tempExpRefList) == 0:
                self.log.warn("No valid background models")
                return

        supplementaryData = self.makeSupplementaryData(dataRef, selectDataList)

        retStruct = self.assemble(skyInfo, inputData.tempExpRefList, inputData.imageScalerList,
                                  inputData.weightList,
                                  inputData.backgroundInfoList if self.config.doMatchBackgrounds else None,
                                  supplementaryData=supplementaryData)

        if self.config.doMatchBackgrounds:
            self.addBackgroundMatchingMetadata(retStruct.coaddExposure, inputData.tempExpRefList,
                                               inputData.backgroundInfoList)

        if self.config.doInterp:
            self.interpImage.run(retStruct.coaddExposure.getMaskedImage(), planeName="NO_DATA")
            # The variance must be positive; work around for DM-3201.
            varArray = retStruct.coaddExposure.getMaskedImage().getVariance().getArray()
            varArray[:] = numpy.where(varArray > 0, varArray, numpy.inf)

        if self.config.doMaskBrightObjects:
            brightObjectMasks = self.readBrightObjectMasks(dataRef)
            self.setBrightObjectMasks(retStruct.coaddExposure, dataRef.dataId, brightObjectMasks)

        if self.config.doWrite:
            self.log.info("Persisting %s" % self.getCoaddDatasetName(self.warpType))
            dataRef.put(retStruct.coaddExposure, self.getCoaddDatasetName(self.warpType))
            if retStruct.nImage is not None:
                dataRef.put(retStruct.nImage, self.getCoaddDatasetName(self.warpType) + '_nImage')

        return retStruct

    def makeSupplementaryData(self, dataRef, selectDataList):
        """!
        \brief Make additional inputs to assemble() specific to subclasses.

        Available to be implemented by subclasses only if they need the
        coadd dataRef for performing preliminary processing before
        assembling the coadd.
        """
        pass

    def getTempExpRefList(self, patchRef, calExpRefList):
        """!
        \brief Generate list data references corresponding to warped exposures that lie within the
        patch to be coadded.

        \param[in] patchRef: Data reference for patch
        \param[in] calExpRefList: List of data references for input calexps
        \return List of Warp/CoaddTempExp data references
        """
        butler = patchRef.getButler()
        groupData = groupPatchExposures(patchRef, calExpRefList, self.getCoaddDatasetName(self.warpType),
                                        self.getTempExpDatasetName(self.warpType))
        tempExpRefList = [getGroupDataRef(butler, self.getTempExpDatasetName(self.warpType),
                                          g, groupData.keys) for
                          g in groupData.groups.keys()]
        return tempExpRefList

    def getBackgroundReferenceScaler(self, dataRef):
        """!
        \brief Construct an image scaler for the background reference frame

        Each Warp has a different background level. A reference background level must be chosen before
        coaddition. If config.autoReference=True, \ref backgroundMatching will pick the reference level and
        this routine is a no-op and None is returned. Otherwise, use the
        \ref ScaleZeroPointTask_ "scaleZeroPoint" subtask to compute an imageScaler object for the provided
        reference image and return it.

        \param[in] dataRef: Data reference for the background reference frame, or None
        \return image scaler, or None
        """
        if self.config.autoReference:
            return None

        # We've been given the data reference
        dataset = self.getTempExpDatasetName(self.warpType)
        if not dataRef.datasetExists(dataset):
            raise RuntimeError("Could not find reference exposure %s %s." % (dataset, dataRef.dataId))

        refExposure = dataRef.get(self.getTempExpDatasetName(self.warpType), immediate=True)
        refImageScaler = self.scaleZeroPoint.computeImageScaler(
            exposure=refExposure,
            dataRef=dataRef,
        )
        return refImageScaler

    def prepareInputs(self, refList):
        """!
        \brief Prepare the input warps for coaddition by measuring the weight for each warp and the scaling
        for the photometric zero point.

        Each Warp has its own photometric zeropoint and background variance. Before coadding these
        Warps together, compute a scale factor to normalize the photometric zeropoint and compute the
        weight for each Warp.

        \param[in] refList: List of data references to tempExp
        \return Struct:
        - tempExprefList: List of data references to tempExp
        - weightList: List of weightings
        - imageScalerList: List of image scalers
        """
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(self.config.sigmaClip)
        statsCtrl.setNumIter(self.config.clipIter)
        statsCtrl.setAndMask(self.getBadPixelMask())
        statsCtrl.setNanSafe(True)
        # compute tempExpRefList: a list of tempExpRef that actually exist
        # and weightList: a list of the weight of the associated coadd tempExp
        # and imageScalerList: a list of scale factors for the associated coadd tempExp
        tempExpRefList = []
        weightList = []
        imageScalerList = []
        tempExpName = self.getTempExpDatasetName(self.warpType)
        for tempExpRef in refList:
            if not tempExpRef.datasetExists(tempExpName):
                self.log.warn("Could not find %s %s; skipping it", tempExpName, tempExpRef.dataId)
                continue

            tempExp = tempExpRef.get(tempExpName, immediate=True)
            maskedImage = tempExp.getMaskedImage()
            imageScaler = self.scaleZeroPoint.computeImageScaler(
                exposure=tempExp,
                dataRef=tempExpRef,
            )
            try:
                imageScaler.scaleMaskedImage(maskedImage)
            except Exception as e:
                self.log.warn("Scaling failed for %s (skipping it): %s", tempExpRef.dataId, e)
                continue
            statObj = afwMath.makeStatistics(maskedImage.getVariance(), maskedImage.getMask(),
                                             afwMath.MEANCLIP, statsCtrl)
            meanVar, meanVarErr = statObj.getResult(afwMath.MEANCLIP)
            weight = 1.0 / float(meanVar)
            if not numpy.isfinite(weight):
                self.log.warn("Non-finite weight for %s: skipping", tempExpRef.dataId)
                continue
            self.log.info("Weight of %s %s = %0.3f", tempExpName, tempExpRef.dataId, weight)

            del maskedImage
            del tempExp

            tempExpRefList.append(tempExpRef)
            weightList.append(weight)
            imageScalerList.append(imageScaler)

        return pipeBase.Struct(tempExpRefList=tempExpRefList, weightList=weightList,
                               imageScalerList=imageScalerList)

    def backgroundMatching(self, inputData, refExpDataRef=None, refImageScaler=None):
        """!
        \brief Perform background matching on the prepared inputs

        Each Warp has a different background level that must be normalized to a reference level
        before coaddition. If no reference is provided, the background matcher selects one. If the background
        matching is performed sucessfully, recompute the weight to be applied to the Warp (coaddTempExp) to be
        consistent with the scaled background.

        \param[in] inputData: Struct from prepareInputs() with tempExpRefList, weightList, imageScalerList
        \param[in] refExpDataRef: Data reference for background reference Warp, or None
        \param[in] refImageScaler: Image scaler for background reference Warp, or None
        \return Struct:
        - tempExprefList: List of data references to warped exposures (coaddTempExps)
        - weightList: List of weightings
        - imageScalerList: List of image scalers
        - backgroundInfoList: result from background matching
        """
        try:
            backgroundInfoList = self.matchBackgrounds.run(
                expRefList=inputData.tempExpRefList,
                imageScalerList=inputData.imageScalerList,
                refExpDataRef=refExpDataRef if not self.config.autoReference else None,
                refImageScaler=refImageScaler,
                expDatasetType=self.getTempExpDatasetName(self.warpType),
            ).backgroundInfoList
        except Exception as e:
            self.log.fatal("Cannot match backgrounds: %s", e)
            raise pipeBase.TaskError("Background matching failed.")

        newWeightList = []
        newTempExpRefList = []
        newBackgroundStructList = []
        newScaleList = []
        # the number of good backgrounds may be < than len(tempExpList)
        # sync these up and correct the weights
        for tempExpRef, bgInfo, scaler, weight in zip(inputData.tempExpRefList, backgroundInfoList,
                                                      inputData.imageScalerList, inputData.weightList):
            if not bgInfo.isReference:
                # skip exposure if it has no backgroundModel
                # or if fit was bad
                if (bgInfo.backgroundModel is None):
                    self.log.info("No background offset model available for %s: skipping", tempExpRef.dataId)
                    continue
                try:
                    varianceRatio = bgInfo.matchedMSE / bgInfo.diffImVar
                except Exception as e:
                    self.log.info("MSE/Var ratio not calculable (%s) for %s: skipping",
                                  e, tempExpRef.dataId)
                    continue
                if not numpy.isfinite(varianceRatio):
                    self.log.info("MSE/Var ratio not finite (%.2f / %.2f) for %s: skipping",
                                  bgInfo.matchedMSE, bgInfo.diffImVar, tempExpRef.dataId)
                    continue
                elif (varianceRatio > self.config.maxMatchResidualRatio):
                    self.log.info("Bad fit. MSE/Var ratio %.2f > %.2f for %s: skipping",
                                  varianceRatio, self.config.maxMatchResidualRatio, tempExpRef.dataId)
                    continue
                elif (bgInfo.fitRMS > self.config.maxMatchResidualRMS):
                    self.log.info("Bad fit. RMS %.2f > %.2f for %s: skipping",
                                  bgInfo.fitRMS, self.config.maxMatchResidualRMS, tempExpRef.dataId)
                    continue
            newWeightList.append(1 / (1 / weight + bgInfo.fitRMS**2))
            newTempExpRefList.append(tempExpRef)
            newBackgroundStructList.append(bgInfo)
            newScaleList.append(scaler)

        return pipeBase.Struct(tempExpRefList=newTempExpRefList, weightList=newWeightList,
                               imageScalerList=newScaleList, backgroundInfoList=newBackgroundStructList)

    def assemble(self, skyInfo, tempExpRefList, imageScalerList, weightList, bgInfoList=None,
                 altMaskList=None, mask=None, supplementaryData=None):
        """!
        \anchor AssembleCoaddTask.assemble_

        \brief Assemble a coadd from input warps

        Assemble the coadd using the provided list of coaddTempExps. Since the full coadd covers a patch (a
        large area), the assembly is performed over small areas on the image at a time in order to
        conserve memory usage. Iterate over subregions within the outer bbox of the patch using
        \ref assembleSubregion to stack the corresponding subregions from the coaddTempExps with the
        statistic specified. Set the edge bits the coadd mask based on the weight map.

        \param[in] skyInfo: Patch geometry information, from getSkyInfo
        \param[in] tempExpRefList: List of data references to Warps (previously called CoaddTempExps)
        \param[in] imageScalerList: List of image scalers
        \param[in] weightList: List of weights
        \param[in] bgInfoList: List of background data from background matching, or None
        \param[in] altMaskList: List of alternate masks to use rather than those stored with tempExp, or None
        \param[in] mask: Mask to ignore when coadding
        \param[in] supplementaryData: pipeBase.Struct with additional data products needed to assemble coadd.
                        Only used by subclasses that implement makeSupplementaryData and override assemble.
        \return pipeBase.Struct with coaddExposure, nImage if requested
        """
        tempExpName = self.getTempExpDatasetName(self.warpType)
        self.log.info("Assembling %s %s", len(tempExpRefList), tempExpName)
        if mask is None:
            mask = self.getBadPixelMask()

        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(self.config.sigmaClip)
        statsCtrl.setNumIter(self.config.clipIter)
        statsCtrl.setAndMask(mask)
        statsCtrl.setNanSafe(True)
        statsCtrl.setWeighted(True)
        statsCtrl.setCalcErrorFromInputVariance(True)
        for plane, threshold in self.config.maskPropagationThresholds.items():
            bit = afwImage.Mask.getMaskPlane(plane)
            statsCtrl.setMaskPropagationThreshold(bit, threshold)

        statsFlags = afwMath.stringToStatisticsProperty(self.config.statistic)

        if bgInfoList is None:
            bgInfoList = [None]*len(tempExpRefList)

        if altMaskList is None:
            altMaskList = [None]*len(tempExpRefList)

        coaddExposure = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
        coaddExposure.setCalib(self.scaleZeroPoint.getCalib())
        coaddExposure.getInfo().setCoaddInputs(self.inputRecorder.makeCoaddInputs())
        self.assembleMetadata(coaddExposure, tempExpRefList, weightList)
        coaddMaskedImage = coaddExposure.getMaskedImage()
        subregionSizeArr = self.config.subregionSize
        subregionSize = afwGeom.Extent2I(subregionSizeArr[0], subregionSizeArr[1])
        # if nImage is requested, create a zero one which can be passed to assembleSubregion
        if self.config.doNImage:
            nImage = afwImage.ImageU(skyInfo.bbox)
        else:
            nImage = None
        for subBBox in _subBBoxIter(skyInfo.bbox, subregionSize):
            try:
                self.assembleSubregion(coaddExposure, subBBox, tempExpRefList, imageScalerList,
                                       weightList, bgInfoList, altMaskList, statsFlags, statsCtrl,
                                       nImage=nImage)
            except Exception as e:
                self.log.fatal("Cannot compute coadd %s: %s", subBBox, e)

        coaddUtils.setCoaddEdgeBits(coaddMaskedImage.getMask(), coaddMaskedImage.getVariance())
        return pipeBase.Struct(coaddExposure=coaddExposure, nImage=nImage)

    def assembleMetadata(self, coaddExposure, tempExpRefList, weightList):
        """!
        \brief Set the metadata for the coadd

        This basic implementation simply sets the filter from the
        first input.

        \param[in] coaddExposure: The target image for the coadd
        \param[in] tempExpRefList: List of data references to tempExp
        \param[in] weightList: List of weights
        """
        assert len(tempExpRefList) == len(weightList), "Length mismatch"
        tempExpName = self.getTempExpDatasetName(self.warpType)
        # We load a single pixel of each coaddTempExp, because we just want to get at the metadata
        # (and we need more than just the PropertySet that contains the header), which is not possible
        # with the current butler (see #2777).
        tempExpList = [tempExpRef.get(tempExpName + "_sub",
                                      bbox=afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(1, 1)),
                                      imageOrigin="LOCAL", immediate=True) for tempExpRef in tempExpRefList]
        numCcds = sum(len(tempExp.getInfo().getCoaddInputs().ccds) for tempExp in tempExpList)

        coaddExposure.setFilter(tempExpList[0].getFilter())
        coaddInputs = coaddExposure.getInfo().getCoaddInputs()
        coaddInputs.ccds.reserve(numCcds)
        coaddInputs.visits.reserve(len(tempExpList))

        for tempExp, weight in zip(tempExpList, weightList):
            self.inputRecorder.addVisitToCoadd(coaddInputs, tempExp, weight)
        coaddInputs.visits.sort()
        if self.warpType == "psfMatched":
            # The modelPsf BBox for a psfMatchedWarp/coaddTempExp was dynamically defined by
            # ModelPsfMatchTask as the square box bounding its spatially-variable, pre-matched WarpedPsf.
            # Likewise, set the PSF of a PSF-Matched Coadd to the modelPsf
            # having the maximum width (sufficient because square)
            modelPsfList = [tempExp.getPsf() for tempExp in tempExpList]
            modelPsfWidthList = [modelPsf.computeBBox().getWidth() for modelPsf in modelPsfList]
            psf = modelPsfList[modelPsfWidthList.index(max(modelPsfWidthList))]
        else:
            psf = measAlg.CoaddPsf(coaddInputs.ccds, coaddExposure.getWcs(),
                                   self.config.coaddPsf.makeControl())
        coaddExposure.setPsf(psf)
        apCorrMap = measAlg.makeCoaddApCorrMap(coaddInputs.ccds, coaddExposure.getBBox(afwImage.PARENT),
                                               coaddExposure.getWcs())
        coaddExposure.getInfo().setApCorrMap(apCorrMap)

    def assembleSubregion(self, coaddExposure, bbox, tempExpRefList, imageScalerList, weightList,
                          bgInfoList, altMaskList, statsFlags, statsCtrl, nImage=None):
        """!
        \brief Assemble the coadd for a sub-region.

        For each coaddTempExp, check for (and swap in) an alternative mask if one is passed. If background
        matching is enabled, add the background and background variance from each coaddTempExp. Remove mask
        planes listed in config.removeMaskPlanes, Finally, stack the actual exposures using
        \ref afwMath.statisticsStack "statisticsStack" with the statistic specified
        by statsFlags. Typically, the statsFlag will be one of afwMath.MEAN for a mean-stack or
        afwMath.MEANCLIP for outlier rejection using an N-sigma clipped mean where N and iterations
        are specified by statsCtrl.  Assign the stacked subregion back to the coadd.

        \param[in] coaddExposure: The target image for the coadd
        \param[in] bbox: Sub-region to coadd
        \param[in] tempExpRefList: List of data reference to tempExp
        \param[in] imageScalerList: List of image scalers
        \param[in] weightList: List of weights
        \param[in] bgInfoList: List of background data from background matching
        \param[in] altMaskList: List of alternate masks to use rather than those stored with tempExp, or None
        \param[in] statsFlags: afwMath.Property object for statistic for coadd
        \param[in] statsCtrl: Statistics control object for coadd
        \param[in] nImage: optional ImageU keeps track of exposure count for each pixel
        """
        self.log.debug("Computing coadd over %s", bbox)
        tempExpName = self.getTempExpDatasetName(self.warpType)
        coaddMaskedImage = coaddExposure.getMaskedImage()
        maskedImageList = []
        if nImage is not None:
            subNImage = afwImage.ImageU(bbox.getWidth(), bbox.getHeight())
        for tempExpRef, imageScaler, bgInfo, altMask in zip(tempExpRefList, imageScalerList, bgInfoList,
                                                            altMaskList):
            exposure = tempExpRef.get(tempExpName + "_sub", bbox=bbox)
            maskedImage = exposure.getMaskedImage()
            if altMask:
                altMaskSub = altMask.Factory(altMask, bbox, afwImage.PARENT)
                maskedImage.getMask().swap(altMaskSub)
            imageScaler.scaleMaskedImage(maskedImage)

            if self.config.doMatchBackgrounds and not bgInfo.isReference:
                backgroundModel = bgInfo.backgroundModel
                backgroundImage = backgroundModel.getImage() if \
                    self.matchBackgrounds.config.usePolynomial else \
                    backgroundModel.getImageF()
                backgroundImage.setXY0(coaddMaskedImage.getXY0())
                maskedImage += backgroundImage.Factory(backgroundImage, bbox, afwImage.PARENT, False)
                var = maskedImage.getVariance()
                var += (bgInfo.fitRMS)**2
            # Add 1 for each pixel which is not excluded by the exclude mask.
            # In legacyCoadd, pixels may also be excluded by afwMath.statisticsStack.
            if nImage is not None:
                subNImage.getArray()[maskedImage.getMask().getArray() & statsCtrl.getAndMask() == 0] += 1
            if self.config.removeMaskPlanes:
                mask = maskedImage.getMask()
                for maskPlane in self.config.removeMaskPlanes:
                    try:
                        mask &= ~mask.getPlaneBitMask(maskPlane)
                    except Exception as e:
                        self.log.warn("Unable to remove mask plane %s: %s", maskPlane, e.args[0])

            maskedImageList.append(maskedImage)

        with self.timer("stack"):
            coaddSubregion = afwMath.statisticsStack(
                maskedImageList, statsFlags, statsCtrl, weightList)
        coaddMaskedImage.assign(coaddSubregion, bbox)
        if nImage is not None:
            nImage.assign(subNImage, bbox)

    def addBackgroundMatchingMetadata(self, coaddExposure, tempExpRefList, backgroundInfoList):
        """!
        \brief Add metadata from the background matching to the coadd

        \param[in] coaddExposure: Coadd
        \param[in] tempExpRefList: List of data references for temp exps to go into coadd
        \param[in] backgroundInfoList: List of background info, results from background matching
        """
        self.log.info("Adding exposure information to metadata")
        metadata = coaddExposure.getMetadata()
        metadata.addString("CTExp_SDQA1_DESCRIPTION",
                           "Background matching: Ratio of matchedMSE / diffImVar")
        for ind, (tempExpRef, backgroundInfo) in enumerate(zip(tempExpRefList, backgroundInfoList)):
            tempExpStr = '&'.join('%s=%s' % (k, v) for k, v in tempExpRef.dataId.items())
            if backgroundInfo.isReference:
                metadata.addString("ReferenceExp_ID", tempExpStr)
            else:
                metadata.addString("CTExp_ID_%d" % (ind), tempExpStr)
                metadata.addDouble("CTExp_SDQA1_%d" % (ind),
                                   backgroundInfo.matchedMSE/backgroundInfo.diffImVar)
                metadata.addDouble("CTExp_SDQA2_%d" % (ind),
                                   backgroundInfo.fitRMS)

    def readBrightObjectMasks(self, dataRef):
        """Returns None on failure"""
        try:
            return dataRef.get("brightObjectMask", immediate=True)
        except Exception as e:
            self.log.warn("Unable to read brightObjectMask for %s: %s", dataRef.dataId, e)
            return None

    def setBrightObjectMasks(self, exposure, dataId, brightObjectMasks):
        """Set the bright object masks

        exposure:          Exposure under consideration
        dataId:            Data identifier dict for patch
        brightObjectMasks: afwTable of bright objects to mask
        """
        #
        # Check the metadata specifying the tract/patch/filter
        #
        if brightObjectMasks is None:
            self.log.warn("Unable to apply bright object mask: none supplied")
            return
        self.log.info("Applying %d bright object masks to %s", len(brightObjectMasks), dataId)
        md = brightObjectMasks.table.getMetadata()
        for k in dataId:
            if not md.exists(k):
                self.log.warn("Expected to see %s in metadata", k)
            else:
                if md.get(k) != dataId[k]:
                    self.log.warn("Expected to see %s == %s in metadata, saw %s", k, md.get(k), dataId[k])

        mask = exposure.getMaskedImage().getMask()
        wcs = exposure.getWcs()
        plateScale = wcs.pixelScale().asArcseconds()

        for rec in brightObjectMasks:
            center = afwGeom.PointI(wcs.skyToPixel(rec.getCoord()))
            if rec["type"] == "box":
                assert rec["angle"] == 0.0, ("Angle != 0 for mask object %s" % rec["id"])
                width = rec["width"].asArcseconds()/plateScale    # convert to pixels
                height = rec["height"].asArcseconds()/plateScale  # convert to pixels

                halfSize = afwGeom.ExtentI(0.5*width, 0.5*height)
                bbox = afwGeom.Box2I(center - halfSize, center + halfSize)

                bbox = afwGeom.BoxI(afwGeom.PointI(int(center[0] - 0.5*width), int(center[1] - 0.5*height)),
                                    afwGeom.PointI(int(center[0] + 0.5*width), int(center[1] + 0.5*height)))
                spans = afwGeom.SpanSet(bbox)
            elif rec["type"] == "circle":
                radius = int(rec["radius"].asArcseconds()/plateScale)   # convert to pixels
                spans = afwGeom.SpanSet.fromShape(radius, offset=center)
            else:
                self.log.warn("Unexpected region type %s at %s" % rec["type"], center)
                continue
            spans.clippedTo(mask.getBBox()).setMask(mask, self.brightObjectBitmask)

    @classmethod
    def _makeArgumentParser(cls):
        """!
        \brief Create an argument parser
        """
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", cls.ConfigClass().coaddName + "Coadd_" +
                               cls.ConfigClass().warpType + "Warp",
                               help="data ID, e.g. --id tract=12345 patch=1,2",
                               ContainerClass=AssembleCoaddDataIdContainer)
        parser.add_id_argument("--selectId", "calexp", help="data ID, e.g. --selectId visit=6789 ccd=0..9",
                               ContainerClass=SelectDataIdContainer)
        return parser


def _subBBoxIter(bbox, subregionSize):
    """!
    \brief Iterate over subregions of a bbox

    \param[in] bbox: bounding box over which to iterate: afwGeom.Box2I
    \param[in] subregionSize: size of sub-bboxes

    \return subBBox: next sub-bounding box of size subregionSize or smaller;
        each subBBox is contained within bbox, so it may be smaller than subregionSize at the edges of bbox,
        but it will never be empty
    """
    if bbox.isEmpty():
        raise RuntimeError("bbox %s is empty" % (bbox,))
    if subregionSize[0] < 1 or subregionSize[1] < 1:
        raise RuntimeError("subregionSize %s must be nonzero" % (subregionSize,))

    for rowShift in range(0, bbox.getHeight(), subregionSize[1]):
        for colShift in range(0, bbox.getWidth(), subregionSize[0]):
            subBBox = afwGeom.Box2I(bbox.getMin() + afwGeom.Extent2I(colShift, rowShift), subregionSize)
            subBBox.clip(bbox)
            if subBBox.isEmpty():
                raise RuntimeError("Bug: empty bbox! bbox=%s, subregionSize=%s, colShift=%s, rowShift=%s" %
                                   (bbox, subregionSize, colShift, rowShift))
            yield subBBox


class AssembleCoaddDataIdContainer(pipeBase.DataIdContainer):
    """!
    \brief A version of lsst.pipe.base.DataIdContainer specialized for assembleCoadd.
    """

    def makeDataRefList(self, namespace):
        """!
           \brief Make self.refList from self.idList.

           Interpret the config.doMatchBackgrounds, config.autoReference,
           and whether a visit/run supplied.
           If a visit/run is supplied, config.autoReference is automatically set to False.
           if config.doMatchBackgrounds == false, then a visit/run will be ignored if accidentally supplied.

        """
        keysCoadd = namespace.butler.getKeys(datasetType=namespace.config.coaddName + "Coadd",
                                             level=self.level)
        keysCoaddTempExp = namespace.butler.getKeys(datasetType=namespace.config.coaddName +
                                                    "Coadd_directWarp", level=self.level)

        if namespace.config.doMatchBackgrounds:
            if namespace.config.autoReference:  # matcher will pick it's own reference image
                datasetType = namespace.config.coaddName + "Coadd"
                validKeys = keysCoadd
            else:
                datasetType = namespace.config.coaddName + "Coadd_directWarp"
                validKeys = keysCoaddTempExp
        else:  # bkg subtracted coadd
            datasetType = namespace.config.coaddName + "Coadd"
            validKeys = keysCoadd

        for dataId in self.idList:
            # tract and patch are required
            for key in validKeys:
                if key not in dataId:
                    raise RuntimeError("--id must include " + key)

            for key in dataId:  # check if users supplied visit/run
                if (key not in keysCoadd) and (key in keysCoaddTempExp):  # user supplied a visit/run
                    if namespace.config.autoReference:
                        # user probably meant: autoReference = False
                        namespace.config.autoReference = False
                        datasetType = namespace.config.coaddName + "Coadd_directWarp"
                        print("Switching config.autoReference to False; applies only to background Matching.")
                        break

            dataRef = namespace.butler.dataRef(
                datasetType=datasetType,
                dataId=dataId,
            )
            self.refList.append(dataRef)


def countMaskFromFootprint(mask, footprint, bitmask, ignoreMask):
    """!
    \brief Function to count the number of pixels with a specific mask in a footprint.

    Find the intersection of mask & footprint. Count all pixels in the mask that are in the intersection that
    have bitmask set but do not have ignoreMask set. Return the count.

    \param[in] mask: mask to define intersection region by.
    \parma[in] footprint: footprint to define the intersection region by.
    \param[in] bitmask: specific mask that we wish to count the number of occurances of.
    \param[in] ignoreMask: pixels to not consider.
    \return count of number of pixels in footprint with specified mask.
    """
    bbox = footprint.getBBox()
    bbox.clip(mask.getBBox(afwImage.PARENT))
    fp = afwImage.Mask(bbox)
    subMask = mask.Factory(mask, bbox, afwImage.PARENT)
    footprint.spans.setMask(fp, bitmask)
    return numpy.logical_and((subMask.getArray() & fp.getArray()) > 0,
                             (subMask.getArray() & ignoreMask) == 0).sum()


class SafeClipAssembleCoaddConfig(AssembleCoaddConfig):
    """!
\anchor SafeClipAssembleCoaddConfig

\brief Configuration parameters for the SafeClipAssembleCoaddTask
    """
    clipDetection = pexConfig.ConfigurableField(
        target=SourceDetectionTask,
        doc="Detect sources on difference between unclipped and clipped coadd")
    minClipFootOverlap = pexConfig.Field(
        doc="Minimum fractional overlap of clipped footprint with visit DETECTED to be clipped",
        dtype=float,
        default=0.6
    )
    minClipFootOverlapSingle = pexConfig.Field(
        doc="Minimum fractional overlap of clipped footprint with visit DETECTED to be "
        "clipped when only one visit overlaps",
        dtype=float,
        default=0.5
    )
    minClipFootOverlapDouble = pexConfig.Field(
        doc="Minimum fractional overlap of clipped footprints with visit DETECTED to be "
        "clipped when two visits overlap",
        dtype=float,
        default=0.45
    )
    maxClipFootOverlapDouble = pexConfig.Field(
        doc="Maximum fractional overlap of clipped footprints with visit DETECTED when "
        "considering two visits",
        dtype=float,
        default=0.15
    )
    minBigOverlap = pexConfig.Field(
        doc="Minimum number of pixels in footprint to use DETECTED mask from the single visits "
        "when labeling clipped footprints",
        dtype=int,
        default=100
    )

    def setDefaults(self):
        # The numeric values for these configuration parameters were empirically determined, future work
        # may further refine them.
        AssembleCoaddConfig.setDefaults(self)
        self.clipDetection.doTempLocalBackground = False
        self.clipDetection.reEstimateBackground = False
        self.clipDetection.returnOriginalFootprints = False
        self.clipDetection.thresholdPolarity = "both"
        self.clipDetection.thresholdValue = 2
        self.clipDetection.nSigmaToGrow = 2
        self.clipDetection.minPixels = 4
        self.clipDetection.isotropicGrow = True
        self.clipDetection.thresholdType = "pixel_stdev"
        self.sigmaClip = 1.5
        self.clipIter = 3
        self.statistic = "MEAN"

    def validate(self):
        if self.doSigmaClip:
            log.warn("Additional Sigma-clipping not allowed in Safe-clipped Coadds. "
                     "Ignoring doSigmaClip.")
            self.doSigmaClip = False
        if self.statistic != "MEAN":
            raise ValueError("Only MEAN statistic allowed for final stacking in SafeClipAssembleCoadd "
                             "(%s chosen). Please set statistic to MEAN."
                             % (self.statistic))
        AssembleCoaddTask.ConfigClass.validate(self)


## \addtogroup LSST_task_documentation
## \{
## \page SafeClipAssembleCoaddTask
## \ref SafeClipAssembleCoaddTask_ "SafeClipAssembleCoaddTask"
## \copybrief SafeClipAssembleCoaddTask
## \}


class SafeClipAssembleCoaddTask(AssembleCoaddTask):
    """!
    \anchor SafeClipAssembleCoaddTask_

    \brief Assemble a coadded image from a set of coadded temporary exposures,
    being careful to clip & flag areas with potential artifacts.

    \section pipe_tasks_assembleCoadd_Contents Contents
      - \ref pipe_tasks_assembleCoadd_SafeClipAssembleCoaddTask_Purpose
      - \ref pipe_tasks_assembleCoadd_SafeClipAssembleCoaddTask_Initialize
      - \ref pipe_tasks_assembleCoadd_SafeClipAssembleCoaddTask_Run
      - \ref pipe_tasks_assembleCoadd_SafeClipAssembleCoaddTask_Config
      - \ref pipe_tasks_assembleCoadd_SafeClipAssembleCoaddTask_Debug
      - \ref pipe_tasks_assembleCoadd_SafeClipAssembleCoaddTask_Example

    \section pipe_tasks_assembleCoadd_SafeClipAssembleCoaddTask_Purpose	Description

    \copybrief SafeClipAssembleCoaddTask

    Read the documentation for \ref AssembleCoaddTask_ "AssembleCoaddTask" first since
    SafeClipAssembleCoaddTask subtasks that task.
    In \ref AssembleCoaddTask_ "AssembleCoaddTask", we compute the coadd as an clipped mean (i.e. we clip
    outliers).
    The problem with doing this is that when computing the coadd PSF at a given location, individual visit
    PSFs from visits with outlier pixels contribute to the coadd PSF and cannot be treated correctly.
    In this task, we correct for this behavior by creating a new badMaskPlane 'CLIPPED'.
    We populate this plane on the input coaddTempExps and the final coadd where i. difference imaging suggests
    that there is an outlier and ii. this outlier appears on only one or two images.
    Such regions will not contribute to the final coadd.
    Furthermore, any routine to determine the coadd PSF can now be cognizant of clipped regions.
    Note that the algorithm implemented by this task is preliminary and works correctly for HSC data.
    Parameter modifications and or considerable redesigning of the algorithm is likley required for other
    surveys.

    SafeClipAssembleCoaddTask uses a \ref SourceDetectionTask_ "clipDetection" subtask and also sub-classes
    \ref AssembleCoaddTask_ "AssembleCoaddTask". You can retarget the
    \ref SourceDetectionTask_ "clipDetection" subtask if you wish.

    \section pipe_tasks_assembleCoadd_SafeClipAssembleCoaddTask_Initialize       Task initialization
    \copydoc \_\_init\_\_

    \section pipe_tasks_assembleCoadd_SafeClipAssembleCoaddTask_Run       Invoking the Task
    \copydoc run

    \section pipe_tasks_assembleCoadd_SafeClipAssembleCoaddTask_Config       Configuration parameters
    See \ref SafeClipAssembleCoaddConfig

    \section pipe_tasks_assembleCoadd_SafeClipAssembleCoaddTask_Debug		Debug variables
    The \link lsst.pipe.base.cmdLineTask.CmdLineTask command line task\endlink interface supports a
    flag \c -d to import \b debug.py from your \c PYTHONPATH; see \ref baseDebug for more about \b debug.py
    files.
    SafeClipAssembleCoaddTask has no debug variables of its own. The \ref SourceDetectionTask_ "clipDetection"
    subtasks may support debug variables. See the documetation for \ref SourceDetectionTask_ "clipDetection"
    for further information.

    \section pipe_tasks_assembleCoadd_SafeClipAssembleCoaddTask_Example	A complete example of using
    SafeClipAssembleCoaddTask

    SafeClipAssembleCoaddTask assembles a set of warped coaddTempExp images into a coadded image.
    The SafeClipAssembleCoaddTask is invoked by running assembleCoadd.py <em>without</em> the flag
    '--legacyCoadd'.
    Usage of assembleCoadd.py expects a data reference to the tract patch and filter to be coadded
    (specified using '--id = [KEY=VALUE1[^VALUE2[^VALUE3...] [KEY=VALUE1[^VALUE2[^VALUE3...] ...]]') along
    with a list of coaddTempExps to attempt to coadd (specified using
    '--selectId [KEY=VALUE1[^VALUE2[^VALUE3...] [KEY=VALUE1[^VALUE2[^VALUE3...] ...]]').
    Only the coaddTempExps that cover the specified tract and patch will be coadded.
    A list of the available optional arguments can be obtained by calling assembleCoadd.py with the --help
    command line argument:
    \code
    assembleCoadd.py --help
    \endcode
    To demonstrate usage of the SafeClipAssembleCoaddTask in the larger context of multi-band processing, we
    will generate the HSC-I & -R band coadds from HSC engineering test data provided in the ci_hsc package. To
    begin, assuming that the lsst stack has been already set up, we must set up the obs_subaru and ci_hsc
    packages.
    This defines the environment variable $CI_HSC_DIR and points at the location of the package. The raw HSC
    data live in the $CI_HSC_DIR/raw directory. To begin assembling the coadds, we must first
    <DL>
      <DT>processCcd</DT>
      <DD> process the individual ccds in $CI_HSC_RAW to produce calibrated exposures</DD>
      <DT>makeSkyMap</DT>
      <DD> create a skymap that covers the area of the sky present in the raw exposures</DD>
      <DT>makeCoaddTempExp</DT>
      <DD> warp the individual calibrated exposures to the tangent plane of the coadd</DD>
    </DL>
    We can perform all of these steps by running
    \code
    $CI_HSC_DIR scons warp-903986 warp-904014 warp-903990 warp-904010 warp-903988
    \endcode
    This will produce warped coaddTempExps for each visit. To coadd the warped data, we call assembleCoadd.py
    as follows:
    \code
    assembleCoadd.py $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-I \
    --selectId visit=903986 ccd=16 --selectId visit=903986 ccd=22 --selectId visit=903986 ccd=23 \
    --selectId visit=903986 ccd=100--selectId visit=904014 ccd=1 --selectId visit=904014 ccd=6 \
    --selectId visit=904014 ccd=12 --selectId visit=903990 ccd=18 --selectId visit=903990 ccd=25 \
    --selectId visit=904010 ccd=4 --selectId visit=904010 ccd=10 --selectId visit=904010 ccd=100 \
    --selectId visit=903988 ccd=16 --selectId visit=903988 ccd=17 --selectId visit=903988 ccd=23 \
    --selectId visit=903988 ccd=24
    \endcode
    This will process the HSC-I band data. The results are written in
    `$CI_HSC_DIR/DATA/deepCoadd-results/HSC-I`.

    You may also choose to run:
    \code
    scons warp-903334 warp-903336 warp-903338 warp-903342 warp-903344 warp-903346
    assembleCoadd.py $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-R --selectId visit=903334 ccd=16 \
    --selectId visit=903334 ccd=22 --selectId visit=903334 ccd=23 --selectId visit=903334 ccd=100 \
    --selectId visit=903336 ccd=17 --selectId visit=903336 ccd=24 --selectId visit=903338 ccd=18 \
    --selectId visit=903338 ccd=25 --selectId visit=903342 ccd=4 --selectId visit=903342 ccd=10 \
    --selectId visit=903342 ccd=100 --selectId visit=903344 ccd=0 --selectId visit=903344 ccd=5 \
    --selectId visit=903344 ccd=11 --selectId visit=903346 ccd=1 --selectId visit=903346 ccd=6 \
    --selectId visit=903346 ccd=12
    \endcode
    to generate the coadd for the HSC-R band if you are interested in following multiBand Coadd processing as
    discussed in \ref pipeTasks_multiBand.
    """
    ConfigClass = SafeClipAssembleCoaddConfig
    _DefaultName = "safeClipAssembleCoadd"

    def __init__(self, *args, **kwargs):
        """!
        \brief Initialize the task and make the \ref SourceDetectionTask_ "clipDetection" subtask.
        """
        AssembleCoaddTask.__init__(self, *args, **kwargs)
        schema = afwTable.SourceTable.makeMinimalSchema()
        self.makeSubtask("clipDetection", schema=schema)

    def assemble(self, skyInfo, tempExpRefList, imageScalerList, weightList, bgModelList,
                 *args, **kwargs):
        """!
        \brief Assemble the coadd for a region

        Compute the difference of coadds created with and without outlier rejection to identify coadd pixels
        that have outlier values in some individual visits. Detect clipped regions on the difference image and
        mark these regions on the one or two individual coaddTempExps where they occur if there is significant
        overlap between the clipped region and a source.
        This leaves us with a set of footprints from the difference image that have been identified as having
        occured on just one or two individual visits. However, these footprints were generated from a
        difference image. It is conceivable for a large diffuse source to have become broken up into multiple
        footprints acrosss the coadd difference in this process.
        Determine the clipped region from all overlapping footprints from the detected sources in each visit -
        these are big footprints.
        Combine the small and big clipped footprints and mark them on a new bad mask plane
        Generate the coadd using \ref AssembleCoaddTask.assemble_ "AssembleCoaddTask.assemble" without outlier
        removal. Clipped footprints will no longer make it into the coadd because they are marked in the new
        bad mask plane.

        N.b. *args and **kwargs are passed but ignored in order to match the call signature expected by the
        parent task.

        @param skyInfo: Patch geometry information, from getSkyInfo
        @param tempExpRefList: List of data reference to tempExp
        @param imageScalerList: List of image scalers
        @param weightList: List of weights
        @param bgModelList: List of background models from background matching
        return pipeBase.Struct with coaddExposure, nImage
        """
        exp = self.buildDifferenceImage(skyInfo, tempExpRefList, imageScalerList, weightList, bgModelList)
        mask = exp.getMaskedImage().getMask()
        mask.addMaskPlane("CLIPPED")

        result = self.detectClip(exp, tempExpRefList)

        self.log.info('Found %d clipped objects', len(result.clipFootprints))

        # Go to individual visits for big footprints
        maskClipValue = mask.getPlaneBitMask("CLIPPED")
        maskDetValue = mask.getPlaneBitMask("DETECTED") | mask.getPlaneBitMask("DETECTED_NEGATIVE")
        bigFootprints = self.detectClipBig(result.tempExpClipList, result.clipFootprints, result.clipIndices,
                                           maskClipValue, maskDetValue)

        # Create mask of the current clipped footprints
        maskClip = mask.Factory(mask.getBBox(afwImage.PARENT))
        afwDet.setMaskFromFootprintList(maskClip, result.clipFootprints, maskClipValue)

        maskClipBig = maskClip.Factory(mask.getBBox(afwImage.PARENT))
        afwDet.setMaskFromFootprintList(maskClipBig, bigFootprints, maskClipValue)
        maskClip |= maskClipBig

        # Assemble coadd from base class, but ignoring CLIPPED pixels
        badMaskPlanes = self.config.badMaskPlanes[:]
        badMaskPlanes.append("CLIPPED")
        badPixelMask = afwImage.Mask.getPlaneBitMask(badMaskPlanes)
        retStruct = AssembleCoaddTask.assemble(self, skyInfo, tempExpRefList, imageScalerList, weightList,
                                               bgModelList, result.tempExpClipList, mask=badPixelMask)

        # Set the coadd CLIPPED mask from the footprints since currently pixels that are masked
        # do not get propagated. (Remove with DM-9953)
        maskExp = retStruct.coaddExposure.getMaskedImage().getMask()
        maskExp |= maskClip

        return retStruct

    def buildDifferenceImage(self, skyInfo, tempExpRefList, imageScalerList, weightList, bgModelList):
        """!
        \brief Return an exposure that contains the difference between and unclipped and clipped coadds.

        Generate a difference image between clipped and unclipped coadds.
        Compute the difference image by subtracting an outlier-clipped coadd from an outlier-unclipped coadd.
        Return the difference image.

        @param skyInfo: Patch geometry information, from getSkyInfo
        @param tempExpRefList: List of data reference to tempExp
        @param imageScalerList: List of image scalers
        @param weightList: List of weights
        @param bgModelList: List of background models from background matching
        @return Difference image of unclipped and clipped coadd wrapped in an Exposure
        """
        # Clone and upcast self.config because current self.config is frozen
        config = AssembleCoaddConfig()
        # getattr necessary because subtasks do not survive Config.toDict()
        configIntersection = {k: getattr(self.config, k)
                              for k, v in self.config.toDict().items() if (k in config.keys())}
        config.update(**configIntersection)

        # statistic MEAN copied from self.config.statistic, but for clarity explicitly assign
        config.statistic = 'MEAN'
        task = AssembleCoaddTask(config=config)
        coaddMean = task.assemble(skyInfo, tempExpRefList, imageScalerList, weightList,
                                  bgModelList).coaddExposure

        config.statistic = 'MEANCLIP'
        task = AssembleCoaddTask(config=config)
        coaddClip = task.assemble(skyInfo, tempExpRefList, imageScalerList, weightList,
                                  bgModelList).coaddExposure

        coaddDiff = coaddMean.getMaskedImage().Factory(coaddMean.getMaskedImage())
        coaddDiff -= coaddClip.getMaskedImage()
        exp = afwImage.ExposureF(coaddDiff)
        exp.setPsf(coaddMean.getPsf())
        return exp

    def detectClip(self, exp, tempExpRefList):
        """!
        \brief Detect clipped regions on an exposure and set the mask on the individual tempExp masks

        Detect footprints in the difference image after smoothing the difference image with a Gaussian kernal.
        Identify footprints that overlap with one or two input coaddTempExps by comparing the computed overlap
        fraction to thresholds set in the config.
        A different threshold is applied depending on the number of overlapping visits (restricted to one or
        two).
        If the overlap exceeds the thresholds, the footprint is considered "CLIPPED" and is marked as such on
        the coaddTempExp.
        Return a struct with the clipped footprints, the indices of the coaddTempExps that end up overlapping
        with the clipped footprints and a list of new masks for the coaddTempExps.

        \param[in] exp: Exposure to run detection on
        \param[in] tempExpRefList: List of data reference to tempExp
        \return struct containing:
        - clippedFootprints: list of clipped footprints
        - clippedIndices: indices for each clippedFootprint in tempExpRefList
        - tempExpClipList: list of new masks for tempExp
        """
        mask = exp.getMaskedImage().getMask()
        maskClipValue = mask.getPlaneBitMask("CLIPPED")
        maskDetValue = mask.getPlaneBitMask("DETECTED") | mask.getPlaneBitMask("DETECTED_NEGATIVE")
        fpSet = self.clipDetection.detectFootprints(exp, doSmooth=True, clearMask=True)
        # Merge positive and negative together footprints together
        fpSet.positive.merge(fpSet.negative)
        footprints = fpSet.positive
        self.log.info('Found %d potential clipped objects', len(footprints.getFootprints()))
        ignoreMask = self.getBadPixelMask()

        clipFootprints = []
        clipIndices = []

        # build a list with a mask for each visit which can be modified with clipping information
        tempExpClipList = [tmpExpRef.get(self.getTempExpDatasetName(self.warpType),
                                         immediate=True).getMaskedImage().getMask() for
                           tmpExpRef in tempExpRefList]

        for footprint in footprints.getFootprints():
            nPixel = footprint.getArea()
            overlap = []  # hold the overlap with each visit
            maskList = []  # which visit mask match
            indexList = []  # index of visit in global list
            for i, tmpExpMask in enumerate(tempExpClipList):
                # Determine the overlap with the footprint
                ignore = countMaskFromFootprint(tmpExpMask, footprint, ignoreMask, 0x0)
                overlapDet = countMaskFromFootprint(tmpExpMask, footprint, maskDetValue, ignoreMask)
                totPixel = nPixel - ignore

                # If we have more bad pixels than detection skip
                if ignore > overlapDet or totPixel <= 0.5*nPixel or overlapDet == 0:
                    continue
                overlap.append(overlapDet/float(totPixel))
                maskList.append(tmpExpMask)
                indexList.append(i)

            overlap = numpy.array(overlap)
            if not len(overlap):
                continue

            keep = False   # Should this footprint be marked as clipped?
            keepIndex = []  # Which tempExps does the clipped footprint belong to

            # If footprint only has one overlap use a lower threshold
            if len(overlap) == 1:
                if overlap[0] > self.config.minClipFootOverlapSingle:
                    keep = True
                    keepIndex = [0]
            else:
                # This is the general case where only visit should be clipped
                clipIndex = numpy.where(overlap > self.config.minClipFootOverlap)[0]
                if len(clipIndex) == 1:
                    keep = True
                    keepIndex = [clipIndex[0]]

                # Test if there are clipped objects that overlap two different visits
                clipIndex = numpy.where(overlap > self.config.minClipFootOverlapDouble)[0]
                if len(clipIndex) == 2 and len(overlap) > 3:
                    clipIndexComp = numpy.where(overlap <= self.config.minClipFootOverlapDouble)[0]
                    if numpy.max(overlap[clipIndexComp]) <= self.config.maxClipFootOverlapDouble:
                        keep = True
                        keepIndex = clipIndex

            if not keep:
                continue

            for index in keepIndex:
                footprint.spans.setMask(maskList[index], maskClipValue)

            clipIndices.append(numpy.array(indexList)[keepIndex])
            clipFootprints.append(footprint)

        return pipeBase.Struct(clipFootprints=clipFootprints, clipIndices=clipIndices,
                               tempExpClipList=tempExpClipList)

    def detectClipBig(self, tempExpClipList, clipFootprints, clipIndices, maskClipValue, maskDetValue):
        """!
        \brief Find footprints from individual tempExp footprints for large footprints.

        Identify big footprints composed of many sources in the coadd difference that may have originated in a
        large diffuse source in the coadd. We do this by indentifying all clipped footprints that overlap
        significantly with each source in all the coaddTempExps.

        \param[in] tempExpClipList: List of tempExp masks with clipping information
        \param[in] clipFootprints: List of clipped footprints
        \param[in] clipIndices: List of which entries in tempExpClipList each footprint belongs to
        \param[in] maskClipValue: Mask value of clipped pixels
        \param[in] maskClipValue: Mask value of detected pixels
        \return list of big footprints
        """
        bigFootprintsCoadd = []
        ignoreMask = self.getBadPixelMask()
        for index, tmpExpMask in enumerate(tempExpClipList):

            # Create list of footprints from the DETECTED pixels
            maskVisitDet = tmpExpMask.Factory(tmpExpMask, tmpExpMask.getBBox(afwImage.PARENT),
                                              afwImage.PARENT, True)
            maskVisitDet &= maskDetValue
            visitFootprints = afwDet.FootprintSet(maskVisitDet, afwDet.Threshold(1))

            # build a mask of clipped footprints that are in this visit
            clippedFootprintsVisit = []
            for foot, clipIndex in zip(clipFootprints, clipIndices):
                if index not in clipIndex:
                    continue
                clippedFootprintsVisit.append(foot)
            maskVisitClip = maskVisitDet.Factory(maskVisitDet.getBBox(afwImage.PARENT))
            afwDet.setMaskFromFootprintList(maskVisitClip, clippedFootprintsVisit, maskClipValue)

            bigFootprintsVisit = []
            for foot in visitFootprints.getFootprints():
                if foot.getArea() < self.config.minBigOverlap:
                    continue
                nCount = countMaskFromFootprint(maskVisitClip, foot, maskClipValue, ignoreMask)
                if nCount > self.config.minBigOverlap:
                    bigFootprintsVisit.append(foot)
                    bigFootprintsCoadd.append(foot)

            # Update single visit masks
            maskVisitClip.clearAllMaskPlanes()
            afwDet.setMaskFromFootprintList(maskVisitClip, bigFootprintsVisit, maskClipValue)
            tmpExpMask |= maskVisitClip

        return bigFootprintsCoadd


class CompareWarpAssembleCoaddConfig(AssembleCoaddConfig):
    assembleStaticSkyModel = pexConfig.ConfigurableField(
        target=AssembleCoaddTask,
        doc="Task to assemble an artifact-free, PSF-matched Coadd to serve as a"
            " naive/first-iteration model of the static sky.",
    )
    detect = pexConfig.ConfigurableField(
        target=SourceDetectionTask,
        doc="Detect outlier sources on difference between each psfMatched warp and static sky model"
    )
    maxNumEpochs = pexConfig.Field(
        doc="Maximum number of epochs/visits in which an artifact candidate can appear and still be masked. "
            "For each footprint detected on the image difference between the psfMatched warp and static sky "
            "model, if a significant fraction of pixels (defined by spatialThreshold) are residuals in more "
            "than maxNumEpochs, the artifact candidate is persistant rather than transient and not masked.",
        dtype=int,
        default=2
    )
    spatialThreshold = pexConfig.Field(
        doc="Unitless fraction of pixels defining how much of the outlier region has to meet the "
            "temporal criteria",
        dtype=float,
        default=0.5
    )

    def setDefaults(self):
        AssembleCoaddConfig.setDefaults(self)
        self.assembleStaticSkyModel.warpType = 'psfMatched'
        self.assembleStaticSkyModel.statistic = 'MEDIAN'
        self.assembleStaticSkyModel.doWrite = False
        self.statistic = 'MEAN'
        self.detect.doTempLocalBackground = False
        self.detect.reEstimateBackground = False
        self.detect.returnOriginalFootprints = False
        self.detect.thresholdPolarity = "both"
        self.detect.thresholdValue = 5
        self.detect.nSigmaToGrow = 2
        self.detect.minPixels = 4
        self.detect.isotropicGrow = True
        self.detect.thresholdType = "pixel_stdev"


## \addtogroup LSST_task_documentation
## \{
## \page CompareWarpAssembleCoaddTask
## \ref CompareWarpAssembleCoaddTask_ "CompareWarpAssembleCoaddTask"
## \copybrief CompareWarpAssembleCoaddTask
## \}

class CompareWarpAssembleCoaddTask(AssembleCoaddTask):
    """!
    \anchor CompareWarpAssembleCoaddTask_

    \brief Assemble a compareWarp coadded image from a set of warps
    by masking artifacts detected by comparing PSF-matched warps

    \section pipe_tasks_assembleCoadd_Contents Contents
      - \ref pipe_tasks_assembleCoadd_CompareWarpAssembleCoaddTask_Purpose
      - \ref pipe_tasks_assembleCoadd_CompareWarpAssembleCoaddTask_Initialize
      - \ref pipe_tasks_assembleCoadd_CompareWarpAssembleCoaddTask_Run
      - \ref pipe_tasks_assembleCoadd_CompareWarpAssembleCoaddTask_Config
      - \ref pipe_tasks_assembleCoadd_CompareWarpAssembleCoaddTask_Debug
      - \ref pipe_tasks_assembleCoadd_CompareWarpAssembleCoaddTask_Example

    \section pipe_tasks_assembleCoadd_CompareWarpAssembleCoaddTask_Purpose Description

    \copybrief CompareWarpAssembleCoaddTask

    In \ref AssembleCoaddTask_ "AssembleCoaddTask", we compute the coadd as an clipped mean (i.e. we clip
    outliers).
    The problem with doing this is that when computing the coadd PSF at a given location, individual visit
    PSFs from visits with outlier pixels contribute to the coadd PSF and cannot be treated correctly.
    In this task, we correct for this behavior by creating a new badMaskPlane 'CLIPPED' which marks
    pixels in the individual warps suspected to contain an artifact.
    We populate this plane on the input warps by comparing PSF-matched warps with a PSF-matched median coadd
    which serves as a model of the static sky. Any group of pixels that deviates from the PSF-matched
    template coadd by more than config.detect.threshold sigma, is an artifact candidate.
    The candidates are then filtered to remove variable sources and sources that are difficult to subtract
    such as bright stars.
    This filter is configured using the config parameters temporalThreshold and spatialThreshold.
    The temporalThreshold is the maximum fraction of epochs that the deviation can
    appear in and still be considered an artifact. The spatialThreshold is the maximum fraction of pixels in
    the footprint of the deviation that appear in other epochs (where other epochs is defined by the
    temporalThreshold). If the deviant region meets this criteria of having a significant percentage of pixels
    that deviate in only a few epochs, these pixels have the 'CLIPPED' bit set in the mask.
    These regions will not contribute to the final coadd.
    Furthermore, any routine to determine the coadd PSF can now be cognizant of clipped regions.
    Note that the algorithm implemented by this task is preliminary and works correctly for HSC data.
    Parameter modifications and or considerable redesigning of the algorithm is likley required for other
    surveys.

    CompareWarpAssembleCoaddTask sub-classes
    \ref AssembleCoaddTask_ "AssembleCoaddTask" and instantiates  \ref AssembleCoaddTask_ "AssembleCoaddTask"
    as a subtask to generate the TemplateCoadd (the model of the static sky)

    \section pipe_tasks_assembleCoadd_CompareWarpAssembleCoaddTask_Initialize       Task initialization
    \copydoc \_\_init\_\_

    \section pipe_tasks_assembleCoadd_CompareWarpAssembleCoaddTask_Run       Invoking the Task
    \copydoc run

    \section pipe_tasks_assembleCoadd_CompareWarpAssembleCoaddTask_Config       Configuration parameters
    See \ref CompareWarpAssembleCoaddConfig

    \section pipe_tasks_assembleCoadd_CompareWarpAssembleCoaddTask_Debug       Debug variables
    The \link lsst.pipe.base.cmdLineTask.CmdLineTask command line task\endlink interface supports a
    flag \c -d to import \b debug.py from your \c PYTHONPATH; see \ref baseDebug for more about \b debug.py
    files.

    This task supports the following debug variables:
    <dl>
        <dt>`saveCountIm`
        <dd> If True then save the Epoch Count Image as a fits file in the `figPath`
        <dt> `saveAltMask`
        <dd> If True then save the new masks with CLIPPED planes as fits files to the `figPath`
        <dt> `figPath`
        <dd> Path to save the debug fits images and figures
    </dl>

    For example, put something like:
    @code{.py}
        import lsstDebug
        def DebugInfo(name):
            di = lsstDebug.getInfo(name)
            if name == "lsst.pipe.tasks.assembleCoadd":
                di.saveCountIm = True
                di.saveAltMask = True
                di.figPath = "/desired/path/to/debugging/output/images"
            return di
        lsstDebug.Info = DebugInfo
    @endcode
    into your `debug.py` file and run `assemebleCoadd.py` with the `--debug`
    flag.
    Some subtasks may have their own debug variables; see individual Task
    documentation

    \section pipe_tasks_assembleCoadd_CompareWarpAssembleCoaddTask_Example A complete example of using
    CompareWarpAssembleCoaddTask

    CompareWarpAssembleCoaddTask assembles a set of warped images into a coadded image.
    The CompareWarpAssembleCoaddTask is invoked by running assembleCoadd.py with the flag
    '--compareWarpCoadd'.
    Usage of assembleCoadd.py expects a data reference to the tract patch and filter to be coadded
    (specified using '--id = [KEY=VALUE1[^VALUE2[^VALUE3...] [KEY=VALUE1[^VALUE2[^VALUE3...] ...]]') along
    with a list of coaddTempExps to attempt to coadd (specified using
    '--selectId [KEY=VALUE1[^VALUE2[^VALUE3...] [KEY=VALUE1[^VALUE2[^VALUE3...] ...]]').
    Only the warps that cover the specified tract and patch will be coadded.
    A list of the available optional arguments can be obtained by calling assembleCoadd.py with the --help
    command line argument:
    \code
    assembleCoadd.py --help
    \endcode
    To demonstrate usage of the CompareWarpAssembleCoaddTask in the larger context of multi-band processing,
    we will generate the HSC-I & -R band coadds from HSC engineering test data provided in the ci_hsc package.
    To begin, assuming that the lsst stack has been already set up, we must set up the obs_subaru and ci_hsc
    packages.
    This defines the environment variable $CI_HSC_DIR and points at the location of the package. The raw HSC
    data live in the $CI_HSC_DIR/raw directory. To begin assembling the coadds, we must first
    <DL>
      <DT>processCcd</DT>
      <DD> process the individual ccds in $CI_HSC_RAW to produce calibrated exposures</DD>
      <DT>makeSkyMap</DT>
      <DD> create a skymap that covers the area of the sky present in the raw exposures</DD>
      <DT>makeCoaddTempExp</DT>
      <DD> warp the individual calibrated exposures to the tangent plane of the coadd</DD>
    </DL>
    We can perform all of these steps by running
    \code
    $CI_HSC_DIR scons warp-903986 warp-904014 warp-903990 warp-904010 warp-903988
    \endcode
    This will produce warped coaddTempExps for each visit. To coadd the warped data, we call assembleCoadd.py
    as follows:
    \code
    assembleCoadd.py --compareWarpCoadd $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-I \
    --selectId visit=903986 ccd=16 --selectId visit=903986 ccd=22 --selectId visit=903986 ccd=23 \
    --selectId visit=903986 ccd=100 --selectId visit=904014 ccd=1 --selectId visit=904014 ccd=6 \
    --selectId visit=904014 ccd=12 --selectId visit=903990 ccd=18 --selectId visit=903990 ccd=25 \
    --selectId visit=904010 ccd=4 --selectId visit=904010 ccd=10 --selectId visit=904010 ccd=100 \
    --selectId visit=903988 ccd=16 --selectId visit=903988 ccd=17 --selectId visit=903988 ccd=23 \
    --selectId visit=903988 ccd=24
    \endcode
    This will process the HSC-I band data. The results are written in
    `$CI_HSC_DIR/DATA/deepCoadd-results/HSC-I`.
    """
    ConfigClass = CompareWarpAssembleCoaddConfig
    _DefaultName = "compareWarpAssembleCoadd"

    def __init__(self, *args, **kwargs):
        """!
        \brief Initialize the task and make the \ref AssembleCoadd_ "assembleStaticSkyModel" subtask.
        """
        AssembleCoaddTask.__init__(self, *args, **kwargs)
        self.makeSubtask("assembleStaticSkyModel")
        detectionSchema = afwTable.SourceTable.makeMinimalSchema()
        self.makeSubtask("detect", schema=detectionSchema)

    def makeSupplementaryData(self, dataRef, selectDataList):
        """!
        \brief Make inputs specific to Subclass

        Generate a templateCoadd to use as a native model of static sky to subtract from warps.
        """
        templateCoadd = self.assembleStaticSkyModel.run(dataRef, selectDataList)

        if templateCoadd is None:
            warpName = (self.assembleStaticSkyModel.warpType[0].upper() +
                        self.assembleStaticSkyModel.warpType[1:])
            message = """No %(warpName)s warps were found to build the template coadd which is
              required to run CompareWarpAssembleCoaddTask. To continue assembling this type of coadd,
              first either rerun makeCoaddTempExp with config.make%(warpName)s=True or
              coaddDriver with config.makeCoadTempExp.make%(warpName)s=True, before assembleCoadd.

              Alternatively, to use another algorithm with existing warps, retarget the CoaddDriverConfig to
              another algorithm like:

                from lsst.pipe.tasks.assembleCoadd import SafeClipAssembleCoaddTask
                config.assemble.retarget(SafeClipAssembleCoaddTask)
            """ % {"warpName": warpName}
            raise RuntimeError(message)

        return pipeBase.Struct(templateCoadd=templateCoadd.coaddExposure)

    def assemble(self, skyInfo, tempExpRefList, imageScalerList, weightList, bgModelList,
                 supplementaryData, *args, **kwargs):
        """!
        \brief Assemble the coadd

        Requires additional inputs Struct `supplementaryData` to contain a `templateCoadd` that serves
        as the model of the static sky.

        Find artifacts and apply them to the warps' masks creating a list of alternative masks with a
        new "CLIPPED" plane and updated "NO_DATA" plane.
        Then pass these alternative masks to the base class's assemble method.

        @param skyInfo: Patch geometry information
        @param tempExpRefList: List of data references to warps
        @param imageScalerList: List of image scalers
        @param weightList: List of weights
        @param bgModelList: List of background models from background matching
        @param supplementaryData: PipeBase.Struct containing a templateCoadd

        return pipeBase.Struct with coaddExposure, nImage if requested
        """
        templateCoadd = supplementaryData.templateCoadd
        spanSetMaskList = self.findArtifacts(templateCoadd, tempExpRefList, imageScalerList)
        maskList = self.computeAltMaskList(tempExpRefList, spanSetMaskList)
        badMaskPlanes = self.config.badMaskPlanes[:]
        badMaskPlanes.append("CLIPPED")
        badPixelMask = afwImage.Mask.getPlaneBitMask(badMaskPlanes)

        retStruct = AssembleCoaddTask.assemble(self, skyInfo, tempExpRefList, imageScalerList, weightList,
                                               bgModelList, maskList, mask=badPixelMask)

        # Set the coadd CLIPPED mask from the footprints since currently pixels that are masked
        # do not get propagated (Remove with DM-9953)
        mask = retStruct.coaddExposure.maskedImage.mask
        for maskClip in maskList:
            maskClip &= mask.getPlaneBitMask("CLIPPED")
            mask |= maskClip

        return retStruct

    def findArtifacts(self, templateCoadd, tempExpRefList, imageScalerList):
        """!
        \brief Find artifacts

        Loop through warps twice. The first loop builds a map with the count of how many
        epochs each pixel deviates from the templateCoadd by more than config.chiThreshold sigma.
        The second loop takes each difference image and filters the artifacts detected
        in each using count map to filter out variable sources and sources that are difficult to
        subtract cleanly.

        @param templateCoadd: Exposure to serve as model of static sky
        @param tempExpRefList: List of data references to warps
        @param imageScalerList: List of image scalers
        """

        self.log.debug("Generating Count Image, and mask lists.")
        coaddBBox = templateCoadd.getBBox()
        slateIm = afwImage.ImageU(coaddBBox)
        epochCountImage = afwImage.ImageU(coaddBBox)
        spanSetArtifactList = []
        spanSetNoDataMaskList = []

        for warpRef, imageScaler in zip(tempExpRefList, imageScalerList):
            warpDiffExp = self._readAndComputeWarpDiff(warpRef, imageScaler, templateCoadd)
            if warpDiffExp is not None:
                fpSet = self.detect.detectFootprints(warpDiffExp, doSmooth=False, clearMask=True)
                fpSet.positive.merge(fpSet.negative)
                footprints = fpSet.positive
                slateIm.set(0)
                spanSetList = [footprint.spans for footprint in footprints.getFootprints()]
                for spans in spanSetList:
                    spans.setImage(slateIm, 1, doClip=True)
                epochCountImage += slateIm

                # PSF-Matched warps have less available area (~the matching kernel) because the calexps
                # undergo a second convolution. Pixels with data in the direct warp
                # but not in the PSF-matched warp will not have their artifacts detected.
                # NaNs from the PSF-matched warp therefore must be masked in the direct warp
                nans = numpy.where(numpy.isnan(warpDiffExp.maskedImage.image.array), 1, 0)
                nansMask = afwImage.makeMaskFromArray(nans.astype(afwImage.MaskPixel))
                nansMask.setXY0(warpDiffExp.getXY0())
            else:
                # If the directWarp has <1% coverage, the psfMatchedWarp can have 0% and not exist
                # In this case, mask the whole epoch
                nansMask = afwImage.MaskX(coaddBBox, 1)
                spanSetList = []

            spanSetNoDataMask = afwGeom.SpanSet.fromMask(nansMask).split()

            spanSetNoDataMaskList.append(spanSetNoDataMask)
            spanSetArtifactList.append(spanSetList)

        if lsstDebug.Info(__name__).saveCountIm:
            path = self._dataRef2DebugPath("epochCountIm", tempExpRefList[0], coaddLevel=True)
            epochCountImage.writeFits(path)

        for i, spanSetList in enumerate(spanSetArtifactList):
            if spanSetList:
                filteredSpanSetList = self._filterArtifacts(spanSetList, epochCountImage)
                spanSetArtifactList[i] = filteredSpanSetList

        return pipeBase.Struct(artifacts=spanSetArtifactList,
                               noData=spanSetNoDataMaskList)

    def computeAltMaskList(self, tempExpRefList, maskSpanSets):
        """!
        \brief Apply artifact span set lists to masks

        @param tempExpRefList: List of data references to warps
        @param maskSpanSets: Struct containing artifact and noData spanSet lists to apply

        return List of alternative masks

        Add artifact span set list as "CLIPPED" plane and NaNs to existing "NO_DATA" plane
        """
        spanSetMaskList = maskSpanSets.artifacts
        spanSetNoDataList = maskSpanSets.noData
        altMaskList = []
        for warpRef, artifacts, noData in zip(tempExpRefList, spanSetMaskList, spanSetNoDataList):
            warp = warpRef.get(self.getTempExpDatasetName(self.config.warpType), immediate=True)
            mask = warp.maskedImage.mask
            maskClipValue = mask.addMaskPlane("CLIPPED")
            noDataValue = mask.addMaskPlane("NO_DATA")
            for artifact in artifacts:
                artifact.clippedTo(mask.getBBox()).setMask(mask, 2**maskClipValue)
            for noDataRegion in noData:
                noDataRegion.clippedTo(mask.getBBox()).setMask(mask, 2**noDataValue)
            altMaskList.append(mask)
            if lsstDebug.Info(__name__).saveAltMask:
                mask.writeFits(self._dataRef2DebugPath("altMask", warpRef))

        return altMaskList

    def _filterArtifacts(self, spanSetList, epochCountImage):
        """!
        \brief Filter artifact candidates

        @param spanSetList: List of SpanSets representing artifact candidates
        @param epochCountImage: Image of accumulated number of warpDiff detections

        return List of SpanSets with artifacts
        """

        maskSpanSetList = []
        x0, y0 = epochCountImage.getXY0()
        for i, span in enumerate(spanSetList):
            y, x = span.indices()
            counts = epochCountImage.array[[y1 - y0 for y1 in y], [x1 - x0 for x1 in x]]
            nCountsBelowThreshold = numpy.count_nonzero((counts > 0) & (counts <= self.config.maxNumEpochs))
            percentBelowThreshold = nCountsBelowThreshold / len(counts)
            if percentBelowThreshold > self.config.spatialThreshold:
                maskSpanSetList.append(span)
        return maskSpanSetList

    def _readAndComputeWarpDiff(self, warpRef, imageScaler, templateCoadd):
        # Warp comparison must use PSF-Matched Warps regardless of requested coadd warp type
        warpName = self.getTempExpDatasetName('psfMatched')
        if not warpRef.datasetExists(warpName):
            self.log.warn("Could not find %s %s; skipping it", warpName, warpRef.dataId)
            return None
        warp = warpRef.get(self.getTempExpDatasetName('psfMatched'), immediate=True)
        # direct image scaler OK for PSF-matched Warp
        imageScaler.scaleMaskedImage(warp.getMaskedImage())
        mi = warp.getMaskedImage()
        mi -= templateCoadd.getMaskedImage()
        return warp

    def _dataRef2DebugPath(self, prefix, warpRef, coaddLevel=False):
        """!
        \brief Return a path to which to write debugging output

        @param prefix: string, prefix for filename
        @param warpRef: Butler dataRef
        @param coaddLevel: bool, optional. If True, include only coadd-level keys
                           (e.g. 'tract', 'patch', 'filter', but no 'visit')

        Creates a hyphen-delimited string of dataId values for simple filenames.
        """
        if coaddLevel:
            keys = warpRef.getButler().getKeys(self.getCoaddDatasetName(self.warpType))
        else:
            keys = warpRef.dataId.keys()
        keyList = sorted(keys, reverse=True)
        directory = lsstDebug.Info(__name__).figPath if lsstDebug.Info(__name__).figPath else "."
        filename = "%s-%s.fits" % (prefix, '-'.join([str(warpRef.dataId[k]) for k in keyList]))
        return os.path.join(directory, filename)
