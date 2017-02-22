from __future__ import division, absolute_import
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
import resource

import lsst.pex.config as pexConfig
import lsst.pex.exceptions as pexExceptions
import lsst.afw.detection as afwDetect
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.afw.detection as afwDet
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
import lsst.meas.algorithms as measAlg
from .coaddBase import CoaddBaseTask, SelectDataIdContainer
from .interpImage import InterpImageTask
from .matchBackgrounds import MatchBackgroundsTask
from .scaleZeroPoint import ScaleZeroPointTask
from .coaddHelpers import groupPatchExposures, getGroupDataRef
from lsst.meas.algorithms import SourceDetectionTask

__all__ = ["AssembleCoaddTask","SafeClipAssembleCoaddTask"]

class AssembleCoaddConfig(CoaddBaseTask.ConfigClass):
    """!
\anchor AssembleCoaddConfig_

\brief Configuration parameters for the \ref AssembleCoaddTask_ "AssembleCoaddTask"
    """
    subregionSize = pexConfig.ListField(
        dtype = int,
        doc = "Width, height of stack subregion size; " \
              "make small enough that a full stack of images will fit into memory at once.",
        length = 2,
        default = (2000, 2000),
    )
    doSigmaClip = pexConfig.Field(
        dtype = bool,
        doc = "Perform sigma clipped outlier rejection? If False then compute a simple mean.",
        default = True,
    )
    sigmaClip = pexConfig.Field(
        dtype = float,
        doc = "Sigma for outlier rejection; ignored if doSigmaClip false.",
        default = 3.0,
    )
    clipIter = pexConfig.Field(
        dtype = int,
        doc = "Number of iterations of outlier rejection; ignored if doSigmaClip false.",
        default = 2,
    )
    scaleZeroPoint = pexConfig.ConfigurableField(
        target = ScaleZeroPointTask,
        doc = "Task to adjust the photometric zero point of the coadd temp exposures",
    )
    doInterp = pexConfig.Field(
        doc = "Interpolate over NaN pixels? Also extrapolate, if necessary, but the results are ugly.",
        dtype = bool,
        default = True,
    )
    interpImage = pexConfig.ConfigurableField(
        target = InterpImageTask,
        doc = "Task to interpolate (and extrapolate) over NaN pixels",
    )
    matchBackgrounds = pexConfig.ConfigurableField(
        target = MatchBackgroundsTask,
        doc = "Task to match backgrounds",
    )
    maxMatchResidualRatio = pexConfig.Field(
        doc = "Maximum ratio of the mean squared error of the background matching model to the variance " \
        "of the difference in backgrounds",
        dtype = float,
        default = 1.1
    )
    maxMatchResidualRMS = pexConfig.Field(
        doc = "Maximum RMS of residuals of the background offset fit in matchBackgrounds.",
        dtype = float,
        default = 1.0
    )
    doWrite = pexConfig.Field(
        doc = "Persist coadd?",
        dtype = bool,
        default = True,
    )
    doMatchBackgrounds = pexConfig.Field(
        doc = "Match backgrounds of coadd temp exposures before coadding them? " \
        "If False, the coadd temp expsosures must already have been background subtracted or matched",
        dtype = bool,
        default = True,
    )
    autoReference = pexConfig.Field(
        doc = "Automatically select the coadd temp exposure to use as a reference for background matching? " \
              "Ignored if doMatchBackgrounds false. " \
              "If False you must specify the reference temp exposure as the data Id",
        dtype = bool,
        default = True,
    )
    maskPropagationThresholds = pexConfig.DictField(
        keytype = str,
        itemtype = float,
        doc = ("Threshold (in fractional weight) of rejection at which we propagate a mask plane to "
               "the coadd; that is, we set the mask bit on the coadd if the fraction the rejected frames "
               "would have contributed exceeds this value."),
        default = {"SAT": 0.1},
    )
    removeMaskPlanes = pexConfig.ListField(dtype=str, default=["CROSSTALK", "NOT_DEBLENDED"],\
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

    def setDefaults(self):
        CoaddBaseTask.ConfigClass.setDefaults(self)
        self.badMaskPlanes = ["NO_DATA", "BAD", "CR",]


## \addtogroup LSST_task_documentation
## \{
## \page AssembleCoaddTask
## \ref AssembleCoaddTask_ "AssembleCoaddTask"
## \copybrief AssembleCoaddTask
## \}

class AssembleCoaddTask(CoaddBaseTask):
    """!
\anchor AssembleCoaddTask_

\brief Assemble a coadded image from a set of coadded temporary exposures.

\section pipe_tasks_assembleCoadd_Contents Contents
  - \ref pipe_tasks_assembleCoadd_AssembleCoaddTask_Purpose
  - \ref pipe_tasks_assembleCoadd_AssembleCoaddTask_Initialize
  - \ref pipe_tasks_assembleCoadd_AssembleCoaddTask_Run
  - \ref pipe_tasks_assembleCoadd_AssembleCoaddTask_Config
  - \ref pipe_tasks_assembleCoadd_AssembleCoaddTask_Debug
  - \ref pipe_tasks_assembleCoadd_AssembleCoaddTask_Example

\section pipe_tasks_assembleCoadd_AssembleCoaddTask_Purpose	Description

\copybrief AssembleCoaddTask_

We want to assemble a coadded image from a set of coadded temporary exposures (coaddTempExps).
Each input coaddTempExp covers a patch on the sky and corresponds to a single run/visit/exposure of the
covered patch. We provide the task with a list of coaddTempExps (selectDataList) from which it selects
coaddTempExps that cover the specified patch (pointed at by dataRef).
Each coaddTempExp that goes into a coadd will typically have an independent photometric zero-point.
Therefore, we must scale each coaddTempExp to set it to a common photometric zeropoint. By default, each
coaddTempExp has backgrounds and hence will require config.doMatchBackgrounds=True.
When background matching is enabled, the task may be configured to automatically select a reference exposure
(config.autoReference=True). If this is not done, we require that the input dataRef provides access to a
coaddTempExp (dataset type coaddName + 'Coadd_tempExp') which is used as the reference exposure.
The coadd is computed as a mean with optional outlier rejection.
Criteria for outlier rejection are set in \ref AssembleCoaddConfig. Finally, coaddTempExps can have bad 'NaN'
pixels which received no input from the source calExps. We interpolate over these bad (NaN) pixels.

AssembleCoaddTask uses several sub-tasks. These are
<DL>
  <DT>\ref ScaleZeroPointTask_ "ScaleZeroPointTask"</DT>
  <DD> create and use an imageScaler object to scale the photometric zeropoint for each coaddTempExp</DD>
  <DT>\ref MatchBackgroundsTask_ "MatchBackgroundsTask"</DT>
  <DD> match background in a coaddTempExp to a reference exposure (and select the reference exposure if one is
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

AssembleCoaddTask assembles a set of warped coaddTempExp images into a coadded image. The AssembleCoaddTask
can be invoked by running assembleCoadd.py with the flag '--legacyCoadd'. Usage of assembleCoadd.py expects
a data reference to the tract patch and filter to be coadded (specified using
'--id = [KEY=VALUE1[^VALUE2[^VALUE3...] [KEY=VALUE1[^VALUE2[^VALUE3...] ...]]') along with a list of
coaddTempExps to attempt to coadd (specified using
'--selectId [KEY=VALUE1[^VALUE2[^VALUE3...] [KEY=VALUE1[^VALUE2[^VALUE3...] ...]]'). Only the coaddTempExps
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
This will produce warped coaddTempExps for each visit. To coadd the warped data, we call assembleCoadd.py as
follows:
\code
assembleCoadd.py --legacyCoadd $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-I --selectId visit=903986 ccd=16 --selectId visit=903986 ccd=22 --selectId visit=903986 ccd=23 --selectId visit=903986 ccd=100 --selectId visit=904014 ccd=1 --selectId visit=904014 ccd=6 --selectId visit=904014 ccd=12 --selectId visit=903990 ccd=18 --selectId visit=903990 ccd=25 --selectId visit=904010 ccd=4 --selectId visit=904010 ccd=10 --selectId visit=904010 ccd=100 --selectId visit=903988 ccd=16 --selectId visit=903988 ccd=17 --selectId visit=903988 ccd=23 --selectId visit=903988 ccd=24\endcode
that will process the HSC-I band data. The results are written in $CI_HSC_DIR/DATA/deepCoadd-results/HSC-I
You may also choose to run:
\code
scons warp-903334 warp-903336 warp-903338 warp-903342 warp-903344 warp-903346
assembleCoadd.py --legacyCoadd $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-R --selectId visit=903334 ccd=16 --selectId visit=903334 ccd=22 --selectId visit=903334 ccd=23 --selectId visit=903334 ccd=100 --selectId visit=903336 ccd=17 --selectId visit=903336 ccd=24 --selectId visit=903338 ccd=18 --selectId visit=903338 ccd=25 --selectId visit=903342 ccd=4 --selectId visit=903342 ccd=10 --selectId visit=903342 ccd=100 --selectId visit=903344 ccd=0 --selectId visit=903344 ccd=5 --selectId visit=903344 ccd=11 --selectId visit=903346 ccd=1 --selectId visit=903346 ccd=6 --selectId visit=903346 ccd=12
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
            mask = afwImage.MaskU()
            try:
                self.brightObjectBitmask = 1 << mask.addMaskPlane(self.config.brightObjectMaskName)
            except pexExceptions.LsstCppException:
                raise RuntimeError("Unable to define mask plane for bright objects; planes used are %s" %
                                   mask.getMaskPlaneDict().keys())
            del mask

    @pipeBase.timeMethod
    def run(self, dataRef, selectDataList=[]):
        """!
        \brief Assemble a coadd from a set of coaddTempExp

        Coadd a set of coaddTempExps. Compute weights to be applied to each coaddTempExp and find scalings to
        match the photometric zeropoint to a reference coaddTempExp. Optionally, match backgrounds across
        coaddTempExps if the background has not already been removed. Assemble the coaddTempExps using
        \ref assemble. Interpolate over NaNs and optionally write the coadd to disk. Return the coadded
        exposure.

        \anchor runParams
        \param[in] dataRef: Data reference defining the patch for coaddition and the reference coaddTempExp
                        (if config.autoReference=False). Used to access the following data products:
                        - [in] self.config.coaddName + "Coadd_skyMap"
                        - [in] self.config.coaddName + "Coadd_tempExp" (optionally)
                        - [out] self.config.coaddName + "Coadd"
        \param[in] selectDataList[in]: List of data references to coaddTempExps. Data to be coadded will be
                                   selected from this list based on overlap with the patch defined by dataRef.

        \return a pipeBase.Struct with fields:
                 - coaddExposure: coadded exposure
        """
        skyInfo = self.getSkyInfo(dataRef)
        calExpRefList = self.selectExposures(dataRef, skyInfo, selectDataList=selectDataList)
        if len(calExpRefList) == 0:
            self.log.warn("No exposures to coadd")
            return
        self.log.info("Coadding %d exposures", len(calExpRefList))

        tempExpRefList = self.getTempExpRefList(dataRef, calExpRefList)
        inputData = self.prepareInputs(tempExpRefList)
        self.log.info("Found %d %s", len(inputData.tempExpRefList), self.getTempExpDatasetName())
        if len(inputData.tempExpRefList) == 0:
            self.log.warn("No coadd temporary exposures found")
            return
        if self.config.doMatchBackgrounds:
            refImageScaler = self.getBackgroundReferenceScaler(dataRef)
            inputData = self.backgroundMatching(inputData, dataRef, refImageScaler)
            if len(inputData.tempExpRefList) == 0:
                self.log.warn("No valid background models")
                return
        if self.config.doMatchBackgrounds:
            backgroundInfoList = inputData.backgroundInfoList
        else:
            backgroundInfoList = None
        coaddExp, coaddCov = self.assemble(skyInfo, inputData.tempExpRefList, inputData.imageScalerList,
                                           inputData.weightList,
                                           backgroundInfoList,
                                           doClip=self.config.doSigmaClip)
        if self.config.doMatchBackgrounds:
            self.addBackgroundMatchingMetadata(coaddExp, inputData.tempExpRefList,
                                               inputData.backgroundInfoList)

        if self.config.doInterp:
            self.interpImage.run(coaddExp.getMaskedImage(), planeName="NO_DATA")
            # The variance must be positive; work around for DM-3201.
            varArray = coaddExp.getMaskedImage().getVariance().getArray()
            varArray[:] = numpy.where(varArray > 0, varArray, numpy.inf)

        if self.config.doMaskBrightObjects:
            brightObjectMasks = self.readBrightObjectMasks(dataRef)
            self.setBrightObjectMasks(coaddExp, dataRef.dataId, brightObjectMasks)
        if self.config.doWrite:
            self.writeCoaddOutput(dataRef, coaddExp)
            self.writeCoaddOutput(dataRef, coaddCov, suffix='Cov')

        return pipeBase.Struct(coaddExposure=coaddExp, coaddCovariance=coaddCov)

    def getTempExpRefList(self, patchRef, calExpRefList):
        """!
        \brief Generate list of coaddTempExp data references corresponding to exposures that lie within the
        patch to be coadded.

        \param[in] patchRef: Data reference for patch
        \param[in] calExpRefList: List of data references for input calexps
        \return List of coaddTempExp data references
        """
        butler = patchRef.getButler()
        groupData = groupPatchExposures(patchRef, calExpRefList, self.getCoaddDatasetName(),
                                        self.getTempExpDatasetName())
        tempExpRefList = [getGroupDataRef(butler, self.getTempExpDatasetName(), g, groupData.keys) for
                          g in groupData.groups.keys()]
        return tempExpRefList

    def getBackgroundReferenceScaler(self, dataRef):
        """!
        \brief Construct an image scaler for the background reference frame

        Each coaddTempExp has a different background level. A reference background level must be chosen before
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
        dataset = self.getTempExpDatasetName()
        if not dataRef.datasetExists(dataset):
            raise RuntimeError("Could not find reference exposure %s %s." % (dataset, dataRef.dataId))

        refExposure = dataRef.get(self.getTempExpDatasetName(), immediate=True)
        refImageScaler = self.scaleZeroPoint.computeImageScaler(
            exposure = refExposure,
            dataRef = dataRef,
            )
        return refImageScaler

    def prepareInputs(self, refList):
        """!
        \brief Prepare the input warps for coaddition by measuring the weight for each warp and the scaling
        for the photometric zero point.

        Each coaddTempExp has its own photometric zeropoint and background variance. Before coadding these
        coaddTempExps together, compute a scale factor to normalize the photometric zeropoint and compute the
        weight for each coaddTempExp.

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
        tempExpName = self.getTempExpDatasetName()
        for tempExpRef in refList:
            if not tempExpRef.datasetExists(tempExpName):
                self.log.warn("Could not find %s %s; skipping it", tempExpName, tempExpRef.dataId)
                continue
            tempExp = tempExpRef.get(tempExpName, immediate=True)
            maskedImage = tempExp.getMaskedImage()
            imageScaler = self.scaleZeroPoint.computeImageScaler(
                exposure = tempExp,
                dataRef = tempExpRef,
            )
            try:
                imageScaler.scaleMaskedImage(maskedImage)
            except Exception as e:
                self.log.warn("Scaling failed for %s (skipping it): %s", tempExpRef.dataId, e)
                continue
            statObj = afwMath.makeStatistics(maskedImage.getVariance(), maskedImage.getMask(),
                afwMath.MEANCLIP, statsCtrl)
            meanVar, meanVarErr = statObj.getResult(afwMath.MEANCLIP);
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

        Each coaddTempExp has a different background level that must be normalized to a reference level
        before coaddition. If no reference is provided, the background matcher selects one. If the background
        matching is performed sucessfully, recompute the weight to be applied to the coaddTempExp to be
        consistent with the scaled background.

        \param[in] inputData: Struct from prepareInputs() with tempExpRefList, weightList, imageScalerList
        \param[in] refExpDataRef: Data reference for background reference tempExp, or None
        \param[in] refImageScaler: Image scaler for background reference tempExp, or None
        \return Struct:
        - tempExprefList: List of data references to tempExp
        - weightList: List of weightings
        - imageScalerList: List of image scalers
        - backgroundInfoList: result from background matching
        """
        try:
            backgroundInfoList = self.matchBackgrounds.run(
                expRefList = inputData.tempExpRefList,
                imageScalerList = inputData.imageScalerList,
                refExpDataRef = refExpDataRef if not self.config.autoReference else None,
                refImageScaler = refImageScaler,
                expDatasetType = self.getTempExpDatasetName(),
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
                    varianceRatio =  bgInfo.matchedMSE / bgInfo.diffImVar
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
                elif ( bgInfo.fitRMS > self.config.maxMatchResidualRMS):
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
                 altMaskList=None, doClip=False, mask=None):
        """!
        \anchor AssembleCoaddTask.assemble_

        \brief Assemble a coadd from input warps

        Assemble the coadd using the provided list of coaddTempExps. Since the full coadd covers a patch (a
        large area), the assembly is performed over small areas on the image at a time in order to
        conserve memory usage. Iterate over subregions within the outer bbox of the patch using
        \ref assembleSubregion to mean-stack the corresponding subregions from the coaddTempExps (with outlier
        rejection if config.doSigmaClip=True). Set the edge bits the the coadd mask based on the weight map.

        \param[in] skyInfo: Patch geometry information, from getSkyInfo
        \param[in] tempExpRefList: List of data references to tempExp
        \param[in] imageScalerList: List of image scalers
        \param[in] weightList: List of weights
        \param[in] bgInfoList: List of background data from background matching, or None
        \param[in] altMaskList: List of alternate masks to use rather than those stored with tempExp, or None
        \param[in] doClip: Use clipping when codding?
        \param[in] mask: Mask to ignore when coadding
        \return coadded exposure
        """
        tempExpName = self.getTempExpDatasetName()
        tempCovName = self.getTempCovDatasetName()
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
            bit = afwImage.MaskU.getMaskPlane(plane)
            statsCtrl.setMaskPropagationThreshold(bit, threshold)

        if doClip:
            statsFlags = afwMath.MEANCLIP
        else:
            statsFlags = afwMath.MEAN

        if bgInfoList is None:
            bgInfoList = [None]*len(tempExpRefList)

        if altMaskList is None:
            altMaskList = [None]*len(tempExpRefList)

        coaddExposure = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
        coaddExposure.setCalib(self.scaleZeroPoint.getCalib())
        coaddExposure.getInfo().setCoaddInputs(self.inputRecorder.makeCoaddInputs())
        self.assembleMetadata(coaddExposure, tempExpRefList, weightList)
        for tempExpRef in tempExpRefList:
            if not tempExpRef.datasetExists(tempExpName):
                continue
            tempExp = tempExpRef.get(tempExpName, immediate=True)
            tempCov = tempExpRef.get(tempCovName, immediate=True)
            multX = int(tempCov.getWidth()/tempExp.getWidth())
            multY = int(tempCov.getHeight()/tempExp.getHeight())
            break
        covBBox = afwGeom.Box2I(afwGeom.Point2I(skyInfo.bbox.getBegin().getX()*multX,
                                                skyInfo.bbox.getBegin().getY()*multY),
                                afwGeom.Extent2I(skyInfo.bbox.getWidth()*multX,
                                                 skyInfo.bbox.getHeight()*multY))
        coaddCovariance = afwImage.ImageD(covBBox)
        coaddMaskedImage = coaddExposure.getMaskedImage()
        subregionSizeArr = self.config.subregionSize
        subregionSize = afwGeom.Extent2I(subregionSizeArr[0], subregionSizeArr[1])
        covSubregionSize = afwGeom.Extent2I(subregionSizeArr[0]*multX, subregionSizeArr[1]*multY)
        for subBBox, covSubBBox in _covSubBBoxIter(skyInfo.bbox, covBBox, subregionSize, covSubregionSize):
            try:
                self.assembleSubregion(coaddExposure, subBBox, coaddCovariance, covSubBBox,
                                       tempExpRefList, imageScalerList,
                                       weightList, bgInfoList, altMaskList, statsFlags, statsCtrl)
            except Exception as e:
                self.log.fatal("Cannot compute coadd %s: %s", subBBox, e)

        coaddUtils.setCoaddEdgeBits(coaddMaskedImage.getMask(), coaddMaskedImage.getVariance())

        return coaddExposure, coaddCovariance

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
        tempExpName = self.getTempExpDatasetName()
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
        if self.config.doPsfMatch:
            psf = self.config.modelPsf.apply()
        else:
            psf = measAlg.CoaddPsf(coaddInputs.ccds, coaddExposure.getWcs())
        coaddExposure.setPsf(psf)
        apCorrMap = measAlg.makeCoaddApCorrMap(coaddInputs.ccds, coaddExposure.getBBox(afwImage.PARENT),
                                               coaddExposure.getWcs())
        coaddExposure.getInfo().setApCorrMap(apCorrMap)

    def assembleSubregion(self, coaddExposure, bbox, coaddCovariance, covBBox,
                          tempExpRefList, imageScalerList, weightList,
                          bgInfoList, altMaskList, statsFlags, statsCtrl):
        """!
        \brief Assemble the coadd for a sub-region.

        For each coaddTempExp, check for (and swap in) an alternative mask if one is passed. If background
        matching is enabled, add the background and background variance from each coaddTempExp. Remove mask
        planes listed in config.removeMaskPlanes, Finally, mean-stack
        the actual exposures using \ref afwMath.statisticsStack "statisticsStack" with outlier rejection if
        config.doSigmaClip=True. Assign the stacked subregion back to the coadd.

        \param[in] coaddExposure: The target image for the coadd
        \param[in] bbox: Sub-region to coadd
        \param[in] tempExpRefList: List of data reference to tempExp
        \param[in] imageScalerList: List of image scalers
        \param[in] weightList: List of weights
        \param[in] bgInfoList: List of background data from background matching
        \param[in] altMaskList: List of alternate masks to use rather than those stored with tempExp, or None
        \param[in] statsFlags: Statistic for coadd
        \param[in] statsCtrl: Statistics control object for coadd
        """
        self.log.debug("Computing coadd over %s", bbox)
        tempExpName = self.getTempExpDatasetName()
        tempCovName = self.getTempCovDatasetName()
        coaddMaskedImage = coaddExposure.getMaskedImage()
        maskedImageList = afwImage.vectorMaskedImageF()  # [] is rejected by afwMath.statisticsStack
        covarianceList = afwImage.vectorImageD()  # [] is rejected by afwMath.statisticsStack
        for tempExpRef, imageScaler, bgInfo, altMask in zip(tempExpRefList, imageScalerList, bgInfoList,
                                                            altMaskList):
            exposure = tempExpRef.get(tempExpName + "_sub", bbox=bbox, immediate=True)
            maskedImage = exposure.getMaskedImage()
            covariance = tempExpRef.get(tempCovName + "_sub", bbox=covBBox, immediate=True)

            if altMask:
                altMaskSub = altMask.Factory(altMask, bbox, afwImage.PARENT)
                maskedImage.getMask().swap(altMaskSub)
            imageScaler.scaleMaskedImage(maskedImage, covImage=covariance)

            if self.config.doMatchBackgrounds and not bgInfo.isReference:
                backgroundModel = bgInfo.backgroundModel
                backgroundImage = backgroundModel.getImage() if \
                    self.matchBackgrounds.config.usePolynomial else \
                    backgroundModel.getImageF()
                backgroundImage.setXY0(coaddMaskedImage.getXY0())
                maskedImage += backgroundImage.Factory(backgroundImage, bbox, afwImage.PARENT, False)
                var = maskedImage.getVariance()
                var += (bgInfo.fitRMS)**2

            if self.config.removeMaskPlanes:
                mask = maskedImage.getMask()
                for maskPlane in self.config.removeMaskPlanes:
                    try:
                        mask &= ~mask.getPlaneBitMask(maskPlane)
                    except Exception as e:
                        self.log.warn("Unable to remove mask plane %s: %s", maskPlane, e)

            maskedImageList.append(maskedImage)
            covarianceList.append(covariance)

        coaddSubregion = afwImage.MaskedImageF(bbox)
        coaddCovarianceSubregion = afwImage.ImageD(covBBox)
        afwMath.statisticsStack(
            coaddSubregion, coaddCovarianceSubregion,
            maskedImageList, covarianceList, statsFlags, statsCtrl, weightList)
        coaddMaskedImage.assign(coaddSubregion, bbox)
        coaddCovariance.assign(coaddCovarianceSubregion, covBBox)

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
            tempExpStr = '&'.join('%s=%s' % (k,v) for k,v in tempExpRef.dataId.items())
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
            radius = rec["radius"].asArcseconds()/plateScale   # convert to pixels

            foot = afwDetect.Footprint(center, radius, exposure.getBBox())
            afwDetect.setMaskFromFootprint(mask, foot, self.brightObjectBitmask)

    @classmethod
    def _makeArgumentParser(cls):
        """!
        \brief Create an argument parser
        """
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", cls.ConfigClass().coaddName + "Coadd_tempExp",
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
                raise RuntimeError("Bug: empty bbox! bbox=%s, subregionSize=%s, colShift=%s, rowShift=%s" % \
                    (bbox, subregionSize, colShift, rowShift))
            yield subBBox


def _covSubBBoxIter(bbox, covBBox, subregionSize, covSubregionSize):
    """!
    \brief Iterate over subregions of two bboxes

    \param[in] bbox: bounding box over which to iterate: afwGeom.Box2I
    \param[in] covBBox: bounding box of covariance image over which to iterate: afwGeom.Box2I
    \param[in] subregionSize: size of sub-bboxes
    \param[in] covSubregionSize: size of cov sub-bboxes

    \return subBBox: next sub-bounding box of size subregionSize or smaller;
        each subBBox is contained within bbox, so it may be smaller than subregionSize at the edges of bbox,
        but it will never be empty
    """
    if bbox.isEmpty():
        raise RuntimeError("bbox %s is empty" % (bbox,))
    if covBBox.isEmpty():
        raise RuntimeError("covBBox %s is empty" % (covBBox,))
    if subregionSize[0] < 1 or subregionSize[1] < 1:
        raise RuntimeError("subregionSize %s must be nonzero" % (subregionSize,))
    if covSubregionSize[0] < 1 or covSubregionSize[1] < 1:
        raise RuntimeError("covSubregionSize %s must be nonzero" % (covSubregionSize,))

    for rowShift, covRowShift in zip(range(0, bbox.getHeight(), subregionSize[1]),
                                     range(0, covBBox.getHeight(), covSubregionSize[1])):
        for colShift, covColShift in zip(range(0, bbox.getWidth(), subregionSize[0]),
                                         range(0, covBBox.getWidth(), covSubregionSize[0])):
            subBBox = afwGeom.Box2I(bbox.getMin() + afwGeom.Extent2I(colShift, rowShift), subregionSize)
            covSubBBox = afwGeom.Box2I(covBBox.getMin() + afwGeom.Extent2I(covColShift, covRowShift),
                                       covSubregionSize)
            subBBox.clip(bbox)
            covSubBBox.clip(covBBox)
            if subBBox.isEmpty():
                raise RuntimeError("Bug: empty bbox! bbox=%s, subregionSize=%s, colShift=%s, rowShift=%s"
                                   % (bbox, subregionSize, colShift, rowShift))
            if covSubBBox.isEmpty():
                raise RuntimeError("Bug: empty covBBox! covBBox=%s, covSubregionSize=%s, colShift=%s, \
                                    rowShift=%s"
                                   % (covBBox, covSubregionSize, covColShift, covRowShift))
            yield subBBox, covSubBBox


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
        keysCoaddTempExp = namespace.butler.getKeys(datasetType=namespace.config.coaddName + "Coadd_tempExp",
                                                    level=self.level)

        if namespace.config.doMatchBackgrounds:
            if namespace.config.autoReference: #matcher will pick it's own reference image
                datasetType = namespace.config.coaddName + "Coadd"
                validKeys = keysCoadd
            else:
                datasetType = namespace.config.coaddName + "Coadd_tempExp"
                validKeys = keysCoaddTempExp
        else: #bkg subtracted coadd
            datasetType = namespace.config.coaddName + "Coadd"
            validKeys = keysCoadd

        for dataId in self.idList:
            # tract and patch are required
            for key in validKeys:
                if key not in dataId:
                    raise RuntimeError("--id must include " + key)

            for key in dataId: # check if users supplied visit/run
                if (key not in keysCoadd) and (key in keysCoaddTempExp):  #user supplied a visit/run
                    if namespace.config.autoReference:
                        # user probably meant: autoReference = False
                        namespace.config.autoReference = False
                        datasetType = namespace.config.coaddName + "Coadd_tempExp"
                        print "Switching config.autoReference to False; applies only to background Matching."
                        break

            dataRef = namespace.butler.dataRef(
                datasetType = datasetType,
                dataId = dataId,
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
    fp = afwImage.MaskU(bbox)
    subMask = mask.Factory(mask, bbox, afwImage.PARENT)
    afwDet.setMaskFromFootprint(fp, footprint, bitmask)
    return numpy.logical_and((subMask.getArray() & fp.getArray()) > 0,
                             (subMask.getArray() & ignoreMask) == 0).sum()


class SafeClipAssembleCoaddConfig(AssembleCoaddConfig):
    """!
\anchor SafeClipAssembleCoaddConfig

\brief Configuration parameters for the SafeClipAssembleCoaddTask
    """
    clipDetection = pexConfig.ConfigurableField(target=SourceDetectionTask,
                                      doc="Detect sources on difference between unclipped and clipped coadd")
    minClipFootOverlap = pexConfig.Field(
        doc = "Minimum fractional overlap of clipped footprint with visit DETECTED to be clipped",
        dtype = float,
        default = 0.6
    )
    minClipFootOverlapSingle = pexConfig.Field(
        doc = "Minimum fractional overlap of clipped footprint with visit DETECTED to be " \
              "clipped when only one visit overlaps",
        dtype = float,
        default = 0.5
    )
    minClipFootOverlapDouble = pexConfig.Field(
        doc = "Minimum fractional overlap of clipped footprints with visit DETECTED to be " \
              "clipped when two visits overlap",
        dtype = float,
        default = 0.45
    )
    maxClipFootOverlapDouble = pexConfig.Field(
        doc = "Maximum fractional overlap of clipped footprints with visit DETECTED when " \
              "considering two visits",
        dtype = float,
        default = 0.15
    )
    minBigOverlap = pexConfig.Field(
        doc = "Minimum number of pixels in footprint to use DETECTED mask from the single visits " \
              "when labeling clipped footprints",
        dtype = int,
        default = 100
    )

    def setDefaults(self):
        # The numeric values for these configuration parameters were empirically determined, future work
        # may further refine them.
        AssembleCoaddConfig.setDefaults(self)
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

## \addtogroup LSST_task_documentation
## \{
## \page SafeClipAssembleCoaddTask
## \ref SafeClipAssembleCoaddTask_ "SafeClipAssembleCoaddTask"
## \copybrief SafeClipAssembleCoaddTask
## \}

class SafeClipAssembleCoaddTask(AssembleCoaddTask):
    """!
    \anchor SafeClipAssembleCoaddTask_

    \brief Assemble a coadded image from a set of coadded temporary exposures, being careful to clip & flag areas
    with potential artifacts.

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

    \section pipe_tasks_assembleCoadd_SafeClipAssembleCoaddTask_Example	A complete example of using SafeClipAssembleCoaddTask

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
    This will produce warped coaddTempExps for each visit. To coadd the wraped data, we call assembleCoadd.py
    as follows:
    \code
    assembleCoadd.py $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-I --selectId visit=903986 ccd=16 --selectId visit=903986 ccd=22 --selectId visit=903986 ccd=23 --selectId visit=903986 ccd=100 --selectId visit=904014 ccd=1 --selectId visit=904014 ccd=6 --selectId visit=904014 ccd=12 --selectId visit=903990 ccd=18 --selectId visit=903990 ccd=25 --selectId visit=904010 ccd=4 --selectId visit=904010 ccd=10 --selectId visit=904010 ccd=100 --selectId visit=903988 ccd=16 --selectId visit=903988 ccd=17 --selectId visit=903988 ccd=23 --selectId visit=903988 ccd=24
    \endcode
    This will process the HSC-I band data. The results are written in $CI_HSC_DIR/DATA/deepCoadd-results/HSC-I
    You may also choose to run:
    \code
    scons warp-903334 warp-903336 warp-903338 warp-903342 warp-903344 warp-903346
    assembleCoadd.py $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-R --selectId visit=903334 ccd=16 --selectId visit=903334 ccd=22 --selectId visit=903334 ccd=23 --selectId visit=903334 ccd=100 --selectId visit=903336 ccd=17 --selectId visit=903336 ccd=24 --selectId visit=903338 ccd=18 --selectId visit=903338 ccd=25 --selectId visit=903342 ccd=4 --selectId visit=903342 ccd=10 --selectId visit=903342 ccd=100 --selectId visit=903344 ccd=0 --selectId visit=903344 ccd=5 --selectId visit=903344 ccd=11 --selectId visit=903346 ccd=1 --selectId visit=903346 ccd=6 --selectId visit=903346 ccd=12
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

    def assemble(self, skyInfo, tempExpRefList, imageScalerList, weightList, bgModelList, *args, **kwargs):
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
        return coadd exposure
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

        # Assemble coadd from base class, but ignoring CLIPPED pixels (doClip is false)
        badMaskPlanes = self.config.badMaskPlanes[:]
        badMaskPlanes.append("CLIPPED")
        badPixelMask = afwImage.MaskU.getPlaneBitMask(badMaskPlanes)
        coaddExp, coaddCov = AssembleCoaddTask.assemble(self, skyInfo, tempExpRefList, imageScalerList,
                                                        weightList, bgModelList, result.tempExpClipList,
                                                        doClip=False,
                                                        mask=badPixelMask)
        # Set the coadd CLIPPED mask from the footprints since currently pixels that are masked
        # do not get propagated
        maskExp = coaddExp.getMaskedImage().getMask()
        maskExp |= maskClip

        return coaddExp, coaddCov

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
        # Build the unclipped coadd
        coaddMean, coaddCovMean = AssembleCoaddTask.assemble(self, skyInfo, tempExpRefList, imageScalerList,
                                                             weightList, bgModelList, doClip=False)

        # Build the clipped coadd
        coaddClip, coaddCovClip = AssembleCoaddTask.assemble(self, skyInfo, tempExpRefList, imageScalerList,
                                                             weightList, bgModelList, doClip=True)

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
        tempExpClipList = [tmpExpRef.get(self.getTempExpDatasetName(),
                           immediate=True).getMaskedImage().getMask() for tmpExpRef in tempExpRefList]

        for footprint in footprints.getFootprints():
            nPixel = footprint.getArea()
            overlap = [] # hold the overlap with each visit
            maskList = [] # which visit mask match
            indexList = []# index of visit in global list
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
            keepIndex = [] # Which tempExps does the clipped footprint belong to

            # If footprint only has one overlap use a lower threshold
            if len(overlap) == 1:
                if overlap[0] > self.config.minClipFootOverlapSingle:
                    keep = True
                    keepIndex = [0]
            else:
                # This is the general case where only visit should be clipped
                clipIndex = numpy.where(overlap > self.config.minClipFootOverlap)[0]
                if len(clipIndex) == 1:
                    keep=True
                    keepIndex = [clipIndex[0]]

                # Test if there are clipped objects that overlap two different visits
                clipIndex = numpy.where(overlap > self.config.minClipFootOverlapDouble)[0]
                if len(clipIndex) == 2 and len(overlap) > 3:
                    clipIndexComp = numpy.where(overlap < self.config.minClipFootOverlapDouble)[0]
                    if numpy.max(overlap[clipIndexComp]) < self.config.maxClipFootOverlapDouble:
                        keep=True
                        keepIndex = clipIndex

            if not keep:
                continue

            for index in keepIndex:
                afwDet.setMaskFromFootprint(maskList[index], footprint, maskClipValue)

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
