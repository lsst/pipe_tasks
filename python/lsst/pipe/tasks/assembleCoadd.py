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
import lsst.pex.config as pexConfig
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

    def setDefaults(self):
        CoaddBaseTask.ConfigClass.setDefaults(self)
        self.badMaskPlanes = ["NO_DATA", "BAD", "CR",]


class AssembleCoaddTask(CoaddBaseTask):
    """Assemble a coadd from a set of coaddTempExp
    """
    ConfigClass = AssembleCoaddConfig
    _DefaultName = "assembleCoadd"

    def __init__(self, *args, **kwargs):
        CoaddBaseTask.__init__(self, *args, **kwargs)
        self.makeSubtask("interpImage")
        self.makeSubtask("matchBackgrounds")
        self.makeSubtask("scaleZeroPoint")

    @pipeBase.timeMethod
    def run(self, dataRef, selectDataList=[]):
        """Assemble a coadd from a set of coaddTempExp

        The coadd is computed as a mean with optional outlier rejection.

        AssembleCoaddTask performs coaddition of "coadd temporary exposures" ("coaddTempExps").  Each
        coaddTempExp is the size of a patch and contains data for one run, visit or (for a non-mosaic camera)
        Exposure. The coaddTempExps to coadd are selected from the provided selectDataList based on their
        overlap with the patch specified by dataRef.

        By default, coaddTempExps contain backgrounds and hence require config.doMatchBackgrounds=True.
        Background-subtracted coaddTempExps can be coadded by setting config.doMatchBackgrounds=False.

        When background matching is enabled, the task may be configured to automatically select a reference
        exposure (config.autoReference=True). If this is not done, then we require that the input dataRef
        provides access to a coaddTempExp (dataset type coaddName + 'Coadd_tempExp') which is used as the
        reference exposure.

        @param dataRef: Data reference defining the patch for coaddition and the reference coaddTempExp
                        (if config.autoReference=False). Used to access the following data products:
                        - [in] self.config.coaddName + "Coadd_skyMap"
                        - [in] self.config.coaddName + "Coadd_tempExp" (optionally)
                        - [out] self.config.coaddName + "Coadd"
        @param selectDataList[in]: List of data references to coaddTempExp. Data to be coadded will be
                                   selected from this list based on overlap with the patch defined by dataRef.

        @return: a pipeBase.Struct with fields:
                 - coaddExposure: coadded exposure
        """
        skyInfo = self.getSkyInfo(dataRef)
        calExpRefList = self.selectExposures(dataRef, skyInfo, selectDataList=selectDataList)
        if len(calExpRefList) == 0:
            self.log.warn("No exposures to coadd")
            return
        self.log.info("Coadding %d exposures" % len(calExpRefList))

        tempExpRefList = self.getTempExpRefList(dataRef, calExpRefList)
        inputData = self.prepareInputs(tempExpRefList)
        self.log.info("Found %d %s" % (len(inputData.tempExpRefList), self.getTempExpDatasetName()))
        if len(inputData.tempExpRefList) == 0:
            self.log.warn("No coadd temporary exposures found")
            return
        if self.config.doMatchBackgrounds:
            refImageScaler = self.getBackgroundReferenceScaler(dataRef)
            inputData = self.backgroundMatching(inputData, dataRef, refImageScaler)
            if len(inputData.tempExpRefList) == 0:
                self.log.warn("No valid background models")
                return

        coaddExp = self.assemble(skyInfo, inputData.tempExpRefList, inputData.imageScalerList,
                                 inputData.weightList,
                                 inputData.backgroundInfoList if self.config.doMatchBackgrounds else None,
                                 doClip=self.config.doSigmaClip)
        if self.config.doMatchBackgrounds:
            self.addBackgroundMatchingMetadata(coaddExp, inputData.tempExpRefList,
                                               inputData.backgroundInfoList)

        if self.config.doInterp:
            self.interpImage.run(coaddExp.getMaskedImage(), planeName="NO_DATA")
            # The variance must be positive; work around for DM-3201.
            varArray = coaddExp.getMaskedImage().getVariance().getArray()
            varArray[:] = numpy.where(varArray > 0, varArray, numpy.inf)

        if self.config.doWrite:
            self.writeCoaddOutput(dataRef, coaddExp)

        return pipeBase.Struct(coaddExposure=coaddExp)

    def getTempExpRefList(self, patchRef, calExpRefList):
        """Generate list of coaddTempExp data references

        @param patchRef: Data reference for patch
        @param calExpRefList: List of data references for input calexps
        @return List of coaddTempExp data references
        """
        butler = patchRef.getButler()
        groupData = groupPatchExposures(patchRef, calExpRefList, self.getCoaddDatasetName(),
                                        self.getTempExpDatasetName())
        tempExpRefList = [getGroupDataRef(butler, self.getTempExpDatasetName(), g, groupData.keys) for
                          g in groupData.groups.keys()]
        return tempExpRefList

    def getBackgroundReferenceScaler(self, dataRef):
        """Construct an image scaler for the background reference frame

        If there is no reference frame ('autoReference'), then this is a no-op
        and None is returned

        @param dataRef: Data reference for the background reference frame, or None
        @return image scaler, or None
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
        """Prepare the input warps for coaddition

        This involves measuring weights and constructing image scalers
        for each of the inputs.

        @param refList: List of data references to tempExp
        @return Struct:
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
                self.log.warn("Could not find %s %s; skipping it" % (tempExpName, tempExpRef.dataId))
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
                self.log.warn("Scaling failed for %s (skipping it): %s" % (tempExpRef.dataId, e))
                continue
            statObj = afwMath.makeStatistics(maskedImage.getVariance(), maskedImage.getMask(),
                afwMath.MEANCLIP, statsCtrl)
            meanVar, meanVarErr = statObj.getResult(afwMath.MEANCLIP);
            weight = 1.0 / float(meanVar)
            if not numpy.isfinite(weight):
                self.log.warn("Non-finite weight for %s: skipping" % (tempExpRef.dataId,))
                continue
            self.log.info("Weight of %s %s = %0.3f" % (tempExpName, tempExpRef.dataId, weight))

            del maskedImage
            del tempExp

            tempExpRefList.append(tempExpRef)
            weightList.append(weight)
            imageScalerList.append(imageScaler)

        return pipeBase.Struct(tempExpRefList=tempExpRefList, weightList=weightList,
                               imageScalerList=imageScalerList)


    def backgroundMatching(self, inputData, refExpDataRef=None, refImageScaler=None):
        """Perform background matching on the prepared inputs

        If no reference is provided, the background matcher will select one.

        This method returns a new inputData Struct that can replace the original.

        @param inputData: Struct from prepareInputs() with tempExpRefList, weightList, imageScalerList
        @param refExpDataRef: Data reference for background reference tempExp, or None
        @param refImageScaler: Image scaler for background reference tempExp, or None
        @return Struct:
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
            self.log.fatal("Cannot match backgrounds: %s" % (e))
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
                    self.log.info("No background offset model available for %s: skipping"%(
                        tempExpRef.dataId))
                    continue
                try:
                    varianceRatio =  bgInfo.matchedMSE / bgInfo.diffImVar
                except Exception as e:
                    self.log.info("MSE/Var ratio not calculable (%s) for %s: skipping" %
                                  (e, tempExpRef.dataId,))
                    continue
                if not numpy.isfinite(varianceRatio):
                    self.log.info("MSE/Var ratio not finite (%.2f / %.2f) for %s: skipping" %
                                  (bgInfo.matchedMSE, bgInfo.diffImVar,
                                   tempExpRef.dataId,))
                    continue
                elif (varianceRatio > self.config.maxMatchResidualRatio):
                    self.log.info("Bad fit. MSE/Var ratio %.2f > %.2f for %s: skipping" % (
                            varianceRatio, self.config.maxMatchResidualRatio, tempExpRef.dataId,))
                    continue
                elif ( bgInfo.fitRMS > self.config.maxMatchResidualRMS):
                    self.log.info("Bad fit. RMS %.2f > %.2f for %s: skipping" % (
                            bgInfo.fitRMS, self.config.maxMatchResidualRMS, tempExpRef.dataId,))
                    continue
            newWeightList.append(1 / (1 / weight + bgInfo.fitRMS**2))
            newTempExpRefList.append(tempExpRef)
            newBackgroundStructList.append(bgInfo)
            newScaleList.append(scaler)

        return pipeBase.Struct(tempExpRefList=newTempExpRefList, weightList=newWeightList,
                               imageScalerList=newScaleList, backgroundInfoList=newBackgroundStructList)

    def assemble(self, skyInfo, tempExpRefList, imageScalerList, weightList, bgInfoList=None, 
                 altMaskList=None, doClip=False, mask=None):
        """Assemble a coadd from input warps

        The assembly is performed over small areas on the image at a time, to
        conserve memory usage.

        @param skyInfo: Patch geometry information, from getSkyInfo
        @param tempExpRefList: List of data references to tempExp
        @param imageScalerList: List of image scalers
        @param weightList: List of weights
        @param bgInfoList: List of background data from background matching, or None
        @param altMaskList: List of alternate masks to use rather than those stored with tempExp, or None
        @param doClip: Use clipping when codding?
        @param mask: Mask to ignore when coadding
        @return coadded exposure
        """
        tempExpName = self.getTempExpDatasetName()
        self.log.info("Assembling %s %s" % (len(tempExpRefList), tempExpName))
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
        coaddMaskedImage = coaddExposure.getMaskedImage()
        subregionSizeArr = self.config.subregionSize
        subregionSize = afwGeom.Extent2I(subregionSizeArr[0], subregionSizeArr[1])
        for subBBox in _subBBoxIter(skyInfo.bbox, subregionSize):
            try:
                self.assembleSubregion(coaddExposure, subBBox, tempExpRefList, imageScalerList,
                                       weightList, bgInfoList, altMaskList, statsFlags, statsCtrl)
            except Exception as e:
                self.log.fatal("Cannot compute coadd %s: %s" % (subBBox, e,))

        coaddUtils.setCoaddEdgeBits(coaddMaskedImage.getMask(), coaddMaskedImage.getVariance())

        return coaddExposure

    def assembleMetadata(self, coaddExposure, tempExpRefList, weightList):
        """Set the metadata for the coadd

        This basic implementation simply sets the filter from the
        first input.

        @param coaddExposure: The target image for the coadd
        @param tempExpRefList: List of data references to tempExp
        @param weightList: List of weights
        """
        tempExpName = self.getTempExpDatasetName()
        # We load a single pixel of each coaddTempExp, because we just want to get at the metadata
        # (and we need more than just the PropertySet that contains the header).  See #2777.
        bbox = afwGeom.Box2I(afwGeom.Point2I(0,0), afwGeom.Extent2I(1,1))
        first = True
        coaddInputs = coaddExposure.getInfo().getCoaddInputs()
        for tempExpRef, weight in zip(tempExpRefList, weightList):
            tempExp = tempExpRef.get(tempExpName + "_sub", bbox=bbox, imageOrigin="LOCAL", immediate=True)
            if first:
                coaddExposure.setFilter(tempExp.getFilter())
                first = False
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

    def assembleSubregion(self, coaddExposure, bbox, tempExpRefList, imageScalerList, weightList,
                          bgInfoList, altMaskList, statsFlags, statsCtrl):
        """Assemble the coadd for a sub-region

        @param coaddExposure: The target image for the coadd
        @param bbox: Sub-region to coadd
        @param tempExpRefList: List of data reference to tempExp
        @param imageScalerList: List of image scalers
        @param weightList: List of weights
        @param bgInfoList: List of background data from background matching
        @param altMaskList: List of alternate masks to use rather than those stored with tempExp, or None
        @param statsFlags: Statistic for coadd
        @param statsCtrl: Statistics control object for coadd
        """
        self.log.logdebug("Computing coadd over %s" % bbox)
        tempExpName = self.getTempExpDatasetName()
        coaddMaskedImage = coaddExposure.getMaskedImage()
        maskedImageList = afwImage.vectorMaskedImageF() # [] is rejected by afwMath.statisticsStack
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

            maskedImageList.append(maskedImage)

        with self.timer("stack"):
            coaddSubregion = afwMath.statisticsStack(
                maskedImageList, statsFlags, statsCtrl, weightList)

        coaddMaskedImage.assign(coaddSubregion, bbox)


    def addBackgroundMatchingMetadata(self, coaddExposure, tempExpRefList, backgroundInfoList):
        """Add metadata from the background matching to the coadd

        @param coaddExposure: Coadd
        @param backgroundInfoList: List of background info, results from background matching
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
    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", cls.ConfigClass().coaddName + "Coadd_tempExp",
                               help="data ID, e.g. --id tract=12345 patch=1,2",
                               ContainerClass=AssembleCoaddDataIdContainer)
        parser.add_id_argument("--selectId", "calexp", help="data ID, e.g. --selectId visit=6789 ccd=0..9",
                               ContainerClass=SelectDataIdContainer)
        return parser



def _subBBoxIter(bbox, subregionSize):
    """Iterate over subregions of a bbox

    @param[in] bbox: bounding box over which to iterate: afwGeom.Box2I
    @param[in] subregionSize: size of sub-bboxes

    @return subBBox: next sub-bounding box of size subregionSize or smaller;
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



class AssembleCoaddDataIdContainer(pipeBase.DataIdContainer):
    """A version of lsst.pipe.base.DataIdContainer specialized for assembleCoadd.
    """
    def makeDataRefList(self, namespace):
        """Make self.refList from self.idList.

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
    """Function to count the number of pixels with a specific mask in a footprint
    """
    bbox = footprint.getBBox()
    bbox.clip(mask.getBBox(afwImage.PARENT))
    fp = afwImage.MaskU(bbox)
    subMask = mask.Factory(mask, bbox, afwImage.PARENT)
    afwDet.setMaskFromFootprint(fp, footprint, bitmask)
    return numpy.logical_and((subMask.getArray() & fp.getArray()) > 0,
                             (subMask.getArray() & ignoreMask) == 0).sum()


class SafeClipAssembleCoaddConfig(AssembleCoaddConfig):
    clipDetection = pexConfig.ConfigurableField(target=SourceDetectionTask,
                                      doc="Detect sources on difference between unclipped and clipped coadd")
    minClipFootOverlap = pexConfig.Field(
        doc = "Minimum fractional overlap of clipped footprint with visit DETECTED to be clipped",
        dtype = float,
        default = 0.65
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
        pexConfig.Config.setDefaults(self)
        self.clipDetection.reEstimateBackground = False
        self.clipDetection.returnOriginalFootprints = False
        self.clipDetection.thresholdPolarity = "both"
        self.clipDetection.thresholdValue = 2
        self.clipDetection.nSigmaToGrow = 4
        self.clipDetection.minPixels = 4
        self.clipDetection.isotropicGrow = True
        self.clipDetection.thresholdType = "pixel_stdev"
        self.sigmaClip = 1.5
        self.clipIter = 3


class SafeClipAssembleCoaddTask(AssembleCoaddTask):
    """Assemble a coadd, being careful to clip areas with potential artifacts

    This is done in a number of steps:
        - identify potential regions that need to be clipped
        - find how much overlap there is between each potential region and the DETECTED mask for each visit
        - apply a cut based on the visit overlaps to determine the clipped regions
        - for big areas do not use the region as determined from the coadd, but rather all overlapping
          footprints from the visit DETECTED plane.  This helps in flagging low surface brightness regions.
        - assemble the coadd, ignoring the clipped regions.

    The cuts used to determine the clipped regions were found empirically by examining a lot of HSC data.
    It is unlikely that they will generalize well to other data sets.  The current algorithm is limited to
    identifying artifacts that are found in only one or two visits.  A different threshold is applied
    depending on the number overlapping visits.  The current thresholds work well at removing a majority
    of artifacts.

    """
    ConfigClass = SafeClipAssembleCoaddConfig
    _DefaultName = "safeClipAssembleCoadd"

    def __init__(self, *args, **kwargs):
        AssembleCoaddTask.__init__(self, *args, **kwargs)
        schema = afwTable.SourceTable.makeMinimalSchema()
        self.makeSubtask("clipDetection", schema=schema)

    def assemble(self, skyInfo, tempExpRefList, imageScalerList, weightList, bgModelList, *args, **kwargs):
        """Assemble the coadd for a region

        Identify clipped regions by detecting objects on the difference between unclipped and clipped coadd
        and then flag these regions on the individual vists so they are ignored in the coaddition process

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

        self.log.info('Found %d clipped objects' % len(result.clipFootprints))

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
        coaddExp = AssembleCoaddTask.assemble(self, skyInfo, tempExpRefList, imageScalerList, weightList,
                                              bgModelList, result.tempExpClipList,
                                              doClip=False,
                                              mask=badPixelMask)

        # Set the coadd CLIPPED mask from the footprints since currently pixels that are masked
        # do not get propagated
        maskExp = coaddExp.getMaskedImage().getMask()
        maskExp |= maskClip

        return coaddExp

    def buildDifferenceImage(self, skyInfo, tempExpRefList, imageScalerList, weightList, bgModelList):
        """Return an exposure that contains the difference between and unclipped and clipped coadd

        @param skyInfo: Patch geometry information, from getSkyInfo
        @param tempExpRefList: List of data reference to tempExp
        @param imageScalerList: List of image scalers
        @param weightList: List of weights
        @param bgModelList: List of background models from background matching
        @return Difference image of unclipped and clipped coadd wrapped in an Exposure
        """
        # Build the unclipped coadd
        coaddMean = AssembleCoaddTask.assemble(self, skyInfo, tempExpRefList, imageScalerList, weightList,
                                               bgModelList, doClip=False)

        # Build the clipped coadd
        coaddClip = AssembleCoaddTask.assemble(self, skyInfo, tempExpRefList, imageScalerList, weightList,
                                               bgModelList, doClip=True)

        coaddDiff = coaddMean.getMaskedImage().Factory(coaddMean.getMaskedImage())
        coaddDiff -= coaddClip.getMaskedImage()
        exp = afwImage.ExposureF(coaddDiff)
        exp.setPsf(coaddMean.getPsf())
        return exp


    def detectClip(self, exp, tempExpRefList):
        """Detect clipped regions on an exposure and set the mask on the individual tempExp masks

        @param exp: Exposure to run detection on
        @param tempExpRefList: List of data reference to tempExp
        @return struct containg:
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
        self.log.info('Found %d potential clipped objects' % len(footprints.getFootprints()))
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
        """Find footprints from individual tempExp footprints for large footprints

        @param tempExpClipList: List of tempExp masks with clipping information
        @param clipFootprints: List of clipped footprints
        @param clipIndices: List of which entries in tempExpClipList each footprint belongs to
        @param maskClipValue: Mask value of clipped pixels
        @param maskClipValue: Mask value of detected pixels
        @return list of big footprints
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


