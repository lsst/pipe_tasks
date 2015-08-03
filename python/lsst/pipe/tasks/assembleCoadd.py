#!/usr/bin/env python
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
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
import lsst.meas.algorithms as measAlg
from .coaddBase import CoaddBaseTask, SelectDataIdContainer
from .interpImage import InterpImageTask
from .matchBackgrounds import MatchBackgroundsTask
from .scaleZeroPoint import ScaleZeroPointTask
from .coaddHelpers import groupPatchExposures, getGroupDataRef

__all__ = ["AssembleCoaddTask"]

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
    interpFwhm = pexConfig.Field(
        doc = "FWHM of PSF used for interpolation (arcsec)",
        dtype = float,
        default = 1.5,
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
        super(AssembleCoaddConfig, self).setDefaults()
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

        assembleCoaddTask only works on the dataset type 'coaddTempExp', which are 'coadd temp exposures'.
        Each coaddTempExp is the size of a patch and contains data for one run, visit or
        (for a non-mosaic camera it will contain data for a single exposure).

        coaddTempExps, by default, will have backgrounds in them and will require
        config.doMatchBackgrounds = True. However, makeCoaddTempExp.py can optionally create background-
        subtracted coaddTempExps which can be coadded here by setting
        config.doMatchBackgrounds = False.

        @param dataRef: data reference for a coadd patch (of dataType 'Coadd') OR a data reference
        for a coadd temp exposure (of dataType 'Coadd_tempExp') which serves as the reference visit
        if config.doMatchBackgrounds true and config.autoReference false)
        If supplying a coadd patch: Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter")
        Used to access the following data products (depending on the config):
        - [in] self.config.coaddName + "Coadd_tempExp"
        - [out] self.config.coaddName + "Coadd"

        @return: a pipeBase.Struct with fields:
        - coaddExposure: coadd exposure
        """
        skyInfo = self.getSkyInfo(dataRef)
        calExpRefList = self.selectExposures(dataRef, skyInfo, selectDataList=selectDataList)
        if len(calExpRefList) == 0:
            self.log.warn("No exposures to coadd")
            return
        self.log.info("Coadding %d exposures" % len(calExpRefList))

        butler = dataRef.getButler()
        groupData = groupPatchExposures(dataRef, calExpRefList, self.getCoaddDatasetName(),
                                        self.getTempExpDatasetName())
        tempExpRefList = [getGroupDataRef(butler, self.getTempExpDatasetName(), g, groupData.keys) for
                          g in groupData.groups.keys()]
        inputData = self.prepareInputs(tempExpRefList)
        tempExpRefList = inputData.tempExpRefList
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
                                 inputData.backgroundInfoList if self.config.doMatchBackgrounds else None)
        if self.config.doMatchBackgrounds:
            self.addBackgroundMatchingMetadata(coaddExp, inputData.tempExpRefList,
                                               inputData.backgroundInfoList)

        if self.config.doInterp:
            self.interpImage.interpolateOnePlane(coaddExp.getMaskedImage(), "NO_DATA", psf=coaddExp.getPsf())

        if self.config.doWrite:
            self.writeCoaddOutput(dataRef, coaddExp)

        return pipeBase.Struct(coaddExposure=coaddExp)


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
            except Exception, e:
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
        except Exception, e:
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
                except Exception, e:
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

    def assemble(self, skyInfo, tempExpRefList, imageScalerList, weightList, bgInfoList=None):
        """Assemble a coadd from input warps

        The assembly is performed over small areas on the image at a time, to
        conserve memory usage.

        @param skyInfo: Patch geometry information, from getSkyInfo
        @param tempExpRefList: List of data references to tempExp
        @param imageScalerList: List of image scalers
        @param weightList: List of weights
        @param bgInfoList: List of background data from background matching
        @return coadded exposure
        """
        tempExpName = self.getTempExpDatasetName()
        self.log.info("Assembling %s %s" % (len(tempExpRefList), tempExpName))

        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(self.config.sigmaClip)
        statsCtrl.setNumIter(self.config.clipIter)
        statsCtrl.setAndMask(self.getBadPixelMask())
        statsCtrl.setNanSafe(True)
        statsCtrl.setWeighted(True)
        statsCtrl.setCalcErrorFromInputVariance(True)
        for plane, threshold in self.config.maskPropagationThresholds.items():
            bit = afwImage.MaskU.getMaskPlane(plane)
            statsCtrl.setMaskPropagationThreshold(bit, threshold)

        if self.config.doSigmaClip:
            statsFlags = afwMath.MEANCLIP
        else:
            statsFlags = afwMath.MEAN

        if bgInfoList is None:
            bgInfoList = [None]*len(tempExpRefList)

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
                                       weightList, bgInfoList, statsFlags, statsCtrl)
            except Exception, e:
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
        tempExpList = []
        numCcds = 0
        for tempExpRef in tempExpRefList:
            tempExp = tempExpRef.get(tempExpName + "_sub", bbox=bbox, imageOrigin="LOCAL", immediate=True)
            if first:
                coaddExposure.setFilter(tempExp.getFilter())
                first = False
            tempExpList.append(tempExp)
            numCcds += len(tempExp.getInfo().getCoaddInputs().ccds)
        coaddInputs.ccds.reserve(numCcds)
        coaddInputs.visits.reserve(len(tempExpList))
        for tempExp, weight in zip(tempExpList, weightList):
            self.inputRecorder.addVisitToCoadd(coaddInputs, tempExp, weight)
        coaddInputs.visits.sort()
        if self.config.doPsfMatch:
            psf = self.config.modelPsf.apply(coaddExposure.getWcs())
        else:
            psf = measAlg.CoaddPsf(coaddInputs.ccds, coaddExposure.getWcs())
        coaddExposure.setPsf(psf)
        apCorrMap = measAlg.makeCoaddApCorrMap(coaddInputs.ccds, coaddExposure.getBBox(afwImage.PARENT),
                                               coaddExposure.getWcs())
        coaddExposure.getInfo().setApCorrMap(apCorrMap)

    def assembleSubregion(self, coaddExposure, bbox, tempExpRefList, imageScalerList, weightList,
                          bgInfoList, statsFlags, statsCtrl):
        """Assemble the coadd for a sub-region

        @param coaddExposure: The target image for the coadd
        @param bbox: Sub-region to coadd
        @param tempExpRefList: List of data reference to tempExp
        @param imageScalerList: List of image scalers
        @param weightList: List of weights
        @param bgInfoList: List of background data from background matching
        @param statsFlags: Statistic for coadd
        @param statsCtrl: Statistics control object for coadd
        """
        self.log.info("Computing coadd over %s" % bbox)
        tempExpName = self.getTempExpDatasetName()
        coaddMaskedImage = coaddExposure.getMaskedImage()
        coaddView = afwImage.MaskedImageF(coaddMaskedImage, bbox, afwImage.PARENT, False)
        maskedImageList = afwImage.vectorMaskedImageF() # [] is rejected by afwMath.statisticsStack
        for tempExpRef, imageScaler, bgInfo in zip(tempExpRefList, imageScalerList, bgInfoList):
            exposure = tempExpRef.get(tempExpName + "_sub", bbox=bbox, imageOrigin="PARENT")
            maskedImage = exposure.getMaskedImage()
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

        coaddView <<= coaddSubregion


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

