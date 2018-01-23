from __future__ import absolute_import, division, print_function

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

from collections import namedtuple
import numpy
from scipy.ndimage.interpolation import shift as scipyShift
from lsst.afw.coord.refraction import differentialRefraction
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
from lsst.afw.image.maskedImage import MaskedImageF as MaskedImageF
import lsst.afw.image.utils as afwImageUtils
import lsst.afw.math as afwMath
import lsst.coadd.utils as coaddUtils
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.assembleCoadd import _subBBoxIter
from lsst.pipe.tasks.assembleCoadd import AssembleCoaddTask
# from lsst.pipe.tasks.assembleCoadd import AssembleCoaddConfig
from lsst.pipe.tasks.assembleCoadd import CompareWarpAssembleCoaddTask
from lsst.pipe.tasks.assembleCoadd import CompareWarpAssembleCoaddConfig

__all__ = ["DcrAssembleCoaddTask"]


class DcrAssembleCoaddConfig(CompareWarpAssembleCoaddConfig):
    filterName = pexConfig.Field(
        dtype=str,
        doc="Common name of the band-defining filter of th observations.",
        default='g',
    )
    lambdaEff = pexConfig.Field(
        dtype=float,
        doc="Effective wavelength of the filter, in nm.",
        default=0.,
    )
    filterWidth = pexConfig.Field(
        dtype=float,
        doc="FWHM of the filter transmission curve, in nm.",
        default=0.,
    )
    bufferSize = pexConfig.Field(
        dtype=int,
        doc="Number of pixels to grow the subregion bounding box by.",
        default=5,
    )
    useFFT = pexConfig.Field(
        dtype=bool,
        doc="Option to use use Fourier transforms for the convolutions.",
        default=False,
    )
    usePsf = pexConfig.Field(
        dtype=bool,
        doc="Option to use the PSF as part of the convolution; requires `useFFT=True`.",
        default=False,
    )
    dcrNSubbands = pexConfig.Field(
        dtype=int,
        doc="Number of sub-bands to forward model chromatic effects to fit the supplied exposures.",
        default=3,
    )
    maxNIter = pexConfig.Field(
        dtype=int,
        doc="Maximum number of iterations of forward modeling.",
        default=8,
    )
    convergenceThreshold = pexConfig.Field(
        dtype=float,
        doc="Target change in convergence between iteration of forward modeling.",
        default=0.001,
    )
    useConvergence = pexConfig.Field(
        dtype=bool,
        doc="Turn on or off the convergence test as a forward modeling end condition.",
        default=True,
    )
    doWeightGain = pexConfig.Field(
        dtype=bool,
        doc="Use the calculated convergence metric to accelerate forward modeling.",
        default=True,
    )
    assembleStaticSkyModel = pexConfig.ConfigurableField(
        target=AssembleCoaddTask,
        doc="Task to assemble an artifact-free, PSF-matched Coadd to serve as a"
            " naive/first-iteration model of the static sky.",
    )

    def setDefaults(self):
        CompareWarpAssembleCoaddConfig.setDefaults(self)
        self.assembleStaticSkyModel.removeMaskPlanes = []
        self.removeMaskPlanes = []
        if self.usePsf:
            self.useFFT = True
        if self.doWeightGain:
            self.useConvergence = True


class DcrAssembleCoaddTask(CompareWarpAssembleCoaddTask):

    ConfigClass = DcrAssembleCoaddConfig
    _DefaultName = "dcrAssembleCoadd"

    def __init__(self, *args, **kwargs):
        """!
        \brief Initialize the task and make the \ref AssembleCoadd_ "assembleStaticSkyModel" subtask.
        """
        CompareWarpAssembleCoaddTask.__init__(self, *args, **kwargs)

        # Note that we can only call afwImageUtils.Filter after the butler has been initialized.
        self.lambdaEff = afwImageUtils.Filter(self.config.filterName).getFilterProperty().getLambdaEff()
        self.filterWidth = self.lambdaEff*0.2

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

        supplementaryData = self.makeSupplementaryData(dataRef, selectDataList)
        # nImage is created if it is requested by self.config.  Otherwise, None.

        retStructGen = self.assemble(
            skyInfo, inputData.tempExpRefList, inputData.imageScalerList, inputData.weightList,
            supplementaryData=supplementaryData,
        )
        coaddExp = None
        for retStruct in retStructGen:

            if self.config.doInterp:
                self.interpImage.run(retStruct.coaddExposure.getMaskedImage(), planeName="NO_DATA")
                # The variance must be positive; work around for DM-3201.
                varArray = retStruct.coaddExposure.getMaskedImage().getVariance().getArray()
                varArray[:] = numpy.where(varArray > 0, varArray, numpy.inf)

            if self.config.doMaskBrightObjects:
                brightObjectMasks = self.readBrightObjectMasks(dataRef)
                self.setBrightObjectMasks(retStruct.coaddExposure, dataRef.dataId, brightObjectMasks)

            if self.config.doWrite:
                self.log.info("Persisting dcrCoadd")
                dataRef.put(retStruct.coaddExposure, "dcrCoadd", subfilter=retStruct.subFilter)

            if coaddExp is None:
                coaddExp = retStruct.coaddExposure
            else:
                mimage = coaddExp.getMaskedImage()
                mimage += retStruct.coaddExposure.getMaskedImage()
        if self.config.doWrite:
            self.log.info("Persisting %s" % self.getCoaddDatasetName(self.warpType))
            dataRef.put(retStruct.coaddExposure, self.getCoaddDatasetName(self.warpType))
            if retStruct.nImage is not None:
                dataRef.put(retStruct.nImage, self.getCoaddDatasetName(self.warpType)+'_nImage')
        return pipeBase.Struct(coaddExposure=coaddExp, nImage=retStruct.nImage)

    def assemble(self, skyInfo, tempExpRefList, imageScalerList, weightList,
                 altMaskList=None, supplementaryData=None, *args, **kwargs):
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
        @param supplementaryData: PipeBase.Struct containing a templateCoadd

        return coadd exposure
        """
        # templateCoadd = supplementaryData.templateCoadd
        # spanSetMaskList = self.findArtifacts(templateCoadd, tempExpRefList, imageScalerList)
        # maskList = self.computeAltMaskList(tempExpRefList, spanSetMaskList)
        # badMaskPlanes = self.config.badMaskPlanes[:]
        # badMaskPlanes.append("CLIPPED")
        # badPixelMask = afwImage.Mask.getPlaneBitMask(badMaskPlanes)

        # templateCoadd = supplementaryData.coaddExposure
        # spanSetMaskList = CompareWarpAssembleCoaddTask.findArtifacts(self, templateCoadd,
        #                                                              tempExpRefList, imageScalerList)
        # maskList = CompareWarpAssembleCoaddTask.computeAltMaskList(self, tempExpRefList, spanSetMaskList)
        badMaskPlanes = self.config.badMaskPlanes[:]
        badMaskPlanes.append("CLIPPED")
        badPixelMask = afwImage.Mask.getPlaneBitMask(badMaskPlanes)
        subBandImages = self.dcrDivideCoadd(supplementaryData.templateCoadd)

        statsCtrl, statsFlags = self.prepareStats(skyInfo, mask=badPixelMask)

        subregionSizeArr = self.config.subregionSize
        subregionSize = afwGeom.Extent2I(subregionSizeArr[0], subregionSizeArr[1])

        if altMaskList is None:
            altMaskList = [None]*len(tempExpRefList)
        for subBBox in _subBBoxIter(skyInfo.bbox, subregionSize):
            iter = 0
            self.scale = None
            convergenceMetric = self.calculateConvergence(subBandImages, subBBox, tempExpRefList,
                                                          imageScalerList, altMaskList,
                                                          statsFlags, statsCtrl)
            self.log.info("Initial convergence of coadd %s: %s", subBBox, convergenceMetric)
            convergenceList = [convergenceMetric]
            convergenceCheck = convergenceMetric
            data_check = [numpy.std(model[subBBox].getImage().getArray()) for model in subBandImages]
            self.log.info("Deviation of model in coadd %s: %s", subBBox, data_check)
            while convergenceCheck > self.config.convergenceThreshold:
                self.log.info("Iteration %s with convergence %s", iter, convergenceMetric)
                try:
                    self.dcrAssembleSubregion(subBandImages, subBBox, tempExpRefList, imageScalerList,
                                              weightList, altMaskList, statsFlags, statsCtrl,
                                              convergenceMetric)
                    convergenceMetric = self.calculateConvergence(subBandImages, subBBox, tempExpRefList,
                                                                  imageScalerList, altMaskList,
                                                                  statsFlags, statsCtrl)
                    convergenceCheck = convergenceList[-1] - convergenceMetric
                    convergenceList.append(convergenceMetric)
                except Exception as e:
                    self.log.warn("Error during iteration %s while computing coadd %s: %s", iter, subBBox, e)
                    break
                if iter > self.config.maxNIter:
                    self.log.warn("Coadd %s reached maximum iterations. Convergence: %s",
                                  subBBox, convergenceMetric)
                    break
                iter += 1
            else:
                self.log.info("Coadd %s finished with convergence %s after %s iterations",
                              subBBox, convergenceMetric, iter)
        dcrCoadd = self.fillCoadd(subBandImages, skyInfo, tempExpRefList, weightList)
        for subFilter, coadd in enumerate(dcrCoadd):
            yield pipeBase.Struct(coaddExposure=coadd, nImage=None, subFilter=subFilter)

    def prepareStats(self, skyInfo, mask=None):
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

        return (statsCtrl, statsFlags)

    def dcrAssembleSubregion(self, dcrModel, bbox, tempExpRefList, imageScalerList, weightList,
                             altMaskList, statsFlags, statsCtrl, convergenceMetric):
        """!
        \brief Assemble the DCR coadd for a sub-region, .

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
        \param[in] altMaskList: List of alternate masks to use rather than those stored with tempExp, or None
        \param[in] statsFlags: afwMath.Property object for statistic for coadd
        \param[in] statsCtrl: Statistics control object for coadd
        """
        self.log.debug("Computing coadd over %s", bbox)
        bbox_grow = afwGeom.Box2I(bbox)
        bbox_grow.grow(self.config.bufferSize)
        for model in dcrModel:
            bbox_grow.clip(model.getBBox())
        tempExpName = self.getTempExpDatasetName(self.warpType)
        maskedImageList2 = [[] for foo in range(self.config.dcrNSubbands)]
        for tempExpRef, imageScaler, altMask in zip(tempExpRefList, imageScalerList, altMaskList):
            exposure = tempExpRef.get(tempExpName + "_sub", bbox=bbox_grow)
            visitInfo = exposure.getInfo().getVisitInfo()
            maskedImage = exposure.getMaskedImage()
            if exposure.getWcs().pixelScale() != self.scale:
                self.log.warn("Incompatible pixel scale for %s %s", tempExpName, tempExpRef.dataId)

            if altMask:
                altMaskSub = altMask.Factory(altMask, bbox_grow, afwImage.PARENT)
                maskedImage.getMask().swap(altMaskSub)
            imageScaler.scaleMaskedImage(maskedImage)

            if self.config.removeMaskPlanes:
                mask = maskedImage.getMask()
                for maskPlane in self.config.removeMaskPlanes:
                    try:
                        mask &= ~mask.getPlaneBitMask(maskPlane)
                    except Exception as e:
                        self.log.warn("Unable to remove mask plane %s: %s", maskPlane, e.message)
            maskedImageList = self.dcrResiduals(dcrModel, maskedImage, visitInfo, bbox_grow)
            for mi, ml in zip(maskedImageList, maskedImageList2):
                ml.append(mi)

        dcrSubModelOut = []
        with self.timer("stack"):
            for maskedImageList in maskedImageList2:
                model = afwMath.statisticsStack(maskedImageList, statsFlags, statsCtrl, weightList)
                model.setXY0(bbox_grow.getBegin())
                dcrSubModelOut.append(model)
        if self.config.doWeightGain:
            convergenceMetricNew = self.calculateConvergence(dcrSubModelOut, bbox, tempExpRefList,
                                                             imageScalerList, altMaskList,
                                                             statsFlags, statsCtrl)
            gain = convergenceMetric/convergenceMetricNew
            convergenceMetric = convergenceMetricNew
        else:
            gain = 1.
        self.conditionDcrModel(dcrModel, dcrSubModelOut, bbox_grow, gain=gain)

        for model, subModel in zip(dcrModel, dcrSubModelOut):
            model.assign(subModel[bbox, afwImage.PARENT], bbox)

    def calculateConvergence(self, dcrModel, bbox, tempExpRefList, imageScalerList,
                             altMaskList, statsFlags, statsCtrl):
        tempExpName = self.getTempExpDatasetName(self.warpType)
        weightList = [1.0]*self.config.dcrNSubbands
        maskRef = dcrModel[0].getMask()
        dcrModelCut = [model[bbox, afwImage.PARENT] for model in dcrModel]
        modelWeights = afwMath.statisticsStack(dcrModelCut, statsFlags, statsCtrl, weightList,
                                               maskRef.getPlaneBitMask("CLIPPED"),
                                               maskRef.getPlaneBitMask("NO_DATA"))
        count = 0
        metric = 0.
        metricList = []
        for tempExpRef, imageScaler, altMask in zip(tempExpRefList, imageScalerList, altMaskList):
            exposure = tempExpRef.get(tempExpName + "_sub", bbox=bbox)
            if self.scale is None:
                self.scale = exposure.getWcs().pixelScale()
            refImage = exposure.getMaskedImage().clone()
            refImage *= modelWeights
            visitInfo = exposure.getInfo().getVisitInfo()
            diffImage = self.buildMatchedTemplate(dcrModel, visitInfo, statsFlags, statsCtrl, bbox)
            diffImage *= modelWeights
            diffImage -= refImage
            refVals = numpy.abs(refImage.getImage().getArray())
            diffVals = numpy.abs(diffImage.getImage().getArray())
            inds = (numpy.isfinite(refVals))*(numpy.isfinite(diffVals))
            singleMetric = numpy.sum(diffVals[inds])/numpy.sum(refVals[inds])
            metric += singleMetric
            metricList.append(singleMetric)
            count += 1
        self.log.info("Indiviual metrics: %s", metricList)
        return metric/count

    def dcrDivideCoadd(self, coaddExposure):
        dcrModel = [coaddExposure.getMaskedImage().clone() for f in range(self.config.dcrNSubbands)]
        for model in dcrModel:
            model.getImage().getArray()[:, :] /= self.config.dcrNSubbands
            model.getVariance().getArray()[:, :] /= self.config.dcrNSubbands
        return dcrModel

    def fillCoadd(self, dcrModel, skyInfo, tempExpRefList, weightList):
        for model in dcrModel:
            coaddExposure = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
            coaddExposure.setCalib(self.scaleZeroPoint.getCalib())
            coaddExposure.getInfo().setCoaddInputs(self.inputRecorder.makeCoaddInputs())
            self.assembleMetadata(coaddExposure, tempExpRefList, weightList)
            coaddUtils.setCoaddEdgeBits(model.getMask(), model.getVariance())
            coaddExposure.setMaskedImage(model)
            yield coaddExposure

    def convolveDcrModelPlane(self, dcrModelPlane, dcr, useInverse=False):
        if self.config.useFFT:
            raise NotImplementedError("The Fourier transform approach has not yet been written.")
        else:
            if useInverse:
                shift = (-dcr.dy, -dcr.dx)
            else:
                shift = (dcr.dy, dcr.dx)
            if isinstance(dcrModelPlane, MaskedImageF):
                # Shift each of image, mask, and variance if a masked image.
                result = dcrModelPlane.clone()
                for array in result.getArrays():
                    array = scipyShift(array, shift)
            else:
                result = scipyShift(dcrModelPlane, shift)
        return result

    def conditionDcrModel(self, oldDcrModel, newDcrModel, bbox, gain=1.):
        for oldModel, newModel in zip(oldDcrModel, newDcrModel):
            # The DcrModels are MaskedImages, which only support in-place operations.
            newModel *= gain
            newModel += oldModel[bbox, afwImage.PARENT]
            newModel /= 1. + gain
            # newModel = (oldModel[bbox] + gain*newModel)/(1. + gain)

    def dcrShiftCalculate(self, visitInfo):
        rotation = visitInfo.getBoresightParAngle() + visitInfo.getBoresightRotAngle()

        dcr = namedtuple("dcr", ["dx", "dy"])
        for wl in self.wavelengthGenerator():
            # Note that refract_amp can be negative, since it's relative to the midpoint of the full band
            diffRefractAmp = differentialRefraction(wl, self.lambdaEff,
                                                    elevation=visitInfo.getBoresightAzAlt().getLatitude(),
                                                    observatory=visitInfo.getObservatory(),
                                                    weather=visitInfo.getWeather())
            diffRefractPix = diffRefractAmp.asArcseconds()/self.scale.asArcseconds()
            yield dcr(dx=diffRefractPix*numpy.sin(rotation.asRadians()),
                      dy=diffRefractPix*numpy.cos(rotation.asRadians()))

    def buildMatchedTemplate(self, dcrModel, visitInfo, statsFlags, statsCtrl, bbox):
        dcrShift = self.dcrShiftCalculate(visitInfo)
        weightList = [1.0]*self.config.dcrNSubbands
        maskedImageList = [self.convolveDcrModelPlane(model[bbox, afwImage.PARENT], dcr)
                           for dcr, model in zip(dcrShift, dcrModel)]
        maskRef = dcrModel[0].getMask()
        templateVals = afwMath.statisticsStack(maskedImageList, statsFlags, statsCtrl, weightList,
                                               maskRef.getPlaneBitMask("CLIPPED"),
                                               maskRef.getPlaneBitMask("NO_DATA"))
        return templateVals

    def dcrResiduals(self, dcrModel, maskedImage, visitInfo, bbox):
        dcrShift = list(self.dcrShiftCalculate(visitInfo))
        shiftedModels = [self.convolveDcrModelPlane(model[bbox, afwImage.PARENT], dcr, useInverse=False)
                         for dcr, model in zip(dcrShift, dcrModel)]
        residualImages = []
        for f, dcr in enumerate(dcrShift):
            mimage = maskedImage.clone()
            for f2 in range(self.config.dcrNSubbands):
                if f2 != f:
                    mimage -= shiftedModels[f2]
            residual = self.convolveDcrModelPlane(mimage, dcr, useInverse=True)
            residual.setXY0(bbox.getBegin())
            residualImages.append(residual)
        return residualImages

    def wavelengthGenerator(self):
        wlRef = self.lambdaEff
        for wl in numpy.linspace(0., self.filterWidth, self.config.dcrNSubbands, endpoint=True):
            yield wlRef - self.filterWidth/2. + wl
