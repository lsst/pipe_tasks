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
import scipy.ndimage.interpolation
from lsst.afw.coord.refraction import differentialRefraction
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.image.utils as afwImageUtils
import lsst.afw.math as afwMath
import lsst.coadd.utils as coaddUtils
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.assembleCoadd import _subBBoxIter
from lsst.pipe.tasks.assembleCoadd import AssembleCoaddTask
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
    minNIter = pexConfig.Field(
        dtype=int,
        doc="Minimum number of iterations of forward modeling.",
        default=3,
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
    clampModel = pexConfig.Field(
        dtype=float,
        doc="Restrict new solutions from changing by more than a factor of `clampModel`.",
        default=2.,
    )
    useNonNegative = pexConfig.Field(
        dtype=bool,
        doc="Force the model to be non-negative.",
        default=False,
    )
    maxGain = pexConfig.Field(
        dtype=float,
        doc="Maximum convergence-weighted gain to apply between forward modeling iterations.",
        default=2.,
    )
    minGain = pexConfig.Field(
        dtype=float,
        doc="Minimum convergence-weighted gain to apply between forward modeling iterations.",
        default=0.5,
    )
    convergenceMaskPlanes = pexConfig.ListField(
        dtype=str,
        default=["DETECTED"],
        doc="Mask planes to use to calculate convergence."
    )

    def setDefaults(self):
        CompareWarpAssembleCoaddConfig.setDefaults(self)
        self.assembleStaticSkyModel.removeMaskPlanes = []
        self.assembleStaticSkyModel.warpType = 'direct'
        self.removeMaskPlanes = []
        self.statistic = 'MEANCLIP'
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
        retStruct = AssembleCoaddTask.run(self, dataRef, selectDataList=selectDataList)
        for subfilter, coaddExposure in enumerate(retStruct.dcrCoadds):
            if self.config.doWrite:
                self.writeDcrCoadd(dataRef, coaddExposure, subfilter)
            
        return pipeBase.Struct(coaddExposure=retStruct.coaddExposure, nImage=retStruct.nImage)

    def writeDcrCoadd(self, dataRef, coaddExposure, subfilter, nImage=None):
            if self.config.doInterp:
                self.interpImage.run(coaddExposure.getMaskedImage(), planeName="NO_DATA")
                # The variance must be positive; work around for DM-3201.
                varArray = coaddExposure.getMaskedImage().getVariance().getArray()
                varArray[:] = numpy.where(varArray > 0, varArray, numpy.inf)

            if self.config.doMaskBrightObjects:
                brightObjectMasks = self.readBrightObjectMasks(dataRef)
                self.setBrightObjectMasks(coaddExposure, dataRef.dataId, brightObjectMasks)

            self.log.info("Persisting dcrCoadd")
            dataRef.put(coaddExposure, "dcrCoadd", subfilter=subfilter)
            if nImage is not None:
                dataRef.put(nImage, self.getCoaddDatasetName(self.warpType)+'_nImage')

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
        templateCoadd = supplementaryData.templateCoadd
        badMaskPlanes = self.config.badMaskPlanes[:]
        badMaskPlanes.append("CLIPPED")
        badPixelMask = afwImage.Mask.getPlaneBitMask(badMaskPlanes)
        subBandImages = self.dcrDivideCoadd(templateCoadd, self.config.dcrNSubbands)

        statsCtrl, statsFlags = self.prepareStats(skyInfo, mask=badPixelMask)

        subregionSizeArr = self.config.subregionSize
        subregionSize = afwGeom.Extent2I(subregionSizeArr[0], subregionSizeArr[1])
        weightList = [1.]*len(tempExpRefList)

        if altMaskList is None:
            altMaskList = [None]*len(tempExpRefList)
        for subBBox in _subBBoxIter(skyInfo.bbox, subregionSize):
            iter = 0
            self.pixelScale = None
            convergenceMetric = self.calculateConvergence(subBandImages, subBBox, tempExpRefList,
                                                          imageScalerList, weightList, altMaskList,
                                                          statsFlags, statsCtrl)
            self.log.info("Computing coadd over %s", subBBox)
            self.log.info("Initial convergence : %s", convergenceMetric)
            convergenceList = [convergenceMetric]
            convergenceCheck = 1.
            while (convergenceCheck > self.config.convergenceThreshold) or (iter < self.config.minNIter):
                try:
                    self.dcrAssembleSubregion(subBandImages, subBBox, tempExpRefList, imageScalerList,
                                              weightList, altMaskList, statsFlags, statsCtrl,
                                              convergenceMetric)
                    convergenceMetric = self.calculateConvergence(subBandImages, subBBox, tempExpRefList,
                                                                  imageScalerList, weightList, altMaskList,
                                                                  statsFlags, statsCtrl)
                    convergenceCheck = (convergenceList[-1] - convergenceMetric)/convergenceMetric
                    convergenceList.append(convergenceMetric)
                except Exception as e:
                    self.log.warn("Error during iteration %s while computing coadd %s: %s", iter, subBBox, e)
                    break
                if iter > self.config.maxNIter:
                    self.log.warn("Coadd %s reached maximum iterations. Convergence: %s",
                                  subBBox, convergenceMetric)
                    break
                self.log.info("Iteration %s with convergence %s, %2.4f%% improvement \n",
                              iter, convergenceMetric, 100.*convergenceCheck)
                iter += 1
            else:
                self.log.info("Coadd %s finished with convergence %s after %s iterations",
                              subBBox, convergenceMetric, iter)
            finalPercent = 100*(convergenceList[0] - convergenceMetric)/convergenceMetric
            self.log.info("Final convergence improvement was %2.4f%% overall", finalPercent)
        dcrCoadds = self.fillCoadd(subBandImages, skyInfo, tempExpRefList, weightList)
        coaddExposure = self.stackCoadd(dcrCoadds)
        return pipeBase.Struct(coaddExposure=coaddExposure, nImage=None, dcrCoadds=dcrCoadds)

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

    def dcrAssembleSubregion(self, dcrModels, bbox, tempExpRefList, imageScalerList, weightList,
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
        bbox_grow = afwGeom.Box2I(bbox)
        bbox_grow.grow(self.config.bufferSize)
        for model in dcrModels:
            bbox_grow.clip(model.getBBox(afwImage.PARENT))
        tempExpName = self.getTempExpDatasetName(self.warpType)
        residualGeneratorList = []
        weightList = []
        convergeMask = afwImage.Mask.getPlaneBitMask(self.config.convergenceMaskPlanes)

        for tempExpRef, imageScaler, altMask in zip(tempExpRefList, imageScalerList, altMaskList):
            exposure = tempExpRef.get(tempExpName + "_sub", bbox=bbox_grow)
            visitInfo = exposure.getInfo().getVisitInfo()
            wcs = exposure.getInfo().getWcs()
            maskedImage = exposure.getMaskedImage()
            templateImage = self.buildMatchedTemplate(dcrModels, visitInfo, statsFlags, statsCtrl, bbox_grow, wcs)
            if exposure.getWcs().pixelScale() != self.pixelScale:
                self.log.warn("Incompatible pixel scale for %s %s", tempExpName, tempExpRef.dataId)
            imageScaler.scaleMaskedImage(maskedImage)
            mask = maskedImage.getMask()
            if altMask is not None:
                self.applyAltMaskPlanes(mask, altMask)

            if self.config.removeMaskPlanes:
                mask = maskedImage.getMask()
                for maskPlane in self.config.removeMaskPlanes:
                    try:
                        mask &= ~mask.getPlaneBitMask(maskPlane)
                    except Exception as e:
                        self.log.warn("Unable to remove mask plane %s: %s", maskPlane, e.message)
            maskedImage -= templateImage
            weightList.append(self.calculateWeight(maskedImage, convergeMask)*1e3)
            residualGeneratorList.append(self.dcrResiduals(dcrModels, maskedImage, visitInfo, bbox_grow, wcs))

        dcrSubModelOut = []
        with self.timer("stack"):
            for oldModel in dcrModels:
                residualsList = [next(residualGenerator) for residualGenerator in residualGeneratorList]
                residual = afwMath.statisticsStack(residualsList, statsFlags, statsCtrl, weightList,
                                                   afwImage.Mask.getPlaneBitMask("CLIPPED"),
                                                   afwImage.Mask.getPlaneBitMask("NO_DATA"))

                newModel = self.clampModel(residual, oldModel, bbox_grow,
                                           useNonNegative=self.config.useNonNegative,
                                           clamp=self.config.clampModel)
                dcrSubModelOut.append(newModel)
        if self.config.doWeightGain:
            convergenceMetricNew = self.calculateConvergence(dcrSubModelOut, bbox, tempExpRefList,
                                                             imageScalerList, weightList, altMaskList,
                                                             statsFlags, statsCtrl)
            gain = convergenceMetric/convergenceMetricNew
            if gain > self.config.maxGain:
                gain = self.config.maxGain
            if gain < self.config.minGain:
                gain = self.config.minGain
            self.log.info("Convergence-weighted gain used: %2.4f", gain)
            self.log.info("Based on old convergence: %2.6f, new convergence: %2.6f",
                          convergenceMetric, convergenceMetricNew)
            convergenceMetric = convergenceMetricNew
        else:
            gain = 1.
        self.conditionDcrModel(dcrModels, dcrSubModelOut, bbox_grow, gain=gain)

        for model, subModel in zip(dcrModels, dcrSubModelOut):
            model.assign(subModel[bbox, afwImage.PARENT], bbox)

    @staticmethod
    def calculateWeight(residual, goodMask):
        residualVals = residual.getImage().getArray()
        finiteInds = (numpy.isfinite(residualVals))
        goodMaskInds = (residual.getMask().getArray() & goodMask) == goodMask
        weight = 1./numpy.std(residualVals[finiteInds*goodMaskInds])
        return weight

    @staticmethod
    def clampModel(residual, oldModel, bbox, useNonNegative=False, clamp=2.):
        newModel = residual
        newModel += oldModel
        newImage = newModel.getImage().getArray()
        newVariance = newModel.getVariance().getArray()
        if useNonNegative:
            negInds = newImage < 0.
            newImage[negInds] = 0.
        oldImage = oldModel[bbox, afwImage.PARENT].getImage().getArray()
        oldVariance = oldModel[bbox, afwImage.PARENT].getVariance().getArray()
        highInds = ((numpy.abs(newImage) > numpy.abs(oldImage*clamp))*
                    (numpy.abs(newVariance) > numpy.abs(oldVariance*clamp)))
        newImage[highInds] = oldImage[highInds]*clamp
        newVariance[highInds] = oldVariance[highInds]*clamp
        lowInds = ((numpy.abs(newImage) < numpy.abs(oldImage/clamp))*
                    (numpy.abs(newVariance) < numpy.abs(oldVariance/clamp)))
        newImage[lowInds] = oldImage[lowInds]/clamp
        newVariance[lowInds] = oldVariance[lowInds]/clamp
        return newModel

    def calculateConvergence(self, dcrModels, bbox, tempExpRefList, imageScalerList,
                             weightList, altMaskList, statsFlags, statsCtrl):
        tempExpName = self.getTempExpDatasetName(self.warpType)
        modelWeightList = [1.0]*self.config.dcrNSubbands
        convergeMask = afwImage.Mask.getPlaneBitMask(self.config.convergenceMaskPlanes)
        dcrModelCut = [model[bbox, afwImage.PARENT] for model in dcrModels]
        modelWeights = afwMath.statisticsStack(dcrModelCut, statsFlags, statsCtrl, modelWeightList,
                                               afwImage.Mask.getPlaneBitMask("CLIPPED"),
                                               afwImage.Mask.getPlaneBitMask("NO_DATA"))
        modelVals = modelWeights.getImage().getArray()
        weight = 0
        metric = 0.
        metricList = []
        zipIterables = zip(tempExpRefList, weightList, imageScalerList, altMaskList)
        len_list = []
        obsid_list = []
        for tempExpRef, expWeight, imageScaler, altMask in zipIterables:
            exposure = tempExpRef.get(tempExpName + "_sub", bbox=bbox)
            if self.pixelScale is None:
                self.pixelScale = exposure.getWcs().pixelScale()
            imageScaler.scaleMaskedImage(refImage)
            refImage = exposure.getMaskedImage()
            refVals = refImage.getImage().getArray()
            visitInfo = exposure.getInfo().getVisitInfo()
            wcs = exposure.getInfo().getWcs()
            templateImage = self.buildMatchedTemplate(dcrModels, visitInfo, statsFlags, statsCtrl, bbox, wcs)
            templateVals = templateImage.getImage().getArray()
            diffVals = numpy.abs(refVals - templateVals)*modelVals
            refVals =  numpy.abs(refVals)*modelVals

            finiteInds = (numpy.isfinite(refVals))*(numpy.isfinite(diffVals))
            goodMaskInds = (refImage.getMask().getArray() & convergeMask) == convergeMask
            inds = finiteInds*goodMaskInds
            len_list.append(numpy.sum(inds))
            if numpy.sum(inds) == 0:
                metricList.append(numpy.nan)
                continue
            singleMetric = numpy.sum(diffVals[inds])/numpy.sum(refVals[inds])
            metric += singleMetric
            metricList.append(singleMetric)
            weight += 1.
            obsid_list.append((visitInfo.getExposureId()-124)//512)
        self.log.info("Indiviual metrics:\n%s", list(zip(obsid_list, metricList)))
        if weight == 0:
            return 1.
        else:
            return metric/weight

    @staticmethod
    def dcrDivideCoadd(coaddExposure, dcrNSubbands):
        dcrModels = [coaddExposure.getMaskedImage().clone() for f in range(dcrNSubbands)]
        for model in dcrModels:
            model.getMask().addMaskPlane("CLIPPED")
            model.getImage().getArray()[:, :] /= dcrNSubbands
            model.getVariance().getArray()[:, :] /= dcrNSubbands
        return dcrModels

    @staticmethod
    def stackCoadd(dcrCoadds):
        coaddGenerator = (coadd for coadd in dcrCoadds)
        coaddExposure = next(coaddGenerator).clone()
        mimage = coaddExposure.getMaskedImage()
        for coadd in coaddGenerator:
            mimage += coadd.getMaskedImage()
        return coaddExposure


    def fillCoadd(self, dcrModels, skyInfo, tempExpRefList, weightList):
        dcrCoadds = []
        for model in dcrModels:
            coaddExposure = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
            coaddExposure.setCalib(self.scaleZeroPoint.getCalib())
            coaddExposure.getInfo().setCoaddInputs(self.inputRecorder.makeCoaddInputs())
            self.assembleMetadata(coaddExposure, tempExpRefList, weightList)
            coaddUtils.setCoaddEdgeBits(model.getMask(), model.getVariance())
            coaddExposure.setMaskedImage(model)
            dcrCoadds.append(coaddExposure)
        return dcrCoadds

    @staticmethod
    def convolveDcrModelPlane(maskedImage, dcr, useFFT=False, useInverse=False):
        if useFFT:
            raise NotImplementedError("The Fourier transform approach has not yet been written.")
        else:
            if useInverse:
                shift = (-dcr.dy, -dcr.dx)
            else:
                shift = (dcr.dy, dcr.dx)
            # Shift each of image, mask, and variance
            result = maskedImage.clone()
            srcImage = result.getImage().getArray()
            srcImage[numpy.isnan(srcImage)] = 0.
            scrVariance = result.getVariance().getArray()
            scrVariance[numpy.isnan(scrVariance)] = 0.
            retImage = scipy.ndimage.interpolation.shift(srcImage, shift)
            result.getImage().getArray()[:, :] = retImage
            retVariance = scipy.ndimage.interpolation.shift(scrVariance, shift)
            result.getVariance().getArray()[:, :] = retVariance
        return result

    @staticmethod
    def conditionDcrModel(oldDcrModels, newDcrModels, bbox, gain=1.):
        for oldModel, newModel in zip(oldDcrModels, newDcrModels):
            # The DcrModels are MaskedImages, which only support in-place operations.
            newModel *= gain
            newModel += oldModel[bbox, afwImage.PARENT]
            newModel /= 1. + gain

    @staticmethod
    def dcrShiftCalculate(visitInfo, wcs, lambdaEff, filterWidth, dcrNSubbands):
        rotation = calculateRotationAngle(visitInfo, wcs)

        dcr = namedtuple("dcr", ["dx", "dy"])
        dcrShift = []
        for wl0, wl1 in wavelengthGenerator(lambdaEff, filterWidth, dcrNSubbands):
            # Note that refract_amp can be negative, since it's relative to the midpoint of the full band
            diffRefractAmp0 = differentialRefraction(wl0, lambdaEff,
                                                    elevation=visitInfo.getBoresightAzAlt().getLatitude(),
                                                    observatory=visitInfo.getObservatory(),
                                                    weather=visitInfo.getWeather())
            diffRefractAmp1 = differentialRefraction(wl1, lambdaEff,
                                                    elevation=visitInfo.getBoresightAzAlt().getLatitude(),
                                                    observatory=visitInfo.getObservatory(),
                                                    weather=visitInfo.getWeather())
            diffRefractAmp = (diffRefractAmp0 + diffRefractAmp1)/2.
            diffRefractPix = diffRefractAmp.asArcseconds()/wcs.pixelScale().asArcseconds()
            dcrShift.append(dcr(dx=diffRefractPix*numpy.cos(rotation.asRadians()),
                                dy=diffRefractPix*numpy.sin(rotation.asRadians())))
        return dcrShift

    def buildMatchedTemplate(self, dcrModels, visitInfo, statsFlags, statsCtrl, bbox, wcs):
        dcrShift = self.dcrShiftCalculate(visitInfo, wcs, self.lambdaEff, self.filterWidth, self.config.dcrNSubbands)
        weightList = [1.0]*self.config.dcrNSubbands
        maskedImageList = [self.convolveDcrModelPlane(model[bbox, afwImage.PARENT],
                                                      dcr, useFFT=self.config.useFFT)
                           for dcr, model in zip(dcrShift, dcrModels)]
        templateImage = afwMath.statisticsStack(maskedImageList, statsFlags, statsCtrl, weightList,
                                               afwImage.Mask.getPlaneBitMask("CLIPPED"),
                                               afwImage.Mask.getPlaneBitMask("NO_DATA"))
        templateImage *= self.config.dcrNSubbands
        return templateImage

    def dcrResiduals(self, dcrModels, residualImageIn, visitInfo, bbox, wcs):
        dcrShift = self.dcrShiftCalculate(visitInfo, wcs, self.lambdaEff, self.filterWidth, self.config.dcrNSubbands)
        for dcr in dcrShift:
            yield self.convolveDcrModelPlane(residualImageIn, dcr, useInverse=True, useFFT=self.config.useFFT)


def wavelengthGenerator(lambdaEff, filterWidth, dcrNSubbands):
    wlStep = filterWidth/dcrNSubbands
    for wl in numpy.linspace(-filterWidth/2., filterWidth/2., dcrNSubbands, endpoint=False):
        wlStart = lambdaEff + wl
        wlEnd = wlStart + wlStep
        yield (wlStart, wlEnd)


def calculateRotationAngle(visitInfo, wcs):
    """Calculate the sky rotation angle of an exposure.

    Parameters
    ----------
    exposure : lsst.afw.image.ExposureD
        An LSST exposure object.

    Returns
    -------
    lsst.afw.geom.Angle
        The rotation of the image axis, East from North.
        A rotation angle of 0 degrees is defined with North along the +y axis and East along the +x axis.
        A rotation angle of 90 degrees is defined with North along the +x axis and East along the -y axis.
    """
    p_angle = visitInfo.getBoresightParAngle().asRadians()
    cd = wcs.getCDMatrix()
    cd_rot = (numpy.arctan2(-cd[0, 1], cd[0, 0]) + numpy.arctan2(cd[1, 0], cd[1, 1]))/2.
    rotation_angle = afwGeom.Angle(cd_rot + p_angle)
    return rotation_angle
