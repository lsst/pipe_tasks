from __future__ import absolute_import, division, print_function

#
# LSST Data Management System
# Copyright 2017-2018 University of Washington.
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

import numpy as np
import scipy.ndimage.interpolation
from lsst.afw.coord.refraction import differentialRefraction
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.coadd.utils as coaddUtils
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from .assembleCoadd import AssembleCoaddTask, CompareWarpAssembleCoaddTask, CompareWarpAssembleCoaddConfig

__all__ = ["DcrAssembleCoaddTask", "DcrAssembleCoaddConfig"]


class DcrAssembleCoaddConfig(CompareWarpAssembleCoaddConfig):
    bufferSize = pexConfig.Field(
        dtype=int,
        doc="Number of pixels to grow the subregion bounding box by to account for DCR at the edge.",
        default=5,
    )
    useFFT = pexConfig.Field(
        dtype=bool,
        doc="Use Fourier transforms for the convolution?",
        default=False,
    )
    usePsf = pexConfig.Field(
        dtype=bool,
        doc="Convolve models with the PSF of the exposures? Requires `useFFT=True`.",
        default=False,
    )
    dcrNumSubfilters = pexConfig.Field(
        dtype=int,
        doc="Number of sub-filters to forward model chromatic effects to fit the supplied exposures.",
        default=3,
    )
    maxNumIter = pexConfig.Field(
        dtype=int,
        doc="Maximum number of iterations of forward modeling.",
        default=8,
    )
    minNumIter = pexConfig.Field(
        dtype=int,
        doc="Minimum number of iterations of forward modeling.",
        default=3,
    )
    convergenceThreshold = pexConfig.Field(
        dtype=float,
        doc="Target relative change in convergence between iterations of forward modeling.",
        default=0.001,
    )
    useConvergence = pexConfig.Field(
        dtype=bool,
        doc="Use convergence test as a forward modeling end condition.",
        default=True,
    )
    doWeightGain = pexConfig.Field(
        dtype=bool,
        doc="Use the calculated convergence metric to accelerate forward modeling?",
        default=True,
    )
    doAirmassWeight = pexConfig.Field(
        dtype=bool,
        doc="Weight exposures by airmass? Useful if there are relatively few high-airmass observations.",
        default=True,
    )
    modelClampFactor = pexConfig.Field(
        dtype=float,
        doc="Maximum relative change of the model allowed between iterations.",
        default=2.,
    )
    regularizeSigma = pexConfig.Field(
        dtype=float,
        doc="Threshold to exclude noise-like pixels from frequency regularization.",
        default=3.,
    )
    clampFrequency = pexConfig.Field(
        dtype=float,
        doc="Maximum relative change of the model allowed between subfilters.",
        default=2.,
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
        self.assembleStaticSkyModel.warpType = 'direct'
        self.warpType = 'direct'
        self.statistic = 'MEAN'
        if self.usePsf:
            self.useFFT = True
        if self.doWeightGain:
            self.useConvergence = True


class DcrAssembleCoaddTask(CompareWarpAssembleCoaddTask):

    ConfigClass = DcrAssembleCoaddConfig
    _DefaultName = "dcrAssembleCoadd"

    @pipeBase.timeMethod
    def run(self, dataRef, selectDataList=[], tempExpRefList=None):
        """Assemble a coadd from a set of Warps.

        Coadd a set of Warps. Compute weights to be applied to each Warp and find scalings to
        match the photometric zeropoint to a reference Warp. Optionally, match backgrounds across
        Warps if the background has not already been removed. Assemble the Warps using
        \ref assemble. Interpolate over NaNs and optionally write the coadd to disk. Return the coadded
        exposure.

        Parameters
        ----------
        dataRef : lsst.daf.persistence.Butler.dataRef
            Data reference defining the patch for coaddition and the reference Warp
        selectDataList : list, optional
            List of data references to Warps. Data to be coadded will be
            selected from this list based on overlap with the patch defined by dataRef.
        tempExpRefList : List of data references to tempExp, optional
            If provided, the data references for the warped temporary exposures to coadd,
            otherwise constructed from the parsed command-line arguments.

        Returns
        -------
        pipeBase.Struct
            The Struct contains the following fields:
            - coaddExposure: coadded exposure
            - nImage: exposure count image
        """
        results = AssembleCoaddTask.run(self, dataRef, selectDataList=selectDataList,
                                        tempExpRefList=tempExpRefList)
        for subfilter, coaddExposure in enumerate(results.dcrCoadds):
            self.processResults(coaddExposure, dataRef)
            if self.config.doWrite:
                self.log.info("Persisting dcrCoadd")
                dataRef.put(coaddExposure, "dcrCoadd", subfilter=subfilter)
        return pipeBase.Struct(coaddExposure=results.coaddExposure, nImage=results.nImage,
                               dcrCoadds=results.dcrCoadds)

    def assemble(self, skyInfo, tempExpRefList, imageScalerList, weightList,
                 supplementaryData=None):
        """Assemble the coadd.

        Requires additional inputs Struct `supplementaryData` to contain a `templateCoadd` that serves
        as the model of the static sky.

        Find artifacts and apply them to the warps' masks creating a list of alternative masks with a
        new "CLIPPED" plane and updated "NO_DATA" plane.
        Then pass these alternative masks to the base class's assemble method.

        Parameters
        ----------
        skyInfo : lsst.skymap.discreteSkyMap.DiscreteSkyMap
            Patch geometry information, from getSkyInfo
        tempExpRefList : List of ButlerDataRefs
            The data references to the input warped exposures.
        imageScalerList : List of image scalers
            The image scalars correct for the zero point of the exposures.
        weightList : list of floats
            The weight to give each input exposure in the coadd
        supplementaryData : PipeBase.Struct
            A templateCoadd constructed from the full band.

        Returns
        -------
        pipeBase.Struct
            The Struct contains the following fields:
            - coaddExposure: coadded exposure
            - nImage: exposure count image
            - dcrCoadds: list of coadded exposures for each subfilter
        """
        templateCoadd = supplementaryData.templateCoadd
        spanSetMaskList = self.findArtifacts(templateCoadd, tempExpRefList, imageScalerList)
        badMaskPlanes = self.config.badMaskPlanes[:]
        badMaskPlanes.append("CLIPPED")
        badPixelMask = afwImage.Mask.getPlaneBitMask(badMaskPlanes)
        subBandImages = self.dcrDivideCoadd(templateCoadd, self.config.dcrNumSubfilters)
        # Propagate PSF-matched EDGE pixels to coadd SENSOR_EDGE and INEXACT_PSF
        # Psf-Matching moves the real edge inwards
        self.applyAltEdgeMask(templateCoadd.mask, spanSetMaskList)

        stats = self.prepareStats(mask=badPixelMask)
        if self.config.doAirmassWeight:
            tempExpName = self.getTempExpDatasetName(self.warpType)
            for visit, tempExpRef in enumerate(tempExpRefList):
                visitInfo = tempExpRef.get(tempExpName + "_visitInfo")
                weightList[visit] *= visitInfo.getBoresightAirmass()

        subregionSize = afwGeom.Extent2I(*self.config.subregionSize)

        baseMask = templateCoadd.mask
        self.filterInfo = templateCoadd.getFilter()
        if np.isnan(self.filterInfo.getFilterProperty().getLambdaMin()):
            raise NotImplementedError("No minimum/maximum wavelength information found"
                                      " in the filter definition! Please add lambdaMin and lambdaMax"
                                      " to the Mapper class in your obs package.")
        for subBBox in self._subBBoxIter(skyInfo.bbox, subregionSize):
            modelIter = 0
            convergenceMetric = self.calculateConvergence(subBandImages, subBBox, tempExpRefList,
                                                          imageScalerList, weightList, spanSetMaskList,
                                                          stats.flags, stats.ctrl)
            self.log.info("Computing coadd over %s", subBBox)
            self.log.info("Initial convergence : %s", convergenceMetric)
            convergenceList = [convergenceMetric]
            convergenceCheck = 1.
            self.subfilterVariance = None
            while (convergenceCheck > self.config.convergenceThreshold or
                   modelIter < self.config.minNumIter):
                try:
                    self.dcrAssembleSubregion(subBandImages, subBBox, tempExpRefList, imageScalerList,
                                              weightList, spanSetMaskList, stats.flags, stats.ctrl,
                                              convergenceMetric, baseMask)
                    if self.config.useConvergence:
                        convergenceMetric = self.calculateConvergence(subBandImages, subBBox, tempExpRefList,
                                                                      imageScalerList, weightList,
                                                                      spanSetMaskList,
                                                                      stats.flags, stats.ctrl)
                        convergenceCheck = (convergenceList[-1] - convergenceMetric)/convergenceMetric
                        convergenceList.append(convergenceMetric)
                except Exception as e:
                    self.log.fatal("Cannot compute coadd %s: %s", subBBox, e.args[0])
                    break
                if modelIter > self.config.maxNumIter:
                    if self.config.useConvergence:
                        self.log.warn("Coadd %s reached maximum iterations before reaching"
                                      " desired convergence improvement of %s.",
                                      " Final convergence improvement: %s",
                                      subBBox, self.config.convergenceThreshold, convergenceCheck)
                    break

                if self.config.useConvergence:
                    self.log.info("Iteration %s with convergence metric %s, %2.4f%% improvement",
                                  modelIter, convergenceMetric, 100.*convergenceCheck)
                modelIter += 1
            else:
                if self.config.useConvergence:
                    self.log.info("Coadd %s finished with convergence metric %s after %s iterations",
                                  subBBox, convergenceMetric, modelIter)
                else:
                    self.log.info("Coadd %s finished after %s iterations", subBBox, modelIter)
            if self.config.useConvergence:
                self.log.info("Final convergence improvement was %2.4f%% overall",
                              100*(convergenceList[0] - convergenceMetric)/convergenceMetric)
        dcrCoadds = self.fillCoadd(subBandImages, skyInfo, tempExpRefList, weightList)
        coaddExposure = self.stackCoadd(dcrCoadds)
        return pipeBase.Struct(coaddExposure=coaddExposure, nImage=None, dcrCoadds=dcrCoadds)

    def dcrAssembleSubregion(self, dcrModels, bbox, tempExpRefList, imageScalerList, weightList,
                             spanSetMaskList, statsFlags, statsCtrl, convergenceMetric, baseMask):
        """Assemble the DCR coadd for a sub-region.

        Build a DCR-matched template for each input exposure, then shift the residuals according to the DCR
        in each subfilter.
        Stack the shifted residuals and apply them as a correction to the solution
        from the previous iteration.
        Restrict the new model solutions from varying by more than a factor of `modelClampFactor`
        from the last solution, and additionally restrict the individual subfilter models
        from varying by more than a factor of `clampFrequency` from their average.
        Finally, mitigate potentially oscillating solutions by averaging the new solution with the solution
        from the previous iteration, weighted by their convergence metric.

        Parameters
        ----------
        dcrModels : list of lsst.afw.image.maskedImageF
            A list of masked images, each containing the model for one subfilter.
        bbox : lsst.afw.geom.box.Box2I
            Bounding box of the subregion to coadd.
        tempExpRefList : List of ButlerDataRefs
            The data references to the input warped exposures.
        imageScalerList : List of image scalers
            The image scalars correct for the zero point of the exposures.
        weightList : list of floats
            The weight to give each input exposure in the coadd
        spanSetMaskList : List of Dicts containing spanSet lists, or None
            Each element is dict with keys = mask plane name to which to add the spans
        statsFlags : lsst.afw.math.Property
            Statistics settings for coaddition.
        statsCtrl : lsst.afw.math.StatisticsControl
            Statistics control object for coadd
        convergenceMetric : `float`
            Quality of fit metric for the matched templates of the input images.
        baseMask : lsst.afw.image.mask
            Mask of the initial template coadd.
        """
        bboxGrow = afwGeom.Box2I(bbox)
        bboxGrow.grow(self.config.bufferSize)
        for model in dcrModels:
            bboxGrow.clip(model.getBBox())
        tempExpName = self.getTempExpDatasetName(self.warpType)
        residualGeneratorList = []
        maskMap = self.setRejectedMaskMapping(statsCtrl)
        clipped = afwImage.Mask.getPlaneBitMask("CLIPPED")

        for tempExpRef, imageScaler, altMaskSpans in zip(tempExpRefList, imageScalerList, spanSetMaskList):
            exposure = tempExpRef.get(tempExpName + "_sub", bbox=bboxGrow)
            visitInfo = exposure.getInfo().getVisitInfo()
            wcs = exposure.getInfo().getWcs()
            maskedImage = exposure.maskedImage
            templateImage = self.buildMatchedTemplate(dcrModels, visitInfo, bboxGrow, wcs, mask=baseMask)
            imageScaler.scaleMaskedImage(maskedImage)
            if altMaskSpans is not None:
                self.applyAltMaskPlanes(maskedImage.mask, altMaskSpans)

            if self.config.removeMaskPlanes:
                for maskPlane in self.config.removeMaskPlanes:
                    try:
                        maskedImage.mask &= ~maskedImage.mask.getPlaneBitMask(maskPlane)
                    except Exception as e:
                        self.log.warn("Unable to remove mask plane %s: %s", maskPlane, e.args[0])
            maskedImage -= templateImage
            residualGeneratorList.append(self.dcrResiduals(dcrModels, maskedImage, visitInfo, bboxGrow, wcs))
        dcrSubModelOut = []
        with self.timer("stack"):
            for oldModel in dcrModels:
                residualsList = [next(residualGenerator) for residualGenerator in residualGeneratorList]
                residual = afwMath.statisticsStack(residualsList, statsFlags, statsCtrl, weightList,
                                                   clipped,  # also set output to CLIPPED if sigma-clipped
                                                   maskMap)
                residual.setXY0(bboxGrow.getBegin())
                newModel = self.clampModel(residual, oldModel, bboxGrow)
                dcrSubModelOut.append(newModel)
        self.setModelVariance(dcrSubModelOut)
        self.regularizeModel(dcrSubModelOut, bboxGrow, baseMask, statsCtrl)
        if self.config.doWeightGain:
            convergenceMetricNew = self.calculateConvergence(dcrSubModelOut, bbox, tempExpRefList,
                                                             imageScalerList, weightList, spanSetMaskList,
                                                             statsFlags, statsCtrl)
            gain = min(max(convergenceMetric/convergenceMetricNew, self.config.minGain), self.config.maxGain)
            self.log.info("Convergence-weighted gain used: %2.4f", gain)
            self.log.info("Based on old convergence: %2.6f, new convergence: %2.6f",
                          convergenceMetric, convergenceMetricNew)
        else:
            gain = 1.
        self.conditionDcrModel(dcrModels, dcrSubModelOut, bboxGrow, gain=gain)

        for model, subModel in zip(dcrModels, dcrSubModelOut):
            model.assign(subModel[bbox, afwImage.PARENT], bbox)

    def clampModel(self, residual, oldModel, bbox):
        """Restrict large variations in the model between iterations.

        Parameters
        ----------
        residual : lsst.afw.image.maskedImageF
            Stacked residual masked image after subtracting DCR-matched templates.
            To save memory, the residual is modified in-place.
        oldModel : lsst.afw.image.maskedImageF
            Description
        bbox : lsst.afw.geom.box.Box2I
            Sub-region to coadd

        Returns
        -------
        lsst.afw.image.maskedImageF
            The sum of the oldModel and residual, with extreme values clipped.
        """
        newModel = residual
        newModel += oldModel[bbox, afwImage.PARENT]
        newImage = newModel.image.array
        newVariance = newModel.variance.array
        nonFinitePixels = ~(np.isfinite(newImage) | np.isfinite(newVariance))
        newModel.mask.array[nonFinitePixels] = newModel.mask.getPlaneBitMask("NO_DATA")
        newImage[nonFinitePixels] = 0.
        newVariance[nonFinitePixels] = 0.
        oldImage = oldModel[bbox, afwImage.PARENT].image.array
        highPixels = np.abs(newImage) > np.abs(oldImage*self.config.modelClampFactor)
        newImage[highPixels] = oldImage[highPixels]*self.config.modelClampFactor
        lowPixels = np.abs(newImage) < np.abs(oldImage/self.config.modelClampFactor)
        newImage[lowPixels] = oldImage[lowPixels]/self.config.modelClampFactor
        return newModel

    def setModelVariance(self, dcrModels):
        """Set the subfilter variance planes from the results of the first iteration.

        We are not solving for the variance, so we need to shift the variance plane
        only once. Otherwise, Otherwise, regions with high variance will bleed into
        neighboring pixels with each successive iteration.

        Parameters
        ----------
        dcrModels : list of lsst.afw.image.maskedImageF
            A list of masked images, each containing the model for one subfilter.
        """
        if self.subfilterVariance is None:
            self.subfilterVariance = [mi.variance.array for mi in dcrModels]
        else:
            for mi, variance in zip(dcrModels, self.subfilterVariance):
                mi.variance.array[:] = variance

    def regularizeModel(self, dcrModels, bbox, mask, statsCtrl):
        """Restrict large variations in the model between subfilters.

        Any flux subtracted by the restriction is accumulated from all subfilters,
        and divided evenly to each afterwards in order to preserve total flux.

        Parameters
        ----------
        dcrModels : list of lsst.afw.image.maskedImageF
            A list of masked images, each containing the model for one subfilter.
        bbox : lsst.afw.geom.box.Box2I
            Sub-region to coadd
        mask : lsst.afw.image.mask
            Reference mask to use for all model planes.
        statsCtrl : lsst.afw.math.StatisticsControl
            Statistics control object for coadd
        """
        nModels = len(dcrModels)
        templateImage = np.mean([model[bbox, afwImage.PARENT].image.array
                                 for model in dcrModels], axis=0)
        excess = np.zeros_like(templateImage)
        convergeMask = mask.getPlaneBitMask(self.config.convergenceMaskPlanes)
        backgroundPixels = mask[bbox, afwImage.PARENT].array & (statsCtrl.getAndMask() ^ convergeMask) == 0
        noiseLevel = self.config.regularizeSigma*np.std(templateImage[backgroundPixels])
        for model in dcrModels:
            modelVals = model.image.array
            highPixels = (modelVals > (templateImage*self.config.clampFrequency + noiseLevel))
            excess[highPixels] += modelVals[highPixels] - templateImage[highPixels]*self.config.clampFrequency
            modelVals[highPixels] = templateImage[highPixels]*self.config.clampFrequency
            lowPixels = (modelVals < templateImage/self.config.clampFrequency - noiseLevel)
            excess[lowPixels] += modelVals[lowPixels] - templateImage[lowPixels]/self.config.clampFrequency
            modelVals[lowPixels] = templateImage[lowPixels]/self.config.clampFrequency
        excess /= nModels
        for model in dcrModels:
            model.image.array += excess

    def calculateConvergence(self, dcrModels, bbox, tempExpRefList, imageScalerList,
                             weightList, spanSetMaskList, statsFlags, statsCtrl):
        """Calculate a quality of fit metric for the matched templates of the input images.

        Parameters
        ----------
        dcrModels : list of lsst.afw.image.maskedImageF
            A list of masked images, each containing the model for one subfilter.
        bbox : lsst.afw.geom.box.Box2I
            Sub-region to coadd
        tempExpRefList : List of ButlerDataRefs
            The data references to the input warped exposures.
        imageScalerList : List of image scalers
            The image scalars correct for the zero point of the exposures.
        weightList : list of floats
            The weight to give each input exposure in the coadd
        spanSetMaskList : List of Dicts containing spanSet lists, or None
            Each element is dict with keys = mask plane name to which to add the spans
        statsFlags : lsst.afw.math.Property
            Statistics settings for coaddition.
        statsCtrl : lsst.afw.math.StatisticsControl
            Statistics control object for coadd

        Returns
        -------
        `float`
            Quality of fit metric for all input exposures, within the sub-region.
        """
        maskMap = self.setRejectedMaskMapping(statsCtrl)
        clipped = afwImage.Mask.getPlaneBitMask("CLIPPED")
        tempExpName = self.getTempExpDatasetName(self.warpType)
        modelWeightList = [1.0]*self.config.dcrNumSubfilters
        dcrModelCut = [model[bbox, afwImage.PARENT] for model in dcrModels]
        modelSum = afwMath.statisticsStack(dcrModelCut, statsFlags, statsCtrl, modelWeightList,
                                           clipped,
                                           maskMap)
        significanceImage = np.abs(modelSum.image.array)
        weight = 0
        metric = 0.
        metricList = {}
        zipIterables = zip(tempExpRefList, weightList, imageScalerList, spanSetMaskList)
        for tempExpRef, expWeight, imageScaler, altMaskSpans in zipIterables:
            exposure = tempExpRef.get(tempExpName + "_sub", bbox=bbox)
            imageScaler.scaleMaskedImage(exposure.maskedImage)
            singleMetric = self.calculateSingleConvergence(exposure, dcrModels, significanceImage, statsCtrl,
                                                           altMaskSpans=altMaskSpans)
            metric += singleMetric*expWeight
            metricList[tempExpRef.dataId["visit"]] = singleMetric
            weight += expWeight
        self.log.info("Individual metrics:\n%s", metricList)
        if weight == 0:
            return 1.
        else:
            return metric/weight

    def calculateSingleConvergence(self, exposure, dcrModels, significanceImage, statsCtrl,
                                   altMaskSpans=None):
        """Calculate a quality of fit metric for a matched template of a single exposure.

        Parameters
        ----------
        exposure : lsst.afw.image.exposure.exposure.ExposureF
            The input warped exposure to evaluate.
        dcrModels : list of lsst.afw.image.maskedImageF
            A list of masked images, each containing the model for one subfilter.
        significanceImage : `np.ndarray`
            Array of weights for each pixel corresponding to its significance for the convergence calculation.

        Returns
        -------
        `float`
            Quality of fit metric for one exposure, within the sub-region.
        """
        convergeMask = afwImage.Mask.getPlaneBitMask(self.config.convergenceMaskPlanes)
        templateImage = self.buildMatchedTemplate(dcrModels, exposure.getInfo().getVisitInfo(),
                                                  exposure.getBBox(), exposure.getInfo().getWcs())
        diffVals = np.abs(exposure.maskedImage.image.array - templateImage.image.array)*significanceImage
        refVals = np.abs(exposure.maskedImage.image.array)*significanceImage

        finitePixels = np.isfinite(refVals) & np.isfinite(diffVals)
        if altMaskSpans is not None:
            self.applyAltMaskPlanes(exposure.mask, altMaskSpans)
        goodMaskPixels = exposure.mask.array & statsCtrl.getAndMask() == 0
        convergeMaskPixels = exposure.mask.array & convergeMask > 0
        usePixels = finitePixels & goodMaskPixels & convergeMaskPixels
        if np.sum(usePixels) == 0:
            metric = 0.
        metric = np.sum(diffVals[usePixels])/np.sum(refVals[usePixels])
        return metric

    @staticmethod
    def dcrDivideCoadd(coaddExposure, dcrNumSubfilters):
        """Divide a coadd into equal subfilter coadds.

        Parameters
        ----------
        coaddExposure : lsst.afw.image.exposure.exposure.ExposureF
            The target image for the coadd
        dcrNumSubfilters : `int`
            The number of subfilters to divide the coadd into.

        Returns
        -------
        list of lsst.afw.image.maskedImageF
            A list of masked images, each containing the model for one subfilter.
        """
        maskedImage = coaddExposure.maskedImage.clone()
        # NANs will potentially contaminate the entire image, depending on the shift or convolution type used.
        badPixels = np.isnan(maskedImage.image.array) ^ np.isnan(maskedImage.variance.array)
        maskedImage.image.array[badPixels] = 0.
        maskedImage.variance.array[badPixels] = 0.
        maskedImage.image.array /= dcrNumSubfilters
        maskedImage.variance.array /= dcrNumSubfilters
        maskedImage.mask.array[badPixels] = maskedImage.mask.getPlaneBitMask("NO_DATA")
        dcrModels = [maskedImage, ]
        for subfilter in range(1, dcrNumSubfilters):
            dcrModels.append(maskedImage.clone())
        return dcrModels

    @staticmethod
    def stackCoadd(dcrCoadds):
        """Add a list of sub-band coadds together.

        Parameters
        ----------
        dcrCoadds : list of lsst.afw.image.exposure.exposure.ExposureF
            A list of coadd exposures, each exposure containing
            the model for one subfilter.

        Returns
        -------
        lsst.afw.image.exposure.exposure.ExposureF
            A single coadd exposure that is the sum of the sub-bands.
        """
        coaddExposure = dcrCoadds[0].clone()
        for coadd in dcrCoadds[1:]:
            coaddExposure.maskedImage += coadd.maskedImage
        return coaddExposure

    def fillCoadd(self, dcrModels, skyInfo, tempExpRefList, weightList):
        """Create a list of coadd exposures from a list of masked images.

        Parameters
        ----------
        dcrModels : list of lsst.afw.image.maskedImageF
            A list of masked images, each containing the model for one subfilter.
        skyInfo : lsst.skymap.discreteSkyMap.DiscreteSkyMap
            Patch geometry information, from getSkyInfo
        tempExpRefList : List of ButlerDataRefs
            The data references to the input warped exposures.
        weightList : list of floats
            The weight to give each input exposure in the coadd

        Returns
        -------
        list of lsst.afw.image.exposure.exposure.ExposureF
            A list of coadd exposures, each exposure containing
            the model for one subfilter.
        """
        dcrCoadds = []
        for model in dcrModels:
            coaddExposure = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
            coaddExposure.setCalib(self.scaleZeroPoint.getCalib())
            coaddExposure.getInfo().setCoaddInputs(self.inputRecorder.makeCoaddInputs())
            self.assembleMetadata(coaddExposure, tempExpRefList, weightList)
            coaddUtils.setCoaddEdgeBits(model.mask, model.variance)
            coaddExposure.setMaskedImage(model)
            dcrCoadds.append(coaddExposure)
        return dcrCoadds

    def convolveDcrModelPlane(self, maskedImage, dcr, bbox=None, useInverse=False):
        """Shift a masked image.

        Parameters
        ----------
        maskedImage : lsst.afw.image.maskedImageF
            The input masked image to shift.
        dcr : lsst.afw.geom.Extent2I
            Shift calculated with `dcrShiftCalculate`.
        bbox : lsst.afw.geom.box.Box2I, optional
            Sub-region of the masked image to shift. Shifts the entire image if None.
        useFFT : bool, optional
            Perform the convolution with an FFT?
        useInverse : bool, optional
            Use the reverse of `dcr` for the shift.

        Returns
        -------
        lsst.afw.image.maskedImageF
            A masked image, with the pixels within the bounding box shifted.

        Raises
        ------
        NotImplementedError
            The Fourier transform approach has not yet been written.
        """
        if self.config.useFFT:
            raise NotImplementedError("The Fourier transform approach has not yet been written.")
        else:
            if useInverse:
                shift = (-dcr.getY(), -dcr.getX())
            else:
                shift = (dcr.getY(), dcr.getX())
            # Shift each of image, mask, and variance
            if bbox is None:
                result = maskedImage.clone()
            else:
                result = maskedImage[bbox, afwImage.PARENT].clone()
            mask = result.mask
            srcImage = result.image.array
            badImagePixels = np.isnan(srcImage)
            srcVariance = result.variance.array
            badVariancePixels = np.isnan(srcVariance)
            mask.array[badImagePixels] = mask.getPlaneBitMask("NO_DATA")
            result.setMask(self.shiftMask(mask, dcr, useInverse=useInverse))

            srcImage[badImagePixels] = 0.
            retImage = scipy.ndimage.interpolation.shift(srcImage, shift)
            result.image.array[:] = retImage
            srcVariance[badVariancePixels] = 0
            retVariance = scipy.ndimage.interpolation.shift(srcVariance, shift)
            result.variance.array[:] = retVariance
        return result

    @staticmethod
    def conditionDcrModel(oldDcrModels, newDcrModels, bbox, gain=1.):
        """Average the current solution with the solution from the last iteration to reduce oscillations.

        Parameters
        ----------
        oldDcrModels : list of lsst.afw.image.maskedImageF
            The models for each subfilter from the previous iteration.
        newDcrModels : list of lsst.afw.image.maskedImageF
            The models for each subfilter from the current iteration.
        bbox : lsst.afw.geom.box.Box2I
            Sub-region of the coadd
        gain : `float`, optional
            Additional weight to apply to the model from the current iteration.
        """
        for oldModel, newModel in zip(oldDcrModels, newDcrModels):
            # The DcrModels are MaskedImages, which only support in-place operations.
            newModel *= gain
            newModel += oldModel[bbox, afwImage.PARENT]
            newModel.image.array[:] /= 1. + gain
            newModel.variance.array[:] /= 1. + gain

    def dcrShiftCalculate(self, visitInfo, wcs):
        """Calculate the shift in pixels of an exposure due to DCR.

        Parameters
        ----------
        visitInfo : lsst.afw.image.VisitInfo
            Metadata for the exposure.
        wcs : lsst.afw.geom.skyWcs.skyWcs.SkyWcs
            Coordinate system definition (wcs) for the exposure.

        Returns
        -------
        lsst.afw.geom.Extent2I
            The 2D shift due to DCR, in pixels.
        """
        rotation = self.calculateRotationAngle(visitInfo, wcs)
        dcrShift = []
        lambdaEff = self.filterInfo.getFilterProperty().getLambdaEff()
        for wl0, wl1 in self.wavelengthGenerator():
            # Note that diffRefractAmp can be negative, since it's relative to the midpoint of the full band
            diffRefractAmp0 = differentialRefraction(wl0, lambdaEff,
                                                     elevation=visitInfo.getBoresightAzAlt().getLatitude(),
                                                     observatory=visitInfo.getObservatory(),
                                                     weather=visitInfo.getWeather())
            diffRefractAmp1 = differentialRefraction(wl1, lambdaEff,
                                                     elevation=visitInfo.getBoresightAzAlt().getLatitude(),
                                                     observatory=visitInfo.getObservatory(),
                                                     weather=visitInfo.getWeather())
            diffRefractAmp = (diffRefractAmp0 + diffRefractAmp1)/2.
            diffRefractPix = diffRefractAmp.asArcseconds()/wcs.getPixelScale().asArcseconds()
            dcrShift.append(afwGeom.Extent2D(diffRefractPix*np.cos(rotation.asRadians()),
                                             diffRefractPix*np.sin(rotation.asRadians())))
        return dcrShift

    def buildMatchedTemplate(self, dcrModels, visitInfo, bbox, wcs, mask=None):
        """Create a DCR-matched template for an exposure.

        Parameters
        ----------
        dcrModels : list of lsst.afw.image.maskedImageF
            A list of masked images, each containing the model for one subfilter.
        visitInfo : lsst.afw.image.VisitInfo
            Metadata for the exposure.
        bbox : lsst.afw.geom.box.Box2I
            Sub-region of the coadd
        wcs : lsst.afw.geom.skyWcs.skyWcs.SkyWcs
            Coordinate system definition (wcs) for the exposure.
        mask : lsst.afw.image.mask, optional
            reference mask to use for the template image.

        Returns
        -------
        lsst.afw.image.maskedImageF
            The DCR-matched template
        """
        dcrShift = self.dcrShiftCalculate(visitInfo, wcs)
        templateImage = afwImage.MaskedImageF(bbox)
        for dcr, model in zip(dcrShift, dcrModels):
            templateImage += self.convolveDcrModelPlane(model, dcr, bbox=bbox)
        if mask is not None:
            templateImage.setMask(mask[bbox, afwImage.PARENT])
        return templateImage

    def dcrResiduals(self, dcrModels, residual, visitInfo, bbox, wcs):
        """Prepare a residual image for stacking in each subfilter by applying the reverse DCR shifts.

        Parameters
        ----------
        dcrModels : list of lsst.afw.image.maskedImageF
            A list of masked images, each containing the model for one subfilter.
        residual : lsst.afw.image.maskedImageF
            The residual masked image for one exposure, after subtracting the matched template
        visitInfo : lsst.afw.image.VisitInfo
            Metadata for the exposure.
        bbox : lsst.afw.geom.box.Box2I
            Sub-region of the coadd
        wcs : lsst.afw.geom.skyWcs.skyWcs.SkyWcs
            Coordinate system definition (wcs) for the exposure.

        Yields
        ------
        lsst.afw.image.maskedImageF
            The residual image for the next subfilter, shifted for DCR.
        """
        dcrShift = self.dcrShiftCalculate(visitInfo, wcs)
        for dcr in dcrShift:
            yield self.convolveDcrModelPlane(residual, dcr, bbox=bbox, useInverse=True)

    def calculateRotationAngle(self, visitInfo, wcs):
        """Calculate the sky rotation angle of an exposure.

        Parameters
        ----------
        visitInfo : lsst.afw.image.VisitInfo
            Metadata for the exposure.
        wcs : lsst.afw.geom.skyWcs.skyWcs.SkyWcs
            Coordinate system definition (wcs) for the exposure.

        Returns
        -------
        lsst.afw.geom.Angle
            The rotation of the image axis, East from North
            Equal to the parallactic angle plus any additional rotation of the coordinate system.
            A rotation angle of 0 degrees is defined with North along the +y axis and East along the +x axis.
            A rotation angle of 90 degrees is defined with North along the +x axis and East along the -y axis.
        """
        parAngle = visitInfo.getBoresightParAngle().asRadians()
        cd = wcs.getCdMatrix()
        cdAngle = (np.arctan2(-cd[0, 1], cd[0, 0]) + np.arctan2(cd[1, 0], cd[1, 1]))/2.
        rotAngle = afwGeom.Angle(cdAngle + parAngle)
        return rotAngle

    def shiftMask(self, mask, shift, useInverse=False):
        """Shift a mask and grow each mask plane by one pixel.

        Parameters
        ----------
        mask : lsst.afw.image.mask
            The input mask to shift.
        shift : lsst.afw.geom.Extent2I
            The shift due to DCR calculated with `dcrShiftCalculate`.
        useInverse : `bool`, optional
            Use the reverse of `shift` for the shift.

        Returns
        -------
        lsst.afw.image.mask
            The mask, shifted to account for DCR.
        """
        if useInverse:
            dx0 = -np.ceil(shift.getX())
            dy0 = -np.ceil(shift.getY())
        else:
            dx0 = np.floor(shift.getX())
            dy0 = np.floor(shift.getY())

        bboxFull = mask.getBBox()
        retMask = mask.Factory(bboxFull)

        bufferXSize = np.abs(dx0) + 1
        bufferYSize = np.abs(dy0) + 1
        bboxBase = mask.getBBox()
        bboxBase.grow(afwGeom.Extent2I(-bufferXSize, -bufferYSize))

        for x0 in range(2):
            for y0 in range(2):
                bbox = mask.getBBox()
                bbox.grow(afwGeom.Extent2I(-bufferXSize, -bufferYSize))
                bbox.shift(afwGeom.Extent2I(dx0 + x0, dy0 + y0))
                retMask[bbox, afwImage.PARENT] |= mask[bboxBase, afwImage.PARENT]
        return retMask

    def wavelengthGenerator(self):
        """Iterate over the wavelength endpoints of subfilters.

        Yields
        ------
        Tuple of two `floats`
            The next set of wavelength endpoints for a subfilter, in nm.
        """
        lambdaMin = self.filterInfo.getFilterProperty().getLambdaMin()
        lambdaMax = self.filterInfo.getFilterProperty().getLambdaMax()
        wlStep = (lambdaMax - lambdaMin)/self.config.dcrNumSubfilters
        for wl in np.linspace(lambdaMin, lambdaMax, self.config.dcrNumSubfilters, endpoint=False):
            yield (wl, wl + wlStep)
