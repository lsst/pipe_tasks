# This file is part of pipe_tasks.
#
# LSST Data Management System
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
# See COPYRIGHT file at the top of the source tree.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#

import numpy as np
from lsst.afw.coord.refraction import differentialRefraction
import lsst.afw.geom as afwGeom
from lsst.afw.geom import AffineTransform
from lsst.afw.geom import makeTransform
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.coadd.utils as coaddUtils
from lsst.geom import radians
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from .assembleCoadd import AssembleCoaddTask, CompareWarpAssembleCoaddTask, CompareWarpAssembleCoaddConfig

__all__ = ["DcrAssembleCoaddTask", "DcrAssembleCoaddConfig"]


class DcrAssembleCoaddConfig(CompareWarpAssembleCoaddConfig):
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
        default=False,
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
        doc="Threshold to exclude noise-like pixels from regularization.",
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
    imageWarpMethod = pexConfig.Field(
        dtype=str,
        doc="Name of the warping kernel to use for shifting the image and variance planes.",
        default="bilinear",
    )
    maskWarpMethod = pexConfig.Field(
        dtype=str,
        doc="Name of the warping kernel to use for shifting the mask plane.",
        default="bilinear",
    )

    def setDefaults(self):
        CompareWarpAssembleCoaddConfig.setDefaults(self)
        self.doNImage = True
        self.warpType = 'direct'
        self.assembleStaticSkyModel.warpType = self.warpType
        self.assembleStaticSkyModel.doNImage = self.doNImage
        self.statistic = 'MEAN'
        if self.usePsf:
            self.useFFT = True
        if self.doWeightGain:
            self.useConvergence = True


class DcrAssembleCoaddTask(CompareWarpAssembleCoaddTask):
    """Assemble DCR coadded images from a set of warps.

    Notes
    -----
    As with AssembleCoaddTask, we want to assemble a coadded image from a set of
    Warps (also called coadded temporary exposures), including the effects of
    Differential Chromatic Refraction (DCR).
    For full details of the mathematics and algorithm, please see
    DMTN-037: DCR-matched template generation (https://dmtn-037.lsst.io).

    This Task produces a DCR-corrected deepCoadd, as well as a dcrCoadd for
    each subfilter used in the iterative calculation.
    It begins by dividing the bandpass-defining filter into N equal bandwidth
    "subfilters", and divides the flux in each pixel from an initial coadd
    equally into each as a "dcrModel". Because the airmass and parallactic
    angle of each individual exposure is known, we can calculate the shift
    relative to the center of the band in each subfilter due to DCR. For each
    exposure we apply this shift as a linear transformation to the dcrModels
    and stack the results to produce a DCR-matched exposure. The matched
    exposures are subtracted from the input exposures to produce a set of
    residual images, and these residuals are reverse shifted for each
    exposures' subfilters and stacked. The shifted and stacked residuals are
    added to the dcrModels to produce a new estimate of the flux in each pixel
    within each subfilter. The dcrModels are solved for iteratively, which
    continues until the solution from a new iteration improves by less than
    a set percentage, or a maximum number of iterations is reached.
    Two forms of regularization are employed to reduce unphysical results.
    First, the new solution is averaged with the solution from the previous
    iteration, which mitigates oscillating solutions where the model
    overshoots with alternating very high and low values.
    Second, a common degeneracy when the data have a limited range of airmass or
    parallactic angle values is for one subfilter to be fit with very low or
    negative values, while another subfilter is fit with very high values. This
    typically appears in the form of holes next to sources in one subfilter,
    and corresponding extended wings in another. Because each subfilter has
    a narrow bandwidth we assume that physical sources that are above the noise
    level will not vary in flux by more than a factor of `clampFrequency`
    between subfilters, and pixels that have flux deviations larger than that
    factor will have the excess flux distributed evenly among all subfilters.
    """

    ConfigClass = DcrAssembleCoaddConfig
    _DefaultName = "dcrAssembleCoadd"

    @pipeBase.timeMethod
    def run(self, dataRef, selectDataList=[]):
        """Assemble a coadd from a set of warps.

        Coadd a set of Warps. Compute weights to be applied to each Warp and
        find scalings to match the photometric zeropoint to a reference Warp.
        Assemble the Warps using assemble.
        Interpolate over NaNs and optionally write the coadd to disk.
        Return the coadded exposure.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference defining the patch for coaddition and the
            reference Warp
        selectDataList : `list`, optional
            List of data references to warps. Data to be coadded will be
            selected from this list based on overlap with the patch defined by
            dataRef.

        Returns
        -------
        `lsst.pipe.base.Struct`
            The Struct contains the following fields:

            - ``coaddExposure``: coadded exposure (`lsst.afw.image.Exposure`)
            - ``nImage``: exposure count image (`lsst.afw.image.imageU`)
            - ``dcrCoadds``: list of coadded exposures for each subfilter
            - ``dcrNImages``: list of exposure count images for each subfilter

        """
        results = AssembleCoaddTask.run(self, dataRef, selectDataList=selectDataList)
        for subfilter in range(self.config.dcrNumSubfilters):
            self.processResults(results.dcrCoadds[subfilter], dataRef)
            if self.config.doWrite:
                self.log.info("Persisting dcrCoadd")
                dataRef.put(results.dcrCoadds[subfilter], "dcrCoadd", subfilter=subfilter,
                            numSubfilters=self.config.dcrNumSubfilters)
            if self.config.doNImage and results.dcrNImages is not None:
                dataRef.put(results.dcrNImages[subfilter], "dcrCoadd_nImage", subfilter=subfilter,
                            numSubfilters=self.config.dcrNumSubfilters)

        return results

    def prepareDcrInputs(self, templateCoadd, tempExpRefList, weightList):
        """Prepare the DCR coadd by iterating through the visitInfo of the input warps.

        Sets the properties ``filterInfo`` and ``bufferSize``.

        Parameters
        ----------
        templateCoadd : `lsst.afw.image.ExposureF`
            The initial coadd exposure before accounting for DCR.
        tempExpRefList : `list` of ButlerDataRefs
            The data references to the input warped exposures.
        weightList : `list` of floats
            The weight to give each input exposure in the coadd
            Will be modified in place if ``doAirmassWeight`` is set.
        """
        self.filterInfo = templateCoadd.getFilter()
        tempExpName = self.getTempExpDatasetName(self.warpType)
        dcrShifts = []
        for visitNum, tempExpRef in enumerate(tempExpRefList):
            visitInfo = tempExpRef.get(tempExpName + "_visitInfo")
            airmass = visitInfo.getBoresightAirmass()
            if self.config.doAirmassWeight:
                weightList[visitNum] *= airmass
            dcrShifts.append(np.max(np.abs(self.dcrShiftCalculate(visitInfo,
                                                                  templateCoadd.getInfo().getWcs()))))
        self.bufferSize = int(np.ceil(np.max(dcrShifts)) + 1)
        # Turn off the warping cache, since we set the linear interpolation length to the entire subregion
        warpCache = 0
        warpInterpLength = max(self.config.subregionSize)
        self.warpCtrl = afwMath.WarpingControl(self.config.imageWarpMethod,
                                               self.config.maskWarpMethod,
                                               warpCache, warpInterpLength)

    def assemble(self, skyInfo, tempExpRefList, imageScalerList, weightList,
                 supplementaryData=None):
        """Assemble the coadd.

        Requires additional inputs Struct ``supplementaryData`` to contain a
        ``templateCoadd`` that serves as the model of the static sky.

        Find artifacts and apply them to the warps' masks creating a list of
        alternative masks with a new "CLIPPED" plane and updated "NO_DATA" plane
        Then pass these alternative masks to the base class's assemble method.

        Parameters
        ----------
        skyInfo : `lsst.skymap.discreteSkyMap.DiscreteSkyMap`
            Patch geometry information, from getSkyInfo
        tempExpRefList : `list` of `lsst.daf.persistence.ButlerDataRef`
            The data references to the input warped exposures.
        imageScalerList : `list` of image scalers
            The image scalars correct for the zero point of the exposures.
        weightList : `list` of `float`
            The weight to give each input exposure in the coadd
        supplementaryData : `lsst.pipe.base.Struct`
            Result struct returned by ``makeSupplementaryData`` with components:

            - ``templateCoadd``: coadded exposure (`lsst.afw.image.Exposure`)

        Returns
        -------
        `lsst.pipe.base.Struct`
            The struct contains the following fields:

            - ``coaddExposure``: coadded exposure (`lsst.afw.image.Exposure`)
            - ``nImage``: exposure count image (`lsst.afw.image.imageU`)
            - ``dcrCoadds``: `list` of coadded exposures for each subfilter
            - ``dcrNImages``: `list` of exposure count images for each subfilter
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
        self.prepareDcrInputs(templateCoadd, tempExpRefList, weightList)
        subregionSize = afwGeom.Extent2I(*self.config.subregionSize)
        if self.config.doNImage:
            dcrNImages = self.calculateNImage(skyInfo, tempExpRefList, spanSetMaskList, stats.ctrl)
            nImage = afwImage.ImageU(skyInfo.bbox)
            # Note that this nImage will be a factor of dcrNumSubfilters higher than
            # the nImage returned by assembleCoadd for most pixels. This is because each
            # subfilter may have a different nImage, and fractional values are not allowed.
            for dcrNImage in dcrNImages:
                nImage += dcrNImage
        else:
            dcrNImages = None

        baseMask = templateCoadd.mask
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
                                      " desired convergence improvement of %s."
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
        return pipeBase.Struct(coaddExposure=coaddExposure, nImage=nImage,
                               dcrCoadds=dcrCoadds, dcrNImages=dcrNImages)

    def calculateNImage(self, skyInfo, tempExpRefList, spanSetMaskList, statsCtrl):
        """Calculate the number of exposures contributing to each subfilter.

        Parameters
        ----------
        dcrNImages : `list` of `lsst.afw.image.imageU`
            List of exposure count images for each subfilter
        bbox : `lsst.afw.geom.box.Box2I`
            Bounding box of the patch to coadd.
        tempExpRefList : `list` of ButlerDataRefs
            The data references to the input warped exposures.
        spanSetMaskList : `list` of `dict` containing spanSet lists, or None
            Each element is dict with keys = mask plane name to add the spans to
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd
        """
        dcrNImages = [afwImage.ImageU(skyInfo.bbox) for subfilter in range(self.config.dcrNumSubfilters)]
        subregionSize = afwGeom.Extent2I(*self.config.subregionSize)
        for subBBox in self._subBBoxIter(skyInfo.bbox, subregionSize):
            bboxGrow = afwGeom.Box2I(subBBox)
            bboxGrow.grow(self.bufferSize)
            bboxGrow.clip(skyInfo.bbox)
            subNImages = [afwImage.ImageU(bboxGrow) for subfilter in range(self.config.dcrNumSubfilters)]
            tempExpName = self.getTempExpDatasetName(self.warpType)
            for tempExpRef, altMaskSpans in zip(tempExpRefList, spanSetMaskList):
                exposure = tempExpRef.get(tempExpName + "_sub", bbox=bboxGrow)
                visitInfo = exposure.getInfo().getVisitInfo()
                wcs = exposure.getInfo().getWcs()
                mask = exposure.mask
                if altMaskSpans is not None:
                    self.applyAltMaskPlanes(mask, altMaskSpans)
                dcrShift = self.dcrShiftCalculate(visitInfo, wcs)
                for dcr, subNImage in zip(dcrShift, subNImages):
                    shiftedImage = self.convolveDcrModelPlane(exposure.maskedImage, dcr, useInverse=True)
                    subNImage.array[shiftedImage.mask.array & statsCtrl.getAndMask() == 0] += 1
            for subfilter, subNImage in enumerate(subNImages):
                dcrNImages[subfilter].assign(subNImage[subBBox, afwImage.PARENT], subBBox)
        return dcrNImages

    def dcrAssembleSubregion(self, dcrModels, bbox, tempExpRefList, imageScalerList, weightList,
                             spanSetMaskList, statsFlags, statsCtrl, convergenceMetric, baseMask):
        """Assemble the DCR coadd for a sub-region.

        Build a DCR-matched template for each input exposure, then shift the
        residuals according to the DCR in each subfilter.
        Stack the shifted residuals and apply them as a correction to the
        solution from the previous iteration.
        Restrict the new model solutions from varying by more than a factor of
        `modelClampFactor` from the last solution, and additionally restrict the
        individual subfilter models from varying by more than a factor of
        `clampFrequency` from their average.
        Finally, mitigate potentially oscillating solutions by averaging the new
        solution with the solution from the previous iteration, weighted by
        their convergence metric.

        Parameters
        ----------
        dcrModels : `list` of `lsst.afw.image.maskedImageF`
            A list of masked images, each containing the model for one subfilter
        bbox : `lsst.afw.geom.box.Box2I`
            Bounding box of the subregion to coadd.
        tempExpRefList : `list` of ButlerDataRefs
            The data references to the input warped exposures.
        imageScalerList : `list` of image scalers
            The image scalars correct for the zero point of the exposures.
        weightList : `list` of `float`
            The weight to give each input exposure in the coadd
        spanSetMaskList : `list` of `dict` containing spanSet lists, or None
            Each element is dict with keys = mask plane name to add the spans to
        statsFlags : `lsst.afw.math.Property`
            Statistics settings for coaddition.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd
        convergenceMetric : `float`
            Quality of fit metric for the matched templates of the input images.
        baseMask : `lsst.afw.image.Mask`
            Mask of the initial template coadd.
        """
        bboxGrow = afwGeom.Box2I(bbox)
        bboxGrow.grow(self.bufferSize)
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
                self._removeMaskPlanes(maskedImage)
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
                newModel = self.clampModel(residual, oldModel, bboxGrow, statsCtrl)
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

    def clampModel(self, residual, oldModel, bbox, statsCtrl):
        """Restrict large variations in the model between iterations.

        Parameters
        ----------
        residual : `lsst.afw.image.maskedImageF`
            Stacked residual masked image after subtracting DCR-matched
            templates. To save memory, the residual is modified in-place.
        oldModel : `lsst.afw.image.maskedImageF`
            Description
        bbox : `lsst.afw.geom.box.Box2I`
            Sub-region to coadd
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd

        Returns
        -------
        lsst.afw.image.maskedImageF
            The sum of the oldModel and residual, with extreme values clipped.
        """
        newModel = residual
        newModel += oldModel[bbox, afwImage.PARENT]
        newImage = newModel.image.array
        oldImage = oldModel[bbox, afwImage.PARENT].image.array
        noiseCutoff = self.calculateNoiseCutoff(oldModel[bbox, afwImage.PARENT], statsCtrl)
        # Catch any invalid values
        nanPixels = np.isnan(newImage)
        newImage[nanPixels] = 0.
        infPixels = np.isinf(newImage)
        newImage[infPixels] = oldImage[infPixels]*self.config.modelClampFactor
        # Clip pixels that have very high amplitude, compared with the previous iteration.
        clampPixels = np.abs(newImage - oldImage) > (np.abs(oldImage*(self.config.modelClampFactor - 1)) +
                                                     noiseCutoff)
        highPixels = newImage > oldImage
        newImage[clampPixels & highPixels] = oldImage[clampPixels & highPixels]*self.config.modelClampFactor
        newImage[clampPixels & ~highPixels] = oldImage[clampPixels & ~highPixels]/self.config.modelClampFactor

        return newModel

    def setModelVariance(self, dcrModels):
        """Set the subfilter variance planes from the first iteration's results.

        We are not solving for the variance, so we need to shift the variance
        plane only once. Otherwise, regions with high variance will bleed into
        neighboring pixels with each successive iteration.

        Parameters
        ----------
        dcrModels : `list` of `lsst.afw.image.maskedImageF`
            A list of masked images, each containing the model for one subfilter
        """
        if self.subfilterVariance is None:
            self.subfilterVariance = [mi.variance.array for mi in dcrModels]
        else:
            for mi, variance in zip(dcrModels, self.subfilterVariance):
                mi.variance.array[:] = variance

    def regularizeModel(self, dcrModels, bbox, mask, statsCtrl):
        """Restrict large variations in the model between subfilters.

        Any flux subtracted by the restriction is accumulated from all
        subfilters, and divided evenly to each afterwards in order to preserve
        total flux.

        Parameters
        ----------
        dcrModels : `list` of `lsst.afw.image.maskedImageF`
            A list of masked images, each containing the model for one subfilter.
        bbox : `lsst.afw.geom.box.Box2I`
            Sub-region to coadd
        mask : `lsst.afw.image.Mask`
            Reference mask to use for all model planes.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd
        """
        nModels = len(dcrModels)
        templateImage = np.mean([model[bbox, afwImage.PARENT].image.array
                                 for model in dcrModels], axis=0)
        excess = np.zeros_like(templateImage)
        for model in dcrModels:
            noiseCutoff = self.calculateNoiseCutoff(model, statsCtrl, mask=mask[bbox, afwImage.PARENT])
            modelVals = model.image.array
            highPixels = (modelVals > (templateImage*self.config.clampFrequency + noiseCutoff))
            excess[highPixels] += modelVals[highPixels] - templateImage[highPixels]*self.config.clampFrequency
            modelVals[highPixels] = templateImage[highPixels]*self.config.clampFrequency
            lowPixels = (modelVals < templateImage/self.config.clampFrequency - noiseCutoff)
            excess[lowPixels] += modelVals[lowPixels] - templateImage[lowPixels]/self.config.clampFrequency
            modelVals[lowPixels] = templateImage[lowPixels]/self.config.clampFrequency
        excess /= nModels
        for model in dcrModels:
            model.image.array += excess

    def calculateNoiseCutoff(self, maskedImage, statsCtrl, mask=None):
        """Helper function to calculate the background noise level of an image.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.maskedImageF`
            The input image to evaluate the background noise properties.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd
        mask : `lsst.afw.image.Mask`, Optional
            Optional alternate mask

        Returns
        -------
        float
            The threshold value to treat pixels as noise in an image,
            set by self.config.regularizeSigma.
        """
        convergeMask = maskedImage.mask.getPlaneBitMask(self.config.convergenceMaskPlanes)
        if mask is None:
            backgroundPixels = maskedImage.mask.array & (statsCtrl.getAndMask() ^ convergeMask) == 0
        else:
            backgroundPixels = mask.array & (statsCtrl.getAndMask() ^ convergeMask) == 0
        noiseCutoff = self.config.regularizeSigma*np.std(maskedImage.image.array[backgroundPixels])
        return noiseCutoff

    def calculateConvergence(self, dcrModels, bbox, tempExpRefList, imageScalerList,
                             weightList, spanSetMaskList, statsFlags, statsCtrl):
        """Calculate a quality of fit metric for the matched templates.

        Parameters
        ----------
        dcrModels : `list` of `lsst.afw.image.maskedImageF`
            A list of masked images, each containing the model for one subfilter
        bbox : `lsst.afw.geom.box.Box2I`
            Sub-region to coadd
        tempExpRefList : `list` of ButlerDataRefs
            The data references to the input warped exposures.
        imageScalerList : `list` of image scalers
            The image scalars correct for the zero point of the exposures.
        weightList : `list` of `float`
            The weight to give each input exposure in the coadd
        spanSetMaskList : `list` of `dict` containing spanSet lists, or None
            Each element is dict with keys = mask plane name to add the spans to
        statsFlags : `lsst.afw.math.Property`
            Statistics settings for coaddition.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd

        Returns
        -------
        `float`
            Quality of fit metric for all input exposures, within the sub-region
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
        """Calculate a quality of fit metric for a single matched template.

        Parameters
        ----------
        exposure : `lsst.afw.image.exposure.ExposureF`
            The input warped exposure to evaluate.
        dcrModels : `list` of `lsst.afw.image.maskedImageF`
            A list of masked images, each containing the model for one subfilter
        significanceImage : `numpy.ndarray`
            Array of weights for each pixel corresponding to its significance
            for the convergence calculation.

        Returns
        -------
        `float`
            Quality of fit metric for one exposure, within the sub-region.
        """
        convergeMask = afwImage.Mask.getPlaneBitMask(self.config.convergenceMaskPlanes)
        templateImage = self.buildMatchedTemplate(dcrModels, exposure.getInfo().getVisitInfo(),
                                                  exposure.getBBox(), exposure.getInfo().getWcs())
        diffVals = np.abs(exposure.image.array - templateImage.image.array)*significanceImage
        refVals = np.abs(templateImage.image.array)*significanceImage

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

    def dcrDivideCoadd(self, coaddExposure, dcrNumSubfilters):
        """Divide a coadd into equal subfilter coadds.

        Parameters
        ----------
        coaddExposure : `lsst.afw.image.exposure.ExposureF`
            The target image for the coadd
        dcrNumSubfilters : `int`
            The number of subfilters to divide the coadd into.

        Returns
        -------
        `list` of `lsst.afw.image.maskedImageF`
            A list of masked images, each containing the model for one subfilter
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

    def stackCoadd(self, dcrCoadds):
        """Add a list of sub-band coadds together.

        Parameters
        ----------
        dcrCoadds : `list` of `lsst.afw.image.exposure.ExposureF`
            A list of coadd exposures, each exposure containing
            the model for one subfilter.

        Returns
        -------
        `lsst.afw.image.exposure.ExposureF`
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
        dcrModels : `list` of `lsst.afw.image.maskedImageF`
            A list of masked images, each containing the model for one subfilter
        skyInfo : `lsst.skymap.discreteSkyMap.DiscreteSkyMap`
            Patch geometry information, from ``getSkyInfo``
        tempExpRefList : `list` of ButlerDataRefs
            The data references to the input warped exposures.
        weightList : `list` of `float`
            The weight to give each input exposure in the coadd

        Returns
        -------
        `list` of `lsst.afw.image.exposure.ExposureF`
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
        maskedImage : `lsst.afw.image.maskedImageF`
            The input masked image to shift.
        dcr : `lsst.afw.geom.Extent2I`
            Shift calculated with ``calculateDcr``.
        bbox : `lsst.afw.geom.box.Box2I`, optional
            Sub-region of the masked image to shift.
            Shifts the entire image if None.
        useFFT : `bool`, optional
            Perform the convolution with an FFT?
        useInverse : `bool`, optional
            Use the reverse of ``dcr`` for the shift.

        Returns
        -------
        `lsst.afw.image.maskedImageF`
            A masked image, with the pixels within the bounding box shifted.

        Raises
        ------
        NotImplementedError
            The Fourier transform approach has not yet been written.
        """
        if self.config.useFFT:
            raise NotImplementedError("The Fourier transform approach has not yet been written.")
        else:
            padValue = afwImage.pixel.SinglePixelF(0., maskedImage.mask.getPlaneBitMask("NO_DATA"), 0)
            if bbox is None:
                bbox = maskedImage.getBBox()
            shiftedImage = afwImage.MaskedImageF(bbox)
            if useInverse:
                transform = makeTransform(AffineTransform(-dcr))
            else:
                transform = makeTransform(AffineTransform(dcr))
            afwMath.warpImage(shiftedImage, maskedImage[bbox, afwImage.PARENT],
                              transform, self.warpCtrl, padValue=padValue)
        return shiftedImage

    @staticmethod
    def conditionDcrModel(oldDcrModels, newDcrModels, bbox, gain=1.):
        """Average two iterations' solutions to reduce oscillations.

        Parameters
        ----------
        oldDcrModels : `list` of `lsst.afw.image.maskedImageF`
            The models for each subfilter from the previous iteration.
        newDcrModels : `list` of `lsst.afw.image.maskedImageF`
            The models for each subfilter from the current iteration.
        bbox : `lsst.afw.geom.box.Box2I`
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
        visitInfo : `lsst.afw.image.VisitInfo`
            Metadata for the exposure.
        wcs : `lsst.afw.geom.SkyWcs`
            Coordinate system definition (wcs) for the exposure.

        Returns
        -------
        `lsst.afw.geom.Extent2I`
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
        dcrModels : `list` of `lsst.afw.image.maskedImageF`
            A list of masked images, each containing the model for one subfilter
        visitInfo : `lsst.afw.image.VisitInfo`
            Metadata for the exposure.
        bbox : `lsst.afw.geom.box.Box2I`
            Sub-region of the coadd
        wcs : `lsst.afw.geom.SkyWcs`
            Coordinate system definition (wcs) for the exposure.
        mask : `lsst.afw.image.Mask`, optional
            reference mask to use for the template image.

        Returns
        -------
        `lsst.afw.image.maskedImageF`
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
        dcrModels : `list` of `lsst.afw.image.maskedImageF`
            A list of masked images, each containing the model for one subfilter.
        residual : `lsst.afw.image.maskedImageF`
            The residual masked image for one exposure,
            after subtracting the matched template
        visitInfo : `lsst.afw.image.VisitInfo`
            Metadata for the exposure.
        bbox : `lsst.afw.geom.box.Box2I`
            Sub-region of the coadd
        wcs : `lsst.afw.geom.SkyWcs`
            Coordinate system definition (wcs) for the exposure.

        Yields
        ------
        `lsst.afw.image.maskedImageF`
            The residual image for the next subfilter, shifted for DCR.
        """
        dcrShift = self.dcrShiftCalculate(visitInfo, wcs)
        for dcr in dcrShift:
            yield self.convolveDcrModelPlane(residual, dcr, bbox=bbox, useInverse=True)

    def calculateRotationAngle(self, visitInfo, wcs):
        """Calculate the sky rotation angle of an exposure.

        Parameters
        ----------
        visitInfo : `lsst.afw.image.VisitInfo`
            Metadata for the exposure.
        wcs : `lsst.afw.geom.SkyWcs`
            Coordinate system definition (wcs) for the exposure.

        Returns
        -------
        `lsst.geom.Angle`
            The rotation of the image axis, East from North.
            Equal to the parallactic angle plus any additional rotation of the
            coordinate system.
            A rotation angle of 0 degrees is defined with
            North along the +y axis and East along the +x axis.
            A rotation angle of 90 degrees is defined with
            North along the +x axis and East along the -y axis.
        """
        parAngle = visitInfo.getBoresightParAngle().asRadians()
        cd = wcs.getCdMatrix()
        cdAngle = (np.arctan2(-cd[0, 1], cd[0, 0]) + np.arctan2(cd[1, 0], cd[1, 1]))/2.
        rotAngle = (cdAngle + parAngle)*radians
        return rotAngle

    def wavelengthGenerator(self):
        """Iterate over the wavelength endpoints of subfilters.

        Yields
        ------
        `tuple` of two `float`
            The next set of wavelength endpoints for a subfilter, in nm.
        """
        lambdaMin = self.filterInfo.getFilterProperty().getLambdaMin()
        lambdaMax = self.filterInfo.getFilterProperty().getLambdaMax()
        wlStep = (lambdaMax - lambdaMin)/self.config.dcrNumSubfilters
        for wl in np.linspace(lambdaMin, lambdaMax, self.config.dcrNumSubfilters, endpoint=False):
            yield (wl, wl + wlStep)
