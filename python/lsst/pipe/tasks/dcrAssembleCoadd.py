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

from math import ceil
import numpy as np
from scipy import ndimage
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.coadd.utils as coaddUtils
from lsst.ip.diffim.dcrModel import applyDcr, calculateDcr, DcrModel
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from .assembleCoadd import AssembleCoaddTask, CompareWarpAssembleCoaddTask, CompareWarpAssembleCoaddConfig

__all__ = ["DcrAssembleCoaddTask", "DcrAssembleCoaddConfig"]


class DcrAssembleCoaddConfig(CompareWarpAssembleCoaddConfig):
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
        doc="Use convergence test as a forward modeling end condition?"
            "If not set, skips calculating convergence and runs for ``maxNumIter`` iterations",
        default=True,
    )
    baseGain = pexConfig.Field(
        dtype=float,
        optional=True,
        doc="Relative weight to give the new solution vs. the last solution when updating the model."
            "A value of 1.0 gives equal weight to both solutions."
            "Small values imply slower convergence of the solution, but can "
            "help prevent overshooting and failures in the fit."
            "If ``baseGain`` is None, a conservative gain "
            "will be calculated from the number of subfilters. ",
        default=None,
    )
    useProgressiveGain = pexConfig.Field(
        dtype=bool,
        doc="Use a gain that slowly increases above ``baseGain`` to accelerate convergence? "
        "When calculating the next gain, we use up to 5 previous gains and convergence values."
        "Can be set to False to force the model to change at the rate of ``baseGain``. ",
        default=True,
    )
    doAirmassWeight = pexConfig.Field(
        dtype=bool,
        doc="Weight exposures by airmass? Useful if there are relatively few high-airmass observations.",
        default=True,
    )
    modelWeightsWidth = pexConfig.Field(
        dtype=float,
        doc="Width of the region around detected sources to include in the DcrModel.",
        default=3,
    )
    useModelWeights = pexConfig.Field(
        dtype=bool,
        doc="Width of the region around detected sources to include in the DcrModel.",
        default=True,
    )
    splitSubfilters = pexConfig.Field(
        dtype=bool,
        doc="Calculate DCR for two evenly-spaced wavelengths in each subfilter."
            "Instead of at the midpoint",
        default=False,
    )
    regularizeModelIterations = pexConfig.Field(
        dtype=float,
        doc="Maximum relative change of the model allowed between iterations."
            "Set to zero to disable.",
        default=2.,
    )
    regularizeModelFrequency = pexConfig.Field(
        dtype=float,
        doc="Maximum relative change of the model allowed between subfilters."
            "Set to zero to disable.",
        default=4.,
    )
    convergenceMaskPlanes = pexConfig.ListField(
        dtype=str,
        default=["DETECTED"],
        doc="Mask planes to use to calculate convergence."
    )
    regularizationWidth = pexConfig.Field(
        dtype=int,
        default=2,
        doc="Minimum radius of a region to include in regularization, in pixels."
    )
    imageWarpMethod = pexConfig.Field(
        dtype=str,
        doc="Name of the warping kernel to use for shifting the image and variance planes.",
        default="lanczos3",
    )
    maskWarpMethod = pexConfig.Field(
        dtype=str,
        doc="Name of the warping kernel to use for shifting the mask plane.",
        default="bilinear",
    )

    def setDefaults(self):
        CompareWarpAssembleCoaddConfig.setDefaults(self)
        self.assembleStaticSkyModel.retarget(CompareWarpAssembleCoaddTask)
        self.doNImage = True
        self.warpType = 'direct'
        self.assembleStaticSkyModel.warpType = self.warpType
        # The deepCoadd and nImage files will be overwritten by this Task, so don't write them the first time
        self.assembleStaticSkyModel.doNImage = False
        self.assembleStaticSkyModel.doWrite = False
        self.statistic = 'MEAN'


class DcrAssembleCoaddTask(CompareWarpAssembleCoaddTask):
    """Assemble DCR coadded images from a set of warps.

    Attributes
    ----------
    bufferSize : `int`
        The number of pixels to grow each subregion by to allow for DCR.
    warpCtrl : `lsst.afw.math.WarpingControl`
        Configuration settings for warping an image

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
    level will not vary in flux by more than a factor of `frequencyClampFactor`
    between subfilters, and pixels that have flux deviations larger than that
    factor will have the excess flux distributed evenly among all subfilters.
    """

    ConfigClass = DcrAssembleCoaddConfig
    _DefaultName = "dcrAssembleCoadd"

    @pipeBase.timeMethod
    def runDataRef(self, dataRef, selectDataList=None, warpRefList=None):
        """Assemble a coadd from a set of warps.

        Coadd a set of Warps. Compute weights to be applied to each Warp and
        find scalings to match the photometric zeropoint to a reference Warp.
        Assemble the Warps using run method.
        Forward model chromatic effects across multiple subfilters,
        and subtract from the input Warps to build sets of residuals.
        Use the residuals to construct a new ``DcrModel`` for each subfilter,
        and iterate until the model converges.
        Interpolate over NaNs and optionally write the coadd to disk.
        Return the coadded exposure.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference defining the patch for coaddition and the
            reference Warp
        selectDataList : `list` of `lsst.daf.persistence.ButlerDataRef`
            List of data references to warps. Data to be coadded will be
            selected from this list based on overlap with the patch defined by
            the data reference.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The Struct contains the following fields:

            - ``coaddExposure``: coadded exposure (`lsst.afw.image.Exposure`)
            - ``nImage``: exposure count image (`lsst.afw.image.ImageU`)
            - ``dcrCoadds``: `list` of coadded exposures for each subfilter
            - ``dcrNImages``: `list` of exposure count images for each subfilter
        """
        if (selectDataList is None and warpRefList is None) or (selectDataList and warpRefList):
            raise RuntimeError("runDataRef must be supplied either a selectDataList or warpRefList")

        results = AssembleCoaddTask.runDataRef(self, dataRef, selectDataList=selectDataList,
                                               warpRefList=warpRefList)
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

        Sets the properties ``warpCtrl`` and ``bufferSize``.

        Parameters
        ----------
        templateCoadd : `lsst.afw.image.ExposureF`
            The initial coadd exposure before accounting for DCR.
        tempExpRefList : `list` of `lsst.daf.persistence.ButlerDataRef`
            The data references to the input warped exposures.
        weightList : `list` of `float`
            The weight to give each input exposure in the coadd
            Will be modified in place if ``doAirmassWeight`` is set.

        Returns
        -------
        dcrModels : `lsst.pipe.tasks.DcrModel`
            Best fit model of the true sky after correcting chromatic effects.

        Raises
        ------
        NotImplementedError
            If ``lambdaMin`` is missing from the Mapper class of the obs package being used.
        """
        filterInfo = templateCoadd.getFilter()
        if np.isnan(filterInfo.getFilterProperty().getLambdaMin()):
            raise NotImplementedError("No minimum/maximum wavelength information found"
                                      " in the filter definition! Please add lambdaMin and lambdaMax"
                                      " to the Mapper class in your obs package.")
        tempExpName = self.getTempExpDatasetName(self.warpType)
        dcrShifts = []
        for visitNum, tempExpRef in enumerate(tempExpRefList):
            visitInfo = tempExpRef.get(tempExpName + "_visitInfo")
            airmass = visitInfo.getBoresightAirmass()
            if self.config.doAirmassWeight:
                weightList[visitNum] *= airmass
            dcrShifts.append(np.max(np.abs(calculateDcr(visitInfo, templateCoadd.getWcs(),
                                                        filterInfo, self.config.dcrNumSubfilters))))
        self.bufferSize = int(np.ceil(np.max(dcrShifts)) + 1)
        # Turn off the warping cache, since we set the linear interpolation length to the entire subregion
        # This warper is only used for applying DCR shifts, which are assumed to be uniform across a patch
        warpCache = 0
        warpInterpLength = max(self.config.subregionSize)
        self.warpCtrl = afwMath.WarpingControl(self.config.imageWarpMethod,
                                               self.config.maskWarpMethod,
                                               cacheSize=warpCache, interpLength=warpInterpLength)
        dcrModels = DcrModel.fromImage(templateCoadd.maskedImage,
                                       self.config.dcrNumSubfilters,
                                       filterInfo=filterInfo,
                                       psf=templateCoadd.getPsf())
        return dcrModels

    def run(self, skyInfo, tempExpRefList, imageScalerList, weightList,
            supplementaryData=None):
        """Assemble the coadd.

        Requires additional inputs Struct ``supplementaryData`` to contain a
        ``templateCoadd`` that serves as the model of the static sky.

        Find artifacts and apply them to the warps' masks creating a list of
        alternative masks with a new "CLIPPED" plane and updated "NO_DATA" plane
        Then pass these alternative masks to the base class's assemble method.

        Divide the ``templateCoadd`` evenly between each subfilter of a
        ``DcrModel`` as the starting best estimate of the true wavelength-
        dependent sky. Forward model the ``DcrModel`` using the known
        chromatic effects in each subfilter and calculate a convergence metric
        based on how well the modeled template matches the input warps. If
        the convergence has not yet reached the desired threshold, then shift
        and stack the residual images to build a new ``DcrModel``. Apply
        conditioning to prevent oscillating solutions between iterations or
        between subfilters.

        Once the ``DcrModel`` reaches convergence or the maximum number of
        iterations has been reached, fill the metadata for each subfilter
        image and make them proper ``coaddExposure``s.

        Parameters
        ----------
        skyInfo : `lsst.pipe.base.Struct`
            Patch geometry information, from getSkyInfo
        tempExpRefList : `list` of `lsst.daf.persistence.ButlerDataRef`
            The data references to the input warped exposures.
        imageScalerList : `list` of `lsst.pipe.task.ImageScaler`
            The image scalars correct for the zero point of the exposures.
        weightList : `list` of `float`
            The weight to give each input exposure in the coadd
        supplementaryData : `lsst.pipe.base.Struct`
            Result struct returned by ``makeSupplementaryData`` with components:

            - ``templateCoadd``: coadded exposure (`lsst.afw.image.Exposure`)

        Returns
        -------
        result : `lsst.pipe.base.Struct`
           Result struct with components:

            - ``coaddExposure``: coadded exposure (`lsst.afw.image.Exposure`)
            - ``nImage``: exposure count image (`lsst.afw.image.ImageU`)
            - ``dcrCoadds``: `list` of coadded exposures for each subfilter
            - ``dcrNImages``: `list` of exposure count images for each subfilter
        """
        templateCoadd = supplementaryData.templateCoadd
        baseMask = templateCoadd.mask.clone()
        # The variance plane is for each subfilter
        # and should be proportionately lower than the full-band image
        baseVariance = templateCoadd.variance.clone()
        baseVariance /= self.config.dcrNumSubfilters
        spanSetMaskList = self.findArtifacts(templateCoadd, tempExpRefList, imageScalerList)
        # Note that the mask gets cleared in ``findArtifacts``, but we want to preserve the mask.
        templateCoadd.setMask(baseMask)
        badMaskPlanes = self.config.badMaskPlanes[:]
        badMaskPlanes.append("CLIPPED")
        badPixelMask = templateCoadd.mask.getPlaneBitMask(badMaskPlanes)

        stats = self.prepareStats(mask=badPixelMask)
        dcrModels = self.prepareDcrInputs(templateCoadd, tempExpRefList, weightList)
        if self.config.doNImage:
            dcrNImages = self.calculateNImage(dcrModels, skyInfo.bbox,
                                              tempExpRefList, spanSetMaskList, stats.ctrl)
            nImage = afwImage.ImageU(skyInfo.bbox)
            # Note that this nImage will be a factor of dcrNumSubfilters higher than
            # the nImage returned by assembleCoadd for most pixels. This is because each
            # subfilter may have a different nImage, and fractional values are not allowed.
            for dcrNImage in dcrNImages:
                nImage += dcrNImage
        else:
            dcrNImages = None

        subregionSize = afwGeom.Extent2I(*self.config.subregionSize)
        nSubregions = (ceil(skyInfo.bbox.getHeight()/subregionSize[1]) *
                       ceil(skyInfo.bbox.getWidth()/subregionSize[0]))
        subIter = 0
        for subBBox in self._subBBoxIter(skyInfo.bbox, subregionSize):
            modelIter = 0
            subIter += 1
            self.log.info("Computing coadd over patch %s subregion %s of %s: %s",
                          skyInfo.patchInfo.getIndex(), subIter, nSubregions, subBBox)
            dcrBBox = afwGeom.Box2I(subBBox)
            dcrBBox.grow(self.bufferSize)
            dcrBBox.clip(dcrModels.bbox)
            if self.config.useModelWeights:
                modelWeights = self.calculateModelWeights(dcrModels, dcrBBox)
            else:
                modelWeights = 1.
            convergenceMetric = self.calculateConvergence(dcrModels, subBBox, tempExpRefList,
                                                          imageScalerList, weightList, spanSetMaskList,
                                                          stats.ctrl)
            self.log.info("Initial convergence : %s", convergenceMetric)
            convergenceList = [convergenceMetric]
            gainList = []
            convergenceCheck = 1.
            subfilterVariance = None
            while (convergenceCheck > self.config.convergenceThreshold or
                   modelIter < self.config.minNumIter):
                gain = self.calculateGain(convergenceList, gainList)
                self.dcrAssembleSubregion(dcrModels, subBBox, dcrBBox, tempExpRefList, imageScalerList,
                                          weightList, spanSetMaskList, stats.flags, stats.ctrl,
                                          convergenceMetric, baseMask, subfilterVariance, gain,
                                          modelWeights)
                if self.config.useConvergence:
                    convergenceMetric = self.calculateConvergence(dcrModels, subBBox, tempExpRefList,
                                                                  imageScalerList, weightList,
                                                                  spanSetMaskList,
                                                                  stats.ctrl)
                    if convergenceMetric == 0:
                        self.log.warn("Coadd patch %s subregion %s had convergence metric of 0.0 which is "
                                      "most likely due to there being no valid data in the region.",
                                      skyInfo.patchInfo.getIndex(), subIter)
                        break
                    convergenceCheck = (convergenceList[-1] - convergenceMetric)/convergenceMetric
                    if convergenceCheck < 0:
                        self.log.warn("Coadd patch %s subregion %s diverged before reaching maximum "
                                      "iterations or desired convergence improvement of %s."
                                      " Divergence: %s",
                                      skyInfo.patchInfo.getIndex(), subIter,
                                      self.config.convergenceThreshold, convergenceCheck)
                        break
                    convergenceList.append(convergenceMetric)
                if modelIter > self.config.maxNumIter:
                    if self.config.useConvergence:
                        self.log.warn("Coadd patch %s subregion %s reached maximum iterations "
                                      "before reaching desired convergence improvement of %s."
                                      " Final convergence improvement: %s",
                                      skyInfo.patchInfo.getIndex(), subIter,
                                      self.config.convergenceThreshold, convergenceCheck)
                    break

                if self.config.useConvergence:
                    self.log.info("Iteration %s with convergence metric %s, %.4f%% improvement (gain: %.2f)",
                                  modelIter, convergenceMetric, 100.*convergenceCheck, gain)
                modelIter += 1
            else:
                if self.config.useConvergence:
                    self.log.info("Coadd patch %s subregion %s finished with "
                                  "convergence metric %s after %s iterations",
                                  skyInfo.patchInfo.getIndex(), subIter, convergenceMetric, modelIter)
                else:
                    self.log.info("Coadd patch %s subregion %s finished after %s iterations",
                                  skyInfo.patchInfo.getIndex(), subIter, modelIter)
            if self.config.useConvergence and convergenceMetric > 0:
                self.log.info("Final convergence improvement was %.4f%% overall",
                              100*(convergenceList[0] - convergenceMetric)/convergenceMetric)

        dcrCoadds = self.fillCoadd(dcrModels, skyInfo, tempExpRefList, weightList,
                                   calibration=self.scaleZeroPoint.getCalib(),
                                   coaddInputs=templateCoadd.getInfo().getCoaddInputs(),
                                   mask=baseMask,
                                   variance=baseVariance)
        coaddExposure = self.stackCoadd(dcrCoadds)
        return pipeBase.Struct(coaddExposure=coaddExposure, nImage=nImage,
                               dcrCoadds=dcrCoadds, dcrNImages=dcrNImages)

    def calculateNImage(self, dcrModels, bbox, tempExpRefList, spanSetMaskList, statsCtrl):
        """Calculate the number of exposures contributing to each subfilter.

        Parameters
        ----------
        dcrModels : `lsst.pipe.tasks.DcrModel`
            Best fit model of the true sky after correcting chromatic effects.
        bbox : `lsst.afw.geom.box.Box2I`
            Bounding box of the patch to coadd.
        tempExpRefList : `list` of `lsst.daf.persistence.ButlerDataRef`
            The data references to the input warped exposures.
        spanSetMaskList : `list` of `dict` containing spanSet lists, or None
            Each element is dict with keys = mask plane name to add the spans to
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd

        Returns
        -------
        dcrNImages : `list` of `lsst.afw.image.ImageU`
            List of exposure count images for each subfilter
        """
        dcrNImages = [afwImage.ImageU(bbox) for subfilter in range(self.config.dcrNumSubfilters)]
        tempExpName = self.getTempExpDatasetName(self.warpType)
        for tempExpRef, altMaskSpans in zip(tempExpRefList, spanSetMaskList):
            exposure = tempExpRef.get(tempExpName + "_sub", bbox=bbox)
            visitInfo = exposure.getInfo().getVisitInfo()
            wcs = exposure.getInfo().getWcs()
            mask = exposure.mask
            if altMaskSpans is not None:
                self.applyAltMaskPlanes(mask, altMaskSpans)
            dcrShift = calculateDcr(visitInfo, wcs, dcrModels.filter, self.config.dcrNumSubfilters)
            for dcr, dcrNImage in zip(dcrShift, dcrNImages):
                shiftedImage = applyDcr(exposure.maskedImage, dcr, self.warpCtrl, useInverse=True)
                dcrNImage.array[shiftedImage.mask.array & statsCtrl.getAndMask() == 0] += 1
        return dcrNImages

    def dcrAssembleSubregion(self, dcrModels, bbox, dcrBBox, tempExpRefList, imageScalerList, weightList,
                             spanSetMaskList, statsFlags, statsCtrl, convergenceMetric,
                             baseMask, subfilterVariance, gain, modelWeights):
        """Assemble the DCR coadd for a sub-region.

        Build a DCR-matched template for each input exposure, then shift the
        residuals according to the DCR in each subfilter.
        Stack the shifted residuals and apply them as a correction to the
        solution from the previous iteration.
        Restrict the new model solutions from varying by more than a factor of
        `modelClampFactor` from the last solution, and additionally restrict the
        individual subfilter models from varying by more than a factor of
        `frequencyClampFactor` from their average.
        Finally, mitigate potentially oscillating solutions by averaging the new
        solution with the solution from the previous iteration, weighted by
        their convergence metric.

        Parameters
        ----------
        dcrModels : `lsst.pipe.tasks.DcrModel`
            Best fit model of the true sky after correcting chromatic effects.
        bbox : `lsst.afw.geom.box.Box2I`
            Bounding box of the subregion to coadd.
        dcrBBox :`lsst.afw.geom.box.Box2I`
            Sub-region of the coadd which includes a buffer to allow for DCR.
        tempExpRefList : `list` of `lsst.daf.persistence.ButlerDataRef`
            The data references to the input warped exposures.
        imageScalerList : `list` of `lsst.pipe.task.ImageScaler`
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
        subfilterVariance : `list` of `numpy.ndarray`
            The variance of each coadded subfilter image.
        gain : `float`, optional
            Relative weight to give the new solution when updating the model.
        modelWeights : `numpy.ndarray` or `float`
            A 2D array of weight values that tapers smoothly to zero away from detected sources.
            Set to a placeholder value of 1.0 if ``self.config.useModelWeights`` is False.
        """
        tempExpName = self.getTempExpDatasetName(self.warpType)
        residualGeneratorList = []

        for tempExpRef, imageScaler, altMaskSpans in zip(tempExpRefList, imageScalerList, spanSetMaskList):
            exposure = tempExpRef.get(tempExpName + "_sub", bbox=dcrBBox)
            visitInfo = exposure.getInfo().getVisitInfo()
            wcs = exposure.getInfo().getWcs()
            maskedImage = exposure.maskedImage
            templateImage = dcrModels.buildMatchedTemplate(warpCtrl=self.warpCtrl, visitInfo=visitInfo,
                                                           bbox=dcrBBox, wcs=wcs, mask=baseMask,
                                                           splitSubfilters=self.config.splitSubfilters)
            imageScaler.scaleMaskedImage(maskedImage)
            if altMaskSpans is not None:
                self.applyAltMaskPlanes(maskedImage.mask, altMaskSpans)

            if self.config.removeMaskPlanes:
                self.removeMaskPlanes(maskedImage)
            maskedImage -= templateImage
            maskedImage.image.array *= modelWeights
            residualGeneratorList.append(self.dcrResiduals(maskedImage, visitInfo, dcrBBox, wcs,
                                                           dcrModels.filter))

        dcrSubModelOut = self.newModelFromResidual(dcrModels, residualGeneratorList, dcrBBox,
                                                   statsFlags, statsCtrl, weightList,
                                                   mask=baseMask, gain=gain)
        dcrModels.assign(dcrSubModelOut, bbox)

    def dcrResiduals(self, residual, visitInfo, bbox, wcs, filterInfo):
        """Prepare a residual image for stacking in each subfilter by applying the reverse DCR shifts.

        Parameters
        ----------
        residual : `lsst.afw.image.MaskedImageF`
            The residual masked image for one exposure,
            after subtracting the matched template
        visitInfo : `lsst.afw.image.VisitInfo`
            Metadata for the exposure.
        bbox : `lsst.afw.geom.box.Box2I`
            Sub-region of the coadd
        wcs : `lsst.afw.geom.SkyWcs`
            Coordinate system definition (wcs) for the exposure.
        filterInfo : `lsst.afw.image.Filter`
            The filter definition, set in the current instruments' obs package.
            Required for any calculation of DCR, including making matched templates.

        Yields
        ------
        residualImage : `lsst.afw.image.maskedImageF`
            The residual image for the next subfilter, shifted for DCR.
        """
        dcrShift = calculateDcr(visitInfo, wcs, filterInfo, self.config.dcrNumSubfilters)
        for dcr in dcrShift:
            yield applyDcr(residual, dcr, self.warpCtrl, bbox=bbox, useInverse=True)

    def newModelFromResidual(self, dcrModels, residualGeneratorList, bbox,
                             statsFlags, statsCtrl, weightList,
                             mask, gain):
        """Calculate a new DcrModel from a set of image residuals.

        Parameters
        ----------
        dcrModels : `lsst.pipe.tasks.DcrModel`
            Current model of the true sky after correcting chromatic effects.
        residualGeneratorList : `generator` of `lsst.afw.image.maskedImageF`
            The residual image for the next subfilter, shifted for DCR.
        bbox : `lsst.afw.geom.box.Box2I`
            Sub-region of the coadd
        statsFlags : `lsst.afw.math.Property`
            Statistics settings for coaddition.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd
        weightList : `list` of `float`
            The weight to give each input exposure in the coadd
        mask : `lsst.afw.image.Mask`
            Mask to use for each new model image.
        gain : `float`
            Relative weight to give the new solution when updating the model.

        Returns
        -------
        dcrModel : `lsst.pipe.tasks.DcrModel`
            New model of the true sky after correcting chromatic effects.
        """
        maskMap = self.setRejectedMaskMapping(statsCtrl)
        clipped = dcrModels.mask.getPlaneBitMask("CLIPPED")
        newModelImages = []
        for subfilter, model in enumerate(dcrModels):
            residualsList = [next(residualGenerator) for residualGenerator in residualGeneratorList]
            residual = afwMath.statisticsStack(residualsList, statsFlags, statsCtrl, weightList,
                                               clipped, maskMap)
            residual.setXY0(bbox.getBegin())
            # `MaskedImage`s only support in-place addition, so rename for readability
            residual += model[bbox]
            newModel = residual
            # Catch any invalid values
            badPixels = ~np.isfinite(newModel.image.array)
            # Overwrite the mask with one calculated previously. If the mask is allowed to adjust
            # every iteration, masked regions will continually expand.
            newModel.setMask(mask[bbox])
            newModel.image.array[badPixels] = model[bbox].image.array[badPixels]
            if self.config.regularizeModelIterations > 0:
                dcrModels.regularizeModelIter(subfilter, newModel, bbox,
                                              self.config.regularizeModelIterations,
                                              self.config.regularizationWidth)
            newModelImages.append(newModel)
        if self.config.regularizeModelFrequency > 0:
            dcrModels.regularizeModelFreq(newModelImages, bbox,
                                          self.config.regularizeModelFrequency,
                                          self.config.regularizationWidth)
        dcrModels.conditionDcrModel(newModelImages, bbox, gain=gain)
        return DcrModel(newModelImages, dcrModels.filter, dcrModels.psf)

    def calculateConvergence(self, dcrModels, bbox, tempExpRefList, imageScalerList,
                             weightList, spanSetMaskList, statsCtrl):
        """Calculate a quality of fit metric for the matched templates.

        Parameters
        ----------
        dcrModels : `lsst.pipe.tasks.DcrModel`
            Best fit model of the true sky after correcting chromatic effects.
        bbox : `lsst.afw.geom.box.Box2I`
            Sub-region to coadd
        tempExpRefList : `list` of `lsst.daf.persistence.ButlerDataRef`
            The data references to the input warped exposures.
        imageScalerList : `list` of `lsst.pipe.task.ImageScaler`
            The image scalars correct for the zero point of the exposures.
        weightList : `list` of `float`
            The weight to give each input exposure in the coadd
        spanSetMaskList : `list` of `dict` containing spanSet lists, or None
            Each element is dict with keys = mask plane name to add the spans to
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd

        Returns
        -------
        convergenceMetric : `float`
            Quality of fit metric for all input exposures, within the sub-region
        """
        significanceImage = np.abs(dcrModels.getReferenceImage(bbox))
        nSigma = 3.
        significanceImage += nSigma*dcrModels.calculateNoiseCutoff(dcrModels[1], statsCtrl,
                                                                   bufferSize=self.bufferSize)
        tempExpName = self.getTempExpDatasetName(self.warpType)
        weight = 0
        metric = 0.
        metricList = {}
        zipIterables = zip(tempExpRefList, weightList, imageScalerList, spanSetMaskList)
        for tempExpRef, expWeight, imageScaler, altMaskSpans in zipIterables:
            exposure = tempExpRef.get(tempExpName + "_sub", bbox=bbox)
            imageScaler.scaleMaskedImage(exposure.maskedImage)
            singleMetric = self.calculateSingleConvergence(dcrModels, exposure, significanceImage, statsCtrl,
                                                           altMaskSpans=altMaskSpans)
            metric += singleMetric*expWeight
            metricList[tempExpRef.dataId["visit"]] = singleMetric
            weight += expWeight
        self.log.info("Individual metrics:\n%s", metricList)
        return 1.0 if weight == 0.0 else metric/weight

    def calculateSingleConvergence(self, dcrModels, exposure, significanceImage,
                                   statsCtrl, altMaskSpans=None):
        """Calculate a quality of fit metric for a single matched template.

        Parameters
        ----------
        dcrModels : `lsst.pipe.tasks.DcrModel`
            Best fit model of the true sky after correcting chromatic effects.
        exposure : `lsst.afw.image.ExposureF`
            The input warped exposure to evaluate.
        significanceImage : `numpy.ndarray`
            Array of weights for each pixel corresponding to its significance
            for the convergence calculation.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd
        altMaskSpans : `dict` containing spanSet lists, or None
            The keys of the `dict` equal the mask plane name to add the spans to

        Returns
        -------
        convergenceMetric : `float`
            Quality of fit metric for one exposure, within the sub-region.
        """
        convergeMask = exposure.mask.getPlaneBitMask(self.config.convergenceMaskPlanes)
        templateImage = dcrModels.buildMatchedTemplate(warpCtrl=self.warpCtrl,
                                                       visitInfo=exposure.getInfo().getVisitInfo(),
                                                       bbox=exposure.getBBox(),
                                                       wcs=exposure.getInfo().getWcs())
        diffVals = np.abs(exposure.image.array - templateImage.image.array)*significanceImage
        refVals = np.abs(templateImage.image.array)*significanceImage

        finitePixels = np.isfinite(diffVals)
        if altMaskSpans is not None:
            self.applyAltMaskPlanes(exposure.mask, altMaskSpans)
        goodMaskPixels = exposure.mask.array & statsCtrl.getAndMask() == 0
        convergeMaskPixels = exposure.mask.array & convergeMask > 0
        usePixels = finitePixels & goodMaskPixels & convergeMaskPixels
        if np.sum(usePixels) == 0:
            metric = 0.
        else:
            diffUse = diffVals[usePixels]
            refUse = refVals[usePixels]
            metric = np.sum(diffUse/np.median(diffUse))/np.sum(refUse/np.median(diffUse))
        return metric

    def stackCoadd(self, dcrCoadds):
        """Add a list of sub-band coadds together.

        Parameters
        ----------
        dcrCoadds : `list` of `lsst.afw.image.ExposureF`
            A list of coadd exposures, each exposure containing
            the model for one subfilter.

        Returns
        -------
        coaddExposure : `lsst.afw.image.ExposureF`
            A single coadd exposure that is the sum of the sub-bands.
        """
        coaddExposure = dcrCoadds[0].clone()
        for coadd in dcrCoadds[1:]:
            coaddExposure.maskedImage += coadd.maskedImage
        return coaddExposure

    def fillCoadd(self, dcrModels, skyInfo, tempExpRefList, weightList, calibration=None, coaddInputs=None,
                  mask=None, variance=None):
        """Create a list of coadd exposures from a list of masked images.

        Parameters
        ----------
        dcrModels : `lsst.pipe.tasks.DcrModel`
            Best fit model of the true sky after correcting chromatic effects.
        skyInfo : `lsst.pipe.base.Struct`
            Patch geometry information, from getSkyInfo
        tempExpRefList : `list` of `lsst.daf.persistence.ButlerDataRef`
            The data references to the input warped exposures.
        weightList : `list` of `float`
            The weight to give each input exposure in the coadd
        calibration : `lsst.afw.Image.Calib`, optional
            Scale factor to set the photometric zero point of an exposure.
        coaddInputs : `lsst.afw.Image.CoaddInputs`, optional
            A record of the observations that are included in the coadd.
        mask : `lsst.afw.image.Mask`, optional
            Optional mask to override the values in the final coadd.
        variance : `lsst.afw.image.Image`, optional
            Optional variance plane to override the values in the final coadd.

        Returns
        -------
        dcrCoadds : `list` of `lsst.afw.image.ExposureF`
            A list of coadd exposures, each exposure containing
            the model for one subfilter.
        """
        dcrCoadds = []
        for model in dcrModels:
            coaddExposure = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
            if calibration is not None:
                coaddExposure.setCalib(calibration)
            if coaddInputs is not None:
                coaddExposure.getInfo().setCoaddInputs(coaddInputs)
            # Set the metadata for the coadd, including PSF and aperture corrections.
            self.assembleMetadata(coaddExposure, tempExpRefList, weightList)
            coaddUtils.setCoaddEdgeBits(model[skyInfo.bbox].mask, model[skyInfo.bbox].variance)
            coaddExposure.setMaskedImage(model[skyInfo.bbox])
            if mask is not None:
                coaddExposure.setMask(mask)
            if variance is not None:
                coaddExposure.setVariance(variance)
            dcrCoadds.append(coaddExposure)
        return dcrCoadds

    def calculateGain(self, convergenceList, gainList):
        """Calculate the gain to use for the current iteration.

        After calculating a new DcrModel, each value is averaged with the
        value in the corresponding pixel from the previous iteration. This
        reduces oscillating solutions that iterative techniques are plagued by,
        and speeds convergence. By far the biggest changes to the model
        happen in the first couple iterations, so we can also use a more
        aggressive gain later when the model is changing slowly.

        Parameters
        ----------
        convergenceList : `list` of `float`
            The quality of fit metric from each previous iteration.
        gainList : `list` of `float`
            The gains used in each previous iteration: appended with the new
            gain value.
            Gains are numbers between ``self.config.baseGain`` and 1.

        Returns
        -------
        gain : `float`
            Relative weight to give the new solution when updating the model.
            A value of 1.0 gives equal weight to both solutions.

        Raises
        ------
        ValueError
            If ``len(convergenceList) != len(gainList)+1``.
        """
        nIter = len(convergenceList)
        if nIter != len(gainList) + 1:
            raise ValueError("convergenceList (%d) must be one element longer than gainList (%d)."
                             % (len(convergenceList), len(gainList)))

        if self.config.baseGain is None:
            # If ``baseGain`` is not set, calculate it from the number of DCR subfilters
            # The more subfilters being modeled, the lower the gain should be.
            baseGain = 1./(self.config.dcrNumSubfilters - 1)
        else:
            baseGain = self.config.baseGain

        if self.config.useProgressiveGain and nIter > 2:
            # To calculate the best gain to use, compare the past gains that have been used
            # with the resulting convergences to estimate the best gain to use.
            # Algorithmically, this is a Kalman filter.
            # If forward modeling proceeds perfectly, the convergence metric should
            # asymptotically approach a final value.
            # We can estimate that value from the measured changes in convergence
            # weighted by the gains used in each previous iteration.
            estFinalConv = [((1 + gainList[i])*convergenceList[i + 1] - convergenceList[i])/gainList[i]
                            for i in range(nIter - 1)]
            # The convergence metric is strictly positive, so if the estimated final convergence is
            # less than zero, force it to zero.
            estFinalConv = np.array(estFinalConv)
            estFinalConv[estFinalConv < 0] = 0
            # Because the estimate may slowly change over time, only use the most recent measurements.
            estFinalConv = np.median(estFinalConv[max(nIter - 5, 0):])
            lastGain = gainList[-1]
            lastConv = convergenceList[-2]
            newConv = convergenceList[-1]
            # The predicted convergence is the value we would get if the new model calculated
            # in the previous iteration was perfect. Recall that the updated model that is
            # actually used is the gain-weighted average of the new and old model,
            # so the convergence would be similarly weighted.
            predictedConv = (estFinalConv*lastGain + lastConv)/(1. + lastGain)
            # If the measured and predicted convergence are very close, that indicates
            # that our forward model is accurate and we can use a more aggressive gain
            # If the measured convergence is significantly worse (or better!) than predicted,
            # that indicates that the model is not converging as expected and
            # we should use a more conservative gain.
            delta = (predictedConv - newConv)/((lastConv - estFinalConv)/(1 + lastGain))
            newGain = 1 - abs(delta)
            # Average the gains to prevent oscillating solutions.
            newGain = (newGain + lastGain)/2.
            gain = max(baseGain, newGain)
        else:
            gain = baseGain
        gainList.append(gain)
        return gain

    def calculateModelWeights(self, dcrModels, dcrBBox):
        """Build an array that smoothly tapers to 0 away from detected sources.

        Parameters
        ----------
        dcrModels : `lsst.pipe.tasks.DcrModel`
            Best fit model of the true sky after correcting chromatic effects.
        dcrBBox : `lsst.afw.geom.box.Box2I`
            Sub-region of the coadd which includes a buffer to allow for DCR.

        Returns
        -------
        weights : `numpy.ndarray` or `float`
            A 2D array of weight values that tapers smoothly to zero away from detected sources.
            Set to a placeholder value of 1.0 if ``self.config.useModelWeights`` is False.

        Raises
        ------
        ValueError
            If ``useModelWeights`` is set and ``modelWeightsWidth`` is negative.
        """
        if self.config.modelWeightsWidth < 0:
            raise ValueError("modelWeightsWidth must not be negative if useModelWeights is set")
        convergeMask = dcrModels.mask.getPlaneBitMask(self.config.convergenceMaskPlanes)
        convergeMaskPixels = dcrModels.mask[dcrBBox].array & convergeMask > 0
        weights = np.zeros_like(dcrModels[0][dcrBBox].image.array)
        weights[convergeMaskPixels] = 1.
        weights = ndimage.filters.gaussian_filter(weights, self.config.modelWeightsWidth)
        weights /= np.max(weights)
        return weights
