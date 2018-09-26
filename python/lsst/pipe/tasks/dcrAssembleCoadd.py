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
        doc="Relative weight to give the new solution when updating the model."
            "A value of 1.0 gives equal weight to both solutions.",
        default=1.,
    )
    useProgressiveGain = pexConfig.Field(
        dtype=bool,
        doc="Use a gain that slowly increases above ``baseGain`` to accelerate convergence?",
        default=True,
    )
    doAirmassWeight = pexConfig.Field(
        dtype=bool,
        doc="Weight exposures by airmass? Useful if there are relatively few high-airmass observations.",
        default=True,
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
        default=2.,
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
        self.doNImage = True
        self.warpType = 'direct'
        self.assembleStaticSkyModel.warpType = self.warpType
        self.assembleStaticSkyModel.doNImage = self.doNImage
        self.statistic = 'MEAN'


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
    level will not vary in flux by more than a factor of `frequencyClampFactor`
    between subfilters, and pixels that have flux deviations larger than that
    factor will have the excess flux distributed evenly among all subfilters.
    """

    ConfigClass = DcrAssembleCoaddConfig
    _DefaultName = "dcrAssembleCoadd"

    @pipeBase.timeMethod
    def runDataRef(self, dataRef, selectDataList=[]):
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
        results = AssembleCoaddTask.runDataRef(self, dataRef, selectDataList=selectDataList)
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
        spanSetMaskList = self.findArtifacts(templateCoadd, tempExpRefList, imageScalerList)
        badMaskPlanes = self.config.badMaskPlanes[:]
        badMaskPlanes.append("CLIPPED")
        badPixelMask = templateCoadd.mask.getPlaneBitMask(badMaskPlanes)
        # Propagate PSF-matched EDGE pixels to coadd SENSOR_EDGE and INEXACT_PSF
        # Psf-Matching moves the real edge inwards
        self.applyAltEdgeMask(templateCoadd.mask, spanSetMaskList)

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

        baseMask = templateCoadd.mask
        subregionSize = afwGeom.Extent2I(*self.config.subregionSize)
        for subBBox in self._subBBoxIter(skyInfo.bbox, subregionSize):
            modelIter = 0
            self.log.info("Computing coadd over %s", subBBox)
            convergenceMetric = self.calculateConvergence(dcrModels, subBBox, tempExpRefList,
                                                          imageScalerList, weightList, spanSetMaskList,
                                                          stats.ctrl)
            self.log.info("Initial convergence : %s", convergenceMetric)
            convergenceList = [convergenceMetric]
            convergenceCheck = 1.
            subfilterVariance = None
            while (convergenceCheck > self.config.convergenceThreshold or
                   modelIter < self.config.minNumIter):
                gain = self.calculateGain(modelIter, self.config.baseGain)
                self.dcrAssembleSubregion(dcrModels, subBBox, tempExpRefList, imageScalerList,
                                          weightList, spanSetMaskList, stats.flags, stats.ctrl,
                                          convergenceMetric, baseMask, subfilterVariance, gain)
                if self.config.useConvergence:
                    convergenceMetric = self.calculateConvergence(dcrModels, subBBox, tempExpRefList,
                                                                  imageScalerList, weightList,
                                                                  spanSetMaskList,
                                                                  stats.ctrl)
                    convergenceCheck = (convergenceList[-1] - convergenceMetric)/convergenceMetric
                    convergenceList.append(convergenceMetric)
                if modelIter > self.config.maxNumIter:
                    if self.config.useConvergence:
                        self.log.warn("Coadd %s reached maximum iterations before reaching"
                                      " desired convergence improvement of %s."
                                      " Final convergence improvement: %s",
                                      subBBox, self.config.convergenceThreshold, convergenceCheck)
                    break

                if self.config.useConvergence:
                    self.log.info("Iteration %s with convergence metric %s, %.4f%% improvement (gain: %.1f)",
                                  modelIter, convergenceMetric, 100.*convergenceCheck, gain)
                modelIter += 1
            else:
                if self.config.useConvergence:
                    self.log.info("Coadd %s finished with convergence metric %s after %s iterations",
                                  subBBox, convergenceMetric, modelIter)
                else:
                    self.log.info("Coadd %s finished after %s iterations", subBBox, modelIter)
            if self.config.useConvergence:
                self.log.info("Final convergence improvement was %.4f%% overall",
                              100*(convergenceList[0] - convergenceMetric)/convergenceMetric)

        dcrCoadds = self.fillCoadd(dcrModels, skyInfo, tempExpRefList, weightList,
                                   calibration=self.scaleZeroPoint.getCalib(),
                                   coaddInputs=self.inputRecorder.makeCoaddInputs(),
                                   mask=templateCoadd.mask)
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

    def dcrAssembleSubregion(self, dcrModels, bbox, tempExpRefList, imageScalerList, weightList,
                             spanSetMaskList, statsFlags, statsCtrl, convergenceMetric,
                             baseMask, subfilterVariance, gain):
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
        """
        bboxGrow = afwGeom.Box2I(bbox)
        bboxGrow.grow(self.bufferSize)
        bboxGrow.clip(dcrModels.bbox)

        tempExpName = self.getTempExpDatasetName(self.warpType)
        residualGeneratorList = []

        for tempExpRef, imageScaler, altMaskSpans in zip(tempExpRefList, imageScalerList, spanSetMaskList):
            exposure = tempExpRef.get(tempExpName + "_sub", bbox=bboxGrow)
            visitInfo = exposure.getInfo().getVisitInfo()
            wcs = exposure.getInfo().getWcs()
            maskedImage = exposure.maskedImage
            templateImage = dcrModels.buildMatchedTemplate(warpCtrl=self.warpCtrl, visitInfo=visitInfo,
                                                           bbox=bboxGrow, wcs=wcs, mask=baseMask)
            imageScaler.scaleMaskedImage(maskedImage)
            if altMaskSpans is not None:
                self.applyAltMaskPlanes(maskedImage.mask, altMaskSpans)

            if self.config.removeMaskPlanes:
                self.removeMaskPlanes(maskedImage)
            maskedImage -= templateImage
            residualGeneratorList.append(self.dcrResiduals(maskedImage, visitInfo, bboxGrow, wcs,
                                                           dcrModels.filter))

        dcrSubModelOut = self.newModelFromResidual(dcrModels, residualGeneratorList, bboxGrow,
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
                  mask=None):
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
            dcrCoadds.append(coaddExposure)
        return dcrCoadds

    def calculateGain(self, modelIter, baseGain=1.):
        """Calculate the gain to use for the current iteration.

        After calculating a new DcrModel, each value is averaged with the
        value in the corresponding pixel from the previous iteration. This
        reduces oscillating solutions that iterative techniques are plagued by,
        and speeds convergence. By far the biggest changes to the model
        happen in the first couple iterations, so we can also use a more
        aggressive gain later when the model is changing slowly.

        Parameters
        ----------
        modelIter : `int`
            The current iteration of forward modeling.
        baseGain : `float`, optional
            Description

        Returns
        -------
        gain : `float`
            Relative weight to give the new solution when updating the model.
            A value of 1.0 gives equal weight to both solutions.
        """
        if self.config.useProgressiveGain:
            iterGain = np.log(modelIter) if modelIter > 0 else baseGain
            return max(baseGain, iterGain)
        return baseGain
