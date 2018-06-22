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

__all__ = ["DcrAssembleCoaddTask", "DcrAssembleCoaddConfig", "DcrModel"]


class DcrAssembleCoaddConfig(CompareWarpAssembleCoaddConfig):
    useFFT = pexConfig.Field(
        dtype=bool,
        doc="Use Fourier transforms for the convolution?"
            "Note: not yet implemented. Intended for use with variable PSFs.",
        default=False,
    )
    usePsf = pexConfig.Field(
        dtype=bool,
        doc="Convolve models with the PSF of the exposures? Requires `useFFT=True`."
            "Note: not yet implemented. Intended for use with variable PSFs.",
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
        doc="Use convergence test as a forward modeling end condition?"
            "If not set, skips calculating convergence and runs for ``maxNumIter`` iterations",
        default=True,
    )
    doWeightGain = pexConfig.Field(
        dtype=bool,
        doc="Use the calculated convergence metric to accelerate forward modeling?"
            "If set, convergence has to be calculated an extra time each iteration,"
            "but fewer iterations may be needed to reach the convergence threshold.",
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
        
        Returns
        -------
        `DcrModel`
            Description
        
        Raises
        ------
        NotImplementedError
            If ``lambdaMin`` is missing from the Mapper class of the obs package being used.
        """
        self.filterInfo = templateCoadd.getFilter()
        if np.isnan(self.filterInfo.getFilterProperty().getLambdaMin()):
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
                                                        self.filterInfo, self.config.dcrNumSubfilters))))
        self.bufferSize = int(np.ceil(np.max(dcrShifts)) + 1)
        # Turn off the warping cache, since we set the linear interpolation length to the entire subregion
        # This warper is only used for applying DCR shifts, which are assumed to be uniform across a patch
        warpCache = 0
        warpInterpLength = max(self.config.subregionSize)
        self.warpCtrl = afwMath.WarpingControl(self.config.imageWarpMethod,
                                               self.config.maskWarpMethod,
                                               cacheSize=warpCache, interpLength=warpInterpLength)
        dcrModels = DcrModel(self.config.dcrNumSubfilters, coaddExposure=templateCoadd,
                             clampFrequency=self.config.clampFrequency,
                             modelClampFactor=self.config.modelClampFactor,
                             regularizeSigma=self.config.regularizeSigma,
                             convergenceMaskPlanes=self.config.convergenceMaskPlanes,
                             warpCtrl=self.warpCtrl,
                             filterInfo=self.filterInfo,
                             )
        return dcrModels

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
        badPixelMask = templateCoadd.mask.getPlaneBitMask(badMaskPlanes)
        # Propagate PSF-matched EDGE pixels to coadd SENSOR_EDGE and INEXACT_PSF
        # Psf-Matching moves the real edge inwards
        self.applyAltEdgeMask(templateCoadd.mask, spanSetMaskList)

        stats = self.prepareStats(mask=badPixelMask)
        dcrModels = self.prepareDcrInputs(templateCoadd, tempExpRefList, weightList)
        if self.config.doNImage:
            dcrNImages = self.calculateNImage(dcrModels, skyInfo.bbox, tempExpRefList, spanSetMaskList, stats.ctrl)
            nImage = afwImage.ImageU(skyInfo.bbox)
            # Note that this nImage will be a factor of dcrNumSubfilters higher than
            # the nImage returned by assembleCoadd for most pixels. This is because each
            # subfilter may have a different nImage, and fractional values are not allowed.
            for dcrNImage in dcrNImages:
                nImage += dcrNImage
        else:
            dcrNImages = None

        tempExpName = self.getTempExpDatasetName(self.warpType)
        baseMask = templateCoadd.mask
        subregionSize = afwGeom.Extent2I(*self.config.subregionSize)
        for subBBox in self._subBBoxIter(skyInfo.bbox, subregionSize):
            modelIter = 0
            self.log.info("Computing coadd over %s", subBBox)
            convergenceMetric = self.calculateConvergence(dcrModels, subBBox, tempExpRefList,
                                                          imageScalerList, weightList, spanSetMaskList,
                                                          stats.ctrl, tempExpName)
            self.log.info("Initial convergence : %s", convergenceMetric)
            convergenceList = [convergenceMetric]
            convergenceCheck = 1.
            subfilterVariance = None
            while (convergenceCheck > self.config.convergenceThreshold or
                   modelIter < self.config.minNumIter):
                # try:
                self.dcrAssembleSubregion(dcrModels, subBBox, tempExpRefList, imageScalerList,
                                          weightList, spanSetMaskList, stats.flags, stats.ctrl,
                                          convergenceMetric, baseMask, subfilterVariance)
                if self.config.useConvergence:
                    convergenceMetric = self.calculateConvergence(dcrModels, subBBox, tempExpRefList,
                                                                  imageScalerList, weightList,
                                                                  spanSetMaskList,
                                                                  stats.ctrl, tempExpName)
                    convergenceCheck = (convergenceList[-1] - convergenceMetric)/convergenceMetric
                    convergenceList.append(convergenceMetric)
                # except Exception as e:
                #     self.log.fatal("Cannot compute coadd %s: %s", subBBox, e)
                if modelIter > self.config.maxNumIter:
                    if self.config.useConvergence:
                        self.log.warn("Coadd %s reached maximum iterations before reaching"
                                      " desired convergence improvement of %s."
                                      " Final convergence improvement: %s",
                                      subBBox, self.config.convergenceThreshold, convergenceCheck)
                    break

                if self.config.useConvergence:
                    self.log.info("Iteration %s with convergence metric %s, %.4f%% improvement",
                                  modelIter, convergenceMetric, 100.*convergenceCheck)
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
                                   coaddInputs=self.inputRecorder.makeCoaddInputs())
        coaddExposure = self.stackCoadd(dcrCoadds)
        return pipeBase.Struct(coaddExposure=coaddExposure, nImage=nImage,
                               dcrCoadds=dcrCoadds, dcrNImages=dcrNImages)

    def calculateNImage(self, dcrModels, bbox, tempExpRefList, spanSetMaskList, statsCtrl):
        """Calculate the number of exposures contributing to each subfilter.

        Parameters
        ----------
        bbox : `lsst.afw.geom.box.Box2I`
            Bounding box of the patch to coadd.
        tempExpRefList : `list` of ButlerDataRefs
            The data references to the input warped exposures.
        spanSetMaskList : `list` of `dict` containing spanSet lists, or None
            Each element is dict with keys = mask plane name to add the spans to
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd

        Returns
        -------
        dcrNImages : `list` of `lsst.afw.image.imageU`
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
            dcrShift = calculateDcr(visitInfo, wcs, self.filterInfo, self.config.dcrNumSubfilters)
            for dcr, dcrNImage in zip(dcrShift, dcrNImages):
                shiftedImage = applyDcr(exposure.maskedImage, dcr, self.warpCtrl, useInverse=True)
                dcrNImage.array[shiftedImage.mask.array & statsCtrl.getAndMask() == 0] += 1
        return dcrNImages

    def dcrAssembleSubregion(self, dcrModels, bbox, tempExpRefList, imageScalerList, weightList,
                             spanSetMaskList, statsFlags, statsCtrl, convergenceMetric,
                             baseMask, subfilterVariance):
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
        subfilterVariance : `list` of `numpy.ndarray`
            The variance of each coadded subfilter image.
        """
        bboxGrow = afwGeom.Box2I(bbox)
        bboxGrow.grow(self.bufferSize)
        for model in dcrModels.modelImages:
            bboxGrow.clip(model.getBBox())

        tempExpName = self.getTempExpDatasetName(self.warpType)
        residualGeneratorList = []

        for tempExpRef, imageScaler, altMaskSpans in zip(tempExpRefList, imageScalerList, spanSetMaskList):
            exposure = tempExpRef.get(tempExpName + "_sub", bbox=bboxGrow)
            visitInfo = exposure.getInfo().getVisitInfo()
            wcs = exposure.getInfo().getWcs()
            maskedImage = exposure.maskedImage
            templateImage = dcrModels.buildMatchedTemplate(visitInfo=visitInfo, bbox=bboxGrow,
                                                           wcs=wcs, mask=baseMask)
            imageScaler.scaleMaskedImage(maskedImage)
            if altMaskSpans is not None:
                self.applyAltMaskPlanes(maskedImage.mask, altMaskSpans)

            if self.config.removeMaskPlanes:
                self.removeMaskPlanes(maskedImage)
            maskedImage -= templateImage
            residualGeneratorList.append(self.dcrResiduals(maskedImage, visitInfo, bboxGrow, wcs))

        dcrSubModelOut = dcrModels.newModelFromResidual(residualGeneratorList, bboxGrow,
                                                        statsFlags, statsCtrl, weightList)
        dcrSubModelOut.regularizeModel(bboxGrow, baseMask, statsCtrl)
        dcrModels.assign(dcrSubModelOut, bbox)

    def dcrResiduals(self, residual, visitInfo, bbox, wcs):
        """Prepare a residual image for stacking in each subfilter by applying the reverse DCR shifts.
        
        Parameters
        ----------
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
        dcrShift = calculateDcr(visitInfo, wcs, self.filterInfo, self.config.dcrNumSubfilters)
        for dcr in dcrShift:
            yield applyDcr(residual, dcr, self.warpCtrl, bbox=bbox, useInverse=True)

    def newModelFromResidual(self, dcrModels, residualGeneratorList, bbox, statsFlags, statsCtrl, weightList):
        """Summary
        
        Parameters
        ----------
        dcrModels : TYPE
            Description
        residualGeneratorList : TYPE
            Description
        bbox : TYPE
            Description
        statsFlags : TYPE
            Description
        statsCtrl : TYPE
            Description
        weightList : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        maskMap = self.setRejectedMaskMapping(statsCtrl)
        clipped = dcrModels.modelImages[0].mask.getPlaneBitMask("CLIPPED")
        newDcrModels = []
        for oldModel in dcrModels.modelImages:
            residualsList = [next(residualGenerator) for residualGenerator in residualGeneratorList]
            residual = afwMath.statisticsStack(residualsList, statsFlags, statsCtrl, weightList,
                                               clipped, maskMap)
            residual.setXY0(bbox.getBegin())
            # `MaskedImage`s only support in-place addition, so rename for readability
            residual.image += oldModel[bbox, afwImage.PARENT].image
            newModel = residual
            dcrModels.clampModel(newModel, oldModel, bbox, statsCtrl)
            dcrModels.conditionDcrModel(newModel, oldModel, bbox, gain=1.)
            newDcrModels.append(newModel)
        return DcrModel(dcrModels.dcrNumSubfilters, modelImages=newDcrModels,
                        clampFrequency=dcrModels.clampFrequency,
                        modelClampFactor=dcrModels.modelClampFactor,
                        regularizeSigma=dcrModels.regularizeSigma,
                        convergenceMaskPlanes=dcrModels.convergenceMaskPlanes,
                        filterInfo=dcrModels.filterInfo)

    def calculateConvergence(self, dcrModels, bbox, tempExpRefList, imageScalerList,
                             weightList, spanSetMaskList, statsCtrl, tempExpName):
        """Calculate a quality of fit metric for the matched templates.
        
        Parameters
        ----------
        dcrModels : TYPE
            Description
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
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd
        tempExpName : str
            Description
        
        Returns
        -------
        `float`
            Quality of fit metric for all input exposures, within the sub-region
        """
        significanceImage = np.sum([np.abs(model[bbox, afwImage.PARENT].image.array)
                                    for model in dcrModels.modelImages], axis=0)
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
        dcrModels : TYPE
            Description
        exposure : `lsst.afw.image.exposure.ExposureF`
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
        `float`
            Quality of fit metric for one exposure, within the sub-region.
        """
        convergeMask = exposure.mask.getPlaneBitMask(self.config.convergenceMaskPlanes)
        templateImage = dcrModels.buildMatchedTemplate(visitInfo=exposure.getInfo().getVisitInfo(),
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

    def fillCoadd(self, dcrModels, skyInfo, tempExpRefList, weightList, calibration=None, coaddInputs=None):
        """Create a list of coadd exposures from a list of masked images.
        
        Parameters
        ----------
        dcrModels : Type
            Description
        skyInfo : `lsst.skymap.discreteSkyMap.DiscreteSkyMap`
            Patch geometry information, from ``getSkyInfo``
        tempExpRefList : `list` of ButlerDataRefs
            The data references to the input warped exposures.
        weightList : `list` of `float`
            The weight to give each input exposure in the coadd
        calibration : None, optional
            Description
        coaddInputs : None, optional
            Description
        
        Returns
        -------
        `list` of `lsst.afw.image.exposure.ExposureF`
            A list of coadd exposures, each exposure containing
            the model for one subfilter.
        """
        dcrCoadds = []
        for model in dcrModels.modelImages:
            coaddExposure = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
            if calibration is not None:
                coaddExposure.setCalib(calibration)
            if coaddInputs is not None:
                coaddExposure.getInfo().setCoaddInputs(coaddInputs)
            self.assembleMetadata(coaddExposure, tempExpRefList, weightList)
            coaddUtils.setCoaddEdgeBits(model.mask, model.variance)
            coaddExposure.setMaskedImage(model)
            dcrCoadds.append(coaddExposure)
        return dcrCoadds


class DcrModel(object):

    """Summary
    
    Attributes
    ----------
    clampFrequency : TYPE
        Description
    convergenceMaskPlanes : TYPE
        Description
    dcrModels : `list` of `lsst.afw.image.maskedImageF`
        A list of masked images, each containing the model for one subfilter
    dcrNumSubfilters : TYPE
        Description
    filterInfo : TYPE
        Description
    modelClampFactor : TYPE
        Description
    regularizeSigma : TYPE
        Description
    """
    
    def __init__(self, dcrNumSubfilters, coaddExposure=None, dcrModels=None,
                 clampFrequency=None, modelClampFactor=None,
                 regularizeSigma=None, convergenceMaskPlanes=None,
                 warpCtrl=None, filterInfo=None):
        """Divide a coadd into equal subfilter coadds.
        
        Parameters
        ----------
        dcrNumSubfilters : TYPE
            Description
        coaddExposure : `lsst.afw.image.exposure.ExposureF`
            The target image for the coadd
        dcrModels : None, optional
            Description
        clampFrequency : None, optional
            Description
        modelClampFactor : None, optional
            Description
        regularizeSigma : None, optional
            Description
        convergenceMaskPlanes : None, optional
            Description
        filterInfo : None, optional
            Description
        
        Raises
        ------
        ValueError
            If neither ``dcrModels`` or ``coaddExposure`` are set.
        """
        self.dcrNumSubfilters = dcrNumSubfilters
        self.clampFrequency = clampFrequency
        self.modelClampFactor = modelClampFactor
        self.regularizeSigma = regularizeSigma
        self.convergenceMaskPlanes = convergenceMaskPlanes
        self.warpCtrl = warpCtrl
        self.filterInfo = filterInfo
        if dcrModels is not None:
            self.modelImages = dcrModels
        elif coaddExposure is not None:
            maskedImage = coaddExposure.maskedImage.clone()
            # NANs will potentially contaminate the entire image,
            #  depending on the shift or convolution type used.
            badPixels = np.isnan(maskedImage.image.array) | np.isnan(maskedImage.variance.array)
            maskedImage.image.array[badPixels] = 0.
            maskedImage.variance.array[badPixels] = 0.
            maskedImage.image.array /= dcrNumSubfilters
            maskedImage.variance.array /= dcrNumSubfilters
            maskedImage.mask.array[badPixels] = maskedImage.mask.getPlaneBitMask("NO_DATA")
            self.modelImages = [maskedImage, ]
            for subfilter in range(1, dcrNumSubfilters):
                self.modelImages.append(maskedImage.clone())
        else:
            raise ValueError("Either dcrModels or coaddExposure must be set.")

    def assign(self, dcrSubModel, bbox):
        """Summary
        
        Parameters
        ----------
        dcrSubModel : TYPE
            Description
        bbox : TYPE
            Description
        """
        for model, subModel in zip(self.modelImages, dcrSubModel.modelImages):
            model.assign(subModel[bbox, afwImage.PARENT], bbox)

    def setModelVariance(self, dcrModels, subfilterVariance=None):
        """Set the subfilter variance planes from the first iteration's results.
        
        We are not solving for the variance, so we need to shift the variance
        plane only once. Otherwise, regions with high variance will bleed into
        neighboring pixels with each successive iteration.
        
        Parameters
        ----------
        dcrModels : `list` of `lsst.afw.image.maskedImageF`
            A list of masked images, each containing the model for one subfilter
        subfilterVariance : None, optional
            Description
        """
        if subfilterVariance is None:
            subfilterVariance = [mi.variance.array for mi in self.modelImages]
        else:
            for mi, variance in zip(self.modelImages, subfilterVariance):
                mi.variance.array[:] = variance

    def buildMatchedTemplate(self, exposure=None, visitInfo=None, bbox=None, wcs=None, mask=None):
        """Create a DCR-matched template for an exposure.

        Parameters
        ----------
        exposure : `lsst.afw.image.exposure.ExposureF`, optional
            The input exposure to build a matched template for.
            May be omitted if all of the metadata is supplied separately
        visitInfo : `lsst.afw.image.VisitInfo`, optional
            Metadata for the exposure.
        bbox : `lsst.afw.geom.box.Box2I`, optional
            Sub-region of the coadd
        wcs : `lsst.afw.geom.SkyWcs`, optional
            Coordinate system definition (wcs) for the exposure.
        mask : `lsst.afw.image.Mask`, optional
            reference mask to use for the template image.

        Returns
        -------
        `lsst.afw.image.maskedImageF`
            The DCR-matched template

        Raises
        ------
        ValueError
            If neither ``exposure`` or all of ``visitInfo``, ``bbox``, and ``wcs`` are set.
        """
        if exposure is not None:
            visitInfo = exposure.getInfo().getVisitInfo()
            bbox = exposure.getBBox()
            wcs = exposure.getInfo().getWcs()
        elif visitInfo is None or bbox is None or wcs is None:
            raise ValueError("Either exposure or visitInfo, bbox, and wcs must be set.")
        dcrShift = calculateDcr(visitInfo, wcs, self.filterInfo, self.dcrNumSubfilters)
        templateImage = afwImage.MaskedImageF(bbox)
        for dcr, model in zip(dcrShift, self.modelImages):
            templateImage += applyDcr(model, dcr, self.warpCtrl, bbox=bbox)
        if mask is not None:
            templateImage.setMask(mask[bbox, afwImage.PARENT])
        return templateImage

    def dcrResiduals(self, residual, visitInfo, bbox, wcs):
        """Prepare a residual image for stacking in each subfilter by applying the reverse DCR shifts.

        Parameters
        ----------
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
        dcrShift = calculateDcr(visitInfo, wcs, self.filterInfo, self.dcrNumSubfilters)
        for dcr in dcrShift:
            yield applyDcr(residual, dcr, self.warpCtrl, bbox=bbox, useInverse=True)

    def newModelFromResidual(self, residualGeneratorList, bbox, statsFlags, statsCtrl, weightList):
        """Summary
        
        Parameters
        ----------
        residualGeneratorList : TYPE
            Description
        bbox : TYPE
            Description
        statsFlags : TYPE
            Description
        statsCtrl : TYPE
            Description
        weightList : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        
        Deleted Parameters
        ------------------
        residual : TYPE
            Description
        """
        maskMap = DcrAssembleCoaddTask.setRejectedMaskMapping(statsCtrl)
        clipped = self.modelImages[0].mask.getPlaneBitMask("CLIPPED")
        newDcrModels = []
        for oldModel in self.modelImages:
            residualsList = [next(residualGenerator) for residualGenerator in residualGeneratorList]
            residual = afwMath.statisticsStack(residualsList, statsFlags, statsCtrl, weightList,
                                               clipped, maskMap)
            residual.setXY0(bbox.getBegin())
            # `MaskedImage`s only support in-place addition, so rename for readability
            residual.image += oldModel[bbox, afwImage.PARENT].image
            newModel = residual
            self.clampModel(newModel, oldModel, bbox, statsCtrl)
            self.conditionDcrModel(newModel, oldModel, bbox, gain=1.)
            newDcrModels.append(newModel)
        return DcrModel(self.dcrNumSubfilters, dcrModels=newDcrModels,
                        clampFrequency=self.clampFrequency,
                        modelClampFactor=self.modelClampFactor,
                        regularizeSigma=self.regularizeSigma,
                        convergenceMaskPlanes=self.convergenceMaskPlanes,
                        filterInfo=self.filterInfo)

    def conditionDcrModel(self, newModel, oldModel, bbox, gain=1.):
        """Average two iterations' solutions to reduce oscillations.
        
        Parameters
        ----------
        newModel : TYPE
            Description
        oldModel : TYPE
            Description
        bbox : `lsst.afw.geom.box.Box2I`
            Sub-region of the coadd
        gain : `float`, optional
            Additional weight to apply to the model from the current iteration.
        
        Deleted Parameters
        ------------------
        newDcrModels : `list` of `lsst.afw.image.maskedImageF`
            The models for each subfilter from the current iteration.
        """
        # The models are MaskedImages, which only support in-place operations.
        newModel *= gain
        newModel += oldModel[bbox, afwImage.PARENT]
        newModel.image.array /= 1. + gain
        newModel.variance.array /= 1. + gain

    def clampModel(self, newModel, oldModel, bbox, statsCtrl):
        """Restrict large variations in the model between iterations.
        
        Parameters
        ----------
        newModel : TYPE
            Description
        oldModel : TYPE
            Description
        bbox : `lsst.afw.geom.box.Box2I`
            Sub-region to coadd
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd
        
        No Longer Returned
        ------------------
        lsst.afw.image.maskedImageF
            The sum of the oldModel and residual, with extreme values clipped.
        
        Deleted Parameters
        ------------------
        residual : `lsst.afw.image.maskedImageF`
            Stacked residual masked image after subtracting DCR-matched
            templates. To save memory, the residual is modified in-place.
        oldModel : `lsst.afw.image.maskedImageF`
            The DCR model from the previous iteration for one subfilter.
        newModels : TYPE
            Description
        """
        newImage = newModel.image.array
        oldImage = oldModel[bbox, afwImage.PARENT].image.array
        noiseCutoff = self.calculateNoiseCutoff(newModel, statsCtrl)
        # Catch any invalid values
        nanPixels = np.isnan(newImage)
        newImage[nanPixels] = 0.
        infPixels = np.isinf(newImage)
        newImage[infPixels] = oldImage[infPixels]*self.modelClampFactor
        # Clip pixels that have very high amplitude, compared with the previous iteration.
        clampPixels = np.abs(newImage - oldImage) > (np.abs(oldImage*(self.modelClampFactor - 1)) +
                                                     noiseCutoff)
        # Set high amplitude pixels to a multiple or fraction of the old model value,
        #  depending on whether the new model is higher or lower than the old
        highPixels = newImage > oldImage
        newImage[clampPixels & highPixels] = oldImage[clampPixels & highPixels]*self.modelClampFactor
        newImage[clampPixels & ~highPixels] = oldImage[clampPixels & ~highPixels]/self.modelClampFactor

    def regularizeModel(self, bbox, mask, statsCtrl):
        """Restrict large variations in the model between subfilters.

        Any flux subtracted by the restriction is accumulated from all
        subfilters, and divided evenly to each afterwards in order to preserve
        total flux.

        Parameters
        ----------
        bbox : `lsst.afw.geom.box.Box2I`
            Sub-region to coadd
        mask : `lsst.afw.image.Mask`
            Reference mask to use for all model planes.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd
        """
        templateImage = np.mean([model[bbox, afwImage.PARENT].image.array
                                 for model in self.modelImages], axis=0)
        excess = np.zeros_like(templateImage)
        for model in self.modelImages:
            noiseCutoff = self.calculateNoiseCutoff(model, statsCtrl, mask=mask[bbox, afwImage.PARENT])
            modelVals = model.image.array
            highPixels = (modelVals > (templateImage*self.clampFrequency + noiseCutoff))
            excess[highPixels] += modelVals[highPixels] - templateImage[highPixels]*self.clampFrequency
            modelVals[highPixels] = templateImage[highPixels]*self.clampFrequency
            lowPixels = (modelVals < templateImage/self.clampFrequency - noiseCutoff)
            excess[lowPixels] += modelVals[lowPixels] - templateImage[lowPixels]/self.clampFrequency
            modelVals[lowPixels] = templateImage[lowPixels]/self.clampFrequency
        excess /= self.dcrNumSubfilters
        for model in self.modelImages:
            model.image.array += excess

    def calculateConvergence(self, bbox, tempExpRefList, imageScalerList,
                             weightList, spanSetMaskList, statsCtrl, tempExpName):
        """Calculate a quality of fit metric for the matched templates.
        
        Parameters
        ----------
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
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd
        tempExpName : str
            Description
        
        Returns
        -------
        `float`
            Quality of fit metric for all input exposures, within the sub-region
        """
        significanceImage = np.sum([np.abs(model[bbox, afwImage.PARENT].image.array)
                                    for model in self.modelImages], axis=0)
        weight = 0
        metric = 0.
        metricList = {}
        zipIterables = zip(tempExpRefList, weightList, imageScalerList, spanSetMaskList)
        for tempExpRef, expWeight, imageScaler, altMaskSpans in zipIterables:
            exposure = tempExpRef.get(tempExpName + "_sub", bbox=bbox)
            imageScaler.scaleMaskedImage(exposure.maskedImage)
            singleMetric = self.calculateSingleConvergence(exposure, significanceImage, statsCtrl,
                                                           altMaskSpans=altMaskSpans)
            metric += singleMetric*expWeight
            metricList[tempExpRef.dataId["visit"]] = singleMetric
            weight += expWeight
        # self.log.info("Individual metrics:\n%s", metricList)
        return 1.0 if weight == 0.0 else metric/weight

    def calculateSingleConvergence(self, exposure, significanceImage, statsCtrl, altMaskSpans=None):
        """Calculate a quality of fit metric for a single matched template.

        Parameters
        ----------
        exposure : `lsst.afw.image.exposure.ExposureF`
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
        `float`
            Quality of fit metric for one exposure, within the sub-region.
        """
        convergeMask = exposure.mask.getPlaneBitMask(self.convergenceMaskPlanes)
        templateImage = self.buildMatchedTemplate(visitInfo=exposure.getInfo().getVisitInfo(),
                                                  bbox=exposure.getBBox(), wcs=exposure.getInfo().getWcs())
        diffVals = np.abs(exposure.image.array - templateImage.image.array)*significanceImage
        refVals = np.abs(templateImage.image.array)*significanceImage

        finitePixels = np.isfinite(diffVals)
        # if altMaskSpans is not None:
        #     self.applyAltMaskPlanes(exposure.mask, altMaskSpans)
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
            set by ``self.regularizeSigma``.
        """
        convergeMask = maskedImage.mask.getPlaneBitMask(self.convergenceMaskPlanes)
        if mask is None:
            backgroundPixels = maskedImage.mask.array & (statsCtrl.getAndMask() | convergeMask) == 0
        else:
            backgroundPixels = mask.array & (statsCtrl.getAndMask() | convergeMask) == 0
        noiseCutoff = self.regularizeSigma*np.std(maskedImage.image.array[backgroundPixels])
        return noiseCutoff


def applyDcr(maskedImage, dcr, warpCtrl, bbox=None, useInverse=False):
    """Shift a masked image.
    
    Parameters
    ----------
    maskedImage : `lsst.afw.image.maskedImageF`
        The input masked image to shift.
    dcr : `lsst.afw.geom.Extent2I`
        Shift calculated with ``calculateDcr``.
    warpCtrl : TYPE
        Description
    bbox : `lsst.afw.geom.box.Box2I`, optional
        Sub-region of the masked image to shift.
        Shifts the entire image if None.
    useInverse : `bool`, optional
        Use the reverse of ``dcr`` for the shift.
    
    Returns
    -------
    `lsst.afw.image.maskedImageF`
        A masked image, with the pixels within the bounding box shifted.
    
    Deleted Parameters
    ------------------
    useFFT : `bool`, optional
        Perform the convolution with an FFT?
    """
    padValue = afwImage.pixel.SinglePixelF(0., maskedImage.mask.getPlaneBitMask("NO_DATA"), 0)
    if bbox is None:
        bbox = maskedImage.getBBox()
    shiftedImage = afwImage.MaskedImageF(bbox)
    transform = makeTransform(AffineTransform((-1.0 if useInverse else 1.0)*dcr))
    afwMath.warpImage(shiftedImage, maskedImage[bbox, afwImage.PARENT],
                      transform, warpCtrl, padValue=padValue)
    return shiftedImage


def calculateDcr(visitInfo, wcs, filterInfo, dcrNumSubfilters):
    """Calculate the shift in pixels of an exposure due to DCR.
    
    Parameters
    ----------
    visitInfo : `lsst.afw.image.VisitInfo`
        Metadata for the exposure.
    wcs : `lsst.afw.geom.SkyWcs`
        Coordinate system definition (wcs) for the exposure.
    filterInfo : TYPE
        Description
    dcrNumSubfilters : TYPE
        Description
    
    Returns
    -------
    `lsst.afw.geom.Extent2I`
        The 2D shift due to DCR, in pixels.
    """
    rotation = calculateRotationAngle(visitInfo, wcs)
    dcrShift = []
    lambdaEff = filterInfo.getFilterProperty().getLambdaEff()
    for wl0, wl1 in wavelengthGenerator(filterInfo, dcrNumSubfilters):
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


def calculateRotationAngle(visitInfo, wcs):
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


def wavelengthGenerator(filterInfo, dcrNumSubfilters):
    """Iterate over the wavelength endpoints of subfilters.
    
    Parameters
    ----------
    filterInfo : TYPE
        Description
    dcrNumSubfilters : TYPE
        Description
    
    Yields
    ------
    `tuple` of two `float`
        The next set of wavelength endpoints for a subfilter, in nm.
    """
    lambdaMin = filterInfo.getFilterProperty().getLambdaMin()
    lambdaMax = filterInfo.getFilterProperty().getLambdaMax()
    wlStep = (lambdaMax - lambdaMin)/dcrNumSubfilters
    for wl in np.linspace(lambdaMin, lambdaMax, dcrNumSubfilters, endpoint=False):
        yield (wl, wl + wlStep)
