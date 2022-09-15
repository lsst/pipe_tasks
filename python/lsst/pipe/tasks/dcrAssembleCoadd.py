# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["DcrAssembleCoaddConnections", "DcrAssembleCoaddTask", "DcrAssembleCoaddConfig"]

from math import ceil
import numpy as np
from scipy import ndimage
import lsst.geom as geom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.coadd.utils as coaddUtils
from lsst.ip.diffim.dcrModel import applyDcr, calculateDcr, DcrModel
import lsst.meas.algorithms as measAlg
from lsst.meas.base import SingleFrameMeasurementTask
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.utils as utils
from lsst.utils.timer import timeMethod
from .assembleCoadd import (AssembleCoaddConnections,
                            AssembleCoaddTask,
                            CompareWarpAssembleCoaddConfig,
                            CompareWarpAssembleCoaddTask)
from .coaddBase import makeSkyInfo
from .measurePsf import MeasurePsfTask


class DcrAssembleCoaddConnections(AssembleCoaddConnections,
                                  dimensions=("tract", "patch", "band", "skymap"),
                                  defaultTemplates={"inputWarpName": "deep",
                                                    "inputCoaddName": "deep",
                                                    "outputCoaddName": "dcr",
                                                    "warpType": "direct",
                                                    "warpTypeSuffix": "",
                                                    "fakesType": ""}):
    inputWarps = pipeBase.connectionTypes.Input(
        doc=("Input list of warps to be assembled i.e. stacked."
             "Note that this will often be different than the inputCoaddName."
             "WarpType (e.g. direct, psfMatched) is controlled by the warpType config parameter"),
        name="{inputWarpName}Coadd_{warpType}Warp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
        deferLoad=True,
        multiple=True
    )
    templateExposure = pipeBase.connectionTypes.Input(
        doc="Input coadded exposure, produced by previous call to AssembleCoadd",
        name="{fakesType}{inputCoaddName}Coadd{warpTypeSuffix}",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band"),
    )
    dcrCoadds = pipeBase.connectionTypes.Output(
        doc="Output coadded exposure, produced by stacking input warps",
        name="{fakesType}{outputCoaddName}Coadd{warpTypeSuffix}",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band", "subfilter"),
        multiple=True,
    )
    dcrNImages = pipeBase.connectionTypes.Output(
        doc="Output image of number of input images per pixel",
        name="{outputCoaddName}Coadd_nImage",
        storageClass="ImageU",
        dimensions=("tract", "patch", "skymap", "band", "subfilter"),
        multiple=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.doWrite:
            self.outputs.remove("dcrCoadds")
        if not config.doNImage:
            self.outputs.remove("dcrNImages")
        # Remove outputs inherited from ``AssembleCoaddConnections`` that are not used
        self.outputs.remove("coaddExposure")
        self.outputs.remove("nImage")


class DcrAssembleCoaddConfig(CompareWarpAssembleCoaddConfig,
                             pipelineConnections=DcrAssembleCoaddConnections):
    dcrNumSubfilters = pexConfig.Field(
        dtype=int,
        doc="Number of sub-filters to forward model chromatic effects to fit the supplied exposures.",
        default=3,
    )
    maxNumIter = pexConfig.Field(
        dtype=int,
        optional=True,
        doc="Maximum number of iterations of forward modeling.",
        default=None,
    )
    minNumIter = pexConfig.Field(
        dtype=int,
        optional=True,
        doc="Minimum number of iterations of forward modeling.",
        default=None,
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
        default=False,
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
        default=True,
    )
    splitThreshold = pexConfig.Field(
        dtype=float,
        doc="Minimum DCR difference within a subfilter to use ``splitSubfilters``, in pixels."
            "Set to 0 to always split the subfilters.",
        default=0.1,
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
    imageInterpOrder = pexConfig.Field(
        dtype=int,
        doc="The order of the spline interpolation used to shift the image plane.",
        default=1,
    )
    accelerateModel = pexConfig.Field(
        dtype=float,
        doc="Factor to amplify the differences between model planes by to speed convergence.",
        default=3,
    )
    doCalculatePsf = pexConfig.Field(
        dtype=bool,
        doc="Set to detect stars and recalculate the PSF from the final coadd."
        "Otherwise the PSF is estimated from a selection of the best input exposures",
        default=False,
    )
    detectPsfSources = pexConfig.ConfigurableField(
        target=measAlg.SourceDetectionTask,
        doc="Task to detect sources for PSF measurement, if ``doCalculatePsf`` is set.",
    )
    measurePsfSources = pexConfig.ConfigurableField(
        target=SingleFrameMeasurementTask,
        doc="Task to measure sources for PSF measurement, if ``doCalculatePsf`` is set."
    )
    measurePsf = pexConfig.ConfigurableField(
        target=MeasurePsfTask,
        doc="Task to measure the PSF of the coadd, if ``doCalculatePsf`` is set.",
    )
    effectiveWavelength = pexConfig.Field(
        doc="Effective wavelength of the filter, in nm."
        "Required if transmission curves aren't used."
        "Support for using transmission curves is to be added in DM-13668.",
        dtype=float,
    )
    bandwidth = pexConfig.Field(
        doc="Bandwidth of the physical filter, in nm."
        "Required if transmission curves aren't used."
        "Support for using transmission curves is to be added in DM-13668.",
        dtype=float,
    )

    def setDefaults(self):
        CompareWarpAssembleCoaddConfig.setDefaults(self)
        self.assembleStaticSkyModel.retarget(CompareWarpAssembleCoaddTask)
        self.doNImage = True
        self.assembleStaticSkyModel.warpType = self.warpType
        # The deepCoadd and nImage files will be overwritten by this Task, so don't write them the first time
        self.assembleStaticSkyModel.doNImage = False
        self.assembleStaticSkyModel.doWrite = False
        self.detectPsfSources.returnOriginalFootprints = False
        self.detectPsfSources.thresholdPolarity = "positive"
        # Only use bright sources for PSF measurement
        self.detectPsfSources.thresholdValue = 50
        self.detectPsfSources.nSigmaToGrow = 2
        # A valid star for PSF measurement should at least fill 5x5 pixels
        self.detectPsfSources.minPixels = 25
        # Use the variance plane to calculate signal to noise
        self.detectPsfSources.thresholdType = "pixel_stdev"
        # The signal to noise limit is good enough, while the flux limit is set
        # in dimensionless units and may not be appropriate for all data sets.
        self.measurePsf.starSelector["objectSize"].doFluxLimit = False
        # Ensure psf candidate size is as large as piff psf size.
        if (self.doCalculatePsf and self.measurePsf.psfDeterminer.name == "piff"
                and self.psfDeterminer["piff"].kernelSize > self.makePsfCandidates.kernelSize):
            self.makePsfCandidates.kernelSize = self.psfDeterminer["piff"].kernelSize


class DcrAssembleCoaddTask(CompareWarpAssembleCoaddTask):
    """Assemble DCR coadded images from a set of warps.

    Attributes
    ----------
    bufferSize : `int`
        The number of pixels to grow each subregion by to allow for DCR.

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
    If `splitSubfilters` is set, then each subfilter will be further sub-
    divided during the forward modeling step (only). This approximates using
    a higher number of subfilters that may be necessary for high airmass
    observations, but does not increase the number of free parameters in the
    fit. This is needed when there are high airmass observations which would
    otherwise have significant DCR even within a subfilter. Because calculating
    the shifted images takes most of the time, splitting the subfilters is
    turned off by way of the `splitThreshold` option for low-airmass
    observations that do not suffer from DCR within a subfilter.
    """

    ConfigClass = DcrAssembleCoaddConfig
    _DefaultName = "dcrAssembleCoadd"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.doCalculatePsf:
            self.schema = afwTable.SourceTable.makeMinimalSchema()
            self.makeSubtask("detectPsfSources", schema=self.schema)
            self.makeSubtask("measurePsfSources", schema=self.schema)
            self.makeSubtask("measurePsf", schema=self.schema)

    @utils.inheritDoc(pipeBase.PipelineTask)
    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docstring to be formatted with info from PipelineTask.runQuantum
        """
        Notes
        -----
        Assemble a coadd from a set of Warps.
        """
        inputData = butlerQC.get(inputRefs)

        # Construct skyInfo expected by run
        # Do not remove skyMap from inputData in case _makeSupplementaryData needs it
        skyMap = inputData["skyMap"]
        outputDataId = butlerQC.quantum.dataId

        inputData['skyInfo'] = makeSkyInfo(skyMap,
                                           tractId=outputDataId['tract'],
                                           patchId=outputDataId['patch'])

        # Construct list of input Deferred Datasets
        warpRefList = inputData['inputWarps']

        inputs = self.prepareInputs(warpRefList)
        self.log.info("Found %d %s", len(inputs.tempExpRefList),
                      self.getTempExpDatasetName(self.warpType))
        if len(inputs.tempExpRefList) == 0:
            self.log.warning("No coadd temporary exposures found")
            return

        supplementaryData = self._makeSupplementaryData(butlerQC, inputRefs, outputRefs)
        retStruct = self.run(inputData['skyInfo'], inputs.tempExpRefList, inputs.imageScalerList,
                             inputs.weightList, supplementaryData=supplementaryData)

        inputData.setdefault('brightObjectMask', None)
        for subfilter in range(self.config.dcrNumSubfilters):
            # Use the PSF of the stacked dcrModel, and do not recalculate the PSF for each subfilter
            retStruct.dcrCoadds[subfilter].setPsf(retStruct.coaddExposure.getPsf())
            self.processResults(retStruct.dcrCoadds[subfilter], inputData['brightObjectMask'], outputDataId)

        if self.config.doWrite:
            butlerQC.put(retStruct, outputRefs)
        return retStruct

    @utils.inheritDoc(AssembleCoaddTask)
    def _makeSupplementaryData(self, butlerQC, inputRefs, outputRefs):
        """Load the previously-generated template coadd.

        Returns
        -------
        templateCoadd : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``templateCoadd``
                Coadded exposure (`lsst.afw.image.ExposureF`).
        """
        templateCoadd = butlerQC.get(inputRefs.templateExposure)

        return pipeBase.Struct(templateCoadd=templateCoadd)

    def measureCoaddPsf(self, coaddExposure):
        """Detect sources on the coadd exposure and measure the final PSF.

        Parameters
        ----------
        coaddExposure : `lsst.afw.image.Exposure`
            The final coadded exposure.
        """
        table = afwTable.SourceTable.make(self.schema)
        detResults = self.detectPsfSources.run(table, coaddExposure, clearMask=False)
        coaddSources = detResults.sources
        self.measurePsfSources.run(
            measCat=coaddSources,
            exposure=coaddExposure
        )
        # Measure the PSF on the stacked subfilter coadds if possible.
        # We should already have a decent estimate of the coadd PSF, however,
        # so in case of any errors simply log them as a warning and use the
        # default PSF.
        try:
            psfResults = self.measurePsf.run(coaddExposure, coaddSources)
        except Exception as e:
            self.log.warning("Unable to calculate PSF, using default coadd PSF: %s", e)
        else:
            coaddExposure.setPsf(psfResults.psf)

    def prepareDcrInputs(self, templateCoadd, warpRefList, weightList):
        """Prepare the DCR coadd by iterating through the visitInfo of the input warps.

        Sets the property ``bufferSize``.

        Parameters
        ----------
        templateCoadd : `lsst.afw.image.ExposureF`
            The initial coadd exposure before accounting for DCR.
        warpRefList : `list` of `lsst.daf.butler.DeferredDatasetHandle`
            The data references to the input warped exposures.
        weightList : `list` of `float`
            The weight to give each input exposure in the coadd.
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
        sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))
        filterLabel = templateCoadd.getFilter()
        dcrShifts = []
        airmassDict = {}
        angleDict = {}
        psfSizeDict = {}
        for visitNum, warpExpRef in enumerate(warpRefList):
            visitInfo = warpExpRef.get(component="visitInfo")
            psf = warpExpRef.get(component="psf")
            visit = warpExpRef.dataId["visit"]
            # Just need a rough estimate; average positions are fine
            psfAvgPos = psf.getAveragePosition()
            psfSize = psf.computeShape(psfAvgPos).getDeterminantRadius()*sigma2fwhm
            airmass = visitInfo.getBoresightAirmass()
            parallacticAngle = visitInfo.getBoresightParAngle().asDegrees()
            airmassDict[visit] = airmass
            angleDict[visit] = parallacticAngle
            psfSizeDict[visit] = psfSize
            if self.config.doAirmassWeight:
                weightList[visitNum] *= airmass
            dcrShifts.append(np.max(np.abs(calculateDcr(visitInfo, templateCoadd.getWcs(),
                                                        self.config.effectiveWavelength,
                                                        self.config.bandwidth,
                                                        self.config.dcrNumSubfilters))))
        self.log.info("Selected airmasses:\n%s", airmassDict)
        self.log.info("Selected parallactic angles:\n%s", angleDict)
        self.log.info("Selected PSF sizes:\n%s", psfSizeDict)
        self.bufferSize = int(np.ceil(np.max(dcrShifts)) + 1)
        try:
            psf = self.selectCoaddPsf(templateCoadd, warpRefList)
        except Exception as e:
            self.log.warning("Unable to calculate restricted PSF, using default coadd PSF: %s", e)
        else:
            psf = templateCoadd.getPsf()
        dcrModels = DcrModel.fromImage(templateCoadd.maskedImage,
                                       self.config.dcrNumSubfilters,
                                       effectiveWavelength=self.config.effectiveWavelength,
                                       bandwidth=self.config.bandwidth,
                                       wcs=templateCoadd.getWcs(),
                                       filterLabel=filterLabel,
                                       psf=psf)
        return dcrModels

    @timeMethod
    def run(self, skyInfo, warpRefList, imageScalerList, weightList,
            supplementaryData=None):
        r"""Assemble the coadd.

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
        image and make them proper ``coaddExposure``\ s.

        Parameters
        ----------
        skyInfo : `lsst.pipe.base.Struct`
            Patch geometry information, from getSkyInfo
        warpRefList : `list` of `lsst.daf.butler.DeferredDatasetHandle`
            The data references to the input warped exposures.
        imageScalerList : `list` of `lsst.pipe.task.ImageScaler`
            The image scalars correct for the zero point of the exposures.
        weightList : `list` of `float`
            The weight to give each input exposure in the coadd
        supplementaryData : `lsst.pipe.base.Struct`
            Result struct returned by ``_makeSupplementaryData`` with attributes:

            ``templateCoadd``
                Coadded exposure (`lsst.afw.image.Exposure`).

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``coaddExposure``
                Coadded exposure (`lsst.afw.image.Exposure`).
            ``nImage``
                Exposure count image (`lsst.afw.image.ImageU`).
            ``dcrCoadds``
                `list` of coadded exposures for each subfilter.
            ``dcrNImages``
                `list` of exposure count images for each subfilter.
        """
        minNumIter = self.config.minNumIter or self.config.dcrNumSubfilters
        maxNumIter = self.config.maxNumIter or self.config.dcrNumSubfilters*3
        templateCoadd = supplementaryData.templateCoadd
        baseMask = templateCoadd.mask.clone()
        # The variance plane is for each subfilter
        # and should be proportionately lower than the full-band image
        baseVariance = templateCoadd.variance.clone()
        baseVariance /= self.config.dcrNumSubfilters
        spanSetMaskList = self.findArtifacts(templateCoadd, warpRefList, imageScalerList)
        # Note that the mask gets cleared in ``findArtifacts``, but we want to preserve the mask.
        templateCoadd.setMask(baseMask)
        badMaskPlanes = self.config.badMaskPlanes[:]
        # Note that is important that we do not add "CLIPPED" to ``badMaskPlanes``
        # This is because pixels in observations that are significantly affect by DCR
        # are likely to have many pixels that are both "DETECTED" and "CLIPPED",
        # but those are necessary to constrain the DCR model.
        badPixelMask = templateCoadd.mask.getPlaneBitMask(badMaskPlanes)

        stats = self.prepareStats(mask=badPixelMask)
        dcrModels = self.prepareDcrInputs(templateCoadd, warpRefList, weightList)
        if self.config.doNImage:
            dcrNImages, dcrWeights = self.calculateNImage(dcrModels, skyInfo.bbox, warpRefList,
                                                          spanSetMaskList, stats.ctrl)
            nImage = afwImage.ImageU(skyInfo.bbox)
            # Note that this nImage will be a factor of dcrNumSubfilters higher than
            # the nImage returned by assembleCoadd for most pixels. This is because each
            # subfilter may have a different nImage, and fractional values are not allowed.
            for dcrNImage in dcrNImages:
                nImage += dcrNImage
        else:
            dcrNImages = None

        subregionSize = geom.Extent2I(*self.config.subregionSize)
        nSubregions = (ceil(skyInfo.bbox.getHeight()/subregionSize[1])
                       * ceil(skyInfo.bbox.getWidth()/subregionSize[0]))
        subIter = 0
        for subBBox in self._subBBoxIter(skyInfo.bbox, subregionSize):
            modelIter = 0
            subIter += 1
            self.log.info("Computing coadd over patch %s subregion %s of %s: %s",
                          skyInfo.patchInfo.getIndex(), subIter, nSubregions, subBBox)
            dcrBBox = geom.Box2I(subBBox)
            dcrBBox.grow(self.bufferSize)
            dcrBBox.clip(dcrModels.bbox)
            modelWeights = self.calculateModelWeights(dcrModels, dcrBBox)
            subExposures = self.loadSubExposures(dcrBBox, stats.ctrl, warpRefList,
                                                 imageScalerList, spanSetMaskList)
            convergenceMetric = self.calculateConvergence(dcrModels, subExposures, subBBox,
                                                          warpRefList, weightList, stats.ctrl)
            self.log.info("Initial convergence : %s", convergenceMetric)
            convergenceList = [convergenceMetric]
            gainList = []
            convergenceCheck = 1.
            refImage = templateCoadd.image
            while (convergenceCheck > self.config.convergenceThreshold or modelIter <= minNumIter):
                gain = self.calculateGain(convergenceList, gainList)
                self.dcrAssembleSubregion(dcrModels, subExposures, subBBox, dcrBBox, warpRefList,
                                          stats.ctrl, convergenceMetric, gain,
                                          modelWeights, refImage, dcrWeights)
                if self.config.useConvergence:
                    convergenceMetric = self.calculateConvergence(dcrModels, subExposures, subBBox,
                                                                  warpRefList, weightList, stats.ctrl)
                    if convergenceMetric == 0:
                        self.log.warning("Coadd patch %s subregion %s had convergence metric of 0.0 which is "
                                         "most likely due to there being no valid data in the region.",
                                         skyInfo.patchInfo.getIndex(), subIter)
                        break
                    convergenceCheck = (convergenceList[-1] - convergenceMetric)/convergenceMetric
                    if (convergenceCheck < 0) & (modelIter > minNumIter):
                        self.log.warning("Coadd patch %s subregion %s diverged before reaching maximum "
                                         "iterations or desired convergence improvement of %s."
                                         " Divergence: %s",
                                         skyInfo.patchInfo.getIndex(), subIter,
                                         self.config.convergenceThreshold, convergenceCheck)
                        break
                    convergenceList.append(convergenceMetric)
                if modelIter > maxNumIter:
                    if self.config.useConvergence:
                        self.log.warning("Coadd patch %s subregion %s reached maximum iterations "
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

        dcrCoadds = self.fillCoadd(dcrModels, skyInfo, warpRefList, weightList,
                                   calibration=self.scaleZeroPoint.getPhotoCalib(),
                                   coaddInputs=templateCoadd.getInfo().getCoaddInputs(),
                                   mask=baseMask,
                                   variance=baseVariance)
        coaddExposure = self.stackCoadd(dcrCoadds)
        return pipeBase.Struct(coaddExposure=coaddExposure, nImage=nImage,
                               dcrCoadds=dcrCoadds, dcrNImages=dcrNImages)

    def calculateNImage(self, dcrModels, bbox, warpRefList, spanSetMaskList, statsCtrl):
        """Calculate the number of exposures contributing to each subfilter.

        Parameters
        ----------
        dcrModels : `lsst.pipe.tasks.DcrModel`
            Best fit model of the true sky after correcting chromatic effects.
        bbox : `lsst.geom.box.Box2I`
            Bounding box of the patch to coadd.
        warpRefList : `list` of `lsst.daf.butler.DeferredDatasetHandle`
            The data references to the input warped exposures.
        spanSetMaskList : `list` of `dict` containing spanSet lists, or `None`
            Each element of the `dict` contains the new mask plane name
            (e.g. "CLIPPED and/or "NO_DATA") as the key,
            and the list of SpanSets to apply to the mask.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd

        Returns
        -------
        dcrNImages : `list` of `lsst.afw.image.ImageU`
            List of exposure count images for each subfilter.
        dcrWeights : `list` of `lsst.afw.image.ImageF`
            Per-pixel weights for each subfilter.
            Equal to 1/(number of unmasked images contributing to each pixel).
        """
        dcrNImages = [afwImage.ImageU(bbox) for subfilter in range(self.config.dcrNumSubfilters)]
        dcrWeights = [afwImage.ImageF(bbox) for subfilter in range(self.config.dcrNumSubfilters)]
        for warpExpRef, altMaskSpans in zip(warpRefList, spanSetMaskList):
            exposure = warpExpRef.get(parameters={'bbox': bbox})
            visitInfo = exposure.getInfo().getVisitInfo()
            wcs = exposure.getInfo().getWcs()
            mask = exposure.mask
            if altMaskSpans is not None:
                self.applyAltMaskPlanes(mask, altMaskSpans)
            weightImage = np.zeros_like(exposure.image.array)
            weightImage[(mask.array & statsCtrl.getAndMask()) == 0] = 1.
            # The weights must be shifted in exactly the same way as the residuals,
            # because they will be used as the denominator in the weighted average of residuals.
            weightsGenerator = self.dcrResiduals(weightImage, visitInfo, wcs,
                                                 dcrModels.effectiveWavelength, dcrModels.bandwidth)
            for shiftedWeights, dcrNImage, dcrWeight in zip(weightsGenerator, dcrNImages, dcrWeights):
                dcrNImage.array += np.rint(shiftedWeights).astype(dcrNImage.array.dtype)
                dcrWeight.array += shiftedWeights
        # Exclude any pixels that don't have at least one exposure contributing in all subfilters
        weightsThreshold = 1.
        goodPix = dcrWeights[0].array > weightsThreshold
        for weights in dcrWeights[1:]:
            goodPix = (weights.array > weightsThreshold) & goodPix
        for subfilter in range(self.config.dcrNumSubfilters):
            dcrWeights[subfilter].array[goodPix] = 1./dcrWeights[subfilter].array[goodPix]
            dcrWeights[subfilter].array[~goodPix] = 0.
            dcrNImages[subfilter].array[~goodPix] = 0
        return (dcrNImages, dcrWeights)

    def dcrAssembleSubregion(self, dcrModels, subExposures, bbox, dcrBBox, warpRefList,
                             statsCtrl, convergenceMetric,
                             gain, modelWeights, refImage, dcrWeights):
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
        subExposures : `dict` of `lsst.afw.image.ExposureF`
            The pre-loaded exposures for the current subregion.
        bbox : `lsst.geom.box.Box2I`
            Bounding box of the subregion to coadd.
        dcrBBox : `lsst.geom.box.Box2I`
            Sub-region of the coadd which includes a buffer to allow for DCR.
        warpRefList : `list` of `lsst.daf.butler.DeferredDatasetHandle`
            The data references to the input warped exposures.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd.
        convergenceMetric : `float`
            Quality of fit metric for the matched templates of the input images.
        gain : `float`, optional
            Relative weight to give the new solution when updating the model.
        modelWeights : `numpy.ndarray` or `float`
            A 2D array of weight values that tapers smoothly to zero away from detected sources.
            Set to a placeholder value of 1.0 if ``self.config.useModelWeights`` is False.
        refImage : `lsst.afw.image.Image`
            A reference image used to supply the default pixel values.
        dcrWeights : `list` of `lsst.afw.image.Image`
            Per-pixel weights for each subfilter.
            Equal to 1/(number of unmasked images contributing to each pixel).
        """
        residualGeneratorList = []

        for warpExpRef in warpRefList:
            visit = warpExpRef.dataId["visit"]
            exposure = subExposures[visit]
            visitInfo = exposure.getInfo().getVisitInfo()
            wcs = exposure.getInfo().getWcs()
            templateImage = dcrModels.buildMatchedTemplate(exposure=exposure,
                                                           bbox=exposure.getBBox(),
                                                           order=self.config.imageInterpOrder,
                                                           splitSubfilters=self.config.splitSubfilters,
                                                           splitThreshold=self.config.splitThreshold,
                                                           amplifyModel=self.config.accelerateModel)
            residual = exposure.image.array - templateImage.array
            # Note that the variance plane here is used to store weights, not the actual variance
            residual *= exposure.variance.array
            # The residuals are stored as a list of generators.
            # This allows the residual for a given subfilter and exposure to be created
            # on the fly, instead of needing to store them all in memory.
            residualGeneratorList.append(self.dcrResiduals(residual, visitInfo, wcs,
                                                           dcrModels.effectiveWavelength,
                                                           dcrModels.bandwidth))

        dcrSubModelOut = self.newModelFromResidual(dcrModels, residualGeneratorList, dcrBBox, statsCtrl,
                                                   gain=gain,
                                                   modelWeights=modelWeights,
                                                   refImage=refImage,
                                                   dcrWeights=dcrWeights)
        dcrModels.assign(dcrSubModelOut, bbox)

    def dcrResiduals(self, residual, visitInfo, wcs, effectiveWavelength, bandwidth):
        """Prepare a residual image for stacking in each subfilter by applying the reverse DCR shifts.

        Parameters
        ----------
        residual : `numpy.ndarray`
            The residual masked image for one exposure,
            after subtracting the matched template.
        visitInfo : `lsst.afw.image.VisitInfo`
            Metadata for the exposure.
        wcs : `lsst.afw.geom.SkyWcs`
            Coordinate system definition (wcs) for the exposure.

        Yields
        ------
        residualImage : `numpy.ndarray`
            The residual image for the next subfilter, shifted for DCR.
        """
        if self.config.imageInterpOrder > 1:
            # Pre-calculate the spline-filtered residual image, so that step can be
            # skipped in the shift calculation in `applyDcr`.
            filteredResidual = ndimage.spline_filter(residual, order=self.config.imageInterpOrder)
        else:
            # No need to prefilter if order=1 (it will also raise an error)
            filteredResidual = residual
        # Note that `splitSubfilters` is always turned off in the reverse direction.
        # This option introduces additional blurring if applied to the residuals.
        dcrShift = calculateDcr(visitInfo, wcs, effectiveWavelength, bandwidth, self.config.dcrNumSubfilters,
                                splitSubfilters=False)
        for dcr in dcrShift:
            yield applyDcr(filteredResidual, dcr, useInverse=True, splitSubfilters=False,
                           doPrefilter=False, order=self.config.imageInterpOrder)

    def newModelFromResidual(self, dcrModels, residualGeneratorList, dcrBBox, statsCtrl,
                             gain, modelWeights, refImage, dcrWeights):
        """Calculate a new DcrModel from a set of image residuals.

        Parameters
        ----------
        dcrModels : `lsst.pipe.tasks.DcrModel`
            Current model of the true sky after correcting chromatic effects.
        residualGeneratorList : `generator` of `numpy.ndarray`
            The residual image for the next subfilter, shifted for DCR.
        dcrBBox : `lsst.geom.box.Box2I`
            Sub-region of the coadd which includes a buffer to allow for DCR.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd.
        gain : `float`
            Relative weight to give the new solution when updating the model.
        modelWeights : `numpy.ndarray` or `float`
            A 2D array of weight values that tapers smoothly to zero away from detected sources.
            Set to a placeholder value of 1.0 if ``self.config.useModelWeights`` is False.
        refImage : `lsst.afw.image.Image`
            A reference image used to supply the default pixel values.
        dcrWeights : `list` of `lsst.afw.image.Image`
            Per-pixel weights for each subfilter.
            Equal to 1/(number of unmasked images contributing to each pixel).

        Returns
        -------
        dcrModel : `lsst.pipe.tasks.DcrModel`
            New model of the true sky after correcting chromatic effects.
        """
        newModelImages = []
        for subfilter, model in enumerate(dcrModels):
            residualsList = [next(residualGenerator) for residualGenerator in residualGeneratorList]
            residual = np.sum(residualsList, axis=0)
            residual *= dcrWeights[subfilter][dcrBBox].array
            # `MaskedImage`s only support in-place addition, so rename for readability
            newModel = model[dcrBBox].clone()
            newModel.array += residual
            # Catch any invalid values
            badPixels = ~np.isfinite(newModel.array)
            newModel.array[badPixels] = model[dcrBBox].array[badPixels]
            if self.config.regularizeModelIterations > 0:
                dcrModels.regularizeModelIter(subfilter, newModel, dcrBBox,
                                              self.config.regularizeModelIterations,
                                              self.config.regularizationWidth)
            newModelImages.append(newModel)
        if self.config.regularizeModelFrequency > 0:
            dcrModels.regularizeModelFreq(newModelImages, dcrBBox, statsCtrl,
                                          self.config.regularizeModelFrequency,
                                          self.config.regularizationWidth)
        dcrModels.conditionDcrModel(newModelImages, dcrBBox, gain=gain)
        self.applyModelWeights(newModelImages, refImage[dcrBBox], modelWeights)
        return DcrModel(newModelImages, dcrModels.filter, dcrModels.effectiveWavelength,
                        dcrModels.bandwidth, dcrModels.psf,
                        dcrModels.mask, dcrModels.variance)

    def calculateConvergence(self, dcrModels, subExposures, bbox, warpRefList, weightList, statsCtrl):
        """Calculate a quality of fit metric for the matched templates.

        Parameters
        ----------
        dcrModels : `lsst.pipe.tasks.DcrModel`
            Best fit model of the true sky after correcting chromatic effects.
        subExposures : `dict` of `lsst.afw.image.ExposureF`
            The pre-loaded exposures for the current subregion.
        bbox : `lsst.geom.box.Box2I`
            Sub-region to coadd.
        warpRefList : `list` of `lsst.daf.butler.DeferredDatasetHandle`
            The data references to the input warped exposures.
        weightList : `list` of `float`
            The weight to give each input exposure in the coadd.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd.

        Returns
        -------
        convergenceMetric : `float`
            Quality of fit metric for all input exposures, within the sub-region.
        """
        significanceImage = np.abs(dcrModels.getReferenceImage(bbox))
        nSigma = 3.
        significanceImage += nSigma*dcrModels.calculateNoiseCutoff(dcrModels[1], statsCtrl,
                                                                   bufferSize=self.bufferSize)
        if np.max(significanceImage) == 0:
            significanceImage += 1.
        weight = 0
        metric = 0.
        metricList = {}
        for warpExpRef, expWeight in zip(warpRefList, weightList):
            visit = warpExpRef.dataId["visit"]
            exposure = subExposures[visit][bbox]
            singleMetric = self.calculateSingleConvergence(dcrModels, exposure, significanceImage, statsCtrl)
            metric += singleMetric
            metricList[visit] = singleMetric
            weight += 1.
        self.log.info("Individual metrics:\n%s", metricList)
        return 1.0 if weight == 0.0 else metric/weight

    def calculateSingleConvergence(self, dcrModels, exposure, significanceImage, statsCtrl):
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
            Statistics control object for coadd.

        Returns
        -------
        convergenceMetric : `float`
            Quality of fit metric for one exposure, within the sub-region.
        """
        convergeMask = exposure.mask.getPlaneBitMask(self.config.convergenceMaskPlanes)
        templateImage = dcrModels.buildMatchedTemplate(exposure=exposure,
                                                       bbox=exposure.getBBox(),
                                                       order=self.config.imageInterpOrder,
                                                       splitSubfilters=self.config.splitSubfilters,
                                                       splitThreshold=self.config.splitThreshold,
                                                       amplifyModel=self.config.accelerateModel)
        diffVals = np.abs(exposure.image.array - templateImage.array)*significanceImage
        refVals = np.abs(exposure.image.array + templateImage.array)*significanceImage/2.

        finitePixels = np.isfinite(diffVals)
        goodMaskPixels = (exposure.mask.array & statsCtrl.getAndMask()) == 0
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

    def fillCoadd(self, dcrModels, skyInfo, warpRefList, weightList, calibration=None, coaddInputs=None,
                  mask=None, variance=None):
        """Create a list of coadd exposures from a list of masked images.

        Parameters
        ----------
        dcrModels : `lsst.pipe.tasks.DcrModel`
            Best fit model of the true sky after correcting chromatic effects.
        skyInfo : `lsst.pipe.base.Struct`
            Patch geometry information, from getSkyInfo.
        warpRefList : `list` of `lsst.daf.butler.DeferredDatasetHandle`
            The data references to the input warped exposures.
        weightList : `list` of `float`
            The weight to give each input exposure in the coadd.
        calibration : `lsst.afw.Image.PhotoCalib`, optional
            Scale factor to set the photometric calibration of an exposure.
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
        refModel = dcrModels.getReferenceImage()
        for model in dcrModels:
            if self.config.accelerateModel > 1:
                model.array = (model.array - refModel)*self.config.accelerateModel + refModel
            coaddExposure = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
            if calibration is not None:
                coaddExposure.setPhotoCalib(calibration)
            if coaddInputs is not None:
                coaddExposure.getInfo().setCoaddInputs(coaddInputs)
            # Set the metadata for the coadd, including PSF and aperture corrections.
            self.assembleMetadata(coaddExposure, warpRefList, weightList)
            # Overwrite the PSF
            coaddExposure.setPsf(dcrModels.psf)
            coaddUtils.setCoaddEdgeBits(dcrModels.mask[skyInfo.bbox], dcrModels.variance[skyInfo.bbox])
            maskedImage = afwImage.MaskedImageF(dcrModels.bbox)
            maskedImage.image = model
            maskedImage.mask = dcrModels.mask
            maskedImage.variance = dcrModels.variance
            coaddExposure.setMaskedImage(maskedImage[skyInfo.bbox])
            coaddExposure.setPhotoCalib(self.scaleZeroPoint.getPhotoCalib())
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
        dcrBBox : `lsst.geom.box.Box2I`
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
        if not self.config.useModelWeights:
            return 1.0
        if self.config.modelWeightsWidth < 0:
            raise ValueError("modelWeightsWidth must not be negative if useModelWeights is set")
        convergeMask = dcrModels.mask.getPlaneBitMask(self.config.convergenceMaskPlanes)
        convergeMaskPixels = dcrModels.mask[dcrBBox].array & convergeMask > 0
        weights = np.zeros_like(dcrModels[0][dcrBBox].array)
        weights[convergeMaskPixels] = 1.
        weights = ndimage.gaussian_filter(weights, self.config.modelWeightsWidth)
        weights /= np.max(weights)
        return weights

    def applyModelWeights(self, modelImages, refImage, modelWeights):
        """Smoothly replace model pixel values with those from a
        reference at locations away from detected sources.

        Parameters
        ----------
        modelImages : `list` of `lsst.afw.image.Image`
            The new DCR model images from the current iteration.
            The values will be modified in place.
        refImage : `lsst.afw.image.MaskedImage`
            A reference image used to supply the default pixel values.
        modelWeights : `numpy.ndarray` or `float`
            A 2D array of weight values that tapers smoothly to zero away from detected sources.
            Set to a placeholder value of 1.0 if ``self.config.useModelWeights`` is False.
        """
        if self.config.useModelWeights:
            for model in modelImages:
                model.array *= modelWeights
                model.array += refImage.array*(1. - modelWeights)/self.config.dcrNumSubfilters

    def loadSubExposures(self, bbox, statsCtrl, warpRefList, imageScalerList, spanSetMaskList):
        """Pre-load sub-regions of a list of exposures.

        Parameters
        ----------
        bbox : `lsst.geom.box.Box2I`
            Sub-region to coadd.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd.
        warpRefList : `list` of `lsst.daf.butler.DeferredDatasetHandle`
            The data references to the input warped exposures.
        imageScalerList : `list` of `lsst.pipe.task.ImageScaler`
            The image scalars correct for the zero point of the exposures.
        spanSetMaskList : `list` of `dict` containing spanSet lists, or `None`
            Each element is dict with keys = mask plane name to add the spans to.

        Returns
        -------
        subExposures : `dict`
            The `dict` keys are the visit IDs,
            and the values are `lsst.afw.image.ExposureF`
            The pre-loaded exposures for the current subregion.
            The variance plane contains weights, and not the variance
        """
        zipIterables = zip(warpRefList, imageScalerList, spanSetMaskList)
        subExposures = {}
        for warpExpRef, imageScaler, altMaskSpans in zipIterables:
            exposure = warpExpRef.get(parameters={'bbox': bbox})
            visit = warpExpRef.dataId["visit"]
            if altMaskSpans is not None:
                self.applyAltMaskPlanes(exposure.mask, altMaskSpans)
            imageScaler.scaleMaskedImage(exposure.maskedImage)
            # Note that the variance plane here is used to store weights, not the actual variance
            exposure.variance.array[:, :] = 0.
            # Set the weight of unmasked pixels to 1.
            exposure.variance.array[(exposure.mask.array & statsCtrl.getAndMask()) == 0] = 1.
            # Set the image value of masked pixels to zero.
            # This eliminates needing the mask plane when stacking images in ``newModelFromResidual``
            exposure.image.array[(exposure.mask.array & statsCtrl.getAndMask()) > 0] = 0.
            subExposures[visit] = exposure
        return subExposures

    def selectCoaddPsf(self, templateCoadd, warpRefList):
        """Compute the PSF of the coadd from the exposures with the best seeing.

        Parameters
        ----------
        templateCoadd : `lsst.afw.image.ExposureF`
            The initial coadd exposure before accounting for DCR.
        warpRefList : `list` of `lsst.daf.butler.DeferredDatasetHandle`
            The data references to the input warped exposures.

        Returns
        -------
        psf : `lsst.meas.algorithms.CoaddPsf`
            The average PSF of the input exposures with the best seeing.
        """
        sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))
        # Note: ``ccds`` is a `lsst.afw.table.ExposureCatalog` with one entry per ccd and per visit
        # If there are multiple ccds, it will have that many times more elements than ``warpExpRef``
        ccds = templateCoadd.getInfo().getCoaddInputs().ccds
        templatePsf = templateCoadd.getPsf()
        # Just need a rough estimate; average positions are fine
        templateAvgPos = templatePsf.getAveragePosition()
        psfRefSize = templatePsf.computeShape(templateAvgPos).getDeterminantRadius()*sigma2fwhm
        psfSizes = np.zeros(len(ccds))
        ccdVisits = np.array(ccds["visit"])
        for warpExpRef in warpRefList:
            psf = warpExpRef.get(component="psf")
            visit = warpExpRef.dataId["visit"]
            psfAvgPos = psf.getAveragePosition()
            psfSize = psf.computeShape(psfAvgPos).getDeterminantRadius()*sigma2fwhm
            psfSizes[ccdVisits == visit] = psfSize
        # Note that the input PSFs include DCR, which should be absent from the DcrCoadd
        # The selected PSFs are those that have a FWHM less than or equal to the smaller
        # of the mean or median FWHM of the input exposures.
        sizeThreshold = min(np.median(psfSizes), psfRefSize)
        goodPsfs = psfSizes <= sizeThreshold
        psf = measAlg.CoaddPsf(ccds[goodPsfs], templateCoadd.getWcs(),
                               self.config.coaddPsf.makeControl())
        return psf
