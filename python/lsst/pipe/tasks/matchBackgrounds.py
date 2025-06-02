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

__all__ = ["MatchBackgroundsConnections", "MatchBackgroundsConfig", "MatchBackgroundsTask"]

import numpy as np

from lsst.afw.image import ImageF, MaskedImageF
from lsst.afw.math import (
    MEANSQUARE,
    NPOINT,
    STDEV,
    VARIANCE,
    ApproximateControl,
    BackgroundControl,
    BackgroundMI,
    StatisticsControl,
    makeBackground,
    makeStatistics,
    stringToInterpStyle,
    stringToStatisticsProperty,
    stringToUndersampleStyle,
)
from lsst.pex.config import ChoiceField, ConfigField, Field, RangeField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct, TaskError
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.pipe.tasks.background import TractBackground, TractBackgroundConfig
from lsst.skymap import BaseSkyMap
from lsst.utils.timer import timeMethod


class MatchBackgroundsConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "band"),
    defaultTemplates={
        "inputCoaddName": "deep",
        "outputCoaddName": "deep",
        "warpType": "direct",
        "warpTypeSuffix": "",
    },
):

    warps = Input(
        doc=("Warps used to construct a list of matched backgrounds."),
        name="{inputCoaddName}Coadd_{warpType}Warp",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "visit"),
        deferLoad=True,
        multiple=True,
    )
    skyMap = Input(
        doc="Input definition of geometry/bbox and projection/wcs for warped exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    backgroundInfoList = Output(
        doc="List of differential backgrounds, with goodness of fit params.",
        name="{warpType}WarpBackground_diff",  # TODO: settle on appropriate name
        dimensions=("skymap", "tract", "visit", "patch"),
        storageClass="Background",
        multiple=True,
    )
    matchedImageList = Output(
        doc="List of background-matched warps.",
        name="{inputCoaddName}Coadd_{warpType}Warp_bgMatched",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "visit", "patch"),
        multiple=True,
    )
    # TODO: binned full-tract images as outputs, for visualization?


class MatchBackgroundsConfig(PipelineTaskConfig, pipelineConnections=MatchBackgroundsConnections):

    # Reference warp selection
    refWarpVisit = Field[int](
        doc="Reference visit ID. If None, the best visit is chosen using the list of warps.",
        optional=True,
    )
    bestRefWeightChi2 = RangeField(
        dtype=float,
        doc="Mean background goodness of fit statistic weight when calculating the best reference exposure. "
        "Higher weights prefer exposures with flatter backgrounds. Ignored when ref visit supplied.",
        default=0.3,
        min=0.0,
        max=1.0,
    )
    bestRefWeightVariance = RangeField(
        dtype=float,
        doc="Image variance weight when calculating the best reference exposure. "
        "Higher weights prefers exposures with low image variances. Ignored when ref visit supplied",
        default=0.3,
        min=0.0,
        max=1.0,
    )
    bestRefWeightGlobalCoverage = RangeField(
        dtype=float,
        doc="Global coverage weight (total number of valid pixels) when calculating the best reference "
        "exposure. Higher weights prefer exposures with high coverage. Ignored when a ref visit supplied.",
        default=0.4,
        min=0.0,
        max=1.0,
    )

    # Background matching
    tractBgModel = ConfigField(
        dtype=TractBackgroundConfig,
        doc="Background model for the entire tract",
    )
    usePolynomial = Field[bool](
        doc="Fit background difference with a Chebychev polynomial interpolation? "
        "If False, fit with spline interpolation instead.",
        default=True,
    )
    order = Field[int](
        doc="Order of Chebyshev polynomial background model. Ignored if ``usePolynomial=False``.",
        default=1,
    )
    gridStatistic = ChoiceField(
        dtype=str,
        doc="Type of statistic to estimate pixel value for the grid points.",
        default="MEANCLIP",
        allowed={"MEAN": "mean", "MEDIAN": "median", "MEANCLIP": "clipped mean"},
    )
    # TODO: binning is done apart from fitting now, making INCREASE_NXNYSAMPLE
    # option unusable here.  Unsure if this will cause problems.
    undersampleStyle = ChoiceField(
        dtype=str,
        doc="Behaviour if there are too few points in the grid for requested interpolation style. ",
        default="REDUCE_INTERP_ORDER",
        allowed={
            "THROW_EXCEPTION": "throw an exception if there are too few points.",
            "REDUCE_INTERP_ORDER": "use an interpolation style with a lower order.",
        },
    )
    interpStyle = ChoiceField(
        dtype=str,
        doc="Algorithm to interpolate the background values; ignored if ``usePolynomial=True``."
        "Maps to an enum; see afw.math.Background for more information.",
        default="AKIMA_SPLINE",
        allowed={
            "CONSTANT": "Use a single constant value.",
            "LINEAR": "Use linear interpolation.",
            "NATURAL_SPLINE": "A cubic spline with zero second derivative at endpoints.",
            "AKIMA_SPLINE": "A higher-level non-linear spline that is more robust to outliers.",
            "NONE": "No background estimation is to be attempted.",
        },
    )
    numSigmaClip = Field[int](
        doc="Sigma for outlier rejection. Ignored if ``gridStatistic != 'MEANCLIP'``.",
        default=3,
    )
    numIter = Field[int](
        doc="Number of iterations of outlier rejection. Ignored if ``gridStatistic != 'MEANCLIP'``.",
        default=3,
    )

    approxWeighting = Field[bool](
        doc="Use inverse-variance weighting when approximating the background offset model? This will fail "
        "when the background offset is constant (usually only the case in testing with artificial images)."
        "Only applied if ``usePolynomial=True``.",
        default=True,
    )
    gridStdevEpsilon = RangeField(
        dtype=float,
        doc="Tolerance on almost zero standard deviation in a background-offset grid bin. If all bins have a "
        "standard deviation below this value, the background offset model is approximated without "
        "inverse-variance weighting. Only applied if ``usePolynomial=True``.",
        default=1e-8,
        min=0.0,
    )


class MatchBackgroundsTask(PipelineTask):
    """Match the backgrounds of a list of warped exposures to a reference.

    This task is a part of the background subtraction pipeline.
    It matches the backgrounds of a list of science exposures to a reference
    science exposure.
    The reference exposure is chosen from the list of science exposures by
    minimizing a cost function that penalizes high background complexity
    (divergence from a plane), high variance, and low global coverage.
    The cost function is a weighted sum of these three metrics.
    The weights are set by the config parameters:
    - ``bestRefWeightChi2``
    - ``bestRefWeightVariance``
    - ``bestRefWeightGlobalCoverage``

    Attributes
    ----------
    config : `MatchBackgroundsConfig`
        Configuration for this task.
    statsCtrl : `~lsst.afw.math.StatisticsControl`
        Statistics control object.
    """

    ConfigClass = MatchBackgroundsConfig
    config: MatchBackgroundsConfig
    _DefaultName = "matchBackgrounds"

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        # Fits on binned images only; masking controlled in background.py
        self.statsFlag = stringToStatisticsProperty(self.config.gridStatistic)
        self.statsCtrl = StatisticsControl()
        self.statsCtrl.setNanSafe(True)
        self.statsCtrl.setNumSigmaClip(self.config.numSigmaClip)
        self.statsCtrl.setNumIter(self.config.numIter)
        self.stringToInterpStyle = stringToInterpStyle(self.config.interpStyle)
        self.undersampleStyle = stringToUndersampleStyle(self.config.undersampleStyle)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    @timeMethod
    def run(self, warps, skyMap):
        """Match the backgrounds of a list of warped exposures to the same
        patches in a reference visit.

        A reference visit ID will be chosen automatically if none is supplied.

        Parameters
        ----------
        warps : `list`[`~lsst.afw.image.Exposure`]
            List of warped science exposures to be background-matched.
        skyMap : `lsst.skyMap.SkyMap`
           SkyMap for deriving the patch/tract dimensions

        Returns
        -------
        result : `~lsst.afw.math.BackgroundList`, `~lsst.afw.image.Exposure`
            Differential background models and associated background-matched
            images.

        Raises
        ------
        RuntimeError
            Raised if an exposure does not exist on disk.
        """
        # TODO: matching currently done in fluence, not surface brightness,
        # units.  Current solution for filling in non-overlapping regions
        # between ref and other images is fitting a plane, but this fails as
        # plane is typically buried beneath warping Jacobian pattern.  Unsure
        # how to resolve this issue, but it's critical to resolve for BG-
        # matching to be viable.
        if (numExp := len(warps)) < 1:
            raise TaskError("No exposures to match")

        if self.config.refWarpVisit is None:
            # Build FFP BG models of each visit
            visitTractBgs = self._makeTractBackgrounds(warps, skyMap)
            # Choose a reference visit using those
            refVisId = self._defineWarps(visitTractBgs)
        else:
            self.log.info("Using user-supplied reference visit %d", self.config.refWarpVisit)
            refVisId = self.config.refWarpVisit

        self.log.info("Matching %d Exposures", numExp)

        backgroundInfoList, matchedImageList = self.matchBackgrounds(warps, skyMap, refVisId)

        # TODO: costly, but consider a post-hoc check on the match quality?

        return Struct(backgroundInfoList=backgroundInfoList, matchedImageList=matchedImageList)

    @timeMethod
    def _makeTractBackgrounds(self, warps, skyMap, refVisitId=None):
        """If no reference visit ID is supplied, create full tract models of
        the backgrounds of all visits.
        If a reference visit ID is supplied, create full tract models of the
        difference image backgrounds between all visits and the reference
        visit.

        Parameters
        ----------
        warps : `list`[`~lsst.daf.butler.DeferredDatasetHandle`]
            List of warped exposures (of type `~lsst.afw.image.ExposureF`).
            This is ordered by patch ID, then by visit ID
        skyMap : `lsst.skyMap.SkyMap`
           SkyMap for deriving the patch/tract dimensions

        refVisitId : `int` optional
            Chosen reference visit ID to match to, if supplied

        Returns
        -------
        visitTractBackrounds : `dict`{`TractBackground`}
            Models of full tract backgrounds (or difference image backgrounds)
            for all visits, in nanojanskies.  Accessed by visit ID.

        Notes
        -----
        Input warps, including reference if ID is supplied, are converted in-
        place to nanojanskies.
        """
        # First, separate warps by visit
        visits = np.unique([i.dataId["visit"] for i in warps])

        # Then build background models for each visit and store
        visitTractBackgrounds = {}
        for i in range(len(visits)):
            visitWarpDDFs = [j for j in warps if j.dataId["visit"] == visits[i]]
            # Set up empty full tract background model object
            bgModelBase = TractBackground(
                config=self.config.tractBgModel, skymap=skyMap, tract=warps[0].dataId["tract"]
            )

            bgModels = []
            for warp in visitWarpDDFs:
                msg = "Constructing FFP background model for visit %d using %d patches"
                self.log.debug(
                    msg,
                    visits[i],
                    len(visitWarpDDFs),
                )
                if refVisitId is not None:
                    msg = "Doing difference imaging: reference warp visit ID: %d"
                    self.log.debug(msg, refVisitId)
                workingWarp = warp.get()
                self._fluxScale(workingWarp)

                # If a reference visit is supplied, makes model of difference
                # image backgrounds
                if refVisitId is not None:
                    patchId = warp.dataId["patch"]
                    refWarpDDFs = [j for j in warps if j.dataId["visit"] == refVisitId]
                    refPatches = [j.dataId["patch"] for j in refWarpDDFs]
                    # On no overlap between working warp and reference visit,
                    # set the image to all NaN
                    try:
                        idx = refPatches.index(patchId)
                        refWarp = refWarpDDFs[idx].get()
                    except ValueError:
                        refWarp = workingWarp.clone()
                        refWarp.image += np.nan
                    self._fluxScale(refWarp)
                    workingWarp.image.array = refWarp.image.array - workingWarp.image.array
                bgModel = bgModelBase.clone()
                bgModel.addWarp(workingWarp)
                bgModels.append(bgModel)

            # Merge warp models to make a single full tract background model
            for bgModel, warp in zip(bgModels, visitWarpDDFs):
                msg = (
                    "Patch %d: Merging %d unmasked pixels (%.1f%s of detector area) into full tract "
                    "background model"
                )
                self.log.debug(
                    msg,
                    warp.dataId["patch"],
                    bgModel._numbers.getArray().sum(),
                    100 * bgModel._numbers.getArray().sum() / workingWarp.getBBox().getArea(),
                    "%",
                )
                bgModelBase.merge(bgModel)

            # When working with diff images, fit full tract background to
            # extrapolate a model into where visit and ref have no overlap
            if refVisitId is not None and visits[i] != refVisitId:
                # Some config and input checks if config.usePolynomial:
                # 1) Check that order/bin size make sense
                # 2) Change order if underconstrained
                bgModelImage = bgModelBase.getStatsImage()
                if self.config.usePolynomial:
                    order = self.config.order
                    dimX, dimY = bgModelImage.array.shape
                    stats = makeStatistics(bgModelImage, NPOINT | STDEV, self.statsCtrl)
                    npoints, _ = stats.getResult(NPOINT)
                    stdev, _ = stats.getResult(STDEV)
                    if stdev < self.config.gridStdevEpsilon:
                        stdev = self.config.gridStdevEpsilon
                    minNumberGridPoints = min(dimX, dimY)
                    if npoints == 0:
                        raise ValueError("No overlap with reference. Nothing to match")
                    elif minNumberGridPoints <= order:
                        # Must lower order or throw exception
                        if self.config.undersampleStyle == "THROW_EXCEPTION":
                            raise ValueError("Image does not cover enough of ref image for order and binsize")
                        elif self.config.undersampleStyle == "REDUCE_INTERP_ORDER":
                            self.log.warning("Reducing order to %d", (minNumberGridPoints - 1))
                            order = minNumberGridPoints - 1

                    # TODO: we fit the full tract image, which has already been
                    # binned, so we set binSize=1 when fitting.  But this
                    # results in an all 0s variance image, meaning can't weight
                    # by inverse variance when doing the fit.
                    weightByInverseVariance = False

                bkgd, _ = self._makeBackground(bgModelImage, binSize=1)  # Already binned
                try:
                    if self.config.usePolynomial:
                        actrl = ApproximateControl(
                            ApproximateControl.CHEBYSHEV, order, order, weightByInverseVariance
                        )
                        undersampleStyle = stringToUndersampleStyle(self.config.undersampleStyle)
                        approx = bkgd.getApproximate(actrl, undersampleStyle)
                        bkgdImage = approx.getImage()
                    else:
                        bkgdImage = bkgd.getImageF(self.config.interpStyle, self.config.undersampleStyle)
                except Exception as e:
                    raise RuntimeError(
                        "Background/Approximation failed to interp image %s: %s" % (warp.dataId, e)
                    )
                # Calculate RMS and MSE of fit and print as log
                resids = ImageF(bgModelImage.array - bkgdImage.array)
                rms = np.sqrt(np.nanmean(resids.array**2))
                mse = makeStatistics(resids, MEANSQUARE, self.statsCtrl).getValue()

                self.log.info(
                    "Visit %d; difference BG fit RMS=%.2f nJy, matched MSE=%.2f nJy",
                    visits[i],
                    rms,
                    mse,
                )
                # Replace binned difference image w/best-fit model; this is our
                # offset image
                bgModelBase._numbers /= bgModelBase._numbers
                bgModelBase._numbers.array[np.isnan(bgModelBase._numbers.array)] = 1.0
                bgModelBase._values = bkgdImage

            visitTractBackgrounds[visits[i]] = bgModelBase
        return visitTractBackgrounds

    @timeMethod
    def _defineWarps(self, visitTractBackgrounds):
        """Define the reference visit.

        This method calculates an appropriate reference visit from the
        supplied full tract visit backgrounds by minimizing a cost function
        that penalizes high background complexity (divergence from a plane),
        high variance, and low global coverage.

        Parameters
        ----------
        visitTractBackgrounds : `dict`{`TractBackground`}
            Models of full tract backgrounds for all visits, accessed by visit
            IDs

        Returns
        -------
        refVisId : `int`
            ID of the reference visit.
        """
        # Extract mean/var/npoints for each visit background model
        fitChi2s = []  # Background goodness of fit
        fitVars = []  # Variance
        fitNPointsGlobal = []  # Global coverage
        visits = []  # To ensure dictionary key order is correct
        for vis in visitTractBackgrounds:
            visits.append(vis)
            # Fit a polynomial model to the full tract plane
            tractBg = visitTractBackgrounds[vis].getStatsImage()
            fitBg, _ = self._makeBackground(tractBg, binSize=1)
            # TODO: as stated above, fitting a pre-binned image results in a
            # null variance image.  But we want to add variance into the cost
            # function.  How best to do that?  Below is a bad temporary
            # solution, just assuming variance = mean
            fitBg.getStatsImage().variance = ImageF(tractBg.array)

            # Return an approximation to the background
            approxCtrl = ApproximateControl(ApproximateControl.CHEBYSHEV, 1, 1, self.config.approxWeighting)
            fitApprox = fitBg.getApproximate(approxCtrl, self.undersampleStyle)

            fitBgSub = MaskedImageF(ImageF(tractBg.array - fitApprox.getImage().array))
            bad_mask_bit_mask = fitBgSub.mask.getPlaneBitMask("BAD")
            fitBgSub.mask.array[np.isnan(fitBgSub.image.array)] = bad_mask_bit_mask

            fitStats = makeStatistics(fitBgSub.image, fitBgSub.mask, VARIANCE | NPOINT, self.statsCtrl)

            good = (fitBgSub.mask.array.astype(int) & bad_mask_bit_mask) == 0
            dof = len(good[good]) - 6  # Assuming eq. of plane
            fitChi2 = (
                np.nansum(fitBgSub.image.array[good] ** 2 / fitBg.getStatsImage().variance.array[good]) / dof
            )
            fitVar, _ = fitStats.getResult(VARIANCE)
            fitNPointGlobal, _ = fitStats.getResult(NPOINT)
            fitChi2s.append(fitChi2)
            fitVars.append(fitVar)
            fitNPointsGlobal.append(int(fitNPointGlobal))

            self.log.info(
                "Sci exp. visit %d; BG fit Chi^2=%.2f, var=%.2f nJy, nPoints global=%d",
                vis,
                fitChi2,
                fitVar,
                fitNPointGlobal,
            )
        # Normalize mean/var/npoints to range from  0 to 1
        fitChi2sFrac = np.array(fitChi2s) / np.nanmax(fitChi2s)
        fitVarsFrac = np.array(fitVars) / np.nanmax(fitVars)
        fitNPointsGlobalFrac = np.nanmin(fitNPointsGlobal) / np.array(fitNPointsGlobal)

        # Calculate cost function values
        costFunctionVals = self.config.bestRefWeightChi2 * fitChi2sFrac
        costFunctionVals += self.config.bestRefWeightVariance * fitVarsFrac
        costFunctionVals += self.config.bestRefWeightGlobalCoverage * fitNPointsGlobalFrac

        ind = np.nanargmin(costFunctionVals)
        refVisitId = visits[ind]
        self.log.info("Using best reference visit %d", refVisitId)
        return refVisitId

    def _makeBackground(self, warp: MaskedImageF, binSize) -> tuple[BackgroundMI, BackgroundControl]:
        """Generate a simple binned background masked image for warped or other
        data.

        Parameters
        ----------
        warp: `~lsst.afw.image.MaskedImageF`
            Warped exposure for which to estimate background.

        Returns
        -------
        bkgd: `~lsst.afw.math.BackgroundMI`
            Background model of masked warp.
        bgCtrl: `~lsst.afw.math.BackgroundControl`
            Background control object.
        """
        # TODO: leaving as-is for now, but currently this only fits the pre-
        # binned images, meaning binSize is always 1.  But we need a solution
        # to the 0 variance image issue, which might mean we need a variable
        # binSize still.
        nx = warp.getWidth() // binSize
        ny = warp.getHeight() // binSize

        bgCtrl = BackgroundControl(nx, ny, self.statsCtrl, self.statsFlag)
        bgCtrl.setUndersampleStyle(self.config.undersampleStyle)
        bkgd = makeBackground(warp, bgCtrl)

        return bkgd, bgCtrl

    def _fluxScale(self, exposure):
        """Scales image to nJy flux using photometric calibration.

        Parameters
        ----------
        exposure: `lsst.afw.image._exposure.ExposureF`
            Exposure to scale.

        Returns
        -------
        fluxZp: `float`
            Counts to nanojanskies conversion factor
        """
        fluxZp = exposure.getPhotoCalib().instFluxToNanojansky(1)
        exposure.image *= fluxZp
        return fluxZp

    @timeMethod
    def matchBackgrounds(self, warps, skyMap, refVisitId):
        """Match science exposures' background level to that of reference
        exposure.

        Process creates binned images of the full focal plane (in tract
        coordinates) for all visit IDs, subtracts each from a similarly
        binned FFP reference image, then generates TractBackground
        objects.  It assumes (but does not require/check) that the mask planes
        already have detections set.  If detections have not been set/masked,
        sources will bias the difference image background estimation.

        The TractBackground objects representing the difference image
        backgrounds are then used to generate 'offset' images for each warp
        comprising the full science exposure visit, which are then added to
        each warp to match the background to that of the reference visit at the
        warp's location within the tract.

        Fit diagnostics are also calculated and returned.

        Parameters
        ----------
        warps : `list`[`~lsst.daf.butler.DeferredDatasetHandle`]
            List of warped exposures (of type `~lsst.afw.image.ExposureF`).
            This is ordered by patch ID, then by visit ID
        skyMap : `lsst.skyMap.SkyMap`
           SkyMap for deriving the patch/tract dimensions
        refVisitId : `int`
            Chosen reference visit ID to match to

        Returns
        -------
        backgroundInfoList : `list`[`TractBackground`]
            List of all difference image backgrounds used to match to reference
            visit warps, in counts
        matchedImageList : `list`[`~lsst.afw.image.ExposureF`]
            List of all background-matched warps, in counts
        """
        visits = np.unique([i.dataId["visit"] for i in warps])
        self.log.info("Processing %d visits", len(visits))

        backgroundInfoList = []
        matchedImageList = []
        diffTractBackgrounds = self._makeTractBackgrounds(warps, skyMap, refVisitId)

        # Reference visit doesn't need an offset image, so use all 0's
        im = warps[0].get()  # Use arbitrary image as base
        bkgd = diffTractBackgrounds[refVisitId].toWarpBackground(im)
        blank = bkgd.getImage()
        blank *= 0

        for warp in warps:
            visId = warp.dataId["visit"]
            if visId == refVisitId:
                backgroundInfoList.append(bkgd)  # Just append a 0 image
                matchedImageList.append(warp.get())
                continue
            self.log.info(
                "Matching background of %s to same patch in visit %s",
                warp.dataId,
                refVisitId,
            )
            im = warp.get()
            maskIm = im.getMaskedImage()
            # Matching must be done at common zeropoint
            instFluxToNanojansky = self._fluxScale(im)
            tractBg = diffTractBackgrounds[visId]
            diffModel = tractBg.toWarpBackground(im)
            bkgdIm = diffModel.getImage()
            maskIm.image += bkgdIm
            # Then convert everything back to counts
            maskIm.image /= instFluxToNanojansky
            bkgdIm /= instFluxToNanojansky

            backgroundInfoList.append(diffModel)
            matchedImageList.append(im)

        return backgroundInfoList, matchedImageList
