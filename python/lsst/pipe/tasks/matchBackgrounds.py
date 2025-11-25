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

__all__ = [
    "MatchBackgroundsConnections",
    "MatchBackgroundsConfig",
    "MatchBackgroundsTask",
    "ChooseReferenceVisitConfig",
    "ChooseReferenceVisitTask",
]

import numpy as np

from lsst.afw.image import ImageF, MaskedImageF
from lsst.afw.math import (
    MEANSQUARE,
    NPOINT,
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
from lsst.pex.config import ChoiceField, Config, ConfigField, ConfigurableField, Field, RangeField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct, Task
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.pipe.tasks.tractBackground import TractBackground, TractBackgroundConfig
from lsst.skymap import BaseSkyMap
from lsst.utils.timer import timeMethod


class ChooseReferenceVisitConfig(Config):

    tractBgModel = ConfigField(
        dtype=TractBackgroundConfig,
        doc="Background model for the entire tract",
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
        "Higher weights prefers exposures with low image variances. Ignored when ref visit supplied.",
        default=0.3,
        min=0.0,
        max=1.0,
    )
    bestRefWeightGlobalCoverage = RangeField(
        dtype=float,
        doc="Global coverage weight (total number of valid pixels) when calculating the best reference "
        "exposure. Higher weights prefer exposures with high coverage. Ignored when ref visit supplied.",
        default=0.4,
        min=0.0,
        max=1.0,
    )
    gridStatistic = ChoiceField(
        dtype=str,
        doc="Type of statistic to estimate pixel value for the grid points",
        default="MEANCLIP",
        allowed={"MEAN": "mean", "MEDIAN": "median", "MEANCLIP": "clipped mean"},
    )
    undersampleStyle = ChoiceField(
        dtype=str,
        doc="Behavior if there are too few points in the grid for requested interpolation style",
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


class ChooseReferenceVisitTask(Task):
    """Select a reference visit from a list of visits by minimizing a cost
    function

    Notes
    -----
    The reference exposure is chosen from the list of science exposures by
    minimizing a cost function that penalizes high background complexity
    (divergence from a plane), high variance, and low global coverage.
    The cost function is a weighted sum of these three metrics.
    The weights are set by the config parameters:

    - ``bestRefWeightChi2``
    - ``bestRefWeightVariance``
    - ``bestRefWeightGlobalCoverage``
    """

    # TODO: this is only used to select reference visits, so may become
    # redundant if we find a means to build reference images instead.
    ConfigClass = ChooseReferenceVisitConfig
    config: ChooseReferenceVisitConfig

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        # Fits on binned images only; masking controlled in tractBackground.py
        self.statsFlag = stringToStatisticsProperty(self.config.gridStatistic)
        self.statsCtrl = StatisticsControl()
        self.statsCtrl.setNanSafe(True)
        self.statsCtrl.setNumSigmaClip(self.config.numSigmaClip)
        self.statsCtrl.setNumIter(self.config.numIter)
        self.stringToInterpStyle = stringToInterpStyle(self.config.interpStyle)
        self.undersampleStyle = stringToUndersampleStyle(self.config.undersampleStyle)

    @timeMethod
    def _makeTractBackgrounds(self, warps, skyMap):
        """Create full tract models of all visit backgrounds

        Parameters
        ----------
        warps : `list`[`~lsst.daf.butler.DeferredDatasetHandle`]
            List of warped exposures (of type `~lsst.afw.image.ExposureF`).
            This is ordered by patch ID, then by visit ID
        skyMap : `lsst.skyMap.SkyMap`
           SkyMap for deriving the patch/tract dimensions

        Returns
        -------
        visitTractBackrounds : `dict` [`int`, `TractBackground`]
            Models of full tract backgrounds for all visits, in nanojanskies.
            Accessed by visit ID.
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
                workingWarp = warp.get()

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
                bgModelBase += bgModel

            visitTractBackgrounds[visits[i]] = bgModelBase

        return visitTractBackgrounds

    @timeMethod
    def _defineWarps(self, visitTractBackgrounds):
        """Define the reference visit

        This method calculates an appropriate reference visit from the
        supplied full tract visit backgrounds by minimizing a cost function
        that penalizes high background complexity (divergence from a plane),
        high variance within the warps comprising the visit, and low global
        coverage.

        Parameters
        ----------
        visitTractBackrounds : `dict` [`int`, `TractBackground`]
            Models of full tract backgrounds for all visits, in nanojanskies.
            Accessed by visit ID.
        Returns
        -------
        refVisId : `int`
            ID of the reference visit.
        """
        # Extract mean/var/npoints for each visit background model
        fitChi2s = []  # Background goodness of fit
        visitVars = []  # Variance of original image
        fitNPointsGlobal = []  # Global coverage
        visits = []  # To ensure dictionary key order is correct
        for vis in visitTractBackgrounds:
            visits.append(vis)
            # Fit a polynomial model to the full tract plane
            tractBg = visitTractBackgrounds[vis].getStatsImage()
            tractVar = visitTractBackgrounds[vis].getVarianceImage()
            fitBg, _ = fitBackground(tractBg, self.statsCtrl, self.statsFlag, self.config.undersampleStyle)
            # Weight the approximation fit by the binned image variance
            fitBg.getStatsImage().variance = tractVar

            # Return an approximation to the background
            approxCtrl = ApproximateControl(ApproximateControl.CHEBYSHEV, 1, 1, True)
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

            visitVar = np.nanmean(fitBg.getStatsImage().variance.array[good])
            fitNPointGlobal, _ = fitStats.getResult(NPOINT)
            fitChi2s.append(fitChi2)
            visitVars.append(visitVar)
            fitNPointsGlobal.append(int(fitNPointGlobal))

            self.log.info(
                "Sci exp. visit %d; BG fit Chi^2=%.2e, var=%.2f nJy, nPoints global=%d",
                vis,
                fitChi2,
                visitVar,
                fitNPointGlobal,
            )
        # Normalize mean/var/npoints to range from  0 to 1
        fitChi2sFrac = np.array(fitChi2s) / np.nanmax(fitChi2s)
        fitVarsFrac = np.array(visitVars) / np.nanmax(visitVars)
        fitNPointsGlobalFrac = np.nanmin(fitNPointsGlobal) / np.array(fitNPointsGlobal)

        # Calculate cost function values
        costFunctionVals = self.config.bestRefWeightChi2 * fitChi2sFrac
        costFunctionVals += self.config.bestRefWeightVariance * fitVarsFrac
        costFunctionVals += self.config.bestRefWeightGlobalCoverage * fitNPointsGlobalFrac

        ind = np.nanargmin(costFunctionVals)
        refVisitId = visits[ind]
        self.log.info("Using best reference visit %d", refVisitId)

        return refVisitId


class MatchBackgroundsConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "band"),
    defaultTemplates={
        "warpType": "psf_matched",
        "warpTypeSuffix": "",
    },
):
    # TODO: add connection for warped backgroundToPhotometricRatio
    warps = Input(
        doc=("Warps used to construct a list of matched backgrounds."),
        name="{warpType}_warp",
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
        name="{warpType}_warp_background_diff",  # TODO: settle on appropriate name
        dimensions=("skymap", "tract", "visit", "patch"),
        storageClass="Background",
        multiple=True,
    )
    matchedImageList = Output(
        doc="List of background-matched warps.",
        name="{warpType}_warp_background_matched",  # TODO: settle on appropriate name
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "visit", "patch"),
        multiple=True,
    )


class MatchBackgroundsConfig(PipelineTaskConfig, pipelineConnections=MatchBackgroundsConnections):

    refWarpVisit = Field[int](
        doc="Reference visit ID. If None, the best visit is chosen using the list of visits.",
        optional=True,
    )
    tractBgModel = ConfigField(
        dtype=TractBackgroundConfig,
        doc="Background model for the entire tract",
    )
    gridStatistic = ChoiceField(
        dtype=str,
        doc="Type of statistic to estimate pixel value for the grid points",
        default="MEANCLIP",
        allowed={"MEAN": "mean", "MEDIAN": "median", "MEANCLIP": "clipped mean"},
    )
    undersampleStyle = ChoiceField(
        dtype=str,
        doc="Behavior if there are too few points in the grid for requested interpolation style",
        default="REDUCE_INTERP_ORDER",
        allowed={
            "THROW_EXCEPTION": "throw an exception if there are too few points.",
            "REDUCE_INTERP_ORDER": "use an interpolation style with a lower order.",
        },
    )
    interpStyle = ChoiceField(
        dtype=str,
        doc="Algorithm to interpolate the background values. "
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
    reference = ConfigurableField(
        target=ChooseReferenceVisitTask, doc="Choose reference visit to match backgrounds to"
    )

    def validate(self):
        if self.undersampleStyle == "INCREASE_NXNYSAMPLE":
            raise RuntimeError("Invalid undersampleStyle: Polynomial fitting not implemented.")


class MatchBackgroundsTask(PipelineTask):
    """Match the backgrounds of a list of warped exposures to a reference

    Attributes
    ----------
    config : `MatchBackgroundsConfig`
        Configuration for this task.
    statsCtrl : `~lsst.afw.math.StatisticsControl`
        Statistics control object.

    Notes
    -----
    This task is a part of the background subtraction pipeline.  It matches the
    backgrounds of a list of warped science exposures to that of a reference
    image.
    """

    ConfigClass = MatchBackgroundsConfig
    config: MatchBackgroundsConfig
    _DefaultName = "matchBackgrounds"

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        # Fits on binned images only; masking controlled in tractBackground.py
        self.statsFlag = stringToStatisticsProperty(self.config.gridStatistic)
        self.statsCtrl = StatisticsControl()
        self.statsCtrl.setNanSafe(True)
        self.statsCtrl.setNumSigmaClip(self.config.numSigmaClip)
        self.statsCtrl.setNumIter(self.config.numIter)
        self.stringToInterpStyle = stringToInterpStyle(self.config.interpStyle)
        self.undersampleStyle = stringToUndersampleStyle(self.config.undersampleStyle)

        self.makeSubtask("reference")

    @timeMethod
    def run(self, warps, skyMap):
        """Match the backgrounds of a list of warped exposures to the same
        patches in a reference visit

        A reference visit ID will be chosen automatically if none is supplied.

        Parameters
        ----------
        warps : `list`[`~lsst.afw.image.Exposure`]
            List of warped science exposures to be background-matched.
        skyMap : `lsst.skyMap.SkyMap`
           SkyMap for deriving the patch/tract dimensions.

        Returns
        -------
        result : `~lsst.afw.math.BackgroundList`, `~lsst.afw.image.Exposure`
            Differential background models and associated background-matched
            images.
        """
        # TODO: include warped backgroundToPhotometricRatio correction
        if (numExp := len(warps)) < 1:
            self.log.warning("No exposures found!  Returning empty lists.")
            return Struct(backgroundInfoList=[], matchedImageList=[])

        if self.config.refWarpVisit is None:
            # Build FFP BG models of each visit
            visitTractBgs = self.reference._makeTractBackgrounds(warps, skyMap)
            # Choose a reference visit using those
            refVisId = self.reference._defineWarps(visitTractBgs)
        else:
            self.log.info("Using user-supplied reference visit %d", self.config.refWarpVisit)
            refVisId = self.config.refWarpVisit

        self.log.info("Matching %d Exposures", numExp)

        backgroundInfoList, matchedImageList = self.matchBackgrounds(warps, skyMap, refVisId)

        return Struct(backgroundInfoList=backgroundInfoList, matchedImageList=matchedImageList)

    @timeMethod
    def _makeTractDifferenceBackgrounds(self, warps, skyMap, refVisitId):
        """Create full tract models of all difference image backgrounds
        (reference visit - visit)

        Parameters
        ----------
        warps : `list`[`~lsst.daf.butler.DeferredDatasetHandle`]
            List of warped exposures (of type `~lsst.afw.image.ExposureF`).
            This is ordered by patch ID, then by visit ID
        skyMap : `lsst.skyMap.SkyMap`
           SkyMap for deriving the patch/tract dimensions.
        refVisitId : `int`
            Visit ID number for the chosen reference visit.

        Returns
        -------
        visitTractDifferenceBackrounds : `dict` [`int`, `TractBackground`]
            Models of full tract backgrounds for all difference images
            (reference visit - visit), in nanojanskies.
            Accessed by visit ID.
        """
        # First, separate warps by visit
        visits = np.unique([i.dataId["visit"] for i in warps])

        # Then build difference image background models for each visit & store
        visitTractDifferenceBackgrounds = {}
        for i in range(len(visits)):
            visitWarpDDFs = [j for j in warps if j.dataId["visit"] == visits[i]]
            refWarpDDFs = [j for j in warps if j.dataId["visit"] == refVisitId]
            refPatches = [j.dataId["patch"] for j in refWarpDDFs]
            # Set up empty full tract background model object
            bgModelBase = TractBackground(
                config=self.config.tractBgModel, skymap=skyMap, tract=warps[0].dataId["tract"]
            )

            bgModels = []
            for warp in visitWarpDDFs:
                msg = "Constructing FFP background model for reference visit %d - visit %d using %d patches"
                self.log.debug(
                    msg,
                    refVisitId,
                    visits[i],
                    len(visitWarpDDFs),
                )
                workingWarp = warp.get()

                patchId = warp.dataId["patch"]
                # On no overlap between working warp and reference visit, set
                # the image to all NaN
                try:
                    idx = refPatches.index(patchId)
                    refWarp = refWarpDDFs[idx].get()
                except ValueError:
                    refWarp = workingWarp.clone()
                    refWarp.image += np.nan
                workingWarp.image.array = refWarp.image.array - workingWarp.image.array

                bgModel = bgModelBase.clone()
                bgModel.addWarp(workingWarp)
                bgModels.append(bgModel)

            # Merge warp difference models to make a single full tract
            # background difference model
            for bgModel, warp in zip(bgModels, visitWarpDDFs):
                msg = (
                    "Patch %d: Merging %d unmasked pixels (%.1f%s of detector area) into full tract "
                    "difference background model"
                )
                self.log.debug(
                    msg,
                    warp.dataId["patch"],
                    bgModel._numbers.getArray().sum(),
                    100 * bgModel._numbers.getArray().sum() / workingWarp.getBBox().getArea(),
                    "%",
                )
                bgModelBase += bgModel

            # Fit full tract background to generate offset image
            if visits[i] != refVisitId:
                bgModelImage = bgModelBase.getStatsImage()
                # Note: this just extrapolates into regions of no overlap
                # between reference and visit
                bkgd, _ = fitBackground(
                    bgModelImage, self.statsCtrl, self.statsFlag, self.config.undersampleStyle
                )
                try:
                    bkgdImage = bkgd.getImageF(self.config.interpStyle, self.config.undersampleStyle)
                except Exception as e:
                    e.add_note(f"on image {warp.dataId}")
                    raise
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
                # Replace binned difference image w/best-fit model.
                # Resetting numbers to override interpolation
                bgModelBase._numbers.array[:] = 1e6  # Arbitrarily large value
                bgModelBase._values.array = bkgdImage.array * bgModelBase._numbers.array

            visitTractDifferenceBackgrounds[visits[i]] = bgModelBase

        return visitTractDifferenceBackgrounds

    @timeMethod
    def matchBackgrounds(self, warps, skyMap, refVisitId):
        """Match science visit's background level to that of reference visit

        Process creates binned images of the full focal plane (in tract
        coordinates) for all visit IDs, subtracts each from a similarly
        binned FFP reference image, then generates TractBackground
        objects.

        The TractBackground objects representing the difference image
        backgrounds are then used to generate 'offset' images for each warp
        comprising the full science exposure visit, which are then added to
        each warp to match the background to that of the reference visit at the
        warp's location within the tract.

        Best practice uses `psf_matched_warp` images without the
        detections mask plane set.  When using `direct_warp`, sources may bias
        the difference image background estimation.  Mask planes are set in
        TractBackgroundConfig.

        Fit diagnostics are also calculated and returned.

        Parameters
        ----------
        warps : `list`[`~lsst.daf.butler.DeferredDatasetHandle`]
            List of warped exposures (of type `~lsst.afw.image.ExposureF`).
            This is ordered by patch ID, then by visit ID
        skyMap : `lsst.skyMap.SkyMap`
           SkyMap for deriving the patch/tract dimensions.
        refVisitId : `int`
            Chosen reference visit ID to match to.

        Returns
        -------
        backgroundInfoList : `list`[`TractBackground`]
            List of all difference image backgrounds used to match to reference
            visit warps, in nanojanskies.
        matchedImageList : `list`[`~lsst.afw.image.ExposureF`]
            List of all background-matched warps, in nanojanskies.
        """
        visits = np.unique([i.dataId["visit"] for i in warps])
        self.log.info("Processing %d visits", len(visits))

        backgroundInfoList = []
        matchedImageList = []
        diffTractBackgrounds = self._makeTractDifferenceBackgrounds(warps, skyMap, refVisitId)

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
            tractBg = diffTractBackgrounds[visId]
            diffModel = tractBg.toWarpBackground(im)
            bkgdIm = diffModel.getImage()
            maskIm.image += bkgdIm

            backgroundInfoList.append(diffModel)
            matchedImageList.append(im)

        return backgroundInfoList, matchedImageList


def fitBackground(
    warp: MaskedImageF, statsCtrl, statsFlag, undersampleStyle
) -> tuple[BackgroundMI, BackgroundControl]:
    """Generate a simple binned background masked image for warped or other
    data

    Parameters
    ----------
    warp: `~lsst.afw.image.MaskedImageF`
        Warped exposure for which to estimate background.
    statsCtrl : `~lsst.afw.math.StatisticsControl`
        Statistics control object.
    statsFlag : `~lsst.afw.math.Property`
        Statistics control type.
    undersampleStyle : `str`
        Behavior if there are too few points in the grid for requested
        interpolation style.

    Returns
    -------
    bkgd: `~lsst.afw.math.BackgroundMI`
        Background model of masked warp.
    bgCtrl: `~lsst.afw.math.BackgroundControl`
        Background control object.
    """
    # This only accesses pre-binned images, so no scaling necessary
    nx = warp.getWidth()
    ny = warp.getHeight()

    bgCtrl = BackgroundControl(nx, ny, statsCtrl, statsFlag)
    bgCtrl.setUndersampleStyle(undersampleStyle)
    bkgd = makeBackground(warp, bgCtrl)

    return bkgd, bgCtrl
