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
from lsst.afw.image import LOCAL, ImageF, Mask, MaskedImageF
from lsst.afw.math import (
    MEANCLIP,
    MEANSQUARE,
    NPOINT,
    VARIANCE,
    ApproximateControl,
    BackgroundControl,
    BackgroundList,
    BackgroundMI,
    StatisticsControl,
    makeBackground,
    makeStatistics,
    stringToInterpStyle,
    stringToStatisticsProperty,
    stringToUndersampleStyle,
)
from lsst.pex.config import ChoiceField, ConfigField, Field, ListField, RangeField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct, TaskError
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.pipe.tasks.background import TractBackground, TractBackgroundConfig
from lsst.utils.timer import timeMethod


class MatchBackgroundsConnections(
    PipelineTaskConnections,
    # How to get it to do visit, warp?
    # Need to kill all collections w/new dataset types to try other combos here...
    # https://pipelines.lsst.io/v/weekly/modules/lsst.pipe.base/creating-a-pipelinetask.html#pipelinetask-processing-multiple-datasets
    # dimensions=("skymap", "tract", "patch", "band"),
    dimensions=("skymap", "tract", "band"),  # Don't want to mix bands...
    defaultTemplates={
        "inputCoaddName": "deep",
        "outputCoaddName": "deep",
        "warpType": "psfMatched",
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
    backgroundInfoList = Output(
        doc="List of differential backgrounds, with goodness of fit params.",
        name="psfMatchedWarpBackground_diff",  # This needs to change
        # dimensions=("skymap", "tract", "patch", "visit"),
        dimensions=("skymap", "tract", "visit", "patch"),
        storageClass="Background",
        multiple=True,
    )
    matchedImageList = Output(
        doc="List of background-matched warps.",
        name="{inputCoaddName}Coadd_{warpType}Warp_bgMatched",
        storageClass="ExposureF",
        # dimensions=("skymap", "tract", "patch", "visit"),
        dimensions=("skymap", "tract", "visit", "patch"),
        multiple=True,
    )


class MatchBackgroundsConfig(PipelineTaskConfig, pipelineConnections=MatchBackgroundsConnections):

    # Reference warp selection
    refWarpVisit = Field[int](
        doc="Visit ID of the reference warp. If None, the best warp is chosen from the list of warps.",
        optional=True,
    )
    bestRefWeightChi2 = RangeField(
        dtype=float,
        doc="Mean background goodness of fit statistic weight when calculating the best reference exposure. "
        "Higher weights prefer exposures with flatter backgrounds. Ignored when ref visit supplied.",
        default=0.2,
        min=0.0,
        max=1.0,
    )
    bestRefWeightVariance = RangeField(
        dtype=float,
        doc="Image variance weight when calculating the best reference exposure. "
        "Higher weights prefers exposures with low image variances. Ignored when ref visit supplied",
        default=0.2,
        min=0.0,
        max=1.0,
    )
    bestRefWeightGlobalCoverage = RangeField(
        dtype=float,
        doc="Global coverage weight (total number of valid pixels) when calculating the best reference "
        "exposure. Higher weights prefer exposures with high coverage. Ignored when a ref visit supplied.",
        default=0.2,
        min=0.0,
        max=1.0,
    )
    bestRefWeightEdgeCoverage = RangeField(
        dtype=float,
        doc="Edge coverage weight (number of valid edge pixels) when calculating the best reference "
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
        default=False,
    )
    order = Field[int](
        doc="Order of Chebyshev polynomial background model. Ignored if ``usePolynomial=False``.",
        default=8,
    )
    badMaskPlanes = ListField[str](
        doc="Names of mask planes to ignore while estimating the background.",
        default=[
            "NO_DATA",
            "DETECTED",
            "DETECTED_NEGATIVE",
            "SAT",
            "BAD",
            "INTRP",
            "CR",
        ],
        itemCheck=lambda x: x in Mask().getMaskPlaneDict(),
    )
    gridStatistic = ChoiceField(
        dtype=str,
        doc="Type of statistic to estimate pixel value for the grid points.",
        default="MEANCLIP",
        allowed={"MEAN": "mean", "MEDIAN": "median", "MEANCLIP": "clipped mean"},
    )
    undersampleStyle = ChoiceField(
        dtype=str,
        doc="Behaviour if there are too few points in the grid for requested interpolation style. "
        "Note: choice ``INCREASE_NXNYSAMPLE`` only allowed for ``usePolynomial=True``.",
        default="REDUCE_INTERP_ORDER",
        allowed={
            "THROW_EXCEPTION": "throw an exception if there are too few points.",
            "REDUCE_INTERP_ORDER": "use an interpolation style with a lower order.",
            "INCREASE_NXNYSAMPLE": "Increase the number of samples used to make the interpolation grid.",
        },
    )
    binSize = Field[int](
        doc="Bin size for gridding the difference image and fitting a spatial model.",
        default=256,
    )
    chi2BinSize = Field[int](
        doc="Bin size for gridding images when choosing best reference exposure.",
        default=1024,
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
    (divergence from a plane), high variance, low global coverage, and low
    coverage along image edges.
    The cost function is a weighted sum of these four metrics.
    The weights are set by the config parameters:
    - ``bestRefWeightChi2``
    - ``bestRefWeightVariance``
    - ``bestRefWeightGlobalCoverage``
    - ``bestRefWeightEdgeCoverage``


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
        self.statsFlag = stringToStatisticsProperty(self.config.gridStatistic)
        self.statsCtrl = StatisticsControl()
        self.statsCtrl.setAndMask(Mask.getPlaneBitMask(self.config.badMaskPlanes))
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
    def run(self, warps):
        """Match the backgrounds of a list of warped exposures to a reference.

        A reference warp will be chosen automatically if none is supplied.

        Parameters
        ----------
        warps : `list`[`~lsst.afw.image.Exposure`]
            List of warped science exposures to be background-matched.

        Returns
        -------
        result : `~lsst.afw.math.BackgroundList`
            Differential background model
            Add this to the science exposure to match the reference exposure.

        Raises
        ------
        RuntimeError
            Raised if an exposure does not exist on disk.
        """
        if (numExp := len(warps)) < 1:
            raise TaskError("No exposures to match")
        # TODO: store ref visit ID between runs, to skip selection process?

        # First, build FFP BG models of each visit
        visitTractBgs = self._makeTractBackgrounds(warps)

        # Define a reference warp; 'warps' is modified in-place to exclude it
        refBg, refVis, bkgd = self._defineWarps(visitTractBgs, refVisitId=self.config.refWarpVisit)

        # Images must be scaled to a common ZP
        # Converting everything to nJy to accomplish this
        refExposure = refWarp.get()
        instFluxToNanojanskyRef = self._fluxScale(refExposure)

        self.log.info("Matching %d Exposures", numExp)

        # Blank ref warp background as reference background
        bkgdIm = bkgd.getImageF()
        bkgdStatsIm = bkgd.getStatsImage()
        bkgdIm *= 0
        bkgdStatsIm *= 0
        blank = BackgroundList(
            (
                bkgd,
                stringToInterpStyle(self.config.interpStyle),
                stringToUndersampleStyle(self.config.undersampleStyle),
                ApproximateControl.UNKNOWN,
                0,
                0,
                False,
            )
        )

        backgroundInfoList = []
        matchedImageList = []
        for exp in warps:
            self.log.info(
                "Matching background of %s to %s",
                exp.dataId,
                refWarp.dataId,
            )
            toMatchExposure = exp.get()
            instFluxToNanojansky = self._fluxScale(toMatchExposure)
            try:
                backgroundInfoStruct = self.matchBackgrounds(
                    refExposure=refExposure,
                    sciExposure=toMatchExposure,
                )
                backgroundInfoStruct.isReference = False
            except Exception as e:
                self.log.warning("Failed to fit background %s: %s", exp.dataId, e)
                backgroundInfoStruct = blank

            backgroundInfoList.append(backgroundInfoStruct)
            toMatchExposure.image /= instFluxToNanojansky  # Back to cts
            matchedImageList.append(toMatchExposure)

        backgroundInfoList.insert(refInd, blank)
        refExposure.image /= instFluxToNanojanskyRef  # Back to cts
        matchedImageList.insert(refInd, refExposure)
        return Struct(backgroundInfoList=backgroundInfoList, matchedImageList=matchedImageList)

    @timeMethod
    def _makeTractBackgrounds(self, warps, refWarpDDF=None):
        """Create full tract model of the backgrounds of all visits.

        Parameters
        ----------
        warps : `list`[`~lsst.daf.butler.DeferredDatasetHandle`]
            List of warped exposures (of type `~lsst.afw.image.ExposureF`).
            This is ordered by patch ID, then by visit ID

        refWarpDDF : ``~lsst.daf.butler.DeferredDatasetHandle` optional
            Chosen reference warp to match to, if supplied

        Returns
        -------
        visitTractBackrounds : `dict`{`TractBackground`}
            Models of full tract backgrounds for all visits, accessed by visit
            IDs
        """
        # First, separate warps by visit
        visits = np.unique([i.dataId["visit"] for i in warps])

        # Then build background models for each visit, and store
        visitTractBackgrounds = {}
        for i in range(len(visits)):
            # Find all the warps for the visit
            visitWarps = [j for j in warps if j.dataId["visit"] == visits[i]]
            # Set up empty full tract background model object
            bgModelBase = TractBackground(self.config.tractBgModel)

            bgModels = []
            for warp in visitWarps:
                msg = "Constructing FFP background model for visit %d using %d patches"
                self.log.debug(
                    msg,
                    visits[i],
                    len(warps),
                )
                if refWarpDDF is not None:
                    msg = "Doing difference imaging: reference warp visit ID: %d"
                    self.log.debug(msg, refWarpDDF.dataId["visit"])
                visitWarp = warp.get()
                # If a reference image is supplied, make a BG of the difference
                if refWarpDDF is not None:
                    refWarp = refWarpDDF.get()
                    visitWarp = refWarp - visitWarp
                bgModel = bgModelBase.clone()
                bgModel.addWarp(visitWarp)
                bgModels.append(bgModel)

            # Merge warp models to make a single full tract background model
            for bgModel, warp in zip(bgModels, visitWarps):
                msg = (
                    "Patch %d: Merging %d unmasked pixels (%.1f%s of detector area) into tract plane BG "
                    "model"
                )
                self.log.debug(
                    msg,
                    warp.dataId["patch"],
                    bgModel._numbers.getArray().sum(),
                    100 * bgModel._numbers.getArray().sum() / visitWarp.getBBox().getArea(),
                    "%",
                )
                bgModelBase.merge(bgModel)
            visitTractBackgrounds[visits[i]] = bgModelBase
        return visitTractBackgrounds

    @timeMethod
    def _defineWarps(self, visitTractBackgrounds, refVisitId=None):
        """Define the reference visit and list of comparison visits.

        If no reference visit ID is supplied, this method calculates an
        appropriate reference exposure from the supplied list of visit
        backgrounds by minimizing a cost function that penalizes high
        background complexity (divergence from a plane), high variance, and low
        global coverage.

        Parameters
        ----------
        visitTractBackgrounds : `dict`{`TractBackground`}
            Models of full tract backgrounds for all visits, accessed by visit
            IDs
        refVisitId : `int`, optional
            ID of the reference visit.
            If None, the best visit is chosen using the dictionary of existing
            backgrounds.

        Returns
        -------
        refBg : `~lsst.afw.math.BackgroundMI`
            Reference background to match to.
        refVis : `int`
            Index of the reference visit removed from the dictionary.
        fitBg : `~lsst.afw.math.BackgroundMI`
            Temporary background model, used to make a blank BG for the ref

        Notes
        -----
        This method modifies the input list of warps in place by removing the
        reference warp from it.

        """
        # User-defined reference visit, if one has been supplied
        if refVisitId:
            visits = [visId for visId in visitTractBackgrounds.keys()]
            try:
                refTractBackground = visitTractBackgrounds.pop(refVisitId)
                self.log.info("Using user-supplied reference visit %d", refVisitId)
                # TODO: need to return a background object here!
                return refTractBackground, refVisitId
            except ValueError:
                raise TaskError(f"Reference visit {refVisitId} is not found in the list of warps.")

        # Extract mean/var/npoints for each warp
        fitChi2s = []  # Background goodness of fit
        # warpVars = []  # Variance
        fitNPointsGlobal = []  # Global coverage
        # warpNPointsEdge = []  # Edge coverage
        visits = []  # To ensure dictionary key order is correct
        for vis in visitTractBackgrounds:
            visits.append(vis)
            # Fit a model to the FFP
            # TODO: need a variance plane in the tractBg as well
            tractBg = visitTractBackgrounds[vis].getStatsImage()
            fitBg, _ = self._makeBackground(tractBg, binSize=1)
            fitBg.getStatsImage().variance = ImageF(np.sqrt(fitBg.getStatsImage().image.array))

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
            # fitVar, _ = fitStats.getResult(VARIANCE)  # Will add this back later
            fitNPointGlobal, _ = fitStats.getResult(NPOINT)
            # This becomes pointless: it's a circle within a square.  Coverage is what's needed for that.
            # warpNPointEdge = (
            #     np.sum(~np.isnan(warp.image.array[:, 0]))  # Left edge
            #     + np.sum(~np.isnan(warp.image.array[:, -1]))  # Right edge
            #     + np.sum(~np.isnan(warp.image.array[0, :]))  # Bottom edge
            #     + np.sum(~np.isnan(warp.image.array[-1, :]))  # Top edge
            # )
            fitChi2s.append(fitChi2)
            # warpVars.append(warpVar)
            fitNPointsGlobal.append(int(fitNPointGlobal))
            # warpNPointsEdge.append(warpNPointEdge)

            self.log.info(
                # "Sci exp. visit %d; BG fit Chi^2=%.1f, var=%.1f nJy, nPoints global=%d, nPoints edge=%d",
                "Sci exp. visit %d; BG fit Chi^2=%.1f, nPoints global=%d",
                vis,
                fitChi2,
                # warpVar,
                fitNPointGlobal,
                # warpNPointEdge,
            )
        # Normalize mean/var/npoints to range from  0 to 1
        fitChi2sFrac = np.array(fitChi2s) / np.nanmax(fitChi2s)
        # warpVarsFrac = np.array(warpVars) / np.nanmax(warpVars)
        fitNPointsGlobalFrac = np.nanmin(fitNPointsGlobal) / np.array(fitNPointsGlobal)
        # warpNPointsEdgeFrac = np.nanmin(warpNPointsEdge) / np.array(warpNPointsEdge)

        # Calculate cost function values
        costFunctionVals = self.config.bestRefWeightChi2 * fitChi2sFrac
        # costFunctionVals += self.config.bestRefWeightVariance * warpVarsFrac
        costFunctionVals += self.config.bestRefWeightGlobalCoverage * fitNPointsGlobalFrac
        # costFunctionVals += self.config.bestRefWeightEdgeCoverage * warpNPointsEdgeFrac

        ind = np.nanargmin(costFunctionVals)
        refVis = visits[ind]
        refBg = visitTractBackgrounds.pop(refVis)
        self.log.info("Using best reference visit %d", refVis)
        return refBg, refVis, fitBg

    def _makeBackground(self, warp: MaskedImageF, binSize) -> tuple[BackgroundMI, BackgroundControl]:
        """Generate a simple binned background masked image for warped data.

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
    def matchBackgrounds(self, refExposure, sciExposure):
        """Match science exposure's background level to that of reference
        exposure.

        Process creates a difference image of the reference exposure minus the
        science exposure, and then generates an afw.math.Background object. It
        assumes (but does not require/check) that the mask plane already has
        detections set. If detections have not been set/masked, sources will
        bias the background estimation.

        The 'background' of the difference image is smoothed by spline
        interpolation (by the Background class) or by polynomial interpolation
        by the Approximate class. This model of difference image is added to
        the science exposure in memory.

        Fit diagnostics are also calculated and returned.

        Parameters
        ----------
        refExposure : `lsst.afw.image.Exposure`
            Reference exposure.
        sciExposure : `lsst.afw.image.Exposure`
            Science exposure; ultimately modified by changing the background
            level to match that of the reference exposure.

        Returns
        -------
        model : `~lsst.afw.math.BackgroundMI`
            Background model of difference image, reference - science
        """
        # Check Configs for polynomials:
        if self.config.usePolynomial:
            x, y = sciExposure.getDimensions()
            shortSideLength = min(x, y)
            if shortSideLength < self.config.binSize:
                raise ValueError(
                    "%d = config.binSize > shorter dimension = %d" % (self.config.binSize, shortSideLength)
                )
            npoints = shortSideLength // self.config.binSize
            if shortSideLength % self.config.binSize != 0:
                npoints += 1

            # If order of polynomial to be fit > number of bins to fit, error
            if self.config.order > npoints - 1:
                raise ValueError("%d = config.order > npoints - 1 = %d" % (self.config.order, npoints - 1))

        # Check that exposures are same shape
        if sciExposure.getDimensions() != refExposure.getDimensions():
            wSci, hSci = sciExposure.getDimensions()
            wRef, hRef = refExposure.getDimensions()
            raise RuntimeError(
                "Exposures are different dimensions. sci:(%i, %i) vs. ref:(%i, %i)" % (wSci, hSci, wRef, hRef)
            )

        im = refExposure.getMaskedImage()
        diffMI = im.clone()
        diffMI -= sciExposure.getMaskedImage()

        bkgd, bctrl = self._makeBackground(diffMI, binSize=self.config.binSize)

        # Some config and input checks if config.usePolynomial:
        # 1) Check that order/bin size make sense:
        # 2) Change binsize or order if underconstrained.
        if self.config.usePolynomial:
            order = self.config.order
            bgX, bgY, bgZ, bgdZ = self._gridImage(diffMI, self.config.binSize, self.statsFlag)
            minNumberGridPoints = min(len(set(bgX)), len(set(bgY)))
            if len(bgZ) == 0:
                raise ValueError("No overlap with reference. Nothing to match")
            elif minNumberGridPoints <= self.config.order:
                # must lower order or raise number of bins, or throw exception
                if self.config.undersampleStyle == "THROW_EXCEPTION":
                    raise ValueError("Image does not cover enough of ref image for order and binsize")
                elif self.config.undersampleStyle == "REDUCE_INTERP_ORDER":
                    self.log.warning("Reducing order to %d", (minNumberGridPoints - 1))
                    order = minNumberGridPoints - 1
                elif self.config.undersampleStyle == "INCREASE_NXNYSAMPLE":
                    newBinSize = (minNumberGridPoints * self.config.binSize) // (self.config.order + 1)
                    bctrl.setNxSample(newBinSize)
                    bctrl.setNySample(newBinSize)
                    bkgd = makeBackground(diffMI, bctrl)  # do over
                    self.log.warning("Decreasing binsize to %d", newBinSize)

            # If there is no variance in any image pixels,
            # do not weight bins by inverse variance
            isUniformImageDiff = not np.any(bgdZ > self.config.gridStdevEpsilon)
            weightByInverseVariance = False if isUniformImageDiff else self.config.approxWeighting

        # Add offset to sciExposure
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
                "Background/Approximation failed to interp image %s: %s" % (sciExposure.dataId, e)
            )

        instFluxToNanojansky = sciExposure.getPhotoCalib().instFluxToNanojansky(1)
        sciMI = sciExposure.getMaskedImage()
        sciMI += bkgdImage
        del sciMI  # sciExposure is now a BG-matched image

        # Need RMS from fit: 2895 will replace this:
        rms = 0.0
        bgX, bgY, bgZ, bgdZ = self._gridImage(diffMI, self.config.binSize, self.statsFlag)
        x0, y0 = diffMI.getXY0()
        modelValueArr = np.empty(len(bgZ))
        for i in range(len(bgX)):
            modelValueArr[i] = bkgdImage[int(bgX[i] - x0), int(bgY[i] - y0), LOCAL]
        resids = bgZ - modelValueArr
        rms = np.sqrt(np.mean(resids[~np.isnan(resids)] ** 2))

        meanVar = makeStatistics(diffMI.getVariance(), diffMI.getMask(), MEANCLIP, self.statsCtrl).getValue()

        diffIm = diffMI.getImage()
        diffIm -= bkgdImage  # diffMI should now have a mean ~ 0
        del diffIm
        mse = makeStatistics(diffMI, MEANSQUARE, self.statsCtrl).getValue()

        outBkgd = approx if self.config.usePolynomial else bkgd
        # Convert this back into counts
        statsIm = outBkgd.getStatsImage()
        statsIm /= instFluxToNanojansky
        bkgdIm = outBkgd.getImageF()
        bkgdIm /= instFluxToNanojansky

        self.log.info(
            "Visit %d; difference BG fit RMS=%.1f cts, matched MSE=%.1f cts, mean variance=%.1f cts",
            sciExposure.getInfo().getVisitInfo().id,
            rms,
            mse,
            meanVar,
        )
        # TODO: verify this is correct (borrowed from background.py)
        return BackgroundList(
            (
                outBkgd,
                stringToInterpStyle(self.config.interpStyle),
                stringToUndersampleStyle(self.config.undersampleStyle),
                ApproximateControl.UNKNOWN,
                0,
                0,
                False,
            )
        )
