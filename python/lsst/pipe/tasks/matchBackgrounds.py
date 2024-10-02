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

import lsstDebug
import numpy as np
from lsst.afw.image import LOCAL, PARENT, ExposureF, ImageF, Mask, MaskedImageF
from lsst.afw.math import (
    MEAN,
    MEANCLIP,
    MEANSQUARE,
    MEDIAN,
    NPOINT,
    STDEV,
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
from lsst.geom import Box2D, Box2I, PointI
from lsst.pex.config import ChoiceField, Field, ListField, RangeField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct, TaskError
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.utils.timer import timeMethod


class MatchBackgroundsConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch", "band"),
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
        dimensions=("skymap", "tract", "patch", "visit"),
        storageClass="Background",
        multiple=True,
    )
    matchedImageList = Output(
        doc="List of background-matched warps.",
        name="{inputCoaddName}Coadd_{warpType}Warp_bgMatched",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "visit"),
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
        default=0.4,
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
        default=0.2,
        min=0.0,
        max=1.0,
    )

    # Background matching
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
        default=["NO_DATA", "DETECTED", "DETECTED_NEGATIVE", "SAT", "BAD", "INTRP", "CR"],
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
        # TODO: Check that setting the mask planes here work - these planes
        # can vary from exposure to exposure, I think?
        # Aaron: I think only the bit values vary, not the names, which this is referencing.
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

        # Define a reference warp; 'warps' is modified in-place to exclude it
        refWarp, refInd = self._defineWarps(warps=warps, refWarpVisit=self.config.refWarpVisit)

        # Images must be scaled to a common ZP
        # Converting everything to nJy to accomplish this
        refExposure = refWarp.get()
        instFluxToNanojanskyRef = self._fluxScale(refExposure)

        self.log.info("Matching %d Exposures", numExp)

        # Creating a null BackgroundList object by fitting a blank image
        statsFlag = stringToStatisticsProperty(self.config.gridStatistic)
        self.statsCtrl.setNumSigmaClip(self.config.numSigmaClip)
        self.statsCtrl.setNumIter(self.config.numIter)

        # TODO: refactor below to construct blank bg model
        im = refExposure.getMaskedImage()
        blankIm = im.clone()
        blankIm.image.array *= 0

        width = blankIm.getWidth()
        height = blankIm.getHeight()
        nx = width // self.config.binSize
        if width % self.config.binSize != 0:
            nx += 1
        ny = height // self.config.binSize
        if height % self.config.binSize != 0:
            ny += 1

        bctrl = BackgroundControl(nx, ny, self.statsCtrl, statsFlag)
        bctrl.setUndersampleStyle(self.config.undersampleStyle)

        bkgd = makeBackground(blankIm, bctrl)
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
            # TODO: simplify what this prints?
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

        # TODO: more elegant solution than inserting blank model at ref ind?
        backgroundInfoList.insert(refInd, blank)
        refExposure.image /= instFluxToNanojanskyRef  # Back to cts
        matchedImageList.insert(refInd, refExposure)
        return Struct(backgroundInfoList=backgroundInfoList, matchedImageList=matchedImageList)

    @timeMethod
    def _defineWarps(self, warps, refWarpVisit=None):
        """Define the reference warp and list of comparison warps.

        If no reference warp ID is supplied, this method calculates an
        appropriate reference exposure from the supplied list of warps by
        minimizing a cost function that penalizes high background complexity
        (divergence from a plane), high variance, low global coverage, and low
        edge coverage.

        Parameters
        ----------
        warps : `list`[`~lsst.daf.butler.DeferredDatasetHandle`]
            List of warped exposures (of type `~lsst.afw.image.ExposureF`).
        refWarpVisit : `int`, optional
            Visit ID of the reference warp.
            If None, the best warp is chosen from the list of warps.

        Returns
        -------
        refWarp : `~lsst.afw.image.ExposureF`
            Reference warped exposure.
        refWarpIndex : `int`
            Index of the reference removed from the list of warps.

        Notes
        -----
        This method modifies the input list of warps in place by removing the
        reference warp from it.
        """
        # User-defined reference visit, if one has been supplied
        if refWarpVisit:
            warpVisits = [warpDDH.dataId["visit"] for warpDDH in warps]
            try:
                refWarpIndex = warpVisits.index(refWarpVisit)
                refWarpDDH = warps.pop(refWarpIndex)
                self.log.info("Using user-supplied reference visit %d", refWarpVisit)
                return refWarpDDH.get(), refWarpIndex
            except ValueError:
                raise TaskError(f"Reference visit {refWarpVisit} is not found in the list of warps.")

        # Extract mean/var/npoints for each warp
        warpChi2s = []  # Background goodness of fit
        warpVars = []  # Variance
        warpNPointsGlobal = []  # Global coverage
        warpNPointsEdge = []  # Edge coverage
        for warpDDH in warps:
            warp = warpDDH.get()
            instFluxToNanojansky = warp.getPhotoCalib().instFluxToNanojansky(1)
            warp.image *= instFluxToNanojansky  # Images in nJy to facilitate difference imaging
            warp.variance *= instFluxToNanojansky
            warpBg, _ = self._makeBackground(warp.getMaskedImage())

            # Return an approximation to the background
            approxCtrl = ApproximateControl(ApproximateControl.CHEBYSHEV, 1, 1, self.config.approxWeighting)
            warpApprox = warpBg.getApproximate(approxCtrl, self.undersampleStyle)
            warpBgSub = ImageF(warp.image.array - warpApprox.getImage().array)

            warpStats = makeStatistics(warpBgSub, warp.mask, VARIANCE | NPOINT, self.statsCtrl)

            bad_mask_bit_mask = warp.mask.getPlaneBitMask(self.config.badMaskPlanes)
            good = (warp.mask.array.astype(int) & bad_mask_bit_mask) == 0
            dof = len(good[good]) - 6  # Assuming eq. of plane
            warpChi2 = np.nansum(warpBgSub.array[good] ** 2 / warp.variance.array[good]) / dof
            warpVar, _ = warpStats.getResult(VARIANCE)
            warpNPointGlobal, _ = warpStats.getResult(NPOINT)
            warpNPointEdge = (
                np.sum(~np.isnan(warp.image.array[:, 0]))  # Left edge
                + np.sum(~np.isnan(warp.image.array[:, -1]))  # Right edge
                + np.sum(~np.isnan(warp.image.array[0, :]))  # Bottom edge
                + np.sum(~np.isnan(warp.image.array[-1, :]))  # Top edge
            )
            warpChi2s.append(warpChi2)
            warpVars.append(warpVar)
            warpNPointsGlobal.append(int(warpNPointGlobal))
            warpNPointsEdge.append(warpNPointEdge)

            self.log.info(
                "Sci exp. visit %d; BG fit Chi^2=%.1f, var=%.1f nJy, nPoints global=%d, nPoints edge=%d",
                warp.getInfo().getVisitInfo().id,
                warpChi2,
                warpVar,
                warpNPointGlobal,
                warpNPointEdge,
            )

        # Normalize mean/var/npoints to range from  0 to 1
        warpChi2sFrac = np.array(warpChi2s) / np.nanmax(warpChi2s)
        warpVarsFrac = np.array(warpVars) / np.nanmax(warpVars)
        warpNPointsGlobalFrac = np.nanmin(warpNPointsGlobal) / np.array(warpNPointsGlobal)
        warpNPointsEdgeFrac = np.nanmin(warpNPointsEdge) / np.array(warpNPointsEdge)

        # Calculate cost function values
        costFunctionVals = self.config.bestRefWeightChi2 * warpChi2sFrac
        costFunctionVals += self.config.bestRefWeightVariance * warpVarsFrac
        costFunctionVals += self.config.bestRefWeightGlobalCoverage * warpNPointsGlobalFrac
        costFunctionVals += self.config.bestRefWeightEdgeCoverage * warpNPointsEdgeFrac

        ind = np.nanargmin(costFunctionVals)
        refWarp = warps.pop(ind)
        self.log.info("Using best reference visit %d", refWarp.dataId["visit"])
        return refWarp, ind

    def _makeBackground(self, warp: MaskedImageF) -> tuple[BackgroundMI, BackgroundControl]:
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
        nx = warp.getWidth() // self.config.binSize
        ny = warp.getHeight() // self.config.binSize

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
        # TODO: this is deprecated
        if lsstDebug.Info(__name__).savefits:
            refExposure.writeFits(lsstDebug.Info(__name__).figpath + "refExposure.fits")
            sciExposure.writeFits(lsstDebug.Info(__name__).figpath + "sciExposure.fits")

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

        bkgd, bctrl = self._makeBackground(diffMI)

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

        # TODO: also deprecated; _gridImage() maybe can go?
        if lsstDebug.Info(__name__).savefits:
            sciExposure.writeFits(lsstDebug.Info(__name__).figpath + "sciMatchedExposure.fits")

        if lsstDebug.Info(__name__).savefig:
            bbox = Box2D(refExposure.getMaskedImage().getBBox())
            try:
                self._debugPlot(bgX, bgY, bgZ, bgdZ, bkgdImage, bbox, modelValueArr, resids)
            except Exception as e:
                self.log.warning("Debug plot not generated: %s", e)

        meanVar = makeStatistics(diffMI.getVariance(), diffMI.getMask(), MEANCLIP, self.statsCtrl).getValue()

        diffIm = diffMI.getImage()
        diffIm -= bkgdImage  # diffMI should now have a mean ~ 0
        del diffIm
        mse = makeStatistics(diffMI, MEANSQUARE, self.statsCtrl).getValue()

        outBkgd = approx if self.config.usePolynomial else bkgd
        # Convert this back into counts
        # TODO: is there a one-line way to do this?
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

    def _debugPlot(self, X, Y, Z, dZ, modelImage, bbox, model, resids):
        """
        Consider deleting this entirely
        Generate a plot showing the background fit and residuals.

        It is called when lsstDebug.Info(__name__).savefig = True.
        Saves the fig to lsstDebug.Info(__name__).figpath.
        Displays on screen if lsstDebug.Info(__name__).display = True.

        Parameters
        ----------
        X : `np.ndarray`, (N,)
            Array of x positions.
        Y : `np.ndarray`, (N,)
            Array of y positions.
        Z : `np.ndarray`
            Array of the grid values that were interpolated.
        dZ : `np.ndarray`, (len(Z),)
            Array of the error on the grid values.
        modelImage : `Unknown`
            Image of the model of the fit.
        model : `np.ndarray`, (len(Z),)
            Array of len(Z) containing the grid values predicted by the model.
        resids : `Unknown`
            Z - model.
        """
        import matplotlib.colors
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid

        zeroIm = MaskedImageF(Box2I(bbox))
        zeroIm += modelImage
        x0, y0 = zeroIm.getXY0()
        dx, dy = zeroIm.getDimensions()
        if len(Z) == 0:
            self.log.warning("No grid. Skipping plot generation.")
        else:
            max, min = np.max(Z), np.min(Z)
            norm = matplotlib.colors.normalize(vmax=max, vmin=min)
            maxdiff = np.max(np.abs(resids))
            diffnorm = matplotlib.colors.normalize(vmax=maxdiff, vmin=-maxdiff)
            rms = np.sqrt(np.mean(resids**2))
            fig = plt.figure(1, (8, 6))
            meanDz = np.mean(dZ)
            grid = ImageGrid(
                fig,
                111,
                nrows_ncols=(1, 2),
                axes_pad=0.1,
                share_all=True,
                label_mode="L",
                cbar_mode="each",
                cbar_size="7%",
                cbar_pad="2%",
                cbar_location="top",
            )
            im = grid[0].imshow(
                zeroIm.getImage().getArray(), extent=(x0, x0 + dx, y0 + dy, y0), norm=norm, cmap="Spectral"
            )
            im = grid[0].scatter(
                X, Y, c=Z, s=15.0 * meanDz / dZ, edgecolor="none", norm=norm, marker="o", cmap="Spectral"
            )
            im2 = grid[1].scatter(X, Y, c=resids, edgecolor="none", norm=diffnorm, marker="s", cmap="seismic")
            grid.cbar_axes[0].colorbar(im)
            grid.cbar_axes[1].colorbar(im2)
            grid[0].axis([x0, x0 + dx, y0 + dy, y0])
            grid[1].axis([x0, x0 + dx, y0 + dy, y0])
            grid[0].set_xlabel("model and grid")
            grid[1].set_xlabel("residuals. rms = %0.3f" % (rms))
            if lsstDebug.Info(__name__).savefig:
                fig.savefig(lsstDebug.Info(__name__).figpath + self.debugDataIdString + ".png")
            if lsstDebug.Info(__name__).display:
                plt.show()
            plt.clf()

    def _gridImage(self, maskedImage, binsize, statsFlag):
        """Private method to grid an image for debugging."""
        width, height = maskedImage.getDimensions()
        x0, y0 = maskedImage.getXY0()
        xedges = np.arange(0, width, binsize)
        yedges = np.arange(0, height, binsize)
        xedges = np.hstack((xedges, width))  # add final edge
        yedges = np.hstack((yedges, height))  # add final edge

        # Use lists/append to protect against the case where
        # a bin has no valid pixels and should not be included in the fit
        bgX = []
        bgY = []
        bgZ = []
        bgdZ = []

        for ymin, ymax in zip(yedges[0:-1], yedges[1:]):
            for xmin, xmax in zip(xedges[0:-1], xedges[1:]):
                subBBox = Box2I(
                    PointI(int(x0 + xmin), int(y0 + ymin)),
                    PointI(int(x0 + xmax - 1), int(y0 + ymax - 1)),
                )
                subIm = MaskedImageF(maskedImage, subBBox, PARENT, False)
                stats = makeStatistics(
                    subIm,
                    MEAN | MEANCLIP | MEDIAN | NPOINT | STDEV,
                    self.statsCtrl,
                )
                npoints, _ = stats.getResult(NPOINT)
                if npoints >= 2:
                    stdev, _ = stats.getResult(STDEV)
                    if stdev < self.config.gridStdevEpsilon:
                        stdev = self.config.gridStdevEpsilon
                    bgX.append(0.5 * (x0 + xmin + x0 + xmax))
                    bgY.append(0.5 * (y0 + ymin + y0 + ymax))
                    bgdZ.append(stdev / np.sqrt(npoints))
                    est, _ = stats.getResult(statsFlag)
                    bgZ.append(est)

        return np.array(bgX), np.array(bgY), np.array(bgZ), np.array(bgdZ)
