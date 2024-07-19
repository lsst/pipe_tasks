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
from lsst.afw.image import LOCAL, PARENT, Mask, MaskedImageF
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


class MatchBackgroundsConfig(PipelineTaskConfig, pipelineConnections=MatchBackgroundsConnections):

    # Reference warp selection
    refWarpVisit = Field[int](
        doc="Visit ID of the reference warp. If None, the best warp is chosen from the list of warps.",
        optional=True,
    )
    bestRefWeightCoverage = RangeField(
        dtype=float,
        doc="Coverage weight (Number of pixels overlapping the patch) when calculating the best reference "
        "exposure. Higher weights prefer exposures with high coverage. Ignored when a ref visit supplied.",
        default=0.4,
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
    bestRefWeightLevel = RangeField(
        dtype=float,
        doc="Mean background level weight when calculating the best reference exposure. "
        "Higher weights prefer exposures with low mean background levels. Ignored when ref visit supplied.",
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
        default="MEAN",
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
        default=2,
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
    minimizing a cost function that penalizes high variance, high background
    level, and low coverage.
    The cost function is a weighted sum of these three metrics.
    The weights are set by the config parameters:
    - ``bestRefWeightCoverage``
    - ``bestRefWeightVariance``
    - ``bestRefWeightLevel``


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
        self.statsCtrl = StatisticsControl()
        # TODO: Check that setting the mask planes here work - these planes
        # can vary from exposure to exposure, I think?
        self.statsCtrl.setAndMask(Mask.getPlaneBitMask(self.config.badMaskPlanes))
        self.statsCtrl.setNanSafe(True)

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
        refWarp = self._defineWarps(warps=warps, refWarpVisit=self.config.refWarpVisit)
        breakpoint()

        # Images must be scaled to a common ZP
        # Converting everything to nJy to accomplish this
        refExposure = refMatchedWarp.get()
        refMI = self._fluxScale(refExposure)  # Also modifies refExposure

        # TODO: figure out what this was creating and why
        # debugIdKeyList = tuple(set(expKeyList) - set(['tract', 'patch']))

        self.log.info("Matching %d Exposures", numExp)

        # Creating a null BackgroundList object by fitting a blank image
        statsFlag = stringToStatisticsProperty(self.config.gridStatistic)
        self.statsCtrl.setNumSigmaClip(self.config.numSigmaClip)
        self.statsCtrl.setNumIter(self.config.numIter)

        # TODO: refactor below to construct blank bg model
        im = refExposure.getMaskedImage()
        blankIm = im.Factory(im, True)  # Don't do this
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

        backgroundInfoList = []
        for ind, exp in enumerate(warps):
            if ind in refIndSet:
                backgroundInfoStruct = BackgroundList(
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
            else:
                # TODO: simplify this maybe, using only visit IDs?
                self.log.info("Matching background of %s to %s", exp.dataId, refMatchedWarp.dataId)
                toMatchExposure = exp.get()
                try:
                    # Seems to require exposure, not masked exposure.
                    toMatchMI = self._fluxScale(toMatchExposure)

                    # store a string specifying the visit to label debug plot
                    # self.debugDataIdString = ''.join([str(toMatchRef.dataId[vk]) for vk in debugIdKeyList])

                    backgroundInfoStruct = self.matchBackgrounds(
                        refExposure=refExposure,
                        sciExposure=toMatchExposure,
                    )
                    backgroundInfoStruct.isReference = False
                except Exception as e:
                    self.log.warning("Failed to fit background %s: %s", exp.dataId, e)
                    backgroundInfoStruct = BackgroundList(
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

            backgroundInfoList.append(backgroundInfoStruct)

        return Struct(backgroundInfoList=backgroundInfoList)

    @timeMethod
    def _defineWarps(self, warps, refWarpVisit=None):
        """Define the reference warp and list of comparison warps.

        If no reference warp ID is supplied, this method calculates an
        appropriate reference exposure from the supplied list of warps by
        minimizing a cost function that penalizes high variance, high
        background level, and low coverage.

        To find a reference warp, the following config parameters are used:
        - ``bestRefWeightCoverage``
        - ``bestRefWeightVariance``
        - ``bestRefWeightLevel``

        Parameters
        ----------
        warps : `list`[`~lsst.afw.image.Exposure`]
            List of warped science exposures.

        Returns
        -------
        refWarp : `~lsst.afw.image.Exposure`
            Reference warped exposure.
        compWarps : `list`[`~lsst.afw.image.Exposure`]
            List of warped science exposures to compare to the reference.
        """
        # User a reference visit, if one has been supplied
        if refWarpVisit:
            warpVisits = [warp.dataId["visit"] for warp in warps]
            try:
                refWarp = warps.pop(warpVisits.index(refWarpVisit))
                self.log.info("Using user-supplied reference visit %d", refWarpVisit)
                return refWarp
            except ValueError:
                raise TaskError(f"Reference visit {refWarpVisit} is not found in the list of warps.")

        # Extract mean/var/npoints for each warp
        warpMeans = []
        warpVars = []
        warpNPoints = []
        for warpDDH in warps:
            warp = warpDDH.get()
            warp.image.array *= warp.getPhotoCalib().instFluxToNanojansky(1)  # Convert image plane to nJy
            warpStats = makeStatistics(warp.image, warp.mask, MEAN | VARIANCE | NPOINT, self.statsCtrl)
            warpMean, _ = warpStats.getResult(MEAN)
            warpVar, _ = warpStats.getResult(VARIANCE)
            warpNPoint, _ = warpStats.getResult(NPOINT)
            warpMeans.append(warpMean)
            warpVars.append(warpVar)
            warpNPoints.append(int(warpNPoint))

        # Normalize mean/var/npoints to range from  0 to 1
        warpMeansFrac = np.array(warpMeans) / np.nanmax(warpMeans)
        warpVarsFrac = np.array(warpVars) / np.nanmax(warpVars)
        warpNPointsFrac = np.nanmin(warpNPoints) / np.array(warpNPoints)

        # Calculate cost function values
        costFunctionVals = self.config.bestRefWeightLevel * warpMeansFrac
        costFunctionVals += self.config.bestRefWeightVariance * warpVarsFrac
        costFunctionVals += self.config.bestRefWeightCoverage * warpNPointsFrac

        refWarp = warps.pop(np.nanargmin(costFunctionVals))
        self.log.info("Using best reference visit %d", refWarp.dataId["visit"])
        return refWarp

    def _fluxScale(self, exposure):
        """Scales image to nJy flux using photometric calibration.

        Parameters
        ----------
        exposure: ``
            Exposure to scale.

        Returns
        -------
        maskedImage: ``
            Flux-scaled masked exposure.
        """
        maskedImage = exposure.getMaskedImage()
        fluxZp = exposure.getPhotoCalib().instFluxToNanojansky(1)
        exposure.image.array *= fluxZp

        return maskedImage

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
            Science exposure; modified by changing the background level
            to match that of the reference exposure.

        Returns
        -------
        model : `lsst.pipe.base.Struct`
            Background model as a struct with attributes:

            ``backgroundModel``
                An afw.math.Approximate or an afw.math.Background.
            ``fitRMS``
                RMS of the fit = sqrt(mean(residuals**2)), (`float`).
            ``matchedMSE``
                The MSE of the reference and matched images:
                mean((refImage - matchedSciImage)**2);
                Should be comparable to difference image's mean variance
                (`float`).
            ``diffImVar``
                The mean variance of the difference image (`float`).
        """
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

            if self.config.order > npoints - 1:
                raise ValueError("%d = config.order > npoints - 1 = %d" % (self.config.order, npoints - 1))

        # Check that exposures are same shape
        if sciExposure.getDimensions() != refExposure.getDimensions():
            wSci, hSci = sciExposure.getDimensions()
            wRef, hRef = refExposure.getDimensions()
            raise RuntimeError(
                "Exposures are different dimensions. sci:(%i, %i) vs. ref:(%i, %i)" % (wSci, hSci, wRef, hRef)
            )

        statsFlag = stringToStatisticsProperty(self.config.gridStatistic)
        self.statsCtrl.setNumSigmaClip(self.config.numSigmaClip)
        self.statsCtrl.setNumIter(self.config.numIter)

        im = refExposure.getMaskedImage()
        diffMI = im.Factory(im, True)
        diffMI -= sciExposure.getMaskedImage()

        width = diffMI.getWidth()
        height = diffMI.getHeight()
        nx = width // self.config.binSize
        if width % self.config.binSize != 0:
            nx += 1
        ny = height // self.config.binSize
        if height % self.config.binSize != 0:
            ny += 1

        bctrl = BackgroundControl(nx, ny, self.statsCtrl, statsFlag)
        bctrl.setUndersampleStyle(self.config.undersampleStyle)

        bkgd = makeBackground(diffMI, bctrl)

        # Some config and input checks if config.usePolynomial:
        # 1) Check that order/bin size make sense:
        # 2) Change binsize or order if underconstrained.
        if self.config.usePolynomial:
            order = self.config.order
            bgX, bgY, bgZ, bgdZ = self._gridImage(diffMI, self.config.binSize, statsFlag)
            minNumberGridPoints = min(len(set(bgX)), len(set(bgY)))
            if len(bgZ) == 0:
                raise ValueError("No overlap with reference. Nothing to match")
            elif minNumberGridPoints <= self.config.order:
                # must either lower order or raise number of bins or throw exception
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

            # If there is no variance in any image pixels, do not weight bins by inverse variance
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
                "Background/Approximation failed to interp image %s: %s" % (self.debugDataIdString, e)
            )

        sciMI = sciExposure.getMaskedImage()
        sciMI += bkgdImage
        del sciMI

        # Need RMS from fit: 2895 will replace this:
        rms = 0.0
        bgX, bgY, bgZ, bgdZ = self._gridImage(diffMI, self.config.binSize, statsFlag)
        x0, y0 = diffMI.getXY0()
        modelValueArr = np.empty(len(bgZ))
        for i in range(len(bgX)):
            modelValueArr[i] = bkgdImage[int(bgX[i] - x0), int(bgY[i] - y0), LOCAL]
        resids = bgZ - modelValueArr
        rms = np.sqrt(np.mean(resids[~np.isnan(resids)] ** 2))

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
        """Generate a plot showing the background fit and residuals.

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


class DataRefMatcher:
    """Match data references for a specified dataset type.

    Note that this is not exact, but should suffice for this task
    until there is better support for this kind of thing in the butler.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler to search for maches in.
    datasetType : `str`
        Dataset type to match.
    """

    def __init__(self, butler, datasetType):
        self._datasetType = datasetType  # for diagnostics
        self._keyNames = butler.getKeys(datasetType)

    def _makeKey(self, ref):
        """Return a tuple of values for the specified keyNames.

        Parameters
        ----------
        ref : `Unknown`
            Data reference.

        Raises
        ------
        KeyError
            Raised if ref.dataId is missing a key in keyNames.
        """
        return tuple(ref.dataId[key] for key in self._keyNames)

    def isMatch(self, ref0, ref1):
        """Return True if ref0 == ref1.

        Parameters
        ----------
        ref0 : `Unknown`
            Data for ref 0.
        ref1 : `Unknown`
            Data for ref 1.

        Raises
        ------
        KeyError
            Raised if either ID is missing a key in keyNames.
        """
        return self._makeKey(ref0) == self._makeKey(ref1)

    def matchList(self, ref0, refList):
        """Return a list of indices of matches.

        Parameters
        ----------
        ref0 : `Unknown`
            Data for ref 0.
        `refList` : `list`

        Returns
        -------
        matches : `tuple`
            Tuple of indices of matches.

        Raises
        ------
        KeyError
            Raised if any ID is missing a key in keyNames.
        """
        key0 = self._makeKey(ref0)
        return tuple(ind for ind, ref in enumerate(refList) if self._makeKey(ref) == key0)
