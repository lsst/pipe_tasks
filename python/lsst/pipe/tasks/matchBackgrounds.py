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

__all__ = ["MatchBackgroundsConfig", "MatchBackgroundsTask"]

import numpy as np
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsstDebug
from lsst.utils.timer import timeMethod

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
import lsst.pipe.base.connectionTypes as cT

import pdb


class MatchBackgroundsConnections(PipelineTaskConnections,
                                  dimensions=("tract", "patch", "band", "skymap"),
                                  defaultTemplates={"inputCoaddName": "deep",
                                                    "outputCoaddName": "deep",
                                                    "warpType": "direct",
                                                    "warpTypeSuffix": ""}):

    psfMatchedWarps = pipeBase.connectionTypes.Input(
        doc=("PSF-matched warps to be subtracted from the reference warp"),
        name="{inputCoaddName}Coadd_psfMatchedWarp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit"),
        deferLoad=True,
        multiple=True,
    )
    # The reference exposure needs to be another psfMatchedWarp
    # Let the software choose it automatically for now
    # refMatchedWarp = cT.Input(
    #     doc="Reference exposure",
    #     dimensions=("visit", "detector", "band"),
    #     storageClass="ExposureF",
    #     name="calexp",
    # )
    # This needs to be the models of each differential BG in warped coords
    backgroundInfoList = cT.Output(
        doc="List of differential backgrounds, w/goodness of fit params",
        name="calexpBackground_diff",  # This needs to change
        dimensions=("visit", "detector",),
        storageClass="Background",
        multiple=True,
    )


# class MatchBackgroundsConfig(pexConfig.Config):
class MatchBackgroundsConfig(PipelineTaskConfig,
                             pipelineConnections=MatchBackgroundsConnections):

    usePolynomial = pexConfig.Field(
        dtype=bool,
        doc="Fit background difference with Chebychev polynomial interpolation "
        "(using afw.math.Approximate)? If False, fit with spline interpolation using afw.math.Background",
        default=False
    )
    order = pexConfig.Field(
        dtype=int,
        doc="Order of Chebyshev polynomial background model. Ignored if usePolynomial False",
        default=8
    )
    badMaskPlanes = pexConfig.ListField(
        doc="Names of mask planes to ignore while estimating the background",
        dtype=str, default=["NO_DATA", "DETECTED", "DETECTED_NEGATIVE", "SAT", "BAD", "INTRP", "CR"],
        itemCheck=lambda x: x in afwImage.Mask().getMaskPlaneDict(),
    )
    gridStatistic = pexConfig.ChoiceField(
        dtype=str,
        doc="Type of statistic to estimate pixel value for the grid points",
        default="MEAN",
        allowed={
            "MEAN": "mean",
            "MEDIAN": "median",
            "MEANCLIP": "clipped mean"
        }
    )
    undersampleStyle = pexConfig.ChoiceField(
        doc="Behaviour if there are too few points in grid for requested interpolation style. "
        "Note: INCREASE_NXNYSAMPLE only allowed for usePolynomial=True.",
        dtype=str,
        default="REDUCE_INTERP_ORDER",
        allowed={
            "THROW_EXCEPTION": "throw an exception if there are too few points",
            "REDUCE_INTERP_ORDER": "use an interpolation style with a lower order.",
            "INCREASE_NXNYSAMPLE": "Increase the number of samples used to make the interpolation grid.",
        }
    )
    binSize = pexConfig.Field(
        doc="Bin size for gridding the difference image and fitting a spatial model",
        dtype=int,
        default=256
    )
    interpStyle = pexConfig.ChoiceField(
        dtype=str,
        doc="Algorithm to interpolate the background values; ignored if usePolynomial is True"
        "Maps to an enum; see afw.math.Background",
        default="AKIMA_SPLINE",
        allowed={
            "CONSTANT": "Use a single constant value",
            "LINEAR": "Use linear interpolation",
            "NATURAL_SPLINE": "cubic spline with zero second derivative at endpoints",
            "AKIMA_SPLINE": "higher-level nonlinear spline that is more robust to outliers",
            "NONE": "No background estimation is to be attempted",
        }
    )
    numSigmaClip = pexConfig.Field(
        dtype=int,
        doc="Sigma for outlier rejection; ignored if gridStatistic != 'MEANCLIP'.",
        default=3
    )
    numIter = pexConfig.Field(
        dtype=int,
        doc="Number of iterations of outlier rejection; ignored if gridStatistic != 'MEANCLIP'.",
        default=2
    )
    bestRefWeightCoverage = pexConfig.RangeField(
        dtype=float,
        doc="Weight given to coverage (number of pixels that overlap with patch), "
        "when calculating best reference exposure. Higher weight prefers exposures with high coverage."
        "Ignored when reference visit is supplied",
        default=0.4,
        min=0., max=1.
    )
    bestRefWeightVariance = pexConfig.RangeField(
        dtype=float,
        doc="Weight given to image variance when calculating best reference exposure. "
        "Higher weight prefers exposures with low image variance. Ignored when reference visit is supplied",
        default=0.4,
        min=0., max=1.
    )
    bestRefWeightLevel = pexConfig.RangeField(
        dtype=float,
        doc="Weight given to mean background level when calculating best reference exposure. "
        "Higher weight prefers exposures with low mean background level. "
        "Ignored when reference visit is supplied.",
        default=0.2,
        min=0., max=1.
    )
    approxWeighting = pexConfig.Field(
        dtype=bool,
        doc=("Use inverse-variance weighting when approximating background offset model? "
             "This will fail when the background offset is constant "
             "(this is usually only the case in testing with artificial images)."
             "(usePolynomial=True)"),
        default=True,
    )
    gridStdevEpsilon = pexConfig.RangeField(
        dtype=float,
        doc="Tolerance on almost zero standard deviation in a background-offset grid bin. "
        "If all bins have a standard deviation below this value, the background offset model "
        "is approximated without inverse-variance weighting. (usePolynomial=True)",
        default=1e-8,
        min=0.
    )


class MatchBackgroundsTask(pipeBase.PipelineTask):
    ConfigClass = MatchBackgroundsConfig
    _DefaultName = "matchBackgrounds"

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

        self.sctrl = afwMath.StatisticsControl()
        self.sctrl.setAndMask(afwImage.Mask.getPlaneBitMask(self.config.badMaskPlanes))
        self.sctrl.setNanSafe(True)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        '''
        Boilerplate for now, until bug-testing commences
        '''
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    @timeMethod
    def run(self, psfMatchedWarps):
        """Match the backgrounds of a list of science exposures to a reference science exposure.

        Choose a refMatchedWarp automatically if none supplied.

        Parameters
        ----------
        psfMatchedWarps : `list` of `lsst.afw.image.Exposure`
            List of warped science exposures to be background-matched;
            all exposures must exist.
        refMatchedWarp : `lsst.afw.image.Exposure`, optional
            Reference science exposure.
            If None, then this task selects the best exposures from psfMatchedWarps.
            If not None then must be one of the exposures in psfMatchedWarps.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``backgroundInfoList``
                A `list` of `pipeBase.Struct`, one per exposure in psfMatchedWarps\s,
                each of which contains these fields:
                - ``isReference``: This is the reference exposure (only one
                                   returned Struct will contain True for this
                                   value, unless the ref exposure is listed multiple times).
                - ``backgroundModel``: Differential background model
                                       (afw.Math.Background or afw.Math.Approximate).
                                       Add this to the science exposure to match the reference exposure.
                - ``fitRMS``: The RMS of the fit. This is the sqrt(mean(residuals**2)).
                - ``matchedMSE``: The MSE of the reference and matched images:
                                  mean((refImage - matchedSciImage)**2);
                    should be comparable to difference image's mean variance.
                - ``diffImVar``: The mean variance of the difference image.
                All fields except isReference will be None if isReference True or the fit failed.

        Raises
        ------
        RuntimeError
            Raised if an exposure does not exist on disk.
        """
        numExp = len(psfMatchedWarps)
        if numExp < 1:
            raise pipeBase.TaskError("No exposures to match")

        # The selectRefExposure() method now only requires one input
        # refInd is the index in psfMatchedWarps corresponding to refMatchedWarp
        # refInd = None
        # if refMatchedWarp is None:
        # select the best reference exposure from psfMatchedWarps
        # note: just selecting from the set for testing purposes
        refInd = self.selectRefExposure(
            psfMatchedWarps=psfMatchedWarps,
        )
        refMatchedWarp = psfMatchedWarps[refInd]

        # refIndSet is the index of all exposures in psfMatchedWarps that match the reference.
        # It is used to avoid background-matching an exposure to itself. It is a list
        # because it is possible (though unlikely) that psfMatchedWarps will contain duplicates.
        refId = refMatchedWarp.dataId
        expIds = [exp.dataId for exp in psfMatchedWarps]
        try:
            refIndSet = [expIds.index(i) for i in [refId]]
        except ValueError:
            raise RuntimeError("Internal error: selected reference %s not found in psfMatchedWarps")

        # This is the original check, probably no longer needed.
        # if refInd is not None and refInd not in refIndSet:
        #     raise RuntimeError("Internal error: selected reference %s not found in psfMatchedWarps")

        # Images must be scaled to a common ZP
        # Converting everything to nJy to accomplish this
        refExposure = refMatchedWarp.get()
        refMI = self._fluxScale(refExposure)  # Also modifies refExposure

        # Will determine later what this is supposed to be
        # All unknown quantities being logged currently commented out
        # debugIdKeyList = tuple(set(expKeyList) - set(['tract', 'patch']))

        self.log.info("Matching %d Exposures", numExp)

        backgroundInfoList = []
        for ind, exp in enumerate(psfMatchedWarps):
            if ind in refIndSet:
                backgroundInfoStruct = pipeBase.Struct(
                    isReference=True,
                    backgroundModel=None,
                    fitRMS=0.0,
                    matchedMSE=None,
                    diffImVar=None,
                )
            else:
                self.log.info("Matching background of %s to %s", exp.dataId, refMatchedWarp.dataId)
                toMatchExposure = exp.get()
                try:
                    # Why use exposure and not maskedImage?
                    toMatchMI = self._fluxScale(toMatchExposure)

                    # store a string specifying the visit to label debug plot
                    # self.debugDataIdString = ''.join([str(toMatchRef.dataId[vk]) for vk in debugIdKeyList])

                    backgroundInfoStruct = self.matchBackgrounds(
                        refExposure=refExposure,
                        sciExposure=toMatchExposure,
                    )
                    backgroundInfoStruct.isReference = False
                except Exception as e:
                    # self.log.warning("Failed to fit background %s: %s", toMatchRef.dataId, e)
                    backgroundInfoStruct = pipeBase.Struct(
                        isReference=False,
                        backgroundModel=None,
                        fitRMS=None,
                        matchedMSE=None,
                        diffImVar=None,
                    )

            backgroundInfoList.append(backgroundInfoStruct)

        return pipeBase.Struct(
            backgroundInfoList=backgroundInfoList)

    @timeMethod
    def selectRefExposure(self, psfMatchedWarps):
        """Find best exposure to use as the reference exposure.

        Calculate an appropriate reference exposure by minimizing a cost function that penalizes
        high variance,  high background level, and low coverage. Use the following config parameters:
        - bestRefWeightCoverage
        - bestRefWeightVariance
        - bestRefWeightLevel

        Parameters
        ----------
        psfMatchedWarps : `list` of `lsst.afw.image.Exposure`
            List of science exposures.
            If an exposure is not found, it is skipped with a warning.

        Returns
        -------
        bestIdx : `int`
            Index of best exposure.

        Raises
        ------
        RuntimeError
            Raised if none of the exposures in psfMatchedWarps are found.
        """
        self.log.info("Calculating best reference visit")
        varList = []
        meanBkgdLevelList = []
        coverageList = []

        for exp in psfMatchedWarps:
            exposure = exp.get()
            # Convert images to nJy before doing statistics
            try:
                maskedImage = self._fluxScale(exposure)
            except Exception:
                # need to put a place holder in Arr
                varList.append(np.nan)
                meanBkgdLevelList.append(np.nan)
                coverageList.append(np.nan)
                continue
            statObjIm = afwMath.makeStatistics(maskedImage.getImage(), maskedImage.getMask(),
                                               afwMath.MEAN | afwMath.NPOINT | afwMath.VARIANCE, self.sctrl)
            meanVar, meanVarErr = statObjIm.getResult(afwMath.VARIANCE)
            meanBkgdLevel, meanBkgdLevelErr = statObjIm.getResult(afwMath.MEAN)
            npoints, npointsErr = statObjIm.getResult(afwMath.NPOINT)
            varList.append(meanVar)
            meanBkgdLevelList.append(meanBkgdLevel)
            coverageList.append(npoints)
        if not coverageList:
            raise pipeBase.TaskError(
                "None of the candidates exist; cannot select best reference exposure")

        # Normalize metrics to range from  0 to 1
        varArr = np.array(varList)/np.nanmax(varList)
        meanBkgdLevelArr = np.array(meanBkgdLevelList)/np.nanmax(meanBkgdLevelList)
        coverageArr = np.nanmin(coverageList)/np.array(coverageList)

        costFunctionArr = self.config.bestRefWeightVariance * varArr
        costFunctionArr += self.config.bestRefWeightLevel * meanBkgdLevelArr
        costFunctionArr += self.config.bestRefWeightCoverage * coverageArr
        return np.nanargmin(costFunctionArr)

    @timeMethod
    def matchBackgrounds(self, refExposure, sciExposure):
        """Match science exposure's background level to that of reference exposure.

        Process creates a difference image of the reference exposure minus the science exposure, and then
        generates an afw.math.Background object. It assumes (but does not require/check) that the mask plane
        already has detections set. If detections have not been set/masked, sources will bias the
        background estimation.

        The 'background' of the difference image is smoothed by spline interpolation (by the Background class)
        or by polynomial interpolation by the Approximate class. This model of difference image
        is added to the science exposure in memory.

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
                RMS of the fit. This is the sqrt(mean(residuals**2)), (`float`).
            ``matchedMSE``
                The MSE of the reference and matched images: mean((refImage - matchedSciImage)**2);
                should be comparable to difference image's mean variance (`float`).
            ``diffImVar``
                The mean variance of the difference image (`float`).
        """
        if lsstDebug.Info(__name__).savefits:
            refExposure.writeFits(lsstDebug.Info(__name__).figpath + 'refExposure.fits')
            sciExposure.writeFits(lsstDebug.Info(__name__).figpath + 'sciExposure.fits')

        # Check Configs for polynomials:
        if self.config.usePolynomial:
            x, y = sciExposure.getDimensions()
            shortSideLength = min(x, y)
            if shortSideLength < self.config.binSize:
                raise ValueError("%d = config.binSize > shorter dimension = %d" % (self.config.binSize,
                                                                                   shortSideLength))
            npoints = shortSideLength // self.config.binSize
            if shortSideLength % self.config.binSize != 0:
                npoints += 1

            if self.config.order > npoints - 1:
                raise ValueError("%d = config.order > npoints - 1 = %d" % (self.config.order, npoints - 1))

        # Check that exposures are same shape
        if (sciExposure.getDimensions() != refExposure.getDimensions()):
            wSci, hSci = sciExposure.getDimensions()
            wRef, hRef = refExposure.getDimensions()
            raise RuntimeError(
                "Exposures are different dimensions. sci:(%i, %i) vs. ref:(%i, %i)" %
                (wSci, hSci, wRef, hRef))

        statsFlag = getattr(afwMath, self.config.gridStatistic)
        self.sctrl.setNumSigmaClip(self.config.numSigmaClip)
        self.sctrl.setNumIter(self.config.numIter)

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

        bctrl = afwMath.BackgroundControl(nx, ny, self.sctrl, statsFlag)
        bctrl.setUndersampleStyle(self.config.undersampleStyle)

        bkgd = afwMath.makeBackground(diffMI, bctrl)

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
                    newBinSize = (minNumberGridPoints*self.config.binSize) // (self.config.order + 1)
                    bctrl.setNxSample(newBinSize)
                    bctrl.setNySample(newBinSize)
                    bkgd = afwMath.makeBackground(diffMI, bctrl)  # do over
                    self.log.warning("Decreasing binsize to %d", newBinSize)

            # If there is no variance in any image pixels, do not weight bins by inverse variance
            isUniformImageDiff = not np.any(bgdZ > self.config.gridStdevEpsilon)
            weightByInverseVariance = False if isUniformImageDiff else self.config.approxWeighting

        # Add offset to sciExposure
        try:
            if self.config.usePolynomial:
                actrl = afwMath.ApproximateControl(afwMath.ApproximateControl.CHEBYSHEV,
                                                   order, order, weightByInverseVariance)
                undersampleStyle = getattr(afwMath, self.config.undersampleStyle)
                approx = bkgd.getApproximate(actrl, undersampleStyle)
                bkgdImage = approx.getImage()
            else:
                bkgdImage = bkgd.getImageF(self.config.interpStyle, self.config.undersampleStyle)
        except Exception as e:
            raise RuntimeError("Background/Approximation failed to interp image %s: %s" % (
                self.debugDataIdString, e))

        sciMI = sciExposure.getMaskedImage()
        sciMI += bkgdImage
        del sciMI

        # Need RMS from fit: 2895 will replace this:
        rms = 0.0
        X, Y, Z, dZ = self._gridImage(diffMI, self.config.binSize, statsFlag)
        x0, y0 = diffMI.getXY0()
        modelValueArr = np.empty(len(Z))
        for i in range(len(X)):
            modelValueArr[i] = bkgdImage[int(X[i]-x0), int(Y[i]-y0), afwImage.LOCAL]
        resids = Z - modelValueArr
        rms = np.sqrt(np.mean(resids[~np.isnan(resids)]**2))

        if lsstDebug.Info(__name__).savefits:
            sciExposure.writeFits(lsstDebug.Info(__name__).figpath + 'sciMatchedExposure.fits')

        if lsstDebug.Info(__name__).savefig:
            bbox = geom.Box2D(refExposure.getMaskedImage().getBBox())
            try:
                self._debugPlot(X, Y, Z, dZ, bkgdImage, bbox, modelValueArr, resids)
            except Exception as e:
                self.log.warning('Debug plot not generated: %s', e)

        meanVar = afwMath.makeStatistics(diffMI.getVariance(), diffMI.getMask(),
                                         afwMath.MEANCLIP, self.sctrl).getValue()

        diffIm = diffMI.getImage()
        diffIm -= bkgdImage  # diffMI should now have a mean ~ 0
        del diffIm
        mse = afwMath.makeStatistics(diffMI, afwMath.MEANSQUARE, self.sctrl).getValue()

        outBkgd = approx if self.config.usePolynomial else bkgd

        return pipeBase.Struct(
            backgroundModel=outBkgd,
            fitRMS=rms,
            matchedMSE=mse,
            diffImVar=meanVar)

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
        import matplotlib.pyplot as plt
        import matplotlib.colors
        from mpl_toolkits.axes_grid1 import ImageGrid
        zeroIm = afwImage.MaskedImageF(geom.Box2I(bbox))
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
            grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.1,
                             share_all=True, label_mode="L", cbar_mode="each",
                             cbar_size="7%", cbar_pad="2%", cbar_location="top")
            im = grid[0].imshow(zeroIm.getImage().getArray(),
                                extent=(x0, x0+dx, y0+dy, y0), norm=norm,
                                cmap='Spectral')
            im = grid[0].scatter(X, Y, c=Z, s=15.*meanDz/dZ, edgecolor='none', norm=norm,
                                 marker='o', cmap='Spectral')
            im2 = grid[1].scatter(X, Y, c=resids, edgecolor='none', norm=diffnorm,
                                  marker='s', cmap='seismic')
            grid.cbar_axes[0].colorbar(im)
            grid.cbar_axes[1].colorbar(im2)
            grid[0].axis([x0, x0+dx, y0+dy, y0])
            grid[1].axis([x0, x0+dx, y0+dy, y0])
            grid[0].set_xlabel("model and grid")
            grid[1].set_xlabel("residuals. rms = %0.3f"%(rms))
            if lsstDebug.Info(__name__).savefig:
                fig.savefig(lsstDebug.Info(__name__).figpath + self.debugDataIdString + '.png')
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
                subBBox = geom.Box2I(geom.PointI(int(x0 + xmin), int(y0 + ymin)),
                                     geom.PointI(int(x0 + xmax-1), int(y0 + ymax-1)))
                subIm = afwImage.MaskedImageF(maskedImage, subBBox, afwImage.PARENT, False)
                stats = afwMath.makeStatistics(subIm,
                                               afwMath.MEAN | afwMath.MEANCLIP | afwMath.MEDIAN
                                               | afwMath.NPOINT | afwMath.STDEV,
                                               self.sctrl)
                npoints, _ = stats.getResult(afwMath.NPOINT)
                if npoints >= 2:
                    stdev, _ = stats.getResult(afwMath.STDEV)
                    if stdev < self.config.gridStdevEpsilon:
                        stdev = self.config.gridStdevEpsilon
                    bgX.append(0.5 * (x0 + xmin + x0 + xmax))
                    bgY.append(0.5 * (y0 + ymin + y0 + ymax))
                    bgdZ.append(stdev/np.sqrt(npoints))
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
