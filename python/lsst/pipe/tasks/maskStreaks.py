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

__all__ = ["MaskStreaksConfig", "MaskStreaksTask", "setDetectionMask"]

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.kht
from lsst.utils.timer import timeMethod

import numpy as np
import scipy
import textwrap
import copy
from skimage.feature import canny
from sklearn.cluster import KMeans
import warnings
from dataclasses import dataclass


def setDetectionMask(maskedImage, forceSlowBin=False, binning=None, detectedPlane="DETECTED",
                     badMaskPlanes=("NO_DATA", "INTRP", "BAD", "SAT", "EDGE"), detectionThreshold=5):
    """Make detection mask and set the mask plane.

    Creat a binary image from a masked image by setting all data with signal-to-
    noise below some threshold to zero, and all data above the threshold to one.
    If the binning parameter has been set, this procedure will be preceded by a
    weighted binning of the data in order to smooth the result, after which the
    result is scaled back to the original dimensions. Set the detection mask
    plane with this binary image.

    Parameters
    ----------
    maskedImage : `lsst.afw.image.maskedImage`
        Image to be (optionally) binned and converted.
    forceSlowBin : `bool`, optional
        Force usage of slower binning method to check that the two methods
        give the same result.
    binning : `int`, optional
        Number of pixels by which to bin image.
    detectedPlane : `str`, optional
        Name of mask with pixels that were detected above threshold in image.
    badMaskPlanes : `set`, optional
        Names of masks with pixels that are rejected.
    detectionThreshold : `float`, optional
        Boundary in signal-to-noise between non-detections and detections for
        making a binary image from the original input image.
    """
    data = maskedImage.image.array
    weights = 1 / maskedImage.variance.array
    mask = maskedImage.getMask()

    detectionMask = ((mask.array & mask.getPlaneBitMask(detectedPlane)))
    badPixelMask = mask.getPlaneBitMask(badMaskPlanes)
    badMask = (mask.array & badPixelMask) > 0
    fitMask = detectionMask.astype(bool) & ~badMask

    fitData = np.copy(data)
    fitData[~fitMask] = 0
    fitWeights = np.copy(weights)
    fitWeights[~fitMask] = 0

    if binning:
        # Do weighted binning:
        ymax, xmax = fitData.shape
        if (ymax % binning == 0) and (xmax % binning == 0) and (not forceSlowBin):
            # Faster binning method
            binNumeratorReshape = (fitData * fitWeights).reshape(ymax // binning, binning,
                                                                 xmax // binning, binning)
            binDenominatorReshape = fitWeights.reshape(binNumeratorReshape.shape)
            binnedNumerator = binNumeratorReshape.sum(axis=3).sum(axis=1)
            binnedDenominator = binDenominatorReshape.sum(axis=3).sum(axis=1)
        else:
            # Slower binning method when (image shape mod binsize) != 0
            warnings.warn('Using slow binning method--consider choosing a binsize that evenly divides '
                          f'into the image size, so that {ymax} mod binning == 0 '
                          f'and {xmax} mod binning == 0')
            xarray = np.arange(xmax)
            yarray = np.arange(ymax)
            xmesh, ymesh = np.meshgrid(xarray, yarray)
            xbins = np.arange(0, xmax + binning, binning)
            ybins = np.arange(0, ymax + binning, binning)
            numerator = fitWeights * fitData
            binnedNumerator, *_ = scipy.stats.binned_statistic_2d(ymesh.ravel(), xmesh.ravel(),
                                                                  numerator.ravel(), statistic='sum',
                                                                  bins=(ybins, xbins))
            binnedDenominator, *_ = scipy.stats.binned_statistic_2d(ymesh.ravel(), xmesh.ravel(),
                                                                    fitWeights.ravel(), statistic='sum',
                                                                    bins=(ybins, xbins))
        binnedData = np.zeros(binnedNumerator.shape)
        ind = binnedDenominator != 0
        np.divide(binnedNumerator, binnedDenominator, out=binnedData, where=ind)
        binnedWeight = binnedDenominator
        binMask = (binnedData * binnedWeight**0.5) > detectionThreshold
        tmpOutputMask = binMask.repeat(binning, axis=0)[:ymax]
        outputMask = tmpOutputMask.repeat(binning, axis=1)[:, :xmax]
    else:
        outputMask = (fitData * fitWeights**0.5) > detectionThreshold

    # Clear existing Detected Plane:
    maskedImage.mask.array &= ~maskedImage.mask.getPlaneBitMask(detectedPlane)

    # Set Detected Plane with the binary detection mask:
    maskedImage.mask.array[outputMask] |= maskedImage.mask.getPlaneBitMask(detectedPlane)


@dataclass
class Line:
    """A simple data class to describe a line profile. The parameter `rho`
    describes the distance from the center of the image, `theta` describes
    the angle, and `sigma` describes the width of the line.
    """

    rho: float
    theta: float
    sigma: float = 0


class LineCollection:
    """Collection of `Line` objects.

    Parameters
    ----------
    rhos : `np.ndarray`
        Array of `Line` rho parameters.
    thetas : `np.ndarray`
        Array  of `Line` theta parameters.
    sigmas : `np.ndarray`, optional
        Array of `Line` sigma parameters.
    """

    def __init__(self, rhos, thetas, sigmas=None):
        if sigmas is None:
            sigmas = np.zeros(len(rhos))

        self._lines = [Line(rho, theta, sigma) for (rho, theta, sigma) in
                       zip(rhos, thetas, sigmas)]

    def __len__(self):
        return len(self._lines)

    def __getitem__(self, index):
        return self._lines[index]

    def __iter__(self):
        return iter(self._lines)

    def __repr__(self):
        joinedString = ", ".join(str(line) for line in self._lines)
        return textwrap.shorten(joinedString, width=160, placeholder="...")

    @property
    def rhos(self):
        return np.array([line.rho for line in self._lines])

    @property
    def thetas(self):
        return np.array([line.theta for line in self._lines])

    def append(self, newLine):
        """Add line to current collection of lines.

        Parameters
        ----------
        newLine : `Line`
            `Line` to add to current collection of lines
        """
        self._lines.append(copy.copy(newLine))


class LineProfile:
    """Construct and/or fit a model for a linear streak.

    This assumes a simple model for a streak, in which the streak
    follows a straight line in pixels space, with a Moffat-shaped profile. The
    model is fit to data using a Newton-Raphson style minimization algorithm.
    The initial guess for the line parameters is assumed to be fairly accurate,
    so only a narrow band of pixels around the initial line estimate is used in
    fitting the model, which provides a significant speed-up over using all the
    data. The class can also be used just to construct a model for the data with
    a line following the given coordinates.

    Parameters
    ----------
    data : `np.ndarray`
        2d array of data.
    weights : `np.ndarray`
        2d array of weights.
    line : `Line`, optional
        Guess for position of line. Data far from line guess is masked out.
        Defaults to None, in which case only data with `weights` = 0 is masked
        out.
    """

    def __init__(self, data, weights, line=None):
        self.data = data
        self.weights = weights
        self._ymax, self._xmax = data.shape
        self._dtype = data.dtype
        xrange = np.arange(self._xmax) - self._xmax / 2.
        yrange = np.arange(self._ymax) - self._ymax / 2.
        self._rhoMax = ((0.5 * self._ymax)**2 + (0.5 * self._xmax)**2)**0.5
        self._xmesh, self._ymesh = np.meshgrid(xrange, yrange)
        self.mask = (weights != 0)

        self._initLine = line
        self.setLineMask(line)

    def setLineMask(self, line):
        """Set mask around the image region near the line.

        Parameters
        ----------
        line : `Line`
            Parameters of line in the image.
        """
        if line:
            # Only fit pixels within 5 sigma of the estimated line
            radtheta = np.deg2rad(line.theta)
            distance = (np.cos(radtheta) * self._xmesh + np.sin(radtheta) * self._ymesh - line.rho)
            m = (abs(distance) < 5 * line.sigma)
            self.lineMask = self.mask & m
        else:
            self.lineMask = np.copy(self.mask)

        self.lineMaskSize = self.lineMask.sum()
        self._maskData = self.data[self.lineMask]
        self._maskWeights = self.weights[self.lineMask]
        self._mxmesh = self._xmesh[self.lineMask]
        self._mymesh = self._ymesh[self.lineMask]

    def _makeMaskedProfile(self, line, fitFlux=True):
        """Construct the line model in the masked region and calculate its
        derivatives.

        Parameters
        ----------
        line : `Line`
            Parameters of line profile for which to make profile in the masked
            region.
        fitFlux : `bool`
            Fit the amplitude of the line profile to the data.

        Returns
        -------
        model : `np.ndarray`
            Model in the masked region.
        dModel : `np.ndarray`
            Derivative of the model in the masked region.
        """
        invSigma = line.sigma**-1
        # Calculate distance between pixels and line
        radtheta = np.deg2rad(line.theta)
        costheta = np.cos(radtheta)
        sintheta = np.sin(radtheta)
        distance = (costheta * self._mxmesh + sintheta * self._mymesh - line.rho)
        distanceSquared = distance**2

        # Calculate partial derivatives of distance
        drad = np.pi / 180
        dDistanceSqdRho = 2 * distance * (-np.ones_like(self._mxmesh))
        dDistanceSqdTheta = (2 * distance * (-sintheta * self._mxmesh + costheta * self._mymesh) * drad)

        # Use pixel-line distances to make Moffat profile
        profile = (1 + distanceSquared * invSigma**2)**-2.5
        dProfile = -2.5 * (1 + distanceSquared * invSigma**2)**-3.5

        if fitFlux:
            # Calculate line flux from profile and data
            flux = ((self._maskWeights * self._maskData * profile).sum()
                    / (self._maskWeights * profile**2).sum())
        else:
            # Approximately normalize the line
            flux = invSigma**-1
        if np.isnan(flux):
            flux = 0

        model = flux * profile

        # Calculate model derivatives
        fluxdProfile = flux * dProfile
        fluxdProfileInvSigma = fluxdProfile * invSigma**2
        dModeldRho = fluxdProfileInvSigma * dDistanceSqdRho
        dModeldTheta = fluxdProfileInvSigma * dDistanceSqdTheta
        dModeldInvSigma = fluxdProfile * distanceSquared * 2 * invSigma

        dModel = np.array([dModeldRho, dModeldTheta, dModeldInvSigma])
        return model, dModel

    def makeProfile(self, line, fitFlux=True):
        """Construct the line profile model.

        Parameters
        ----------
        line : `Line`
            Parameters of the line profile to model.
        fitFlux : `bool`, optional
            Fit the amplitude of the line profile to the data.

        Returns
        -------
        finalModel : `np.ndarray`
            Model for line profile.
        """
        model, _ = self._makeMaskedProfile(line, fitFlux=fitFlux)
        finalModel = np.zeros((self._ymax, self._xmax), dtype=self._dtype)
        finalModel[self.lineMask] = model
        return finalModel

    def _lineChi2(self, line, grad=True):
        """Construct the chi2 between the data and the model.

        Parameters
        ----------
        line : `Line`
            `Line` parameters for which to build model and calculate chi2.
        grad : `bool`, optional
            Whether or not to return the gradient and hessian.

        Returns
        -------
        reducedChi : `float`
            Reduced chi2 of the model.
        reducedDChi : `np.ndarray`
            Derivative of the chi2 with respect to rho, theta, invSigma.
        reducedHessianChi : `np.ndarray`
            Hessian of the chi2 with respect to rho, theta, invSigma.
        """
        # Calculate chi2
        model, dModel = self._makeMaskedProfile(line)
        chi2 = (self._maskWeights * (self._maskData - model)**2).sum()
        if not grad:
            return chi2.sum() / self.lineMaskSize

        # Calculate derivative and Hessian of chi2
        derivChi2 = ((-2 * self._maskWeights * (self._maskData - model))[None, :] * dModel).sum(axis=1)
        hessianChi2 = (2 * self._maskWeights * dModel[:, None, :] * dModel[None, :, :]).sum(axis=2)

        reducedChi = chi2 / self.lineMaskSize
        reducedDChi = derivChi2 / self.lineMaskSize
        reducedHessianChi = hessianChi2 / self.lineMaskSize
        return reducedChi, reducedDChi, reducedHessianChi

    def fit(self, dChi2Tol=0.1, maxIter=100, log=None):
        """Perform Newton-Raphson minimization to find line parameters.

        This method takes advantage of having known derivative and Hessian of
        the multivariate function to quickly and efficiently find the minimum.
        This is more efficient than the scipy implementation of the Newton-
        Raphson method, which doesn't take advantage of the Hessian matrix. The
        method here also performs a line search in the direction of the steepest
        derivative at each iteration, which reduces the number of iterations
        needed.

        Parameters
        ----------
        dChi2Tol : `float`, optional
            Change in Chi2 tolerated for fit convergence.
        maxIter : `int`, optional
            Maximum number of fit iterations allowed. The fit should converge in
            ~10 iterations, depending on the value of dChi2Tol, but this
            maximum provides a backup.
        log : `lsst.utils.logging.LsstLogAdapter`, optional
            Logger to use for reporting more details for fitting failures.

        Returns
        -------
        outline : `np.ndarray`
            Coordinates and inverse width of fit line.
        chi2 : `float`
            Reduced Chi2 of model fit to data.
        fitFailure : `bool`
            Boolean where `False` corresponds to a successful  fit.
        """
        # Do minimization on inverse of sigma to simplify derivatives:
        x = np.array([self._initLine.rho, self._initLine.theta, self._initLine.sigma**-1])

        dChi2 = 1
        iter = 0
        oldChi2 = 0
        fitFailure = False

        def line_search(c, dx):
            testx = x - c * dx
            testLine = Line(testx[0], testx[1], testx[2]**-1)
            return self._lineChi2(testLine, grad=False)

        while abs(dChi2) > dChi2Tol:
            line = Line(x[0], x[1], x[2]**-1)
            chi2, b, A = self._lineChi2(line)
            if chi2 == 0:
                break
            if not np.isfinite(A).all():
                fitFailure = True
                if log is not None:
                    log.warning("Hessian matrix has non-finite elements.")
                break
            dChi2 = oldChi2 - chi2
            try:
                cholesky = scipy.linalg.cho_factor(A)
            except np.linalg.LinAlgError:
                fitFailure = True
                if log is not None:
                    log.warning("Hessian matrix is not invertible.")
                break
            dx = scipy.linalg.cho_solve(cholesky, b)

            factor, fmin, _, _ = scipy.optimize.brent(line_search, args=(dx,), full_output=True, tol=0.05)
            x -= factor * dx
            if (abs(x[0]) > 1.5 * self._rhoMax) or (iter > maxIter):
                fitFailure = True
                break
            oldChi2 = chi2
            iter += 1

        outline = Line(x[0], x[1], abs(x[2])**-1)

        return outline, chi2, fitFailure


class MaskStreaksConfig(pexConfig.Config):
    """Configuration parameters for `MaskStreaksTask`.
    """

    minimumKernelHeight = pexConfig.Field(
        doc="Minimum height of the streak-finding kernel relative to the tallest kernel",
        dtype=float,
        default=0.0,
    )
    absMinimumKernelHeight = pexConfig.Field(
        doc="Minimum absolute height of the streak-finding kernel",
        dtype=float,
        default=5,
    )
    clusterMinimumSize = pexConfig.Field(
        doc="Minimum size in pixels of detected clusters",
        dtype=int,
        default=50,
    )
    clusterMinimumDeviation = pexConfig.Field(
        doc="Allowed deviation (in pixels) from a straight line for a detected "
            "line",
        dtype=int,
        default=2,
    )
    delta = pexConfig.Field(
        doc="Stepsize in angle-radius parameter space",
        dtype=float,
        default=0.2,
    )
    nSigma = pexConfig.Field(
        doc="Number of sigmas from center of kernel to include in voting "
            "procedure",
        dtype=float,
        default=2,
    )
    rhoBinSize = pexConfig.Field(
        doc="Binsize in pixels for position parameter rho when finding "
            "clusters of detected lines",
        dtype=float,
        default=30,
    )
    thetaBinSize = pexConfig.Field(
        doc="Binsize in degrees for angle parameter theta when finding "
            "clusters of detected lines",
        dtype=float,
        default=2,
    )
    invSigma = pexConfig.Field(
        doc="Inverse of the Moffat sigma parameter (in units of pixels)"
            "describing the profile of the streak",
        dtype=float,
        default=10.**-1,
    )
    footprintThreshold = pexConfig.Field(
        doc="Threshold at which to determine edge of line, in units of "
            "nanoJanskys",
        dtype=float,
        default=0.01
    )
    dChi2Tolerance = pexConfig.Field(
        doc="Absolute difference in Chi2 between iterations of line profile"
            "fitting that is acceptable for convergence",
        dtype=float,
        default=0.1
    )
    detectedMaskPlane = pexConfig.Field(
        doc="Name of mask with pixels above detection threshold, used for first"
            "estimate of streak locations",
        dtype=str,
        default="DETECTED"
    )
    streaksMaskPlane = pexConfig.Field(
        doc="Name of mask plane holding detected streaks",
        dtype=str,
        default="STREAK"
    )


class MaskStreaksTask(pipeBase.Task):
    """Find streaks or other straight lines in image data.

    Nearby objects passing through the field of view of the telescope leave a
    bright trail in images. This class uses the Kernel Hough Transform (KHT)
    (Fernandes and Oliveira, 2007), implemented in `lsst.houghtransform`. The
    procedure works by taking a binary image, either provided as put or produced
    from the input data image, using a Canny filter to make an image of the
    edges in the original image, then running the KHT on the edge image. The KHT
    identifies clusters of non-zero points, breaks those clusters of points into
    straight lines, keeps clusters with a size greater than the user-set
    threshold, then performs a voting procedure to find the best-fit coordinates
    of any straight lines. Given the results of the KHT algorithm, clusters of
    lines are identified and grouped (generally these correspond to the two
    edges of a strea) and a profile is fit to the streak in the original
    (non-binary) image.
    """

    ConfigClass = MaskStreaksConfig
    _DefaultName = "maskStreaks"

    @timeMethod
    def find(self, maskedImage):
        """Find streaks in a masked image.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.maskedImage`
            The image in which to search for streaks.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``originalLines``
                Lines identified by kernel hough transform.
            ``lineClusters``
                Lines grouped into clusters in rho-theta space.
            ``lines``
                Final result for lines after line-profile fit.
            ``mask``
                2-d boolean mask where detected lines are True.
        """
        mask = maskedImage.getMask()
        detectionMask = (mask.array & mask.getPlaneBitMask(self.config.detectedMaskPlane))

        self.edges = self._cannyFilter(detectionMask)
        self.lines = self._runKHT(self.edges)

        if len(self.lines) == 0:
            lineMask = np.zeros(detectionMask.shape, dtype=bool)
            fitLines = LineCollection([], [])
            clusters = LineCollection([], [])
        else:
            clusters = self._findClusters(self.lines)
            fitLines, lineMask = self._fitProfile(clusters, maskedImage)

        # The output mask is the intersection of the fit streaks and the image detections
        outputMask = lineMask & detectionMask.astype(bool)

        return pipeBase.Struct(
            lines=fitLines,
            lineClusters=clusters,
            originalLines=self.lines,
            mask=outputMask,
        )

    @timeMethod
    def run(self, maskedImage):
        """Find and mask streaks in a masked image.

        Finds streaks in the image and modifies maskedImage in place by adding a
        mask plane with any identified streaks.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.maskedImage`
            The image in which to search for streaks. The mask detection plane
            corresponding to `config.detectedMaskPlane` must be set with the
            detected pixels.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``originalLines``
                Lines identified by kernel hough transform.
            ``lineClusters``
                Lines grouped into clusters in rho-theta space.
            ``lines``
                Final result for lines after line-profile fit.
        """
        streaks = self.find(maskedImage)

        maskedImage.mask.addMaskPlane(self.config.streaksMaskPlane)
        maskedImage.mask.array[streaks.mask] |= maskedImage.mask.getPlaneBitMask(self.config.streaksMaskPlane)

        return pipeBase.Struct(
            lines=streaks.lines,
            lineClusters=streaks.lineClusters,
            originalLines=streaks.originalLines,
        )

    def _cannyFilter(self, image):
        """Apply a canny filter to the data in order to detect edges.

        Parameters
        ----------
        image : `np.ndarray`
            2-d image data on which to run filter.

        Returns
        -------
        cannyData : `np.ndarray`
            2-d image of edges found in input image.
        """
        filterData = image.astype(int)
        return canny(filterData, low_threshold=0, high_threshold=1, sigma=0.1)

    def _runKHT(self, image):
        """Run Kernel Hough Transform on image.

        Parameters
        ----------
        image : `np.ndarray`
            2-d image data on which to detect lines.

        Returns
        -------
        result : `LineCollection`
            Collection of detected lines, with their detected rho and theta
            coordinates.
        """
        lines = lsst.kht.find_lines(image, self.config.clusterMinimumSize,
                                    self.config.clusterMinimumDeviation, self.config.delta,
                                    self.config.minimumKernelHeight, self.config.nSigma,
                                    self.config.absMinimumKernelHeight)
        self.log.info("The Kernel Hough Transform detected %s line(s)", len(lines))

        return LineCollection(lines.rho, lines.theta)

    def _findClusters(self, lines):
        """Group lines that are close in parameter space and likely describe
        the same streak.

        Parameters
        ----------
        lines : `LineCollection`
            Collection of lines to group into clusters.

        Returns
        -------
        result : `LineCollection`
            Average `Line` for each cluster of `Line`s in the input
            `LineCollection`.
        """
        # Scale variables by threshold bin-size variable so that rho and theta
        # are on the same scale. Since the clustering algorithm below stops when
        # the standard deviation <= 1, after rescaling each cluster will have a
        # standard deviation at or below the bin-size.
        x = lines.rhos / self.config.rhoBinSize
        y = lines.thetas / self.config.thetaBinSize
        X = np.array([x, y]).T
        nClusters = 1

        # Put line parameters in clusters by starting with all in one, then
        # subdividing until the parameters of each cluster have std dev=1.
        # If nClusters == len(lines), each line will have its own 'cluster', so
        # the standard deviations of each cluster must be zero and the loop
        # is guaranteed to stop.
        while True:
            kmeans = KMeans(n_clusters=nClusters).fit(X)
            clusterStandardDeviations = np.zeros((nClusters, 2))
            for c in range(nClusters):
                inCluster = X[kmeans.labels_ == c]
                clusterStandardDeviations[c] = np.std(inCluster, axis=0)
            # Are the rhos and thetas in each cluster all below the threshold?
            if (clusterStandardDeviations <= 1).all():
                break
            nClusters += 1

        # The cluster centers are final line estimates
        finalClusters = kmeans.cluster_centers_.T

        # Rescale variables:
        finalRhos = finalClusters[0] * self.config.rhoBinSize
        finalThetas = finalClusters[1] * self.config.thetaBinSize
        result = LineCollection(finalRhos, finalThetas)
        self.log.info("Lines were grouped into %s potential streak(s)", len(finalRhos))

        return result

    def _fitProfile(self, lines, maskedImage):
        """Fit the profile of the streak.

        Given the initial parameters of detected lines, fit a model for the
        streak to the original (non-binary image). The assumed model is a
        straight line with a Moffat profile.

        Parameters
        ----------
        lines : `LineCollection`
            Collection of guesses for `Line`s detected in the image.
        maskedImage : `lsst.afw.image.maskedImage`
            Original image to be used to fit profile of streak.

        Returns
        -------
        lineFits : `LineCollection`
            Collection of `Line` profiles fit to the data.
        finalMask : `np.ndarray`
            2d mask array with detected streaks=1.
        """
        data = maskedImage.image.array
        weights = maskedImage.variance.array**-1
        # Mask out any pixels with non-finite weights
        weights[~np.isfinite(weights) | ~np.isfinite(data)] = 0

        lineFits = LineCollection([], [])
        finalLineMasks = [np.zeros(data.shape, dtype=bool)]
        nFinalLines = 0
        for line in lines:
            line.sigma = self.config.invSigma**-1
            lineModel = LineProfile(data, weights, line=line)
            # Skip any lines that do not cover any data (sometimes happens because of chip gaps)
            if lineModel.lineMaskSize == 0:
                continue

            fit, chi2, fitFailure = lineModel.fit(dChi2Tol=self.config.dChi2Tolerance, log=self.log)
            if fitFailure:
                self.log.warning("Streak fit failed.")

            # Initial estimate should be quite close: fit is deemed unsuccessful if rho or theta
            # change more than the allowed bin in rho or theta:
            if ((abs(fit.rho - line.rho) > 2 * self.config.rhoBinSize)
                    or (abs(fit.theta - line.theta) > 2 * self.config.thetaBinSize)):
                fitFailure = True
                self.log.warning("Streak fit moved too far from initial estimate. Line will be dropped.")

            if fitFailure:
                continue

            self.log.debug("Best fit streak parameters are rho=%.2f, theta=%.2f, and sigma=%.2f", fit.rho,
                           fit.theta, fit.sigma)

            # Make mask
            lineModel.setLineMask(fit)
            finalModel = lineModel.makeProfile(fit)
            # Take absolute value, as streaks are allowed to be negative
            finalModelMax = abs(finalModel).max()
            finalLineMask = abs(finalModel) > self.config.footprintThreshold
            # Drop this line if the model profile is below the footprint threshold
            if not finalLineMask.any():
                continue
            fit.chi2 = chi2
            fit.finalModelMax = finalModelMax
            lineFits.append(fit)
            finalLineMasks.append(finalLineMask)
            nFinalLines += 1

        finalMask = np.array(finalLineMasks).any(axis=0)
        nMaskedPixels = finalMask.sum()
        percentMasked = (nMaskedPixels / finalMask.size) * 100
        self.log.info("%d streak(s) fit, with %d pixels masked (%0.2f%% of image)", nFinalLines,
                      nMaskedPixels, percentMasked)

        return lineFits, finalMask
