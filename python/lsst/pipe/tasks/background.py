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
    "BackgroundConfig",
    "FocalPlaneBackground",
    "FocalPlaneBackgroundConfig",
    "MaskObjectsConfig",
    "MaskObjectsTask",
    "SkyMeasurementConfig",
    "SkyMeasurementTask",
    "SkyStatsConfig",
    "TractBackground",
    "TractBackgroundConfig",
]

import importlib
import itertools
import sys

import lsst.afw.cameraGeom as afwCameraGeom
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.meas.algorithms as measAlg
import numpy
from lsst.pex.config import ChoiceField, Config, ConfigField, ConfigurableField, Field, ListField, RangeField
from lsst.pipe.base import Task
from scipy.ndimage import gaussian_filter


def robustMean(array, rej=3.0):
    """Measure a robust mean of an array

    Parameters
    ----------
    array : `numpy.ndarray`
        Array for which to measure the mean.
    rej : `float`
        k-sigma rejection threshold.

    Returns
    -------
    mean : `array.dtype`
        Robust mean of `array`.
    """
    q1, median, q3 = numpy.percentile(array, [25.0, 50.0, 100.0])
    good = numpy.abs(array - median) < rej * 0.74 * (q3 - q1)
    return array[good].mean()


class BackgroundConfig(Config):
    """Configuration for background measurement"""

    statistic = ChoiceField(
        dtype=str,
        default="MEANCLIP",
        doc="type of statistic to use for grid points",
        allowed={"MEANCLIP": "clipped mean", "MEAN": "unclipped mean", "MEDIAN": "median"},
    )
    xBinSize = RangeField(dtype=int, default=32, min=1, doc="Superpixel size in x")
    yBinSize = RangeField(dtype=int, default=32, min=1, doc="Superpixel size in y")
    algorithm = ChoiceField(
        dtype=str,
        default="NATURAL_SPLINE",
        optional=True,
        doc="How to interpolate the background values. " "This maps to an enum; see afw::math::Background",
        allowed={
            "CONSTANT": "Use a single constant value",
            "LINEAR": "Use linear interpolation",
            "NATURAL_SPLINE": "cubic spline with zero second derivative at endpoints",
            "AKIMA_SPLINE": "higher-level nonlinear spline that is more robust" " to outliers",
            "NONE": "No background estimation is to be attempted",
        },
    )
    mask = ListField(
        dtype=str,
        default=["SAT", "BAD", "EDGE", "DETECTED", "DETECTED_NEGATIVE", "NO_DATA"],
        doc="Names of mask planes to ignore while estimating the background",
    )


class SkyStatsConfig(Config):
    """Parameters controlling the measurement of sky statistics"""

    statistic = ChoiceField(
        dtype=str,
        default="MEANCLIP",
        doc="type of statistic to use for grid points",
        allowed={"MEANCLIP": "clipped mean", "MEAN": "unclipped mean", "MEDIAN": "median"},
    )
    clip = Field(doc="Clipping threshold for background", dtype=float, default=3.0)
    nIter = Field(doc="Clipping iterations for background", dtype=int, default=3)
    mask = ListField(
        doc="Mask planes to reject",
        dtype=str,
        default=["SAT", "DETECTED", "DETECTED_NEGATIVE", "BAD", "NO_DATA"],
    )


class SkyMeasurementConfig(Config):
    """Configuration for SkyMeasurementTask"""

    skyIter = Field(dtype=int, default=3, doc="k-sigma rejection iterations for sky scale")
    skyRej = Field(dtype=float, default=3.0, doc="k-sigma rejection threshold for sky scale")
    background = ConfigField(dtype=BackgroundConfig, doc="Background measurement")
    xNumSamples = Field(dtype=int, default=4, doc="Number of samples in x for scaling sky frame")
    yNumSamples = Field(dtype=int, default=4, doc="Number of samples in y for scaling sky frame")
    stats = ConfigField(dtype=SkyStatsConfig, doc="Measurement of sky statistics in the samples")


class SkyMeasurementTask(Task):
    """Task for creating, persisting and using sky frames

    A sky frame is like a fringe frame (the sum of many exposures of the night sky,
    combined with rejection to remove astrophysical objects) except the structure
    is on larger scales, and hence we bin the images and represent them as a
    background model (a `lsst.afw.math.BackgroundMI`).  The sky frame represents
    the dominant response of the camera to the sky background.
    """

    ConfigClass = SkyMeasurementConfig

    @staticmethod
    def exposureToBackground(bgExp):
        """Convert an exposure to background model

        Calibs need to be persisted as an Exposure, so we need to convert
        the persisted Exposure to a background model.

        Parameters
        ----------
        bgExp : `lsst.afw.image.Exposure`
            Background model in Exposure format.

        Returns
        -------
        bg : `lsst.afw.math.BackgroundList`
            Background model
        """
        header = bgExp.getMetadata()
        xMin = header.getScalar("BOX.MINX")
        yMin = header.getScalar("BOX.MINY")
        xMax = header.getScalar("BOX.MAXX")
        yMax = header.getScalar("BOX.MAXY")
        algorithm = header.getScalar("ALGORITHM")
        bbox = geom.Box2I(geom.Point2I(xMin, yMin), geom.Point2I(xMax, yMax))
        return afwMath.BackgroundList(
            (
                afwMath.BackgroundMI(bbox, bgExp.getMaskedImage()),
                afwMath.stringToInterpStyle(algorithm),
                afwMath.stringToUndersampleStyle("REDUCE_INTERP_ORDER"),
                afwMath.ApproximateControl.UNKNOWN,
                0,
                0,
                False,
            )
        )

    def backgroundToExposure(self, statsImage, bbox):
        """Convert a background model to an exposure

        Calibs need to be persisted as an Exposure, so we need to convert
        the background model to an Exposure.

        Parameters
        ----------
        statsImage : `lsst.afw.image.MaskedImageF`
            Background model's statistics image.
        bbox : `lsst.geom.Box2I`
            Bounding box for image.

        Returns
        -------
        exp : `lsst.afw.image.Exposure`
            Background model in Exposure format.
        """
        exp = afwImage.makeExposure(statsImage)
        header = exp.getMetadata()
        header.set("BOX.MINX", bbox.getMinX())
        header.set("BOX.MINY", bbox.getMinY())
        header.set("BOX.MAXX", bbox.getMaxX())
        header.set("BOX.MAXY", bbox.getMaxY())
        header.set("ALGORITHM", self.config.background.algorithm)
        return exp

    def measureBackground(self, image):
        """Measure a background model for image

        This doesn't use a full-featured background model (e.g., no Chebyshev
        approximation) because we just want the binning behaviour.  This will
        allow us to average the bins later (`averageBackgrounds`).

        The `BackgroundMI` is wrapped in a `BackgroundList` so it can be
        pickled and persisted.

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage`
            Image for which to measure background.

        Returns
        -------
        bgModel : `lsst.afw.math.BackgroundList`
            Background model.
        """
        stats = afwMath.StatisticsControl()
        stats.setAndMask(image.getMask().getPlaneBitMask(self.config.background.mask))
        stats.setNanSafe(True)
        ctrl = afwMath.BackgroundControl(
            self.config.background.algorithm,
            max(int(image.getWidth() / self.config.background.xBinSize + 0.5), 1),
            max(int(image.getHeight() / self.config.background.yBinSize + 0.5), 1),
            "REDUCE_INTERP_ORDER",
            stats,
            self.config.background.statistic,
        )

        bg = afwMath.makeBackground(image, ctrl)

        return afwMath.BackgroundList(
            (
                bg,
                afwMath.stringToInterpStyle(self.config.background.algorithm),
                afwMath.stringToUndersampleStyle("REDUCE_INTERP_ORDER"),
                afwMath.ApproximateControl.UNKNOWN,
                0,
                0,
                False,
            )
        )

    def averageBackgrounds(self, bgList):
        """Average multiple background models

        The input background models should be a `BackgroundList` consisting
        of a single `BackgroundMI`.

        Parameters
        ----------
        bgList : `list` of `lsst.afw.math.BackgroundList`
            Background models to average.

        Returns
        -------
        bgExp : `lsst.afw.image.Exposure`
            Background model in Exposure format.
        """
        assert all(len(bg) == 1 for bg in bgList), "Mixed bgList: %s" % ([len(bg) for bg in bgList],)
        images = [bg[0][0].getStatsImage() for bg in bgList]
        boxes = [bg[0][0].getImageBBox() for bg in bgList]
        assert (
            len(set((box.getMinX(), box.getMinY(), box.getMaxX(), box.getMaxY()) for box in boxes)) == 1
        ), "Bounding boxes not all equal"
        bbox = boxes.pop(0)

        # Ensure bad pixels are masked
        maskVal = afwImage.Mask.getPlaneBitMask("BAD")
        for img in images:
            bad = numpy.isnan(img.getImage().getArray())
            img.getMask().getArray()[bad] = maskVal

        stats = afwMath.StatisticsControl()
        stats.setAndMask(maskVal)
        stats.setNanSafe(True)
        combined = afwMath.statisticsStack(images, afwMath.MEANCLIP, stats)

        # Set bad pixels to the median
        # Specifically NOT going to attempt to interpolate the bad values because we're only working on a
        # single CCD here and can't use the entire field-of-view to do the interpolation (which is what we
        # would need to avoid introducing problems at the edge of CCDs).
        array = combined.getImage().getArray()
        bad = numpy.isnan(array)
        median = numpy.median(array[~bad])
        array[bad] = median

        # Put it into an exposure, which is required for calibs
        return self.backgroundToExposure(combined, bbox)

    def measureScale(self, image, skyBackground):
        """Measure scale of background model in image

        We treat the sky frame much as we would a fringe frame
        (except the length scale of the variations is different):
        we measure samples on the input image and the sky frame,
        which we will use to determine the scaling factor in the
        'solveScales` method.

        Parameters
        ----------
        image : `lsst.afw.image.Exposure` or `lsst.afw.image.MaskedImage`
            Science image for which to measure scale.
        skyBackground : `lsst.afw.math.BackgroundList`
            Sky background model.

        Returns
        -------
        imageSamples : `numpy.ndarray`
            Sample measurements on image.
        skySamples : `numpy.ndarray`
            Sample measurements on sky frame.
        """
        if isinstance(image, afwImage.Exposure):
            image = image.getMaskedImage()
        # Ensure more samples than pixels
        xNumSamples = min(self.config.xNumSamples, image.getWidth())
        yNumSamples = min(self.config.yNumSamples, image.getHeight())
        xLimits = numpy.linspace(0, image.getWidth(), xNumSamples + 1, dtype=int)
        yLimits = numpy.linspace(0, image.getHeight(), yNumSamples + 1, dtype=int)
        sky = skyBackground.getImage()
        maskVal = image.getMask().getPlaneBitMask(self.config.stats.mask)
        ctrl = afwMath.StatisticsControl(self.config.stats.clip, self.config.stats.nIter, maskVal)
        statistic = afwMath.stringToStatisticsProperty(self.config.stats.statistic)
        imageSamples = []
        skySamples = []
        for xIndex, yIndex in itertools.product(range(xNumSamples), range(yNumSamples)):
            # -1 on the stop because Box2I is inclusive of the end point and we don't want to overlap boxes
            xStart, xStop = xLimits[xIndex], xLimits[xIndex + 1] - 1
            yStart, yStop = yLimits[yIndex], yLimits[yIndex + 1] - 1
            box = geom.Box2I(geom.Point2I(xStart, yStart), geom.Point2I(xStop, yStop))
            subImage = image.Factory(image, box)
            subSky = sky.Factory(sky, box)
            imageSamples.append(afwMath.makeStatistics(subImage, statistic, ctrl).getValue())
            skySamples.append(afwMath.makeStatistics(subSky, statistic, ctrl).getValue())
        return imageSamples, skySamples

    def solveScales(self, scales):
        """Solve multiple scales for a single scale factor

        Having measured samples from the image and sky frame, we
        fit for the scaling factor.

        Parameters
        ----------
        scales : `list` of a `tuple` of two `numpy.ndarray` arrays
            A `list` of the results from `measureScale` method.

        Returns
        -------
        scale : `float`
            Scale factor.
        """
        imageSamples = []
        skySamples = []
        for ii, ss in scales:
            imageSamples.extend(ii)
            skySamples.extend(ss)
        assert len(imageSamples) == len(skySamples)
        imageSamples = numpy.array(imageSamples)
        skySamples = numpy.array(skySamples)

        def solve(mask):
            # Make sure we return a float, not an array.
            return afwMath.LeastSquares.fromDesignMatrix(skySamples[mask].reshape(mask.sum(), 1),
                                                         imageSamples[mask],
                                                         afwMath.LeastSquares.DIRECT_SVD).getSolution()[0]

        mask = numpy.isfinite(imageSamples) & numpy.isfinite(skySamples)
        for ii in range(self.config.skyIter):
            solution = solve(mask)
            residuals = imageSamples - solution * skySamples
            lq, uq = numpy.percentile(residuals[mask], [25, 75])
            stdev = 0.741 * (uq - lq)  # Robust stdev from IQR
            with numpy.errstate(invalid="ignore"):  # suppress NAN warnings
                bad = numpy.abs(residuals) > self.config.skyRej * stdev
            mask[bad] = False

        return solve(mask)

    def subtractSkyFrame(self, image, skyBackground, scale, bgList=None):
        """Subtract sky frame from science image

        Parameters
        ----------
        image : `lsst.afw.image.Exposure` or `lsst.afw.image.MaskedImage`
            Science image.
        skyBackground : `lsst.afw.math.BackgroundList`
            Sky background model.
        scale : `float`
            Scale to apply to background model.
        bgList : `lsst.afw.math.BackgroundList`
            List of backgrounds applied to image
        """
        if isinstance(image, afwImage.Exposure):
            image = image.getMaskedImage()
        if isinstance(image, afwImage.MaskedImage):
            image = image.getImage()
        image.scaledMinus(scale, skyBackground.getImage())
        if bgList is not None:
            # Append the sky frame to the list of applied background models
            bgData = list(skyBackground[0])
            bg = bgData[0]
            statsImage = bg.getStatsImage().clone()
            statsImage *= scale
            newBg = afwMath.BackgroundMI(bg.getImageBBox(), statsImage)
            newBgData = [newBg] + bgData[1:]
            bgList.append(newBgData)


def interpolate1D(method, xSample, ySample, xInterp):
    """Interpolate in one dimension

    Interpolates the curve provided by `xSample` and `ySample` at
    the positions of `xInterp`. Automatically backs off the
    interpolation method to achieve successful interpolation.

    Parameters
    ----------
    method : `lsst.afw.math.Interpolate.Style`
        Interpolation method to use.
    xSample : `numpy.ndarray`
        Vector of ordinates.
    ySample : `numpy.ndarray`
        Vector of coordinates.
    xInterp : `numpy.ndarray`
        Vector of ordinates to which to interpolate.

    Returns
    -------
    yInterp : `numpy.ndarray`
        Vector of interpolated coordinates.

    """
    if len(xSample) == 0:
        return numpy.ones_like(xInterp) * numpy.nan
    try:
        return afwMath.makeInterpolate(xSample.astype(float), ySample.astype(float), method).interpolate(
            xInterp.astype(float)
        )
    except Exception:
        if method == afwMath.Interpolate.CONSTANT:
            # We've already tried the most basic interpolation and it failed
            return numpy.ones_like(xInterp) * numpy.nan
        newMethod = afwMath.lookupMaxInterpStyle(len(xSample))
        if newMethod == method:
            newMethod = afwMath.Interpolate.CONSTANT
        return interpolate1D(newMethod, xSample, ySample, xInterp)


def interpolateBadPixels(array, isBad, interpolationStyle):
    """Interpolate bad pixels in an image array

    The bad pixels are modified in the array.

    Parameters
    ----------
    array : `numpy.ndarray`
        Image array with bad pixels.
    isBad : `numpy.ndarray` of type `bool`
        Boolean array indicating which pixels are bad.
    interpolationStyle : `str`
        Style for interpolation (see `lsst.afw.math.Background`);
        supported values are CONSTANT, LINEAR, NATURAL_SPLINE,
        AKIMA_SPLINE.
    """
    if numpy.all(isBad):
        raise RuntimeError("No good pixels in image array")
    height, width = array.shape
    xIndices = numpy.arange(width, dtype=float)
    yIndices = numpy.arange(height, dtype=float)
    method = afwMath.stringToInterpStyle(interpolationStyle)
    isGood = ~isBad
    for y in range(height):
        if numpy.any(isBad[y, :]) and numpy.any(isGood[y, :]):
            array[y][isBad[y]] = interpolate1D(
                method, xIndices[isGood[y]], array[y][isGood[y]], xIndices[isBad[y]]
            )

    isBad = numpy.isnan(array)
    isGood = ~isBad
    for x in range(width):
        if numpy.any(isBad[:, x]) and numpy.any(isGood[:, x]):
            array[:, x][isBad[:, x]] = interpolate1D(
                method, yIndices[isGood[:, x]], array[:, x][isGood[:, x]], yIndices[isBad[:, x]]
            )


class FocalPlaneBackgroundConfig(Config):
    """Configuration for FocalPlaneBackground

    Note that `xSize` and `ySize` are floating-point values, as
    the focal plane frame is usually defined in units of microns
    or millimetres rather than pixels. As such, their values will
    need to be revised according to each particular camera. For
    this reason, no defaults are set for those.
    """

    xSize = Field(dtype=float, doc="Bin size in x")
    ySize = Field(dtype=float, doc="Bin size in y")
    pixelSize = Field(dtype=float, default=1.0, doc="Pixel size in same units as xSize/ySize")
    minFrac = Field(dtype=float, default=0.1, doc="Minimum fraction of bin size for good measurement")
    mask = ListField(
        dtype=str,
        doc="Mask planes to treat as bad",
        default=["BAD", "SAT", "INTRP", "DETECTED", "DETECTED_NEGATIVE", "EDGE", "NO_DATA"],
    )
    interpolation = ChoiceField(
        doc="how to interpolate the background values. This maps to an enum; see afw::math::Background",
        dtype=str,
        default="AKIMA_SPLINE",
        optional=True,
        allowed={
            "CONSTANT": "Use a single constant value",
            "LINEAR": "Use linear interpolation",
            "NATURAL_SPLINE": "cubic spline with zero second derivative at endpoints",
            "AKIMA_SPLINE": "higher-level nonlinear spline that is more robust to outliers",
            "NONE": "No background estimation is to be attempted",
        },
    )
    doSmooth = Field(dtype=bool, default=False, doc="Do smoothing?")
    smoothScale = Field(dtype=float, default=2.0, doc="Smoothing scale, as a multiple of the bin size")
    binning = Field(dtype=int, default=64, doc="Binning to use for CCD background model (pixels)")


class FocalPlaneBackground:
    """Background model for a focal plane camera

    We model the background empirically with the "superpixel" method: we
    measure the background in each superpixel and interpolate between
    superpixels to yield the model.

    The principal difference between this and `lsst.afw.math.BackgroundMI`
    is that here the superpixels are defined in the frame of the focal
    plane of the camera which removes discontinuities across detectors.

    The constructor you probably want to use is the `fromCamera` classmethod.

    There are two use patterns for building a background model:

    * Serial: create a `FocalPlaneBackground`, then `addCcd` for each of the
      CCDs in an exposure.

    * Parallel: create a `FocalPlaneBackground`, then `clone` it for each
      of the CCDs in an exposure and use those to `addCcd` their respective
      CCD image. Finally, `merge` all the clones into the original.

    Once you've built the background model, you can apply it to individual
    CCDs with the `toCcdBackground` method.
    """

    @classmethod
    def fromCamera(cls, config, camera):
        """Construct from a camera object

        Parameters
        ----------
        config : `FocalPlaneBackgroundConfig`
            Configuration for measuring backgrounds.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera for which to measure backgrounds.
        """
        cameraBox = geom.Box2D()
        for ccd in camera:
            for point in ccd.getCorners(afwCameraGeom.FOCAL_PLANE):
                cameraBox.include(point)

        width, height = cameraBox.getDimensions()
        # Offset so that we run from zero
        offset = geom.Extent2D(cameraBox.getMin()) * -1
        # Add an extra pixel buffer on either side
        dims = geom.Extent2I(
            int(numpy.ceil(width / config.xSize)) + 2, int(numpy.ceil(height / config.ySize)) + 2
        )
        # Transform takes us from focal plane coordinates --> sample coordinates
        transform = (
            geom.AffineTransform.makeTranslation(geom.Extent2D(1, 1))
            * geom.AffineTransform.makeScaling(1.0 / config.xSize, 1.0 / config.ySize)
            * geom.AffineTransform.makeTranslation(offset)
        )

        return cls(config, dims, afwGeom.makeTransform(transform))

    @classmethod
    def fromSimilar(cls, other):
        """Construct from an object that has the same interface.

        Parameters
        ----------
        other : `FocalPlaneBackground`-like
            An object that matches the interface of `FocalPlaneBackground`
            but which may be different.

        Returns
        -------
        background : `FocalPlaneBackground`
            Something guaranteed to be a `FocalPlaneBackground`.
        """
        return cls(other.config, other.dims, other.transform, other._values, other._numbers)

    def __init__(self, config, dims, transform, values=None, numbers=None):
        """Constructor

        Developers should note that changes to the signature of this method
        require coordinated changes to the `__reduce__` and `clone` methods.

        Parameters
        ----------
        config : `FocalPlaneBackgroundConfig`
            Configuration for measuring backgrounds.
        dims : `lsst.geom.Extent2I`
            Dimensions for background samples.
        transform : `lsst.afw.geom.TransformPoint2ToPoint2`
            Transformation from focal plane coordinates to sample coordinates.
        values : `lsst.afw.image.ImageF`
            Measured background values.
        numbers : `lsst.afw.image.ImageF`
            Number of pixels in each background measurement.
        """
        self.config = config
        self.dims = dims
        self.transform = transform

        if values is None:
            values = afwImage.ImageF(self.dims)
            values.set(0.0)
        else:
            values = values.clone()
        assert values.getDimensions() == self.dims
        self._values = values
        if numbers is None:
            numbers = afwImage.ImageF(self.dims)  # float for dynamic range and convenience
            numbers.set(0.0)
        else:
            numbers = numbers.clone()
        assert numbers.getDimensions() == self.dims
        self._numbers = numbers

    def __reduce__(self):
        return self.__class__, (self.config, self.dims, self.transform, self._values, self._numbers)

    def clone(self):
        return self.__class__(self.config, self.dims, self.transform, self._values, self._numbers)

    def addCcd(self, exposure):
        """Add CCD to model

        We measure the background on the CCD (clipped mean), and record
        the results in the model.  For simplicity, measurements are made
        in a box on the CCD corresponding to the warped coordinates of the
        superpixel rather than accounting for little rotations, etc.
        We also record the number of pixels used in the measurement so we
        can have a measure of confidence in each bin's value.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            CCD exposure to measure
        """
        detector = exposure.getDetector()
        transform = detector.getTransformMap().getTransform(
            detector.makeCameraSys(afwCameraGeom.PIXELS), detector.makeCameraSys(afwCameraGeom.FOCAL_PLANE)
        )
        image = exposure.getMaskedImage()
        maskVal = image.getMask().getPlaneBitMask(self.config.mask)

        # Warp the binned image to the focal plane
        toSample = transform.then(self.transform)  # CCD pixels --> focal plane --> sample

        warped = afwImage.ImageF(self._values.getBBox())
        warpedCounts = afwImage.ImageF(self._numbers.getBBox())
        width, height = warped.getDimensions()

        stats = afwMath.StatisticsControl()
        stats.setAndMask(maskVal)
        stats.setNanSafe(True)
        # Iterating over individual pixels in python is usually bad because it's slow, but there aren't many.
        pixels = itertools.product(range(width), range(height))
        for xx, yy in pixels:
            llc = toSample.applyInverse(geom.Point2D(xx - 0.5, yy - 0.5))
            urc = toSample.applyInverse(geom.Point2D(xx + 0.5, yy + 0.5))
            bbox = geom.Box2I(geom.Point2I(llc), geom.Point2I(urc))
            bbox.clip(image.getBBox())
            if bbox.isEmpty():
                continue
            subImage = image.Factory(image, bbox)
            result = afwMath.makeStatistics(subImage, afwMath.MEANCLIP | afwMath.NPOINT, stats)
            mean = result.getValue(afwMath.MEANCLIP)
            num = result.getValue(afwMath.NPOINT)
            if not numpy.isfinite(mean) or not numpy.isfinite(num):
                continue
            warped[xx, yy, afwImage.LOCAL] = mean * num
            warpedCounts[xx, yy, afwImage.LOCAL] = num

        self._values += warped
        self._numbers += warpedCounts

    def toCcdBackground(self, detector, bbox):
        """Produce a background model for a CCD

        The superpixel background model is warped back to the
        CCD frame, for application to the individual CCD.

        Parameters
        ----------
        detector : `lsst.afw.cameraGeom.Detector`
            CCD for which to produce background model.
        bbox : `lsst.geom.Box2I`
            Bounding box of CCD exposure.

        Returns
        -------
        bg : `lsst.afw.math.BackgroundList`
            Background model for CCD.
        """
        transform = detector.getTransformMap().getTransform(
            detector.makeCameraSys(afwCameraGeom.PIXELS), detector.makeCameraSys(afwCameraGeom.FOCAL_PLANE)
        )
        binTransform = geom.AffineTransform.makeScaling(
            self.config.binning
        ) * geom.AffineTransform.makeTranslation(geom.Extent2D(0.5, 0.5))

        # Binned image on CCD --> unbinned image on CCD --> focal plane --> binned focal plane
        toSample = afwGeom.makeTransform(binTransform).then(transform).then(self.transform)

        focalPlane = self.getStatsImage()
        fpNorm = afwImage.ImageF(focalPlane.getBBox())
        fpNorm.set(1.0)

        image = afwImage.ImageF(bbox.getDimensions() // self.config.binning)
        norm = afwImage.ImageF(image.getBBox())
        ctrl = afwMath.WarpingControl("bilinear")
        afwMath.warpImage(image, focalPlane, toSample.inverted(), ctrl)
        afwMath.warpImage(norm, fpNorm, toSample.inverted(), ctrl)
        image /= norm

        mask = afwImage.Mask(image.getBBox())
        isBad = numpy.isnan(image.getArray())
        mask.getArray()[isBad] = mask.getPlaneBitMask("BAD")
        image.getArray()[isBad] = image.getArray()[~isBad].mean()

        return afwMath.BackgroundList(
            (
                afwMath.BackgroundMI(bbox, afwImage.makeMaskedImage(image, mask)),
                afwMath.stringToInterpStyle(self.config.interpolation),
                afwMath.stringToUndersampleStyle("REDUCE_INTERP_ORDER"),
                afwMath.ApproximateControl.UNKNOWN,
                0,
                0,
                False,
            )
        )

    def merge(self, other):
        """Merge with another FocalPlaneBackground

        This allows multiple background models to be constructed from
        different CCDs, and then merged to form a single consistent
        background model for the entire focal plane.

        Parameters
        ----------
        other : `FocalPlaneBackground`
            Another background model to merge.

        Returns
        -------
        self : `FocalPlaneBackground`
            The merged background model.
        """
        if (self.config.xSize, self.config.ySize) != (other.config.xSize, other.config.ySize):
            raise RuntimeError(
                "Size mismatch: %s vs %s"
                % ((self.config.xSize, self.config.ySize), (other.config.xSize, other.config.ySize))
            )
        if self.dims != other.dims:
            raise RuntimeError("Dimensions mismatch: %s vs %s" % (self.dims, other.dims))
        self._values += other._values
        self._numbers += other._numbers
        return self

    def __iadd__(self, other):
        """Merge with another FocalPlaneBackground

        Parameters
        ----------
        other : `FocalPlaneBackground`
            Another background model to merge.

        Returns
        -------
        self : `FocalPlaneBackground`
            The merged background model.
        """
        return self.merge(other)

    def getStatsImage(self):
        """Return the background model data

        This is the measurement of the background for each of the superpixels.
        """
        values = self._values.clone()
        values /= self._numbers
        thresh = (
            self.config.minFrac
            * (self.config.xSize / self.config.pixelSize)
            * (self.config.ySize / self.config.pixelSize)
        )
        isBad = self._numbers.getArray() < thresh
        if self.config.doSmooth:
            array = values.getArray()
            array[:] = smoothArray(array, isBad, self.config.smoothScale)
            isBad = numpy.isnan(values.array)
        if numpy.any(isBad):
            interpolateBadPixels(values.getArray(), isBad, self.config.interpolation)
        return values


class TractBackgroundConfig(Config):
    """Configuration for TractBackground

    Note that `xBin` and `yBin` are in pixels, as unlike FocalPlaneBackground,
    translation from warps to tract and back only requires geometric
    transformations in the warped pixel plane.
    """

    xBin = Field(dtype=float, default=500, doc="Bin size in x")
    yBin = Field(dtype=float, default=500, doc="Bin size in y")
    minFrac = Field(dtype=float, default=0.1, doc="Minimum fraction of bin size for good measurement")
    mask = ListField(
        dtype=str,
        doc="Mask planes to treat as bad",
        default=["BAD", "SAT", "INTRP", "DETECTED", "DETECTED_NEGATIVE", "EDGE", "NO_DATA"],
    )
    interpolation = ChoiceField(
        doc="how to interpolate the background values. This maps to an enum; see afw::math::Background",
        dtype=str,
        default="AKIMA_SPLINE",
        optional=True,
        allowed={
            "CONSTANT": "Use a single constant value",
            "LINEAR": "Use linear interpolation",
            "NATURAL_SPLINE": "cubic spline with zero second derivative at endpoints",
            "AKIMA_SPLINE": "higher-level nonlinear spline that is more robust to outliers",
            "NONE": "No background estimation is to be attempted",
        },
    )
    doSmooth = Field(dtype=bool, default=False, doc="Do smoothing?")
    smoothScale = Field(dtype=float, default=2.0, doc="Smoothing scale, as a multiple of the bin size")
    binning = Field(dtype=int, default=64, doc="Binning to use for warp background model (pixels)")


class TractBackground:
    """
    As FocalPlaneBackground, but works in warped tract coordinates
    """
    @classmethod
    def fromSimilar(cls, other):
        """Construct from an object that has the same interface.

        Parameters
        ----------
        other : `TractBackground`-like
            An object that matches the interface of `TractBackground`
            but which may be different.

        Returns
        -------
        background : `TractBackground`
            Something guaranteed to be a `TractBackground`.
        """
        return cls(other.config, other.tract, other.dims, other.transform, other._values, other._numbers)

    def __init__(self, config, skymap, tract, values=None, numbers=None):
        """Constructor

        Developers should note that changes to the signature of this method
        require coordinated changes to the `__reduce__` and `clone` methods.

        Parameters
        ----------
        config : `TractBackgroundConfig`
            Configuration for measuring tract backgrounds.
        skymap : `lsst.skymap.ringsSkyMap.RingsSkyMap`
            Skymap object
        tract : `int`
            Placeholder Tract ID
        transform : `lsst.afw.geom.TransformPoint2ToPoint2`
            Transformation from tract coordinates to warp coordinates.
        values : `lsst.afw.image.ImageF`
            Measured background values.
        numbers : `lsst.afw.image.ImageF`
            Number of pixels in each background measurement.
        """
        self.config = config
        self.skymap = skymap
        self.tract = tract
        self.tractInfo = self.skymap.generateTract(tract)
        tractDimX, tractDimY = self.tractInfo.getBBox().getDimensions()
        self.dims = geom.Extent2I(tractDimX / self.config.xBin,
                                  tractDimY / self.config.yBin)

        if values is None:
            values = afwImage.ImageF(self.dims)
            values.set(0.0)
        else:
            values = values.clone()
        assert values.getDimensions() == self.dims
        self._values = values
        if numbers is None:
            numbers = afwImage.ImageF(self.dims)  # float for dynamic range and convenience
            numbers.set(0.0)
        else:
            numbers = numbers.clone()
        assert numbers.getDimensions() == self.dims
        self._numbers = numbers

    def __reduce__(self):
        return self.__class__, (self.config, self.skymap, self.tract, self._values, self._numbers)

    def clone(self):
        return self.__class__(self.config, self.skymap, self.tract, self._values, self._numbers)

    def addWarp(self, warp):
        """
        Equivalent to FocalPlaneBackground.addCcd(), but on warps instead.
        Bins masked images of warps and adds these values into a blank image
        with the binned tract dimensions at the location of the warp in the
        tract.

        Parameters
        ----------
        warp : `lsst.afw.image.ExposureF`
            Warped image corresponding to a single patch in a single visit
        """
        image = warp.getMaskedImage()
        maskVal = image.getMask().getPlaneBitMask(self.config.mask)
        # Photometric scaling necessary for contiguous background across tract
        image.image.array *= warp.getPhotoCalib().instFluxToNanojansky(1)

        warped = afwImage.ImageF(self._values.getBBox())
        warpedCounts = afwImage.ImageF(self._numbers.getBBox())

        stats = afwMath.StatisticsControl()
        stats.setAndMask(maskVal)
        stats.setNanSafe(True)

        # Pixel locations in binned tract-scale image
        pixels = itertools.product(
            numpy.arange(warped.getBBox().getMinX(), warped.getBBox().getMaxX() + 1),
            numpy.arange(warped.getBBox().getMinY(), warped.getBBox().getMaxY() + 1),
        )
        for xx, yy in pixels:
            llc = geom.Point2D((xx - 0.5) * self.config.xBin, (yy - 0.5) * self.config.yBin)
            urc = geom.Point2D(
                (xx + 0.5) * self.config.xBin + self.config.xBin - 1,
                (yy + 0.5) * self.config.yBin + self.config.yBin - 1,
            )
            bbox = geom.Box2I(geom.Point2I(llc), geom.Point2I(urc))
            bbox.clip(image.getBBox())  # Works in tract coordinates
            if bbox.isEmpty():
                continue
            subImage = image.Factory(image, bbox)
            result = afwMath.makeStatistics(subImage, afwMath.MEANCLIP | afwMath.NPOINT, stats)
            mean = result.getValue(afwMath.MEANCLIP)
            num = result.getValue(afwMath.NPOINT)
            if not numpy.isfinite(mean) or not numpy.isfinite(num):
                continue
            warped[xx, yy, afwImage.LOCAL] = mean * num
            warpedCounts[xx, yy, afwImage.LOCAL] = num

        self._values += warped
        self._numbers += warpedCounts

    def merge(self, other):
        """Merge with another TractBackground

        This allows multiple background models to be constructed from
        different warps, and then merged to form a single consistent
        background model for the entire tract.

        Parameters
        ----------
        other : `TractBackground`
            Another background model to merge.

        Returns
        -------
        self : `TractBackground`
            The merged background model.
        """
        if (self.config.xBin, self.config.yBin) != (other.config.xBin, other.config.yBin):
            raise RuntimeError(
                "Size mismatch: %s vs %s"
                % ((self.config.xBin, self.config.yBin), (other.config.xBin, other.config.yBin))
            )
        if self.dims != other.dims:
            raise RuntimeError("Dimensions mismatch: %s vs %s" % (self.dims, other.dims))
        self._values += other._values
        self._numbers += other._numbers
        return self

    def __iadd__(self, other):
        """Merge with another TractBackground

        Parameters
        ----------
        other : `TractBackground`
            Another background model to merge.

        Returns
        -------
        self : `TractBackground`
            The merged background model.
        """
        return self.merge(other)

    def toWarpBackground(self, warp):
        """
        Equivalent of FocalPlaneBackground.toCcdBackground(), but creates a
        background model for a warp using a full tract model.

        Parameters
        ----------
        warp : `lsst.afw.image.ExposureF`
            Warped image corresponding to a single patch in a single visit

        Returns
        -------
        bg : `lsst.afw.math.BackgroundList`
            Background model for warp
        """
        # Transform to binned warp plane
        binTransform = geom.AffineTransform.makeScaling(self.config.binning)

        # Transform from binned tract plane to tract plane
        # Start at the patch corner, not the warp corner overlap region
        wcs = self.tractInfo.getWcs()
        coo = wcs.pixelToSky(1, 1)
        ptch = self.tractInfo.findPatch(coo)
        ptchDimX, ptchDimY = ptch.getInnerBBox().getDimensions()
        if ptchDimX != ptchDimY:
            raise ValueError(
                "Patch dimensions %d,%d are unequal: cannot proceed as written."
                % (ptchDimX, ptchDimY)
            )
        ptchOutDimX, _ = ptch.getOuterBBox().getDimensions()
        overlap = ptchDimX - ptchOutDimX
        corner = warp.getBBox().getMin()
        if corner[0] % ptchDimX != 0:
            corner[0] += overlap
            corner[1] += overlap
        offset = geom.Extent2D(corner[0], corner[1])
        tractTransform = (
            geom.AffineTransform.makeTranslation(geom.Extent2D(-0.5, -0.5))
            * geom.AffineTransform.makeScaling(1.0 / self.config.xBin, 1.0 / self.config.yBin)
            * geom.AffineTransform.makeTranslation(offset)
        )
        transform = afwGeom.makeTransform(tractTransform)

        # Full transform
        toSample = afwGeom.makeTransform(binTransform).then(transform)

        # Full tract sky model and normalization array
        tractPlane = self.getStatsImage()
        tpNorm = afwImage.ImageF(tractPlane.getBBox())
        tpNorm.set(1.0)

        # Binned warp image and normalization array
        bbox = warp.getBBox()
        image = afwImage.ImageF(bbox.getDimensions() // self.config.binning)
        norm = afwImage.ImageF(image.getBBox())

        # Warps full tract model to warp image scale
        ctrl = afwMath.WarpingControl("bilinear")
        afwMath.warpImage(image, tractPlane, toSample.inverted(), ctrl)
        afwMath.warpImage(norm, tpNorm, toSample.inverted(), ctrl)
        image /= norm
        # Convert back to counts so the model can be subtracted w/o conversion
        image /= warp.getPhotoCalib().instFluxToNanojansky(1)

        # Only sky background model, so include only null values in mask plane
        mask = afwImage.Mask(image.getBBox())
        isBad = numpy.isnan(image.getArray())
        mask.getArray()[isBad] = mask.getPlaneBitMask("BAD")
        image.getArray()[isBad] = image.getArray()[~isBad].mean()

        return afwMath.BackgroundList(
            (
                afwMath.BackgroundMI(warp.getBBox(), afwImage.makeMaskedImage(image, mask)),
                afwMath.stringToInterpStyle(self.config.interpolation),
                afwMath.stringToUndersampleStyle("REDUCE_INTERP_ORDER"),
                afwMath.ApproximateControl.UNKNOWN,
                0,
                0,
                False,
            )
        )

    def getStatsImage(self):
        """Return the background model data

        This is the measurement of the background for each of the superpixels.
        """
        values = self._values.clone()
        values /= self._numbers
        # TODO: filling in bad pixels.  Currently BAD mask plane includes both
        # chip gaps and regions outside FP, so interpolating across chip gaps
        # also includes extrapolating flux outside the FP, which is
        # undesirable.  But interpolation and extrapolation aren't currently
        # separable, so for now this step is just not done.

        return values


class MaskObjectsConfig(Config):
    """Configuration for MaskObjectsTask"""

    nIter = Field(dtype=int, default=3, doc="Number of iterations")
    subtractBackground = ConfigurableField(
        target=measAlg.SubtractBackgroundTask, doc="Background subtraction"
    )
    detection = ConfigurableField(target=measAlg.SourceDetectionTask, doc="Source detection")
    detectSigma = Field(dtype=float, default=5.0, doc="Detection threshold (standard deviations)")
    doInterpolate = Field(dtype=bool, default=True, doc="Interpolate when removing objects?")
    interpolate = ConfigurableField(target=measAlg.SubtractBackgroundTask, doc="Interpolation")

    def setDefaults(self):
        self.detection.reEstimateBackground = False
        self.detection.doTempLocalBackground = False
        self.detection.doTempWideBackground = False
        self.detection.thresholdValue = 2.5
        self.subtractBackground.binSize = 1024
        self.subtractBackground.useApprox = False
        self.interpolate.binSize = 256
        self.interpolate.useApprox = False

    def validate(self):
        if (
            self.detection.reEstimateBackground
            or self.detection.doTempLocalBackground
            or self.detection.doTempWideBackground
        ):
            raise RuntimeError(
                "Incorrect settings for object masking: reEstimateBackground, "
                "doTempLocalBackground and doTempWideBackground must be False"
            )


class MaskObjectsTask(Task):
    """Iterative masking of objects on an Exposure

    This task makes more exhaustive object mask by iteratively doing detection
    and background-subtraction. The purpose of this task is to get true
    background removing faint tails of large objects. This is useful to get a
    clean sky estimate from relatively small number of visits.

    We deliberately use the specified ``detectSigma`` instead of the PSF,
    in order to better pick up the faint wings of objects.
    """

    ConfigClass = MaskObjectsConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Disposable schema suppresses warning from SourceDetectionTask.__init__
        self.makeSubtask("detection", schema=afwTable.Schema())
        self.makeSubtask("interpolate")
        self.makeSubtask("subtractBackground")

    def run(self, exposure, maskPlanes=None):
        """Mask objects on Exposure

        Objects are found and removed.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure on which to mask objects.
        maskPlanes : iterable of `str`, optional
            List of mask planes to remove.
        """
        self.findObjects(exposure)
        self.removeObjects(exposure, maskPlanes)

    def findObjects(self, exposure):
        """Iteratively find objects on an exposure

        Objects are masked with the ``DETECTED`` mask plane.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure on which to mask objects.
        """
        for _ in range(self.config.nIter):
            bg = self.subtractBackground.run(exposure).background
            self.detection.detectFootprints(exposure, sigma=self.config.detectSigma, clearMask=True)
            exposure.maskedImage += bg.getImage()

    def removeObjects(self, exposure, maskPlanes=None):
        """Remove objects from exposure

        We interpolate over using a background model if ``doInterpolate`` is
        set; otherwise we simply replace everything with the median.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure on which to mask objects.
        maskPlanes : iterable of `str`, optional
            List of mask planes to remove. ``DETECTED`` will be added as well.
        """
        image = exposure.image
        mask = exposure.mask
        maskVal = mask.getPlaneBitMask("DETECTED")
        if maskPlanes is not None:
            maskVal |= mask.getPlaneBitMask(maskPlanes)
        isBad = mask.array & maskVal > 0

        if self.config.doInterpolate:
            smooth = self.interpolate.fitBackground(exposure.maskedImage)
            replace = smooth.getImageF().array[isBad]
            mask.array &= ~mask.getPlaneBitMask(["DETECTED"])
        else:
            replace = numpy.median(image.array[~isBad])
        image.array[isBad] = replace


def smoothArray(array, bad, sigma):
    """Gaussian-smooth an array while ignoring bad pixels

    It's not sufficient to set the bad pixels to zero, as then they're treated
    as if they are zero, rather than being ignored altogether. We need to apply
    a correction to that image that removes the effect of the bad pixels.

    Parameters
    ----------
    array : `numpy.ndarray` of floating-point
        Array to smooth.
    bad : `numpy.ndarray` of `bool`
        Flag array indicating bad pixels.
    sigma : `float`
        Gaussian sigma.

    Returns
    -------
    convolved : `numpy.ndarray`
        Smoothed image.
    """
    convolved = gaussian_filter(numpy.where(bad, 0.0, array), sigma, mode="constant", cval=0.0)
    denominator = gaussian_filter(numpy.where(bad, 0.0, 1.0), sigma, mode="constant", cval=0.0)
    return convolved / denominator


def _create_module_child(name):
    """Create an empty module attached to the relevant parent."""
    parent, child = name.rsplit(".", 1)
    spec = importlib.machinery.ModuleSpec(name, None)
    newmod = importlib.util.module_from_spec(spec)
    setattr(sys.modules[parent], child, newmod)
    sys.modules[name] = newmod
    return newmod


# This module used to be located in pipe_drivers as
# lsst.pipe.drivers.background.  All pickled datasets using this name
# require that it still exists as that name. Therefore we create a faked
# version of lsst.pipe.drivers if that package is not around.
try:
    import lsst.pipe.drivers.background  # noqa: F401
except ImportError:
    # Create a fake lsst.pipe.drivers module and attach it to lsst.pipe.
    pipe_drivers = _create_module_child("lsst.pipe.drivers")

    # Create a background module and attach that to drivers.
    pipe_drivers_background = _create_module_child("lsst.pipe.drivers.background")

    # Attach the classes to the faked pipe_drivers variant.
    setattr(pipe_drivers_background, FocalPlaneBackground.__name__, FocalPlaneBackground)
    setattr(pipe_drivers_background, FocalPlaneBackgroundConfig.__name__, FocalPlaneBackgroundConfig)
