from __future__ import absolute_import, division, print_function

import numpy
import itertools

import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as afwCameraGeom

from lsst.pex.config import Config, Field, ListField, ChoiceField, ConfigField, RangeField
from lsst.pipe.base import Task


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
    good = numpy.abs(array - median) < rej*0.74*(q3 - q1)
    return array[good].mean()


class BackgroundConfig(Config):
    """Configuration for background measurement"""
    statistic = ChoiceField(dtype=str, default="MEANCLIP", doc="type of statistic to use for grid points",
                            allowed={"MEANCLIP": "clipped mean",
                                     "MEAN": "unclipped mean",
                                     "MEDIAN": "median",})
    xBinSize = RangeField(dtype=int, default=32, min=1, doc="Superpixel size in x")
    yBinSize = RangeField(dtype=int, default=32, min=1, doc="Superpixel size in y")
    algorithm = ChoiceField(dtype=str, default="NATURAL_SPLINE", optional=True,
                            doc="How to interpolate the background values. "
                                "This maps to an enum; see afw::math::Background",
                            allowed={
                                "CONSTANT": "Use a single constant value",
                                "LINEAR": "Use linear interpolation",
                                "NATURAL_SPLINE": "cubic spline with zero second derivative at endpoints",
                                "AKIMA_SPLINE": "higher-level nonlinear spline that is more robust to outliers",
                                "NONE": "No background estimation is to be attempted",
                            })
    mask = ListField(dtype=str, default=["SAT", "BAD", "EDGE", "DETECTED", "DETECTED_NEGATIVE", "NO_DATA",],
                     doc="Names of mask planes to ignore while estimating the background")


class SkyStatsConfig(Config):
    """Parameters controlling the measurement of sky statistics"""
    statistic = ChoiceField(dtype=str, default="MEANCLIP", doc="type of statistic to use for grid points",
                            allowed={"MEANCLIP": "clipped mean",
                                     "MEAN": "unclipped mean",
                                     "MEDIAN": "median",})
    clip = Field(doc="Clipping threshold for background", dtype=float, default=3.0)
    nIter = Field(doc="Clipping iterations for background", dtype=int, default=3)
    mask = ListField(doc="Mask planes to reject", dtype=str,
                     default=["SAT", "DETECTED", "DETECTED_NEGATIVE", "BAD", "NO_DATA",])


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

    def getSkyData(self, butler, calibId):
        """Retrieve sky frame from the butler

        Parameters
        ----------
        butler : `lsst.daf.persistence.Butler`
            Data butler
        calibId : `dict`
            Data identifier for calib

        Returns
        -------
        sky : `lsst.afw.math.BackgroundList`
            Sky frame
        """
        exp = butler.get("sky", calibId)
        return self.exposureToBackground(exp)

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
        xMin = header.get("BOX.MINX")
        yMin = header.get("BOX.MINY")
        xMax = header.get("BOX.MAXX")
        yMax = header.get("BOX.MAXY")
        algorithm = header.get("ALGORITHM")
        bbox = afwGeom.Box2I(afwGeom.Point2I(xMin, yMin), afwGeom.Point2I(xMax, yMax))
        return afwMath.BackgroundList(
                (afwMath.BackgroundMI(bbox, bgExp.getMaskedImage()),
                 afwMath.stringToInterpStyle(algorithm),
                 afwMath.stringToUndersampleStyle("REDUCE_INTERP_ORDER"),
                 afwMath.ApproximateControl.UNKNOWN,
                 0, 0, False))

    def backgroundToExposure(self, statsImage, bbox):
        """Convert a background model to an exposure

        Calibs need to be persisted as an Exposure, so we need to convert
        the background model to an Exposure.

        Parameters
        ----------
        statsImage : `lsst.afw.image.MaskedImageF`
            Background model's statistics image.
        bbox : `lsst.afw.geom.Box2I`
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
            max(int(image.getWidth()/self.config.background.xBinSize + 0.5), 1),
            max(int(image.getHeight()/self.config.background.yBinSize + 0.5), 1),
            "REDUCE_INTERP_ORDER",
            stats,
            self.config.background.statistic
        )

        bg = afwMath.makeBackground(image, ctrl)

        return afwMath.BackgroundList((
            bg,
            afwMath.stringToInterpStyle(self.config.background.algorithm),
            afwMath.stringToUndersampleStyle("REDUCE_INTERP_ORDER"),
            afwMath.ApproximateControl.UNKNOWN,
            0, 0, False
        ))

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
        assert len(set((box.getMinX(), box.getMinY(), box.getMaxX(), box.getMaxY()) for box in boxes)) == 1, \
            "Bounding boxes not all equal"
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
            box = afwGeom.Box2I(afwGeom.Point2I(xStart, yStart), afwGeom.Point2I(xStop, yStop))
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
            return afwMath.LeastSquares.fromDesignMatrix(skySamples[mask].reshape(mask.sum(), 1),
                                                         imageSamples[mask],
                                                         afwMath.LeastSquares.DIRECT_SVD).getSolution()

        mask = numpy.isfinite(imageSamples) & numpy.isfinite(skySamples)
        for ii in range(self.config.skyIter):
            solution = solve(mask)
            residuals = imageSamples - solution*skySamples
            lq, uq = numpy.percentile(residuals[mask], [25, 75])
            stdev = 0.741*(uq - lq)  # Robust stdev from IQR
            with numpy.errstate(invalid="ignore"):  # suppress NAN warnings
                bad = numpy.abs(residuals) > self.config.skyRej*stdev
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
        return numpy.ones_like(xInterp)*numpy.nan
    try:
        return afwMath.makeInterpolate(xSample.astype(float), ySample.astype(float),
                                       method).interpolate(xInterp.astype(float))
    except:
        if method == afwMath.Interpolate.CONSTANT:
            # We've already tried the most basic interpolation and it failed
            return numpy.ones_like(xInterp)*numpy.nan
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
            array[y][isBad[y]] = interpolate1D(method, xIndices[isGood[y]], array[y][isGood[y]],
                                               xIndices[isBad[y]])

    isBad = numpy.isnan(array)
    isGood = ~isBad
    for x in range(width):
        if numpy.any(isBad[:, x]) and numpy.any(isGood[:, x]):
            array[:, x][isBad[:, x]] = interpolate1D(method, yIndices[isGood[:, x]],
                                                     array[:, x][isGood[:, x]], yIndices[isBad[:, x]])


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
    minFrac = Field(dtype=float, default=0.1, doc="Minimum fraction of bin size for good measurement")
    mask = ListField(dtype=str, doc="Mask planes to treat as bad",
                     default=["BAD", "SAT", "INTRP", "DETECTED", "DETECTED_NEGATIVE", "EDGE", "NO_DATA"])
    interpolation = ChoiceField(
        doc="how to interpolate the background values. This maps to an enum; see afw::math::Background",
        dtype=str, default="AKIMA_SPLINE", optional=True,
        allowed={
            "CONSTANT": "Use a single constant value",
            "LINEAR": "Use linear interpolation",
            "NATURAL_SPLINE": "cubic spline with zero second derivative at endpoints",
            "AKIMA_SPLINE": "higher-level nonlinear spline that is more robust to outliers",
            "NONE": "No background estimation is to be attempted",
        },
    )
    binning = Field(dtype=int, default=64, doc="Binning to use for CCD background model (pixels)")


class FocalPlaneBackground(object):
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
        cameraBox = afwGeom.Box2D()
        for ccd in camera:
            for point in ccd.getCorners(afwCameraGeom.FOCAL_PLANE):
                cameraBox.include(point)

        width, height = cameraBox.getDimensions()
        # Offset so that we run from zero
        offset = afwGeom.Extent2D(cameraBox.getMin())*-1
        # Add an extra pixel buffer on either side
        dims = afwGeom.Extent2I(int(numpy.ceil(width/config.xSize)) + 2,
                                int(numpy.ceil(height/config.ySize)) + 2)
        # Transform takes us from focal plane coordinates --> sample coordinates
        transform = (afwGeom.AffineTransform.makeTranslation(afwGeom.Extent2D(1, 1))*
                     afwGeom.AffineTransform.makeScaling(1.0/config.xSize, 1.0/config.ySize)*
                     afwGeom.AffineTransform.makeTranslation(offset))

        return cls(config, dims, afwGeom.makeTransform(transform))

    def __init__(self, config, dims, transform, values=None, numbers=None):
        """Constructor

        Developers should note that changes to the signature of this method
        require coordinated changes to the `__reduce__` and `clone` methods.

        Parameters
        ----------
        config : `FocalPlaneBackgroundConfig`
            Configuration for measuring backgrounds.
        dims : `lsst.afw.geom.Extent2I`
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
        assert(values.getDimensions() == self.dims)
        self._values = values
        if numbers is None:
            numbers = afwImage.ImageF(self.dims)  # float for dynamic range and convenience
            numbers.set(0.0)
        else:
            numbers = numbers.clone()
        assert(numbers.getDimensions() == self.dims)
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
        transform = detector.getTransformMap().getTransform(detector.makeCameraSys(afwCameraGeom.PIXELS),
                                                            detector.makeCameraSys(afwCameraGeom.FOCAL_PLANE))
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
            llc = toSample.applyInverse(afwGeom.Point2D(xx - 0.5, yy - 0.5))
            urc = toSample.applyInverse(afwGeom.Point2D(xx + 0.5, yy + 0.5))
            bbox = afwGeom.Box2I(afwGeom.Point2I(llc), afwGeom.Point2I(urc))
            bbox.clip(image.getBBox())
            if bbox.isEmpty():
                continue
            subImage = image.Factory(image, bbox)
            result = afwMath.makeStatistics(subImage, afwMath.MEANCLIP | afwMath.NPOINT, stats)
            mean = result.getValue(afwMath.MEANCLIP)
            num = result.getValue(afwMath.NPOINT)
            if not numpy.isfinite(mean) or not numpy.isfinite(num):
                continue
            warped[xx, yy, afwImage.LOCAL] = mean*num
            warpedCounts[xx, yy,afwImage.LOCAL] = num

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
        bbox : `lsst.afw.geom.Box2I`
            Bounding box of CCD exposure.

        Returns
        -------
        bg : `lsst.afw.math.BackgroundList`
            Background model for CCD.
        """
        transform = detector.getTransformMap().getTransform(detector.makeCameraSys(afwCameraGeom.PIXELS),
                                                            detector.makeCameraSys(afwCameraGeom.FOCAL_PLANE))
        binTransform = (afwGeom.AffineTransform.makeScaling(self.config.binning)*
                        afwGeom.AffineTransform.makeTranslation(afwGeom.Extent2D(0.5, 0.5)))

        # Binned image on CCD --> unbinned image on CCD --> focal plane --> binned focal plane
        toSample = afwGeom.makeTransform(binTransform).then(transform).then(self.transform)

        focalPlane = self.getStatsImage()
        fpNorm = afwImage.ImageF(focalPlane.getBBox())
        fpNorm.set(1.0)

        image = afwImage.ImageF(bbox.getDimensions()//self.config.binning)
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
            (afwMath.BackgroundMI(bbox, afwImage.makeMaskedImage(image, mask)),
             afwMath.stringToInterpStyle(self.config.interpolation),
             afwMath.stringToUndersampleStyle("REDUCE_INTERP_ORDER"),
             afwMath.ApproximateControl.UNKNOWN,
             0, 0, False)
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
            raise RuntimeError("Size mismatch: %s vs %s" % ((self.config.xSize, self.config.ySize),
                                                            (other.config.xSize, other.config.ySize)))
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
        thresh = self.config.minFrac*self.config.xSize*self.config.ySize
        isBad = self._numbers.getArray() < thresh
        interpolateBadPixels(values.getArray(), isBad, self.config.interpolation)
        return values


