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
    "TractBackground",
    "TractBackgroundConfig",
]

import itertools

import numpy

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom as geom
from lsst.pex.config import ChoiceField, Config, Field, ListField
from lsst.pipe.tasks.background import interpolateBadPixels, smoothArray


class TractBackgroundConfig(Config):
    """Configuration for TractBackground

    Parameters `xBin` and `yBin` are in pixels as translation from warps to
    tract and back only requires geometric transformations in the warped pixel
    plane.
    """

    xBin = Field(dtype=float, default=200, doc="Bin size in x")
    yBin = Field(dtype=float, default=200, doc="Bin size in y")
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
    """Background model for a tract, comprised of warped exposures.

    Similar to background.FocalPlaneBackground class, in that we model the
    background using the "superpixel" method, measuring the background in each
    superpixel and interpolating between them.

    There is one use pattern for building a background model: create a
    `TractBackground`, then `addWarp` for each of the patches in a tract.

    Once you've built the background model, you can apply it to individual
    warps with the `toWarpBackground` method.

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
    variances : `lsst.afw.image.ImageF`
        Measured background variances.
    """

    def __init__(self, config, skymap, tract, values=None, numbers=None, variances=None):
        # Developers should note that changes to the signature of this method
        # require coordinated changes to the `__reduce__` and `clone` methods.
        self.config = config
        self.skymap = skymap
        self.tract = tract
        self.tractInfo = self.skymap[tract]
        tractDimX, tractDimY = self.tractInfo.getBBox().getDimensions()
        self.dims = geom.Extent2I(tractDimX / self.config.xBin, tractDimY / self.config.yBin)

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
        if variances is None:
            variances = afwImage.ImageF(self.dims)
            variances.set(0.0)
        else:
            variances = variances.clone()
        assert variances.getDimensions() == self.dims
        self._variances = variances

    def __reduce__(self):
        return self.__class__, (
            self.config,
            self.skymap,
            self.tract,
            self._values,
            self._numbers,
            self._variances,
        )

    def clone(self):
        return self.__class__(
            self.config, self.skymap, self.tract, self._values, self._numbers, self._variances
        )

    def addWarp(self, warp):
        """Bins masked images of warps and adds these values into a blank image
        with the binned tract dimensions at the location of the warp in the
        tract.

        Parameters
        ----------
        warp : `lsst.afw.image.ExposureF`
            Warped image corresponding to a single patch in a single visit
        """
        image = warp.getMaskedImage()
        maskVal = image.getMask().getPlaneBitMask(self.config.mask)

        warped = afwImage.ImageF(self._values.getBBox())
        warpedCounts = afwImage.ImageF(self._numbers.getBBox())
        warpedVariances = afwImage.ImageF(self._variances.getBBox())

        stats = afwMath.StatisticsControl()
        stats.setAndMask(maskVal)
        stats.setNanSafe(True)

        # Pixel locations in binned tract-scale image
        pixels = itertools.product(
            warped.getBBox().x.arange(),
            warped.getBBox().y.arange(),
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
            result = afwMath.makeStatistics(
                subImage, afwMath.MEANCLIP | afwMath.NPOINT | afwMath.VARIANCECLIP, stats
            )
            mean = result.getValue(afwMath.MEANCLIP)
            num = result.getValue(afwMath.NPOINT)
            var = result.getValue(afwMath.VARIANCECLIP)
            if not numpy.isfinite(mean) or not numpy.isfinite(num):
                continue
            warped[xx, yy, afwImage.LOCAL] = mean * num
            warpedCounts[xx, yy, afwImage.LOCAL] = num
            warpedVariances[xx, yy, afwImage.LOCAL] = var * num

        self._values += warped
        self._numbers += warpedCounts
        self._variances += warpedVariances

    def __iadd__(self, other):
        """Merges with another TractBackground

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
        self._variances += other._variances
        return self

    def toWarpBackground(self, warp):
        """Produce a background model for a warp

        The superpixel model is transformed back to the native pixel
        resolution, for application to the background of an individual warp.

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
                "Patch dimensions %d,%d are unequal: cannot proceed as written." % (ptchDimX, ptchDimY)
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

        thresh = self.config.minFrac * (self.config.xBin) * (self.config.yBin)
        isBad = self._numbers.getArray() < thresh
        if self.config.doSmooth:
            array = values.getArray()
            array[:] = smoothArray(array, isBad, self.config.smoothScale)
            isBad = numpy.isnan(values.array)
        # This also extrapolates outside the focal plane to the tract edges
        if numpy.any(isBad):
            interpolateBadPixels(values.getArray(), isBad, self.config.interpolation)

        return values

    def getVarianceImage(self):
        """Return a variance image with the background model dimensions

        Notes
        -----
        Does not interpolate or extrapolate over bad values.
        """
        variances = self._variances.clone()
        variances /= self._numbers

        return variances
