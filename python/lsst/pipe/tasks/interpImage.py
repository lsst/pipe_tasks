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

__all__ = (
    "CloughTocher2DInterpolateConfig",
    "CloughTocher2DInterpolateTask",
    "InterpImageConfig",
    "InterpImageTask",
)


from contextlib import contextmanager
from itertools import product
from typing import Iterable

import lsst.pex.config as pexConfig
import lsst.geom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.ip.isr as ipIsr
import lsst.meas.algorithms as measAlg
import lsst.pipe.base as pipeBase
from lsst.utils.timer import timeMethod
from scipy.interpolate import CloughTocher2DInterpolator


class InterpImageConfig(pexConfig.Config):
    """Config for InterpImageTask
    """
    modelPsf = measAlg.GaussianPsfFactory.makeField(doc="Model Psf factory")

    useFallbackValueAtEdge = pexConfig.Field(
        dtype=bool,
        doc="Smoothly taper to the fallback value at the edge of the image?",
        default=True,
    )
    fallbackValueType = pexConfig.ChoiceField(
        dtype=str,
        doc="Type of statistic to calculate edge fallbackValue for interpolation",
        allowed={
            "MEAN": "mean",
            "MEDIAN": "median",
            "MEANCLIP": "clipped mean",
            "USER": "user value set in fallbackUserValue config",
        },
        default="MEDIAN",
    )
    fallbackUserValue = pexConfig.Field(
        dtype=float,
        doc="If fallbackValueType is 'USER' then use this as the fallbackValue; ignored otherwise",
        default=0.0,
    )
    negativeFallbackAllowed = pexConfig.Field(
        dtype=bool,
        doc=("Allow negative values for egde interpolation fallbackValue?  If False, set "
             "fallbackValue to max(fallbackValue, 0.0)"),
        default=False,
    )
    transpose = pexConfig.Field(dtype=int, default=False,
                                doc="Transpose image before interpolating? "
                                    "This allows the interpolation to act over columns instead of rows.")

    def validate(self):
        pexConfig.Config.validate(self)
        if self.useFallbackValueAtEdge:
            if (not self.negativeFallbackAllowed and self.fallbackValueType == "USER"
                    and self.fallbackUserValue < 0.0):
                raise ValueError("User supplied fallbackValue is negative (%.2f) but "
                                 "negativeFallbackAllowed is False" % self.fallbackUserValue)


class InterpImageTask(pipeBase.Task):
    """Interpolate over bad image pixels
    """
    ConfigClass = InterpImageConfig
    _DefaultName = "interpImage"

    def _setFallbackValue(self, mi=None):
        """Set the edge fallbackValue for interpolation

        Parameters
        ----------
        mi : `lsst.afw.image.MaskedImage`, optional
            Input maskedImage on which to calculate the statistics
            Must be provided if fallbackValueType != "USER".

        Returns
        -------
        fallbackValue : `float`
            The value set/computed based on the fallbackValueType
            and negativeFallbackAllowed config parameters.
        """
        if self.config.fallbackValueType != 'USER':
            assert mi, "No maskedImage provided"
        if self.config.fallbackValueType == 'MEAN':
            fallbackValue = afwMath.makeStatistics(mi, afwMath.MEAN).getValue()
        elif self.config.fallbackValueType == 'MEDIAN':
            fallbackValue = afwMath.makeStatistics(mi, afwMath.MEDIAN).getValue()
        elif self.config.fallbackValueType == 'MEANCLIP':
            fallbackValue = afwMath.makeStatistics(mi, afwMath.MEANCLIP).getValue()
        elif self.config.fallbackValueType == 'USER':
            fallbackValue = self.config.fallbackUserValue
        else:
            raise NotImplementedError("%s : %s not implemented" %
                                      ("fallbackValueType", self.config.fallbackValueType))

        if not self.config.negativeFallbackAllowed and fallbackValue < 0.0:
            self.log.warning("Negative interpolation edge fallback value computed but "
                             "negativeFallbackAllowed is False: setting fallbackValue to 0.0")
            fallbackValue = max(fallbackValue, 0.0)

        self.log.info("fallbackValueType %s has been set to %.4f",
                      self.config.fallbackValueType, fallbackValue)

        return fallbackValue

    @timeMethod
    def run(self, image, planeName=None, fwhmPixels=None, defects=None):
        """Interpolate in place over pixels in a maskedImage marked as bad

        Pixels to be interpolated are set by either a mask planeName provided
        by the caller OR a defects list of type `~lsst.meas.algorithms.Defects`
        If both are provided an exception is raised.

        Note that the interpolation code in meas_algorithms currently doesn't
        use the input PSF (though it's a required argument), so it's not
        important to set the input PSF parameters exactly.  This PSF is set
        here as the psf attached to the "image" (i.e if the image passed in
        is an Exposure).  Otherwise, a psf model is created using
        measAlg.GaussianPsfFactory with the value of fwhmPixels (the value
        passed in by the caller, or the default defaultFwhm set in
        measAlg.GaussianPsfFactory if None).

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage` or `lsst.afw.image.exposure.Exposure`
            MaskedImage OR Exposure to be interpolated.
        planeName : `str`, optional
            Name of mask plane over which to interpolate.
            If None, must provide a defects list.
        fwhmPixels : `int`, optional
            FWHM of core star (pixels).
            If None the default is used, where the default
            is set to the exposure psf if available.
        defects : `lsst.meas.algorithms.Defects`, optional
            List of defects of type ipIsr.Defects
            over which to interpolate.
        """
        try:
            maskedImage = image.getMaskedImage()
        except AttributeError:
            maskedImage = image

        # set defectList from defects OR mask planeName provided
        if planeName is None:
            if defects is None:
                raise ValueError("No defects or plane name provided")
            else:
                if not isinstance(defects, ipIsr.Defects):
                    defectList = ipIsr.Defects(defects)
                else:
                    defectList = defects
                planeName = "defects"
        else:
            if defects is not None:
                raise ValueError("Provide EITHER a planeName OR a list of defects, not both")
            if planeName not in maskedImage.getMask().getMaskPlaneDict():
                raise ValueError("maskedImage does not contain mask plane %s" % planeName)
            defectList = ipIsr.Defects.fromMask(maskedImage, planeName)

        # set psf from exposure if provided OR using modelPsf with fwhmPixels provided
        try:
            psf = image.getPsf()
            self.log.info("Setting psf for interpolation from image")
        except AttributeError:
            self.log.info("Creating psf model for interpolation from fwhm(pixels) = %s",
                          str(fwhmPixels) if fwhmPixels is not None else
                          (str(self.config.modelPsf.defaultFwhm)) + " [default]")
            psf = self.config.modelPsf.apply(fwhm=fwhmPixels)

        fallbackValue = 0.0  # interpolateOverDefects needs this to be a float, regardless if it is used
        if self.config.useFallbackValueAtEdge:
            fallbackValue = self._setFallbackValue(maskedImage)

        self.interpolateImage(maskedImage, psf, defectList, fallbackValue)

        self.log.info("Interpolated over %d %s pixels.", len(defectList), planeName)

    @contextmanager
    def transposeContext(self, maskedImage, defects):
        """Context manager to potentially transpose an image

        This applies the ``transpose`` configuration setting.

        Transposing the image allows us to interpolate along columns instead
        of rows, which is useful when the saturation trails are typically
        oriented along rows on the warped/coadded images, instead of along
        columns as they typically are in raw CCD images.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image on which to perform interpolation.
        defects : `lsst.meas.algorithms.Defects`
            List of defects to interpolate over.

        Yields
        ------
        useImage : `lsst.afw.image.MaskedImage`
            Image to use for interpolation; it may have been transposed.
        useDefects : `lsst.meas.algorithms.Defects`
            List of defects to use for interpolation; they may have been
            transposed.
        """
        def transposeImage(image):
            """Transpose an image

            Parameters
            ----------
            image : `Unknown`
            """
            transposed = image.array.T.copy()  # Copy to force row-major; required for ndarray+pybind
            return image.Factory(transposed, False, lsst.geom.Point2I(*reversed(image.getXY0())))

        useImage = maskedImage
        useDefects = defects
        if self.config.transpose:
            useImage = afwImage.makeMaskedImage(transposeImage(maskedImage.image),
                                                transposeImage(maskedImage.mask),
                                                transposeImage(maskedImage.variance))
            useDefects = defects.transpose()
        yield useImage, useDefects
        if self.config.transpose:
            maskedImage.image.array = useImage.image.array.T
            maskedImage.mask.array = useImage.mask.array.T
            maskedImage.variance.array = useImage.variance.array.T

    def interpolateImage(self, maskedImage, psf, defectList, fallbackValue):
        """Interpolate over defects in an image

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image on which to perform interpolation.
        psf : `lsst.afw.detection.Psf`
            Point-spread function; currently unused.
        defectList : `lsst.meas.algorithms.Defects`
            List of defects to interpolate over.
        fallbackValue : `float`
            Value to set when interpolation fails.
        """
        if not defectList:
            return
        with self.transposeContext(maskedImage, defectList) as (image, defects):
            measAlg.interpolateOverDefects(image, psf, defects, fallbackValue,
                                           self.config.useFallbackValueAtEdge)


class CloughTocher2DInterpolateConfig(pexConfig.Config):
    """Config for CloughTocher2DInterpolateTask."""

    badMaskPlanes = pexConfig.ListField[str](
        doc="List of mask planes to interpolate over.",
        default=["BAD", "SAT", "CR"],
    )
    fillValue = pexConfig.Field[float](
        doc="Constant value to fill outside of the convex hull of the good "
        "pixels. A long (longer than twice the ``interpLength``) streak of "
        "bad pixels at an edge will be set to this value.",
        default=0.0,
    )
    interpLength = pexConfig.Field[int](
        doc="Maximum number of pixels away from a bad pixel to include in "
        "building the interpolant. Must be greater than or equal to 1.",
        default=4,
        check=lambda x: x >= 1,
    )


class CloughTocher2DInterpolateTask(pipeBase.Task):
    """Interpolated over bad pixels using CloughTocher2DInterpolator.

    Pixels with mask bits set to any of those listed ``badMaskPlanes`` config
    are considered bad and are interpolated over. All good (non-bad) pixels
    within ``interpLength`` pixels of a bad pixel in either direction are used
    to construct the interpolant.  An extended streak of bad pixels at an edge,
    longer than ``interpLength``, is set to `fillValue`` specified in config.
    """

    ConfigClass = CloughTocher2DInterpolateConfig
    _DefaultName = "cloughTocher2DInterpolate"

    def run(self, maskedImage, badpix: set[tuple[int, int]] | None = None, goodpix: dict | None = None):
        """Interpolate over bad pixels in a masked image.

        This modifies the ``image`` attribute of the ``maskedImage`` in place.
        This method returns, and accepts, the coordinates of the bad pixels
        that were interpolated over, and the coordinates and values of the
        good pixels that were used to construct the interpolant. This avoids
        having to search for the bad and the good pixels repeatedly when the
        mask plane is shared among many images, as would be the case with
        noise realizations.

        Parameters
        ----------
        maskedImage : `~lsst.afw.image.MaskedImage`
            Image on which to perform interpolation (and modify in-place).
        badpix: `set` [`tuple` [`int`, `int`]], optional
            The coordinates of the bad pixels to interpolate over.
            If None, then the coordinates of the bad pixels are determined by
            an exhaustive search over the image.
        goodpix: `dict` [`tuple` [`int`, `int`], `float`], optional
            A mapping whose keys are the coordinates of the good pixels around
            ``badpix`` that must be included when constructing the
            interpolant. If ``badpix`` is provided, then the pixels in
            ``goodpix`` are used as to construct the interpolatant. If not,
            any additional good pixels around internally determined ``badpix``
            are added to ``goodpix`` and used to construct the interpolant. In
            all cases, the values are populated from the image plane of the
            ``maskedImage`` (provided values will be ignored.

        Returns
        -------
        badpix: `set` [`tuple` [`int`, `int`]]
            The coordinates of the bad pixels that were interpolated over.
        goodpix: `dict` [`tuple` [`int`, `int`], `float`]
            Mapping of the coordinates of the good pixels around ``badpix``
            to their values that were included when constructing the
            interpolant.

        Raises
        ------
        RuntimeError
            If a pixel passed in as ``goodpix`` is found to be bad as specified by
            ``maskPlanes``.
        ValueError
            If an input ``badpix`` is not found to be bad as specified by
            ``maskPlanes``.
        """
        max_window_extent = lsst.geom.Extent2I(
            2 * self.config.interpLength + 1, 2 * self.config.interpLength + 1
        )
        # Even if badpix and/or goodpix is provided, make sure to update
        # the values of goodpix.
        badpix, goodpix = find_good_pixels_around_bad_pixels(
            maskedImage,
            self.config.badMaskPlanes,
            max_window_extent=max_window_extent,
            badpix=badpix,
            goodpix=goodpix,
        )

        # Construct the interpolant.
        interpolator = CloughTocher2DInterpolator(
            list(goodpix.keys()),
            list(goodpix.values()),
            fill_value=self.config.fillValue,
        )

        # Fill in the bad pixels.
        for x, y in badpix:
            maskedImage.image[x, y] = interpolator((x, y))

        return badpix, goodpix


def find_good_pixels_around_bad_pixels(
    image: afwImage.MaskedImage,
    maskPlanes: Iterable[str],
    *,
    max_window_extent: lsst.geom.Extent2I,
    badpix: set | None = None,
    goodpix: dict | None = None,
):
    """Find the location of bad pixels, and neighboring good pixels.

    Parameters
    ----------
    image : `~lsst.afw.image.MaskedImage`
        Image from which to find the bad and the good pixels.
    maskPlanes : `list` [`str`]
        List of mask planes to consider as bad pixels.
    max_window_extent : `lsst.geom.Extent2I`
        Maximum extent of the window around a bad pixel to consider when
        looking for good pixels.
    badpix : `list` [`tuple` [`int`, `int`]], optional
        A known list of bad pixels. If provided, the function does not look for
        any additional bad pixels, but it verifies that the provided
        coordinates correspond to bad pixels. If an input``badpix`` is not
        found to be bad as specified by ``maskPlanes``, an exception is raised.
    goodpix : `dict` [`tuple` [`int`, `int`], `float`], optional
        A known mapping of the coordinates of good pixels to their values, to
        which any newly found good pixels locations will be added, and the
        values (even for existing items) will be updated.

    Returns
    -------
    badpix : `list` [`tuple` [`int`, `int`]]
        The coordinates of the bad pixels. If ``badpix`` was provided as an
        input argument, the returned quantity is the same as the input.
    goodpix : `dict` [`tuple` [`int`, `int`], `float`]
        Updated mapping of the coordinates of good pixels to their values.

    Raises
    ------
    RuntimeError
        If a pixel passed in as ``goodpix`` is found to be bad as specified by
        ``maskPlanes``.
    ValueError
        If an input ``badpix`` is not found to be bad as specified by
        ``maskPlanes``.
    """

    bbox = image.getBBox()
    if badpix is None:
        iterator = product(range(bbox.minX, bbox.maxX + 1), range(bbox.minY, bbox.maxY + 1))
        badpix = set()
    else:
        iterator = badpix

    if goodpix is None:
        goodpix = {}

    for x, y in iterator:
        if image.mask[x, y] & afwImage.Mask.getPlaneBitMask(maskPlanes):
            if (x, y) in goodpix:
                raise RuntimeError(f"Pixel ({x}, {y}) is bad as specified by maskPlanes {maskPlanes} but "
                                   "passed in as goodpix")
            badpix.add((x, y))
            window = lsst.geom.Box2I.makeCenteredBox(
                center=lsst.geom.Point2D(x, y),  # center has to be a Point2D instance.
                size=max_window_extent,
            )
            # Restrict to the bounding box of the image.
            window.clip(bbox)

            for xx, yy in product(range(window.minX, window.maxX + 1), range(window.minY, window.maxY + 1)):
                if not (image.mask[xx, yy] & afwImage.Mask.getPlaneBitMask(maskPlanes)):
                    goodpix[(xx, yy)] = image.image[xx, yy]
        elif (x, y) in badpix:
            # If (x, y) is in badpix, but did not get flagged as bad,
            # raise an exception.
            raise ValueError(f"Pixel ({x}, {y}) is not bad as specified by maskPlanes {maskPlanes}")

    return badpix, goodpix
