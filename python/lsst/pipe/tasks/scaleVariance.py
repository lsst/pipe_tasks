#
# LSST Data Management System
# Copyright 2018 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

from __future__ import absolute_import, division, print_function

from contextlib import contextmanager
import numpy as np

from lsst.pex.config import Config, Field, ListField, ConfigurableField
from lsst.pipe.base import Task
from lsst.meas.algorithms import SubtractBackgroundTask



class ScaleVarianceConfig(Config):
    background = ConfigurableField(target=SubtractBackgroundTask, doc="Background subtraction")
    maskPlanes = ListField(
        dtype=str,
        default=["DETECTED", "DETECTED_NEGATIVE", "BAD", "SAT", "NO_DATA", "INTRP"],
        doc="Mask planes for pixels to ignore when scaling variance",
    )
    limit = Field(dtype=float, default=10.0, doc="Maximum variance scaling value to permit")

    def setDefaults(self):
        self.background.binSize = 32
        self.background.useApprox = False
        self.background.undersampleStyle = "REDUCE_INTERP_ORDER"
        self.background.ignoredPixelMask = ["DETECTED", "DETECTED_NEGATIVE", "BAD", "SAT", "NO_DATA", "INTRP"]


class ScaleVarianceTask(Task):
    """Scale the variance in a MaskedImage

    The variance plane in a convolved or warped image (or a coadd derived
    from warped images) does not accurately reflect the noise properties of
    the image because variance has been lost to covariance. This Task
    attempts to correct for this by scaling the variance plane to match
    the observed variance in the image. This is not perfect (because we're
    not tracking the covariance) but it's simple and is often good enough.
    """
    ConfigClass = ScaleVarianceConfig
    _DefaultName = "scaleVariance"

    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)
        self.makeSubtask("background")

    @contextmanager
    def subtractedBackground(self, maskedImage):
        """Context manager for subtracting the background

        We need to subtract the background so that the entire image
        (apart from objects, which should be clipped) will have the
        image/sqrt(variance) distributed about zero.

        This context manager subtracts the background, and ensures it
        is restored on exit.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image+mask+variance to have background subtracted and restored.

        Returns
        -------
        context : context manager
            Context manager that ensure the background is restored.
        """
        bg = self.background.fitBackground(maskedImage)
        bgImage = bg.getImageF()
        maskedImage -= bgImage
        try:
            yield
        finally:
            maskedImage += bgImage

    def run(self, maskedImage):
        """Rescale the variance in a maskedImage

        Parameters
        ----------
        maskedImage :  `lsst.afw.image.MaskedImage`
            Image for which to determine the variance rescaling factor.

        Returns
        -------
        factor : `float`
            Variance rescaling factor.

        Raises
        ------
        RuntimeError
            If the estimated variance rescaling factor exceeds the
            configured limit.
        """
        with self.subtractedBackground(maskedImage):
            factor = self.pixelBased(maskedImage)
            if np.isnan(factor) or factor > self.config.limit:
                self.log.warn("Pixel-based variance rescaling factor (%f) exceeds configured limit (%f); "
                              "trying image-based method", factor, self.config.limit)
                factor = self.imageBased(maskedImage)
                if np.isnan(factor) or factor > self.config.limit:
                    raise RuntimeError("Variance rescaling factor (%f) exceeds configured limit (%f)" %
                                       (factor, self.config.limit))
            self.log.info("Renormalizing variance by %f" % (factor,))
            maskedImage.variance *= factor
        return factor

    def pixelBased(self, maskedImage):
        """Determine the variance rescaling factor from pixel statistics

        We calculate SNR = image/sqrt(variance), and the distribution
        for most of the background-subtracted image should have a standard
        deviation of unity. The variance rescaling factor is the factor that
        brings that distribution to have unit standard deviation.

        This may not work well if the image has a lot of structure in it, as
        the assumptions are violated. In that case, use an alternate
        method.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image for which to determine the variance rescaling factor.

        Returns
        -------
        factor : `float`
            Variance rescaling factor.
        """
        variance = maskedImage.variance
        snr = maskedImage.image.array/np.sqrt(variance.array)
        maskVal = maskedImage.mask.getPlaneBitMask(self.config.maskPlanes)
        isGood = ((maskedImage.mask.array & maskVal) == 0) & (maskedImage.variance.array > 0)
        # Robust measurement of stdev using inter-quartile range
        q1, q3 = np.percentile(snr[isGood], (25, 75))
        stdev = 0.74*(q3 - q1)
        return stdev**2

    def imageBased(self, maskedImage):
        """Determine the variance rescaling factor from image statistics

        We calculate average(SNR) = stdev(image)/median(variance), and
        the value should be unity. The variance rescaling factor is the
        factor that brings this value to unity.

        This may not work well if the pixels from which we measure the
        standard deviation of the image are not effectively the same pixels
        from which we measure the median of the variance. In that case, use
        an alternate method.

        Parameters
        ----------
        maskedImage :  `lsst.afw.image.MaskedImage`
            Image for which to determine the variance rescaling factor.

        Returns
        -------
        factor : `float`
            Variance rescaling factor.
        """
        isGood = ((maskedImage.mask.array & maskVal) == 0) & (maskedImage.variance.array > 0)
        # Robust measurement of stdev
        q1, q3 = np.percentile(maskedImage.image.array[isGood], (25, 75))
        ratio = 0.74*(q3 - q1)/np.sqrt(np.median(maskedImage.variance.array[isGood]))
        return ratio**2
