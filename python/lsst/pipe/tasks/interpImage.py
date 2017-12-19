#
# LSST Data Management System
# Copyright 2008-2015 AURA/LSST.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
from __future__ import absolute_import, division, print_function
import lsst.pex.config as pexConfig
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
import lsst.pipe.base as pipeBase
import lsst.ip.isr as ipIsr

__all__ = ["InterpImageConfig", "InterpImageTask"]


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

    def validate(self):
        pexConfig.Config.validate(self)
        if self.useFallbackValueAtEdge:
            if (not self.negativeFallbackAllowed and self.fallbackValueType == "USER" and
                    self.fallbackUserValue < 0.0):
                raise ValueError("User supplied fallbackValue is negative (%.2f) but "
                                 "negativeFallbackAllowed is False" % self.fallbackUserValue)


class InterpImageTask(pipeBase.Task):
    """Interpolate over bad image pixels
    """
    ConfigClass = InterpImageConfig
    _DefaultName = "interpImage"

    def _setFallbackValue(self, mi=None):
        """Set the edge fallbackValue for interpolation

        \param[in] mi  input maksedImage on which to calculate the statistics
                       Must be provided if fallbackValueType != "USER".

        \return fallbackValue  The value set/computed based on the fallbackValueType
                               and negativeFallbackAllowed config parameters
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
            self.log.warn("Negative interpolation edge fallback value computed but "
                          "negativeFallbackAllowed is False: setting fallbackValue to 0.0")
            fallbackValue = max(fallbackValue, 0.0)

        self.log.info("fallbackValueType %s has been set to %.4f" %
                      (self.config.fallbackValueType, fallbackValue))

        return fallbackValue

    @pipeBase.timeMethod
    def run(self, image, planeName=None, fwhmPixels=None, defects=None):
        """!Interpolate in place over pixels in a maskedImage marked as bad

        Pixels to be interpolated are set by either a mask planeName provided
        by the caller OR a defects list of type measAlg.DefectListT.  If both
        are provided an exception is raised.

        Note that the interpolation code in meas_algorithms currently doesn't
        use the input PSF (though it's a required argument), so it's not
        important to set the input PSF parameters exactly.  This PSF is set
        here as the psf attached to the "image" (i.e if the image passed in
        is an Exposure).  Otherwise, a psf model is created using
        measAlg.GaussianPsfFactory with the value of fwhmPixels (the value
        passed in by the caller, or the default defaultFwhm set in
        measAlg.GaussianPsfFactory if None).

        \param[in,out] image       MaskedImage OR Exposure to be interpolated
        \param[in]     planeName   name of mask plane over which to interpolate
                                   If None, must provide a defects list.
        \param[in]     fwhmPixels  FWHM of core star (pixels)
                                   If None the default is used, where the default
                                   is set to the exposure psf if available
        \param[in]     defects     List of defects of type measAlg.DefectListT
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
                defectList = defects
                planeName = "defects"
        else:
            if defects is not None:
                raise ValueError("Provide EITHER a planeName OR a list of defects, not both")
            if planeName not in maskedImage.getMask().getMaskPlaneDict():
                raise ValueError("maskedImage does not contain mask plane %s" % planeName)
            defectList = ipIsr.getDefectListFromMask(maskedImage, planeName)

        # set psf from exposure if provided OR using modelPsf with fwhmPixels provided
        try:
            psf = image.getPsf()
            self.log.info("Setting psf for interpolation from image")
        except AttributeError:
            self.log.info("Creating psf model for interpolation from fwhm(pixels) = %s" %
                          (str(fwhmPixels) if fwhmPixels is not None else
                           (str(self.config.modelPsf.defaultFwhm)) + " [default]"))
            psf = self.config.modelPsf.apply(fwhm=fwhmPixels)

        fallbackValue = 0.0  # interpolateOverDefects needs this to be a float, regardless if it is used
        if self.config.useFallbackValueAtEdge:
            fallbackValue = self._setFallbackValue(maskedImage)

        measAlg.interpolateOverDefects(maskedImage, psf, defectList, fallbackValue,
                                       self.config.useFallbackValueAtEdge)

        self.log.info("Interpolated over %d %s pixels." % (len(defectList), planeName))
