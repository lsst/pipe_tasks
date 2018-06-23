# This file is part of pipe_tasks.
#
# LSST Data Management System
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
# See COPYRIGHT file at the top of the source tree.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#

import numpy as np
from lsst.afw.coord.refraction import differentialRefraction
import lsst.afw.geom as afwGeom
from lsst.afw.geom import AffineTransform
from lsst.afw.geom import makeTransform
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
from lsst.geom import radians

__all__ = ["DcrModel", "applyDcr", "calculateDcr", "calculateRotationAngle"]


class DcrModel(object):
    """A model of the true scene after correcting chromatic effects.

    Attributes
    ----------
    dcrNumSubfilters : `int`
        Number of sub-filters used to model chromatic effects within a band.
    filterInfo : `lsst.afw.image.Filter`
        The filter definition, set in the current instruments' obs package.
    modelImages : `list` of `lsst.afw.image.MaskedImageF`
        A list of masked images, each containing the model for one subfilter

    Notes
    -----
    The ``DcrModel`` contains an estimate of the true scene, at a higher
    wavelength resolution than the input observations. It can be forward-
    modeled to produce Differential Chromatic Refraction (DCR) matched
    templates for a given ``Exposure``, and provides utilities for conditioning
    the model in ``dcrAssembleCoadd`` to avoid oscillating solutions between
    iterations of forward modeling or between the subfilters of the model.
    """

    def __init__(self, dcrNumSubfilters, filterInfo, coaddExposure=None, modelImages=None):
        """Divide a coadd into equal subfilter coadds.

        Parameters
        ----------
        dcrNumSubfilters : `int`
            Number of sub-filters used to model chromatic effects within a band.
        filterInfo : `lsst.afw.image.Filter`
            The filter definition, set in the current instruments' obs package.
        coaddExposure : `lsst.afw.image.ExposureF`, optional
            The target image for the coadd.
        modelImages : `list` of `lsst.afw.image.MaskedImageF`, optional
            A list of masked images, each containing the model for one subfilter.

        Raises
        ------
        ValueError
            If neither ``modelImages`` or ``coaddExposure`` are set.
            If ``modelImages`` is supplied but does not match ``dcrNumSubfilters``.
        """
        self.dcrNumSubfilters = dcrNumSubfilters
        self.filterInfo = filterInfo
        if modelImages is not None:
            if len(modelImages) != dcrNumSubfilters:
                raise ValueError("The dimension of modelImages must equal"
                                 " the supplied dcrNumSubfilters.")
            self.modelImages = modelImages
        elif coaddExposure is not None:
            maskedImage = coaddExposure.maskedImage.clone()
            # NANs will potentially contaminate the entire image,
            #  depending on the shift or convolution type used.
            badPixels = np.isnan(maskedImage.image.array) | np.isnan(maskedImage.variance.array)
            maskedImage.image.array[badPixels] = 0.
            maskedImage.variance.array[badPixels] = 0.
            maskedImage.image.array /= dcrNumSubfilters
            maskedImage.variance.array /= dcrNumSubfilters
            maskedImage.mask.array[badPixels] = maskedImage.mask.getPlaneBitMask("NO_DATA")
            self.modelImages = [maskedImage, ]
            for subfilter in range(1, dcrNumSubfilters):
                self.modelImages.append(maskedImage.clone())
        else:
            raise ValueError("Either dcrModels or coaddExposure must be set.")

    def getImage(self, subfilter, bbox=None):
        """Return the image of the DCR model for a given subfilter.

        Parameters
        ----------
        subfilter : `int`
            Index of the current subfilter within the full band.
        bbox : `lsst.afw.geom.box.Box2I`, optional
            Sub-region of the coadd. Returns the entire image if None.

        Returns
        -------
        `lsst.afw.image.MaskedImageF`
            The DCR model for the given ``subfilter``.
        """
        return self.modelImages[subfilter] if bbox is None \
            else self.modelImages[subfilter][bbox, afwImage.PARENT]

    def getReferenceImage(self, bbox=None):
        """Create a simple template from the DCR model.

        Parameters
        ----------
        bbox : `lsst.afw.geom.box.Box2I`, optional
            Sub-region of the coadd. Returns the entire image if None.

        Returns
        -------
        `numpy.ndarray`
            The template with no chromatic effects applied.
        """
        return np.mean([self.getImage(subfilter, bbox).image.array
                        for subfilter in range(self.dcrNumSubfilters)], axis=0)

    def assign(self, dcrSubModel, bbox):
        """Update a sub-region of the ``DcrModel`` with new values.

        Parameters
        ----------
        dcrSubModel : `lsst.pipe.tasks.DcrModel`
            New model of the true scene after correcting chromatic effects.
        bbox : `lsst.afw.geom.box.Box2I`
            Sub-region of the coadd
        """
        for model, subModel in zip(self.modelImages, dcrSubModel.modelImages):
            model.assign(subModel[bbox, afwImage.PARENT], bbox)

    def buildMatchedTemplate(self, warpCtrl, exposure=None, visitInfo=None, bbox=None, wcs=None, mask=None):
        """Create a DCR-matched template for an exposure.

        Parameters
        ----------
        warpCtrl : `lsst.afw.Math.WarpingControl`
            Configuration settings for warping an image
        exposure : `lsst.afw.image.ExposureF`, optional
            The input exposure to build a matched template for.
            May be omitted if all of the metadata is supplied separately
        visitInfo : `lsst.afw.image.VisitInfo`, optional
            Metadata for the exposure. Ignored if ``exposure`` is set.
        bbox : `lsst.afw.geom.box.Box2I`, optional
            Sub-region of the coadd. Ignored if ``exposure`` is set.
        wcs : `lsst.afw.geom.SkyWcs`, optional
            Coordinate system definition (wcs) for the exposure.
            Ignored if ``exposure`` is set.
        mask : `lsst.afw.image.Mask`, optional
            reference mask to use for the template image.
            Ignored if ``exposure`` is set.

        Returns
        -------
        `lsst.afw.image.maskedImageF`
            The DCR-matched template

        Raises
        ------
        ValueError
            If neither ``exposure`` or all of ``visitInfo``, ``bbox``, and ``wcs`` are set.
        """
        if exposure is not None:
            visitInfo = exposure.getInfo().getVisitInfo()
            bbox = exposure.getBBox()
            wcs = exposure.getInfo().getWcs()
        elif visitInfo is None or bbox is None or wcs is None:
            raise ValueError("Either exposure or visitInfo, bbox, and wcs must be set.")
        dcrShift = calculateDcr(visitInfo, wcs, self.filterInfo, self.dcrNumSubfilters)
        templateImage = afwImage.MaskedImageF(bbox)
        for subfilter, dcr in enumerate(dcrShift):
            templateImage += applyDcr(self.getImage(subfilter, bbox), dcr, warpCtrl)
        if mask is not None:
            templateImage.setMask(mask[bbox, afwImage.PARENT])
        return templateImage

    def conditionDcrModel(self, subfilter, newModel, bbox, gain=1.):
        """Average two iterations' solutions to reduce oscillations.

        Parameters
        ----------
        subfilter : `int`
            Index of the current subfilter within the full band.
        newModel : `lsst.afw.image.MaskedImageF`
            The new DCR model for one subfilter from the current iteration.
            Values in ``newModel`` that are extreme compared with the last
            iteration are modified in place.
        bbox : `lsst.afw.geom.box.Box2I`
            Sub-region of the coadd
        gain : `float`, optional
            Additional weight to apply to the model from the current iteration.
            Defaults to 1.0, which gives equal weight to both solutions.
        """
        # The models are MaskedImages, which only support in-place operations.
        newModel *= gain
        newModel += self.getImage(subfilter, bbox)
        newModel.image.array /= 1. + gain
        newModel.variance.array /= 1. + gain

    def clampModel(self, subfilter, newModel, bbox, statsCtrl, regularizeSigma, modelClampFactor,
                   convergenceMaskPlanes=None):
        """Restrict large variations in the model between iterations.

        Parameters
        ----------
        subfilter : `int`
            Index of the current subfilter within the full band.
        newModel : `lsst.afw.image.MaskedImageF`
            The new DCR model for one subfilter from the current iteration.
            Values in ``newModel`` that are extreme compared with the last
            iteration are modified in place.
        bbox : `lsst.afw.geom.box.Box2I`
            Sub-region to coadd
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd
        regularizeSigma : `float`
            Threshold to exclude noise-like pixels from regularization.
        modelClampFactor : `float`
            Maximum relative change of the model allowed between iterations.
        convergenceMaskPlanes : `list` of `str`, or `str`, optional
            Mask planes to use to calculate convergence.
            Default value is set in ``calculateNoiseCutoff`` if not supplied.
        """
        newImage = newModel.image.array
        oldImage = self.getImage(subfilter, bbox).image.array
        noiseCutoff = self.calculateNoiseCutoff(newModel, statsCtrl, regularizeSigma,
                                                convergenceMaskPlanes=convergenceMaskPlanes)
        # Catch any invalid values
        nanPixels = np.isnan(newImage)
        newImage[nanPixels] = 0.
        infPixels = np.isinf(newImage)
        newImage[infPixels] = oldImage[infPixels]*modelClampFactor
        # Clip pixels that have very high amplitude, compared with the previous iteration.
        clampPixels = np.abs(newImage - oldImage) > (np.abs(oldImage*(modelClampFactor - 1)) +
                                                     noiseCutoff)
        # Set high amplitude pixels to a multiple or fraction of the old model value,
        #  depending on whether the new model is higher or lower than the old
        highPixels = newImage > oldImage
        newImage[clampPixels & highPixels] = oldImage[clampPixels & highPixels]*modelClampFactor
        newImage[clampPixels & ~highPixels] = oldImage[clampPixels & ~highPixels]/modelClampFactor

    def regularizeModel(self, bbox, mask, statsCtrl, regularizeSigma, clampFrequency,
                        convergenceMaskPlanes=None):
        """Restrict large variations in the model between subfilters.

        Any flux subtracted by the restriction is accumulated from all
        subfilters, and divided evenly to each afterwards in order to preserve
        total flux.

        Parameters
        ----------
        bbox : `lsst.afw.geom.box.Box2I`
            Sub-region to coadd
        mask : `lsst.afw.image.Mask`
            Reference mask to use for all model planes.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd
        regularizeSigma : `float`
            Threshold to exclude noise-like pixels from regularization.
        clampFrequency : `float`
            Maximum relative change of the model allowed between subfilters.
        convergenceMaskPlanes : `list` of `str`, or `str`, optional
            Mask planes to use to calculate convergence.
            Default value is set in ``calculateNoiseCutoff`` if not supplied.
        """
        templateImage = self.getReferenceImage(bbox)
        excess = np.zeros_like(templateImage)
        for model in self.modelImages:
            noiseCutoff = self.calculateNoiseCutoff(model, statsCtrl, regularizeSigma,
                                                    mask=mask[bbox, afwImage.PARENT],
                                                    convergenceMaskPlanes=convergenceMaskPlanes)
            modelVals = model.image.array
            highPixels = (modelVals > (templateImage*clampFrequency + noiseCutoff))
            excess[highPixels] += modelVals[highPixels] - templateImage[highPixels]*clampFrequency
            modelVals[highPixels] = templateImage[highPixels]*clampFrequency
            lowPixels = (modelVals < templateImage/clampFrequency - noiseCutoff)
            excess[lowPixels] += modelVals[lowPixels] - templateImage[lowPixels]/clampFrequency
            modelVals[lowPixels] = templateImage[lowPixels]/clampFrequency
        excess /= self.dcrNumSubfilters
        for model in self.modelImages:
            model.image.array += excess

    def calculateNoiseCutoff(self, maskedImage, statsCtrl, regularizeSigma, mask=None,
                             convergenceMaskPlanes=None):
        """Helper function to calculate the background noise level of an image.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.maskedImageF`
            The input image to evaluate the background noise properties.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd
        regularizeSigma : `float`
            Threshold to exclude noise-like pixels from regularization.
        mask : `lsst.afw.image.Mask`, Optional
            Optional alternate mask
        convergenceMaskPlanes : `list` of `str`, or `str`, optional
            Mask planes to use to calculate convergence.
            The "DETECTED" mask plane is used by default if not supplied.

        Returns
        -------
        float
            The threshold value to treat pixels as noise in an image..
        """
        if convergenceMaskPlanes is None:
            convergeMask = maskedImage.mask.getPlaneBitMask("DETECTED")
        else:
            convergeMask = maskedImage.mask.getPlaneBitMask(convergenceMaskPlanes)
        if mask is None:
            backgroundPixels = maskedImage.mask.array & (statsCtrl.getAndMask() | convergeMask) == 0
        else:
            backgroundPixels = mask.array & (statsCtrl.getAndMask() | convergeMask) == 0
        noiseCutoff = regularizeSigma*np.std(maskedImage.image.array[backgroundPixels])
        return noiseCutoff


def applyDcr(maskedImage, dcr, warpCtrl, bbox=None, useInverse=False):
    """Shift a masked image.

    Parameters
    ----------
    maskedImage : `lsst.afw.image.maskedImageF`
        The input masked image to shift.
    dcr : `lsst.afw.geom.Extent2I`
        Shift calculated with ``calculateDcr``.
    warpCtrl : `lsst.afw.math.WarpingControl`
        Configuration settings for warping an image
    bbox : `lsst.afw.geom.box.Box2I`, optional
        Sub-region of the masked image to shift.
        Shifts the entire image if None (Default).
    useInverse : `bool`, optional
        Use the reverse of ``dcr`` for the shift. Default: False

    Returns
    -------
    `lsst.afw.image.maskedImageF`
        A masked image, with the pixels within the bounding box shifted.

    Deleted Parameters
    ------------------
    useFFT : `bool`, optional
        Perform the convolution with an FFT?
    """
    padValue = afwImage.pixel.SinglePixelF(0., maskedImage.mask.getPlaneBitMask("NO_DATA"), 0)
    if bbox is None:
        bbox = maskedImage.getBBox()
    shiftedImage = afwImage.MaskedImageF(bbox)
    transform = makeTransform(AffineTransform((-1.0 if useInverse else 1.0)*dcr))
    afwMath.warpImage(shiftedImage, maskedImage[bbox, afwImage.PARENT],
                      transform, warpCtrl, padValue=padValue)
    return shiftedImage


def calculateDcr(visitInfo, wcs, filterInfo, dcrNumSubfilters):
    """Calculate the shift in pixels of an exposure due to DCR.

    Parameters
    ----------
    visitInfo : `lsst.afw.image.VisitInfo`
        Metadata for the exposure.
    wcs : `lsst.afw.geom.SkyWcs`
        Coordinate system definition (wcs) for the exposure.
    filterInfo : `lsst.afw.image.Filter`
        The filter definition, set in the current instruments' obs package.
    dcrNumSubfilters : `int`
        Number of sub-filters used to model chromatic effects within a band.

    Returns
    -------
    `lsst.afw.geom.Extent2I`
        The 2D shift due to DCR, in pixels.
    """
    rotation = calculateRotationAngle(visitInfo, wcs)
    dcrShift = []
    lambdaEff = filterInfo.getFilterProperty().getLambdaEff()
    for wl0, wl1 in wavelengthGenerator(filterInfo, dcrNumSubfilters):
        # Note that diffRefractAmp can be negative, since it's relative to the midpoint of the full band
        diffRefractAmp0 = differentialRefraction(wl0, lambdaEff,
                                                 elevation=visitInfo.getBoresightAzAlt().getLatitude(),
                                                 observatory=visitInfo.getObservatory(),
                                                 weather=visitInfo.getWeather())
        diffRefractAmp1 = differentialRefraction(wl1, lambdaEff,
                                                 elevation=visitInfo.getBoresightAzAlt().getLatitude(),
                                                 observatory=visitInfo.getObservatory(),
                                                 weather=visitInfo.getWeather())
        diffRefractAmp = (diffRefractAmp0 + diffRefractAmp1)/2.
        diffRefractPix = diffRefractAmp.asArcseconds()/wcs.getPixelScale().asArcseconds()
        dcrShift.append(afwGeom.Extent2D(diffRefractPix*np.cos(rotation.asRadians()),
                                         diffRefractPix*np.sin(rotation.asRadians())))
    return dcrShift


def calculateRotationAngle(visitInfo, wcs):
    """Calculate the sky rotation angle of an exposure.

    Parameters
    ----------
    visitInfo : `lsst.afw.image.VisitInfo`
        Metadata for the exposure.
    wcs : `lsst.afw.geom.SkyWcs`
        Coordinate system definition (wcs) for the exposure.

    Returns
    -------
    `lsst.geom.Angle`
        The rotation of the image axis, East from North.
        Equal to the parallactic angle plus any additional rotation of the
        coordinate system.
        A rotation angle of 0 degrees is defined with
        North along the +y axis and East along the +x axis.
        A rotation angle of 90 degrees is defined with
        North along the +x axis and East along the -y axis.
    """
    parAngle = visitInfo.getBoresightParAngle().asRadians()
    cd = wcs.getCdMatrix()
    cdAngle = (np.arctan2(-cd[0, 1], cd[0, 0]) + np.arctan2(cd[1, 0], cd[1, 1]))/2.
    rotAngle = (cdAngle + parAngle)*radians
    return rotAngle


def wavelengthGenerator(filterInfo, dcrNumSubfilters):
    """Iterate over the wavelength endpoints of subfilters.

    Parameters
    ----------
    filterInfo : `lsst.afw.image.Filter`
        The filter definition, set in the current instruments' obs package.
    dcrNumSubfilters : `int`
        Number of sub-filters used to model chromatic effects within a band.

    Yields
    ------
    `tuple` of two `float`
        The next set of wavelength endpoints for a subfilter, in nm.
    """
    lambdaMin = filterInfo.getFilterProperty().getLambdaMin()
    lambdaMax = filterInfo.getFilterProperty().getLambdaMax()
    wlStep = (lambdaMax - lambdaMin)/dcrNumSubfilters
    for wl in np.linspace(lambdaMin, lambdaMax, dcrNumSubfilters, endpoint=False):
        yield (wl, wl + wlStep)
