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

__all__ = ["DcrModel", "applyDcr", "calculateDcr", "calculateImageParallacticAngle"]


class DcrModel:
    """A model of the true sky after correcting chromatic effects.

    Attributes
    ----------
    dcrNumSubfilters : `int`
        Number of sub-filters used to model chromatic effects within a band.
    filterInfo : `lsst.afw.image.Filter`
        The filter definition, set in the current instruments' obs package.
    modelImages : `list` of `lsst.afw.image.MaskedImage`
        A list of masked images, each containing the model for one subfilter

    Parameters
    ----------
    modelImages : `list` of `lsst.afw.image.MaskedImage`
        A list of masked images, each containing the model for one subfilter.
    filterInfo : `lsst.afw.image.Filter`, optional
        The filter definition, set in the current instruments' obs package.
        Required for any calculation of DCR, including making matched templates.

    Notes
    -----
    The ``DcrModel`` contains an estimate of the true sky, at a higher
    wavelength resolution than the input observations. It can be forward-
    modeled to produce Differential Chromatic Refraction (DCR) matched
    templates for a given ``Exposure``, and provides utilities for conditioning
    the model in ``dcrAssembleCoadd`` to avoid oscillating solutions between
    iterations of forward modeling or between the subfilters of the model.
    """

    def __init__(self, modelImages, filterInfo=None, psf=None):
        self.dcrNumSubfilters = len(modelImages)
        self.modelImages = modelImages
        self._filter = filterInfo
        self._psf = psf

    @classmethod
    def fromImage(cls, maskedImage, dcrNumSubfilters, filterInfo=None, psf=None):
        """Initialize a DcrModel by dividing a coadd between the subfilters.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Input coadded image to divide equally between the subfilters.
        dcrNumSubfilters : `int`
            Number of sub-filters used to model chromatic effects within a band.
        filterInfo : `lsst.afw.image.Filter`, optional
            The filter definition, set in the current instruments' obs package.
            Required for any calculation of DCR, including making matched templates.
        psf : `lsst.afw.detection.Psf`, optional
            Point spread function (PSF) of the model.
            Required if the ``DcrModel`` will be persisted.

        Returns
        -------
        dcrModel : `lsst.pipe.tasks.DcrModel`
            Best fit model of the true sky after correcting chromatic effects.
        """
        # NANs will potentially contaminate the entire image,
        # depending on the shift or convolution type used.
        model = maskedImage.clone()
        badPixels = np.isnan(model.image.array) | np.isnan(model.variance.array)
        model.image.array[badPixels] = 0.
        model.variance.array[badPixels] = 0.
        model.image.array /= dcrNumSubfilters
        model.variance.array /= dcrNumSubfilters
        model.mask.array[badPixels] = model.mask.getPlaneBitMask("NO_DATA")
        modelImages = [model, ]
        for subfilter in range(1, dcrNumSubfilters):
            modelImages.append(model.clone())
        return cls(modelImages, filterInfo, psf)

    @classmethod
    def fromDataRef(cls, dataRef, datasetType="dcrCoadd", numSubfilters=None, **kwargs):
        """Load an existing DcrModel from a repository.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference defining the patch for coaddition and the
            reference Warp
        datasetType : `str`, optional
            Name of the DcrModel in the registry {"dcrCoadd", "dcrCoadd_sub"}
        numSubfilters : `int`
            Number of sub-filters used to model chromatic effects within a band.
        **kwargs
            Additional keyword arguments to pass to look up the model in the data registry.
            Common keywords and their types include: ``tract``:`str`, ``patch``:`str`,
            ``bbox``:`lsst.afw.geom.Box2I`

        Returns
        -------
        dcrModel : `lsst.pipe.tasks.DcrModel`
            Best fit model of the true sky after correcting chromatic effects.
        """
        modelImages = []
        filterInfo = None
        psf = None
        for subfilter in range(numSubfilters):
            dcrCoadd = dataRef.get(datasetType, subfilter=subfilter,
                                   numSubfilters=numSubfilters, **kwargs)
            if filterInfo is None:
                filterInfo = dcrCoadd.getFilter()
            if psf is None:
                psf = dcrCoadd.getPsf()
            modelImages.append(dcrCoadd.maskedImage)
        return cls(modelImages, filterInfo, psf)

    def __len__(self):
        """Return the number of subfilters.

        Returns
        -------
        dcrNumSubfilters : `int`
            The number of DCR subfilters in the model.
        """
        return self.dcrNumSubfilters

    def __getitem__(self, subfilter):
        """Iterate over the subfilters of the DCR model.

        Parameters
        ----------
        subfilter : `int`
            Index of the current ``subfilter`` within the full band.
            Negative indices are allowed, and count in reverse order
            from the highest ``subfilter``.

        Returns
        -------
        modelImage : `lsst.afw.image.MaskedImage`
            The DCR model for the given ``subfilter``.

        Raises
        ------
        IndexError
            If the requested ``subfilter`` is greater or equal to the number
            of subfilters in the model.
        """
        if np.abs(subfilter) >= len(self):
            raise IndexError("subfilter out of bounds.")
        return self.modelImages[subfilter]

    def __setitem__(self, subfilter, maskedImage):
        """Update the model image for one subfilter.

        Parameters
        ----------
        subfilter : `int`
            Index of the current subfilter within the full band.
        maskedImage : `lsst.afw.image.MaskedImage`
            The DCR model to set for the given ``subfilter``.

        Raises
        ------
        IndexError
            If the requested ``subfilter`` is greater or equal to the number
            of subfilters in the model.
        ValueError
            If the bounding box of the new image does not match.
        """
        if np.abs(subfilter) >= len(self):
            raise IndexError("subfilter out of bounds.")
        if maskedImage.getBBox() != self.getBBox():
            raise ValueError("The bounding box of a subfilter must not change.")
        self.modelImages[subfilter] = maskedImage

    @property
    def filter(self):
        """Return the filter of the model.

        Returns
        -------
        filter : `lsst.afw.image.Filter`
            The filter definition, set in the current instruments' obs package.
        """
        return self._filter

    @property
    def psf(self):
        """Return the psf of the model.

        Returns
        -------
        psf : `lsst.afw.detection.Psf`
            Point spread function (PSF) of the model.
        """
        return self._psf

    @property
    def bbox(self):
        """Return the common bounding box of each subfilter image.

        Returns
        -------
        bbox : `lsst.afw.geom.Box2I`
            Bounding box of the DCR model.
        """
        return self[0].getBBox()

    def getReferenceImage(self, bbox=None):
        """Create a simple template from the DCR model.

        Parameters
        ----------
        bbox : `lsst.afw.geom.Box2I`, optional
            Sub-region of the coadd. Returns the entire image if `None`.

        Returns
        -------
        templateImage : `numpy.ndarray`
            The template with no chromatic effects applied.
        """
        return np.mean([model[bbox].image.array for model in self], axis=0)

    def assign(self, dcrSubModel, bbox=None):
        """Update a sub-region of the ``DcrModel`` with new values.

        Parameters
        ----------
        dcrSubModel : `lsst.pipe.tasks.DcrModel`
            New model of the true scene after correcting chromatic effects.
        bbox : `lsst.afw.geom.Box2I`, optional
            Sub-region of the coadd.
            Defaults to the bounding box of ``dcrSubModel``.

        Raises
        ------
        ValueError
            If the new model has a different number of subfilters.
        """
        if len(dcrSubModel) != len(self):
            raise ValueError("The number of DCR subfilters must be the same "
                             "between the old and new models.")
        if bbox is None:
            bbox = self.getBBox()
        for model, subModel in zip(self, dcrSubModel):
            model.assign(subModel[bbox], bbox)

    def buildMatchedTemplate(self, exposure=None, warpCtrl=None,
                             visitInfo=None, bbox=None, wcs=None, mask=None):
        """Create a DCR-matched template image for an exposure.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`, optional
            The input exposure to build a matched template for.
            May be omitted if all of the metadata is supplied separately
        warpCtrl : `lsst.afw.Math.WarpingControl`, optional
            Configuration settings for warping an image.
            If not set, defaults to a lanczos3 warping kernel for the image,
            and a bilinear kernel for the mask
        visitInfo : `lsst.afw.image.VisitInfo`, optional
            Metadata for the exposure. Ignored if ``exposure`` is set.
        bbox : `lsst.afw.geom.Box2I`, optional
            Sub-region of the coadd. Ignored if ``exposure`` is set.
        wcs : `lsst.afw.geom.SkyWcs`, optional
            Coordinate system definition (wcs) for the exposure.
            Ignored if ``exposure`` is set.
        mask : `lsst.afw.image.Mask`, optional
            reference mask to use for the template image.

        Returns
        -------
        templateImage : `lsst.afw.image.maskedImageF`
            The DCR-matched template

        Raises
        ------
        ValueError
            If neither ``exposure`` or all of ``visitInfo``, ``bbox``, and ``wcs`` are set.
        """
        if self.filter is None:
            raise ValueError("'filterInfo' must be set for the DcrModel in order to calculate DCR.")
        if exposure is not None:
            visitInfo = exposure.getInfo().getVisitInfo()
            bbox = exposure.getBBox()
            wcs = exposure.getInfo().getWcs()
        elif visitInfo is None or bbox is None or wcs is None:
            raise ValueError("Either exposure or visitInfo, bbox, and wcs must be set.")
        if warpCtrl is None:
            # Turn off the warping cache, since we set the linear interpolation length to the entire subregion
            # This warper is only used for applying DCR shifts, which are assumed to be uniform across a patch
            warpCtrl = afwMath.WarpingControl("lanczos3", "bilinear",
                                              cacheSize=0, interpLength=max(bbox.getDimensions()))

        dcrShift = calculateDcr(visitInfo, wcs, self.filter, len(self))
        templateImage = afwImage.MaskedImageF(bbox)
        for subfilter, dcr in enumerate(dcrShift):
            templateImage += applyDcr(self[subfilter][bbox], dcr, warpCtrl)
        if mask is not None:
            templateImage.setMask(mask[bbox])
        return templateImage

    def buildMatchedExposure(self, exposure=None, warpCtrl=None,
                             visitInfo=None, bbox=None, wcs=None, mask=None):
        """Wrapper to create an exposure from a template image.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`, optional
            The input exposure to build a matched template for.
            May be omitted if all of the metadata is supplied separately
        warpCtrl : `lsst.afw.Math.WarpingControl`
            Configuration settings for warping an image
        visitInfo : `lsst.afw.image.VisitInfo`, optional
            Metadata for the exposure. Ignored if ``exposure`` is set.
        bbox : `lsst.afw.geom.Box2I`, optional
            Sub-region of the coadd. Ignored if ``exposure`` is set.
        wcs : `lsst.afw.geom.SkyWcs`, optional
            Coordinate system definition (wcs) for the exposure.
            Ignored if ``exposure`` is set.
        mask : `lsst.afw.image.Mask`, optional
            reference mask to use for the template image.

        Returns
        -------
        templateExposure : `lsst.afw.image.exposureF`
            The DCR-matched template
        """
        templateImage = self.buildMatchedTemplate(exposure, warpCtrl, visitInfo, bbox, wcs, mask)
        templateExposure = afwImage.ExposureF(bbox, wcs)
        templateExposure.setMaskedImage(templateImage)
        templateExposure.setPsf(self.psf)
        templateExposure.setFilter(self.filter)
        return templateExposure

    def conditionDcrModel(self, subfilter, newModel, bbox, gain=1.):
        """Average two iterations' solutions to reduce oscillations.

        Parameters
        ----------
        subfilter : `int`
            Index of the current subfilter within the full band.
        newModel : `lsst.afw.image.MaskedImage`
            The new DCR model for one subfilter from the current iteration.
            Values in ``newModel`` that are extreme compared with the last
            iteration are modified in place.
        bbox : `lsst.afw.geom.Box2I`
            Sub-region of the coadd
        gain : `float`, optional
            Additional weight to apply to the model from the current iteration.
            Defaults to 1.0, which gives equal weight to both solutions.
        """
        # Calculate weighted averages of the image and variance planes.
        # Note that ``newModel *= gain`` would multiply the variance by ``gain**2``
        newModel.image *= gain
        newModel.image += self[subfilter][bbox].image
        newModel.image /= 1. + gain
        newModel.variance *= gain
        newModel.variance += self[subfilter][bbox].variance
        newModel.variance /= 1. + gain

    def clampModel(self, subfilter, newModel, bbox, statsCtrl, regularizeSigma, modelClampFactor,
                   convergenceMaskPlanes="DETECTED"):
        """Restrict large variations in the model between iterations.

        Parameters
        ----------
        subfilter : `int`
            Index of the current subfilter within the full band.
        newModel : `lsst.afw.image.MaskedImage`
            The new DCR model for one subfilter from the current iteration.
            Values in ``newModel`` that are extreme compared with the last
            iteration are modified in place.
        bbox : `lsst.afw.geom.Box2I`
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
        oldImage = self[subfilter][bbox].image.array
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
                        convergenceMaskPlanes="DETECTED"):
        """Restrict large variations in the model between subfilters.

        Any flux subtracted by the restriction is accumulated from all
        subfilters, and divided evenly to each afterwards in order to preserve
        total flux.

        Parameters
        ----------
        bbox : `lsst.afw.geom.Box2I`
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
            Mask planes to use to calculate convergence. (Default is "DETECTED")
            Default value is set in ``calculateNoiseCutoff`` if not supplied.
        """
        templateImage = self.getReferenceImage(bbox)
        excess = np.zeros_like(templateImage)
        for model in self:
            noiseCutoff = self.calculateNoiseCutoff(model, statsCtrl, regularizeSigma,
                                                    convergenceMaskPlanes=convergenceMaskPlanes,
                                                    mask=mask[bbox])
            modelVals = model.image.array
            highPixels = (modelVals > (templateImage*clampFrequency + noiseCutoff))
            excess[highPixels] += modelVals[highPixels] - templateImage[highPixels]*clampFrequency
            modelVals[highPixels] = templateImage[highPixels]*clampFrequency
            lowPixels = (modelVals < templateImage/clampFrequency - noiseCutoff)
            excess[lowPixels] += modelVals[lowPixels] - templateImage[lowPixels]/clampFrequency
            modelVals[lowPixels] = templateImage[lowPixels]/clampFrequency
        excess /= len(self)
        for model in self:
            model.image.array += excess

    def calculateNoiseCutoff(self, maskedImage, statsCtrl, regularizeSigma,
                             convergenceMaskPlanes="DETECTED", mask=None):
        """Helper function to calculate the background noise level of an image.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            The input image to evaluate the background noise properties.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd
        regularizeSigma : `float`
            Threshold to exclude noise-like pixels from regularization.
        convergenceMaskPlanes : `list` of `str`, or `str`
            Mask planes to use to calculate convergence.
        mask : `lsst.afw.image.Mask`, Optional
            Optional alternate mask

        Returns
        -------
        noiseCutoff : `float`
            The threshold value to treat pixels as noise in an image..
        """
        convergeMask = maskedImage.mask.getPlaneBitMask(convergenceMaskPlanes)
        if mask is None:
            mask = maskedImage.mask
        backgroundPixels = mask.array & (statsCtrl.getAndMask() | convergeMask) == 0
        noiseCutoff = regularizeSigma*np.std(maskedImage.image.array[backgroundPixels])
        return noiseCutoff


def applyDcr(maskedImage, dcr, warpCtrl, bbox=None, useInverse=False):
    """Shift a masked image.

    Parameters
    ----------
    maskedImage : `lsst.afw.image.MaskedImage`
        The input masked image to shift.
    dcr : `lsst.afw.geom.Extent2I`
        Shift calculated with ``calculateDcr``.
    warpCtrl : `lsst.afw.math.WarpingControl`
        Configuration settings for warping an image
    bbox : `lsst.afw.geom.Box2I`, optional
        Sub-region of the masked image to shift.
        Shifts the entire image if None (Default).
    useInverse : `bool`, optional
        Use the reverse of ``dcr`` for the shift. Default: False

    Returns
    -------
    `lsst.afw.image.maskedImageF`
        A masked image, with the pixels within the bounding box shifted.
    """
    padValue = afwImage.pixel.SinglePixelF(0., maskedImage.mask.getPlaneBitMask("NO_DATA"), 0)
    if bbox is None:
        bbox = maskedImage.getBBox()
    shiftedImage = afwImage.MaskedImageF(bbox)
    transform = makeTransform(AffineTransform((-1.0 if useInverse else 1.0)*dcr))
    afwMath.warpImage(shiftedImage, maskedImage[bbox],
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
    rotation = calculateImageParallacticAngle(visitInfo, wcs)
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


def calculateImageParallacticAngle(visitInfo, wcs):
    """Calculate the total sky rotation angle of an exposure.

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
