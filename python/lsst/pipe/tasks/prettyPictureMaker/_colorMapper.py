__all__ = ("mapUpperBounds", "latLum", "colorConstantSat", "lsstRGB", "mapUpperBounds")

import logging
import numpy as np
import skimage
import colour
import cv2
from scipy.special import erf
from scipy.interpolate import pchip_interpolate

from ._localContrast import localContrast, makeGaussianPyramid, makeLapPyramid, levelPadder
from lsst.cpputils import fixGamutOK

from numpy.typing import NDArray
from typing import Callable, Mapping


def latLum(
    values,
    stretch: float = 400,
    max: float = 85,
    A: float = 1,
    b0: float = 0.0,
    minimum: float = 0,
    floor: float = 0.00,
    Q: float = 0.7,
    doDenoise: bool = False,
) -> NDArray:
    """
    Scale the input luminosity values to maximize the dynamic range visible.

    Parameters
    ----------
    values : `NDArray`
        The input image luminosity data of of.
    stretch : `float`, optional
        A parameter for the arcsinh function.
    max : `float`, optional
        Maximum value for intensity scaling.
    A : `float`, optional
        Linear scaling factor for the transformed intensities.
    b0 : `float` optional
        Offset term added to the arcsinh transformation.
    minimum : `float`
        Threshold below which pixel values are set to zero.
    floor : `float`
        A value added to each pixel in arcsinh transform, this ensures values in
        the arcsinh transform will be no smaller than the supplied value.
    Q : `float`
        Another parameter for the arcsinh function and scaling factor for
        softening.
    doDenoise : `bool`
        Denoise the image if desired.

    Returns:
        luminance : `NDArray`
            The stretched luminosity data.
    """

    # De-noise the input image using wavelet de-noising.
    if doDenoise:
        values = skimage.restoration.denoise_wavelet(values)
        values = abs(values)

    # Scale values from 0-1 to 0-100 as various algorithm expect that range.
    values *= 100

    # Find what fraction of 100 the brightest pixel is. This is then used to
    # re-normalize after all the non-linear scaling such that the brightest part
    # of the image corresponds to the same absolute brightness.
    maxRatio = values.max() / 100

    # Calculate the slope for arcsinh transformation based on Q and stretch
    # parameters.
    slope = 0.1 * 100 / np.arcsinh(0.1 * Q)

    # Apply the modified luminosity transformation using arcsinh function.
    soften = Q / stretch
    intensities = A * np.arcsinh((abs(values) * soften + floor) * slope) + b0

    intensities /= intensities.max() / maxRatio
    np.clip(intensities, 0, 1, out=intensities)
    intensities *= 100

    # Apply a specific tone cure to the luminocity defined by the below interpolant.
    # This is calculated on the median of the image to smooth out pixel to pixel
    # variations that are most likely due to noise. The sharpness of the image
    # is preserved as we only apply this filter to the luminocity data.
    # filtered = medfilt2d(intensities, 3)

    control_points = (
        [0, 0.5, 2, 5, 13.725490196078432, 25, 30, 55.294117647058826, 73.72549019607844, 98, 100],
        [0, 10, 15, 20, 25.686274509803921, 40, 50, 80.35294117647058, 94.11764705882352, 98, 100],
    )
    scaled = pchip_interpolate(*control_points, intensities)
    scaled[scaled == 0] = 1e-7
    intensities = scaled
    intensities[intensities > max] = max

    # If values end up near 100 it's best to "bend" them a little to help
    # the out of gamut fixer to appropriately handle luminosity and chroma
    # issues. This is an empirically derived formula that returns
    # scaling factors. For most of the domain it will return a value that
    # is close to 1. Right near the upper part of the domain, it
    # returns values slightly below 1 such that it scales a value of 100
    # to a value near 95.
    intensities *= (-1 * erf(-1 * (1 / intensities * 210))) ** 20
    intensities[np.isnan(intensities)] = 0

    # Reset the input array.
    values /= 100

    # Rescale the output array.
    intensities /= 100

    return intensities


def mapUpperBounds(
    img: NDArray, quant: float = 0.9, absMax: float | None = None, scaleBoundFactor: float | None = None
) -> NDArray:
    """Bound images to a range between zero and one.

    Some images supplied aren't properly bounded with a maximum value of 1.
    Either the images exceed the bounds of 1, or that no value seems to close,
    implying indeterminate maximum value. This function determines an
    appropriate maximum either by taking the value supplied in the absMax
    argument or by scaling the maximum across all channels with the
    supplied quant variable.

    Parameters
    ----------
    img : `NDArray` like
        Must have dimensions of y,x,3 where the channels are in RGB order
    quant : `float`
        Value to scale the maximum pixel value, in any channel, by to
        determine the maximum flux allowable in all channels. Ignored
        if absMax isn't None.
    absMax : `float` or `None`
        If this value is not None, use it as the maximum pixel value
        for all channels, unless scaleBoundFactor is set in which case
        it is only the maximum if the value determined from the image
        and quant is larger than scaleBoundFactor times absMax. This is
        to prevent individual frames in a mosaic from being scaled too
        faint if absMax is too large for one region.
    scaleBoundFactor : `float` or `None`
        Factor used to compare absMax and the emperically determined
        maximim. if emperical_max is less than scaleBoundFactor*absMax
        then the emperical_max is used instead of absMax, even if it
        is set. Set to None to skip this comparison.

    Returns
    -------
    image : `NDArray`
        The result of the remapping process
    """
    if np.max(img) == 1:
        return img

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    turnover = np.max(np.vstack((r, g, b)))

    # If scaleBoundFactor is not none and absMax is not None, check that the
    # determined turnover is not less than the supplied absMax times the
    # scaleBoundFactor. This fixes patches that may have max values much less
    # than others for some processing reason.
    if absMax is not None:
        if scaleBoundFactor is not None and turnover < scaleBoundFactor * absMax:
            scale = turnover * quant
        else:
            scale = absMax
    else:
        scale = turnover * quant

    image = np.empty(img.shape)
    image[:, :, 0] = r / scale
    image[:, :, 1] = g / scale
    image[:, :, 2] = b / scale

    # Clip values that exceed the bound to ensure all values are within [0, absMax]
    np.clip(image, 0, 1, out=image)

    return image


def colorConstantSat(
    oldLum: NDArray, luminance: NDArray, a: NDArray, b: NDArray, saturation: float = 1, maxChroma: float = 50
) -> tuple[NDArray, NDArray]:
    """
    Adjusts the color saturation while keeping the hue constant.

    This function adjusts the chromaticity (a, b) of colors to maintain a
    consistent saturation level, based on their original luminance. It uses
    the CIELAB color space representation and the `luminance` is the new target
    luminance for all colors.

    Parameters
    ----------
    oldLum : `NDArray`
        Luminance values of the original colors.
    luminance : `NDArray`
        Target luminance values for the transformed colors.
    a : `NDArray`
        Chromaticity parameter 'a' corresponding to green-red axis in CIELAB.
    b : `NDArray`
        Chromaticity parameter 'b' corresponding to blue-yellow axis in CIELAB.
    saturation : `float`, optional
        Desired saturation level for the output colors. Defaults to 1.
    maxChroma : `float`, optional
        Maximum chroma value allowed for any color. Defaults to 50.

    Returns
    -------
    new_a : NDArray
        New a values representing the adjusted chromaticity.
    new_b : NDArray
        New b values representing the adjusted chromaticity.
    """
    # Calculate the square of the chroma, which is the distance from origin in
    # the a-b plane.
    chroma1_2 = a**2 + b**2
    chroma1 = np.sqrt(chroma1_2)

    # Calculate the hue angle, taking the absolute value to ensure non-negative
    # angle representation.
    chromaMask = chroma1 == 0
    chroma1[chromaMask] = 1
    sinHue = b / chroma1
    cosHue = a / chroma1
    sinHue[chromaMask] = 0
    cosHue[chromaMask] = 0

    # Compute a divisor for saturation calculation, adding 1 to avoid division
    # by zero.
    div = chroma1_2 + oldLum**2
    div[div <= 0] = 1

    # Calculate the square of the new chroma based on desired saturation
    sat_original_2 = chroma1_2 / div
    chroma2_2 = saturation * sat_original_2 * luminance**2 / (1 - sat_original_2)

    # Cap the chroma to avoid excessive values that are visually unrealistic
    chroma2_2[chroma2_2 > maxChroma**2] = maxChroma**2

    # Compute new 'a' values using the square root of adjusted chroma and
    # considering hue direction.
    chroma2 = np.sqrt(chroma2_2)
    new_a = chroma2 * cosHue

    # Compute new 'b' values by scaling 'new_a' with the tangent of the sin
    # angle.
    new_b = chroma2 * sinHue

    return new_a, new_b


def fixOutOfGamutColors(
    Lab: NDArray,
    colourspace: str = "Display P3",
) -> None:
    """Remap colors that fall outside an RGB color gamut back into it.

    This function modifies the input Lab array in-place for memory reasons.

    Parameters
    ----------
    Lab : `NDArray`
        A NxMX3 array that contains data in the Lab colorspace.
    colourspace : `str`
        The target colourspace to map outlying pixels into. This must
        correspond to an RGB colourspace understood by the colour-science
        python package
    """
    # Convert back into the CIE XYZ colourspace.
    xyz_prime = colour.Oklab_to_XYZ(Lab)

    # And then back to the specified RGB colourspace.
    rgb_prime = colour.XYZ_to_RGB(xyz_prime, colourspace=colourspace)

    # Determine if there are any out of bounds pixels
    outOfBounds = np.bitwise_or(
        np.bitwise_or(rgb_prime[:, :, 0] > 1, rgb_prime[:, :, 1] > 1), rgb_prime[:, :, 2] > 1
    )

    # If all pixels are in bounds, return immediately.
    if not np.any(outOfBounds):
        logging.info("There are no out of gamut pixels.")
        return

    logging.info("There are out of gamut pixels, remapping colors")
    results = fixGamutOK(Lab[outOfBounds])
    logging.debug(f"The total number of remapped pixels is: {np.sum(outOfBounds)}")
    Lab[outOfBounds] = results
    return


def _fuseExposure(images, sigma=0.2, maxLevel=3):
    weights = np.zeros((len(images), *images[0].shape[:2]))
    for i, image in enumerate(images):
        exposure = np.exp(-((image[:, :, 0] - 0.5) ** 2) / (2 * sigma))

        weights[i, :, :] = exposure
    norm = np.sum(weights, axis=0)
    np.divide(weights, norm, out=weights)

    # loop over each image again to build pyramids
    g_pyr = []
    l_pyr = []
    maxImageLevel = int(np.min(np.log2(images[0].shape[:2])))
    if maxLevel is None:
        maxLevel = maxImageLevel
    if maxImageLevel < maxLevel:
        raise ValueError(
            f"The supplied max level {maxLevel} is is greater than the max of the image: {maxImageLevel}"
        )
    support = 1 << (maxLevel - 1)
    padY_amounts = levelPadder(image.shape[0] + support, maxLevel)
    padX_amounts = levelPadder(image.shape[1] + support, maxLevel)
    for image, weight in zip(images, weights):
        imagePadded = cv2.copyMakeBorder(
            image, *(0, support), *(0, support), cv2.BORDER_REPLICATE, None, None
        ).astype(image.dtype)
        weightPadded = cv2.copyMakeBorder(
            weight, *(0, support), *(0, support), cv2.BORDER_REPLICATE, None, None
        ).astype(image.dtype)

        g_pyr.append(list(makeGaussianPyramid(weightPadded, padY_amounts, padX_amounts, None)))
        l_pyr.append(list(makeLapPyramid(imagePadded, padY_amounts, padX_amounts, None, None)))

    # time to blend
    blended = []
    for level in range(len(padY_amounts)):
        accumulate = np.zeros_like(l_pyr[0][level])
        for img in range(len(g_pyr)):
            for i in range(3):
                accumulate[:, :, i] += l_pyr[img][level][:, :, i] * g_pyr[img][level]
        blended.append(accumulate)

    # time to reconstruct
    output = blended[-1]
    for i in range(-2, -1 * len(blended) - 1, -1):
        upsampled = cv2.pyrUp(output)
        upsampled = upsampled[
            : upsampled.shape[0] - 2 * padY_amounts[i + 1], : upsampled.shape[1] - 2 * padX_amounts[i + 1]
        ]
        output = blended[i] + upsampled
    return output[:-support, :-support]


def _handelLuminance(
    img: NDArray,
    scaleLum: Callable[..., NDArray] | None = latLum,
    scaleLumKWargs: Mapping | None = None,
    remapBounds: Callable | None = mapUpperBounds,
    remapBoundsKwargs: Mapping | None = None,
    doLocalContrast: bool = True,
    sigma: float = 30,
    highlights: float = -0.9,
    shadows: float = 0.5,
    clarity: float = 0.1,
    maxLevel: int | None = None,
    cieWhitePoint: tuple[float, float] = (0.28, 0.28),
    bracket: float = 1,
    psf: NDArray | None = None,
):
    # remap the bounds of the image if there is a function to do so.
    if remapBounds is not None:
        img = remapBounds(img, **(remapBoundsKwargs or {}))

    # scale to the supplied bracket
    img /= bracket

    # Convert the starting image into the OK L*a*b* color space.
    # https://en.wikipedia.org/wiki/Oklab_color_space

    Lab = colour.XYZ_to_Oklab(
        colour.RGB_to_XYZ(
            img,
            colourspace="CIE RGB",
            illuminant=np.array(cieWhitePoint),
            chromatic_adaptation_transform="bradford",
        )
    )
    lum = Lab[:, :, 0]

    # This works because lum must be between zero and one, so the max it the ratio of the max
    maxRatio = lum.max()

    # Enhance the contrast of the input image before mapping.
    if doLocalContrast:
        newLum = localContrast(lum, sigma, highlights, shadows, clarity=clarity, maxLevel=maxLevel)
        # Sometimes at the faint end the shadows can be driven a bit negative.
        # Take the abs to avoid black clipping issues.
        newLum = abs(newLum)
        # because contrast enhancement can change the maximum value, linearly
        # rescale the image such that the maximum is at the same ratio as the
        # original maximum.
        newLum /= newLum.max() / maxRatio
    else:
        newLum = lum

    # Scale the luminance channel if possible.
    if scaleLum is not None:
        lRemapped = scaleLum(newLum, **(scaleLumKWargs or {}))
    else:
        lRemapped = newLum

    if psf is not None:
        lRemapped = skimage.restoration.richardson_lucy(lRemapped, psf=psf, clip=False, num_iter=2)
    return lRemapped, Lab


def lsstRGB(
    rArray: NDArray,
    gArray: NDArray,
    bArray: NDArray,
    doLocalContrast: bool = True,
    scaleLum: Callable[..., NDArray] | None = latLum,
    scaleLumKWargs: Mapping | None = None,
    scaleColor: Callable[..., tuple[NDArray, NDArray]] | None = colorConstantSat,
    scaleColorKWargs: Mapping | None = None,
    remapBounds: Callable | None = mapUpperBounds,
    remapBoundsKwargs: Mapping | None = None,
    cieWhitePoint: tuple[float, float] = (0.28, 0.28),
    sigma: float = 30,
    highlights: float = -0.9,
    shadows: float = 0.5,
    clarity: float = 0.1,
    maxLevel: int | None = None,
    psf: NDArray | None = None,
    brackets: list[float] | None = None,
) -> NDArray:
    """Enhance the lightness and color preserving hue using perceptual methods.

    Parameters
    ----------
    rArray : `NDArray`
        The array used as the red channel
    gArray : `NDArray`
        The array used as the green channel
    bArray : `NDArray`
        The array used as the blue channel
    doLocalContrast: `bool`
        Apply local contrast enhancement algorithms to the luminance channel.
    scaleLum : `Callable` or `None`
        This is a callable that's passed the luminance values as well as
        any defined scaleLumKWargs, and should return a scaled luminance array
        the same shape as the input. Set to None for no scaling.
    scaleLumKWargs : `Mapping` or `None`
        Key word arguments that passed to the scaleLum function.
    scaleColor : `Callable` or `None`
        This is a callable that's passed the original luminance, the remapped
        luminance values, the a values for each pixel, and the b values for
        each pixel. The function is also passed any parameters defined in
        scaleColorKWargs. This function is responsible for scaling chroma
        values. This should return two arrays corresponding to the scaled a and
        b values. Set to None for no modification.
    scaleColorKWargs : `Mapping` or `None`
        Key word arguments passed to the scaleColor function.
    remapBounds : `Callable` or `None`
        This is a callable that should remaps the input arrays such that each of
        them fall within a zero to one range. This callable is given the
        initial image as well as any parameters defined in the remapBoundsKwargs
        parameter. Set to None for no remapping.
    remapBoundsKwargs : `Mapping` or None
    cieWhitePoint : `tuple` of `float`, `float`
        This is the white point of the input of the input arrays in CIE XY
        coordinates. Altering this affects the relative balance of colors
        in the input image, and therefore also the output image.
    sigma : `float`
        The scale over which local contrast considers edges real and not noise.
    highlights : `float`
        A parameter that controls how local contrast enhances or reduces
        highlights. Contrary to intuition, negative values increase highlights.
    shadows : `float`
        A parameter that controls how local contrast will deepen or reduce
        shadows.
    clarity : `float`
        A parameter that relates to the local contrast between highlights and
        shadow.
    maxLevel : `int` or `None`
        The maximum number of image pyramid levels to enhance the local contrast
        over. Each level has a spatial scale of roughly 2^(level) pixels.
    psf : `NDArray` or `None`
        If this parameter is an image of a PSF kernel the luminance channel is
        deconvolved with it. Set to None to skip deconvolution.
    brackets : `list` of `float` or `None`
        If a list brackets is supplied, an image will  be generated at each of
        the brackets and the results will be used in exposure fusioning to
        increase the apparent dynamic range of the image. The image post bounds
        remapping will be divided by each of the values specified in this list,
        which can be used to create for instance an under, over, and ballanced
        expoisure. Theese will then be fusioned into a final single exposure
        selecting the proper elements from each of the images.

    Returns
    -------
    result : `NDArray`
        The brightness and color calibrated image.

    Raises
    ------
    ValueError
        Raised if the shapes of the input array don't match
    """
    if rArray.shape != gArray.shape or rArray.shape != bArray.shape:
        raise ValueError("The shapes of all the input arrays must be the same")

    # Construct a new image array in the proper byte ordering.
    img = np.empty((*rArray.shape, 3))
    img[:, :, 0] = rArray
    img[:, :, 1] = gArray
    img[:, :, 2] = bArray

    # The image might contain pixels less than zero due to noise. The options
    # for handling this are to either set them to zero, which creates weird
    # holes in the scaled output image, throw an exception and have the user
    # handle it, which they might not have to proper understanding to, or take
    # the abs. Here the code uses the later, though this may have the effect of
    # raising the floor of the image a bit, this isn't really a bad thing as
    # it makes the background a grey color rather that pitch black which
    # can cause perceptual contrast issues.
    img = abs(img)

    # If there are nan's in the image there is no real option other than to
    # set them to zero or throw.
    img[np.isnan(img)] = 0

    if not brackets:
        brackets = [1]

    exposures = []
    for im_num, bracket in enumerate(brackets):
        tmp_lum, Lab = _handelLuminance(
            img,
            scaleLum,
            scaleLumKWargs=scaleLumKWargs,
            remapBounds=remapBounds,
            remapBoundsKwargs=remapBoundsKwargs,
            doLocalContrast=doLocalContrast,
            sigma=sigma,
            highlights=highlights,
            clarity=clarity,
            maxLevel=maxLevel,
            cieWhitePoint=cieWhitePoint,
            bracket=bracket,
            psf=psf,
        )
        if scaleColor is not None:
            new_a, new_b = scaleColor(
                Lab[:, :, 0], tmp_lum, Lab[:, :, 1], Lab[:, :, 2], **(scaleColorKWargs or {})
            )
            # Replace the color information with the new scaled color information.
            Lab[:, :, 1] = new_a
            Lab[:, :, 2] = new_b
        # Replace the luminance information with the new scaled luminance information
        Lab[:, :, 0] = tmp_lum
        exposures.append(Lab)
    if len(brackets) > 1:
        Lab = _fuseExposure(exposures)

    # Fix any colors that fall outside of the RGB colour gamut.
    fixOutOfGamutColors(Lab)

    # Transform back to RGB coordinates
    result = colour.XYZ_to_RGB(colour.Oklab_to_XYZ(Lab), colourspace="Display P3")

    # explicitly cut at 1 even though the mapping above was to map colors
    # appropriately because the Z matrix transform can produce values above
    # 1 and is a known feature of the transform.
    result[result > 1] = 1
    result[result < 0] = 0
    return result
