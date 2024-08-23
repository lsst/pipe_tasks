__all__ = ("mapUpperBounds", "latLum", "colorConstantSat", "lsstRGB", "mapUpperBounds")

import logging
import numpy as np
import skimage
import colour

from ._localContrast import localContrast
from .._pipeTasksLib import _fixGamut

from numpy.typing import NDArray
from typing import Callable, Mapping


def calcDelta0(a, b, c, d):
    """
    Calculates delta_0 = b^2 - 3ac.

    Parameters
    ----------
    a : float or int
        coefficient of x^2 term.
    b : float or int
        coefficient of x term.
    c : float or int
        constant term.
    d : not used, reserved for future extension.

    Returns
    -------
    delta_0 : float
        calculated value of delta_0
    """
    return b**2 - 3 * a * c


def calcDelta1(a, b, c, d):
    """
    Calculate delta_1 value for the cubic equation.

    Parameters
    ----------
    a : float
        Coefficient of x^3 term.
    b : float
        Coefficient of x^2 term.
    c : float
        Coefficient of x term.
    d : float
        Constant coefficient.

    Returns
    -------
    delta_1 : float
        Calculated value of delta_1

    """
    return 2 * b**3 - 9 * a * b * c + 27 * a**2 * d


def calcRoots(a, b, c, d):
    """
    Compute the three roots of a quartic polynomial.

    Parameters
    ----------
    a : array_like
        Coefficient of the highest degree term.
    b : array_like
        Coefficient of the second-highest degree term.
    c : array_like
        Coefficient of the first-highest degree term.
    d : array_like
        Constant term.

    Returns
    -------
    roots : list of complex128 arrays
        The three roots of the quartic polynomial, each with shape like 'a'.

    Notes
    -----
    This function uses the Cardano's formula for solving cubic equations,
    adapted to compute all three roots of a quartic equation.

    References
    ----------
    For more information on Cardano's formula and its application to quartic
    polynomials, see:

        http://mathworld.wolfram.com/CubicFormula.html
    """

    # Compute the discriminant delta0 for the cubic equation.
    delta0 = np.array(calcDelta0(a, b, c, d), dtype=np.complex128)

    # Compute the value of delta1.
    delta1 = calcDelta1(a, b, c, d)

    # Define Chi as (-1 + sqrt(-3)) / 2, which is a complex cube root.
    chi = (-1 + np.sqrt(-3 + 0j)) / 2.0

    # Compute C, the value needed to compute the roots.
    C = ((delta1 + np.sqrt(delta1**2 - 4 * delta0**3)) / 2) ** (1 / 3)

    # Create a mask for non-zero values of C.
    mask = C != 0

    # Initialize an empty list to store the roots.
    roots = []

    # Loop over each root and compute its value.
    for k in range(3):
        # Append a zero-filled array with shape like 'a'.
        roots.append(np.zeros(a.shape, dtype=np.complex128))

        # Compute the root using Cardano's formula.
        roots[k][mask] = (-1 / (3 * a[mask])) * (
            b[mask] + chi**k * C[mask] + delta0[mask] / (chi**k * C[mask])
        )

    return roots


def calcIntersection(h, fY, vert0, vert1, inputWhitepoint):
    """
    Calculate the intersection point in a,b Lab coordinate of the bounds of
    RGB space for a given luminocity given in XYZ coordinates along a line
    of constant hue.

    Parameters
    ----------
    h : NDArray
        An array of angles in radians for which to calculate the intersections.
    fY : float
        The luminance coordinate in XYZ space for the given points
    vert0 : NDArray
        Coordinates of the starting vertex as a 2xN array where N is the number
        of vertices.
    vert1 : NDArray
        Coordinates of the ending vertex as a 2xN array where N is the number of
        vertices.
    intputWhitepoint: tuple of float, float
        This is the white point of the input of the input arrays in CIE XY
        coordinates.

    Returns
    -------
    aInt : NDArray
        Intersection points along the cubic curve for each angle in 'h' (in Lab
        space).
    bInt : NDArray
        Corresponding values of 'bInt' at the intersection points.
    """

    # Check if vertices have the same x-coordinate to avoid division by zero
    eqMask = vert0[0] == vert1[0]
    # Adjust the x-coordinates slightly to avoid singularity
    vert0[0][eqMask] += 1e-5

    # Calculate the linear slope and intercept between the vertices
    m = (vert1[1] - vert0[1]) / (vert1[0] - vert0[0])
    n = vert1[1] - m * vert1[0]

    # fix m, n for the illuminant used
    xn, _, zn = colour.xyY_to_XYZ(colour.xy_to_xyY(inputWhitepoint))
    m = xn * m / zn
    n = n / zn

    # Compute tangent for each angle in h
    tanh = np.tan(h)

    # Calculate coefficients for the cubic equation representing the
    # intersection problem
    coeff1 = -1 * tanh**3 / 200**3 - m / 500**3
    coeff2 = (3 * tanh**2 / 200**2 - 3 * m / 500**2) * fY
    coeff3 = (-3 * tanh / 200 - 3 * m / 500) * fY**2
    coeff4 = fY**3 * (1 - m) - n

    # Find the roots of the cubic equation
    roots = calcRoots(coeff1, coeff2, coeff3, coeff4)

    # Check which root is real
    reals = np.imag(roots) == 0
    try:
        place = np.where(reals)[0][0]
    except Exception:
        return (-1e6 * np.ones(h.shape), -1e6 * np.ones(h.shape))

    # Extract the real root and use it to calculate the a intersection
    # coordinates (in Lab space)
    aInt = np.sign(h) * np.real(roots[place])
    # Calculate corresponding b intersection coordinates in (Lab space)
    bInt = tanh * aInt

    return (aInt, bInt)


def calcIntersection2(a, b, fy, vert0, vert1, inputWhitepoint):
    eqMask = vert1[0] == vert1[0]
    vert0[0][eqMask] += 1e-5
    # calculate the linear slope and intercept between the vertices
    m = (vert1[1] - vert0[1]) / (vert1[0] - vert0[0])
    n = vert1[1] - m * vert1[0]

    # fix m, n for the illuminant used
    xn, _, zn = colour.xyY_to_XYZ(colour.xy_to_xyY(inputWhitepoint))
    m = xn * m / zn
    n = n / zn

    c = np.sqrt(a**2 + b**2)
    h = np.arctan2(b, a)

    coeff0 = fy**3 - c**3 * m / 500**3 - 3 * m * c**2 * fy / 500**2 - 3 * c * m * fy**2 / 500 - m * fy**3 - n
    coeff1 = -3 * c * fy**2 / 100
    coeff2 = (
        3 * c**2 * fy / 100**2
        + 3 * fy**3
        + 3 * c**3 * m / 500**3
        + 3 * m * c**2 * fy / 500**2
        - 3 * m * c * fy**2 / 500
        - 3 * m * fy**3
        - n
    )
    coeff3 = -1 * c**3 / 100**3 - 6 * c * fy**2 / 100
    coeff4 = (
        3 * c**2 * fy / 100**2
        + 3 * fy**3
        - 3 * c**3 * m / 500**3
        + 3 * m * c**2 * fy / 500**2
        + 3 * c * m * fy**2 / 500
        - 3 * m * fy
    )
    coeff5 = -3 * c * fy**2 / 100
    coeff6 = 1 + c**3 * m / 500**3 - 3 * m * c**2 * fy / 500**2 + 3 * m * c * fy**2 / 500 - m * fy**3

    prop_a = []
    prop_b = []
    for i in range(len(fy)):
        roots = np.polynomial.polynomial.polyroots(
            (coeff0[i], coeff1[i], coeff2[i], coeff3[i], coeff4[i], coeff5[i], coeff6[i])
        )
        reals = np.real(roots[np.imag(roots) == 0])
        # transform t to theta in math
        reals = np.sign(b[i]) * abs(2 * np.arctan(reals))
        if len(reals) < 1:
            prop_a.append(-1000)
            prop_b.append(-1000)
            continue
        realMin = np.argmin(np.abs(reals - h[i]))
        theta = np.real(reals[realMin])
        prop_a.append(c[i] * np.cos(theta))
        prop_b.append(c[i] * np.sin(theta))
    return np.array(prop_a), np.array(prop_b)


def logAsinhLum(values: NDArray, stretch: float = 80, max=80, impact=0.8, minimum=0) -> NDArray:
    intensities = np.log(impact * np.arcsinh(values * stretch) + 1)
    intensities = intensities / intensities.max() * max
    intensities[intensities < minimum] = 0
    return intensities
    # return (s := np.log(impact * np.arcsinh(values * stretch) + 1)) / s.max() * max


# def latLum(values, stretch: float = 80, max: float = 90, A: float = 0.9, b0: float = 1.2, minimum=0):
def latLum(
    values,
    stretch: float = 100,
    max: float = 85,
    A: float = 0.9,
    b0: float = 0.05,
    minimum=0,
    floor: float = 0.00,
    Q: float = 8,
):
    import skimage

    values = skimage.restoration.denoise_wavelet(values)
    # what was there before!
    # intensities = A * np.arcsinh((values + floor) * stretch) / np.arcsinh(stretch) + b0
    # end what was there
    slope = 0.1 * 100 / np.arcsinh(0.1 * Q)
    soften = Q / stretch
    intensities = A * np.arcsinh((values * soften + floor) * slope) + b0
    intensities = intensities / intensities.max() * max
    intensities[intensities < minimum] = 0

    intensities = skimage.exposure.equalize_adapthist(
        intensities / max, tuple(k // 2 for k in intensities.shape)
    )
    intensities /= intensities.max()
    intensities *= max
    return intensities


def latLum(
    values,
    stretch: float = 100,
    max: float = 85,
    A: float = 0.9,
    b0: float = 0.05,
    minimum: float = 0,
    floor: float = 0.00,
    Q: float = 8,
    doDenoise: bool = True,
) -> NDArray:
    """
    Scale the input luminosity values to maximize the dynamic range visible.

    Parameters
    ----------
    values : `NDArray`
        The input image luminosity data of of.
    stretch : `float`, optional
        A parameter for the arcsinh function. Default is 100.
    max : `float`, optional
        Maximum value for intensity scaling. Default is 85.
    A `float`, optional
        Linear scaling factor for the transformed intensities. Default is 0.9.
    b0 `float` optional
        Offset term added to the arcsinh transformation. Default is 0.05.
    minimum: `float` optional
        Threshold below which pixel values are set to zero. Default is 0.
    floor : `float`, optional
        A value added to each pixel in arcsinh transform, this ensures values in
        the arcsinh transform will be no smaller than the supplied value.
        Default is 0.00.
    Q : `float`, optional
        Another parameter for the arcsinh function and scaling factor for
        softening. Default is 8.
    doDenoise : `bool`

    Returns:
        luminance : `NDArray`
            The stretched luminosity data.
    """

    # Denoise the input image using wavelet denoising.
    if doDenoise:
        values = skimage.restoration.denoise_wavelet(values)

    # Calculate the slope for arcsinh transformation based on Q and stretch parameters
    slope = 0.1 * 100 / np.arcsinh(0.1 * Q)

    # Apply the modified luminosity transformation using arcsinh function
    soften = Q / stretch
    intensities = A * np.arcsinh((values * soften + floor) * slope) + b0

    # Scale intensities to a maximum value of 'max' and normalize
    intensities = intensities / intensities.max() * max

    # If values end up near 100 it's best to "bend" them a little to help
    # the out of gamut fixer to appropriately handle luminosity and chroma
    # issues. This is an empirically derived formula that returns
    # scaling factors. For most of the domain it will return a value that
    # is close to 1. Right near the upper part of the domain, it
    # returns values slightly below 1 such that it scales a value of 100
    # to a value near 95.
    intensities *= (-1 * erf(-1 * (1 / dom * 210))) ** 15

    # Set pixel values below the specified minimum to zero
    intensities[intensities < minimum] = 0

    # Further enhance contrast using adaptive histogram equalization
    intensities = skimage.exposure.equalize_adapthist(
        intensities / max, tuple(k // 2 for k in intensities.shape)
    )

    # Normalize the final image intensities to a maximum value of 'max'
    intensities /= intensities.max()
    intensities *= max

    return intensities


def mapUpperBounds(img: NDArray, quant: float = 0.8, absMax: float | None = None) -> NDArray:
    """Bound images to a maximum of 1 by remapping with tanh.

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

    if absMax is None:
        turnover = np.max(np.vstack((r, g, b)))
        scale = turnover * quant

    else:
        scale = absMax

    image = np.empty(img.shape)
    image[:, :, 0] = r / scale
    image[:, :, 1] = g / scale
    image[:, :, 2] = b / scale

    # Clip values that exceed the bound to ensure all values are within [0, absMax]
    np.clip(image, None, scale, out=image)

    return image


def colorConstantSat_old(
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
    # hue = abs(np.arctan2(b, a))
    # tanHue = abs(b / a)
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
    sat_2 = chroma1_2 / div
    chroma2_2 = saturation * sat_2 * luminance**2 / (1 - sat_2)

    # Cap the chroma to avoid excessive values that are visually unrealistic
    chroma2_2[chroma2_2 > maxChroma**2] = maxChroma**2

    # Calculate the tangent of the hue angle, used for adjusting 'a' and 'b'
    # based on hue rotation.
    # tanHue = np.tan(hue)

    # Compute new 'a' values using the square root of adjusted chroma and
    # considering hue direction.
    # new_a = np.sign(a) * np.sqrt(chroma2_2 / (1 + tanHue**2))
    chroma2 = np.sqrt(chroma2_2)
    new_a = chroma2 * cosHue

    # Compute new 'b' values by scaling 'new_a' with the tangent of the hue
    # angle.
    ################# THis is the wrong sign!
    # new_b = np.sign(b) * new_a * tanHue
    new_b = chroma2 * sinHue

    return new_a, new_b


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

    # Calculate the hue angle
    chromaMask = chroma1 == 0
    chroma1[chromaMask] = 1
    sinHue = b / chroma1
    cosHue = a / chroma1
    sinHue[chromaMask] = 0
    cosHue[chromaMask] = 0

    # Compute a divisor for saturation calculation, adding 1 to avoid division
    # by zero.
    div = np.sqrt(1.155 + oldLum)
    div[div <= 0] = 1

    # Calculate the square of the new chroma based on desired saturation
    sat = chroma1 / div
    chroma2 = saturation * sat * np.sqrt(1.155 + luminance)

    # Cap the chroma to avoid excessive values that are visually unrealistic
    chroma2[chroma2 > maxChroma**2] = maxChroma**2

    # Compute new 'a' values using the adjusted chroma and considering hue
    # direction.
    new_a = chroma2 * cosHue

    # Compute new 'b' values by scaling 'new_a' with the tangent of the hue
    # angle.
    new_b = chroma2 * sinHue

    return new_a, new_b


def fixOutOfGamutColors(
    Lab: NDArray,
    inputWhitepoint: tuple[float, float],
    colourspace: str = "Display P3",
) -> None:
    """Remap colors that fall outside an RGB color gamut back into it.

    This function modifies the input Lab array inplace for memory reasons.

    Parameters
    ----------
    Lab : `NDArray`
        A NxMX3 array that contains data in the Lab colorspace.
    inputWhitepoint : tuple of float, float
        This is the white point of the input of the input arrays in CIE XY
        coordinates.
    colourspace : `str`
        The target colourspace to map outlying pixels into. This must
        correspond to an RGB colourspace understood by the colour-science
        python package

    Notes
    -----
    This method is based on the work done in
    https://doi.org/10.1007/s10043-019-00499-2. This implementaiton makes used
    of the bounds and basic ideas outlined in that paper, but implements the
    algorithm differently and expands it to include considerations of chroma.
    """
    # Convert back into the CIE XYZ colourspace.
    xyz_prime = colour.Lab_to_XYZ(Lab)

    # And then back to the specified RGB colourspace.
    rgb_prime = colour.XYZ_to_RGB(xyz_prime, colourspace=colourspace)

    # Determine if there are any out of bounds pixels
    outOfBounds = np.bitwise_or(
        np.bitwise_or(rgb_prime[:, :, 0] > 1, rgb_prime[:, :, 1] > 1), rgb_prime[:, :, 2] > 1
    )

    # If all pixels are in bounds, return immediately.
    if not np.any(outOfBounds):
        print("no out of bounds")
        return

    logging.info("There are out of gamut pixels, remapping colors")
    xn, yn, zn = colour.xyY_to_XYZ(colour.xy_to_xyY(inputWhitepoint))
    results = _fixGamut(Lab[outOfBounds], xn, yn, zn)
    Lab[outOfBounds] = results
    return

    # older solution, held onto for now
    Y = xyz_prime[:, :, 1]

    bounds = (
        (0, 0.0722),
        (0.0722, 0.2126),
        (0.2126, 0.2848),
        (0.2848, 0.7152),
        (0.7152, 0.7874),
        (0.7874, 0.9278),
        (0.9278, 1),
    )
    totalRemapped = 0

    # calculate the hue of each pixel
    h = np.arctan2(Lab[:, :, 2], Lab[:, :, 1])

    # These are the rows of the transformation matrices from XYZ to RGB

    # r, g, b: transformation coefficients for X
    C = (0.4124, 0.3576, 0.1805)
    # r, g, b: transformation coefficients for Y
    D = np.array((0.2126, 0.7152, 0.0722))
    # r, g, b: transformation coefficients for Z
    E = (0.0193, 0.1192, 0.9505)

    allMask = np.zeros(Y.shape, dtype=bool)
    for n, (bmin, bmax) in enumerate(bounds):
        # Create a mask of pixels in the brightness bin under consideration
        # that are also finite and out of bounds.
        mask = (Y > bmin) * (Y <= bmax) * np.isfinite(Y) * outOfBounds

        # Get the number of pixels out of bounds
        maskSize = np.sum(mask)

        # If the mask is empty it means there are no out of bounds pixels
        # in this luminance band, continue to the next.
        if not np.any(mask):
            continue

        # Extract just those values to work on
        val = Y[mask]

        # Each luminance bin have different intersection verities with the
        # RGB cube, so jump to the corresponding mapping. These all can't
        # be done ahead of time, as they depend on the exact values of Y
        # in the image. Make use of numpy vectorization to do everything
        # point by point though.
        match n:
            # each pair is the algebraically expanded coordinate in XZ space
            # at a given Y
            case 0:
                logging.debug("case 0")
                # intersection points of equal lightness plane
                # (0,0,X); (X, 0, 0); (0, X, 0)
                tmp = (val / D[2], val / D[0], val / D[1])
                vertices = (
                    (C[2] * tmp[0], E[2] * tmp[0]),
                    (C[0] * tmp[1], E[0] * tmp[1]),
                    (C[1] * tmp[2], E[1] * tmp[2]),
                )
            case 1:
                logging.debug("case 1")
                # rgb intersection points of equal lightness plane
                # (0,X,1); (X, 0, 1); (X, 0, 0); (0, X, 0)
                tmp = ((val - D[2]) / D[1], (val - D[2]) / D[0], val / D[0], val / D[1])
                vertices = (
                    (C[1] * tmp[0] + C[2], E[1] * tmp[0] + E[2]),
                    (C[0] * tmp[1] + C[2], E[0] * tmp[1] + E[2]),
                    (C[0] * tmp[2], E[0] * tmp[2]),
                    (C[1] * tmp[3], E[1] * tmp[3]),
                )
            case 2:
                logging.debug("case 2")
                # rgb intersection points of equal lightness plane
                # (0,X,1); (X, 0, 1); (1, 0, X); (1, X, 0); (0, X, 0)
                tmp = (
                    (val - D[2]) / D[1],
                    (val - D[2]) / D[0],
                    (val - D[0]) / D[2],
                    (val - D[0]) / D[1],
                    val / D[1],
                )
                vertices = (
                    (C[1] * tmp[0] + C[2], E[1] * tmp[0] + E[2]),
                    (C[0] * tmp[1] + C[2], E[0] * tmp[1] + E[2]),
                    (C[0] + C[2] * tmp[2], E[0] + E[2] * tmp[2]),
                    (C[0] + C[1] * tmp[3], E[0] + E[1] * tmp[3]),
                    (C[1] * tmp[4], E[1] * tmp[4]),
                )
            case 3:
                logging.debug("case 3")
                # rgb intersection points of equal lightness plane
                # (0,X,1); (1, X, 1); (1, X, 0); (0, X, 0)
                tmp = ((val - D[2]) / D[1], (val - D[0] - D[2]) / D[1], (val - D[0]) / D[1], val / D[1])
                vertices = (
                    (C[1] * tmp[0] + C[2], E[1] * tmp[0] + E[2]),
                    (C[0] + C[1] * tmp[1] + C[2], E[0] + E[1] * tmp[1] + E[2]),
                    (C[0] + C[1] * tmp[2], E[0] + E[1] * tmp[2]),
                    (C[1] * tmp[3], E[1] * tmp[3]),
                )
            case 4:
                logging.debug("case 4")
                # rgb intersection points of equal lightness plane
                # (0,X,1); (1, X, 1); (1, X, 0); (X, 1, 0); (0, 1, X)
                tmp = (
                    (val - D[2]) / D[1],
                    (val - D[0] - D[2]) / D[1],
                    (val - D[0]) / D[1],
                    (val - D[1]) / D[0],
                    (val - D[1]) / D[2],
                )
                vertices = (
                    (C[1] * tmp[0] + C[2], E[1] * tmp[0] + E[2]),
                    (C[0] + C[1] * tmp[1] + C[2], E[0] + E[1] * tmp[1] + E[2]),
                    (C[0] + C[1] * tmp[2], E[0] + E[1] * tmp[2]),
                    (C[0] * tmp[3] + C[1], E[0] * tmp[3] + E[1]),
                    (C[1] + tmp[4] * C[2], E[1] + tmp[4] * E[2]),
                )
            case 5:
                logging.debug("case 5")
                # rgb intersection points of equal lightness plane
                # (X,1,1); (1, X, 1); (1, X, 0); (X, 1, 0)
                tmp = (
                    (val - D[1] - D[2]) / D[0],
                    (val - D[0] - D[2]) / D[1],
                    (val - D[0]) / D[1],
                    (val - D[1]) / D[0],
                )
                vertices = (
                    (C[0] * tmp[0] + C[1] + C[2], E[0] * tmp[0] + E[1] + E[2]),
                    (C[0] + C[1] * tmp[1] + C[2], E[0] + E[1] * tmp[1] + E[2]),
                    (C[0] + C[1] * tmp[2], E[0] + E[1] * tmp[2]),
                    (C[0] * tmp[3] + C[1], E[0] * tmp[3] + E[1]),
                )
            case 6:
                logging.debug("case 6")
                # rgb intersection points of equal lightness plane
                # (X,1,1); (1, X, 1); (1, 1, X)
                tmp = ((val - D[1] - D[2]) / D[0], (val - D[0] - D[2]) / D[1], (val - D[0] - D[1]) / D[2])
                vertices = (
                    (C[0] * tmp[0] + C[1] + C[2], E[0] * tmp[0] + E[1] + E[2]),
                    (C[0] + C[1] * tmp[1] + C[2], E[0] + E[1] * tmp[1] + E[2]),
                    (C[0] + C[1] + C[2] * tmp[2], E[0] + E[1] + E[2] * tmp[2]),
                )
            case _:
                logging.debug("case _")
                vertices = tuple()

        a_container = np.zeros(maskSize)
        b_container = np.zeros(maskSize)
        dist_cur = None

        # Iterate over the vertices calculated preceding finding adjacent pairs.
        # Those pairs correspond to a boundary. Find the best color along a line
        # intersecting that boundary.
        for pos in range(len(vertices)):
            next = (pos + 1) % len(vertices)

            vert0 = (vertices[pos][0], vertices[pos][1])
            vert1 = (vertices[next][0], vertices[next][1])

            # really the complete answer would be jointly to solve for the
            # closest point in c, h space, but this is an expensive operation
            # and the results are good enough. This can be revisited in the
            # future.

            # calculate the shortest path to the line holding hue fixed
            prop_a, prop_b = calcIntersection(h[mask], (val) ** (1 / 3), vert0, vert1, inputWhitepoint)

            # calculate the shortest path to the line holding chroma fixed
            # prop_a2, prop_b2 = calcIntersection2(
            # Lab[mask, 1], Lab[mask, 2], val ** (1 / 3), vert0, vert1, inputWhitepoint
            # )

            if dist_cur is None:
                a_container[:] = prop_a
                b_container[:] = prop_b
                dist_cur = np.sqrt((a_container - Lab[mask, 1]) ** 2 + (b_container - Lab[mask, 2]) ** 2)
            # for pa, pb in ((prop_a, prop_b), (prop_a2, prop_b2)):
            for pa, pb in ((prop_a, prop_b),):
                dist_prop = np.sqrt((pa - Lab[mask, 1]) ** 2 + (pb - Lab[mask, 2]) ** 2)
                dist_mask = dist_prop < dist_cur

                a_container[dist_mask] = pa[dist_mask]
                b_container[dist_mask] = pb[dist_mask]
                dist_cur[dist_mask] = dist_prop[dist_mask]
        # testing something
        a_container = a_container / (1 + (a_container / b_container) ** 2) ** 0.5
        b_container = b_container / (1 + (a_container / b_container) ** 2) ** 0.5

        totalRemapped += a_container.size
        Lab[mask, 1] = a_container
        Lab[mask, 2] = b_container
        breakpoint()
        allMask += mask

    # make this a log message
    logging.debug(f"The total number of remapped pixels is: {totalRemapped}")


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
    sigma: float = 20,
    highlights: float = -1.5,
    shadows: float = 0.4,
    clarity: float = 0.2,
    maxLevel: int | None = None,
    psf: NDArray | None = None,
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
    # the abs. Here we choose the later, though this may have the effect of
    # raising the floor of the image a bit, this isn't really a bad thing as
    # it makes the background a grey color rather that pitch black which
    # can cause perceptual contrast issues.
    img = abs(img)

    # If there are nan's in the image there is no real option other than to
    # set them to zero or throw.
    img[np.isnan(img)] = 0

    # remap the bounds of the image if there is a function to do so.
    if remapBounds is not None:
        img = remapBounds(img, **(remapBoundsKwargs or {}))

    # Convert the starting image into the CIE L*a*b* color space.
    Lab = colour.XYZ_to_Lab(
        colour.RGB_to_XYZ(
            img,
            colourspace="CIE RGB",
            illuminant=np.array(cieWhitePoint),
            chromatic_adaptation_transform="bradford",
        )
    )
    # adjust lab for perceptual uniformity CIE94
    Lab[Lab[:, :, 2] == 0, 2] = 1e-7
    Lab[:, :, 1] = Lab[:, :, 1] / (1 + (Lab[:, :, 1] / Lab[:, :, 2]) ** 2) ** 0.5
    Lab[:, :, 2] = Lab[:, :, 2] / (1 + (Lab[:, :, 1] / Lab[:, :, 2]) ** 2) ** 0.5

    # make an alias for the luminance channel.
    lum = Lab[:, :, 0]

    if psf is not None:
        # This algorithm requires values between 0 and 1, but luminance is
        # between 0 and 100, so scale the input to and output of the
        # deconvolution algorithm.
        lum2 = skimage.restoration.richardson_lucy(lum / 100, psf=psf, clip=False, num_iter=10)
        lum2 *= 100
    else:
        lum2 = lum

    # Enhance the contrast of the input image before mapping.
    if doLocalContrast:
        newLum = localContrast(lum2, sigma, highlights, shadows, clarity=clarity, maxLevel=maxLevel)
        # Sometimes at the faint end the shadows can be driven a bit negative.
        # Take the abs to avoid black clipping issues.
        newLum = abs(newLum)
    else:
        newLum = lum2

    # Scale the luminance channnel if possible.
    if scaleLum is not None:
        lRemapped = scaleLum(newLum, **(scaleLumKWargs or {}))
    else:
        lRemapped = newLum

    # Only apply post luminance scaling contrast enhancement if the luminance
    # was actually scaled.
    if scaleLum is not None and doLocalContrast:
        maximumLumValue = lRemapped.max()
        lRemapped = localContrast(lRemapped, 50, highlights, shadows, clarity=clarity, maxLevel=maxLevel)
        lRemapped = abs(lRemapped)

        # make sure the maximum value does not change with the contrast
        # enhancement.
        lRemapped /= lRemapped.max()
        lRemapped *= maximumLumValue

    if scaleColor is not None:
        new_a, new_b = scaleColor(lum, lRemapped, Lab[:, :, 1], Lab[:, :, 2], **(scaleColorKWargs or {}))
        # Replace the color information with the new scaled color information.
        Lab[:, :, 1] = new_a
        Lab[:, :, 2] = new_b

    # Replace the luminance information with the new scaled luminance information
    Lab[:, :, 0] = lRemapped

    # Fix any colors that fall outside of the RGB colour gamut.
    fixOutOfGamutColors(Lab, inputWhitepoint=cieWhitePoint)

    """
    # the preceding transforms can produce values in Lab space that aren't able
    # to be represented in rgb space. What follows is finding closes color in
    # the equal lightness plane.
    xyz_prime = colour.Lab_to_XYZ(Lab)
    rgb_prime = colour.XYZ_to_RGB(xyz_prime, colourspace="Display P3")
    outOfBounds = np.bitwise_or(rgb_prime[:, :, 0] >= 1, rgb_prime[:, :, 1] >= 1, rgb_prime[:, :, 2] >= 1)
    if not np.any(outOfBounds):
        # result = colour.XYZ_to_RGB(colour.Lab_to_XYZ(Lab), colourspace="Display P3")
        result = rgb_prime

        # explicitly cut at 1 even though the mapping above was to map colors
        # appropriately because the Z matrix transform can produce values above
        # 1 and is a known feature of the transform.
        result[result > 1] = 1
        result[result < 0] = 0
        return result
    logging.info("There are out of gamut pixels, remapping colors")
    Y = xyz_prime[:, :, 1]

    bounds = (
        (0, 0.0722),
        (0.0722, 0.2126),
        (0.2126, 0.2848),
        (0.2848, 0.7152),
        (0.7152, 0.7874),
        (0.7874, 0.9278),
        (0.9278, 1),
    )
    totalRemapped = 0

    # calculate the hue of each pixel
    h = np.arctan2(Lab[:, :, 2], Lab[:, :, 1])

    # r, g, b: transformation coefficients for X
    C = (0.4124, 0.3576, 0.1805)
    # r, g, b: transformation coefficients for Y
    D = np.array((0.2126, 0.7152, 0.0722))
    # r, g, b: transformation coefficients for Z
    E = (0.0193, 0.1192, 0.9505)

    allMask = np.zeros(Y.shape, dtype=bool)
    for n, (bmin, bmax) in enumerate(bounds):
        mask = (Y > bmin) * (Y <= bmax) * np.isfinite(Y) * outOfBounds
        # val = np.zeros(Y.shape)
        # val[mask] = Y[mask]
        val = Y[mask]
        # valScale = 0.2*val + 1
        if not np.any(mask):
            continue
        match n:
            # each pair is the algebraically expanded coordinate in XZ space
            # at a given Y
            case 0:
                logging.debug("case 0")
                # intersection points of equal lightness plane
                # (0,0,X); (X, 0, 0); (0, X, 0)
                tmp = (val / D[2], val / D[0], val / D[1])
                vertices = (
                    (C[2] * tmp[0], E[2] * tmp[0]),
                    (C[0] * tmp[1], E[0] * tmp[1]),
                    (C[1] * tmp[2], E[1] * tmp[2]),
                )
            case 1:
                logging.debug("case 1")
                # rgb intersection points of equal lightness plane
                # (0,X,1); (X, 0, 1); (X, 0, 0); (0, X, 0)
                tmp = ((val - D[2]) / D[1], (val - D[2]) / D[0], val / D[0], val / D[1])
                vertices = (
                    (C[1] * tmp[0] + C[2], E[1] * tmp[0] + E[2]),
                    (C[0] * tmp[1] + C[2], E[0] * tmp[1] + E[2]),
                    (C[0] * tmp[2], E[0] * tmp[2]),
                    (C[1] * tmp[3], E[1] * tmp[3]),
                )
            case 2:
                logging.debug("case 2")
                # rgb intersection points of equal lightness plane
                # (0,X,1); (X, 0, 1); (1, 0, X); (1, X, 0); (0, X, 0)
                tmp = (
                    (val - D[2]) / D[1],
                    (val - D[2]) / D[0],
                    (val - D[0]) / D[2],
                    (val - D[0]) / D[1],
                    val / D[1],
                )
                vertices = (
                    (C[1] * tmp[0] + C[2], E[1] * tmp[0] + E[2]),
                    (C[0] * tmp[1] + C[2], E[0] * tmp[1] + E[2]),
                    (C[0] + C[2] * tmp[2], E[0] + E[2] * tmp[2]),
                    (C[0] + C[1] * tmp[3], E[0] + E[1] * tmp[3]),
                    (C[1] * tmp[4], E[1] * tmp[4]),
                )
            case 3:
                logging.debug("case 3")
                # rgb intersection points of equal lightness plane
                # (0,X,1); (1, X, 1); (1, X, 0); (0, X, 0)
                tmp = ((val - D[2]) / D[1], (val - D[0] - D[2]) / D[1], (val - D[0]) / D[1], val / D[1])
                vertices = (
                    (C[1] * tmp[0] + C[2], E[1] * tmp[0] + E[2]),
                    (C[0] + C[1] * tmp[1] + C[2], E[0] + E[1] * tmp[1] + E[2]),
                    (C[0] + C[1] * tmp[2], E[0] + E[1] * tmp[2]),
                    (C[1] * tmp[3], E[1] * tmp[3]),
                )
            case 4:
                logging.debug("case 4")
                # rgb intersection points of equal lightness plane
                # (0,X,1); (1, X, 1); (1, X, 0); (X, 1, 0); (0, 1, X)
                tmp = (
                    (val - D[2]) / D[1],
                    (val - D[0] - D[2]) / D[1],
                    (val - D[0]) / D[1],
                    (val - D[1]) / D[0],
                    (val - D[1]) / D[2],
                )
                vertices = (
                    (C[1] * tmp[0] + C[2], E[1] * tmp[0] + E[2]),
                    (C[0] + C[1] * tmp[1] + C[2], E[0] + E[1] * tmp[1] + E[2]),
                    (C[0] + C[1] * tmp[2], E[0] + E[1] * tmp[2]),
                    (C[0] * tmp[3] + C[1], E[0] * tmp[3] + E[1]),
                    (C[1] + tmp[4] * C[2], E[1] + tmp[4] * E[2]),
                )
            case 5:
                logging.debug("case 5")
                # rgb intersection points of equal lightness plane
                # (X,1,1); (1, X, 1); (1, X, 0); (X, 1, 0)
                tmp = (
                    (val - D[1] - D[2]) / D[0],
                    (val - D[0] - D[2]) / D[1],
                    (val - D[0]) / D[1],
                    (val - D[1]) / D[0],
                )
                vertices = (
                    (C[0] * tmp[0] + C[1] + C[2], E[0] * tmp[0] + E[1] + E[2]),
                    (C[0] + C[1] * tmp[1] + C[2], E[0] + E[1] * tmp[1] + E[2]),
                    (C[0] + C[1] * tmp[2], E[0] + E[1] * tmp[2]),
                    (C[0] * tmp[3] + C[1], E[0] * tmp[3] + E[1]),
                )
            case 6:
                logging.debug("case 6")
                # rgb intersection points of equal lightness plane
                # (X,1,1); (1, X, 1); (1, 1, X)
                tmp = ((val - D[1] - D[2]) / D[0], (val - D[0] - D[2]) / D[1], (val - D[0] - D[1]) / D[2])
                vertices = (
                    (C[0] * tmp[0] + C[1] + C[2], E[0] * tmp[0] + E[1] + E[2]),
                    (C[0] + C[1] * tmp[1] + C[2], E[0] + E[1] * tmp[1] + E[2]),
                    (C[0] + C[1] + C[2] * tmp[2], E[0] + E[1] + E[2] * tmp[2]),
                )
            case _:
                logging.debug("case _")
                vertices = tuple()

        maskSize = np.sum(mask)
        a_container = np.zeros(maskSize)
        b_container = np.zeros(maskSize)
        dist_cur = None
        for pos in range(len(vertices)):
            next = (pos + 1) % len(vertices)

            vert0 = (vertices[pos][0], vertices[pos][1])
            vert1 = (vertices[next][0], vertices[next][1])

            # really the complete answer would be jointly to solve for the
            # closest point in c, h space, but this is an expensive operation
            # and the results are good enough. This can be revisited in the
            # future.

            # calculate the shortest path to the line holding hue fixed
            prop_a, prop_b = calcIntersection(h[mask], (val) ** (1 / 3), vert0, vert1)

            # calculate the shortest path to the line holding chroma fixed
            prop_a2, prop_b2 = calcIntersection2(Lab[mask, 1], Lab[mask, 2], val ** (1 / 3), vert0, vert1)

            if dist_cur is None:
                a_container[:] = prop_a
                b_container[:] = prop_b
                dist_cur = np.sqrt((a_container - Lab[mask, 1]) ** 2 + (b_container - Lab[mask, 2]) ** 2)
            # for pa, pb in ((prop_a, prop_b), (prop_a2, prop_b2)):
            for pa, pb in ((prop_a, prop_b),):
                dist_prop = np.sqrt((pa - Lab[mask, 1]) ** 2 + (pb - Lab[mask, 2]) ** 2)
                dist_mask = dist_prop < dist_cur

                a_container[dist_mask] = pa[dist_mask]
                b_container[dist_mask] = pb[dist_mask]
                dist_cur[dist_mask] = dist_prop[dist_mask]

        totalRemapped += a_container.size
        Lab[mask, 1] = a_container
        Lab[mask, 2] = b_container
        allMask += mask

    # make this a log message
    logging.debug(f"The total number of remapped pixels is: {totalRemapped}")
    """
    result = colour.XYZ_to_RGB(colour.Lab_to_XYZ(Lab), colourspace="Display P3")

    # explicitly cut at 1 even though the mapping above was to map colors
    # appropriately because the Z matrix transform can produce values above
    # 1 and is a known feature of the transform.
    result[result > 1] = 1
    result[result < 0] = 0
    return result
