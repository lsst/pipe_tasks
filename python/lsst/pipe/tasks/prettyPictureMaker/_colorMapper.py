
__all__ = ("mapUpperBounds", "latLum", "colorConstantSat", "lsstRGB", "mapUpperBounds")

import logging
import numpy as np
import colour

from ._localContrast import localContrast

from numpy.typing import NDArray
from typing import Callable


def calcDelta0(a, b, c, d):
    return b**2 - 3 * a * c


def calcDelta1(a, b, c, d):
    return 2 * b**3 - 9 * a * b * c + 27 * a**2 * d


def calcRoots(a, b, c, d):
    delta0 = np.array(calcDelta0(a, b, c, d), dtype=np.complex128)
    delta1 = calcDelta1(a, b, c, d)

    chi = (-1 + np.sqrt(-3 + 0j)) / 2.0
    C = ((delta1 + np.sqrt(delta1**2 - 4 * delta0**3)) / 2) ** (1 / 3)
    mask = C != 0
    roots = []
    for k in range(3):
        roots.append(np.zeros(a.shape, dtype=np.complex128))
        roots[k][mask] = (-1 / (3 * a[mask])) * (
            b[mask] + chi**k * C[mask] + delta0[mask] / (chi**k * C[mask])
        )
    return roots


def calcIntersection(h, fY, vert0, vert1):
    eqMask = vert1[0] == vert1[0]
    vert0[0][eqMask] += 1e-5
    # calculate the linear slope and intercept between the vertices
    m = (vert1[1] - vert0[1]) / (vert1[0] - vert0[0])
    n = vert1[1] - m * vert1[0]

    tanh = np.tan(h)
    coeff1 = -1 * tanh**3 / 200**3 - m / 500**3
    coeff2 = (3 * tanh**2 / 200**2 - 3 * m / 500**2) * fY
    coeff3 = (-3 * tanh / 200 - 3 * m / 500) * fY**2
    coeff4 = fY**3 * (1 - m) - n

    roots = calcRoots(coeff1, coeff2, coeff3, coeff4)
    reals = np.imag(roots) == 0
    try:
        place = np.where(reals)[0][0]
    except Exception:
        return (-1e6 * np.ones(h.shape), -1e6 * np.ones(h.shape))
    aInt = np.sign(h) * np.real(roots[place])
    bInt = tanh * aInt
    return (aInt, bInt)


def calcIntersection2(a, b, fy, vert0, vert1):
    eqMask = vert1[0] == vert1[0]
    vert0[0][eqMask] += 1e-5
    # calculate the linear slope and intercept between the vertices
    m = (vert1[1] - vert0[1]) / (vert1[0] - vert0[0])
    n = vert1[1] - m * vert1[0]

    c = np.sqrt(a**2 + b**2)
    h = np.arctan2(b, a)

    coeff0 = (
        fy**3
        - c**3 * m / 500**3
        - 3 * m * c**2 * fy / 500**2
        - 3 * c * m * fy**2 / 500
        - m * fy**3
        - n
    )
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
    coeff6 = (
        1 + c**3 * m / 500**3 - 3 * m * c**2 * fy / 500**2 + 3 * m * c * fy**2 / 500 - m * fy**3
    )

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
    intensities = intensities/intensities.max() * max
    intensities[intensities < minimum] = 0
    return intensities
    #return (s := np.log(impact * np.arcsinh(values * stretch) + 1)) / s.max() * max


#def latLum(values, stretch: float = 80, max: float = 90, A: float = 0.9, b0: float = 1.2, minimum=0):
def latLum(values, stretch: float = 100, max: float = 85, A: float = 0.9, b0: float = 0.05, minimum=0, floor: float = 0.00):
    intensities = A*np.arcsinh((values+floor)*stretch)/np.arcsinh(stretch)+b0
    intensities = intensities/intensities.max()*max
    intensities[intensities < minimum] = 0
    return intensities


def mapUpperBounds(img: NDArray, quant=0.8, absMax=None) -> NDArray:
    """Bound images to a maximum of 1 by remapping with tanh.

    Some images supplied are not properly bounded with a maximum value of 1.
    Either the images exceed the bounds of 1, or that no value seems to close,
    implying indeterminate maximum value. This function determines an
    appropriate maximum by mapping the 98th quantile has the value of two. This
    new domain is then fed though a tanh function.

    Parameters
    ----------
    img : array like
        Must have dimensions of y,x,3 where the channels are in rgb order

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
        #scale = turnover / quant
        scale = turnover * quant
        #scale = np.quantile(np.vstack((r, g, b)), quant)

    else:
        scale = absMax
    print(scale)

    image = np.empty(img.shape)
    image[:, :, 0] = r / scale
    image[:, :, 1] = g / scale
    image[:, :, 2] = b / scale
    image[image > scale] = scale

    return image


def colorConstantSat(
    oldLum: NDArray,
    luminance: NDArray,
    a: NDArray,
    b: NDArray,
    saturation: float = 1,
    maxChroma: float = 50,
) -> tuple[NDArray, NDArray]:
    chroma1 = np.sqrt(a**2 + b**2)
    hue = abs(np.arctan2(b, a))
    div = chroma1**2 + oldLum**2
    div[div <= 0] = 1
    sat_2 = chroma1**2 / div
    chroma2_2 = saturation * sat_2 * luminance**2 / (1 - sat_2)
    chroma2 = np.sqrt(chroma2_2)
    chroma2[chroma2 > maxChroma] = maxChroma
    chroma2_2 = chroma2**2
    new_a = np.sign(a) * np.sqrt(chroma2_2 / (1 + np.tan(hue) ** 2))
    new_b = np.sign(b) * new_a * np.tan(hue)
    return new_a, new_b


def lsstRGB(
    rArray: NDArray,
    gArray: NDArray,
    bArray: NDArray,
    doLocalContrast: bool = True,
    scaleLum: Callable[..., NDArray] = latLum,
    scaleLumKWargs: dict | None = None,
    scaleColor: Callable[..., tuple[NDArray, NDArray]] = colorConstantSat,
    scaleColorKWargs: dict | None = None,
    remapBounds: Callable | None = mapUpperBounds,
    remapBoundsKwargs: dict | None = None,
    cieWhitePoint: tuple[float, float] = (0.28, 0.28),
    sigma: float = 20,
    highlights: float = -1.5,
    shadows: float = 0.4,
    clarity: float = 0.2

) -> NDArray:
    """Enhance the lightness and color preserving hue using perceptual methods.

    Parameters
    ----------
    rArray : `NDArray`
        The array to be used as the red channel
    gArray : `NDArray`
        The array to be used as the green channel
    bArray : `NDArray`
        The array to be used as the blue channel
    scaleLum : `Callable`
        This is a callable that will be passed the luminance values as well as
        any defined scaleLumKWargs, and should return a scaled luminance array
        the same shape as the input.
    scaleLumKWargs : `dict` or `None`
        Key word arguments that will be passed to the scaleLum function.
    scaleColor : `Callable`
        This is a callable that will be passed the remapped luminance values,
        the chroma for each pixel in the image, and any defined
        scaleColorKWargs. This function is responsible for determining any
        chroma scaling that will be applied. This should return an array of
        scaling factors that will be applied to the chroma, preserving hues.
    scaleColorKWargs : `dict` or `None`
        Key word arguments that will be passed to the scaleColor function.
    doRemapBounds : `Callable`
        Remap the input image such that all values are between 0 and 1, either
        by scaling values down, or determining a reasonable maximum from the
        input if the inputs are all less than 1. Scaling is done by mapping
        the 98th quantile to the value of 2 which is then put though an arctan
        transformation.

    Returns
    -------
    result : `NDArray`
        The brightness and color calibrated image.

    Raises
    ------
    ValueError
        Raised if the shapes of the input array do not match
    """
    if rArray.shape != gArray.shape or rArray.shape != bArray.shape:
        raise ValueError("The shapes of all the input arrays must be the same")

    img = np.empty((*rArray.shape, 3))
    img[:, :, 0] = rArray
    img[:, :, 1] = gArray
    img[:, :, 2] = bArray
    img[img < 0] = 0
    img[np.isnan(img)] = 0

    if remapBounds is not None:
        img = mapUpperBounds(img, **(remapBoundsKwargs or {}))

    # Lab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2Lab)
    # Lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(img))
    #Lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(img, colourspace="Display P3"))
    # d65 =(0.31373, 0.32903 
    Lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(img, colourspace="CIE RGB", illuminant=np.array(cieWhitePoint), chromatic_adaptation_transform="bradford"))

    lum = Lab[:, :, 0]
    if doLocalContrast:
        newLum = localContrast(lum, sigma, highlights, shadows, clarity=clarity)
    else:
        newLum = lum
    if scaleLum is not None:
        lRemapped = scaleLum(newLum, **(scaleLumKWargs or {}))
    else:
        lRemapped = newLum
    #if doLocalContrast:
    #    lumMax = lRemapped.max()
    #    lRemapped = localContrast(lRemapped, sigma, alpha, beta, clarity=clarity)
    #    lRemapped = lRemapped/lRemapped.max() * lumMax
    if scaleColor is not None:
        # cFactor = scaleColor(lum, lRemapped, chroma, **(scaleColorKWargs or {}))
        new_a, new_b = scaleColor(
            lum, 0.8 * lRemapped, Lab[:, :, 1], Lab[:, :, 2], **(scaleColorKWargs or {})
        )
        Lab[:, :, 1] = new_a
        Lab[:, :, 2] = new_b

    Lab[:, :, 0] = lRemapped

    # the above transforms can produce values in Lab space that are not able to
    # be represented in rgb space. What follows is finding the equal lightness
    # plane
    xyz_prime = colour.Lab_to_XYZ(Lab)
    rgb_prime = colour.XYZ_to_RGB(xyz_prime, colourspace="Display P3")
    outOfBounds = np.bitwise_or(rgb_prime[:, :, 0] >= 1, rgb_prime[:, :, 1] >= 1, rgb_prime[:, :, 2] >= 1)
    if not np.any(outOfBounds):
        result = colour.XYZ_to_RGB(colour.Lab_to_XYZ(Lab), colourspace="Display P3")

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
            for pa, pb in ((prop_a, prop_b), (prop_a2, prop_b2)):
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
    result = colour.XYZ_to_RGB(colour.Lab_to_XYZ(Lab), colourspace="Display P3")

    # explicitly cut at 1 even though the mapping above was to map colors
    # appropriately because the Z matrix transform can produce values above
    # 1 and is a known feature of the transform.
    result[result > 1] = 1
    result[result < 0] = 0
    return result
