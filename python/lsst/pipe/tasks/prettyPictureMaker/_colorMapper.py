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

__all__ = ("lsstRGB",)

import numpy as np
import skimage

from .types import (
    FloatImagePlane,
    RGBImage,
    LocalContrastFunction,
    ScaleLumFunction,
    ScaleColorFunction,
    RemapBoundsFunction,
    BracketingFunction,
    GamutRemappingFunction,
)

from ._functors import (
    LocalContrastEnhansor,
    LumCompressor,
    ColorScaler,
    BoundsRemapper,
    ExposureBracketer,
    GamutFixer,
)


from lsst.rubinoxide import rgb


class _SentinalDefault:
    pass


default = _SentinalDefault()


def lsstRGB(
    rArray: FloatImagePlane,
    gArray: FloatImagePlane,
    bArray: FloatImagePlane,
    local_contrast: LocalContrastFunction | None | _SentinalDefault = default,
    scale_lum: ScaleLumFunction | None | _SentinalDefault = default,
    scale_color: ScaleColorFunction | None | _SentinalDefault = default,
    remap_bounds: RemapBoundsFunction | None | _SentinalDefault = default,
    bracketing_function: BracketingFunction | None | _SentinalDefault = default,
    gamut_remapping_function: GamutRemappingFunction | None | _SentinalDefault = default,
    psf: FloatImagePlane | None = None,
    cieWhitePoint: tuple[float, float] = (0.28, 0.28),
) -> RGBImage:
    """Enhance the lightness and color preserving hue using perceptual methods.

    Parameters
    ----------
    rArray : `NDArray`
        The array used as the red channel
    gArray : `NDArray`
        The array used as the green channel
    bArray : `NDArray`
        The array used as the blue channel
    local_contrast : `LocalContrastFunction` or `None`
        This is a callable that's passed the luminance values, and is expected
        to do local contrast enhcment. Set to None to bypass.
    scale_lum : `ScaleLumFunction` or `None`
        This is a callable that's passed the luminance values and should
        return a scaled luminance array the same shape as the input.
        Set to None for no scaling.
    scale_color : `ScaleColorFunction` or `None`
        This is a callable that's passed the original luminance, the remapped
        luminance values, the a values for each pixel, and the b values for
        each pixel. This function is responsible for scaling chroma
        values. This should return two arrays corresponding to the scaled a and
        b values. Set to None for no modification.
    remap_bounds : `RemapBoundsFunction` or `None`
        This is a callable that remaps the input arrays such that each of
        them fall within a zero to one range. This callable is given the
        initial image. Set to None for no remapping.
    bracketing_function : `BracketingFunction` or `None`
        This is a callable that is passed the input luminance, and should
        create various exposure levels and then fuse them together.
        Set to None for no bracketing.
    gamut_remapping_function : `GamutRemappingFunction` or `None`
        This is a callable that is passed the final OkLab image. It's job
        is to detect and correct any pixel values that would fall outside
        an RGB P3 colorspace. Set to None for no fixes.
    psf : `NDArray` or `None`
        If this parameter is an image of a PSF kernel the luminance channel is
        deconvolved with it. Set to None to skip deconvolution.
    cieWhitePoint : `tuple` of `float`, `float`
        This is the white point of the input of the input arrays in CIE XY
        coordinates. Altering this affects the relative balance of colors
        in the input image, and therefore also the output image.

    Returns
    -------
    result : `RGBImage`
        The brightness and color calibrated image.

    Raises
    ------
    ValueError
        Raised if the shapes of the input array don't match
    """

    # Default construct functors to be used as callables
    if local_contrast is default:
        local_contrast = LocalContrastEnhansor()
    if scale_lum is default:
        scale_lum = LumCompressor()
    if scale_color is default:
        scale_color = ColorScaler()
    if remap_bounds is default:
        remap_bounds = BoundsRemapper()
    if bracketing_function is default:
        bracketing_function = ExposureBracketer()
    if gamut_remapping_function is default:
        gamut_remapping_function = GamutFixer()

    # Validate inputs
    if rArray.shape != gArray.shape or rArray.shape != bArray.shape:
        raise ValueError("The shapes of all the input arrays must be the same")

    # Construct a new image array in the proper byte ordering.
    img: RGBImage = np.empty((*rArray.shape, 3))
    img[:, :, 0] = rArray
    img[:, :, 1] = gArray
    img[:, :, 2] = bArray
    # If there are nan's in the image there is no real option other than to
    # set them to zero or throw.
    img[np.isnan(img)] = 0

    import time

    t1 = time.time()

    if remap_bounds is not None:
        img = remap_bounds(img)
        print(f"doing remap took {time.time() - t1}s")
        t1 = time.time()

    # Convert the starting image into the OK L*a*b* color space.
    # https://en.wikipedia.org/wiki/Oklab_color_space
    Lab = rgb.RGB_to_Oklab(img, cieWhitePoint)
    print(f"lab conversion took {time.time() - t1}")
    t1 = time.time()
    lum = Lab[:, :, 0]

    # potentially needed for remapping color, so save what it origionaly was
    lum_save = np.copy(lum)

    if bracketing_function is not None:
        lum = bracketing_function(lum)
        print(f"bracketing took {time.time() - t1}")
        t1 = time.time()

    if scale_lum is not None:
        lum = scale_lum(lum)
        print(f"lum scale took {time.time() - t1}")
        t1 = time.time()

    if local_contrast is not None:
        lum = local_contrast(lum)
        print(f"local_contrast took {time.time() - t1}")
        t1 = time.time()

    if psf is not None:
        lum = skimage.restoration.richardson_lucy(lum, psf=psf, clip=False, num_iter=2)
        print(f"psf took {time.time() - t1}")
        t1 = time.time()

    if scale_color is not None:
        new_a, new_b = scale_color(lum_save, lum, Lab[..., 1], Lab[..., 2])
        Lab[..., 1] = new_a
        Lab[..., 2] = new_b
        print(f"color correction took {time.time() - t1}")
        t1 = time.time()
    Lab[..., 0] = lum

    # The target output profile whitepoint
    cie_white_point_d65 = (0.31272, 0.32903)
    if gamut_remapping_function is not None:
        result = gamut_remapping_function(Lab, cie_white_point_d65)
        print(f"gamut fixing took {time.time() - t1}")
        t1 = time.time()
    else:
        result = rgb.Oklab_to_RGB(np.ascontiguousarray(Lab), cie_white_point_d65)
        print(f"RGB conversion took {time.time() - t1}")
        t1 = time.time()

    result = np.clip(result, 0, 1)
    return result
