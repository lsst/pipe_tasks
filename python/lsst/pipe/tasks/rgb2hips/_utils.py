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

from __future__ import annotations

__all__ = ("_write_hips_image",)

from PIL import Image
import numpy as np
from numpy.typing import NDArray


from lsst.resources import ResourcePath


# allow PIL to work with really large images
Image.MAX_IMAGE_PIXELS = None


def _get_dir_number(pixel: int) -> int:
    """Compute the directory number from a pixel.

    Parameters
    ----------
    pixel : `int`
        HEALPix pixel number.

    Returns
    -------
    dir_number : `int`
        HiPS directory number.
    """
    return (pixel // 10000) * 10000


def _write_hips_image(
    image_data: NDArray,
    pixel_id: int,
    hpx_level: int,
    hips_base_path: ResourcePath,
    file_extension: str,
    output_type: str,
) -> None:
    """Write a processed image to disk in the HealPix tile format.

    This function takes processed image data, converts it to the specified output
    type, and saves it into the appropriate directory structure based on the HealPix
    pixel ID and order level.

    Parameters
    ----------
    image_data : `NDArray`
        The RGB image data array to be written as a HealPix tile.
    pixel_id : `int`
        The unique HealPix ID corresponding to the output tile.
    hpx_level : `int`
        The HealPix order level of the output tile.
    hips_base_path : `ResourcePath`
        Base directory path where the HealPix tiles will be stored.
    file_extension : `str`
        File extension (format) for saving the image ('png' or 'webp').
    output_type : `str`
        Data type of the output array, which can be:
            - "uint8": 8-bit unsigned integers (0-255)
            - "uint16": 16-bit unsigned integers (0-65535)
            - "half": 16-bit floating-point numbers
            - "float": 32-bit floating-point numbers

    """
    # clip in case any of the warping caused values over 1
    image_data = np.clip(image_data, 0, 1)
    # remap the image_data to the chosen output_type
    match output_type:
        case "uint8":
            image_data = (image_data * 255.0).astype(np.uint8)
        case "uint16":
            image_data = (image_data * 65535.0).astype(np.uint16)
        case "half":
            image_data = image_data.astype(np.float16)
        case "float":
            pass

    # mangle the URI where to write
    dir_number = _get_dir_number(pixel_id)
    hips_dir = hips_base_path.join(f"Norder{hpx_level}", forceDirectory=True).join(
        f"Dir{dir_number}", forceDirectory=True
    )

    # Create the file URI for saving
    uri = hips_dir.join(f"Npix{pixel_id}.{file_extension}")

    # Convert numpy array to PIL Image and save with appropriate arguments
    im = Image.fromarray(image_data, mode="RGB")

    extra_args = {}
    if file_extension == "webp":
        # Set WebP-specific parameters for lossless compression
        extra_args["lossless"] = True
        extra_args["quality"] = 80

    # Save the image to a temporary file and transfer to final location
    with ResourcePath.temporary_uri(suffix=uri.getExtension()) as temporary_uri:
        im.save(temporary_uri.ospath, **extra_args)
        uri.transfer_from(temporary_uri, transfer="copy", overwrite=True)
