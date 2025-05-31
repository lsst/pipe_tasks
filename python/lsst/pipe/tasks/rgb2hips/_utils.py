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
    uri = hips_dir.join(f"Npix{pixel_id}.{file_extension}")

    # Finally, need to turn the array into an image
    im = Image.fromarray(image_data, mode="RGB")

    extra_args = {}
    if file_extension == "webp":
        extra_args["lossless"] = True
        extra_args["quality"] = 80
    with ResourcePath.temporary_uri(suffix=uri.getExtension()) as temporary_uri:
        im.save(temporary_uri.ospath, **extra_args)

        uri.transfer_from(temporary_uri, transfer="copy", overwrite=True)
