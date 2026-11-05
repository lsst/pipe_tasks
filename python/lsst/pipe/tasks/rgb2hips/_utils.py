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

__all__ = ("_area_downsample", "downsample_hips_array", "_write_hips_image")

import cv2
from PIL import Image
import numpy as np
from numpy.typing import NDArray


from lsst.resources import ResourcePath


# allow PIL to work with really large images
Image.MAX_IMAGE_PIXELS = None


def _area_downsample(image: NDArray, out_hw: tuple[int, int]) -> NDArray:
    """Exact box-average downsample via the stellaRGB.warp AREA kernel (D10).

    The proper pyramid/downsample kernel, replacing the previous
    ``cv2.resize(..., INTER_LANCZOS4)`` step: a Lanczos resize has NO
    prefilter, so any energy above the output Nyquist aliases into the
    level as ringing; the area kernel is the proper prefilter (exact
    weighted box mean, verified against ``cv2.resize(INTER_AREA)`` and
    bit-exact 2x2 box averages in the engine's parity suite). Lanczos
    remains selectable via the tasks' ``downsample_kernel`` knob for its
    edge-contrast (isophote) preservation.

    Destination pixel ``(y, x)`` is the clip-overlap box mean over the
    source box ``[fy*y - 0.5, fy*y + fy - 0.5] x [fx*x - 0.5, fx*x + fx -
    0.5]`` — exactly what the engine's D10 AREA kernel computes for the
    strided lattice ``map_x = fx*x + (fx-1)/2`` (whose per-axis diff-mean
    scale is ``fx``): integer factors reduce to uniform weight 1.0 on the
    ``f`` source pixels (touching taps carry exact 0.0 weight, the engine's
    support-exempt structure), non-integer (odd-dim) factors get the exact
    partial-overlap weights — cv2 lanczos has no defined box semantics
    there at all.

    ENGINE-FEASIBILITY NOTE (why the box mean is evaluated here in numpy,
    not by calling ``stellaRGB.warp.remap(kernel="area")`` per level): the
    engine's area tap window is the UNIFORM GLOBAL SPAN of the call
    ("area is tile-size only" — the whole-table ``AREA_TOP_CAP`` /
    ``AREA_WEIGHTS_CAP`` prescan). A whole-level 4096->2048 strided remap
    has per-axis spans of the full source (4098 taps) and trips those caps
    with ``ValueError``; at factor 8/16 it would fit the caps yet burn
    ``O(span^2 x dest)`` work (the bench's affine-area hang lesson). The
    pyramid therefore evaluates the IDENTICAL D10 box mean as a separable
    integral-image reduction (float64 accumulation, cast last). D10
    equivalence on grids where the engine call IS feasible (tile-sized)
    is pinned by ``tests/test_hips.py`` (this helper vs the engine
    ``remap(kernel="area")`` on the same strided lattice).

    NaN source pixels poison exactly the boxes touching them (matching the
    engine's D9 contract; ``run()`` zeroes NaN before pyramiding, so
    production is unaffected).

    Parameters
    ----------
    image : `numpy.ndarray`
        ``(H, W)`` or ``(H, W, 3)`` source array (any float dtype).
    out_hw : `tuple` of ``int``
        Output ``(height, width)``; must not exceed the source shape.

    Returns
    -------
    downsampled : `numpy.ndarray`
        Float32 box-averaged array of shape ``out_hw + image.shape[2:]``.
    """
    src_h, src_w = image.shape[:2]
    out_h, out_w = out_hw
    if out_h > src_h or out_w > src_w or out_h < 1 or out_w < 1:
        raise ValueError(
            f"out_hw {out_hw} must be a non-empty downsample of {image.shape[:2]}"
        )
    out = np.empty((out_h, out_w) + image.shape[2:], dtype=np.float32)

    def _reduce_axis(plane: NDArray, axis: int, out_len: int) -> NDArray:
        """Exact D10 clip-overlap box mean along one axis, float64.

        ``plane`` is 2-D; all per-destination-index vectors are broadcast
        along ``axis`` explicitly (a bare ``(out_len,)`` vector would
        broadcast against the LAST dimension and silently misalign the
        axis=0 reduction whenever the grid is square).
        """
        n = plane.shape[axis]
        f = n / out_len
        pos = np.arange(out_len, dtype=np.float64)
        lo = f * pos - 0.5
        hi = lo + f  # == f*pos + f - 0.5, the box edges (see above)
        # Full pixels are [ceil(lo + 0.5), floor(hi - 0.5)] inclusive; the
        # touching partials carry the sub-pixel overhang weights (never
        # reaching outside the array for this lattice: hi <= n - 0.5).
        first = np.ceil(lo + 0.5).astype(np.intp)
        last = np.floor(hi - 0.5).astype(np.intp)
        clean = np.nan_to_num(plane)
        pref = np.cumsum(clean, axis=axis)
        nan_pref = np.cumsum(np.isnan(plane), axis=axis)

        def _take(arr: NDArray, idx: NDArray) -> NDArray:
            return np.take(arr, np.clip(idx, 0, n - 1), axis=axis)

        def _b(vec: NDArray) -> NDArray:
            return vec[:, None] if axis == 0 else vec[None, :]

        wl = _b((first - 0.5) - lo)  # in [0, 1); partial pixel `first - 1`
        wr = _b(hi - (last + 0.5))  # in [0, 1); partial pixel `last + 1`
        has_l = _b(first > 0)
        has_r = _b(last + 1 < n)
        # sum of full pixels [first .. last] (empty window => 0):
        s = _take(pref, last) - np.where(has_l, _take(pref, first - 1), 0.0)
        s = s + np.where(has_l, wl * _take(clean, first - 1), 0.0)
        s = s + np.where(has_r, wr * _take(clean, last + 1), 0.0)
        # NaN poisons exactly the boxes whose window touched a NaN pixel.
        nan_box = _take(nan_pref, last) - np.where(has_l, _take(nan_pref, first - 1), 0.0)
        nan_box = nan_box + np.where(
            has_l & (wl > 0.0),
            _take(nan_pref, first - 1) - np.where(_b(first > 1), _take(nan_pref, first - 2), 0.0),
            0.0,
        ) + np.where(
            has_r & (wr > 0.0),
            _take(nan_pref, last + 1) - _take(nan_pref, last),
            0.0,
        )
        result = s / f
        return np.where(nan_box > 0.0, np.nan, result)

    planes = image if image.ndim == 3 else image[..., None]
    for ch in range(planes.shape[2]):
        plane = np.ascontiguousarray(planes[..., ch], dtype=np.float64)
        reduced = _reduce_axis(plane, axis=1, out_len=out_w)
        reduced = _reduce_axis(np.ascontiguousarray(reduced), axis=0, out_len=out_h)
        out[..., ch] = reduced
    return out if image.ndim == 3 else out[..., 0]


def downsample_hips_array(image: NDArray, out_hw: tuple[int, int], kernel: str) -> NDArray:
    """Downsample an RGB tile array with the configured pyramid kernel.

    Shared dispatch for the ``downsample_kernel`` config knob of
    ``HighOrderHipsTask`` (orders 10-8 tile pyramid and the 256x256
    butler array) and ``LowOrderHipsTask`` (the order-7 -> min_order
    cascade). "area" is the exact box-average prefilter (see
    ``_area_downsample``): flux-conserving, alias-free, and
    minimum-variance for white noise, but it smooths faint gradients
    (visible isophote erosion on low-surface-brightness edges).
    "lanczos" is the historical ``cv2.resize(..., INTER_LANCZOS4)``
    step: no prefilter, so it keeps edge contrast at the cost of
    ringing and passing ~4-7x more high-frequency noise power at 4-8x
    reduction.

    Parameters
    ----------
    image : `numpy.ndarray`
        ``(H, W)`` or ``(H, W, 3)`` float source array.
    out_hw : `tuple` of ``int``
        Output ``(height, width)``.
    kernel : `str`
        ``"area"`` or ``"lanczos"`` (validated upstream by the
        ``ChoiceField``).

    Returns
    -------
    downsampled : `numpy.ndarray`
        Float32 array of shape ``out_hw + image.shape[2:]``.
    """
    if kernel == "lanczos":
        return cv2.resize(image, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LANCZOS4)
    return _area_downsample(image, out_hw)


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
