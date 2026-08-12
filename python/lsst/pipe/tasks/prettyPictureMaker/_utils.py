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

__all__ = ("FeatheredMosaicCreator", "write_array_png", "write_array_exr", "write_array_hdr_avif")

import numpy as np
import imagecodecs
from typing import TYPE_CHECKING

from .types import RGBImage


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from lsst.geom import Box2I


def _linear_to_pq(linear_data, lum_norm=1000.0):
    """Converts linear data (where 1.0 is diffuse white) into PQ space."""
    # Convert linear data to nits and normalize to [0, 1]
    Y = (linear_data * 1) / lum_norm
    if Y.max() > 1.0:
        raise RuntimeError("The normalized data excedes a value of 1")

    # SMPTE ST 2084 constants
    m1 = 2610 / 4096 / 4
    m2 = 2523 / 4096 * 128
    c1 = 3424 / 4096
    c2 = 2413 / 4096 * 32
    c3 = 2392 / 4096 * 32

    Y_pow = np.power(Y, m1)
    numerator = c1 + c2 * Y_pow
    denominator = 1 + c3 * Y_pow

    return np.round(np.power(numerator / denominator, m2) * 1023).astype(np.uint16)


def write_array_png(location: str, array: RGBImage) -> None:
    if not location.endswith(".png"):
        raise ValueError("Output location should have png extension")
    if not (array.dtype == np.uint8 or array.dtype == np.uint16):
        raise TypeError("Png image must be 8 bit or 16 bit int")
    imagecodecs.imwrite(location, array)


def write_array_exr(location: str, array: RGBImage) -> None:
    if not location.endswith(".exr"):
        raise ValueError("Output location should have exr extension")
    if not (array.dtype == np.float32):
        raise TypeError("EXR image must be 32bit float")

    exr_bytes = imagecodecs.exr_encode(array, compression="ZIP")

    with open(location, "wb") as f:
        f.write(exr_bytes)


def write_array_hdr_avif(location: str, array: RGBImage, clamp_values: float | None = None) -> None:
    if not location.endswith(".avif"):
        raise ValueError("Output location should have avif extension")

    if clamp_values is not None:
        array = np.clip(array, 0, clamp_values)

    pq_data = _linear_to_pq(array, 100)
    encoded_avif = imagecodecs.avif_encode(
        pq_data,
        primaries=9,
        transfer=16,
        matrix=9,
        bitspersample=10,
        level=90,
        speed=8,
        numthreads=3,
        pixelformat=2,
    )
    with open(location, "wb") as f:
        f.write(encoded_avif)


class FeatheredMosaicCreator:
    """Create feathering masks for seamless image patch blending.

    This class generates feathering masks used to smoothly blend image patches
    into a larger mosaic. The feathering gradually transitions from full opacity
    to zero opacity across the patch boundaries, reducing visible seams.

    Parameters
    ----------
    patch_grow : `int`
        Number of pixels to grow the patch boundaries for feathering.
    bin_factor : `int`, optional
        Binning factor for the feathering calculation. Reduces resolution
        of feathering masks for faster computation. Default is 1.

    Notes
    -----
    The feathering masks are created as linear ramps from 0 to 1 (or 1 to 0)
    over the `patch_grow` distance. The masks are stored in `self.featherings`
    as a list of arrays in the order: [top, bottom, left, right].
    """

    def __init__(self, patch_grow: int, bin_factor: int = 1) -> None:
        self.patch_grow = patch_grow
        self.bin_factor = bin_factor
        self.featherings = None

    def _make_featherings(self, dimensions: tuple[int, int]) -> None:
        """Create feathering masks for all edges of the patch.

        Parameters
        ----------
        dimensions : `tuple` of `int`
            Shape of the patch (height, width) for which to create feathering masks.

        Notes
        -----
        This method creates four feathering masks (top, bottom, left, right) using
        linear ramps from 0 to 1 (or 1 to 0) over the `patch_grow` distance. The
        featherings are stored in `self.featherings` as a list of arrays.
        """
        extent = self.patch_grow * 2
        if self.bin_factor != 1:
            extent = int(np.floor(extent / self.bin_factor))
        ramp = np.linspace(0, 1, extent)
        ramp[0] = 1e-17
        top = np.ones(dimensions)
        top[:extent, :] = np.repeat(np.expand_dims(ramp, 1), top.shape[1], axis=1)

        bottom = np.ones(dimensions)
        bottom[-1 * extent :, :] = np.repeat(  # noqa: E203
            np.expand_dims(1 - ramp, 1), bottom.shape[1], axis=1
        )

        left = np.ones(dimensions)
        left[:, :extent] = np.repeat(np.expand_dims(ramp, 0), left.shape[0], axis=0)

        right = np.ones(dimensions)
        right[:, -1 * extent :] = np.repeat(  # noqa: E203
            np.expand_dims(1 - ramp, 0), right.shape[0], axis=0
        )
        self.featherings = [
            top,
            bottom,
            left,
            right,
        ]

    def add_to_image(
        self, image: NDArray, patch: NDArray, newBox: Box2I, box: Box2I, reverse: bool = True
    ) -> None:
        """Add a patch to an image with feathering at the edges.

        Parameters
        ----------
        image : `NDArray`
            Target image to which the patch will be added. Modified in-place.
        patch : `NDArray`
            Patch array to be added to the image.
        newBox : `Box2I`
            New bounding box position of the patch.
        box : `Box2I`
            Original bounding box position of the patch.
        reverse : `bool`, optional
            If True, reverse the patch along the first axis before adding.
            Default is True.

        Notes
        -----
        This method applies feathering to smoothly blend the patch into the image
        at the edges where the patch overlaps with existing image content. The
        feathering is applied based on which edges of the patch differ between
        `box` and `newBox`. The patch is multiplied by a mixer array that gradually
        transitions from 0 to 1 across the feathering region.
        """
        base_shape = patch.shape if patch.ndim == 2 else patch.shape[:2]
        mixer = np.ones(base_shape)
        if self.featherings is None:
            self._make_featherings(base_shape)
        if box.getBeginY() != newBox.getBeginY():
            mixer *= self.featherings[0]
        if box.getEndY() != newBox.getEndY():
            mixer *= self.featherings[1]
        if box.getBeginX() != newBox.getBeginX():
            mixer *= self.featherings[2]
        if box.getEndX() != newBox.getEndX():
            mixer *= self.featherings[3]

        if image.ndim > 2:
            mixer = np.repeat(np.expand_dims(mixer, 2), 3, axis=2)

        patch = mixer * patch

        image[*box.slices] += patch[::-1, :, :] if reverse else patch


def highlight_taper(
    x: np.ndarray, max_val: float = 1.0, knee_ratio: float = 0.9, width: float = 0.1
) -> np.ndarray:
    """
    Applies a highlight roll-off (taper) to input values, smoothly compressing
    values above a configurable threshold towards a maximum limit.

    This function is designed to handle both Standard Dynamic Range (SDR) and
    High Dynamic Range (HDR) workflows. It preserves linearity below a specified
    "knee" point and applies a rational function to smoothly asymptote towards
    `max_val` for values exceeding that point.

    Parameters
    ----------
    x : array_like
        Input pixel values Values can exceed `max_val` (e.g., HDR data), but
        the output will be capped at `max_val`.
    max_val : float, optional
        The target maximum value.
        - For SDR (0-1 range): typically `1.0`.
        - For HDR (e.g., 0-1000 nits): e.g., `10.0`.
        Default is 1.0.
    knee_ratio : float, optional
        The fraction of `max_val` where the tapering begins.
        The absolute knee point is calculated as `max_val * knee_ratio`.
        - `0.9`: Tapering starts late (highlights remain bright longer).
        - `0.7`: Tapering starts earlier (more aggressive compression).
        Must be between 0.0 and 1.0. Default is 0.9.
    width : float, optional
        Controls the steepness (smoothness) of the transition curve.
        - Small values (e.g., 0.01) create a sharp, near-hard clip.
        - Larger values (e.g., 0.5) create a smooth, gradual roll-off.
        This is an absolute value, not a ratio. Default is 0.1.

    Returns
    -------
    np.ndarray
        The tapered output values with the same shape as `x`.
        Values will be in the range `[min(x), max_val]`.

    Raises
    ------
    ValueError
        If `knee_ratio` is not between 0.0 and 1.0, or if `max_val` or `width`
        are non-positive.

    Notes
    -----
    The function uses the following logic:
    1. If `x <= knee`, output `y = x` (Linear).
    2. If `x > knee`, output `y = knee + (max_val - knee) * (excess / (excess + width))`,
       where `excess = x - knee`.

    This rational function ensures that as `x` approaches infinity, `y` approaches
    `max_val` asymptotically, preventing hard clipping artifacts.

    """
    # Validate inputs
    if not 0.0 < knee_ratio <= 1.0:
        raise ValueError("knee_ratio must be between 0.0 (exclusive) and 1.0 (inclusive).")
    if max_val <= 0:
        raise ValueError("max_val must be positive.")
    if width <= 0:
        raise ValueError("width must be positive.")

    # Convert input to numpy array for vectorization
    x = np.asarray(x, dtype=float)

    # Calculate the absolute knee point
    knee = max_val * knee_ratio

    # Initialize output array
    y = np.empty_like(x)

    # Create a mask for values above the knee
    mask = x > knee

    # Apply linear mapping for values below or equal to knee
    y[~mask] = x[~mask]

    # Apply rational taper for values above knee
    if np.any(mask):
        excess = x[mask] - knee
        # Formula: knee + (max_val - knee) * (excess / (excess + width))
        # As excess -> infinity, the fraction -> 1, so y -> max_val
        y[mask] = knee + (max_val - knee) * (excess / (excess + width))

    return y
