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

__all__ = ("FeatheredMosaicCreator",)

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from lsst.images import Box


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
    spanning `2 * patch_grow` pixels at each edge. The masks are stored in
    `self.featherings` as a list of arrays in the order:
    [top, bottom, left, right].
    """

    featherings: list[np.ndarray] | None

    def __init__(self, patch_grow: int, bin_factor: int = 1) -> None:
        """Initialize the feathering creator.

        Parameters
        ----------
        patch_grow : `int`
            Number of pixels to grow the patch boundaries for feathering.
        bin_factor : `int`, optional
            Binning factor for the feathering calculation. Reduces resolution
            of feathering masks for faster computation. Default is 1.
        """
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
        linear ramps from 0 to 1 (or 1 to 0) spanning `2 * patch_grow` pixels at
        each edge. The featherings are stored in `self.featherings` as a list of
        arrays.
        """
        extent = self.patch_grow * 2
        if self.bin_factor != 1:
            extent = int(np.floor(extent / self.bin_factor))
        ramp = np.linspace(0, 1, extent)
        # Replace the exact zero at the outer edge of the feathering ramp. The
        # ramp is used as a multiplicative mask (see `add_to_image`), so a hard
        # 0 would blend the boundary pixel to pure black, producing a visible
        # seam. A tiny nonzero keeps that pixel effectively transparent while
        # avoiding the exact-zero value.
        ramp[0] = 1e-17
        top = np.ones(dimensions)
        top[:extent, :] = np.repeat(np.expand_dims(ramp, 1), top.shape[1], axis=1)

        bottom = np.ones(dimensions)
        bottom[-1 * extent :, :] = np.repeat(np.expand_dims(1 - ramp, 1), bottom.shape[1], axis=1)

        left = np.ones(dimensions)
        left[:, :extent] = np.repeat(np.expand_dims(ramp, 0), left.shape[0], axis=0)

        right = np.ones(dimensions)
        right[:, -1 * extent :] = np.repeat(np.expand_dims(1 - ramp, 0), right.shape[0], axis=0)
        self.featherings = [
            top,
            bottom,
            left,
            right,
        ]

    def add_to_image(
        self,
        image: NDArray,
        patch: NDArray,
        new_box: Box,
        box: Box,
        reverse: bool = True,
        weight: NDArray | None = None,
    ) -> None:
        """Add a patch to an image with feathering at the edges.

        Parameters
        ----------
        image : `NDArray`
            Target image to which the patch will be added. Modified in-place.
            When ``weight`` is provided this holds the un-normalized weighted sum
            of the contributions; call :meth:`finalize` to normalize it.
        patch : `NDArray`
            Patch array to be added to the image.
        new_box : `lsst.images.Box`
            New bounding box position of the patch.
        box : `lsst.images.Box`
            Original bounding box position of the patch.
        reverse : `bool`, optional
            If True, reverse the patch along the first axis before adding.
            Default is True.
        weight : `NDArray`, optional
            Same-shaped accumulator that receives the feather weight of each
            contribution. When provided, pixels of ``patch`` that are NaN are
            excluded (their data contributes 0 and their weight contributes 0)
            so a NaN in one patch does not corrupt the blend. Must be zero
            initialized by the caller. Default is None (plain feathered sum).

        Notes
        -----
        This method applies feathering to smoothly blend the patch into the image
        at the edges where the patch overlaps with existing image content. The
        feathering is applied based on which edges of the patch differ between
        `box` and `new_box`. The patch is multiplied by a mixer array that gradually
        transitions from 0 to 1 across the feathering region.

        When ``weight`` is given, call :meth:`finalize` after all contributions
        have been added to obtain the normalized (NaN-reweighted) blend.
        """
        base_shape = patch.shape if patch.ndim == 2 else patch.shape[:2]
        mixer = np.ones(base_shape)
        if self.featherings is None:
            self._make_featherings(base_shape)
        assert self.featherings is not None
        if box.y.start != new_box.y.start:
            mixer *= self.featherings[0]
        if box.y.stop != new_box.y.stop:
            mixer *= self.featherings[1]
        if box.x.start != new_box.x.start:
            mixer *= self.featherings[2]
        if box.x.stop != new_box.x.stop:
            mixer *= self.featherings[3]

        if image.ndim > 2:
            mixer = np.repeat(np.expand_dims(mixer, 2), 3, axis=2)

        slice_ = (slice(box.y.start, box.y.stop), slice(box.x.start, box.x.stop))
        if reverse:
            slice_ += (slice(None),) if image.ndim > 2 else ()

        if weight is None:
            image[slice_] += (mixer * patch)[::-1] if reverse else mixer * patch
            return

        finite = np.isfinite(patch)
        clean = np.where(finite, patch, 0.0)
        image[slice_] += (mixer * clean)[::-1] if reverse else mixer * clean
        weight[slice_] += (mixer * finite)[::-1] if reverse else mixer * finite

    @staticmethod
    def finalize(image: NDArray, weight: NDArray) -> NDArray:
        """Normalize a weighted-sum image into a NaN-reweighted blend.

        Each pixel becomes ``sum(weight_i * data_i) / sum(weight_i)``. Pixels with
        no valid contribution (total weight zero) are set to NaN, the missing-data
        marker.

        Parameters
        ----------
        image : `NDArray`
            The weighted-sum image built by :meth:`add_to_image`. Modified in-place.
        weight : `NDArray`
            The weight accumulator built by :meth:`add_to_image`.

        Returns
        -------
        image : `NDArray`
            The normalized image.
        """
        mask = weight > 0
        image[mask] /= weight[mask]
        image[~mask] = np.nan
        return image
