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

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from lsst.geom import Box2I


class FeatheredMosaicCreator:
    def __init__(self, patch_grow: int, bin_factor: int = 1):
        self.patch_grow = patch_grow
        self.bin_factor = bin_factor
        self.featherings = None

    def _make_featherings(self, dimensions: tuple[int, int]) -> None:
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

        image[*box.slices] += patch[:, :, ::-1] if reverse else patch
