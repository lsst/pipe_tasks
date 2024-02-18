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

__all__ = ("PluginsRegistry", "plugins")

from enum import Enum, auto
from collections.abc import Callable
from typing import TYPE_CHECKING, Generator
import numpy as np
from sklearn.cluster import KMeans
import cv2

from lsst.afw.image import Mask

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from collections.abc import Mapping
    PLUGIN_TYPE = Callable[[NDArray, NDArray, Mapping[str, int]], NDArray]


class PluginType(Enum):
    CHANNEL = auto()
    PARTIAL = auto()
    FULL = auto()


class PluginsRegistry:
    def __init__(self) -> None:
        self._full_values: list[tuple[float, PLUGIN_TYPE]] = []
        self._partial_values: list[tuple[float, PLUGIN_TYPE]] = []
        self._channel_values: list[tuple[float, PLUGIN_TYPE]] = []

    def channel(self) -> Generator[PLUGIN_TYPE, None, None]:
        yield from (func for _, func in self._channel_values)

    def partial(self) -> Generator[PLUGIN_TYPE, None, None]:
        yield from (func for _, func in self._partial_values)

    def full(self) -> Generator[PLUGIN_TYPE, None, None]:
        yield from (func for _, func in self._full_values)

    def register(self, order: float, kind: PluginType) -> Callable:
        def wrapper(
                func: Callable[[NDArray, NDArray, Mapping[str, int]], NDArray]
        ) -> Callable[[NDArray, NDArray, Mapping[str, int]], NDArray]:
            match kind:
                case PluginType.PARTIAL:
                    self._partial_values.append((order, func))
                case PluginType.FULL:
                    self._full_values.append((order, func))
                case PluginType.CHANNEL:
                    self._channel_values.append((order, func))
            return func

        return wrapper


plugins = PluginsRegistry()


# @plugins.register(100, PluginType.FULL)
def externalProcessing(image: NDArray, mask: NDArray) -> NDArray:
    return np.array((0, 0))


# @plugins.register(1, PluginType.CHANNEL)
def fixBackgrounds(image: NDArray, fullMask: NDArray, maskDict: Mapping[str, int]) -> NDArray:
    yind, xind = np.indices(image.shape)
    mask = ~(((fullMask & 2 ** Mask.getMaskPlane("DETECTED")) > 0).astype(bool))  # type: ignore
    number = np.sum(mask)
    container = np.empty((number, 3))  # type: ignore
    container[:, 0] = yind[mask]
    container[:, 1] = xind[mask]

    for _ in range(2):
        container[:, 2] = image[mask]

        kmeans = KMeans(init="random", n_clusters=100)
        kmeans.fit(container)

        what = np.zeros((image.shape), dtype=float)
        what[yind[mask], xind[mask]] = kmeans.cluster_centers_[:, 2][kmeans.labels_] * 0.9

        image = np.abs(image - what)
    return image


# @plugins.register(2, PluginType.PARTIAL)
def fixChromaticStars(image: NDArray, mask: NDArray, maskDict: Mapping[str, int]) -> NDArray:
    return np.array((0, 0))


@plugins.register(1, PluginType.PARTIAL)
def fixNoData(image: NDArray, mask: NDArray, maskDict: Mapping[str, int]) -> NDArray:
    print("running mask fixup")
    m = (mask & 2**maskDict['NO_DATA']).astype(bool)
    print("done making mask")
    # loop over arrays and apply, this supports more than 8 bit arrays
    for i in range(3):
        image[:, :, i] = cv2.inpaint(image[:, :, i], m.astype(image.dtype), 3, cv2.INPAINT_TELEA)
    print("done mask fixup")
    return image
