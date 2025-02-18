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
from scipy.stats import median_abs_deviation
from typing import TYPE_CHECKING, Generator
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.interpolate import griddata, RBFInterpolator

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from collections.abc import Mapping

    PLUGIN_TYPE = Callable[[NDArray, NDArray, Mapping[str, int]], NDArray]


class PluginType(Enum):
    """Enumeration to mark the type of data a plugin expects to work on"""

    CHANNEL = auto()
    """A plugin of this type expects to work on an individual channel
    from a partial region of a mosaic, such as a `patch`.
    """
    PARTIAL = auto()
    """A pluging that expects to work on a 3 channel image that is
    a partial region of a mosaic, such as a `patch`.
    """
    FULL = auto()
    """FULL plugins operate on a 3 channel image corresponding to
    a complete mosaic.
    """


class PluginsRegistry:
    """A class to serve as a registry for all pretty picture manipulation
    plugins.

    This class should not be instantiated directly other than the one
    instantiation in this module.

    example
    -------
    Using this registry to create a plugin would look somehting like the
    following.

    @plugins.register(1, PluginType.PARTIAL)
    def fixNoData(
        image: NDArray,
        mask: NDArray,
        maskDict: Mapping[str, int]
        ) -> NDArray:
        m = (mask & 2 ** maskDict["NO_DATA"]).astype(bool)
        for i in range(3):
            image[:, :, i] = cv2.inpaint(
                image[:, :, i].astype(np.float32),
                m.astype(np.uint8),
                3,
                cv2.INPAINT_TELEA
            ).astype(image.dtype)
        return image

    """

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
        """Register a plugin which is to be run when producing a
        pretty picture.

        parameters
        ----------
        order : `float`
            This determines in what order plugins will be run. For
            example, if plugin A specifies order 2, and plugin B
            specifies order 1, and both are the same ``kind`` of
            plugin type, plugin B will be run before plugin A.
        kind : `PluginType`
            This specifies what data the registered plugin expects
            to run on, a channel, a partial image, or a full mosaic.

        """

        def wrapper(
            func: Callable[[NDArray, NDArray, Mapping[str, int]], NDArray],
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
"""
This is the only instance of the plugin registry there should be. Users
should import from here and use the register method as a decorator to
register any plugins. Or, preferably, add them to this file to avoid
needing any other import time logic elsewhere.
"""

def neg_log_likelihood(params, x):
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    M = np.max(x)
    if mu < M - 1e-8:  # Allow for floating point precision issues
        return np.inf
    loglikelihood = 0.0
    z = (x - mu) / sigma
    term = np.log(2) - np.log(sigma) + norm.logpdf(z)
    loglikelihood = np.sum(term)
    return -loglikelihood

def tile_slices(arr, R, C):
    M = arr.shape[0]
    N = arr.shape[1]

    # Function to compute slices for a given dimension size and number of divisions
    def get_slices(total_size, num_divisions):
        base = total_size // num_divisions
        remainder = total_size % num_divisions
        slices = []
        start = 0
        for i in range(num_divisions):
            end = start + base
            if i < remainder:
                end += 1
            slices.append((start, end))
            start = end
        return slices

    # Get row and column slices
    row_slices = get_slices(M, R)
    col_slices = get_slices(N, C)

    # Generate all possible tile combinations of row and column slices
    tiles = []
    for rs in row_slices:
        r_start, r_end = rs
        for cs in col_slices:
            c_start, c_end = cs
            tile_slice = (slice(r_start, r_end), slice(c_start, c_end))
            tiles.append(tile_slice)

    return tiles


@plugins.register(1, PluginType.CHANNEL)
def fixBackground(image, mask, maskDict):
    maxLikely = np.median(image, axis=None)

    mask = image < maxLikely
    initial_std = (image[mask] - maxLikely).std()

    if len(image[mask]) > 0:
        result = minimize(neg_log_likelihood, (maxLikely, initial_std), args=(image[mask]), bounds=((maxLikely, None), (1e-8, None)))
        mu_hat, sigma_hat = result.x
    else:
        mu_hat, sigma_hat = (maxLikely, 2*initial_std)
    threshhold = mu_hat + sigma_hat
    image_mask = image < threshhold

    tiles = tile_slices(image, 25, 25)

    yloc = []
    xloc = []
    values = []

    for (xslice, yslice) in tiles:
        ypos = (yslice.stop - yslice.start)/2 + yslice.start
        xpos = (xslice.stop - xslice.start)/2 + xslice.start
        yloc.append(ypos)
        xloc.append(xpos)
        value = np.median(image[yslice, xslice][image_mask[yslice, xslice]])
        if np.isnan(value):
            value = 0
        values.append(value)

    positions = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    inter = RBFInterpolator(np.vstack((yloc, xloc)).T, values, kernel='thin_plate_spline', degree=4, smoothing=0.05)
    backgrounds = inter(np.array(positions)[::-1].reshape(2,-1).T).reshape(image.shape)

    return backgrounds
