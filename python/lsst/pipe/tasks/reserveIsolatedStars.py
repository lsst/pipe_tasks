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

"""Task to make a flexible and repeatable selection of reserve stars.
"""

__all__ = ['ReserveIsolatedStarsConfig',
           'ReserveIsolatedStarsTask']

import numpy as np
import hashlib

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase


class ReserveIsolatedStarsConfig(pexConfig.Config):
    """Configuration for ReserveIsolatedStarsTask."""
    reserve_name = pexConfig.Field(
        doc='Name to use for random seed selection hash.',
        dtype=str,
        default='reserved',
    )
    reserve_fraction = pexConfig.RangeField(
        doc='Fraction of stars to reserve.  None if == 0.',
        dtype=float,
        default=0.1,
        min=0.0,
        max=1.0,
        inclusiveMin=True,
        inclusiveMax=True,
    )


class ReserveIsolatedStarsTask(pipeBase.Task):
    """Reserve isolated stars with repeatable hash."""
    ConfigClass = ReserveIsolatedStarsConfig
    _DefaultName = 'reserve_isolated_stars'

    def run(self, nstar, extra=''):
        """Retrieve a selection of reserved stars.

        Parameters
        ----------
        nstar : `int`
            Number of stars to select from.
        extra : `str`, optional
            Extra name to appended to reserve_name, often tract or pixel,
            and may be combined with band name.

        Returns
        -------
        selection : `np.ndarray` (N,)
            Boolean index array, with ``True`` for reserved stars.
        """
        selection = np.zeros(nstar, dtype=bool)

        if self.config.reserve_fraction == 0.0:
            return selection

        # Full name combines the configured reserve name and the tract.
        name = self.config.reserve_name + '_' + extra
        # Random seed is the lower 32 bits of the hashed name.
        # We use hashlib.sha256 for guaranteed repeatability.
        hex_hash = hashlib.sha256(name.encode('UTF-8')).hexdigest()
        seed = int('0x' + hex_hash, 0) & 0xFFFFFFFF

        rng = np.random.default_rng(seed)

        selection[:int(nstar*self.config.reserve_fraction)] = True
        rng.shuffle(selection)

        return selection
