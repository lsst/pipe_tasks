#
# LSST Data Management System
# Copyright 2008-2022 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

__all__ = ['ReserveIsolatedStarsConfig',
           'ReserveIsolatedStarsTask']

import numpy as np

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

    def run(self, tract, nstar):
        """Retrieve a selection of reserved stars.

        Parameters
        ----------
        tract : `int`
            Tract number (used in creation of random seed).
        nstar : `int`
            Number of stars to select from.

        Returns
        -------
        selection : `np.ndarray` (N,)
            Boolean index array, with ``True`` for reserved stars.
        """
        selection = np.zeros(nstar, dtype=bool)

        if self.config.reserve_fraction == 0.0:
            return selection

        # Full name combines the configured reserve name and the tract.
        name = self.config.reserve_name + '_' + str(tract)
        # Random seed is the lower 32 bits of the hashed name.
        seed = hash(name) & 0xFFFFFFFF

        rng = np.random.default_rng(seed)

        selection[:int(nstar*self.config.reserve_fraction)] = True
        rng.shuffle(selection)

        return selection
