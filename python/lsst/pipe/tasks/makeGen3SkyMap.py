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


import logging

import lsst.pex.config as pexConfig
from lsst.skymap import skyMapRegistry


_log = logging.getLogger(__name__.partition(".")[2])


class MakeGen3SkyMapConfig(pexConfig.Config):
    """Config for makeGen3SkyMap.
    """
    name = pexConfig.Field(
        doc="Name assigned to created skymap in butler registry",
        dtype=str,
        default=None,
        optional=True
    )
    skyMap = skyMapRegistry.makeField(
        doc="type of skyMap",
        default="dodeca",
    )

    def validate(self):
        if self.name is None:
            raise ValueError("The name field must be set to the name of the specific "
                             "skymap to use when writing to the butler")


def makeGen3SkyMap(butler, config):
    """Construct and save a SkyMap into a gen3 butler repository.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler repository to which the new skymap will be written.
    config : `MakeGen3SkyMapConfig` or None
        Instance of a configuration class specifying task options.
    """
    skyMap = config.skyMap.apply()
    skyMap.logSkyMapInfo(_log)
    skyMap.register(config.name, butler)
