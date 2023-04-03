# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging

from lsst.daf.butler import Butler
from lsst.resources import ResourcePath
import lsst.pex.config as pexConfig
from lsst.skymap import skyMapRegistry


_log = logging.getLogger(__name__)


class MakeSkyMapConfig(pexConfig.Config):
    """Config for makeSkyMap.
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


def makeSkyMap(butler, config):
    """Construct and save a SkyMap into a gen3 butler repository.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler repository to which the new skymap will be written.
    config : `MakeSkyMapConfig` or None
        Instance of a configuration class specifying task options.
    """
    skyMap = config.skyMap.apply()
    skyMap.logSkyMapInfo(_log)
    skyMap.register(config.name, butler)


def registerSkymap(repo, config, config_file):
    """Make and register a SkyMap in a butler repository.

    Parameters
    ----------
    repo : `str`
        URI to the location of the butler repository.
    config : `dict` [`str`, `str`] or `None`
        Key-value pairs to apply as overrides to the ingest config.
    config_file : `str` or `None`
        Path to a config overrides file. Can be a URI.

    Raises
    ------
    RuntimeError
        If a config overrides file is given and does not exist.
    """

    skyMapConfig = MakeSkyMapConfig()
    if config_file:
        # pex_config can not support URIs but in the script interface
        # we trust that the caller trusts the remote resource they are
        # specifying (configs allow arbitrary python code to run).
        resource = ResourcePath(config_file)
        with resource.as_local() as local_config:
            skyMapConfig.load(local_config.ospath)

    if config:
        skyMapConfig.update(**config)

    butler = Butler(repo, writeable=True)
    makeSkyMap(butler, skyMapConfig)
