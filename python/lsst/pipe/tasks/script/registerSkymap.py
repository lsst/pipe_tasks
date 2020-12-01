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

from lsst.daf.butler import Butler
# Preserve the import path of makeGen3SkyMap so that it is easily mockable:
import lsst.pipe.tasks.makeGen3SkyMap


def registerSkymap(repo, config, config_file):
    """Make and register a SkyMap in a butler repository.

    Parameters
    ----------
    repo : `str`
        URI to the location of the butler repository.
    config : `dict` [`str`, `str`] or `None`
        Key-value pairs to apply as overrides to the ingest config.
    config_file : `str` or `None`
        Path to a config overrides file.

    Raises
    ------
    RuntimeError
        If a config overrides file is given and does not exist.
    """
    skyMapConfig = lsst.pipe.tasks.makeGen3SkyMap.MakeGen3SkyMapConfig()
    if config_file:
        skyMapConfig.load(config_file)

    if config:
        skyMapConfig.update(**config)

    butler = Butler(repo, writeable=True)
    lsst.pipe.tasks.makeGen3SkyMap.makeGen3SkyMap(butler, skyMapConfig)
