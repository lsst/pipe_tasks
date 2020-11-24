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
from lsst.pipe.tasks.makeGen3SkyMap import MakeGen3SkyMapConfig, MakeGen3SkyMapTask


def registerSkymap(repo, config_file):
    """Make and register a SkyMap in a butler repository.

    Parameters
    ----------
    repo : `str`
        URI to the location of the butler repository.
    config_file : `str` or `None`
        Path to a config overrides file.

    Raises
    ------
    RuntimeError
        If a config overrides file is given and does not exist.
    """
    config = MakeGen3SkyMapConfig()
    if config_file:
        config.load(config_file)

    # Construct the SkyMap Creation task and run it
    skyMapTask = MakeGen3SkyMapTask(config=config)
    butler = Butler(repo, writeable=True)
    skyMapTask.run(butler)
