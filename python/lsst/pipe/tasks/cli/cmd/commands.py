# This file is part of obs_base.
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

import click

from lsst.daf.butler.cli.opt import (
    collections_option,
    config_option,
    config_file_option,
    options_file_option,
    repo_argument,
)
from lsst.daf.butler.cli.utils import ButlerCommand
from lsst.pipe.base.cli.opt import instrument_argument
from .opt import (
    band_names_argument,
    num_subfilters_argument,
)
from ... import script


@click.command(cls=ButlerCommand, short_help="Define a discrete skymap from calibrated exposures.")
@repo_argument(required=True)
@instrument_argument(required=True)
@config_file_option(help="URI to a pex_config override to be included after the Instrument config overrides"
                         "are applied.")
@collections_option(help="The collections to be searched (in order) when reading datasets. "
                         "This includes the seed skymap if --append is specified.",
                    required=True)
@click.option("--skymap-id",
              help="The identifier of the skymap to write.",
              type=str, default="discrete", show_default=True)
@click.option("--old-skymap-id",
              help=("The identifier of the previous skymap to append to, if config.doAppend is True."),
              type=str, default=None)
@options_file_option()
def make_discrete_skymap(*args, **kwargs):
    """Define a discrete skymap from calibrated exposures in the butler registry."""
    script.makeDiscreteSkyMap(*args, **kwargs)


@click.command(cls=ButlerCommand)
@repo_argument(required=True)
@config_option()
@config_file_option(help="URI to a config file overrides file.")
@options_file_option()
def register_skymap(*args, **kwargs):
    """Make a SkyMap and add it to a repository."""
    script.registerSkymap.registerSkymap(*args, **kwargs)


@click.command(cls=ButlerCommand,
               short_help="Add subfilters for chaotic modeling.")
@repo_argument(required=True)
@num_subfilters_argument()
@band_names_argument()
@options_file_option()
def register_dcr_subfilters(**kwargs):
    """Construct a set of subfilters for chromatic modeling and add them to a
    registry."""
    print(script.registerDcrSubfilters.registerDcrSubfilters(**kwargs))
