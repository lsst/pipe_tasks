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

from lsst.daf.butler.cli.opt import (repo_argument, config_file_option)
from lsst.daf.butler.cli.utils import (cli_handle_exception, split_commas, typeStrAcceptsMultiple)
from lsst.obs.base.cli.opt import instrument_option
from ... import script


@click.command(short_help="Define a discrete skymap from calibrated exposures.")
@repo_argument(required=True)
@config_file_option(help="Path to a pex_config override to be included after the Instrument config overrides"
                         "are applied.")
@click.option("--collections",
              help=("The collections to be searched (in order) when reading datasets. "
                    "This includes the seed skymap if --append is specified."),
              multiple=True,
              callback=split_commas,
              metavar=typeStrAcceptsMultiple,
              required=True)
@click.option("--out-collection",
              help=("The collection to write the skymap to."),
              type=str, default="skymaps", show_default=True)
@click.option("--skymap-id",
              help=("The identifier of the skymap to write."),
              type=str, default="discrete", show_default=True)
@instrument_option(required=True)
def make_discrete_skymap(*args, **kwargs):
    """Define a discrete skymap from calibrated exposures in the butler registry."""
    cli_handle_exception(script.makeDiscreteSkyMap, *args, **kwargs)
