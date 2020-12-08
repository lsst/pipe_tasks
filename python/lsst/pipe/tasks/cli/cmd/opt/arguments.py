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


import click

from lsst.daf.butler.cli.utils import (
    MWArgumentDecorator,
    split_commas,
    unwrap,
)


num_subfilters_argument = MWArgumentDecorator(
    "NUM_SUBFILTERS",
    help="NUM_SUBFILTERS is the number of subfilters to be used for chromatic modeling.",
    type=click.IntRange(min=1),
    required=True
)

band_names_argument = MWArgumentDecorator(
    "BAND_NAMES",
    help=unwrap("""BAND_NAMES names of the bands to define chromatic subfilters for in the registry. Each
                band will have the same number of subfilters defined, for example 'g0', 'g1', and 'g2' for
                three subfilters and band 'g'."""),
    callback=split_commas,
    required=True,
    nargs=-1,
)
