#
# LSST Data Management System
# Copyright 2012 LSST Corporation.
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
import argparse
import re

import lsst.pipe.base as pipeBase


class SkyTileIdAction(argparse.Action):
    """Action callback to add one sky-tile ID to namespace.skyTileIdSet.
    """
    def __call__(self, parser, namespace, values, option_string):
        """Parse --id skyTile=value_list and add results to namespace.skyTileIdSet.

           The format of value_list is value(^value)*, where each value must be
           an integer. Values are added to a set, so the final list of sky-tile IDs
           is duplicate-free.
        """
        if len(values) != 1:
            parser.error(str.format(
                "{} must be followed by exactly one argument", option_string))
        name, _, values = values[0].partition("=")
        if name != "skyTile":
            parser.error(str.format(
                "{} argument must be of the form skyTile=VALUES", option_string))
        if not hasattr(namespace, "skyTileIds"):
            namespace.skyTileIds = set()
        for v in values.split("^"):
            interval = re.search(r"^(\d+)\.\.(\d+)$", v)
            if interval:
                namespace.skyTileIds.update(xrange(int(interval.group(1)),
                                                   int(interval.group(2))))
            elif re.match(r"^\d+$", v):
                namespace.skyTileIds.add(int(v))
            else:
                parser.error(str.format(
                    "cannot parse {} as an integer for {}", v, option_string))


class SourceAssocArgumentParser(pipeBase.ArgumentParser):
    """A version of lsst.pipe.base.ArgumentParser
       specialized for source association.
    """ 
    def __init__(self, **kwargs):
        """Construct an option parser.
        """
        # need to swap out the behaviour of --id with something that makes
        # sense for sourceAssoc
        pipeBase.ArgumentParser.__init__(
            self, conflict_handler='resolve', **kwargs)
        self.add_argument("--id", action=SkyTileIdAction,
            help="skytile ID. Can be specified multiple times; if omitted, all "
                 "sky-tiles are processed. Example: --id skyTile=12..23^45^67",
            metavar="skyTile=VALUES")

