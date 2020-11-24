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

"""Unit tests for daf_butler CLI make-discrete-skymap command.
"""

import unittest

from lsst.daf.butler.tests import CliCmdTestBase
from lsst.pipe.tasks.cli.cmd import make_discrete_skymap


class DefineMakeDiscreteSkymap(CliCmdTestBase, unittest.TestCase):

    @staticmethod
    def defaultExpected():
        return dict(config_file=None,
                    collections=())

    @staticmethod
    def command():
        return make_discrete_skymap

    def test_repoBasic(self):
        """Test the most basic required arguments."""
        self.run_test(["make-discrete-skymap",
                       "--collections", "foo/bar,baz",
                       "here", "a.b.c"],
                      self.makeExpected(repo="here",
                                        collections=("foo/bar", "baz"),
                                        skymap_id="discrete",
                                        instrument="a.b.c",
                                        old_skymap_id=None))

    def test_all(self):
        """Test all the arguments."""
        self.run_test(["make-discrete-skymap",
                       "--collections", "foo/bar,baz",
                       "--config-file", "/path/to/config",
                       "--collections", "boz",
                       "--skymap-id", "wiz",
                       "--old-skymap-id", "nuz",
                       "here", "a.b.c"],
                      self.makeExpected(repo="here",
                                        instrument="a.b.c",
                                        config_file="/path/to/config",
                                        skymap_id="wiz",
                                        old_skymap_id="nuz",
                                        # The list of collections must be in
                                        # exactly the same order as it is
                                        # passed in the list of arguments to
                                        # run_test.
                                        collections=("foo/bar", "baz", "boz")))

    def test_missing(self):
        """test a missing argument"""
        self.run_missing(["make-discrete-skymap", "--collections", "foo/bar,baz"],
                         "Missing argument ['\"]REPO['\"]")
        self.run_missing(["make-discrete-skymap", "--collections", "foo/bar,baz", "here"],
                         "Missing argument ['\"]INSTRUMENT['\"]")
        self.run_missing(["make-discrete-skymap", "here", "a.b.c"],
                         "Error: Missing option ['\"]--collections['\"].")


if __name__ == "__main__":
    unittest.main()
