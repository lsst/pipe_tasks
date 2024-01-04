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

from astropy.table import Table as AstropyTable
import unittest

from lsst.daf.butler.cli.butler import cli as butlerCli
from lsst.daf.butler import Butler
from lsst.daf.butler.cli.utils import clickResultMsg, LogCliRunner
from lsst.daf.butler.tests import CliCmdTestBase
from lsst.daf.butler.tests.utils import ButlerTestHelper, readTable
from lsst.pipe.tasks.script import registerSkymap, registerDcrSubfilters
from lsst.pipe.tasks.cli.cmd import make_discrete_skymap, register_skymap


class RegisterSkymapTest(CliCmdTestBase, unittest.TestCase):

    mockFuncName = "lsst.pipe.tasks.cli.cmd.commands.script.registerSkymap.registerSkymap"

    @staticmethod
    def defaultExpected():
        return dict(config={}, config_file=None)

    @staticmethod
    def command():
        return register_skymap

    def test_minimal(self):
        self.run_test(["register-skymap", "repo"],
                      self.makeExpected(repo="repo"))

    def test_all(self):
        self.run_test(["register-skymap", "repo",
                       "--config-file", "path/to/file",
                       "--config", "foo=bar"],
                      self.makeExpected(repo="repo",
                                        config_file="path/to/file",
                                        config=dict(foo="bar")))

    def test_missing(self):
        self.run_missing(["register-skymap"],
                         "Missing argument ['\"]REPO['\"]")


class RegisterSkymapConfigTest(unittest.TestCase):

    def setUp(self):
        self.runner = LogCliRunner()

    def testNoConfigOverride(self):
        """Verify expected arguments are passed to makeSkyMap with no config
        overrides."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(butlerCli, ["create", "repo"])
            self.assertEqual(result.exit_code, 0, clickResultMsg(result))
            with unittest.mock.patch("lsst.pipe.tasks.script.registerSkymap.makeSkyMap") as mock:
                # call without any config overrides
                result = self.runner.invoke(butlerCli, ["register-skymap", "repo"])
                self.assertEqual(result.exit_code, 0, clickResultMsg(result))
                expectedConfig = registerSkymap.MakeSkyMapConfig()
                mock.assert_called_once()
                # assert that the first argument to the call to makeSkyMap was a butler
                self.assertIsInstance(mock.call_args[0][0], Butler)
                # assert that the second argument matches the expected config
                self.assertEqual(mock.call_args[0][1], expectedConfig)

    def testConfigOverride(self):
        """Verify expected arguments are passed to makeSkyMap with config
        overrides."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(butlerCli, ["create", "repo"])
            self.assertEqual(result.exit_code, 0, clickResultMsg(result))
            with unittest.mock.patch("lsst.pipe.tasks.script.registerSkymap.makeSkyMap") as mock:
                # call and override the name parameter of the config
                result = self.runner.invoke(butlerCli, ["register-skymap", "repo",
                                                        "--config", "name=bar"])
                self.assertEqual(result.exit_code, 0, clickResultMsg(result))
                expectedConfig = registerSkymap.MakeSkyMapConfig()
                expectedConfig.update(name="bar")
                mock.assert_called_once()
                # assert that the first argument to the makeSkyMap call was a butler
                self.assertIsInstance(mock.call_args[0][0], Butler)
                # assert that the second argument matches the expected config
                self.assertEqual(mock.call_args[0][1], expectedConfig)

    def testNonExistantConfigFile(self):
        """Verify an expected error when a given config override file does not
        exist. """
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(butlerCli, ["create", "repo"])
            self.assertEqual(result.exit_code, 0, clickResultMsg(result))
            result = self.runner.invoke(butlerCli, ["register-skymap", "repo",
                                                    "--config-file", "foo.py"])
            # foo.py does not exist; exit could should be non-zero.
            self.assertNotEqual(result.exit_code, 0, clickResultMsg(result))


class DefineMakeDiscreteSkymap(CliCmdTestBase, unittest.TestCase):

    mockFuncName = "lsst.pipe.tasks.cli.cmd.commands.script.makeDiscreteSkyMap"

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


class RegisterDcrSubfiltersTest(unittest.TestCase, ButlerTestHelper):

    def setUp(self):
        self.runner = LogCliRunner()
        self.repo = "here"

    def testRegisterFilters(self):
        """Register a few filters and verify they are added to the repo."""

        with self.runner.isolated_filesystem():
            result = self.runner.invoke(butlerCli, ["create", self.repo])
            self.assertEqual(result.exit_code, 0, clickResultMsg(result))

            # Since a subfilter requires a band, and a band is implied by a
            # physical_filter, we register an instrument to define these.
            result = self.runner.invoke(
                butlerCli,
                ["register-instrument", self.repo, "lsst.obs.base.instrument_tests.DummyCam"],
            )
            self.assertEqual(result.exit_code, 0, clickResultMsg(result))

            result = self.runner.invoke(butlerCli, ["register-dcr-subfilters", self.repo, "3", "u"])
            self.assertEqual(result.exit_code, 0, clickResultMsg(result))
            self.assertIn(registerDcrSubfilters.registeredMsg.format(band="u", subfilters="[0, 1, 2]"),
                          result.output)

            result = self.runner.invoke(butlerCli, ["query-dimension-records", self.repo, "subfilter"])
            self.assertEqual(result.exit_code, 0, clickResultMsg(result))
            self.assertAstropyTablesEqual(
                AstropyTable((("u", "u", "u"), (0, 1, 2)), names=("band", "id")),
                readTable(result.output))

            # Verify expected output message for registering subfilters in a
            # band that already has subfilters
            result = self.runner.invoke(butlerCli, ["register-dcr-subfilters", self.repo, "5", "u"])
            self.assertEqual(result.exit_code, 0, clickResultMsg(result))
            self.assertIn(registerDcrSubfilters.notRegisteredMsg.format(band="u", subfilters="[0, 1, 2]"),
                          result.output)

            # Add subfilters for two filters, one new filter and one existing.
            # Verify expected result messages and registry values.
            result = self.runner.invoke(butlerCli, ["register-dcr-subfilters", self.repo, "3", "u", "g"])
            self.assertEqual(result.exit_code, 0, clickResultMsg(result))
            self.assertIn(registerDcrSubfilters.notRegisteredMsg.format(band="u", subfilters="[0, 1, 2]"),
                          result.output)
            self.assertIn(registerDcrSubfilters.registeredMsg.format(band="g", subfilters="[0, 1, 2]"),
                          result.output)
            result = self.runner.invoke(butlerCli, ["query-dimension-records", self.repo, "subfilter"])
            self.assertEqual(result.exit_code, 0, clickResultMsg(result))
            resultTable = readTable(result.output)
            resultTable.sort(["band", "id"])
            self.assertAstropyTablesEqual(
                AstropyTable((("g", "g", "g", "u", "u", "u"),
                              (0, 1, 2, 0, 1, 2)), names=("band", "id")),
                resultTable)


if __name__ == "__main__":
    unittest.main()
