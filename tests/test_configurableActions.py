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

import unittest

import lsst.utils.tests

from lsst.pipe.tasks.configurableActions import ConfigurableActionsField, ConfigurableAction
from lsst.pex.config import Config, Field


class TestAction1(ConfigurableAction):
    var = Field(doc="test field", dtype=int, default=0)

    def __call__(self):
        return self.var


class TestAction2(ConfigurableAction):
    var = Field(doc="test field", dtype=int, default=1)

    def __call__(self):
        return self.var


class TestAction3(ConfigurableAction):
    var = Field(doc="test field", dtype=int, default=3)

    def __call__(self):
        return self.var


class ConfigurableActionsTestCase(unittest.TestCase):
    def _createConfig(self, default=None):
        class TestConfig(Config):
            actions = ConfigurableActionsField(doc="Actions to be tested", default=default)
        return TestConfig

    def testConfigInstatiation(self):
        # This will raise if there is an issue instatiating something
        configClass = self._createConfig()
        config = configClass()
        self.assertTrue(hasattr(config, "actions"))

        # test again with default values
        configClass = self._createConfig(default={"test1": TestAction1})
        config = configClass()

        # verify the defaults were created
        self.assertTrue(hasattr(config.actions, "test1"))
        self.assertTrue(hasattr(config.actions.test1, "var"))
        self.assertEqual(config.actions.test1.var, 0)

    def testAssignment(self):
        # Test that a new action can be added with assignment
        configClass = self._createConfig(default={"test1": TestAction1})
        config = configClass()
        config.actions.test2 = TestAction2

        self.assertEqual(tuple(config.actions.fieldNames), ("test1", "test2"))
        self.assertEqual(config.actions.test2.var, 1)

        ### NOTE DESIGN DISCUSSION TO BE HAD HERE
        # Test that a dictionary can be used to add multiple parallel
        # assignments (useful for pipeline assingment)
        configClass = self._createConfig(default={"test1": TestAction1})
        config = configClass()
        config.actions = {"test2": TestAction2, "test3": TestAction3}

        self.assertEqual(tuple(config.actions.fieldNames), ("test1", "test2", "test3"))

        # Test that a string can be used in place of a Action class to support
        # pipeline assingment
        configClass = self._createConfig(default={"test1": TestAction1})
        config = configClass()
        config.actions = {"test2": 'TestAction2', "test3": 'TestAction3'}
        self.assertEqual(tuple(config.actions.fieldNames), ("test1", "test2", "test3"))
        self.assertIsInstance(config.actions.test2, TestAction2)







def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
