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
from io import StringIO

from lsst.pipe.tasks.configurableActions import (ConfigurableActionStructField, ConfigurableAction,
                                                 ConfigurableActionField)
from lsst.pex.config import Config, Field, FieldValidationError


class TestAction1(ConfigurableAction):
    var = Field(doc="test field", dtype=int, default=0)

    def __call__(self):
        return self.var

    def validate(self):
        assert(self.var is not None)


class TestAction2(ConfigurableAction):
    var = Field(doc="test field", dtype=int, default=1)

    def __call__(self):
        return self.var

    def validate(self):
        assert(self.var is not None)


class TestAction3(ConfigurableAction):
    var = Field(doc="test field", dtype=int, default=3)

    def __call__(self):
        return self.var

    def validate(self):
        assert(self.var is not None)


class ConfigurableActionsTestCase(unittest.TestCase):
    def _createConfig(self, default=None, singleDefault=None):
        class TestConfig(Config):
            actions = ConfigurableActionStructField(doc="Actions to be tested", default=default)
            singleAction = ConfigurableActionField(doc="A configurable action", default=singleDefault)
        return TestConfig

    def testConfigInstatiation(self):
        # This will raise if there is an issue instantiating something
        configClass = self._createConfig()
        config = configClass()
        self.assertTrue(hasattr(config, "actions"))
        self.assertTrue(hasattr(config, "singleAction"))

        # test again with default values
        configClass = self._createConfig(default={"test1": TestAction1}, singleDefault=TestAction1)
        config = configClass()

        # verify the defaults were created
        self.assertTrue(hasattr(config.actions, "test1"))
        self.assertTrue(hasattr(config.actions.test1, "var"))
        self.assertEqual(config.actions.test1.var, 0)

        self.assertTrue(hasattr(config.singleAction, "var"))
        self.assertEqual(config.singleAction.var, 0)

    def testAssignment(self):
        # Struct actions
        # Test that a new action can be added with assignment
        configClass = self._createConfig(default={"test1": TestAction1})
        config = configClass()
        config.actions.test2 = TestAction2

        self.assertEqual(tuple(config.actions.fieldNames), ("test1", "test2"))
        self.assertEqual(config.actions.test2.var, 1)

        # verify the same as above, but assigning with instances
        configClass = self._createConfig(default={"test1": TestAction1})
        config = configClass()
        config.actions.test3 = TestAction3()

        self.assertEqual(tuple(config.actions.fieldNames), ("test1", "test3"))
        self.assertEqual(config.actions.test3.var, 3)

        # The following is designed to support pipeline config setting
        # Test assignment using the update accessor
        configClass = self._createConfig(default={"test1": TestAction1})
        config = configClass()
        config.actions.update = {"test2": TestAction2, "test3": TestAction3}

        self.assertEqual(tuple(config.actions.fieldNames), ("test1", "test2", "test3"))

        configClass = self._createConfig(default={"test1": TestAction1})
        configClass2 = self._createConfig(default={"test2": TestAction2, "test3": TestAction3})
        config = configClass()
        config2 = configClass2()
        config.actions.update = config2.actions

        self.assertEqual(tuple(config.actions.fieldNames), ("test1", "test2", "test3"))

        # Test remove "assignment" using the remove accessor
        configClass = self._createConfig(default={"test1": TestAction1, "test2": TestAction2,
                                                  "test3": TestAction3})
        config = configClass()
        config.actions.remove = ("test1", "test2")
        self.assertEqual(tuple(config.actions.fieldNames), ("test3", ))

        configClass = self._createConfig(default={"test1": TestAction1, "test2": TestAction2,
                                                  "test3": TestAction3})
        config = configClass()
        config.actions.remove = "test1"
        self.assertEqual(tuple(config.actions.fieldNames), ("test2", "test3"))

        # singleAction
        # Test that an action can be reassigned
        configClass = self._createConfig(singleDefault=TestAction1)
        config = configClass()
        self.assertEqual(config.singleAction(), 0)

        config.singleAction = TestAction2
        self.assertEqual(config.singleAction(), 1)

        config.singleAction = TestAction3()
        self.assertEqual(config.singleAction(), 3)

    def testValidate(self):
        configClass = self._createConfig(default={"test1": TestAction1, "test2": TestAction2,
                                                  "test3": TestAction3}, singleDefault=TestAction1)
        config = configClass()
        config.validate()

    def testFreeze(self):
        configClass = self._createConfig(default={"test1": TestAction1, "test2": TestAction2},
                                         singleDefault=TestAction1)
        config = configClass()
        config.freeze()

        with self.assertRaises(FieldValidationError):
            config.actions.test3 = TestAction3

        with self.assertRaises(FieldValidationError):
            config.actions.test1.var = 2

        with self.assertRaises(FieldValidationError):
            config.actions.test2.var = 0

        with self.assertRaises(FieldValidationError):
            config.singleAction = TestAction2

        with self.assertRaises(FieldValidationError):
            config.singleAction.var = 3

    def testCompare(self):
        configClass = self._createConfig(default={"test1": TestAction1, "test2": TestAction2},
                                         singleDefault=TestAction1)
        config = configClass()
        config2 = configClass()

        self.assertTrue(config.compare(config2))

        # Test equality fail for ConfigurableActionsStructField
        config3 = configClass()
        config3.actions.test1.var = 99
        self.assertFalse(config.compare(config3))

        # Test equality fail for ConfigurableActionsField
        config4 = configClass()
        config4.singleAction.var = 99
        self.assertFalse(config.compare(config4))

    def testSave(self):
        # This method will also test rename, as it is part of the
        # implementation in pex_config
        ioObject = StringIO()
        configClass = self._createConfig(default={"test1": TestAction1},
                                         singleDefault=TestAction1)
        config = configClass()

        config.saveToStream(ioObject)
        loadedConfig = configClass()
        loadedConfig.loadFromStream(ioObject.read())
        self.assertTrue(config.compare(loadedConfig))
        # Be sure that the fields are actually there
        self.assertEqual(loadedConfig.actions.test1.var, 0)
        self.assertEqual(loadedConfig.singleAction.var, 0)

    def testToDict(self):
        """Test the toDict interface"""
        configClass = self._createConfig(default={"test1": TestAction1},
                                         singleDefault=TestAction1)
        config = configClass()
        self.assertEqual(config.toDict(), {'actions': {'test1': {'var': 0}}, 'singleAction': {'var': 0}})


if __name__ == "__main__":
    unittest.main()
