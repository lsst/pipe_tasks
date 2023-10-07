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
from types import SimpleNamespace

from lsst.pipe.tasks.configurableActions import (ConfigurableActionStructField, ConfigurableAction,
                                                 ConfigurableActionField)
from lsst.pex.config import Config, Field, FieldValidationError
from lsst.pipe.base import Struct


class ActionTest1(ConfigurableAction):
    var = Field(doc="test field", dtype=int, default=0)

    def __call__(self):
        return self.var

    def validate(self):
        assert self.var is not None


class ActionTest2(ConfigurableAction):
    var = Field(doc="test field", dtype=int, default=1)

    def __call__(self):
        return self.var

    def validate(self):
        assert self.var is not None


class ActionTest3(ConfigurableAction):
    var = Field(doc="test field", dtype=int, default=3)

    def __call__(self):
        return self.var

    def validate(self):
        assert self.var is not None


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
        configClass = self._createConfig(default={"test1": ActionTest1}, singleDefault=ActionTest1)
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
        configClass = self._createConfig(default={"test1": ActionTest1})
        config = configClass()
        config.actions.test2 = ActionTest2

        self.assertEqual(tuple(config.actions.fieldNames), ("test1", "test2"))
        self.assertEqual(config.actions.test2.var, 1)

        # verify the same as above, but assigning with instances
        configClass = self._createConfig(default={"test1": ActionTest1})
        config = configClass()
        config.actions.test3 = ActionTest3()

        self.assertEqual(tuple(config.actions.fieldNames), ("test1", "test3"))
        self.assertEqual(config.actions.test3.var, 3)

        # The following is designed to support pipeline config setting
        # Test assignment using the update accessor
        configClass = self._createConfig(default={"test1": ActionTest1})
        config = configClass()
        config.actions.update = {"test2": ActionTest2, "test3": ActionTest3}

        self.assertEqual(tuple(config.actions.fieldNames), ("test1", "test2", "test3"))

        configClass = self._createConfig(default={"test1": ActionTest1})
        configClass2 = self._createConfig(default={"test2": ActionTest2, "test3": ActionTest3})
        config = configClass()
        config2 = configClass2()
        config.actions.update = config2.actions

        self.assertEqual(tuple(config.actions.fieldNames), ("test1", "test2", "test3"))

        # verify tha the update interface cannot be used to assign invalid
        # identifiers
        configClass = self._createConfig()
        config = configClass()
        with self.assertRaises(ValueError):
            config.actions.update = {"name with space": ActionTest2}

        with self.assertRaises(ValueError):
            config.actions.update = {"9leading_number": ActionTest2}

        # Test remove "assignment" using the remove accessor
        configClass = self._createConfig(default={"test1": ActionTest1, "test2": ActionTest2,
                                                  "test3": ActionTest3})
        config = configClass()
        config.actions.remove = ("test1", "test2")
        self.assertEqual(tuple(config.actions.fieldNames), ("test3", ))

        configClass = self._createConfig(default={"test1": ActionTest1, "test2": ActionTest2,
                                                  "test3": ActionTest3})
        config = configClass()
        config.actions.remove = "test1"
        self.assertEqual(tuple(config.actions.fieldNames), ("test2", "test3"))

        # singleAction
        # Test that an action can be reassigned
        configClass = self._createConfig(singleDefault=ActionTest1)
        config = configClass()
        self.assertEqual(config.singleAction(), 0)

        config.singleAction = ActionTest2
        self.assertEqual(config.singleAction(), 1)

        config.singleAction = ActionTest3()
        self.assertEqual(config.singleAction(), 3)

        # Verify that ConfigurableActionStructField can be assigned to with
        # a ConfigurableActionStruct, Struct, and SimpleNamespace
        otherConfigClass = self._createConfig(default={"test1": ActionTest1(var=1),
                                                       "test2": ActionTest2(var=2)})
        assignSource1 = otherConfigClass().actions
        assignSource2 = Struct(test1=ActionTest1(var=1), test2=ActionTest2(var=2))
        assignSource3 = SimpleNamespace(test1=ActionTest1(var=1), test2=ActionTest2(var=2))

        for source in (assignSource1, assignSource2, assignSource3):
            configClass = self._createConfig()
            config = configClass()
            config.actions = source

            self.assertEqual(tuple(config.actions.fieldNames), ("test1", "test2"))
            self.assertEqual((config.actions.test1.var, config.actions.test2.var), (1, 2))

        # Fail if assigment is ConfigurableActionStructField
        with self.assertRaises(ValueError):
            configClass = self._createConfig()
            config = configClass()
            config.actions = otherConfigClass.actions

        # Fail if assignment is some other type
        with self.assertRaises(ValueError):
            configClass = self._createConfig()
            config = configClass()
            config.actions = {}

    def testValidate(self):
        configClass = self._createConfig(default={"test1": ActionTest1, "test2": ActionTest2,
                                                  "test3": ActionTest3}, singleDefault=ActionTest1)
        config = configClass()
        config.validate()

    def testFreeze(self):
        configClass = self._createConfig(default={"test1": ActionTest1, "test2": ActionTest2},
                                         singleDefault=ActionTest1)
        config = configClass()
        config.freeze()

        with self.assertRaises(FieldValidationError):
            config.actions.test3 = ActionTest3

        with self.assertRaises(FieldValidationError):
            config.actions.test1.var = 2

        with self.assertRaises(FieldValidationError):
            config.actions.test2.var = 0

        with self.assertRaises(FieldValidationError):
            config.singleAction = ActionTest2

        with self.assertRaises(FieldValidationError):
            config.singleAction.var = 3

    def testCompare(self):
        configClass = self._createConfig(default={"test1": ActionTest1, "test2": ActionTest2},
                                         singleDefault=ActionTest1)
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
        configClass = self._createConfig(default={"test1": ActionTest1},
                                         singleDefault=ActionTest1)
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
        configClass = self._createConfig(default={"test1": ActionTest1},
                                         singleDefault=ActionTest1)
        config = configClass()
        self.assertEqual(config.toDict(), {'actions': {'test1': {'var': 0}}, 'singleAction': {'var': 0}})


if __name__ == "__main__":
    unittest.main()
