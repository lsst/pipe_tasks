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

"""Check that the prompt_source pipeline produces exactly the columns defined
by the ``PromptSource`` table in ``sdm_schemas``. Only check column names, not
datatypes, since that requires running the detection and measurement task
which can't be imported here.
"""

import os
import unittest

import numpy as np
from astropy.table import Table

import lsst.utils.tests
from lsst.utils import getPackageDir

from lsst.pipe.tasks.postprocess import TransformSourceTableTask, TransformSourceTableConfig
from lsst.pipe.tasks.schemaUtils import readSdmSchemaFile
from lsst.pipe.tasks.split_primary import SplitPrimaryTask

FUNCTOR_FILE = os.path.join(getPackageDir("pipe_tasks"), "schemas", "prompt_source.yaml")

SCHEMA_FILE = os.path.join("${SDM_SCHEMAS_DIR}", "yml", "lsstcam.yaml")
TABLE_NAME = "PromptSource"


class PromptSourceSchemaTestCase(lsst.utils.tests.TestCase):
    """Check the persisted prompt_source columns against the PromptSource DDL.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        schemaFile = os.path.expandvars(SCHEMA_FILE)
        if not os.path.exists(schemaFile):
            raise unittest.SkipTest(f"SDM schema file not available: {SCHEMA_FILE}")

        schema = readSdmSchemaFile(schemaFile)
        if TABLE_NAME not in schema:
            raise ValueError(f"Table {TABLE_NAME!r} not in {schemaFile}; skipping conformance check.")

        cls.schemaColumns = {column.name for column in schema[TABLE_NAME].columns}

        config = TransformSourceTableConfig()
        config.functorFile = FUNCTOR_FILE
        transformTask = TransformSourceTableTask(config=config)
        producedColumns = set(transformTask.funcs.funcDict) | set(config.columnsFromDataId)

        # Run the real splitPromptSource step to check the final set of columns
        splitConfig = SplitPrimaryTask.ConfigClass()
        splitConfig.discard_primary_columns = ["sky_source"]
        splitTask = SplitPrimaryTask(config=splitConfig)

        # SplitPrimaryTask only needs the boolean primary-flag column to mask
        # rows; the other columns are carried through (or dropped) by name, so
        # a minimal two-row table with placeholder values is sufficient.
        data = {}
        for name in producedColumns:
            if name == splitConfig.primary_flag_column:
                data[name] = np.array([True, False])
            else:
                data[name] = np.zeros(2)
        full = Table(data)

        cls.persistedColumns = set(splitTask.run(full=full).primary.colnames)

    def testColumnsConform(self):
        """The persisted prompt_source columns must equal the schema exactly.
        """
        missing = self.schemaColumns - self.persistedColumns
        extra = self.persistedColumns - self.schemaColumns

        message = (
            f"persisted prompt_source columns do not conform to the "
            f"{TABLE_NAME} schema.\n"
            f"  In schema but not persisted ({len(missing)}): "
            f"{sorted(missing)}\n"
            f"  Persisted but not in schema ({len(extra)}): "
            f"{sorted(extra)}"
        )
        self.assertEqual(self.persistedColumns, self.schemaColumns, message)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
