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

import os
import unittest

import pandas as pd

from lsst.pipe.tasks.schemaUtils import readSdmSchemaFile, make_empty_catalog, convertDataFrameToSdmSchema


class TestSchemaUtils(unittest.TestCase):

    def test_make_empty_catalog(self):
        """Check that an empty catalog has the correct format.
        """
        schemaFile = os.path.join("${SDM_SCHEMAS_DIR}", "yml", "apdb.yaml")
        schema = readSdmSchemaFile(schemaFile)

        tableNames = ["DiaObject", "DiaSource", "DiaForcedSource"]
        for tableName in tableNames:
            emptyDiaObjects = make_empty_catalog(schema, tableName=tableName)
            self.assertTrue(emptyDiaObjects.empty)

            emptyColumns = set(emptyDiaObjects.columns)
            self.assertIn("ra", emptyColumns)
            self.assertIn("dec", emptyColumns)
            self.assertIn("diaObjectId", emptyColumns)

            emptyDf = pd.DataFrame(columns=["diaObjectId",])
            emptyDf.set_index("diaObjectId")
            convertedEmptyDiaObjects = convertDataFrameToSdmSchema(schema, emptyDf, tableName=tableName)
            # TODO: we have no tests of convertDataFrameToSdmSchema, so it's dangerous to use it as an oracle.
            emptyTypes = dict(zip(emptyDiaObjects.columns, emptyDiaObjects.dtypes))
            convertedEmptyTypes = dict(zip(convertedEmptyDiaObjects.columns, convertedEmptyDiaObjects.dtypes))
            self.assertEqual(emptyTypes, convertedEmptyTypes)
