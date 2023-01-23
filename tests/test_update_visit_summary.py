# This file is part of pipe_tasks.
#
# LSST Data Management System
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
# See COPYRIGHT file at the top of the source tree.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.

import unittest

from lsst.afw.table import ExposureCatalog, ExposureTable
from lsst.afw.image import ExposureSummaryStats
from lsst.pipe.tasks.update_visit_summary import (
    UpdateVisitSummaryConfig,
    UpdateVisitSummaryConnections,
    UpdateVisitSummaryTask,
)


class UpdateVisitSummaryTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.input_schema = ExposureTable.makeMinimalSchema()
        ExposureSummaryStats.update_schema(self.input_schema)
        self.init_inputs = {"input_summary_schema": ExposureCatalog(self.input_schema)}

    def test_wcs_provider(self) -> None:
        """Test the wcs_provider config option's effect on connections.
        """
        config = UpdateVisitSummaryConfig()
        config.wcs_provider = "input_summary"
        connections = UpdateVisitSummaryConnections(config=config)
        self.assertNotIn("wcs_overrides_tract", connections.inputs)
        self.assertNotIn("wcs_overrides_global", connections.inputs)
        task = UpdateVisitSummaryTask(config=config, initInputs=self.init_inputs)
        self.assertEqual(task.schema, self.input_schema)
        config.wcs_provider = "tract"
        connections = UpdateVisitSummaryConnections(config=config)
        self.assertIn("wcs_overrides_tract", connections.inputs)
        self.assertNotIn("wcs_overrides_global", connections.inputs)
        task = UpdateVisitSummaryTask(config=config, initInputs=self.init_inputs)
        self.assertTrue(task.schema.contains(self.input_schema))
        self.assertIn("wcsTractId", task.schema.getNames())
        config.wcs_provider = "global"
        connections = UpdateVisitSummaryConnections(config=config)
        self.assertNotIn("wcs_overrides_tract", connections.inputs)
        self.assertIn("wcs_overrides_global", connections.inputs)
        task = UpdateVisitSummaryTask(config=config, initInputs=self.init_inputs)
        self.assertEqual(task.schema, self.input_schema)

    def test_photo_calib_provider(self) -> None:
        """Test the photo_calib_provider config option's effect on connections.
        """
        config = UpdateVisitSummaryConfig()
        config.photo_calib_provider = "input_summary"
        connections = UpdateVisitSummaryConnections(config=config)
        self.assertNotIn("photo_calib_overrides_tract", connections.inputs)
        self.assertNotIn("photo_calib_overrides_global", connections.inputs)
        task = UpdateVisitSummaryTask(config=config, initInputs=self.init_inputs)
        self.assertEqual(task.schema, self.input_schema)
        config.photo_calib_provider = "tract"
        connections = UpdateVisitSummaryConnections(config=config)
        self.assertIn("photo_calib_overrides_tract", connections.inputs)
        self.assertNotIn("photo_calib_overrides_global", connections.inputs)
        task = UpdateVisitSummaryTask(config=config, initInputs=self.init_inputs)
        self.assertTrue(task.schema.contains(self.input_schema))
        self.assertIn("photoCalibTractId", task.schema.getNames())
        config.photo_calib_provider = "global"
        connections = UpdateVisitSummaryConnections(config=config)
        self.assertNotIn("photo_calib_overrides_tract", connections.inputs)
        self.assertIn("photo_calib_overrides_global", connections.inputs)
        task = UpdateVisitSummaryTask(config=config, initInputs=self.init_inputs)
        self.assertEqual(task.schema, self.input_schema)

    def test_background_provider(self) -> None:
        """Test the background_provider config option's effect on connections.
        """
        config = UpdateVisitSummaryConfig()
        config.background_provider = "input_summary"
        connections = UpdateVisitSummaryConnections(config=config)
        self.assertNotIn("background_overrides", connections.inputs)
        self.assertNotIn("background_originals", connections.inputs)
        config.background_provider = "replacement"
        connections = UpdateVisitSummaryConnections(config=config)
        self.assertIn("background_overrides", connections.inputs)
        self.assertIn("background_originals", connections.inputs)


if __name__ == "__main__":
    unittest.main()
