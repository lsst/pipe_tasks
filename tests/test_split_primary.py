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

import numpy as np
import astropy.table

import lsst.utils.tests
from lsst.daf.butler import DimensionUniverse
from lsst.pipe.base import PipelineGraph
from lsst.pipe.tasks.split_primary import SplitPrimaryConfig, SplitPrimaryTask


class SplitPrimaryTaskTestCase(lsst.utils.tests.TestCase):
    def test_connections(self):
        """Test that we can reconfigure the dimensions of the connections."""
        universe = DimensionUniverse()
        config = SplitPrimaryConfig()
        config.dimensions = ["visit"]
        pg = PipelineGraph()
        pg.add_task("split_primary", SplitPrimaryTask, config=config)
        pg.resolve(dimensions=universe)
        self.assertEqual(pg.tasks["split_primary"].dimensions, universe.conform(["visit"]))
        self.assertEqual(pg.dataset_types["full"].dimensions, universe.conform(["visit"]))
        self.assertEqual(pg.dataset_types["primary"].dimensions, universe.conform(["visit"]))
        self.assertEqual(pg.dataset_types["nonprimary"].dimensions, universe.conform(["visit"]))

    def test_run(self):
        """Test the 'run' method."""
        rng = np.random.default_rng(10)
        n = 20
        full = astropy.table.Table(
            {
                "primary": rng.random(n) > 0.5,
                "a": rng.standard_normal(n),
                "b": rng.standard_normal(n),
                "c": rng.standard_normal(n),
            }
        )
        config = SplitPrimaryConfig()
        config.primary_flag_column = "primary"
        # Note that column 'd' is not present; it should be ignored.
        config.discard_primary_columns = ["b", "d"]
        config.discard_nonprimary_columns = ["c", "d"]
        task = SplitPrimaryTask(config=config)
        result = task.run(full=full)
        self.assertEqual(result.primary.colnames, ["a", "c"])
        self.assertEqual(result.nonprimary.colnames, ["a", "b"])
        self.assertFloatsEqual(result.primary["a"], full["a"][full["primary"]])
        self.assertFloatsEqual(result.primary["c"], full["c"][full["primary"]])
        self.assertFloatsEqual(result.nonprimary["a"], full["a"][np.logical_not(full["primary"])])
        self.assertFloatsEqual(result.nonprimary["b"], full["b"][np.logical_not(full["primary"])])


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
