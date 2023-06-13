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

"""Tests for HealSparseMapFormatter.
"""

import unittest
import os
import numpy as np
import healsparse as hsp

from lsst.daf.butler import Butler, DatasetType
from lsst.daf.butler.tests.utils import makeTestTempDir, removeTestTempDir

TESTDIR = os.path.abspath(os.path.dirname(__file__))


class HealSparseMapFormatterTestCase(unittest.TestCase):
    """Test for HealSparseMapFormatter.
    """

    def setUp(self):
        self.root = makeTestTempDir(TESTDIR)
        Butler.makeRepo(self.root)

        self.hspMap = hsp.HealSparseMap.make_empty(nside_coverage=32, nside_sparse=4096, dtype=np.float32)
        self.hspMap[0: 10000] = 1.0
        self.hspMap[100000: 110000] = 2.0
        self.hspMap[500000: 510000] = 3.0

    def tearDown(self):
        removeTestTempDir(self.root)
        del self.hspMap

    def testHealSparseMapFormatter(self):
        butler = Butler(self.root, run="testrun")
        datasetType = DatasetType("map", [], "HealSparseMap",
                                  universe=butler.dimensions)
        butler.registry.registerDatasetType(datasetType)
        ref = butler.put(self.hspMap, datasetType)
        uri = butler.getURI(ref)
        self.assertEqual(uri.getExtension(), '.hsp')

        # Retrieve the full map.
        hspMap = butler.get('map')
        self.assertTrue(np.all(hspMap._sparse_map == self.hspMap._sparse_map))

        # Retrieve the coverage map
        coverage = butler.get('map.coverage')
        self.assertTrue(np.all(coverage.coverage_mask == self.hspMap.coverage_mask))

        # Retrieve a partial map
        pixels = [0, 6]
        partialMap = butler.get('map', parameters={'pixels': pixels})

        self.assertTrue(np.all(np.where(partialMap.coverage_mask)[0] == np.array(pixels)))
        self.assertTrue(np.all(partialMap[0: 10000] == self.hspMap[0: 10000]))
        self.assertTrue(np.all(partialMap[100000: 110000] == self.hspMap[100000: 110000]))

        # Retrieve a degraded map
        degradedMapRead = butler.get('map', parameters={'degrade_nside': 512})
        degradedMap = self.hspMap.degrade(512)

        self.assertTrue(np.all(degradedMapRead._sparse_map == degradedMap._sparse_map))


if __name__ == "__main__":
    unittest.main()
