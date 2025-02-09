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

from __future__ import annotations

import unittest
import lsst.utils.tests
import numpy as np

from lsst.afw.image import Mask
from lsst.afw.math import StatisticsControl
from lsst.pipe.tasks.coaddBase import (
    removeMaskPlanes,
    setRejectedMaskMapping,
)


class CoaddBaseTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        self.statsCtrl = StatisticsControl()
        self.mask = Mask(100, 100)
        self.mask.array[:, :] = 0
        self.mask.array[50, 50] = Mask.getPlaneBitMask("BAD")
        self.mask.array[55, 58:62] = Mask.getPlaneBitMask("CR")

    def test_removeMaskPlanes(self):
        """Exercise removeMaskPlanes"""
        # Test that removing non-existing mask planes does not crash.
        removeMaskPlanes(self.mask, ["BLAH"])
        # Test that removing existing mask planes works.
        for maskPlane in ["BAD", "CR"]:
            with self.subTest(maskPlane=maskPlane):
                self.assertGreater((self.mask.array & Mask.getPlaneBitMask(maskPlane)).sum(), 0)
                removeMaskPlanes(self.mask, [maskPlane])
                self.assertEqual((self.mask.array & Mask.getPlaneBitMask(maskPlane)).sum(), 0)

    def test_setRejectedMaskMapping(self):
        mask_map = setRejectedMaskMapping(self.statsCtrl)
        self.assertEqual(len(mask_map), 3)
        # Check that all values are powers of 2
        for _, v in mask_map:
            logv = np.log2(v)
            self.assertEqual(logv, int(logv))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    """Check for resource/memory leaks."""


def setup_module(module):  # noqa: D103
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
