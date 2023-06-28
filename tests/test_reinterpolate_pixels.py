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

import lsst.ip.isr as ipIsr
import lsst.utils.tests
from lsst.meas.algorithms import ReinterpolatePixelsConfig, ReinterpolatePixelsTask


class TestReinterpolatePixels(lsst.utils.tests.TestCase):
    """Test that ReinterpolatePixelsTask for meas_algorithms produces expected
    outputs.
    """

    def setUp(self):
        self.inputExp = ipIsr.isrMock.TrimmedRawMock().run()
        self.mi = self.inputExp.getMaskedImage()

    def test_reinterpolate_pixels(self):
        """Expect number of interpolated pixels to be non-zero."""
        ipIsr.makeThresholdMask(self.mi, 200, growFootprints=2, maskName="SAT")
        config = ReinterpolatePixelsConfig()
        config.maskNameList = ["SAT"]
        task = ReinterpolatePixelsTask(config=config)
        task.run(self.inputExp)
        interpMaskedImage = self.inputExp.getMaskedImage()
        numBit = ipIsr.countMaskedPixels(interpMaskedImage, "INTRP")
        self.assertEqual(numBit, 40800)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
