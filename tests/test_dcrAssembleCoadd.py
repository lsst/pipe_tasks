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

from lsst.pipe.tasks.dcrAssembleCoadd import DcrAssembleCoaddTask, DcrAssembleCoaddConfig


class DcrAssembleCoaddCalculateGainTestCase(lsst.utils.tests.TestCase):
    """Tests of dcrAssembleCoaddTask.calculateGain()."""
    def setUp(self):
        self.baseGain = 0.5
        self.gainList = [self.baseGain, self.baseGain]
        self.convergenceList = [0.2]
        # Calculate the convergence we would expect if the model was converging perfectly,
        # so that the improvement is limited only by our conservative gain.
        for i in range(2):
            self.convergenceList.append(self.convergenceList[i]/(self.baseGain + 1))
        self.nextGain = (1 + self.baseGain) / 2

        self.config = DcrAssembleCoaddConfig()
        self.config.effectiveWavelength = 500.0
        self.config.bandwidth = 100.0
        self.task = DcrAssembleCoaddTask(config=self.config)

    def testUnbalancedLists(self):
        gainList = [1, 2, 3, 4]
        convergenceList = [1, 2]
        with self.assertRaises(ValueError):
            self.task.calculateGain(convergenceList, gainList)

    def testNoProgressiveGain(self):
        self.config.useProgressiveGain = False
        self.config.baseGain = self.baseGain
        expectGain = self.baseGain
        expectGainList = self.gainList + [expectGain]
        result = self.task.calculateGain(self.convergenceList, self.gainList)
        self.assertEqual(result, expectGain)
        self.assertEqual(self.gainList, expectGainList)

    def testBaseGainNone(self):
        """If baseGain is None, gain is calculated from the default values."""
        self.config.useProgressiveGain = False
        expectGain = 1 / (self.config.dcrNumSubfilters - 1)
        expectGainList = self.gainList + [expectGain]
        result = self.task.calculateGain(self.convergenceList, self.gainList)
        self.assertEqual(result, expectGain)
        self.assertEqual(self.gainList, expectGainList)

    def testProgressiveFirstStep(self):
        """The first and second steps always return baseGain."""
        convergenceList = self.convergenceList[:1]
        gainList = []
        self.config.baseGain = self.baseGain
        expectGain = self.baseGain
        expectGainList = [expectGain]
        result = self.task.calculateGain(convergenceList, gainList)
        self.assertEqual(result, expectGain)
        self.assertEqual(gainList, expectGainList)

    def testProgressiveSecondStep(self):
        """The first and second steps always return baseGain."""
        convergenceList = self.convergenceList[:2]
        gainList = self.gainList[:1]
        self.config.baseGain = self.baseGain
        expectGain = self.baseGain
        expectGainList = gainList + [expectGain]
        result = self.task.calculateGain(convergenceList, gainList)
        self.assertEqual(result, expectGain)
        self.assertEqual(gainList, expectGainList)

    def testProgressiveGain(self):
        """Test that gain follows the "perfect" situation defined in setUp."""
        self.config.baseGain = self.baseGain
        expectGain = self.nextGain
        expectGainList = self.gainList + [expectGain]
        result = self.task.calculateGain(self.convergenceList, self.gainList)
        self.assertFloatsAlmostEqual(result, expectGain)
        self.assertEqual(self.gainList, expectGainList)

    def testProgressiveGainBadFit(self):
        """Test that gain is reduced if the predicted convergence does not
        match the measured convergence (in this case, converging too quickly).
        """
        wrongGain = 1.0
        gainList = [self.baseGain, self.baseGain]
        convergenceList = [0.2]
        for i in range(2):
            convergenceList.append(convergenceList[i]/(wrongGain + 1))
        # The below math is a simplified version of the full algorithm,
        # assuming the predicted convergence is zero.
        # Note that in this case, nextGain is smaller than wrongGain.
        nextGain = (self.baseGain + (1 + self.baseGain) / (1 + wrongGain)) / 2

        self.config.baseGain = self.baseGain
        expectGain = nextGain
        expectGainList = self.gainList + [expectGain]
        result = self.task.calculateGain(convergenceList, gainList)
        self.assertFloatsAlmostEqual(result, nextGain)
        self.assertEqual(gainList, expectGainList)


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
