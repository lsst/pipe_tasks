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

import lsst.utils.tests
import lsst.afw.image
import lsst.afw.math
from lsst.pipe.tasks.maskStreaks import MaskStreaksTask, LineProfile, Line, setDetectionMask


class TestMaskStreaks(lsst.utils.tests.TestCase):

    def setUp(self):
        self.config = MaskStreaksTask.ConfigClass()
        self.config.dChi2Tolerance = 1e-6
        self.fst = MaskStreaksTask(config=self.config)

        self.testx = 500
        self.testy = 600
        self.exposure = lsst.afw.image.ExposureF(self.testy, self.testx)
        rand = lsst.afw.math.Random(seed=98765)
        lsst.afw.math.randomGaussianImage(self.exposure.image, rand)
        self.exposure.maskedImage.variance.set(1)
        self.maskName = "STREAK"
        self.detectedPlane = "DETECTED"

    def test_binning(self):
        """Test the two binning methods and the no-binning method"""

        binSize = 4
        self.assertEqual(self.testx % binSize, 0)
        self.assertEqual(self.testy % binSize, 0)

        testExposure1 = self.exposure.clone()
        setDetectionMask(testExposure1, binning=binSize, detectedPlane=self.detectedPlane)
        mask1 = testExposure1.getMask()
        reshapeBinning = mask1.array & mask1.getPlaneBitMask(self.detectedPlane)
        testExposure2 = self.exposure.clone()
        with self.assertWarns(Warning):
            setDetectionMask(testExposure2, binning=binSize, forceSlowBin=True)
        mask2 = testExposure2.getMask()
        scipyBinning = mask2.array & mask2.getPlaneBitMask(self.detectedPlane)
        self.assertAlmostEqual(reshapeBinning.tolist(), scipyBinning.tolist())

    def test_canny(self):
        """Test that Canny filter returns binary of equal shape"""

        zeroExposure = lsst.afw.image.ExposureF(self.testy, self.testx)
        cannyZeroExposure = self.fst._cannyFilter(zeroExposure.image.array)
        self.assertEqual(cannyZeroExposure.tolist(), zeroExposure.image.array.tolist())

        exposure = self.exposure.clone()
        setDetectionMask(exposure, detectedPlane=self.detectedPlane)
        mask = exposure.getMask()
        processedImage = mask.array & mask.getPlaneBitMask(self.detectedPlane)
        cannyNonZero = self.fst._cannyFilter(processedImage)
        self.assertEqual(cannyNonZero.tolist(), cannyNonZero.astype(bool).tolist())

    def test_runkht(self):
        """Test the whole thing"""

        # Empty image:
        zeroArray = np.zeros((self.testx, self.testy))
        zeroLines = self.fst._runKHT(zeroArray)
        self.assertEqual(len(zeroLines), 0)
        testExposure = self.exposure.clone()
        result = self.fst.run(testExposure)
        self.assertEqual(len(result.lines), 0)
        resultMask = testExposure.mask.array & testExposure.mask.getPlaneBitMask(self.maskName)
        self.assertEqual(resultMask.tolist(), zeroArray.tolist())

        # Make image with line and check that input line is recovered:
        testExposure1 = self.exposure.clone()
        inputRho = 150
        inputTheta = 45
        inputSigma = 3
        testLine = Line(inputRho, inputTheta, inputSigma)
        lineProfile = LineProfile(testExposure.image.array, testExposure.variance.array**-1,
                                  line=testLine)
        testData = lineProfile.makeProfile(testLine, fitFlux=False)
        testExposure1.image.array = testData * 100
        detectedInd = abs(testData) > 0.1 * (abs(testData)).max()

        testExposure1.mask.addMaskPlane(self.detectedPlane)
        testExposure1.mask.array[detectedInd] |= testExposure1.mask.getPlaneBitMask(self.detectedPlane)
        testExposure2 = testExposure1.clone()
        setDetectionMask(testExposure2, detectedPlane=self.detectedPlane)

        result_withSetDetMask = self.fst.run(testExposure2)
        self.assertEqual(len(result_withSetDetMask.lines), 1)
        self.assertAlmostEqual(inputRho, result_withSetDetMask.lines[0].rho, places=2)
        self.assertAlmostEqual(inputTheta, result_withSetDetMask.lines[0].theta, places=2)
        self.assertAlmostEqual(inputSigma, result_withSetDetMask.lines[0].sigma, places=2)

        result = self.fst.run(testExposure1)
        self.assertEqual(len(result.lines), 1)
        self.assertAlmostEqual(inputRho, result.lines[0].rho, places=2)
        self.assertAlmostEqual(inputTheta, result.lines[0].theta, places=2)
        self.assertAlmostEqual(inputSigma, result.lines[0].sigma, places=2)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
