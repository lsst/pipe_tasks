#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import unittest

import lsst.afw.image as afwImage
import lsst.utils.tests
from lsst.pipe.tasks.scaleZeroPoint import ScaleZeroPointTask


class ScaleZeroPointTaskTestCase(lsst.utils.tests.TestCase):

    """A test case for ScaleZeroPointTask
    """

    def testBasics(self):
        for outZeroPoint in (23, 24):
            config = ScaleZeroPointTask.ConfigClass()
            config.zeroPoint = outZeroPoint
            zpScaler = ScaleZeroPointTask(config=config)
            outPhotoCalib = zpScaler.getPhotoCalib()

            self.assertAlmostEqual(outPhotoCalib.instFluxToMagnitude(1.0), outZeroPoint)

            for inZeroPoint in (24, 25.5):
                exposure = afwImage.ExposureF(10, 10)
                mi = exposure.getMaskedImage()
                mi.set(1.0)
                var = mi.getVariance()
                var.set(1.0)

                inPhotoCalib = self.makePhotoCalib(inZeroPoint)
                exposure.setPhotoCalib(inPhotoCalib)
                imageScaler = zpScaler.computeImageScaler(exposure)

                predScale = 1.0 / inPhotoCalib.magnitudeToInstFlux(outZeroPoint)
                self.assertAlmostEqual(predScale, imageScaler._scale)

                inFluxAtOutZeroPoint = exposure.getPhotoCalib().magnitudeToInstFlux(outZeroPoint)
                outFluxAtOutZeroPoint = outPhotoCalib.magnitudeToInstFlux(outZeroPoint)
                self.assertAlmostEqual(outFluxAtOutZeroPoint / imageScaler._scale, inFluxAtOutZeroPoint)

                inFluxMag0 = exposure.getPhotoCalib().getInstFluxAtZeroMagnitude()
                outFluxMag0 = outPhotoCalib.getInstFluxAtZeroMagnitude()
                self.assertFloatsAlmostEqual(outFluxMag0 / imageScaler._scale, inFluxMag0, rtol=5e-15)

                imageScaler.scaleMaskedImage(mi)
                self.assertAlmostEqual(mi.image[1, 1, afwImage.LOCAL], predScale)
                self.assertAlmostEqual(mi.variance[1, 1, afwImage.LOCAL], predScale**2)

    def makePhotoCalib(self, zeroPoint):
        fluxMag0 = 10**(0.4 * zeroPoint)
        return afwImage.makePhotoCalibFromCalibZeroPoint(fluxMag0, 1.0)


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
