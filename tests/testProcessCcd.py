#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008-2013 LSST Corporation.
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

"""Test ProcessCcdTask

Run the task on one obs_test image and perform various checks on the results
"""

import os
import shutil
import tempfile
import unittest

import numpy

import eups
import lsst.afw.geom as afwGeom
import lsst.utils.tests
from lsst.pipe.tasks.processCcd import ProcessCcdTask

obsTestDir = eups.productDir("obs_test")
if obsTestDir is None:
    raise RuntimeError("obs_test must be setup")
InputDir = os.path.join(obsTestDir, "data", "input")

class ProcessCcdTestCase(lsst.utils.tests.TestCase):
    def testProcessCcd(self):
        outPath = tempfile.mkdtemp()
        try:
            fullResult = ProcessCcdTask.parseAndRun(
                args = [InputDir, "--output", outPath, "--id", "visit=1"],
                doReturnResults = True,
            )
            self.assertEqual(len(fullResult.resultList), 1)
            runResult = fullResult.resultList[0]
            result = runResult.result

            self.assertEqual(len(result.backgrounds), 4)
            bg0Arr = result.backgrounds[0][0].getImageF().getArray()
            self.assertAlmostEqual(bg0Arr.mean(), 325.5736, places=3)
            self.assertAlmostEqual(bg0Arr.std(),    0.0825, places=3)

            self.assertEqual(len(result.calib.sources), 28)
            psfShape = result.calib.psf.computeShape()
            self.assertAlmostEqual(psfShape.getIxx(), 2.8800, places=3)
            self.assertAlmostEqual(psfShape.getIyy(), 2.2986, places=3)
            self.assertAlmostEqual(psfShape.getIxy(), 0.1880, places=3)

            self.assertEqual(result.exposure.getBBox(),
                afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(1018, 2000)))
            maskedImage = result.exposure.getMaskedImage()
            maskArr = maskedImage.getMask().getArray()
            self.assertGreater(numpy.sum(maskArr == 0), 1900000) # most pixels are good
            imageArr = maskedImage.getImage().getArray()
            self.assertAlmostEqual(imageArr.mean(),   1.9073, places=3)
            self.assertAlmostEqual(imageArr.std(),  145.8494, places=3)
            varArr = maskedImage.getVariance().getArray()
            self.assertAlmostEqual(varArr.mean(), 246.1445, places=3)
            self.assertAlmostEqual(varArr.std(),  104.8250, places=3)

            self.assertEqual(len(result.sources), 167)

        finally:
            shutil.rmtree(outPath)


def suite():
    lsst.utils.tests.init()
    suites = []
    suites += unittest.makeSuite(ProcessCcdTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
