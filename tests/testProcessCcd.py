#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
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
        """test ProcessCcdTask

        This is intended as a sanity check of the task, not a detailed test of its sub-tasks. As such
        comparisons are intentionally loose, so as to allow evolution of the sub-tasks over time
        without breaking this test.
        """
        outPath = tempfile.mkdtemp()
        try:
            dataId = dict(visit=1)
            dataIdStrList = ["%s=%s" % (key, val) for key, val in dataId.iteritems()]
            fullResult = ProcessCcdTask.parseAndRun(
                args = [InputDir, "--output", outPath, "--id"] + dataIdStrList,
                doReturnResults = True,
            )
            butler = fullResult.parsedCmd.butler
            self.assertEqual(len(fullResult.resultList), 1)
            runResult = fullResult.resultList[0]
            fullDataId = runResult.dataRef.dataId
            self.assertEqual(len(fullDataId), 2)
            self.assertEqual(fullDataId["visit"], dataId["visit"])
            self.assertEqual(fullDataId["filter"], "g")
            result = runResult.result

            calexpBackground = butler.get("calexpBackground", dataId)
            self.assertGreaterEqual(len(calexpBackground), 1)
            bg0Arr = calexpBackground.getImage().getArray()
            self.assertAlmostEqual(bg0Arr.mean(dtype=numpy.float64), 325.4484, places=1)
            self.assertLess(bg0Arr.std(dtype=numpy.float64), 1.0) # 0.40578011 as of 2014-11

            icSrc = butler.get("icSrc", dataId)
            self.assertGreater(len(icSrc), 20) # 28 as of 2014-11

            for exposure in (
                butler.get("calexp", dataId),
                result.exposure,
            ):
                self.assertEqual(exposure.getBBox(),
                    afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(1018, 2000)))
                maskedImage = exposure.getMaskedImage()
                maskArr = maskedImage.getMask().getArray()
                self.assertGreater(numpy.sum(maskArr == 0, dtype=numpy.float64), 1900000) # most pixels are good
                imageArr = maskedImage.getImage().getArray()
                self.assertAlmostEqual(imageArr.mean(dtype=numpy.float64),   1.9073, places=1)
                self.assertAlmostEqual(imageArr.std(dtype=numpy.float64),  145.8494, places=1)
                varArr = maskedImage.getVariance().getArray()
                self.assertAlmostEqual(varArr.mean(dtype=numpy.float64), 246.1445, places=1)
                self.assertAlmostEqual(varArr.std(dtype=numpy.float64),  104.8250, places=1)

                psfShape = exposure.getPsf().computeShape()
                self.assertAlmostEqual(psfShape.getIxx(), 2.8800, places=1)
                self.assertAlmostEqual(psfShape.getIyy(), 2.2986, places=1)
                self.assertAlmostEqual(psfShape.getIxy(), 0.1880, places=1)

            sources = butler.get("src", dataId)
            self.assertGreater(len(sources), 100) # 167 as of 2014-11

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
