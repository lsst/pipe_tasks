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

import lsst.utils
import lsst.afw.geom as afwGeom
import lsst.utils.tests
from lsst.pipe.tasks.processCcd import ProcessCcdTask

obsTestDir = lsst.utils.getPackageDir('obs_test')
InputDir = os.path.join(obsTestDir, "data", "input")

OutputName = None # specify a name (as a string) to save the output repository

class ProcessCcdTestCase(lsst.utils.tests.TestCase):
    def testProcessCcd(self):
        """test ProcessCcdTask

        This is intended as a sanity check of the task, not a detailed test of its sub-tasks. As such
        comparisons are intentionally loose, so as to allow evolution of the sub-tasks over time
        without breaking this test.
        """
        outPath = tempfile.mkdtemp() if OutputName is None else OutputName
        try:
            dataId = dict(visit=1)
            dataIdStrList = ["%s=%s" % (key, val) for key, val in dataId.iteritems()]
            fullResult = ProcessCcdTask.parseAndRun(
                args = [InputDir, "--output", outPath, "--clobber-config", "--id"] + dataIdStrList,
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

            icExpBackground = butler.get("icExpBackground", dataId, immediate=True)
            bg0Arr = icExpBackground.getImage().getArray()
            bgMean = bg0Arr.mean(dtype=numpy.float64)
            bgStdDev = bg0Arr.std(dtype=numpy.float64)

            icSrc = butler.get("icSrc", dataId)
            src = butler.get("src", dataId)

            # the following makes pyflakes linter happy and the code more robust
            oldImMean = None
            oldImStdDev = None
            oldVarMean = None
            oldVarStdDev = None
            oldPsfIxx = None
            oldPsfIyy = None
            oldPsfIxy = None

            for i, exposure in enumerate((
                butler.get("calexp", dataId),
                result.exposure,
            )):
                self.assertEqual(exposure.getBBox(),
                    afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(1018, 2000)))
                maskedImage = exposure.getMaskedImage()
                maskArr = maskedImage.getMask().getArray()
                numGoodPix = numpy.sum(maskArr == 0)

                imageArr = maskedImage.getImage().getArray()
                imMean = imageArr.mean(dtype=numpy.float64)
                imStdDev = imageArr.std(dtype=numpy.float64)
                varArr = maskedImage.getVariance().getArray()
                varMean = varArr.mean(dtype=numpy.float64)
                varStdDev = varArr.std(dtype=numpy.float64)

                psfShape = exposure.getPsf().computeShape()
                psfIxx = psfShape.getIxx()
                psfIyy = psfShape.getIyy()
                psfIxy = psfShape.getIxy()

                if i == 0:
                    print("\nMeasured results:")
                    print("background mean = %r, stdDev = %r" % (bgMean, bgStdDev))
                    print("len(icSrc) :", len(icSrc))
                    print("len(src) :", len(src))

                    print("numGoodPix =", numGoodPix)
                    print("image mean = %r, stdDev = %r" % (imMean, imStdDev))
                    print("variance mean = %r, stdDev = %r" % (varMean, varStdDev))
                    print("psf Ixx = %r, Iyy = %r, Ixy = %r" % (psfIxx, psfIyy, psfIxy))

                    self.assertAlmostEqual(bgMean, 327.73706832701379, places=7)
                    self.assertAlmostEqual(bgStdDev, 0.478942723145386, places=7)
                    self.assertEqual(len(icSrc), 28)
                    self.assertEqual(len(src), 183)
                    self.assertEqual(numGoodPix, 1966976)

                    self.assertAlmostEqual(imMean,      1.6992922142160001, places=7)
                    self.assertAlmostEqual(imStdDev,  163.66265133203555, places=7)
                    self.assertAlmostEqual(varMean,   212.5416436210289, places=7)
                    self.assertAlmostEqual(varStdDev, 121.25084470503148, places=7)
                    self.assertAlmostEqual(psfIxx, 2.205197497931509, places=7)
                    self.assertAlmostEqual(psfIyy, 2.0893651175583723, places=7)
                    self.assertAlmostEqual(psfIxy,  -0.07647011192069166, places=7)
                else:
                    self.assertEqual(imMean, oldImMean)
                    self.assertEqual(imStdDev, oldImStdDev)
                    self.assertEqual(varMean, oldVarMean)
                    self.assertEqual(varStdDev, oldVarStdDev)
                    self.assertEqual(psfIxx, oldPsfIxx)
                    self.assertEqual(psfIyy, oldPsfIyy)
                    self.assertEqual(psfIxy, oldPsfIxy)

                oldImMean = imMean
                oldImStdDev = imStdDev
                oldVarMean = varMean
                oldVarStdDev = varStdDev
                oldPsfIxx = psfIxx
                oldPsfIyy = psfIyy
                oldPsfIxy = psfIxy

        finally:
            if OutputName is None:
                shutil.rmtree(outPath)
            else:
                print("testProcessCcd.py's output data saved to %r" % (OutputName,))


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
