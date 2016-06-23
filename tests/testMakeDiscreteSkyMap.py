#!/usr/bin/env python

#
# LSST Data Management System
# Copyright 2012 LSST Corporation.
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

import os.path
import sys
import shutil
import unittest

from lsst.utils import getPackageDir
import lsst.utils.tests as utilsTests
from lsst.afw.geom import Extent2I, Box2D
from lsst.daf.persistence import Butler
from lsst.pipe.tasks.makeDiscreteSkyMap import MakeDiscreteSkyMapTask, DiscreteSkyMap


class MakeDiscreteSkyMapTestCase(unittest.TestCase):
    """Test MakeDiscreteSkyMapTask"""

    def setUp(self):
        self.inPath = os.path.join(getPackageDir("obs_test"), "data", "input")
        self.outPath = os.path.join(os.path.dirname(__file__), "testMakeDiscreteSkyMapOutput")
        self.config = MakeDiscreteSkyMapTask.ConfigClass()
        self.config.doWrite = False  # repo has no place to put the data

    def tearDown(self):
        del self.config
        if True:
            shutil.rmtree(self.outPath)

    def testBasics(self):
        """Test construction of a discrete sky map
        """
        butler = Butler(root=self.inPath, outputRoot=self.outPath)
        coordList = []  # list of sky coords of all corners of all calexp
        for dataId in (
            dict(visit=1, filter="g"),
            dict(visit=2, filter="g"),
            dict(visit=3, filter="r"),
        ):
            rawImage = butler.get("raw", dataId)
            # fake calexp by simply copying raw data; the task just cares about its bounding box
            # (which is slightly larger for raw, but that doesn't matter for this test)
            calexp = rawImage
            butler.put(calexp, "calexp", dataId)
            calexpWcs = calexp.getWcs()
            calexpBoxD = Box2D(calexp.getBBox())
            coordList += [calexpWcs.pixelToSky(corner) for corner in calexpBoxD.getCorners()]

        # use the calexp to make a sky map
        retVal = MakeDiscreteSkyMapTask.parseAndRun(
            args=[self.inPath, "--output", self.outPath, "--id", "filter=g^r"],
            config=self.config,
            doReturnResults=True,
        )
        self.assertEqual(len(retVal.resultList), 1)
        skyMap = retVal.resultList[0].result.skyMap
        self.assertEqual(type(skyMap), DiscreteSkyMap)
        self.assertEqual(len(skyMap), 1)
        tractInfo = skyMap[0]
        self.assertEqual(tractInfo.getId(), 0)
        self.assertEqual(tractInfo.getNumPatches(), Extent2I(3, 3))
        tractWcs = tractInfo.getWcs()
        tractBoxD = Box2D(tractInfo.getBBox())
        for skyPoint in coordList:
            self.assertTrue(tractBoxD.contains(tractWcs.skyToPixel(skyPoint)))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(MakeDiscreteSkyMapTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)


def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    if "--display" in sys.argv:
        display = True
    run(True)
