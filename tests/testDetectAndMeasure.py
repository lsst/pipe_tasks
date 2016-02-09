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
"""Test DetectAndMeasureTask
"""
import os
import unittest

import lsst.utils.tests
from lsst.utils import getPackageDir
from lsst.afw.geom import Box2I, Point2I, Extent2I
from lsst.afw.table import SourceTable
from lsst.daf.persistence import Butler
from lsst.daf.butlerUtils import ExposureIdInfo
from lsst.meas.algorithms.testUtils import plantSources
from lsst.pipe.tasks.detectAndMeasure import DetectAndMeasureTask

try:
    obsTestDir = getPackageDir("obs_test")
    InputDir = os.path.join(obsTestDir, "data", "input")
except Exception:
    InputDir = None

class ProcessCcdTestCase(lsst.utils.tests.TestCase):
    def testBasics(self):
        """Test detection and measurement on simple synthesized data
        """
        bbox = Box2I(Point2I(256, 100), Extent2I(128, 127))
        minCounts = 5000
        maxCounts = 50000
        starSigma = 1.5
        numX = 5
        numY = 5
        coordList = self.makeCoordList(
            bbox = bbox,
            numX = numX,
            numY = numY,
            minCounts = minCounts,
            maxCounts = maxCounts,
            sigma = starSigma,
        )
        kwid = 11 # kernel width
        sky = 2000
        # create an exposure without a Wcs; add the Wcs later
        exposure = plantSources(bbox=bbox, kwid=kwid, sky=sky, coordList=coordList, addPoissonNoise=True)

        schema = SourceTable.makeMinimalSchema()
        psfUsedKey = schema.addField("calib_psfUsed", type="Flag") # needed to measure aperture correction
        def setPsfUsed(sourceCat):
            # claim that all detected sources make good PSF candidates
            for source in sourceCat:
                source.set(psfUsedKey, True)

        config = DetectAndMeasureTask.ConfigClass()
        config.measurement.doApplyApCorr = "yes"
        task = DetectAndMeasureTask(config=config, schema=schema)

        butler = Butler(root=InputDir)
        dataRef = butler.dataRef("calexp", dataId=dict(visit=1))
        wcs = dataRef.get("raw").getWcs()
        exposure.setWcs(wcs)
        exposureIdInfo = ExposureIdInfo.fromDataRef(dataRef)
        taskRes = task.run(exposure=exposure, exposureIdInfo=exposureIdInfo, preMeasApCorrFunc=setPsfUsed)
        self.assertEqual(len(taskRes.sourceCat), numX * numY)
        schema = taskRes.sourceCat.schema
        centroidFlagKey = schema.find("slot_Centroid_flag").getKey()
        parentKey = schema.find("parent").getKey()
        psfFluxFlagKey = schema.find("slot_PsfFlux_flag").getKey()
        psfFluxKey = schema.find("slot_PsfFlux_flux").getKey()
        for src in taskRes.sourceCat:
            self.assertFalse(src.get(centroidFlagKey))  # centroid found
            self.assertEqual(src.get(parentKey), 0)     # not debelended
            self.assertFalse(src.get(psfFluxFlagKey))   # flux measured
            self.assertGreater(src.get(psfFluxKey), 4000)  # flux sane

    def makeCoordList(self, bbox, numX, numY, minCounts, maxCounts, sigma):
        """Make a coordList for plantSources

        Coords are uniformly spaced in a rectangular grid, with linearly increasing counts
        """
        dX = bbox.getWidth() / float(numX)
        dY = bbox.getHeight() / float(numY)
        minX = bbox.getMinX() + (dX / 2.0)
        minY = bbox.getMinY() + (dY / 2.0)
        dCounts = (maxCounts - minCounts) / (numX * numY - 1)
    
        coordList = []
        counts = minCounts
        for i in range(numX):
            x = minX + (dX * i)
            for j in range(numY):
                y = minY + (dY * j)
                coordList.append([x, y, counts, sigma])
                counts += dCounts
        return coordList

    def testNoPsfUsed(self):
        """Test that the "calib_psfUsed" is required to measure aperture correction

        I hope someday DetectAndMeasureTask can determine for itself
        which sources are suitable for measuring aperture correction,
        at which point I expect this test to be deleted.
        """
        schema = SourceTable.makeMinimalSchema()
        config = DetectAndMeasureTask.ConfigClass()
        with self.assertRaises(Exception):
            DetectAndMeasureTask(config=config, schema=schema)

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
