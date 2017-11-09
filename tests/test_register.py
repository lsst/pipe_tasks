from __future__ import division, print_function, absolute_import
from builtins import zip
from builtins import range
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

import unittest

import numpy as np

import lsst.utils.tests
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom
import lsst.afw.display.ds9 as ds9
from lsst.pipe.base import Struct
from lsst.pipe.tasks.registerImage import RegisterConfig, RegisterTask

try:
    display
except NameError:
    display = False


class RegisterTestCase(unittest.TestCase):

    """A test case for RegisterTask."""

    def setUp(self):
        self.dx = -5
        self.dy = +3
        self.numSources = 123
        self.border = 10  # Must be larger than dx,dy
        self.width = 1000
        self.height = 1000
        self.pixelScale = 0.1 * afwGeom.arcseconds  # So dx,dy is not larger than RegisterConfig.matchRadius

    def tearDown(self):
        del self.pixelScale

    def create(self):
        """Create test images and sources

        We will create two fake images with some 'sources', which are just single bright pixels.
        The images will have the same sources with a constant offset between them.  The WCSes
        of the two images are identical, despite the offset; this simulates a small e.g., pointing
        error, or misalignment that the RegisterTask should rectify.
        """
        np.random.seed(0)
        templateImage = afwImage.MaskedImageF(self.width, self.height)
        templateImage.set(0)
        inputImage = afwImage.MaskedImageF(self.width, self.height)
        inputImage.set(0)

        templateArray = templateImage.getImage().getArray()
        inputArray = inputImage.getImage().getArray()

        # Sources are at integer positions to ensure warped pixels have value of unity
        xTemplate = np.random.randint(self.border, self.width - self.border, self.numSources)
        yTemplate = np.random.randint(self.border, self.width - self.border, self.numSources)
        xInput = xTemplate + self.dx
        yInput = yTemplate + self.dy

        # Note: numpy indices are backwards: [y,x]
        templateArray[(yTemplate).astype(int), (xTemplate).astype(int)] = 1
        inputArray[(yInput).astype(int), (xInput).astype(int)] = 1

        # Create WCSes
        centerCoord = afwCoord.IcrsCoord(0*afwGeom.degrees, 0*afwGeom.degrees)
        centerPixel = afwGeom.Point2D(self.width/2, self.height/2)
        wcs = afwImage.makeWcs(centerCoord, centerPixel, self.pixelScale.asDegrees(), 0, 0,
                               self.pixelScale.asDegrees())

        # Note that one of the WCSes must be "wrong", since they are the same, but the sources are offset.
        # It is the job of the RegisterTask to align the images, despite the "wrong" WCS.
        templateExp = afwImage.makeExposure(templateImage, wcs)
        inputExp = afwImage.makeExposure(inputImage, wcs)

        # Generate catalogues
        schema = afwTable.SourceTable.makeMinimalSchema()
        centroidKey = afwTable.Point2DKey.addFields(schema, "center", "center", "pixel")

        def newCatalog():
            catalog = afwTable.SourceCatalog(schema)
            catalog.getTable().defineCentroid("center")
            return catalog

        templateSources = newCatalog()
        inputSources = newCatalog()

        coordKey = templateSources.getCoordKey()
        for xt, yt, xi, yi in zip(xTemplate, yTemplate, xInput, yInput):
            tRecord = templateSources.addNew()
            iRecord = inputSources.addNew()

            tPoint = afwGeom.Point2D(float(xt), float(yt))
            iPoint = afwGeom.Point2D(float(xi), float(yi))

            tRecord.set(centroidKey, tPoint)
            iRecord.set(centroidKey, iPoint)
            tRecord.set(coordKey, wcs.pixelToSky(tPoint))
            iRecord.set(coordKey, wcs.pixelToSky(iPoint))

        self.showImage(inputExp, inputSources, "Input", 1)
        self.showImage(templateExp, templateSources, "Template", 2)

        return Struct(xInput=xInput, yInput=yInput, xTemplate=xTemplate, yTemplate=yTemplate, wcs=wcs,
                      inputExp=inputExp, inputSources=inputSources,
                      templateExp=templateExp, templateSources=templateSources)

    def runTask(self, inData, config=RegisterConfig()):
        """Run the task on the data"""
        config.sipOrder = 2
        task = RegisterTask(name="register", config=config)
        results = task.run(inData.inputSources, inData.inputExp.getWcs(),
                           inData.inputExp.getBBox(afwImage.LOCAL), inData.templateSources)
        warpedExp = task.warpExposure(inData.inputExp, results.wcs, inData.templateExp.getWcs(),
                                      inData.templateExp.getBBox(afwImage.LOCAL))
        warpedSources = task.warpSources(inData.inputSources, results.wcs, inData.templateExp.getWcs(),
                                         inData.templateExp.getBBox(afwImage.LOCAL))

        self.showImage(warpedExp, warpedSources, "Aligned", 3)
        return Struct(warpedExp=warpedExp, warpedSources=warpedSources, matches=results.matches,
                      wcs=results.wcs, task=task)

    def assertRegistered(self, inData, outData, bad=set()):
        """Assert that the registration task is registering images"""
        xTemplate = np.array([x for i, x in enumerate(inData.xTemplate) if i not in bad])
        yTemplate = np.array([y for i, y in enumerate(inData.yTemplate) if i not in bad])
        alignedArray = outData.warpedExp.getMaskedImage().getImage().getArray()
        self.assertTrue((alignedArray[yTemplate, xTemplate] == 1.0).all())
        for dx in (-1, 0, +1):
            for dy in range(-1, 0, +1):
                # The density of points is such that I can assume that no point is next to another.
                # The values are not quite zero because the "image" is undersampled, so we get ringing.
                self.assertTrue((alignedArray[yTemplate+dy, xTemplate+dx] < 0.1).all())

        xAligned = np.array([x for i, x in enumerate(outData.warpedSources["center_x"]) if i not in bad])
        yAligned = np.array([y for i, y in enumerate(outData.warpedSources["center_y"]) if i not in bad])
        self.assertAlmostEqual((xAligned - xTemplate).mean(), 0, 8)
        self.assertAlmostEqual((xAligned - xTemplate).std(), 0, 8)
        self.assertAlmostEqual((yAligned - yTemplate).mean(), 0, 8)
        self.assertAlmostEqual((yAligned - yTemplate).std(), 0, 8)

    def assertMetadata(self, outData, numRejected=0):
        """Assert that the registration task is populating the metadata"""
        metadata = outData.task.metadata
        self.assertEqual(metadata.get("MATCH_NUM"), self.numSources)
        self.assertAlmostEqual(metadata.get("SIP_RMS"), 0.0)
        self.assertEqual(metadata.get("SIP_GOOD"), self.numSources-numRejected)
        self.assertEqual(metadata.get("SIP_REJECTED"), numRejected)

    def testRegister(self):
        """Test image registration"""
        inData = self.create()
        outData = self.runTask(inData)
        self.assertRegistered(inData, outData)
        self.assertMetadata(outData)

    def testRejection(self):
        """Test image registration with rejection"""
        inData = self.create()

        # Tweak a source to have a bad offset
        badIndex = 111

        coordKey = inData.inputSources[badIndex].getTable().getCoordKey()
        centroidKey = inData.inputSources[badIndex].getTable().getCentroidKey()
        x, y = float(inData.xInput[badIndex] + 0.01), float(inData.yInput[badIndex] - 0.01)
        point = afwGeom.Point2D(x, y)
        inData.inputSources[badIndex].set(centroidKey, point)
        inData.inputSources[badIndex].set(coordKey, inData.wcs.pixelToSky(point))

        config = RegisterConfig()
        config.sipRej = 10.0

        outData = self.runTask(inData)
        self.assertRegistered(inData, outData, bad=set([badIndex]))
        self.assertMetadata(outData, numRejected=1)

    def showImage(self, image, sources, title, frame):
        """Display an image

        Images are only displayed if 'display' is turned on.

        @param image: Image to display
        @param sources: Sources to mark on the display
        @param title: Title to give frame
        @param frame: Frame on which to display
        """
        if not display:
            return
        ds9.mtv(image, title=title, frame=frame)
        with ds9.Buffering():
            for s in sources:
                center = s.getCentroid()
                ds9.dot("o", center.getX(), center.getY(), frame=frame)


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
