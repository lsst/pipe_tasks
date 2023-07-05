#
# LSST Data Management System
# Copyright 2008-2016 LSST Corporation.
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
"""
Tests for lsst.afw.table.CoaddInputs

Note: generating the version 1 data requires some complicated machinations
because CoaddInputRecorderTask had bugs:
- Setup pipe_tasks 12.1, or a similar version old enough to have version 1 CoaddInput/ExposureTable
- Edit this file as follows:
    - Set SaveCoadd = True
    - Set self.version to 2 in CoaddInputsTestCase
- Edit CoaddInputRecorderTask as follows:
    - Apply all fixes to CoaddInputRecorderTask from DM-7976
    - Comment out the line that saves VisitInfo (since version 1 doesn't have it)
- Edit ups/pipe_tasks.table as follows:
    - Do not import obs_base (since it did not exist at that time)
- "setup -r . 'j" to setup this version of pipe_tasks
- "python tests/testCoaddInputs.py" to run this test.
    You will see some errors, but the needed data file will be written:
    tests/data/testCoaddInputs_coadd_with_version_1_data.fits
- Save the data file to the repository (but do not save any other changes).
"""
import os.path
import unittest

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
from lsst.daf.base import DateTime
import lsst.afw.cameraGeom.testUtils
from lsst.afw.coord import Observatory, Weather
import lsst.geom
import lsst.afw.geom
import lsst.afw.image
from lsst.afw.detection import GaussianPsf
from lsst.afw.math import ChebyshevBoundedField
from lsst.pipe.tasks.coaddInputRecorder import CoaddInputRecorderTask

SaveCoadd = False  # if True then save coadd even if test passes (always saved if a test fails)


class MockExposure:

    """Factory to make simple mock exposures suitable to put in a coadd

    The metadata is set, but not the pixels
    """

    def __init__(self, numExp, coaddInputRecorder, version=2):
        """Make a MockExposure

        @param[in] numExp  total number of exposures that will be made
        @param[in] coaddInputRecoder  an instance of CoaddInputRecorderTask
        @param[in] version  desired version of CoaddInput/ExposureTable;
                    1 for no VisitInfo; this will only produce the desired result
                        if you run the test with code from before VisitInfo, e.g. version 12.1)
                    2 to include VisitInfo
        """
        self.numExp = int(numExp)
        self.coaddInputRecorder = coaddInputRecorder
        self.version = int(version)

    def makeExposure(self, universalId):
        """Make a tiny exposure with exposure info set, but no pixels

        In particular, exposure info is set as a record in a table, so it can be recorded in a coadd
        """
        inputRecorder = self.coaddInputRecorder.makeCoaddTempExpRecorder(universalId, self.numExp)
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(100, 100), lsst.geom.Extent2I(10, 10))

        detectorName = "detector {}".format(universalId)
        detector = lsst.afw.cameraGeom.testUtils.DetectorWrapper(name=detectorName, id=universalId).detector

        exp = lsst.afw.image.ExposureF(bbox)
        exp.setDetector(detector)

        expInfo = exp.getInfo()
        expInfo.id = 10313423
        scale = 5.1e-5*lsst.geom.degrees
        cdMatrix = lsst.afw.geom.makeCdMatrix(scale=scale)
        wcs = lsst.afw.geom.makeSkyWcs(
            crpix=lsst.geom.Point2D(5, 5),
            crval=lsst.geom.SpherePoint(10, 45, lsst.geom.degrees),
            cdMatrix=cdMatrix,
        )
        expInfo.setWcs(wcs)
        expInfo.setPsf(GaussianPsf(5, 5, 2.5))
        expInfo.setPhotoCalib(lsst.afw.image.makePhotoCalibFromCalibZeroPoint(1.1e12, 2.2e10))
        expInfo.setApCorrMap(self.makeApCorrMap())
        expInfo.setValidPolygon(lsst.afw.geom.Polygon(lsst.geom.Box2D(bbox).getCorners()))
        expInfo.setFilter(lsst.afw.image.FilterLabel(physical="fakeFilter", band="fake"))
        if self.version > 1:
            expInfo.setVisitInfo(self.makeVisitInfo())

        inputRecorder.addCalExp(calExp=exp, ccdId=universalId, nGoodPix=100)
        inputRecorder.finish(coaddTempExp=exp, nGoodPix=100)

        return exp

    @staticmethod
    def makeWcs():
        scale = 5.1e-5*lsst.geom.degrees
        cdMatrix = lsst.afw.geom.makeCdMatrix(scale=scale)
        return lsst.afw.geom.makeSkyWcs(
            crpix=lsst.geom.Point2D(5, 5),
            crval=lsst.geom.SpherePoint(10, 45, lsst.geom.degrees),
            cdMatrix=cdMatrix,
        )

    @staticmethod
    def makeVisitInfo():
        return lsst.afw.image.VisitInfo(
            10.01,
            11.02,
            DateTime(65321.1, DateTime.MJD, DateTime.TAI),
            12345.1,
            45.1*lsst.geom.degrees,
            lsst.geom.SpherePoint(23.1, 73.2, lsst.geom.degrees),
            lsst.geom.SpherePoint(134.5, 33.3, lsst.geom.degrees),
            1.73,
            73.2*lsst.geom.degrees,
            lsst.afw.image.RotType.SKY,
            Observatory(11.1*lsst.geom.degrees, 22.2*lsst.geom.degrees, 0.333),
            Weather(1.1, 2.2, 34.5),
        )

    @staticmethod
    def makeApCorrMap():
        """Make a trivial ApCorrMap with three elements"""
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(-5, -5), lsst.geom.Point2I(5, 5))
        apCorrMap = lsst.afw.image.ApCorrMap()
        for name in ("a", "b", "c"):
            apCorrMap.set(name, ChebyshevBoundedField(bbox, np.zeros((3, 3), dtype=float)))
        return apCorrMap


class MockCoadd:

    """Class to make a mock coadd
    """

    def __init__(self, numExp, version=2):
        """Create a coadd with the specified number of exposures.

        @param[in] numExp  total number of exposures that will be made
        @param[in] version  desired version of CoaddInput/ExposureTable; see MockExposure for details

        Useful fields include:
        - coadd
        - exposures  a list of exposures that went into the coadd
        """
        coaddInputRecorder = CoaddInputRecorderTask(name="coaddInputRecorder")
        mockExposure = MockExposure(numExp=numExp, coaddInputRecorder=coaddInputRecorder, version=version)
        self.exposures = [mockExposure.makeExposure(i) for i in range(numExp)]

        exp0 = self.exposures[0]
        self.coadd = lsst.afw.image.ExposureF(exp0.getBBox())
        self.coadd.setWcs(exp0.getWcs())

        coaddInputs = coaddInputRecorder.makeCoaddInputs()
        for exp in self.exposures:
            coaddInputRecorder.addVisitToCoadd(coaddInputs=coaddInputs, coaddTempExp=exp, weight=1.0/numExp)
        self.coadd.getInfo().setCoaddInputs(coaddInputs)


class CoaddInputsTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        self.version = 2  # desired version of ExposureTable/CoaddInput
        self.numExp = 3
        mockCoadd = MockCoadd(numExp=self.numExp, version=self.version)
        self.coadd = mockCoadd.coadd
        self.exposures = mockCoadd.exposures
        self.dataDir = os.path.join(os.path.dirname(__file__), "data")

    def tearDown(self):
        del self.coadd
        del self.exposures

    def assertPsfsAlmostEqual(self, psf1, psf2):
        im1 = psf1.computeImage(psf1.getAveragePosition())
        im2 = psf2.computeImage(psf2.getAveragePosition())
        self.assertImagesAlmostEqual(im1, im2)

    def getCoaddPath(self, version):
        return os.path.join(self.dataDir,
                            "testCoaddInputs_coadd_with_version_{}_data.fits".format(version))

    def testPersistence(self):
        """Read and write a coadd and check the CoaddInputs"""
        coaddPath = self.getCoaddPath(version=self.version)
        self.coadd.writeFits(coaddPath)
        coadd = lsst.afw.image.ExposureF(coaddPath)
        coaddInputs = coadd.getInfo().getCoaddInputs()
        self.assertCoaddInputsOk(coaddInputs, version=self.version)
        if not SaveCoadd:
            os.unlink(coaddPath)
        else:
            print("SaveCoadd true; saved coadd as: %r" % (coaddPath,))

    def testReadV1Coadd(self):
        """Read a coadd that contains version 1 CoaddInputs

        The test code in question has FK5 WCS
        """
        coaddPath = self.getCoaddPath(version=1)
        print("coaddPath=", coaddPath)
        coadd = lsst.afw.image.ExposureF(coaddPath)
        coaddWcs = coadd.getWcs()
        # the exposure in question uses FK5 for its WCS so update the exposures
        for exposure in self.exposures:
            exposure.setWcs(coaddWcs)
        coaddInputs = coadd.getInfo().getCoaddInputs()
        self.assertCoaddInputsOk(coaddInputs, version=1)

    def assertCoaddInputsOk(self, coaddInputs, version):
        self.assertIsNotNone(coaddInputs)
        for expTable in (coaddInputs.ccds, coaddInputs.visits):
            self.assertEqual(len(expTable), 3)
            for i, expRec in enumerate(expTable):
                exp = self.exposures[i]
                expInfo = exp.getInfo()
                self.assertEqual(expRec.getId(), i)
                self.assertEqual(expRec.getBBox(), exp.getBBox())
                self.assertWcsAlmostEqualOverBBox(expRec.getWcs(), expInfo.getWcs(), expRec.getBBox())
                self.assertPsfsAlmostEqual(expRec.getPsf(), exp.getPsf())
                self.assertEqual(expRec.getPhotoCalib(), expInfo.getPhotoCalib())
                self.assertEqual(len(expRec.getApCorrMap()), 3)
                self.assertEqual(set(expRec.getApCorrMap().keys()), set(expInfo.getApCorrMap().keys()))
                self.assertFloatsAlmostEqual(np.array(expRec.getValidPolygon().getVertices()),
                                             np.array(expInfo.getValidPolygon().getVertices()))
                if version > 1:
                    self.assertEqual(expRec.getVisitInfo(), expInfo.getVisitInfo())
                else:
                    self.assertIsNone(expRec.getVisitInfo())


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
