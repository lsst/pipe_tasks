#
# LSST Data Management System
# Copyright 2018 University of Washington
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
import unittest.mock
import numpy as np

import lsst.utils.tests
import lsst.afw.image
import lsst.daf.persistence
import lsst.geom
import lsst.afw.geom
from lsst.pipe.tasks import selectImages


class BestSeeingWcsSelectImagesTest(lsst.utils.tests.TestCase):

    def setUp(self):
        self.config = selectImages.BestSeeingWcsSelectImageConfig()
        self.plateScale = 0.2  # arcseconds/pixel
        # define an avgFwhm that is in the config's FWHM range
        self.avgFwhm = np.mean([self.config.minPsfFwhm, self.config.maxPsfFwhm])
        self.dataId = "visit=mock"

    def makeDataId(self, fwhm):
        """Provide convenience function for dataId mangling.
        Parameters
        ----------
        fwhm: float
            FWHM in arcseconds of image to be mocked.
        """
        return f'{self.dataId} fwhm={fwhm}'

    def localSetUp(self, mockFWHMs=None):
        """Set up test inputs with mocked variable seeing images.

        Parameters
        ----------
        mockFWHMs : iterable or `None`
            FWHM in arcseconds of images to be mocked.
        """
        if mockFWHMs is not None:
            self.mockFWHMs = mockFWHMs
        else:
            self.mockFWHMs = []

        def mockDeterminantRadii():
            return [f / np.sqrt(8.*np.log(2.)) for f in self.mockFWHMs]

        self.calexp = unittest.mock.Mock(spec=lsst.afw.image.ExposureD)
        self.calexp.getPsf.return_value.computeShape.return_value.getDeterminantRadius.side_effect \
            = mockDeterminantRadii()
        self.calexp.getWcs.return_value.getPixelScale.return_value.asArcseconds.side_effect \
            = [self.plateScale for f in self.mockFWHMs]
        self.dataRef = unittest.mock.Mock(spec=lsst.daf.persistence.ButlerDataRef, dataId=self.dataId)
        self.dataRef.get.return_value = self.calexp

        point1 = lsst.geom.SpherePoint(0, 0, lsst.geom.degrees)
        point2 = lsst.geom.SpherePoint(2, 0, lsst.geom.degrees)
        point3 = lsst.geom.SpherePoint(0, 2, lsst.geom.degrees)

        self.coordList = [point1, point2, point3]

        center = lsst.geom.SpherePoint(1, 1, lsst.geom.degrees)
        rotateAxis = lsst.geom.SpherePoint(0, 0, lsst.geom.degrees)
        rotateAngle = 0*lsst.geom.degrees
        dims = lsst.geom.Extent2I(3600, 3600)
        scale = 0.5*lsst.geom.arcseconds

        crpix = lsst.geom.Point2D(lsst.geom.Extent2D(dims)*0.5)
        center = center.rotated(rotateAxis, rotateAngle)
        cdMatrix = lsst.afw.geom.makeCdMatrix(scale=scale)
        wcs = lsst.afw.geom.makeSkyWcs(crpix=crpix, crval=center, cdMatrix=cdMatrix)
        self.bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Point2I(dims[0], dims[1]))

        self.selectList = []
        for fwhm in self.mockFWHMs:
            # encode the FWHM in the dataId
            dataRef = unittest.mock.Mock(spec=lsst.daf.persistence.ButlerDataRef,
                                         dataId=self.makeDataId(fwhm))
            dataRef.get.return_value = self.calexp
            # also create a new attribute to store the fwhm since side_effect
            # can only be called once per value
            dataRef.fwhm = fwhm
            selectStruct = selectImages.SelectStruct(dataRef, wcs, self.bbox)
            self.selectList.append(selectStruct)

    def testConfigNImagesGTZero(self):
        """Test that the config requests 0 or more images.
        """
        self.assertGreater(self.config.nImagesMax, 0)

    def testConfigPsfFwhmRangeIsValid(self):
        """Test that the minimum PSF FWHM is greater than zero
        and less than the maximum.
        """
        self.assertGreaterEqual(self.config.minPsfFwhm, 0)
        self.assertGreater(self.config.maxPsfFwhm, self.config.minPsfFwhm)

    def testNoInputImages(self):
        """Test that providing no images returns nothing.
        """
        self.localSetUp(mockFWHMs=[])
        task = selectImages.BestSeeingWcsSelectImagesTask(config=self.config)
        result = task.runDataRef(self.dataRef, self.coordList, selectDataList=self.selectList)
        self.assertEqual(len(result.exposureInfoList), 0)

    def testSelectOne(self):
        """Test that configs requesting one image return one image.
        """
        self.config.nImagesMax = 1
        self.localSetUp(mockFWHMs=[self.avgFwhm for i in range(self.config.nImagesMax)])
        task = selectImages.BestSeeingWcsSelectImagesTask(config=self.config)
        result = task.runDataRef(self.dataRef, self.coordList, selectDataList=self.selectList)
        self.assertEqual(len(result.exposureInfoList), 1)
        self.assertEqual(result.exposureInfoList[0].dataId, self.makeDataId(self.avgFwhm))
        self.assertEqual(result.dataRefList[0].fwhm, self.avgFwhm)

    def testSelectDefault(self):
        """Test that the default configuration is self-consistent.
        """
        self.localSetUp(mockFWHMs=[self.avgFwhm for i in range(self.config.nImagesMax)])
        task = selectImages.BestSeeingWcsSelectImagesTask(config=self.config)
        result = task.runDataRef(self.dataRef, self.coordList, selectDataList=self.selectList)
        self.assertEqual(len(result.exposureInfoList), self.config.nImagesMax)
        for dataRef in result.dataRefList:
            self.assertEqual(dataRef.fwhm, self.avgFwhm)

    def testNImagesLessThanNMax(self):
        """Test case of input selectDataList shorter than nImagesMax.
        """
        self.config.nImagesMax = 5
        self.localSetUp(mockFWHMs=[self.avgFwhm for i in range(3)])
        task = selectImages.BestSeeingWcsSelectImagesTask(config=self.config)
        result = task.runDataRef(self.dataRef, self.coordList, selectDataList=self.selectList)
        self.assertEqual(len(result.exposureInfoList), 3)
        for dataRef in result.dataRefList:
            self.assertEqual(dataRef.fwhm, self.avgFwhm)

    def testNImagesGreaterThanNMax(self):
        """Test case of input selectDataList greater than nImagesMax.
        """
        self.localSetUp(mockFWHMs=[self.avgFwhm for i in range(self.config.nImagesMax+3)])
        task = selectImages.BestSeeingWcsSelectImagesTask(config=self.config)
        result = task.runDataRef(self.dataRef, self.coordList, selectDataList=self.selectList)
        self.assertEqual(len(result.exposureInfoList), self.config.nImagesMax)
        for dataRef in result.dataRefList:
            self.assertEqual(dataRef.fwhm, self.avgFwhm)

    def testFwhmSelection(self):
        """Test that minimum and maximum FWHM selections work.
        """
        self.config.minPsfFwhm = 1.0*self.plateScale
        self.config.maxPsfFwhm = 1.5*self.plateScale
        self.config.nImagesMax = 10
        mockFWHMs = [0.5, 1.0, 1.25, 1.5, 2.0]
        self.localSetUp(mockFWHMs=mockFWHMs)
        task = selectImages.BestSeeingWcsSelectImagesTask(config=self.config)
        result = task.runDataRef(self.dataRef, self.coordList, selectDataList=self.selectList)
        self.assertEqual(len(result.exposureInfoList), 3)
        for i, fwhm in enumerate([1.0, 1.25, 1.5]):
            self.assertEqual(result.exposureInfoList[i].dataId, self.makeDataId(fwhm))
            self.assertEqual(result.dataRefList[i].fwhm, fwhm)

    def testBestSeeingFwhmSelection(self):
        """Test that the selection picks the best seeing images.
        """
        self.config.minPsfFwhm = 1.0*self.plateScale
        self.config.maxPsfFwhm = 1.5*self.plateScale
        self.config.nImagesMax = 3
        mockFWHMs = [0.5, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 2.0]
        self.localSetUp(mockFWHMs=mockFWHMs)
        task = selectImages.BestSeeingWcsSelectImagesTask(config=self.config)
        result = task.runDataRef(self.dataRef, self.coordList, selectDataList=self.selectList)
        self.assertEqual(len(result.exposureInfoList), self.config.nImagesMax)
        for i, fwhm in enumerate([1.0, 1.1, 1.2]):
            self.assertEqual(result.exposureInfoList[i].dataId, self.makeDataId(fwhm))
            self.assertEqual(result.dataRefList[i].fwhm, fwhm)

    def testSelectDefaultMakeDataRefListIsFalse(self):
        """Test appropriate behavior for makeDataRefList = False.
        """
        self.localSetUp(mockFWHMs=[self.avgFwhm for i in range(self.config.nImagesMax)])
        task = selectImages.BestSeeingWcsSelectImagesTask(config=self.config)
        result = task.runDataRef(self.dataRef, self.coordList,
                                 selectDataList=self.selectList, makeDataRefList=False)
        self.assertIsNone(result.dataRefList)


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
