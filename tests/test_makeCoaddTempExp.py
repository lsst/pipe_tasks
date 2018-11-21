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
import unittest.mock

import numpy as np

import lsst.utils.tests

import lsst.afw.image
from lsst.daf.persistence import NoResults, ButlerDataRef
from lsst.pipe.tasks.makeCoaddTempExp import (MakeCoaddTempExpTask,
                                              MakeCoaddTempExpConfig,
                                              MissingExposureError)


class GetCalibratedExposureTestCase(lsst.utils.tests.TestCase):
    """Tests of MakeCoaddTempExpTask.getCalibratedExposure()"""
    def setUp(self):
        np.random.seed(10)

        self.config = MakeCoaddTempExpConfig()

        # TODO DM-10156: once Calib is replaced, this will be much cleaner
        meanCalibration = 1e-4
        calibrationErr = 1e-5
        self.exposurePhotoCalib = lsst.afw.image.PhotoCalib(meanCalibration, calibrationErr)
        # a "jointcal_photoCalib" calibration to return
        self.jointcalPhotoCalib = lsst.afw.image.PhotoCalib(1e-6, 1e-8)

        crpix = lsst.geom.Point2D(0, 0)
        crval = lsst.geom.SpherePoint(0, 45, lsst.geom.degrees)
        cdMatrix = lsst.afw.geom.makeCdMatrix(scale=1.0*lsst.geom.arcseconds)
        self.skyWcs = lsst.afw.geom.makeSkyWcs(crpix, crval, cdMatrix)
        jointcalCdMatrix = lsst.afw.geom.makeCdMatrix(scale=0.9*lsst.geom.arcseconds)
        # a "jointcal_wcs" skyWcs to return
        self.jointcalSkyWcs = lsst.afw.geom.makeSkyWcs(crpix, crval, jointcalCdMatrix)

        self.exposure = lsst.afw.image.ExposureF(10, 10)
        self.exposure.maskedImage.image.array = np.random.random((10, 10)).astype(np.float32) * 1000
        self.exposure.maskedImage.variance.array = np.random.random((10, 10)).astype(np.float32)
        # mask at least one pixel
        self.exposure.maskedImage.mask[5, 5] = 3
        # set the Calib and Wcs objects of this exposure.
        self.exposure.getCalib().setFluxMag0(1/meanCalibration, calibrationErr/meanCalibration**2)
        self.exposure.setWcs(self.skyWcs)

        # set to True in a test to raise NoResults for get('calexp')
        self.raiseOnGetCalexp = False

        def mockGet(datasetType, dataId=None, immediate=None):
            """Minimally fake a butler.get()."""
            if "calexp" in datasetType:
                if self.raiseOnGetCalexp:
                    raise NoResults("failed!", datasetType, dataId)
                else:
                    return self.exposure.clone()
            if "jointcal_photoCalib" in datasetType:
                return self.jointcalPhotoCalib
            if "jointcal_wcs" in datasetType:
                return self.jointcalSkyWcs

        self.dataRef = unittest.mock.Mock(spec=ButlerDataRef)
        self.dataRef.get.side_effect = mockGet
        self.dataRef.dataId = {"ccd": 10000, "visit": 1}

    def test_getCalibratedExposureGetCalexpRaises(self):
        """If get('calexp') raises NoResults, we should get a usefully
        chained exception.
        """
        task = MakeCoaddTempExpTask(config=self.config)

        self.raiseOnGetCalexp = True

        with self.assertRaises(MissingExposureError) as cm:
            task.getCalibratedExposure(self.dataRef, True)
        self.assertIsInstance(cm.exception.__cause__, NoResults)
        self.assertIn('Exposure not found', str(cm.exception))

    def test_getCalibratedExposure(self):
        task = MakeCoaddTempExpTask(config=self.config)

        expect = self.exposurePhotoCalib.calibrateImage(self.exposure.maskedImage)
        expect /= self.exposurePhotoCalib.getCalibrationMean()
        result = task.getCalibratedExposure(self.dataRef, True)

        self.assertMaskedImagesEqual(result.maskedImage, expect)
        # TODO: once RFC-545 is implemented, this should be 1.0
        self.assertEqual(result.getCalib().getFluxMag0()[0], 1/self.exposurePhotoCalib.getCalibrationMean())
        self.assertEqual(result.getWcs(), self.skyWcs)

    def test_getCalibratedExposureJointcal(self):
        self.config.doApplyUberCal = True
        task = MakeCoaddTempExpTask(config=self.config)

        expect = self.jointcalPhotoCalib.calibrateImage(self.exposure.maskedImage)
        expect /= self.jointcalPhotoCalib.getCalibrationMean()
        result = task.getCalibratedExposure(self.dataRef, True)
        self.assertMaskedImagesEqual(result.maskedImage, expect)
        # TODO: once RFC-545 is implemented, this should be 1.0
        self.assertEqual(result.getCalib().getFluxMag0()[0], 1/self.jointcalPhotoCalib.getCalibrationMean())
        self.assertEqual(result.getWcs(), self.jointcalSkyWcs)


class MakeCoaddTempExpRunTestCase(lsst.utils.tests.TestCase):
    """Tests of MakeCoaddTempExpTask.run()."""
    def setUp(self):
        self.config = MakeCoaddTempExpConfig()
        self.task = MakeCoaddTempExpTask(self.config)
        dataId = "visit=mock"
        fakeDataRef = unittest.mock.NonCallableMock(lsst.daf.persistence.ButlerDataRef,
                                                    dataId=dataId,
                                                    butlerSubset=unittest.mock.Mock())
        self.calexpRefList = [fakeDataRef]

        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(100, 100))
        self.skyInfo = unittest.mock.Mock(bbox=bbox)

        self.fakeImage = lsst.afw.image.ExposureF(bbox)
        self.fakeImage.getMaskedImage().set(np.nan, lsst.afw.image.Mask.getPlaneBitMask("NO_DATA"), np.inf)

        target = "lsst.pipe.tasks.makeCoaddTempExp.MakeCoaddTempExpTask._prepareEmptyExposure"
        preparePatch = unittest.mock.patch(target, return_value=self.fakeImage)
        preparePatch.start()
        self.addCleanup(preparePatch.stop)

    def testGetCalibratedExposureRaisesRuntimeError(self):
        """If getCalibratedExposure() raises anything other than
        MissingExposureError, it should be passed up the chain.

        Previously, all Exceptions were caught, logged at `warn` level,
        and then dropped.
        """
        mockErr = "Mock Error!"
        target = "lsst.pipe.tasks.makeCoaddTempExp.MakeCoaddTempExpTask.getCalibratedExposure"
        patch = unittest.mock.patch(target,
                                    new_callable=unittest.mock.Mock,
                                    side_effect=RuntimeError(mockErr))
        patch.start()
        self.addCleanup(patch.stop)
        with self.assertRaises(RuntimeError) as cm:
            self.task.run(self.calexpRefList, self.skyInfo)
        self.assertIn(mockErr, str(cm.exception))

    def testGetCalibratedExposureRaisesMissingExposureError(self):
        """If getCalibratedExposure() raises MissingExposureError,
        processing should continue uninterrupted.
        In this case, that means no data is returned, because there is only
        one dataRef available (`self.fakeImage`).
        """
        mockErr = "No data files exist."
        target = "lsst.pipe.tasks.makeCoaddTempExp.MakeCoaddTempExpTask.getCalibratedExposure"
        patch = unittest.mock.patch(target,
                                    new_callable=unittest.mock.Mock,
                                    side_effect=MissingExposureError(mockErr))
        patch.start()
        self.addCleanup(patch.stop)
        result = self.task.run(self.calexpRefList, self.skyInfo)
        self.assertEqual(result.exposures, {"direct": None})


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
