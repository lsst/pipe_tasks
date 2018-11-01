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
import lsst.daf.persistence
from lsst.pipe.tasks.coaddBase import CoaddBaseTask, CoaddBaseConfig


class CoaddBaseTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        np.random.seed(10)

        self.config = CoaddBaseConfig()
        self.task = CoaddBaseTask(self.config, "testCoaddBase")

        self.exposure = lsst.afw.image.ExposureF(10, 10)
        self.exposure.maskedImage.image.array = np.random.random((10, 10)).astype(np.float32) * 1000
        self.exposure.maskedImage.variance.array = np.random.random((10, 10)).astype(np.float32)
        # mask at least one pixel
        self.exposure.maskedImage.mask[5, 5] = 3

        # TODO DM-10156: once Calib is replaced, this will be much cleaner
        meanCalibration = 1e-4
        calibrationErr = 1e-5
        self.exposure.getCalib().setFluxMag0(1/meanCalibration, calibrationErr/meanCalibration**2)
        self.exposurePhotoCalib = lsst.afw.image.PhotoCalib(meanCalibration, calibrationErr)

        # a "jointcal_photoCalib" calibration to return
        self.jointcalPhotoCalib = lsst.afw.image.PhotoCalib(1e-6, 1e-8)

        # set to False in a test to return self.jointcalPhotoCalib for get('jointcal_PhotoCalib'
        self.raiseOnGetPhotoCalib = True

        def mockGet(datasetType, dataId=None, immediate=None):
            """Minimally fake a butler.get()."""
            if "calexp" in datasetType:
                return self.exposure.clone()
            if "jointcal_photoCalib" in datasetType:
                if self.raiseOnGetPhotoCalib:
                    raise lsst.daf.persistence.NoResults("failed!", datasetType, dataId)
                else:
                    return self.jointcalPhotoCalib

        self.dataRef = unittest.mock.Mock(spec=lsst.daf.persistence.ButlerDataRef)
        self.dataRef.get.side_effect = mockGet
        self.dataRef.dataId = {"ccd": 10000, "visit": 1}
        # unittest.mock.patch()
        # butlerPatcher = unittest.mock.patch("lsst.daf.persistence.Butler", autospec=True)
        # self._butler = butlerPatcher.start()
        # self._butler.getMapperClass.return_value = lsst.obs.test.TestMapper
        # self._butler.return_value.get = mockGet
        # self.addCleanup(butlerPatcher.stop)

    def test_getCalibratedExposure(self):
        expect = self.exposurePhotoCalib.calibrateImage(self.exposure.maskedImage)
        result = self.task.getCalibratedExposure(self.dataRef, True)

        self.assertMaskedImagesEqual(result.maskedImage, expect)
        # The new exposure has a flux calibration of 1.
        self.assertEqual(result.getCalib().getFluxMag0()[0], 1.0)

    def test_getCalibratedExposureJointcalNoDataset(self):
        self.config.doApplyJointcal = True
        with(self.assertRaises(RuntimeError)):
            self.task.getCalibratedExposure(self.dataRef, True)

    def test_getCalibratedExposureJointcal(self):
        self.config.doApplyJointcal = True
        self.raiseOnGetPhotoCalib = False

        expect = self.jointcalPhotoCalib.calibrateImage(self.exposure.maskedImage)
        result = self.task.getCalibratedExposure(self.dataRef, True)
        self.assertMaskedImagesEqual(result.maskedImage, expect)
        self.assertEqual(result.getCalib().getFluxMag0()[0], 1.0)


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
