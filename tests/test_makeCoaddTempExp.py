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
        # TODO: this is needed until DM-10153 is done and Calib is gone
        referenceFlux = 1e23 * 10**(48.6 / -2.5) * 1e9
        self.exposure.getCalib().setFluxMag0(referenceFlux/meanCalibration,
                                             referenceFlux*calibrationErr/meanCalibration**2)
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
        """Test that getCalibratedExposure returns expected Calib and WCS
        """
        self._checkCalibratedExposure(doApplyUberCal=False, includeCalibVar=True)
        self._checkCalibratedExposure(doApplyUberCal=False, includeCalibVar=False)
        self._checkCalibratedExposure(doApplyUberCal=True, includeCalibVar=True)
        self._checkCalibratedExposure(doApplyUberCal=True, includeCalibVar=False)

    def _checkCalibratedExposure(self, doApplyUberCal, includeCalibVar):
        self.config.doApplyUberCal = doApplyUberCal
        self.config.includeCalibVar = includeCalibVar
        task = MakeCoaddTempExpTask(config=self.config)

        photoCalib = self.jointcalPhotoCalib if doApplyUberCal else self.exposurePhotoCalib
        expect = photoCalib.calibrateImage(self.exposure.maskedImage, includeCalibVar)
        expect /= photoCalib.getCalibrationMean()
        result = task.getCalibratedExposure(self.dataRef, True)
        self.assertMaskedImagesEqual(result.maskedImage, expect)
        # TODO: once RFC-545 is implemented, this should be 1.0
        self.assertEqual(result.getCalib().getFluxMag0()[0], photoCalib.getInstFluxAtZeroMagnitude())

        targetWcs = self.jointcalSkyWcs if doApplyUberCal else self.skyWcs
        self.assertEqual(result.getWcs(), targetWcs)


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
