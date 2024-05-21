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

import numpy as np

import lsst.utils.tests

import lsst.afw.image
from lsst.pipe.tasks.makeWarp import (MakeWarpTask, MakeWarpConfig)
from lsst.pipe.tasks.coaddBase import makeSkyInfo
import lsst.skymap as skyMap
from lsst.afw.detection import GaussianPsf
import lsst.afw.cameraGeom.testUtils


class MakeWarpTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        np.random.seed(12345)

        self.config = MakeWarpConfig()

        meanCalibration = 1e-4
        calibrationErr = 1e-5
        self.exposurePhotoCalib = lsst.afw.image.PhotoCalib(meanCalibration, calibrationErr)
        # An external photoCalib calibration to return
        self.externalPhotoCalib = lsst.afw.image.PhotoCalib(1e-6, 1e-8)

        crpix = lsst.geom.Point2D(0, 0)
        crval = lsst.geom.SpherePoint(0, 45, lsst.geom.degrees)
        cdMatrix = lsst.afw.geom.makeCdMatrix(scale=1.0*lsst.geom.arcseconds)
        self.skyWcs = lsst.afw.geom.makeSkyWcs(crpix, crval, cdMatrix)
        externalCdMatrix = lsst.afw.geom.makeCdMatrix(scale=0.9*lsst.geom.arcseconds)
        # An external skyWcs to return
        self.externalSkyWcs = lsst.afw.geom.makeSkyWcs(crpix, crval, externalCdMatrix)

        self.exposure = lsst.afw.image.ExposureF(10, 10)
        self.exposure.maskedImage.image.array = np.random.random((10, 10)).astype(np.float32) * 1000
        self.exposure.maskedImage.variance.array = np.random.random((10, 10)).astype(np.float32)
        # mask at least one pixel
        self.exposure.maskedImage.mask[5, 5] = 3
        # set the PhotoCalib and Wcs objects of this exposure.
        self.exposure.setPhotoCalib(lsst.afw.image.PhotoCalib(meanCalibration, calibrationErr))
        self.exposure.setWcs(self.skyWcs)
        self.exposure.setPsf(GaussianPsf(5, 5, 2.5))
        self.exposure.setFilter(lsst.afw.image.FilterLabel(physical="fakeFilter", band="fake"))

        self.visit = 100
        self.detector = 5
        detectorName = f"detector {self.detector}"
        detector = lsst.afw.cameraGeom.testUtils.DetectorWrapper(name=detectorName, id=self.detector).detector
        self.exposure.setDetector(detector)

        simpleMapConfig = skyMap.discreteSkyMap.DiscreteSkyMapConfig()
        simpleMapConfig.raList = [crval.getRa().asDegrees()]
        simpleMapConfig.decList = [crval.getDec().asDegrees()]
        simpleMapConfig.radiusList = [0.1]

        self.simpleMap = skyMap.DiscreteSkyMap(simpleMapConfig)
        self.tractId = 0
        self.patchId = self.simpleMap[0].findPatch(crval).sequential_index
        self.skyInfo = makeSkyInfo(self.simpleMap, self.tractId, self.patchId)

    def test_makeWarp(self):
        """Test basic MakeWarpTask."""
        makeWarp = MakeWarpTask(config=self.config)

        result = makeWarp.run(
            calExpList=[self.exposure],
            ccdIdList=[self.detector],
            skyInfo=self.skyInfo,
            visitId=self.visit,
            dataIdList=[{'visit': self.visit, 'detector': self.detector}],
        )

        warp = result.exposures['direct']

        # Ensure we got an exposure out
        self.assertIsInstance(warp, lsst.afw.image.ExposureF)
        # Ensure the warp has valid pixels
        self.assertGreater(np.isfinite(warp.image.array.ravel()).sum(), 0)
        # Ensure the warp has the correct WCS
        self.assertEqual(warp.getWcs(), self.skyInfo.wcs)

    def test_psfMatched(self):
        self.config.makePsfMatched = False
        task1 = MakeWarpTask(config=self.config)
        self.config.makePsfMatched = True
        task2 = MakeWarpTask(config=self.config)

        result1 = task1.run(
            calExpList=[self.exposure],
            ccdIdList=[self.detector],
            skyInfo=self.skyInfo,
            visitId=self.visit,
            dataIdList=[{'visit': self.visit, 'detector': self.detector}],
        )

        result2 = task2.run(
            calExpList=[self.exposure],
            ccdIdList=[self.detector],
            skyInfo=self.skyInfo,
            visitId=self.visit,
            dataIdList=[{'visit': self.visit, 'detector': self.detector}],
        )

        # Ensure we got an exposure out
        self.assertIsInstance(result2.exposures['psfMatched'], lsst.afw.image.ExposureF)
        self.assertMaskedImagesEqual(result1.exposures['direct'].maskedImage,
                result2.exposures['direct'].maskedImage)


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
