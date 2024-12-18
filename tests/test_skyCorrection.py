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
from copy import deepcopy

import lsst.utils.tests
import numpy as np
from lsst.afw.image import ExposureF
from lsst.afw.math import BackgroundMI
from lsst.obs.base.instrument_tests import DummyCam
from lsst.pipe.tasks.skyCorrection import SkyCorrectionConfig, SkyCorrectionTask


class SkyCorrectionTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        dummyCam = DummyCam()
        self.camera = dummyCam.getCamera()
        bbox = self.camera[0].getBBox()

        # Configs below set to approximate HSC defaults
        self.skyCorrectionConfig = SkyCorrectionConfig()
        self.skyCorrectionConfig.doMaskObjects = False
        # Set bgModel1 size to a single bin for the whole plane (aka constant)
        self.skyCorrectionConfig.bgModel1.xSize = 8192 * 0.015
        self.skyCorrectionConfig.bgModel1.ySize = 8192 * 0.015
        self.skyCorrectionConfig.bgModel1.pixelSize = 0.015
        self.skyCorrectionConfig.bgModel2.xSize = 256 * 0.015
        self.skyCorrectionConfig.bgModel2.ySize = 256 * 0.015
        self.skyCorrectionConfig.bgModel2.pixelSize = 0.015

        # Generate calexp/calexpBackground/sky for all detectors
        self.calExps = []
        self.calBkgs = []
        self.skyFrames = []
        self.background_level = 3000
        self.sky_level = 5
        for detector in [0, 1]:
            rng = np.random.default_rng(detector)

            # Science image
            calexp = ExposureF(bbox)
            calexp.maskedImage.set(0.0, 0x0, 650.0)
            calexp.setDetector(self.camera[detector])
            # Add a sky frame signature to a subregion of the image
            sky_frame_bin_size = 32
            x_start = 32 * sky_frame_bin_size
            x_stop = 64 * sky_frame_bin_size
            y_start = 31 * sky_frame_bin_size
            y_stop = 63 * sky_frame_bin_size
            calexp.image.array[:, x_start:x_stop] += self.sky_level
            calexp.image.array[y_start:y_stop, :] += self.sky_level
            # Add random noise
            calexp.image.array += rng.normal(0.0, 25.0, (bbox.getDimensions().y, bbox.getDimensions().x))
            self.calExps.append(calexp)

            # Background image
            backgroundImage = ExposureF(bbox)
            backgroundImage.maskedImage.set(0.0, 0x0, 1.0)
            backgroundImage.setDetector(self.camera[detector])
            backgroundImage.image.array += self.background_level
            background = BackgroundMI(bbox, backgroundImage.getMaskedImage())
            calexpBackground = lsst.afw.math.BackgroundList(
                (
                    background,
                    lsst.afw.math.Interpolate.CONSTANT,
                    lsst.afw.math.UndersampleStyle.REDUCE_INTERP_ORDER,
                    lsst.afw.math.ApproximateControl.UNKNOWN,
                    0,
                    0,
                    False,
                )
            )
            self.calBkgs.append(calexpBackground)

            # Sky frame
            sky = ExposureF(128, 125)
            sky.maskedImage.set(0.0, 0x0, 1.0)
            sky.setDetector(self.camera[detector])
            header = sky.getMetadata()
            header.set("BOX.MINX", bbox.getMinX())
            header.set("BOX.MINY", bbox.getMinY())
            header.set("BOX.MAXX", bbox.getMaxX())
            header.set("BOX.MAXY", bbox.getMaxY())
            header.set("ALGORITHM", "NATURAL_SPLINE")
            sky.image.array[:, 32:64] += 1  # x
            sky.image.array[31:63, :] += 1  # y
            # Add random noise
            sky.image.array += rng.normal(0.0, 0.1, (125, 128))
            sky.image.array -= np.sum(sky.image.array) / (125 * 128)
            self.skyFrames.append(sky)

    def tearDown(self):
        del self.camera
        del self.calExps
        del self.calBkgs
        del self.skyFrames

    def testSkyCorrectionDefault(self):
        """Test SkyCorrectionTask with mostly default configuration values."""

        skyCorrectionTask = SkyCorrectionTask(config=self.skyCorrectionConfig)
        # Pass in deep copies, as the task modifies the input data
        results = skyCorrectionTask.run(
            deepcopy(self.calExps), deepcopy(self.calBkgs), self.skyFrames, self.camera
        )
        skyFrameScale = results.skyFrameScale
        skyCorr = results.skyCorr
        self.assertEqual(len(skyCorr), len(self.calExps))
        self.assertAlmostEqual(skyFrameScale, self.sky_level, delta=1e-1)
        self.assertAlmostEqual(np.nanmean(results.calExpMosaic.array), 0, delta=1e-2)

    def testSkyCorrectionSkyFrameOnly(self):
        """Test SkyCorrectionTask with the config undoBgModel1 set to True."""

        skyCorrectionConfig = deepcopy(self.skyCorrectionConfig)
        skyCorrectionConfig.undoBgModel1 = True
        skyCorrectionConfig.doBgModel2 = False
        skyCorrectionTask = SkyCorrectionTask(config=skyCorrectionConfig)
        # Pass in deep copies, as the task modifies the input data
        results = skyCorrectionTask.run(
            deepcopy(self.calExps), deepcopy(self.calBkgs), self.skyFrames, self.camera
        )
        self.assertAlmostEqual(
            np.nanmean(results.calExpMosaic.array),
            np.nanmean(self.calExps[0].image.array) + self.background_level,
            delta=1e-2,
        )


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
