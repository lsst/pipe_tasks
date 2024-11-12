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

from __future__ import annotations

import unittest

import numpy as np

import lsst.utils.tests

import lsst.afw.image

from lsst.pipe.tasks.make_direct_warp import MakeDirectWarpTask, WarpDetectorInputs
from lsst.pipe.tasks.make_psf_matched_warp import MakePsfMatchedWarpTask
import lsst.afw.cameraGeom.testUtils

from test_make_direct_warp import MakeWarpTestCase


class MakePsfMatchedWarpTestCase(MakeWarpTestCase):
    def test_makeWarp(self):
        """Test basic MakePsfMatchedWarpTask

        This constructs a direct_warp using `MakeDirectWarpTask` and then
        runs `MakePsfMatchedWarpTask` on it.
        """

        makeWarp = MakeDirectWarpTask(config=self.config)
        warp_detector_inputs = {
            self.dataRef.dataId.detector.id: WarpDetectorInputs(
                exposure_or_handle=self.dataRef, data_id=self.dataRef.dataId
            )
        }
        result = makeWarp.run(
            warp_detector_inputs,
            sky_info=self.skyInfo,
            visit_summary=None
        )

        warp = result.warp

        config = MakePsfMatchedWarpTask.ConfigClass()
        makePsfMatchedWarp = MakePsfMatchedWarpTask(config=config)
        result = makePsfMatchedWarp.run(
            warp,
            bbox=self.skyInfo.bbox,
        )

        psf_matched_warp = result.psf_matched_warp
        # Ensure we got an exposure out
        self.assertIsInstance(psf_matched_warp, lsst.afw.image.ExposureF)
        # Check that the PSF is not None.
        psf = psf_matched_warp.getPsf()
        assert psf is not None
        # Ensure the warp has valid pixels
        self.assertGreater(np.isfinite(psf_matched_warp.image.array.ravel()).sum(), 0)


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
