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

import os
import unittest

import lsst.afw.image as afwImage
import lsst.utils.tests
from lsst.utils import getPackageDir
from lsst.log import Log
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig
from lsst.pipe.tasks.calibrate import CalibrateTask, CalibrateConfig


class IsPrimaryTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        # Load sample input from disk
        expPath = os.path.join(getPackageDir("pipe_tasks"), "tests", "data", "v695833-e0-c000-a00.sci.fits")
        self.exposure = afwImage.ExposureF(expPath)
        # set log level so that warnings do not display
        Log.getLogger("calibrate").setLevel(Log.ERROR)

    def tearDown(self):
        del self.exposure

    def testIsPrimaryFlag(self):
        """Tests detect_isPrimary column gets added when run, and that sources
        labelled as detect_isPrimary are not sky sources and have no children.
        """
        charImConfig = CharacterizeImageConfig()
        charImTask = CharacterizeImageTask(config=charImConfig)
        charImResults = charImTask.run(self.exposure)
        calibConfig = CalibrateConfig()
        calibConfig.doAstrometry = False
        calibConfig.doPhotoCal = False
        calibTask = CalibrateTask(config=calibConfig)
        calibResults = calibTask.run(charImResults.exposure)
        outputCat = calibResults.outputCat
        self.assertTrue("detect_isPrimary" in outputCat.schema.getNames())
        # make sure all sky sources are flagged as not primary
        self.assertEqual(sum((outputCat["detect_isPrimary"]) & (outputCat["sky_source"])), 0)
        # make sure all parent sources are flagged as not primary
        self.assertEqual(sum((outputCat["detect_isPrimary"]) & (outputCat["deblend_nChild"] > 0)), 0)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
