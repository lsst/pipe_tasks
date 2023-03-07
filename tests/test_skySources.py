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
import logging

import lsst.afw.image as afwImage
import lsst.utils.tests
from lsst.utils import getPackageDir
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig
from lsst.pipe.tasks.calibrate import CalibrateTask, CalibrateConfig


class SkySourcesTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        # Load sample input from disk
        expPath = os.path.join(getPackageDir("pipe_tasks"), "tests", "data", "v695833-e0-c000-a00.sci.fits")
        self.exposure = afwImage.ExposureF(expPath)
        # set log level so that warnings do not display
        logging.getLogger("lsst.calibrate").setLevel(logging.ERROR)

    def tearDown(self):
        del self.exposure

    def testDoSkySources(self):
        """Tests sky_source column gets added when run.
        """
        self._checkSkySourceColumnExistence(doSkySources=True)
        self._checkSkySourceColumnExistence(doSkySources=False)

    def _checkSkySourceColumnExistence(self, doSkySources):
        """Implements sky_source column checking.

        Parameters
        ----------
        doSkySource : `bool`
            Value of the config flag determining whether to insert sky sources.
        """
        charImConfig = CharacterizeImageConfig()
        charImConfig.measurePsf.psfDeterminer = 'piff'
        charImConfig.measurePsf.psfDeterminer['piff'].spatialOrder = 0
        charImConfig.measureApCorr.sourceSelector["science"].doSignalToNoise = False
        charImTask = CharacterizeImageTask(config=charImConfig)
        charImResults = charImTask.run(self.exposure)
        calibConfig = CalibrateConfig()
        calibConfig.doAstrometry = False
        calibConfig.doPhotoCal = False
        calibConfig.doSkySources = doSkySources
        calibConfig.doComputeSummaryStats = False
        calibTask = CalibrateTask(config=calibConfig)
        calibResults = calibTask.run(charImResults.exposure)
        if doSkySources:
            self.assertTrue('sky_source' in calibResults.outputCat.schema.getNames())
        else:
            self.assertFalse('sky_source' in calibResults.outputCat.schema.getNames())


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
