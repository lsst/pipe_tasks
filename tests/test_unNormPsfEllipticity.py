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
import lsst.meas.extensions.piff.piffPsfDeterminer
import lsst.pipe.base as pipeBase
import lsst.utils.tests
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig

TESTDIR = os.path.abspath(os.path.dirname(__file__))


class UnNormPsfEllipticityTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        # Load sample input from disk
        expPath = os.path.join(TESTDIR, "data", "v695833-e0-c000-a00.sci.fits")
        self.exposure = afwImage.ExposureF(expPath)
        self.band = self.exposure.filter.bandLabel

    def tearDown(self):
        del self.exposure

    def testUnNormPsfEllipticity(self):
        """Tests that UnprocessableDataError is raised when the unnormalized
           PSF model ellipticity excedes config.maxUnNormPsfEllipticity.
        """
        self._checkUnNormPsfEllipticity(allowPass=True)
        self._checkUnNormPsfEllipticity(allowPass=False)

    def testUnNormPsfEllipticityFallback(self):
        """Tests that the fallback value is set when band is missing from
           config.maxUnNormPsfEllipticityPerBand.
        """
        charImConfig = CharacterizeImageConfig()
        charImConfig.measurePsf.psfDeterminer = 'piff'
        charImConfig.measurePsf.psfDeterminer['piff'].spatialOrder = 0
        charImConfig.measureApCorr.sourceSelector["science"].doSignalToNoise = False
        # Pop the band entry from the config dict to impose need for fallback.
        tempDict = charImConfig.maxUnNormPsfEllipticityPerBand
        tempDict.pop(self.band)
        charImConfig.maxUnNormPsfEllipticityPerBand = tempDict
        charImTask = CharacterizeImageTask(config=charImConfig)
        charImTask.run(self.exposure)
        with self.assertRaises(pipeBase.UnprocessableDataError):
            charImConfig.maxUnNormPsfEllipticityFallback = 0.0
            charImTask = CharacterizeImageTask(config=charImConfig)
            charImTask.run(self.exposure)

    def _checkUnNormPsfEllipticity(self, allowPass):
        """Check unnormalized model PSF ellipticity threshold functionality.

        Parameters
        ----------
        allowPass : `bool`
            Whether to update from the default config to allow this exporsure
            to pass the threshold check.
        """
        charImConfig = CharacterizeImageConfig()
        charImConfig.measurePsf.psfDeterminer = 'piff'
        charImConfig.measurePsf.psfDeterminer['piff'].spatialOrder = 0
        charImConfig.measureApCorr.sourceSelector["science"].doSignalToNoise = False
        if not allowPass:
            charImConfig.maxUnNormPsfEllipticityPerBand[self.band] = 0.0
        charImTask = CharacterizeImageTask(config=charImConfig)
        if allowPass:
            charImTask.run(self.exposure)
        else:
            with self.assertRaises(pipeBase.UnprocessableDataError):
                charImTask.run(self.exposure)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
