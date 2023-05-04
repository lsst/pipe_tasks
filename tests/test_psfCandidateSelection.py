#
# LSST Data Management System
# Copyright 2008-2017 AURA/LSST
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
import os
import unittest
import logging

import lsst.afw.image as afwImage
import lsst.utils.tests
import lsst.meas.extensions.piff.piffPsfDeterminer
from lsst.utils import getPackageDir
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig


class PsfFlagTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        # Load sample input from disk
        expPath = os.path.join(getPackageDir("pipe_tasks"), "tests", "data", "v695833-e0-c000-a00.sci.fits")
        self.exposure = afwImage.ExposureF(expPath)
        # set log level so that warnings do not display
        logging.getLogger("lsst.characterizeImage").setLevel(logging.ERROR)

    def tearDown(self):
        del self.exposure

    def testFlags(self):
        # Test that all of the flags are defined and there is no reservation by default
        # also test that used sources are a subset of candidate sources
        config = CharacterizeImageConfig()
        config.measurePsf.psfDeterminer = 'piff'
        config.measurePsf.psfDeterminer['piff'].spatialOrder = 0
        config.measureApCorr.sourceSelector["science"].doSignalToNoise = False
        task = CharacterizeImageTask(config=config)
        results = task.run(self.exposure)
        used = 0
        reserved = 0
        for source in results.sourceCat:
            if source.get("calib_psf_used"):
                used += 1
                self.assertTrue(source.get("calib_psf_candidate"))
            if source.get("calib_psf_reserved"):
                reserved += 1
        self.assertGreater(used, 0)
        self.assertEqual(reserved, 0)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
