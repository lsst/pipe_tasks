#
# LSST Data Management System
# Copyright 2008-2016 AURA/LSST.
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
# see <https://www.lsstcorp.org/LegalNotices/>.
#
"""Test configuration of CalexpCutoutTask and the run method.

Run the task on one test image and perform various checks on the results
"""
import json
import os
import unittest

import lsst.utils
import lsst.utils.tests
from lsst.afw.image import ExposureF
from lsst.pipe.tasks.calexpCutout import CalexpCutoutTask

packageDir = lsst.utils.getPackageDir('pipe_tasks')
datadir = os.path.join(packageDir, 'tests', "data")


class CalexpCutoutTestCase(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with open(os.path.join(datadir, 'cutout_data.json'), 'r') as fh:
            cls.data = json.load(fh)
        cls.exp = ExposureF.readFits(os.path.join(datadir, 'v695833-e0-c000-a00.sci.fits'))

    @classmethod
    def tearDownClass(cls):
        del cls.data
        del cls.exp

    def testCalexpCutout(self):
        config = CalexpCutoutTask.ConfigClass()
        task = CalexpCutoutTask(config=config)
        result = task.run(self.data['good'], self.exp)
        self.assertEqual(len(result.cutouts), len(self.data['good']['ra']))

        # Test configuration of the max number of cutouts
        config.max_cutouts = 4
        task = CalexpCutoutTask(config=config)
        result = task.run(self.data['good'], self.exp)
        self.assertEqual(len(result.cutouts), task.config.max_cutouts)

    def testEdge(self):
        # Currently edge cutouts are handled the same way
        # as cutouts completely outside the image.  That
        # could change in the future
        config = CalexpCutoutTask.ConfigClass()
        task = CalexpCutoutTask(config=config)
        result = task.run(self.data['edge'], self.exp)
        # Cutouts on the edge should be skipped
        self.assertEqual(len(result.cutouts), 0)

        config.skip_bad = False
        task = CalexpCutoutTask(config=config)
        # Should now raise for cutouts on the edge
        with self.assertRaises(ValueError):
            result = task.run(self.data['edge'], self.exp)

    def testBad(self):
        config = CalexpCutoutTask.ConfigClass()
        task = CalexpCutoutTask(config=config)
        result = task.run(self.data['bad'], self.exp)
        # Cutouts outside the image should be skipped
        self.assertEqual(len(result.cutouts), 0)

        config.skip_bad = False
        task = CalexpCutoutTask(config=config)
        # Should now raise for cutouts outside the image
        with self.assertRaises(ValueError):
            result = task.run(self.data['bad'], self.exp)


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
