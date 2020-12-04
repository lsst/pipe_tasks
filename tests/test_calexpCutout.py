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
import os
import random
import unittest
from astropy.coordinates import SkyCoord
from astropy.table import QTable
import astropy.units as u

import lsst.utils
import lsst.utils.tests
from lsst.afw.image import ExposureF
from lsst.pipe.tasks.calexpCutout import CalexpCutoutTask

random.seed(208241138)

packageDir = lsst.utils.getPackageDir('pipe_tasks')
datadir = os.path.join(packageDir, 'tests', "data")


def make_data(bbox, wcs, border=100, ngood=13, nbad=7, nedge=3):
    dx, dy = bbox.getDimensions()
    data = {}

    ident = []
    pt = []
    size = []
    for i in range(ngood):
        x = random.random()*(dx - 2*border) + border
        y = random.random()*(dy - 2*border) + border
        sphpt = wcs.pixelToSky(x, y)
        pt.append(SkyCoord(sphpt.getRa().asDegrees(), sphpt.getDec().asDegrees(),
                           frame='icrs', unit=u.deg))
        ident.append((i+1)*u.dimensionless_unscaled)
        size.append(random.randint(13, 26)*u.dimensionless_unscaled)
    data['good'] = QTable([ident, pt, size], names=['id', 'position', 'size'])

    ident = []
    pt = []
    size = []
    for i in range(nbad):
        x = random.random()*(dx - 2*border) + border
        y = random.random()*(dy - 2*border) + border
        sphpt = wcs.pixelToSky(-x, -y)
        pt.append(SkyCoord(sphpt.getRa().asDegrees(), sphpt.getDec().asDegrees(),
                           frame='icrs', unit=u.deg))
        ident.append((i+1)*u.dimensionless_unscaled)
        size.append(random.randint(13, 26)*u.dimensionless_unscaled)
    data['bad'] = QTable([ident, pt, size], names=['id', 'position', 'size'])

    ident = []
    pt = []
    size = []
    for i in range(nedge):
        x_or_y = random.randint(0, 1)
        if x_or_y:
            x = random.random()*dx
            y = [0, dy][random.randint(0, 1)]
        else:
            x = [0, dx][random.randint(0, 1)]
            y = random.random()*dy

        sphpt = wcs.pixelToSky(x, y)
        pt.append(SkyCoord(sphpt.getRa().asDegrees(), sphpt.getDec().asDegrees(),
                           frame='icrs', unit=u.deg))
        ident.append((i+1)*u.dimensionless_unscaled)
        size.append(random.randint(13, 26)*u.dimensionless_unscaled)
    data['edge'] = QTable([ident, pt, size], names=['id', 'position', 'size'])
    return data


class CalexpCutoutTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.exp = ExposureF.readFits(os.path.join(datadir, 'v695833-e0-c000-a00.sci.fits'))
        self.data = make_data(self.exp.getBBox(), self.exp.getWcs())

    def tearDown(self):
        del self.data
        del self.exp

    def testCalexpCutout(self):
        config = CalexpCutoutTask.ConfigClass()
        task = CalexpCutoutTask(config=config)
        result = task.run(self.data['good'], self.exp)
        self.assertEqual(len(result.cutouts), len(self.data['good']))

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
        self.assertEqual(len(result.skipped_positions), len(self.data['edge']))

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
        self.assertEqual(len(result.skipped_positions), len(self.data['bad']))

        config.skip_bad = False
        task = CalexpCutoutTask(config=config)
        # Should now raise for cutouts outside the image
        with self.assertRaises(ValueError):
            result = task.run(self.data['bad'], self.exp)

    def testBadColumns(self):
        config = CalexpCutoutTask.ConfigClass()
        task = CalexpCutoutTask(config=config)
        table = QTable([[], []], names=['one', 'two'])
        with self.assertRaises(ValueError):
            result = task.run(table, self.exp)  # noqa


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
