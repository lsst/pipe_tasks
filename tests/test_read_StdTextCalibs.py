#!/usr/bin/env python

#
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2019 LSST Corporation.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

import os
import glob
import unittest


import lsst.utils.tests
from lsst.utils import getPackageDir
import lsst.obs.base
from lsst.pipe.tasks.read_stdText_calibs import read_all
import lsst.daf.persistence as dafPersist

ROOT = os.path.join(getPackageDir('obs_base'), 'tests')


def setup_module(module):
    lsst.utils.tests.init()


class BaseMapper(lsst.obs.base.CameraMapper):
    packageName = 'base'

    def __init__(self):
        policy = dafPersist.Policy(os.path.join(ROOT, "BaseMapper.yaml"))
        lsst.obs.base.CameraMapper.__init__(self, policy=policy, repositoryDir=ROOT, root=ROOT)
        return

    @classmethod
    def getPackageDir(cls):
        return "/path/to/nowhere"


class ReadDefectsTestCase(unittest.TestCase):
    """A test case for the mapper used by the data butler."""

    def setUp(self):
        self.mapper = BaseMapper()

    def tearDown(self):
        del self.mapper

    def test_read_defects(self):
        butler = dafPersist.ButlerFactory(mapper=self.mapper).create()
        cam = butler.get('camera')
        defects_path = os.path.join(ROOT, 'trivial_camera', 'defects')
        defects, data_type = read_all(defects_path, cam)
        self.assertEqual(len(defects.keys()), 1)  # One sensor
        self.assertEqual(data_type, 'defects')
        for s in defects:
            self.assertEqual(len(defects[s].keys()), 2)  # Two validity ranges
            for d in defects[s]:
                self.assertEqual(len(defects[s][d]), 4)  # Four defects


class ReadQeTestCase(unittest.TestCase):
    """A test case for the mapper used by the data butler."""

    def setUp(self):
        self.mapper = BaseMapper()

    def tearDown(self):
        del self.mapper

    def cleanupLinks(self, files):
        for f in files:
            try:
                os.unlink(f)
            except OSError:
                pass  # Carry on since the file didn't exist

    def test_read_qe_amp(self):
        butler = dafPersist.ButlerFactory(mapper=self.mapper).create()
        cam = butler.get('camera')
        qe_path = os.path.join(ROOT, 'trivial_camera', 'qe_curves')
        files = glob.glob(os.path.join(qe_path, 'ccd00', 'per_amp', '*'))
        dest_files = [os.path.join(qe_path, 'ccd00', os.path.split(f)[1]) for f in files]
        self.cleanupLinks(dest_files)
        for f, df in zip(files, dest_files):
            os.symlink(f, df)
        curves, data_type = read_all(qe_path, cam)
        self.assertEqual(len(curves.keys()), 1)  # One sensor
        self.assertEqual(data_type, 'qe_curves')
        for s in curves:
            self.assertEqual(len(curves[s].keys()), 2)  # Two validity ranges
            for d in curves[s]:
                self.assertEqual(len(curves[s][d].data), 2)  # Two amps

    def test_read_qe_det(self):
        butler = dafPersist.ButlerFactory(mapper=self.mapper).create()
        cam = butler.get('camera')
        qe_path = os.path.join(ROOT, 'trivial_camera', 'qe_curves')
        files = glob.glob(os.path.join(qe_path, 'ccd00', 'per_detector', '*'))
        dest_files = [os.path.join(qe_path, 'ccd00', os.path.split(f)[1]) for f in files]
        self.cleanupLinks(dest_files)
        for f, df in zip(files, dest_files):
            os.symlink(f, df)
        curves, data_type = read_all(qe_path, cam)
        self.assertEqual(len(curves.keys()), 1)  # One sensor
        self.assertEqual(data_type, 'qe_curves')
        for s in curves:
            self.assertEqual(len(curves[s].keys()), 2)  # Two validity ranges


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == '__main__':
    lsst.utils.tests.init()
    unittest.main()
