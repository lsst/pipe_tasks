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
import unittest


import lsst.utils.tests
from lsst.utils import getPackageDir
import lsst.obs.base
from lsst.pipe.tasks.read_defects import read_all_defects
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
        defects = read_all_defects(defects_path, cam)
        self.assertEqual(len(defects.keys()), 1)  # One sensor
        for s in defects:
            self.assertEqual(len(defects[s].keys()), 2)  # Two validity ranges
            for d in defects[s]:
                self.assertEqual(len(defects[s][d]), 4)  # Four defects


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == '__main__':
    lsst.utils.tests.init()
    unittest.main()
