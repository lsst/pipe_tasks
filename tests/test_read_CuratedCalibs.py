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
import glob
import unittest
import tempfile


import lsst.utils.tests
from lsst.utils import getPackageDir
import lsst.obs.base
from lsst.pipe.tasks.read_curated_calibs import read_all
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
    """A test case for the defect reader."""

    def setUp(self):
        butler = dafPersist.ButlerFactory(mapper=BaseMapper()).create()
        self.cam = butler.get('camera')
        self.defects_path = os.path.join(ROOT, 'trivial_camera', 'defects')

    def tearDown(self):
        del self.cam
        del self.defects_path

    def test_read_defects(self):
        defects, data_type = read_all(self.defects_path, self.cam)
        self.assertEqual(len(defects.keys()), 1)  # One sensor
        self.assertEqual(data_type, 'defects')
        for s in defects:
            self.assertEqual(len(defects[s].keys()), 2)  # Two validity ranges
            for d in defects[s]:
                self.assertEqual(len(defects[s][d]), 4)  # Four defects


class ReadQeTestCase(unittest.TestCase):
    """A test case for the qe_curve reader"""

    def setUp(self):
        butler = dafPersist.ButlerFactory(mapper=BaseMapper()).create()
        self.cam = butler.get('camera')
        self.qe_path = os.path.join(ROOT, 'trivial_camera', 'qe_curves')
        self.tmp_dir_obj = tempfile.TemporaryDirectory()

    def tearDown(self):
        del self.cam
        del self.qe_path
        self.tmp_dir_obj.cleanup()

    def read_qe_tester(self, per_amp):
        if per_amp:
            path_str = 'per_amp'
        else:
            path_str = 'per_detector'
        files = glob.glob(os.path.join(self.qe_path, 'ccd00', path_str, '*'))
        dest_path = os.path.join(self.tmp_dir_obj.name, 'trivial_camera',
                                 'qe_curves', 'ccd00')
        os.makedirs(dest_path)
        dest_files = [os.path.join(dest_path, os.path.split(f)[1]) for f in files]
        for f, df in zip(files, dest_files):
            os.symlink(f, df)
        curves, data_type = read_all(os.path.join(self.tmp_dir_obj.name, 'trivial_camera', 'qe_curves'),
                                     self.cam)
        self.assertEqual(len(curves.keys()), 1)  # One sensor
        self.assertEqual(data_type, 'qe_curves')
        for s in curves:
            self.assertEqual(len(curves[s].keys()), 2)  # Two validity ranges
            if per_amp:
                for d in curves[s]:
                    self.assertEqual(len(curves[s][d].data), 2)  # Two amps

    def test_read_qe_amp(self):
        self.read_qe_tester(True)

    def test_read_qe_det(self):
        self.read_qe_tester(False)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == '__main__':
    lsst.utils.tests.init()
    unittest.main()
