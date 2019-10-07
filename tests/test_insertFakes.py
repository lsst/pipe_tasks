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
import os
import unittest
import tempfile
import shutil

import lsst.utils.tests
import lsst.utils as lsst_utils

from lsst.pipe.tasks.processCcd import ProcessCcdTask


class TestFakeInserstion(unittest.TestCase):
    """
    Test case for the methods in insertFakes.py
    """

    @classmethod
    def setUpClass(cls):
        current_dir = os.path.join(lsst_utils.getPackageDir('pipe_tasks'),
                                   'tests')
        cls._workspace = tempfile.mkdtemp(dir=current_dir,
                                          prefix='insertFakes_workspace_')

        print('\nworkspace:\n%s\n\n' % cls._workspace)

        obs_test_dir = lsst_utils.getPackageDir('obs_test')
        input_dir = os.path.join(obs_test_dir, 'data', 'input')
        cls._data_id = {'visit': 1}
        data_id_string = ['%s=%s' % (key, val)
                          for key, val in cls._data_id.items()]
        ProcessCcdTask.parseAndRun(args=[input_dir, '--output', cls._workspace,
                                         '--clobber-config', '--clobber-output',
                                         '--doraise',
                                         '--id'] + data_id_string)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._workspace)

    def test_dummy(self):
        pass


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
