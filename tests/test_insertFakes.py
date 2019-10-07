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
import astropy.table as astropy_table

import lsst.utils.tests
import lsst.utils as lsst_utils

import lsst.geom as lsst_geom
import lsst.daf.persistence as daf_persistence
from lsst.pipe.tasks.processCcd import ProcessCcdTask
from lsst.pipe.tasks.insertFakes import InsertFakesTask


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

    def setUp(self):
        current_dir = os.path.join(lsst_utils.getPackageDir('pipe_tasks'),
                                   'tests')
        data_dir = os.path.join(current_dir, 'data', 'insertFakes')
        self.star_truth = os.path.join(data_dir, 'starTruth.txt')
        self.star_catalog = os.path.join(data_dir, 'starCatalog.fits')

    def test_insertFakeStars(self):
        butler = daf_persistence.Butler(self._workspace)
        calexp = butler.get('calexp', dataId=self._data_id)
        psf = calexp.getPsf()
        wcs = calexp.getWcs()
        photo_calib = butler.get('calexp_photoCalib', dataId=self._data_id)
        insert_fakes_task = InsertFakesTask()
        fake_star_table = astropy_table.Table.read(self.star_catalog)
        fake_star_table = fake_star_table.to_pandas()
        fake_star_table = insert_fakes_task.addPixCoords(fake_star_table,
                                                         wcs)

        fake_star_image_list = insert_fakes_task.mkFakeStars(fake_star_table,
                                                             'g',
                                                             photo_calib,
                                                             psf,
                                                             calexp)
        # there should be 4 fake stars
        self.assertEqual(len(fake_star_image_list), 4)

        fake_star_ra = []
        fake_star_dec = []
        fake_star_mag = []
        with open(self.star_truth, 'r') as in_file:
            for line in in_file:
                if line.startswith('#'):
                    continue
                params = line.strip().split()
                fake_star_ra.append(float(params[0]))
                fake_star_dec.append(float(params[1]))
                fake_star_mag.append(float(params[2]))

        # test that stars are added at the correct place
        for ii, (rr, dd) in enumerate(zip(fake_star_table['raJ2000'],
                                      fake_star_table['decJ2000'])):
            sky_pt = lsst_geom.SpherePoint(rr*lsst_geom.radians,
                                           dd*lsst_geom.radians)
            pixel_pt = wcs.skyToPixel(sky_pt)
            self.assertEqual(pixel_pt, fake_star_image_list[ii][1])


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
