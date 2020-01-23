# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import unittest
import pandas as pd

import lsst.utils.tests

# TODO: Remove skipUnless and this try block DM-22256
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lsst.pipe.tasks.parquetTable import MultilevelParquetTable
    from lsst.pipe.tasks.functors import HsmFwhm
    from lsst.pipe.tasks.postprocess import TransformObjectCatalogTask, TransformObjectCatalogConfig
    havePyArrow = True
except ImportError:
    havePyArrow = False

ROOT = os.path.abspath(os.path.dirname(__file__))


def setup_module(module):
    lsst.utils.tests.init()


@unittest.skipUnless(havePyArrow, "Requires pyarrow")
class TransformObjectCatalogTestCase(unittest.TestCase):
    def setUp(self):
        # Note that this test input includes HSC-G, HSC-R, and HSC-I data
        df = pd.read_csv(os.path.join(ROOT, 'data', 'test_multilevel_parq.csv.gz'),
                         header=[0, 1, 2], index_col=0)
        with lsst.utils.tests.getTempFilePath('*.parq') as filename:
            table = pa.Table.from_pandas(df)
            pq.write_table(table, filename, compression='none')
            self.parq = MultilevelParquetTable(filename)

        self.dataId = {"tract": 9615, "patch": "4,4"}

    def testNullFilter(self):
        """Test that columns for all filters are created despite they may not
        exist in the input data.
        """
        config = TransformObjectCatalogConfig()
        # Want y band columns despite the input data do not have them
        # Exclude g band columns despite the input data have them
        filterMap = {"HSC-R": "r", "HSC-I": "i", "HSC-Y": "y"}
        config.filterMap = filterMap
        task = TransformObjectCatalogTask(config=config)
        funcs = {'Fwhm': HsmFwhm(dataset='meas')}
        df = task.run(self.parq, funcs=funcs, dataId=self.dataId)
        self.assertIsInstance(df, pd.DataFrame)
        for column in ('coord_ra', 'coord_dec'):
            self.assertIn(column, df.columns)

        for filt in filterMap.values():
            self.assertIn(filt + 'Fwhm', df.columns)

        self.assertNotIn('gFwhm', df.columns)
        self.assertTrue(df['yFwhm'].isnull().all())
        self.assertTrue(df['iFwhm'].notnull().all())

    def testUnderscoreColumnFormat(self):
        """Test the per-filter column format with an underscore"""
        config = TransformObjectCatalogConfig()
        filterMap = {"HSC-G": "g", "HSC-R": "r", "HSC-I": "i"}
        config.filterMap = filterMap
        config.camelCase = False
        task = TransformObjectCatalogTask(config=config)
        funcs = {'Fwhm': HsmFwhm(dataset='meas')}
        df = task.run(self.parq, funcs=funcs, dataId=self.dataId)
        self.assertIsInstance(df, pd.DataFrame)
        for filt in filterMap.values():
            self.assertIn(filt + '_Fwhm', df.columns)

    def testMultilevelOutput(self):
        """Test the non-flattened result dataframe with a multilevel column index"""
        config = TransformObjectCatalogConfig()
        filterMap = {"HSC-R": "r", "HSC-I": "i"}
        config.filterMap = filterMap
        config.multilevelOutput = True
        task = TransformObjectCatalogTask(config=config)
        funcs = {'Fwhm': HsmFwhm(dataset='meas')}
        df = task.run(self.parq, funcs=funcs, dataId=self.dataId)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertNotIn('HSC-G', df)
        for filt in filterMap:
            self.assertIsInstance(df[filt], pd.DataFrame)
            self.assertIn('Fwhm', df[filt].columns)
