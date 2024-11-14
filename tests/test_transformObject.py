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
import numpy as np

import lsst.utils.tests

from lsst.pipe.base import InMemoryDatasetHandle
from lsst.pipe.tasks.functors import HsmFwhm, Column
from lsst.pipe.tasks.postprocess import TransformObjectCatalogTask, TransformObjectCatalogConfig

ROOT = os.path.abspath(os.path.dirname(__file__))


def setup_module(module):
    lsst.utils.tests.init()


class TransformObjectCatalogTestCase(unittest.TestCase):
    def setUp(self):
        # Note that this test input includes HSC-G, HSC-R, and HSC-I data
        df = pd.read_csv(os.path.join(ROOT, 'data', 'test_multilevel_parq.csv.gz'),
                         header=[0, 1, 2], index_col=0)

        self.dataId = {"tract": 9615, "patch": "4,4"}
        self.handle = InMemoryDatasetHandle(df, storageClass="DataFrame", dataId=self.dataId)

    def testNullFilter(self):
        """Test that columns for all filters are created despite they may not
        exist in the input data.
        """
        config = TransformObjectCatalogConfig()
        config.camelCase = True
        # Want y band columns despite the input data do not have them
        # Exclude g band columns despite the input data have them
        config.outputBands = ["r", "i", "y"]
        # Arbitrarily choose a boolean flag column to be "good"
        config.goodFlags = ['GoodFlagColumn']
        task = TransformObjectCatalogTask(config=config)
        # Add in a float column, an integer column, a good flag, and
        # a bad flag.  It does not matter which columns we choose, just
        # that they have the appropriate type.
        funcs = {'FloatColumn': HsmFwhm(dataset='meas'),
                 'IntColumn': Column('base_InputCount_value', dataset='meas'),
                 'GoodFlagColumn': Column('slot_GaussianFlux_flag', dataset='meas'),
                 'BadFlagColumn': Column('slot_Centroid_flag', dataset='meas')}
        df = task.run(self.handle, funcs=funcs, dataId=self.dataId).outputCatalog
        self.assertIsInstance(df, pd.DataFrame)

        for filt in config.outputBands:
            self.assertIn(filt + 'FloatColumn', df.columns)
            self.assertIn(filt + 'IntColumn', df.columns)
            self.assertIn(filt + 'BadFlagColumn', df.columns)
            self.assertIn(filt + 'GoodFlagColumn', df.columns)

        # Check that the default filling has worked.
        self.assertNotIn('gFloatColumn', df.columns)
        self.assertTrue(df['yFloatColumn'].isnull().all())
        self.assertTrue(df['iFloatColumn'].notnull().all())
        self.assertTrue(np.all(df['iIntColumn'].values >= 0))
        self.assertTrue(np.all(df['yIntColumn'].values < 0))
        self.assertTrue(np.all(~df['yGoodFlagColumn'].values))
        self.assertTrue(np.all(df['yBadFlagColumn'].values))

        # Check that the datatypes are preserved.
        self.assertEqual(df['iFloatColumn'].dtype, np.dtype(np.float64))
        self.assertEqual(df['yFloatColumn'].dtype, np.dtype(np.float64))
        self.assertEqual(df['iIntColumn'].dtype, np.dtype(np.int64))
        self.assertEqual(df['yIntColumn'].dtype, np.dtype(np.int64))
        self.assertEqual(df['iGoodFlagColumn'].dtype, np.dtype(np.bool_))
        self.assertEqual(df['yGoodFlagColumn'].dtype, np.dtype(np.bool_))
        self.assertEqual(df['iBadFlagColumn'].dtype, np.dtype(np.bool_))
        self.assertEqual(df['yBadFlagColumn'].dtype, np.dtype(np.bool_))

    def testUnderscoreColumnFormat(self):
        """Test the per-filter column format with an underscore"""
        config = TransformObjectCatalogConfig()
        config.outputBands = ["g", "r", "i"]
        config.camelCase = False
        task = TransformObjectCatalogTask(config=config)
        funcs = {'Fwhm': HsmFwhm(dataset='meas')}
        df = task.run(self.handle, funcs=funcs, dataId=self.dataId).outputCatalog
        self.assertIsInstance(df, pd.DataFrame)
        for filt in config.outputBands:
            self.assertIn(filt + '_Fwhm', df.columns)

    def testMultilevelOutput(self):
        """Test the non-flattened result dataframe with a multilevel column index"""
        config = TransformObjectCatalogConfig()
        config.outputBands = ["r", "i"]
        config.multilevelOutput = True
        task = TransformObjectCatalogTask(config=config)
        funcs = {'Fwhm': HsmFwhm(dataset='meas')}
        df = task.run(self.handle, funcs=funcs, dataId=self.dataId).outputCatalog
        self.assertIsInstance(df, pd.DataFrame)
        self.assertNotIn('g', df)
        for filt in config.outputBands:
            self.assertIsInstance(df[filt], pd.DataFrame)
            self.assertIn('Fwhm', df[filt].columns)

    def testNoOutputBands(self):
        """All the input bands should go into the output, and nothing else.
        """
        config = TransformObjectCatalogConfig()
        config.multilevelOutput = True
        task = TransformObjectCatalogTask(config=config)
        funcs = {'Fwhm': HsmFwhm(dataset='meas')}
        df = task.run(self.handle, funcs=funcs, dataId=self.dataId).outputCatalog
        self.assertIsInstance(df, pd.DataFrame)
        self.assertNotIn('HSC-G', df)
        for filt in ['g', 'r', 'i']:
            self.assertIsInstance(df[filt], pd.DataFrame)
            self.assertIn('Fwhm', df[filt].columns)


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
