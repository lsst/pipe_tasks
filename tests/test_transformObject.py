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
import astropy.table
import pandas as pd
import numpy as np

import lsst.utils.tests

from lsst.pipe.base import InMemoryDatasetHandle
from lsst.pipe.tasks.functors import CoordColumn, Column
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
        n_rows = len(df)
        tab_epoch = astropy.table.Table({df.index.name: df.index, "r_epoch": [0.]*n_rows})
        tab_ref = astropy.table.Table({df.index.name: df.index, "refBand": ["r"]*n_rows})
        tab_exp = astropy.table.Table({df.index.name: df.index, "exp_n_iter": [0] * n_rows})
        tab_sersic = astropy.table.Table({df.index.name: df.index, "sersic_n_iter": [0] * n_rows})
        self.kwargs_task = {
            "handle_epoch": InMemoryDatasetHandle(
                tab_epoch, storageClass="ArrowAstropy", dataId=self.dataId,
            ),
            "handle_ref": InMemoryDatasetHandle(
                tab_ref, storageClass="ArrowAstropy", dataId=self.dataId,
            ),
            "handle_Exp_multiprofit": InMemoryDatasetHandle(
                tab_exp, storageClass="ArrowAstropy", dataId=self.dataId,
            ),
            "handle_Sersic_multiprofit": InMemoryDatasetHandle(
                tab_sersic, storageClass="ArrowAstropy", dataId=self.dataId,
            ),
        }
        self.funcs_multi = {
            "epoch": Column("r_epoch", dataset="epoch"),
            "refBand": Column("refBand", dataset="ref"),
            "exp_n_iter": Column("exp_n_iter", dataset="Exp_multiprofit"),
            "sersic_n_iter": Column("sersic_n_iter", dataset="Sersic_multiprofit"),
        }

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
        funcs = {'FloatColumn': CoordColumn('coord_ra', dataset='meas'),
                 'IntColumn': Column('base_InputCount_value', dataset='meas'),
                 'GoodFlagColumn': Column('slot_GaussianFlux_flag', dataset='meas'),
                 'BadFlagColumn': Column('slot_Centroid_flag', dataset='meas')}
        funcs.update(self.funcs_multi)
        tbl = task.run(self.handle, funcs=funcs, dataId=self.dataId, **self.kwargs_task).outputCatalog
        self.assertIsInstance(tbl, astropy.table.Table)

        for filt in config.outputBands:
            self.assertIn(filt + 'FloatColumn', tbl.columns)
            self.assertIn(filt + 'IntColumn', tbl.columns)
            self.assertIn(filt + 'BadFlagColumn', tbl.columns)
            self.assertIn(filt + 'GoodFlagColumn', tbl.columns)

        # Check that the default filling has worked.
        self.assertNotIn('gFloatColumn', tbl.columns)
        self.assertTrue(np.all(np.ma.is_masked(tbl['yFloatColumn'])))
        self.assertFalse(np.any(np.ma.is_masked(tbl['iFloatColumn'])))
        self.assertTrue(np.all(tbl['iIntColumn'] >= 0))
        self.assertTrue(np.all(tbl['yIntColumn'] < 0))
        self.assertTrue(np.all(~tbl['yGoodFlagColumn']))
        self.assertTrue(np.all(tbl['yBadFlagColumn']))

        # Check that the datatypes are preserved.
        self.assertEqual(tbl['iFloatColumn'].dtype, np.dtype(np.float64))
        self.assertEqual(tbl['yFloatColumn'].dtype, np.dtype(np.float64))
        self.assertEqual(tbl['iIntColumn'].dtype, np.dtype(np.int64))
        self.assertEqual(tbl['yIntColumn'].dtype, np.dtype(np.int64))
        self.assertEqual(tbl['iGoodFlagColumn'].dtype, np.dtype(np.bool_))
        self.assertEqual(tbl['yGoodFlagColumn'].dtype, np.dtype(np.bool_))
        self.assertEqual(tbl['iBadFlagColumn'].dtype, np.dtype(np.bool_))
        self.assertEqual(tbl['yBadFlagColumn'].dtype, np.dtype(np.bool_))

    def testUnderscoreColumnFormat(self):
        """Test the per-filter column format with an underscore"""
        config = TransformObjectCatalogConfig()
        config.outputBands = ["g", "r", "i"]
        config.camelCase = False
        task = TransformObjectCatalogTask(config=config)
        funcs = {'ra': CoordColumn('coord_ra', dataset='meas')}
        funcs.update(self.funcs_multi)
        tbl = task.run(self.handle, funcs=funcs, dataId=self.dataId, **self.kwargs_task).outputCatalog
        self.assertIsInstance(tbl, astropy.table.Table)
        for filt in config.outputBands:
            self.assertIn(filt + '_ra', tbl.columns)

    def testMultilevelOutput(self):
        """Test the non-flattened result dataframe with a multilevel column index"""
        config = TransformObjectCatalogConfig()
        config.outputBands = ["r", "i"]
        config.multilevelOutput = True
        task = TransformObjectCatalogTask(config=config)
        funcs = {'ra': CoordColumn('coord_ra', dataset='meas')}
        funcs.update(self.funcs_multi)
        df = task.run(self.handle, funcs=funcs, dataId=self.dataId, **self.kwargs_task).outputCatalog
        self.assertIsInstance(df, pd.DataFrame)
        self.assertNotIn('g', df)
        for filt in config.outputBands:
            self.assertIsInstance(df[filt], pd.DataFrame)
            self.assertIn('ra', df[filt].columns)

    def testNoOutputBands(self):
        """All the input bands should go into the output, and nothing else.
        """
        config = TransformObjectCatalogConfig()
        task = TransformObjectCatalogTask(config=config)
        funcs = {'ra': CoordColumn('coord_ra', dataset='meas')}
        funcs.update(self.funcs_multi)
        tbl = task.run(self.handle, funcs=funcs, dataId=self.dataId, **self.kwargs_task).outputCatalog
        self.assertIsInstance(tbl, astropy.table.Table)
        self.assertNotIn('HSC-G_Fwhm', tbl.columns)
        for filt in ['g', 'r', 'i']:
            self.assertIn(f'{filt}_ra', tbl.columns)


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
