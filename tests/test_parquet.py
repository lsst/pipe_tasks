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

import warnings
import unittest
import copy
import functools
import pandas as pd
from pandas.testing import assert_frame_equal

import lsst.utils.tests

import pyarrow as pa
import pyarrow.parquet as pq

from lsst.pipe.tasks.parquetTable import ParquetTable, MultilevelParquetTable


def setup_module(module):
    lsst.utils.tests.init()


class ParquetTableTestCase(unittest.TestCase):
    """Test case for ParquetTable
    """

    def simulateDF(self):
        """Create a simple test DataFrame
        """
        df = pd.DataFrame({
            "coord_ra": [3.77654137, 3.77643059, 3.77621148, 3.77611944, 3.77610396],
            "coord_dec": [0.01127624, 0.01127787, 0.01127543, 0.01127543, 0.01127543],
            "slot_Centroid_flag": [True, True, True, True, True],
            "slot_Centroid_x": [16208., 16344., 16613., 16726., 16745.],
            "slot_Centroid_y": [15905., 15907., 15904., 15904., 15904.],
            "slot_PsfFlux_apCorr": [0.98636465, 0.98437287, 0.97212515, 0.97179828, 0.97182371],
            "slot_PsfFlux_apCorrSigma": [0., 0., 0., 0., 0.],
            "slot_PsfFlux_flag": [True, True, True, True, True],
            "slot_PsfFlux_instFlux": [0.28106412, 1.98260751, 0.08900771, 1.11375753, 1.3835924],
            "slot_PsfFlux_instFluxSigma": [0.22967081, 0.25409701, 0.2120654, 0.23031162, 0.24262261],
            "calib_psfUsed": [False, False, False, False, False],
            "detect_isPatchInner": [False, False, False, False, False],
            "detect_isPrimary": [False, False, False, False, False],
            "detect_isTractInner": [True, True, True, True, True]})
        return df

    def setUp(self):
        self.df = self.simulateDF()
        with lsst.utils.tests.getTempFilePath('*.parq') as filename:
            table = pa.Table.from_pandas(self.df)
            pq.write_table(table, filename)
            self.parq, self.dfParq = self.getParq(filename, self.df)

    def tearDown(self):
        del self.df
        del self.parq

    def getParq(self, filename, df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fromFile, fromDF = ParquetTable(filename), ParquetTable(dataFrame=df)

        return fromFile, fromDF

    def testRoundTrip(self):
        self.assertTrue(self.parq.toDataFrame().equals(self.df))

    def testColumns(self):
        columns = ['coord_ra', 'coord_dec']
        self.assertTrue(self.parq.toDataFrame(columns=columns).equals(self.df[columns]))

        # TO DO: DM-21976 Confirm this is the behavior we want
        # Quietly ignore nonsense columns
        self.assertTrue(self.parq.toDataFrame(columns=columns + ['hello']).equals(self.df[columns]))


class MultilevelParquetTableTestCase(ParquetTableTestCase):
    """Test case for MultilevelParquetTable
    """

    def simulateDF(self):
        self.datasets = ['meas', 'ref']
        self.filters = ['G', 'R']
        self.columns = ['coord_ra', 'coord_dec']
        simpleDF = super(MultilevelParquetTableTestCase, self).simulateDF()
        dfFilterDSCombos = []
        for ds in self.datasets:
            for filterName in self.filters:
                df = copy.copy(simpleDF)
                df.reindex(sorted(df.columns), axis=1)
                df['dataset'] = 'meas'
                df['filter'] = filterName
                df.columns = pd.MultiIndex.from_tuples([(ds, filterName, c) for c in df.columns],
                                                       names=('dataset', 'filter', 'column'))
                dfFilterDSCombos.append(df)

        return functools.reduce(lambda d1, d2: d1.join(d2), dfFilterDSCombos)

    def getParq(self, filename, df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fromFile = MultilevelParquetTable(filename)
            fromDf = MultilevelParquetTable(dataFrame=df)
        return fromFile, fromDf

    def testProperties(self):
        self.assertTrue(all([x == y for x, y in zip(self.parq.columnLevels, self.df.columns.names)]))
        self.assertEqual(len(self.parq.columns), len(self.df.columns))

        self.assertTrue(all([x == y for x, y in zip(self.dfParq.columnLevels, self.df.columns.names)]))
        self.assertEqual(len(self.dfParq.columns), len(self.df.columns))

    def testColumns(self):
        df = self.df
        parq = self.parq

        # Case A, each level has multiple values
        datasets_A = self.datasets
        filters_A = self.filters
        columns_A = self.columns
        columnDict_A = {'dataset': datasets_A,
                        'filter': filters_A,
                        'column': columns_A
                        }
        colTuples_A = [(self.datasets[0], self.filters[0], self.columns[0]),
                       (self.datasets[0], self.filters[0], self.columns[1]),
                       (self.datasets[0], self.filters[1], self.columns[0]),
                       (self.datasets[0], self.filters[1], self.columns[1]),
                       (self.datasets[1], self.filters[0], self.columns[0]),
                       (self.datasets[1], self.filters[0], self.columns[1]),
                       (self.datasets[1], self.filters[1], self.columns[0]),
                       (self.datasets[1], self.filters[1], self.columns[1])]
        df_A = df[colTuples_A]
        assert_frame_equal(parq.toDataFrame(columns=columnDict_A), df_A)

        # Case A1, add a bogus column and test that it gets ignored
        datasets_A1 = self.datasets
        filters_A1 = self.filters
        columns_A1 = self.columns + ['garbage']
        columnDict_A1 = {'dataset': datasets_A1,
                         'filter': filters_A1,
                         'column': columns_A1}
        colTuples_A1 = [(self.datasets[0], self.filters[0], self.columns[0]),
                        (self.datasets[0], self.filters[0], self.columns[1]),
                        (self.datasets[0], self.filters[1], self.columns[0]),
                        (self.datasets[0], self.filters[1], self.columns[1]),
                        (self.datasets[1], self.filters[0], self.columns[0]),
                        (self.datasets[1], self.filters[0], self.columns[1]),
                        (self.datasets[1], self.filters[1], self.columns[0]),
                        (self.datasets[1], self.filters[1], self.columns[1])]
        df_A1 = df[colTuples_A1]
        assert_frame_equal(parq.toDataFrame(columns=columnDict_A1), df_A1)

        # Case B: One level has only a single value
        datasets_B = self.datasets[0]
        filters_B = self.filters
        columns_B = self.columns
        columnDict_B = {'dataset': datasets_B,
                        'filter': filters_B,
                        'column': columns_B}
        colTuples_B = [(self.datasets[0], self.filters[0], self.columns[0]),
                       (self.datasets[0], self.filters[0], self.columns[1]),
                       (self.datasets[0], self.filters[1], self.columns[0]),
                       (self.datasets[0], self.filters[1], self.columns[1])]
        df_B = df[colTuples_B]
        df_B.columns = df_B.columns.droplevel('dataset')
        assert_frame_equal(parq.toDataFrame(columns=columnDict_B), df_B)
        assert_frame_equal(df_B, parq.toDataFrame(columns=colTuples_B))

        # When explicit columns are not provided, comparison requires
        # first getting the column index in sorted order.  Apparently this
        # happens by default in parq.toDataFrame(); to be honest, I'm not
        # exactly sure how/why.

        # Case C: Two levels have a single value; third is not provided
        datasets_C = self.datasets[0]
        filters_C = self.filters[0]
        columnDict_C = {'dataset': datasets_C,
                        'filter': filters_C}
        df_C = df[datasets_C][filters_C].sort_index(axis=1)

        self.assertTrue(parq.toDataFrame(columns=columnDict_C).equals(df_C))

        # Case D: Only one level (first level) is provided
        dataset_D = self.datasets[0]
        columnDict_D = {'dataset': dataset_D}
        df_D = df[dataset_D].sort_index(axis=1)
        self.assertTrue(parq.toDataFrame(columns=columnDict_D).equals(df_D))

        # Case E: Only one level (second level) is provided
        filters_E = self.filters[1]
        columnDict_E = {'filter': filters_E}
        # get second level of multi-index column using .xs()
        df_E = df.xs(filters_E, level=1, axis=1).sort_index(axis=1)
        self.assertTrue(parq.toDataFrame(columns=columnDict_E).equals(df_E))

        # Case when all requested columns don't exist
        columnDictNonsense = {'dataset': 'meas', 'filter': 'G', 'column': ('hello')}
        self.assertRaises(ValueError, parq.toDataFrame, columns=columnDictNonsense)

        # Case when some requested columns don't exist.
        # TO DO: DM-21976 Confirm this is the behavior we want
        # Quietly ignore nonsense columns
        columnDictSomeNonsense = {'dataset': 'meas', 'filter': 'G', 'column': ('coord_ra', 'hello')}
        dfGood = pd.DataFrame(df['meas']['G']['coord_ra'])
        self.assertTrue(parq.toDataFrame(columns=columnDictSomeNonsense).equals(dfGood))


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
