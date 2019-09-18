# This file is part of qa_explorer.
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


from __future__ import print_function

import os
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from pandas.util.testing import assert_frame_equal

import lsst.utils.tests

from lsst.qa.explorer.parquetTable import ParquetTable, MultilevelParquetTable

ROOT = os.path.abspath(os.path.dirname(__file__))

def setup_module(module):
    lsst.utils.tests.init()

class ParquetTableTestCase(unittest.TestCase):
    """Test case for ParquetTable
    """
    testFilename = 'simple_test.parq'

    def setUp(self):
        self.df = pq.read_table(os.path.join(ROOT, self.testFilename)).to_pandas()
        with lsst.utils.tests.getTempFilePath('*.parq') as filename:
            table = pa.Table.from_pandas(self.df)
            pq.write_table(table, filename, compression='none')
            self.parq, self.dfParq = self.getParq(filename, self.df)

    def tearDown(self):
        del self.df
        del self.parq

    def getParq(self, filename, df):
        return ParquetTable(filename), ParquetTable(dataFrame=df)

    def testRoundTrip(self):
        assert_frame_equal(self.parq.toDataFrame(), self.df)

    def testColumns(self):
        columns = ['coord_ra', 'coord_dec']
        assert_frame_equal(self.parq.toDataFrame(columns=columns),
                           self.df[columns])

        # Quietly ignore nonsense columns
        assert_frame_equal(self.parq.toDataFrame(columns=columns+['hello']),
                           self.df[columns])

class MultilevelParquetTableTestCase(ParquetTableTestCase):
    """Test case for MultilevelParquetTable
    """
    testFilename = 'multilevel_test.parq'

    def setUp(self):
        super(MultilevelParquetTableTestCase, self).setUp()

        self.datasets = ['meas', 'ref']
        self.filters = ['HSC-G', 'HSC-R']
        self.columns = ['coord_ra', 'coord_dec']

    def getParq(self, filename, df):
        return MultilevelParquetTable(filename), MultilevelParquetTable(dataFrame=df)

    def testProperties(self):
        assert(all([x==y for x,y in zip(self.parq.columnLevels, self.df.columns.names)]))
        assert(len(self.parq.columns)==len(self.df.columns))

        assert(all([x==y for x,y in zip(self.dfParq.columnLevels, self.df.columns.names)]))
        assert(len(self.dfParq.columns)==len(self.df.columns))


    def testColumns(self):
        df = self.df
        parq = self.parq

        # Case A, each level has multiple values
        datasets_A = self.datasets
        filters_A = self.filters
        columns_A = self.columns
        columnDict_A = {'dataset':datasets_A,
                       'filter':filters_A,
                       'column':columns_A}
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
        columnDict_A1 = {'dataset':datasets_A1,
                       'filter':filters_A1,
                       'column':columns_A1}
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
        columnDict_B = {'dataset':datasets_B,
                       'filter':filters_B,
                       'column':columns_B}
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
        columnDict_C = {'dataset':datasets_C,
                       'filter':filters_C}
        df_C = df[datasets_C][filters_C].sort_index(axis=1)
        assert_frame_equal(parq.toDataFrame(columns=columnDict_C), df_C)

        # Case D: Only one level (first level) is provided
        dataset_D = self.datasets[0]
        columnDict_D = {'dataset':dataset_D}
        df_D = df[dataset_D].sort_index(axis=1)
        assert_frame_equal(parq.toDataFrame(columns=columnDict_D), df_D)

        # Case E: Only one level (second level) is provided
        filters_E = self.filters[1]
        columnDict_E = {'filter':filters_E}
        # get second level of multi-index column using .xs()
        df_E = df.xs(filters_E, level=1, axis=1).sort_index(axis=1)
        assert_frame_equal(parq.toDataFrame(columns=columnDict_E), df_E)

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
