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
import itertools

import pandas as pd

import lsst.utils.tests

from lsst.qa.explorer.parquetTable import MultilevelParquetTable
from lsst.qa.explorer.postprocess import PostprocessAnalysis, MultibandPostprocessTask

ROOT = os.path.abspath(os.path.dirname(__file__))


def setup_module(module):
    lsst.utils.tests.init()


class PostprocessTestCase(unittest.TestCase):

    catFilename = 'multilevel_test.parq'
    yamlFile = 'testFunc.yaml'

    def setUp(self):
        self.parq = MultilevelParquetTable(os.path.join(ROOT, self.catFilename))
        self.filters = self.parq.columnLevelNames['filter']
        self.task = MultibandPostprocessTask()
        self.shortFilters = [f for k, f in self.task.config.filterMap.items()
                             if k in self.filters]
        self.task.config.functorFile = self.yamlFile
        self.funcs = self.task.getFunctors()
        self.columnNames = list(self.funcs.funcDict.keys())
        self.columnNames += ['ra', 'dec'] + list(PostprocessAnalysis._defaultFlags)
        self.noDupCols = [k for k, f in self.funcs.funcDict.items() if f.noDup]

    def tearDown(self):
        del self.parq

    def checkMultiLevelResults(self, df, dataId=None):
        assert type(df.columns) == pd.core.indexes.multi.MultiIndex

        assert len(df.columns.levels) == 2  # Make sure two levels
        assert df.columns.names == ['filter', 'column']

        # Make sure the correct columns are there
        assert all([f in df.columns.levels[0] for f in self.filters])
        assert all([c in df.columns.levels[1] for c in self.columnNames])

        if dataId is not None:
            for k in dataId.keys():
                assert all([k in df[f].columns for f in self.filters])

    def checkFlatResults(self, df, dataId=None):
        assert type(df.columns) == pd.core.indexes.base.Index

        noDupCols = list(self.noDupCols)  # Copy
        if dataId is not None:
            noDupCols += list(dataId.keys())

        assert all([c in df.columns for c in self.noDupCols])

        missing = []
        for filt, col in itertools.product(self.shortFilters, self.columnNames):
            if col not in self.noDupCols:
                mungedCol = '{0}_{1}'.format(filt, col)
                if mungedCol not in df.columns:
                    missing.append(mungedCol)

        assert len(missing) == 0

    def testRun(self):

        dataId = {'patch': '4,4'}

        # Test with multilevel output
        self.task.config.multilevelOutput = True

        df = self.task.run(self.parq)
        self.checkMultiLevelResults(df)

        df = self.task.run(self.parq, funcs=self.funcs)
        self.checkMultiLevelResults(df)

        df = self.task.run(self.parq, dataId=dataId)
        self.checkMultiLevelResults(df)

        # Test with flat output
        self.task.config.multilevelOutput = False

        df = self.task.run(self.parq)
        self.checkFlatResults(df)

        df = self.task.run(self.parq, funcs=self.funcs)
        self.checkFlatResults(df)

        df = self.task.run(self.parq, dataId=dataId)
        self.checkFlatResults(df)


