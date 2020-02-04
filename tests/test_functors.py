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

import copy
import functools
import numpy as np
import os
import pandas as pd
import unittest

import lsst.utils.tests

# TODO: Remove skipUnless and this try block DM-22256
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lsst.pipe.tasks.parquetTable import MultilevelParquetTable
    from lsst.pipe.tasks.functors import (CompositeFunctor, CustomFunctor, Column, RAColumn,
                                          DecColumn, Mag, MagDiff, Color, StarGalaxyLabeller,
                                          DeconvolvedMoments, SdssTraceSize, PsfSdssTraceSizeDiff,
                                          HsmTraceSize, PsfHsmTraceSizeDiff, HsmFwhm)
    havePyArrow = True
except ImportError:
    havePyArrow = False

ROOT = os.path.abspath(os.path.dirname(__file__))


def setup_module(module):
    lsst.utils.tests.init()


@unittest.skipUnless(havePyArrow, "Requires pyarrow")
class FunctorTestCase(unittest.TestCase):

    def simulateMultiParquet(self, dataDict):
        """Create a simple test MultilevelParquetTable
        """
        simpleDF = pd.DataFrame(dataDict)
        dfFilterDSCombos = []
        for ds in self.datasets:
            for filterName in self.filters:
                df = copy.copy(simpleDF)
                df.reindex(sorted(df.columns), axis=1)
                df['dataset'] = ds
                df['filter'] = filterName
                df.columns = pd.MultiIndex.from_tuples(
                    [(ds, filterName, c) for c in df.columns],
                    names=('dataset', 'filter', 'column'))
                dfFilterDSCombos.append(df)

        df = functools.reduce(lambda d1, d2: d1.join(d2), dfFilterDSCombos)

        return MultilevelParquetTable(dataFrame=df)

    def setUp(self):
        np.random.seed(1234)
        self.datasets = ['forced_src', 'meas', 'ref']
        self.filters = ['HSC-G', 'HSC-R']
        self.columns = ['coord_ra', 'coord_dec']
        self.nRecords = 5
        self.dataDict = {
            "coord_ra": [3.77654137, 3.77643059, 3.77621148, 3.77611944, 3.77610396],
            "coord_dec": [0.01127624, 0.01127787, 0.01127543, 0.01127543, 0.01127543]}

    def _funcVal(self, functor, parq):
        self.assertIsInstance(functor.name, str)
        self.assertIsInstance(functor.shortname, str)

        val = functor(parq)
        self.assertIsInstance(val, pd.Series)

        val = functor(parq, dropna=True)
        self.assertEqual(val.isnull().sum(), 0)

        return val

    def testColumn(self):
        self.columns.append("base_FootprintArea_value")
        self.dataDict["base_FootprintArea_value"] = \
            np.full(self.nRecords, 1)
        parq = self.simulateMultiParquet(self.dataDict)
        func = Column('base_FootprintArea_value', filt='HSC-G')
        self._funcVal(func, parq)

    def testCustom(self):
        self.columns.append("base_FootprintArea_value")
        self.dataDict["base_FootprintArea_value"] = \
            np.random.rand(self.nRecords)
        parq = self.simulateMultiParquet(self.dataDict)
        func = CustomFunctor('2*base_FootprintArea_value', filt='HSC-G')
        val = self._funcVal(func, parq)

        func2 = Column('base_FootprintArea_value', filt='HSC-G')

        np.allclose(val.values, 2*func2(parq).values, atol=1e-13, rtol=0)

    def testCoords(self):
        parq = self.simulateMultiParquet(self.dataDict)
        ra = self._funcVal(RAColumn(), parq)
        dec = self._funcVal(DecColumn(), parq)

        columnDict = {'dataset': 'ref', 'filter': 'HSC-G',
                      'column': ['coord_ra', 'coord_dec']}
        coords = parq.toDataFrame(columns=columnDict) / np.pi * 180.

        self.assertTrue(np.allclose(ra, coords['coord_ra'], atol=1e-13, rtol=0))
        self.assertTrue(np.allclose(dec, coords['coord_dec'], atol=1e-13, rtol=0))

    def testMag(self):
        self.columns.extend(["base_PsfFlux_instFlux", "base_PsfFlux_instFluxErr"])
        self.dataDict["base_PsfFlux_instFlux"] = np.full(self.nRecords, 1000)
        self.dataDict["base_PsfFlux_instFluxErr"] = np.full(self.nRecords, 10)
        parq = self.simulateMultiParquet(self.dataDict)
        fluxName = 'base_PsfFlux'

        # Check that things work when you provide dataset explicitly
        for dataset in ['forced_src', 'meas']:
            psfMag_G = self._funcVal(Mag(fluxName, dataset=dataset,
                                         filt='HSC-G'),
                                     parq)
            psfMag_R = self._funcVal(Mag(fluxName, dataset=dataset,
                                         filt='HSC-R'),
                                     parq)

            psfColor_GR = self._funcVal(Color(fluxName, 'HSC-G', 'HSC-R',
                                              dataset=dataset),
                                        parq)

            self.assertTrue(np.allclose((psfMag_G - psfMag_R).dropna(), psfColor_GR, rtol=0, atol=1e-13))

        # Check that behavior as expected when dataset not provided;
        #  that is, that the color comes from forced and default Mag is meas
        psfMag_G = self._funcVal(Mag(fluxName, filt='HSC-G'), parq)
        psfMag_R = self._funcVal(Mag(fluxName, filt='HSC-R'), parq)

        psfColor_GR = self._funcVal(Color(fluxName, 'HSC-G', 'HSC-R'), parq)

        # These should *not* be equal.
        self.assertTrue(np.allclose((psfMag_G - psfMag_R).dropna(), psfColor_GR))

    def testMagDiff(self):
        self.columns.extend(["base_PsfFlux_instFlux", "base_PsfFlux_instFluxErr",
                             "modelfit_CModel_instFlux", "modelfit_CModel_instFluxErr"])
        self.dataDict["base_PsfFlux_instFlux"] = np.full(self.nRecords, 1000)
        self.dataDict["base_PsfFlux_instFluxErr"] = np.full(self.nRecords, 10)
        self.dataDict["modelfit_CModel_instFlux"] = np.full(self.nRecords, 1000)
        self.dataDict["modelfit_CModel_instFluxErr"] = np.full(self.nRecords, 10)
        parq = self.simulateMultiParquet(self.dataDict)

        for filt in self.filters:
            filt = 'HSC-G'
            val = self._funcVal(MagDiff('base_PsfFlux', 'modelfit_CModel', filt=filt), parq)

            mag1 = self._funcVal(Mag('modelfit_CModel', filt=filt), parq)
            mag2 = self._funcVal(Mag('base_PsfFlux', filt=filt), parq)
            self.assertTrue(np.allclose((mag2 - mag1).dropna(), val, rtol=0, atol=1e-13))

    def testLabeller(self):
        # Covering the code is better than nothing
        parq = self.simulateMultiParquet(self.dataDict)
        labels = self._funcVal(StarGalaxyLabeller(), parq)  # noqa

    def testOther(self):
        parq = self.simulateMultiParquet(self.dataDict)
        # Covering the code is better than nothing
        for filt in self.filters:
            for Func in [DeconvolvedMoments,
                         SdssTraceSize,
                         PsfSdssTraceSizeDiff,
                         HsmTraceSize, PsfHsmTraceSizeDiff, HsmFwhm]:
                val = self._funcVal(Func(filt=filt), parq)  # noqa

    def _compositeFuncVal(self, functor, parq):
        self.assertIsInstance(functor, CompositeFunctor)

        df = functor(parq)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(np.all([k in df.columns for k in functor.funcDict.keys()]))

        df = functor(parq, dropna=True)

        # Check that there are no nulls
        self.assertFalse(df.isnull().any(axis=None))

        return df

    def testComposite(self):
        self.dataDict["base_PsfFlux_instFlux"] = np.full(self.nRecords, 1000)
        self.dataDict["base_PsfFlux_instFluxErr"] = np.full(self.nRecords, 10)
        self.dataDict["modelfit_CModel_instFlux"] = np.full(self.nRecords, 1000)
        self.dataDict["modelfit_CModel_instFluxErr"] = np.full(self.nRecords, 10)
        parq = self.simulateMultiParquet(self.dataDict)
        filt = 'HSC-G'
        funcDict = {'psfMag_ref': Mag('base_PsfFlux', dataset='ref'),
                    'ra': RAColumn(),
                    'dec': DecColumn(),
                    'psfMag': Mag('base_PsfFlux', filt=filt),
                    'cmodel_magDiff': MagDiff('base_PsfFlux',
                                              'modelfit_CModel', filt=filt)}
        func = CompositeFunctor(funcDict)
        df = self._compositeFuncVal(func, parq)

        # Repeat same, but define filter globally instead of individually
        funcDict2 = {'psfMag_ref': Mag('base_PsfFlux', dataset='ref'),
                     'ra': RAColumn(),
                     'dec': DecColumn(),
                     'psfMag': Mag('base_PsfFlux'),
                     'cmodel_magDiff': MagDiff('base_PsfFlux',
                                               'modelfit_CModel')}

        func2 = CompositeFunctor(funcDict2, filt=filt)
        df2 = self._compositeFuncVal(func2, parq)
        self.assertTrue(df.equals(df2))

        func2.filt = 'HSC-R'
        df3 = self._compositeFuncVal(func2, parq)
        self.assertFalse(df2.equals(df3))

        # Make sure things work with passing list instead of dict
        funcs = [Mag('base_PsfFlux', dataset='ref'),
                 RAColumn(),
                 DecColumn(),
                 Mag('base_PsfFlux', filt=filt),
                 MagDiff('base_PsfFlux', 'modelfit_CModel', filt=filt)]

        df = self._compositeFuncVal(CompositeFunctor(funcs), parq)

    def testCompositeColor(self):
        self.dataDict["base_PsfFlux_instFlux"] = np.full(self.nRecords, 1000)
        self.dataDict["base_PsfFlux_instFluxErr"] = np.full(self.nRecords, 10)
        parq = self.simulateMultiParquet(self.dataDict)
        funcDict = {'a': Mag('base_PsfFlux', dataset='meas', filt='HSC-G'),
                    'b': Mag('base_PsfFlux', dataset='forced_src', filt='HSC-G'),
                    'c': Color('base_PsfFlux', 'HSC-G', 'HSC-R')}
        # Covering the code is better than nothing
        df = self._compositeFuncVal(CompositeFunctor(funcDict), parq)  # noqa


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

