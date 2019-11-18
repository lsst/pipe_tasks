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
    def setUp(self):
        df = pd.read_csv(os.path.join(ROOT, 'data', 'test_multilevel_parq.csv.gz'),
                         header=[0, 1, 2], index_col=0)
        with lsst.utils.tests.getTempFilePath('*.parq') as filename:
            table = pa.Table.from_pandas(df)
            pq.write_table(table, filename, compression='none')
            self.parq = MultilevelParquetTable(filename)
        self.filters = ['HSC-G', 'HSC-R']

    def _funcVal(self, functor):
        self.assertIsInstance(functor.name, str)
        self.assertIsInstance(functor.shortname, str)

        val = functor(self.parq)
        self.assertIsInstance(val, pd.Series)

        val = functor(self.parq, dropna=True)
        self.assertEqual(val.isnull().sum(), 0)

        return val

    def testColumn(self):
        func = Column('base_FootprintArea_value', filt='HSC-G')
        self._funcVal(func)

    def testCustom(self):
        func = CustomFunctor('2*base_FootprintArea_value', filt='HSC-G')
        val = self._funcVal(func)

        func2 = Column('base_FootprintArea_value', filt='HSC-G')

        np.allclose(val.values, 2*func2(self.parq).values, atol=1e-13, rtol=0)

    def testCoords(self):
        ra = self._funcVal(RAColumn())
        dec = self._funcVal(DecColumn())

        columnDict = {'dataset': 'ref', 'filter': 'HSC-G',
                      'column': ['coord_ra', 'coord_dec']}
        coords = self.parq.toDataFrame(columns=columnDict) / np.pi * 180.

        self.assertTrue(np.allclose(ra, coords['coord_ra'], atol=1e-13, rtol=0))
        self.assertTrue(np.allclose(dec, coords['coord_dec'], atol=1e-13, rtol=0))

    def testMag(self):
        fluxName = 'base_PsfFlux'

        # Check that things work when you provide dataset explicitly
        for dataset in ['forced_src', 'meas']:
            psfMag_G = self._funcVal(Mag(fluxName, dataset=dataset,
                                         filt='HSC-G'))
            psfMag_R = self._funcVal(Mag(fluxName, dataset=dataset,
                                         filt='HSC-R'))

            psfColor_GR = self._funcVal(Color(fluxName, 'HSC-G', 'HSC-R',
                                              dataset=dataset))

            self.assertTrue(np.allclose((psfMag_G - psfMag_R).dropna(), psfColor_GR, rtol=0, atol=1e-13))

        # Check that behavior as expected when dataset not provided;
        #  that is, that the color comes from forced and default Mag is meas
        psfMag_G = self._funcVal(Mag(fluxName, filt='HSC-G'))
        psfMag_R = self._funcVal(Mag(fluxName, filt='HSC-R'))

        psfColor_GR = self._funcVal(Color(fluxName, 'HSC-G', 'HSC-R'))

        # These should *not* be equal.
        self.assertFalse(np.allclose((psfMag_G - psfMag_R).dropna(), psfColor_GR))

    def testMagDiff(self):
        for filt in self.filters:
            filt = 'HSC-G'
            val = self._funcVal(MagDiff('base_PsfFlux', 'modelfit_CModel', filt=filt))

            mag1 = self._funcVal(Mag('modelfit_CModel', filt=filt))
            mag2 = self._funcVal(Mag('base_PsfFlux', filt=filt))
            self.assertTrue(np.allclose((mag2 - mag1).dropna(), val, rtol=0, atol=1e-13))

    def testLabeller(self):
        # Covering the code is better than nothing
        labels = self._funcVal(StarGalaxyLabeller())  # noqa

    def testOther(self):
        # Covering the code is better than nothing
        for filt in self.filters:
            for Func in [DeconvolvedMoments,
                         SdssTraceSize,
                         PsfSdssTraceSizeDiff,
                         HsmTraceSize, PsfHsmTraceSizeDiff, HsmFwhm]:
                val = self._funcVal(Func(filt=filt))  # noqa

    def _compositeFuncVal(self, functor):
        self.assertIsInstance(functor, CompositeFunctor)

        df = functor(self.parq)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(np.all([k in df.columns for k in functor.funcDict.keys()]))

        df = functor(self.parq, dropna=True)

        # Check that there are no nulls
        self.assertFalse(df.isnull().any(axis=None))

        return df

    def testComposite(self):
        filt = 'HSC-G'
        funcDict = {'psfMag_ref': Mag('base_PsfFlux', dataset='ref'),
                    'ra': RAColumn(),
                    'dec': DecColumn(),
                    'psfMag': Mag('base_PsfFlux', filt=filt),
                    'cmodel_magDiff': MagDiff('base_PsfFlux',
                                              'modelfit_CModel', filt=filt)}
        func = CompositeFunctor(funcDict)
        df = self._compositeFuncVal(func)

        # Repeat same, but define filter globally instead of individually
        funcDict2 = {'psfMag_ref': Mag('base_PsfFlux', dataset='ref'),
                     'ra': RAColumn(),
                     'dec': DecColumn(),
                     'psfMag': Mag('base_PsfFlux'),
                     'cmodel_magDiff': MagDiff('base_PsfFlux',
                                               'modelfit_CModel')}

        func2 = CompositeFunctor(funcDict2, filt=filt)
        df2 = self._compositeFuncVal(func2)
        self.assertTrue(df.equals(df2))

        func2.filt = 'HSC-R'
        df3 = self._compositeFuncVal(func2)
        self.assertFalse(df2.equals(df3))

        # Make sure things work with passing list instead of dict
        funcs = [Mag('base_PsfFlux', dataset='ref'),
                 RAColumn(),
                 DecColumn(),
                 Mag('base_PsfFlux', filt=filt),
                 MagDiff('base_PsfFlux', 'modelfit_CModel', filt=filt)]

        df = self._compositeFuncVal(CompositeFunctor(funcs))

    def testCompositeColor(self):
        funcDict = {'a': Mag('base_PsfFlux', dataset='meas', filt='HSC-G'),
                    'b': Mag('base_PsfFlux', dataset='forced_src', filt='HSC-G'),
                    'c': Color('base_PsfFlux', 'HSC-G', 'HSC-R')}
        # Covering the code is better than nothing
        df = self._compositeFuncVal(CompositeFunctor(funcDict))  # noqa
