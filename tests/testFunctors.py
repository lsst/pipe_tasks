import os
import unittest

import pandas as pd
import numpy as np

from pandas.util.testing import assert_frame_equal, assert_series_equal

from lsst.utils import getPackageDir
import lsst.utils.tests
from lsst.qa.explorer.parquetTable import MultilevelParquetTable
from lsst.qa.explorer.functors import (CompositeFunctor, CustomFunctor, Column, RAColumn,
                     DecColumn, Mag, MagDiff, Color, StarGalaxyLabeller,
                     DeconvolvedMoments, SdssTraceSize, PsfSdssTraceSizeDiff,
                     HsmTraceSize, PsfHsmTraceSizeDiff, Seeing)


ROOT = os.path.abspath(os.path.dirname(__file__))

def setup_module(module):
    lsst.utils.tests.init()

class FunctorTestCase(unittest.TestCase):
    def setUp(self):
        filename = os.path.join(ROOT, 'multilevel_test.parq')
        self.parq = MultilevelParquetTable(filename)
        self.filters = ['HSC-G', 'HSC-R']

    def _funcVal(self, functor):
        assert type(functor.name) == type('')
        assert type(functor.shortname) == type('')

        val = functor(self.parq)
        assert type(val)==pd.Series

        val = functor(self.parq, dropna=True)
        assert val.isnull().sum()==0

        return val

    def testColumn(self):
        func = Column('base_FootprintArea_value', filt='HSC-G')
        self._funcVal(func)

    def testCustom(self):
        func = CustomFunctor('2*base_FootprintArea_value', filt='HSC-G')
        val = self._funcVal(func)

        func2 = Column('base_FootprintArea_value', filt='HSC-G')

        ## The following fails with different dtypes:
        # assert_series_equal(val, 2*func2(self.parq))

        assert np.all(val.values == 2*func2(self.parq).values)

    def testCoords(self):
        ra = self._funcVal(RAColumn())
        dec = self._funcVal(DecColumn())

        columnDict = {'dataset':'ref', 'filter':'HSC-G',
                      'column':['coord_ra', 'coord_dec']}
        coords = self.parq.toDataFrame(columns=columnDict) / np.pi * 180.

        assert_series_equal(ra, coords['coord_ra'])
        assert_series_equal(dec, coords['coord_dec'])

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

            assert_series_equal((psfMag_G - psfMag_R).dropna(), psfColor_GR)

        # Check that behavior as expected when dataset not provided;
        #  that is, that the color comes from forced and default Mag is meas
        psfMag_G = self._funcVal(Mag(fluxName, filt='HSC-G'))
        psfMag_R = self._funcVal(Mag(fluxName, filt='HSC-R'))

        psfColor_GR = self._funcVal(Color(fluxName, 'HSC-G', 'HSC-R'))

        # These should *not* be equal.
        try:
            assert_series_equal((psfMag_G - psfMag_R).dropna(), psfColor_GR)
            raise AssertionError('Series are equal!')
        except AssertionError:
            pass

    def testMagDiff(self):
        for filt in self.filters:
            filt = 'HSC-G'
            val = self._funcVal(MagDiff('base_PsfFlux', 'modelfit_CModel', filt=filt))

            mag1 = self._funcVal(Mag('modelfit_CModel', filt=filt))
            mag2 = self._funcVal(Mag('base_PsfFlux', filt=filt))

            assert_series_equal((mag2 - mag1).dropna(), val)

    def testLabeller(self):
        labels = self._funcVal(StarGalaxyLabeller())

    def testOther(self):

        for filt in self.filters:
            for Func in [DeconvolvedMoments,
                         SdssTraceSize,
                         PsfSdssTraceSizeDiff,
                         HsmTraceSize, PsfHsmTraceSizeDiff, Seeing]:
                val = self._funcVal(Func(filt=filt))

    def _compositeFuncVal(self, functor):
        assert type(functor) == CompositeFunctor

        df = functor(self.parq)

        assert type(df) == pd.DataFrame
        assert np.all([k in df.columns for k in functor.funcDict.keys()])

        df = functor(self.parq, dropna=True)
        assert (df.isnull().sum() == 0).all()

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
        assert_frame_equal(df, df2)

        func2.filt = 'HSC-R'
        df3 = self._compositeFuncVal(func2)
        try:
            assert_frame_equal(df2, df3)
            raise AssertionError('Dataframes are equal!')
        except:
            pass

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

        df = self._compositeFuncVal(CompositeFunctor(funcDict))

    def testYaml(self):
        functorFile = os.path.join(getPackageDir("qa_explorer"),
                                   'data', 'QAfunctors.yaml')

        func = CompositeFunctor.from_file(functorFile, filt='HSC-G')
        df = self._compositeFuncVal(func)
