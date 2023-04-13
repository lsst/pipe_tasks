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

import astropy.units as u
import copy
import functools
import numpy as np
import os
import pandas as pd
import unittest

import lsst.daf.base as dafBase
import lsst.afw.geom as afwGeom
import lsst.geom as geom
from lsst.sphgeom import HtmPixelization
import lsst.meas.base as measBase
import lsst.utils.tests
from lsst.pipe.base import InMemoryDatasetHandle
# from lsst.pipe.tasks.parquetTable import MultilevelParquetTable, ParquetTable
from lsst.pipe.tasks.functors import (CompositeFunctor, CustomFunctor, Column, RAColumn,
                                      DecColumn, Mag, MagDiff, Color, StarGalaxyLabeller,
                                      DeconvolvedMoments, SdssTraceSize, PsfSdssTraceSizeDiff,
                                      HsmTraceSize, PsfHsmTraceSizeDiff, HsmFwhm,
                                      LocalPhotometry, LocalNanojansky, LocalNanojanskyErr,
                                      LocalMagnitude, LocalMagnitudeErr,
                                      LocalDipoleMeanFlux, LocalDipoleMeanFluxErr,
                                      LocalDipoleDiffFlux, LocalDipoleDiffFluxErr,
                                      LocalWcs, ComputePixelScale, ConvertPixelToArcseconds,
                                      ConvertPixelSqToArcsecondsSq, Ratio, HtmIndex20)

ROOT = os.path.abspath(os.path.dirname(__file__))


class FunctorTestCase(unittest.TestCase):

    def simulateMultiParquet(self, dataDict):
        """Create a simple test MultilevelParquetTable
        """
        simpleDF = pd.DataFrame(dataDict)
        dfFilterDSCombos = []
        for ds in self.datasets:
            for band in self.bands:
                df = copy.copy(simpleDF)
                df.reindex(sorted(df.columns), axis=1)
                df['dataset'] = ds
                df['band'] = band
                df.columns = pd.MultiIndex.from_tuples(
                    [(ds, band, c) for c in df.columns],
                    names=('dataset', 'band', 'column'))
                dfFilterDSCombos.append(df)

        df = functools.reduce(lambda d1, d2: d1.join(d2), dfFilterDSCombos)

        # return MultilevelParquetTable(dataFrame=df)
        return df

    def simulateParquet(self, dataDict):
        df = pd.DataFrame(dataDict)
        # return ParquetTable(dataFrame=df)
        return df

    def getDatasetHandle(self, parq):
        # df = parq._df
        df = parq
        lo, hi = HtmPixelization(7).universe().ranges()[0]
        value = np.random.randint(lo, hi)
        return InMemoryDatasetHandle(df, storageClass="DataFrame", dataId={"htm7": value})

    def setUp(self):
        np.random.seed(12345)
        self.datasets = ['forced_src', 'meas', 'ref']
        self.bands = ['g', 'r']
        self.columns = ['coord_ra', 'coord_dec']
        self.nRecords = 5
        self.dataDict = {
            "coord_ra": [3.77654137, 3.77643059, 3.77621148, 3.77611944, 3.77610396],
            "coord_dec": [0.01127624, 0.01127787, 0.01127543, 0.01127543, 0.01127543]}

    def _funcVal(self, functor, parq):
        self.assertIsInstance(functor.name, str)
        self.assertIsInstance(functor.shortname, str)

        handle = self.getDatasetHandle(parq)

        val = functor(parq)
        val2 = functor(handle)

        self.assertTrue((val == val2).all())
        self.assertIsInstance(val, pd.Series)

        val = functor(parq, dropna=True)
        val2 = functor(handle, dropna=True)
        self.assertTrue((val == val2).all())
        self.assertEqual(val.isnull().sum(), 0)

        return val

    def _differenceVal(self, functor, parq1, parq2):
        self.assertIsInstance(functor.name, str)
        self.assertIsInstance(functor.shortname, str)

        handle1 = self.getDatasetHandle(parq1)
        handle2 = self.getDatasetHandle(parq2)

        val = functor.difference(parq1, parq2)
        val2 = functor.difference(handle1, handle2)
        self.assertTrue(val.equals(val2))
        self.assertIsInstance(val, pd.Series)

        val = functor.difference(parq1, parq2, dropna=True)
        val2 = functor.difference(handle1, handle2, dropna=True)
        self.assertTrue(val.equals(val2))
        self.assertEqual(val.isnull().sum(), 0)

        val1 = self._funcVal(functor, parq1)
        val2 = self._funcVal(functor, parq2)

        self.assertTrue(np.allclose(val, val1 - val2))

        return val

    def testColumn(self):
        self.columns.append("base_FootprintArea_value")
        self.dataDict["base_FootprintArea_value"] = \
            np.full(self.nRecords, 1)
        parq = self.simulateMultiParquet(self.dataDict)
        func = Column('base_FootprintArea_value', filt='g')
        self._funcVal(func, parq)

        parq = self.simulateParquet(self.dataDict)
        func = Column('base_FootprintArea_value')
        self._funcVal(func, parq)

    def testCustom(self):
        self.columns.append("base_FootprintArea_value")
        self.dataDict["base_FootprintArea_value"] = \
            np.random.rand(self.nRecords)
        parq = self.simulateMultiParquet(self.dataDict)
        func = CustomFunctor('2*base_FootprintArea_value', filt='g')
        val = self._funcVal(func, parq)

        func2 = Column('base_FootprintArea_value', filt='g')

        np.allclose(val.values, 2*func2(parq).values, atol=1e-13, rtol=0)

        parq = self.simulateParquet(self.dataDict)
        func = CustomFunctor('2 * base_FootprintArea_value')
        val = self._funcVal(func, parq)
        func2 = Column('base_FootprintArea_value')

        np.allclose(val.values, 2*func2(parq).values, atol=1e-13, rtol=0)

    def testCoords(self):
        parq = self.simulateMultiParquet(self.dataDict)
        ra = self._funcVal(RAColumn(), parq)
        dec = self._funcVal(DecColumn(), parq)

        columnDict = {'dataset': 'ref', 'band': 'g',
                      'column': ['coord_ra', 'coord_dec']}

        coords = parq.toDataFrame(columns=columnDict, droplevels=True) / np.pi * 180.

        self.assertTrue(np.allclose(ra, coords[('ref', 'g', 'coord_ra')], atol=1e-13, rtol=0))
        self.assertTrue(np.allclose(dec, coords[('ref', 'g', 'coord_dec')], atol=1e-13, rtol=0))

        # single-level column index table
        parq = self.simulateParquet(self.dataDict)
        ra = self._funcVal(RAColumn(), parq)
        dec = self._funcVal(DecColumn(), parq)

        coords = parq.toDataFrame(columns=['coord_ra', 'coord_dec']) / np.pi * 180.

        self.assertTrue(np.allclose(ra, coords['coord_ra'], atol=1e-13, rtol=0))
        self.assertTrue(np.allclose(dec, coords['coord_dec'], atol=1e-13, rtol=0))

    def testMag(self):
        self.columns.extend(["base_PsfFlux_instFlux", "base_PsfFlux_instFluxErr"])
        self.dataDict["base_PsfFlux_instFlux"] = np.full(self.nRecords, 1000)
        self.dataDict["base_PsfFlux_instFluxErr"] = np.full(self.nRecords, 10)
        parq = self.simulateMultiParquet(self.dataDict)
        # Change one dataset filter combinations value.
        # parq._df[("meas", "g", "base_PsfFlux_instFlux")] -= 1
        parq[("meas", "g", "base_PsfFlux_instFlux")] -= 1

        fluxName = 'base_PsfFlux'

        # Check that things work when you provide dataset explicitly
        for dataset in ['forced_src', 'meas']:
            psfMag_G = self._funcVal(Mag(fluxName, dataset=dataset,
                                         filt='g'),
                                     parq)
            psfMag_R = self._funcVal(Mag(fluxName, dataset=dataset,
                                         filt='r'),
                                     parq)

            psfColor_GR = self._funcVal(Color(fluxName, 'g', 'r',
                                              dataset=dataset),
                                        parq)

            self.assertTrue(np.allclose((psfMag_G - psfMag_R).dropna(), psfColor_GR, rtol=0, atol=1e-13))

        # Check that behavior as expected when dataset not provided;
        #  that is, that the color comes from forced and default Mag is meas
        psfMag_G = self._funcVal(Mag(fluxName, filt='g'), parq)
        psfMag_R = self._funcVal(Mag(fluxName, filt='r'), parq)

        psfColor_GR = self._funcVal(Color(fluxName, 'g', 'r'), parq)

        # These should *not* be equal.
        self.assertFalse(np.allclose((psfMag_G - psfMag_R).dropna(), psfColor_GR))

    def testMagDiff(self):
        self.columns.extend(["base_PsfFlux_instFlux", "base_PsfFlux_instFluxErr",
                             "modelfit_CModel_instFlux", "modelfit_CModel_instFluxErr"])
        self.dataDict["base_PsfFlux_instFlux"] = np.full(self.nRecords, 1000)
        self.dataDict["base_PsfFlux_instFluxErr"] = np.full(self.nRecords, 10)
        self.dataDict["modelfit_CModel_instFlux"] = np.full(self.nRecords, 1000)
        self.dataDict["modelfit_CModel_instFluxErr"] = np.full(self.nRecords, 10)
        parq = self.simulateMultiParquet(self.dataDict)

        for filt in self.bands:
            filt = 'g'
            val = self._funcVal(MagDiff('base_PsfFlux', 'modelfit_CModel', filt=filt), parq)

            mag1 = self._funcVal(Mag('modelfit_CModel', filt=filt), parq)
            mag2 = self._funcVal(Mag('base_PsfFlux', filt=filt), parq)
            self.assertTrue(np.allclose((mag2 - mag1).dropna(), val, rtol=0, atol=1e-13))

    def testDifference(self):
        """Test .difference method using MagDiff as the example.
        """
        self.columns.extend(["base_PsfFlux_instFlux", "base_PsfFlux_instFluxErr",
                             "modelfit_CModel_instFlux", "modelfit_CModel_instFluxErr"])

        self.dataDict["base_PsfFlux_instFlux"] = np.full(self.nRecords, 1000)
        self.dataDict["modelfit_CModel_instFlux"] = np.full(self.nRecords, 1000)
        parq1 = self.simulateMultiParquet(self.dataDict)

        self.dataDict["base_PsfFlux_instFlux"] = np.full(self.nRecords, 999)
        self.dataDict["modelfit_CModel_instFlux"] = np.full(self.nRecords, 999)
        parq2 = self.simulateMultiParquet(self.dataDict)

        magDiff = MagDiff('base_PsfFlux', 'modelfit_CModel', filt='g')

        # Asserts that differences computed properly
        self._differenceVal(magDiff, parq1, parq2)

    def testLabeller(self):
        # Covering the code is better than nothing
        self.columns.append("base_ClassificationExtendedness_value")
        self.dataDict["base_ClassificationExtendedness_value"] = np.full(self.nRecords, 1)
        parq = self.simulateMultiParquet(self.dataDict)
        labels = self._funcVal(StarGalaxyLabeller(), parq)  # noqa

    def testPixelScale(self):
        # Test that the pixel scale and pix->arcsec calculations perform as
        # expected.
        pass

    def testOther(self):
        self.columns.extend(["ext_shapeHSM_HsmSourceMoments_xx", "ext_shapeHSM_HsmSourceMoments_yy",
                             "base_SdssShape_xx", "base_SdssShape_yy",
                             "ext_shapeHSM_HsmPsfMoments_xx", "ext_shapeHSM_HsmPsfMoments_yy",
                             "base_SdssShape_psf_xx", "base_SdssShape_psf_yy"])
        self.dataDict["ext_shapeHSM_HsmSourceMoments_xx"] = np.full(self.nRecords, 1 / np.sqrt(2))
        self.dataDict["ext_shapeHSM_HsmSourceMoments_yy"] = np.full(self.nRecords, 1 / np.sqrt(2))
        self.dataDict["base_SdssShape_xx"] = np.full(self.nRecords, 1 / np.sqrt(2))
        self.dataDict["base_SdssShape_yy"] = np.full(self.nRecords, 1 / np.sqrt(2))
        self.dataDict["ext_shapeHSM_HsmPsfMoments_xx"] = np.full(self.nRecords, 1 / np.sqrt(2))
        self.dataDict["ext_shapeHSM_HsmPsfMoments_yy"] = np.full(self.nRecords, 1 / np.sqrt(2))
        self.dataDict["base_SdssShape_psf_xx"] = np.full(self.nRecords, 1)
        self.dataDict["base_SdssShape_psf_yy"] = np.full(self.nRecords, 1)
        parq = self.simulateMultiParquet(self.dataDict)
        # Covering the code is better than nothing
        for filt in self.bands:
            for Func in [DeconvolvedMoments,
                         SdssTraceSize,
                         PsfSdssTraceSizeDiff,
                         HsmTraceSize, PsfHsmTraceSizeDiff, HsmFwhm]:
                val = self._funcVal(Func(filt=filt), parq)  # noqa

    def _compositeFuncVal(self, functor, parq):
        self.assertIsInstance(functor, CompositeFunctor)

        handle = self.getDatasetHandle(parq)

        df = functor(parq)
        df2 = functor(handle)
        self.assertTrue(df.equals(df2))

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(np.all([k in df.columns for k in functor.funcDict.keys()]))

        df = functor(parq, dropna=True)
        df2 = functor(handle, dropna=True)
        self.assertTrue(df.equals(df2))

        # Check that there are no nulls
        self.assertFalse(df.isnull().any(axis=None))

        return df

    def _compositeDifferenceVal(self, functor, parq1, parq2):
        self.assertIsInstance(functor, CompositeFunctor)

        handle1 = self.getDatasetHandle(parq1)
        handle2 = self.getDatasetHandle(parq2)

        df = functor.difference(parq1, parq2)
        df2 = functor.difference(handle1, handle2)
        self.assertTrue(df.equals(df2))

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(np.all([k in df.columns for k in functor.funcDict.keys()]))

        df = functor.difference(parq1, parq2, dropna=True)
        df2 = functor.difference(handle1, handle2, dropna=True)
        self.assertTrue(df.equals(df2))

        # Check that there are no nulls
        self.assertFalse(df.isnull().any(axis=None))

        df1 = functor(parq1)
        df2 = functor(parq2)

        self.assertTrue(np.allclose(df.values, df1.values - df2.values))

        return df

    def testComposite(self):
        self.columns.extend(["modelfit_CModel_instFlux", "base_PsfFlux_instFlux"])
        self.dataDict["modelfit_CModel_instFlux"] = np.full(self.nRecords, 1)
        self.dataDict["base_PsfFlux_instFlux"] = np.full(self.nRecords, 1)

        parq = self.simulateMultiParquet(self.dataDict)
        # Modify r band value slightly.
        # parq._df[("meas", "r", "base_PsfFlux_instFlux")] -= 0.1
        parq[("meas", "r", "base_PsfFlux_instFlux")] -= 0.1

        filt = 'g'
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

        func2.filt = 'r'
        df3 = self._compositeFuncVal(func2, parq)
        # Because we modified the R filter this should fail.
        self.assertFalse(df2.equals(df3))

        # Make sure things work with passing list instead of dict
        funcs = [Mag('base_PsfFlux', dataset='ref'),
                 RAColumn(),
                 DecColumn(),
                 Mag('base_PsfFlux', filt=filt),
                 MagDiff('base_PsfFlux', 'modelfit_CModel', filt=filt)]

        df = self._compositeFuncVal(CompositeFunctor(funcs), parq)

    def testCompositeSimple(self):
        """Test single-level composite functor for functionality
        """
        self.columns.extend(["modelfit_CModel_instFlux", "base_PsfFlux_instFlux"])
        self.dataDict["modelfit_CModel_instFlux"] = np.full(self.nRecords, 1)
        self.dataDict["base_PsfFlux_instFlux"] = np.full(self.nRecords, 1)

        parq = self.simulateParquet(self.dataDict)
        funcDict = {'ra': RAColumn(),
                    'dec': DecColumn(),
                    'psfMag': Mag('base_PsfFlux'),
                    'cmodel_magDiff': MagDiff('base_PsfFlux',
                                              'modelfit_CModel')}
        func = CompositeFunctor(funcDict)
        df = self._compositeFuncVal(func, parq)  # noqa

    def testCompositeColor(self):
        self.dataDict["base_PsfFlux_instFlux"] = np.full(self.nRecords, 1000)
        self.dataDict["base_PsfFlux_instFluxErr"] = np.full(self.nRecords, 10)
        parq = self.simulateMultiParquet(self.dataDict)
        funcDict = {'a': Mag('base_PsfFlux', dataset='meas', filt='g'),
                    'b': Mag('base_PsfFlux', dataset='forced_src', filt='g'),
                    'c': Color('base_PsfFlux', 'g', 'r')}
        # Covering the code is better than nothing
        df = self._compositeFuncVal(CompositeFunctor(funcDict), parq)  # noqa

    def testCompositeDifference(self):
        self.dataDict["base_PsfFlux_instFlux"] = np.full(self.nRecords, 1000)
        self.dataDict["base_PsfFlux_instFluxErr"] = np.full(self.nRecords, 10)
        parq1 = self.simulateMultiParquet(self.dataDict)

        self.dataDict["base_PsfFlux_instFlux"] = np.full(self.nRecords, 999)
        self.dataDict["base_PsfFlux_instFluxErr"] = np.full(self.nRecords, 9)
        parq2 = self.simulateMultiParquet(self.dataDict)

        funcDict = {'a': Mag('base_PsfFlux', dataset='meas', filt='g'),
                    'b': Mag('base_PsfFlux', dataset='forced_src', filt='g'),
                    'c': Color('base_PsfFlux', 'g', 'r')}
        # Covering the code is better than nothing
        df = self._compositeDifferenceVal(CompositeFunctor(funcDict), parq1, parq2)  # noqa

    def testCompositeFail(self):
        """Test a composite functor where one of the functors should be junk.
        """
        self.dataDict["base_PsfFlux_instFlux"] = np.full(self.nRecords, 1000)
        parq = self.simulateMultiParquet(self.dataDict)

        funcDict = {'good': Column("base_PsfFlux_instFlux"),
                    'bad': Column('not_a_column')}

        df = self._compositeFuncVal(CompositeFunctor(funcDict), parq)  # noqa

    def testLocalPhotometry(self):
        """Test the local photometry functors.
        """
        flux = 1000
        fluxErr = 10
        calib = 10
        calibErr = 1
        self.dataDict["base_PsfFlux_instFlux"] = np.full(self.nRecords, flux)
        self.dataDict["base_PsfFlux_instFluxErr"] = np.full(self.nRecords,
                                                            fluxErr)
        self.dataDict["base_LocalPhotoCalib"] = np.full(self.nRecords, calib)
        self.dataDict["base_LocalPhotoCalibErr"] = np.full(self.nRecords,
                                                           calibErr)
        parq = self.simulateMultiParquet(self.dataDict)
        func = LocalPhotometry("base_PsfFlux_instFlux",
                               "base_PsfFlux_instFluxErr",
                               "base_LocalPhotoCalib",
                               "base_LocalPhotoCalibErr")
        df = parq.toDataFrame(columns={"dataset": "meas",
                                       "band": "g",
                                       "columns": ["base_PsfFlux_instFlux",
                                                   "base_PsfFlux_instFluxErr",
                                                   "base_LocalPhotoCalib",
                                                   "base_LocalPhotoCalibErr"]})
        nanoJansky = func.instFluxToNanojansky(
            df[("meas", "g", "base_PsfFlux_instFlux")],
            df[("meas", "g", "base_LocalPhotoCalib")])
        mag = func.instFluxToMagnitude(
            df[("meas", "g", "base_PsfFlux_instFlux")],
            df[("meas", "g", "base_LocalPhotoCalib")])
        nanoJanskyErr = func.instFluxErrToNanojanskyErr(
            df[("meas", "g", "base_PsfFlux_instFlux")],
            df[("meas", "g", "base_PsfFlux_instFluxErr")],
            df[("meas", "g", "base_LocalPhotoCalib")],
            df[("meas", "g", "base_LocalPhotoCalibErr")])
        magErr = func.instFluxErrToMagnitudeErr(
            df[("meas", "g", "base_PsfFlux_instFlux")],
            df[("meas", "g", "base_PsfFlux_instFluxErr")],
            df[("meas", "g", "base_LocalPhotoCalib")],
            df[("meas", "g", "base_LocalPhotoCalibErr")])

        self.assertTrue(np.allclose(nanoJansky.values,
                                    flux * calib,
                                    atol=1e-13,
                                    rtol=0))
        self.assertTrue(np.allclose(mag.values,
                                    (flux * calib * u.nJy).to_value(u.ABmag),
                                    atol=1e-13,
                                    rtol=0))
        self.assertTrue(np.allclose(nanoJanskyErr.values,
                                    np.hypot(fluxErr * calib, flux * calibErr),
                                    atol=1e-13,
                                    rtol=0))
        self.assertTrue(np.allclose(
            magErr.values,
            2.5 / np.log(10) * nanoJanskyErr.values / nanoJansky.values,
            atol=1e-13,
            rtol=0))

        # Test functors against the values computed above.
        self._testLocalPhotometryFunctors(LocalNanojansky,
                                          parq,
                                          nanoJansky)
        self._testLocalPhotometryFunctors(LocalNanojanskyErr,
                                          parq,
                                          nanoJanskyErr)
        self._testLocalPhotometryFunctors(LocalMagnitude,
                                          parq,
                                          mag)
        self._testLocalPhotometryFunctors(LocalMagnitudeErr,
                                          parq,
                                          magErr)

    def _testLocalPhotometryFunctors(self, functor, parq, testValues):
        func = functor("base_PsfFlux_instFlux",
                       "base_PsfFlux_instFluxErr",
                       "base_LocalPhotoCalib",
                       "base_LocalPhotoCalibErr")
        val = self._funcVal(func, parq)
        self.assertTrue(np.allclose(testValues.values,
                                    val.values,
                                    atol=1e-13,
                                    rtol=0))

    def testDipPhotometry(self):
        """Test calibrated flux calculations for dipoles."""
        fluxNeg = -100
        fluxPos = 101
        fluxErr = 10
        calib = 10
        calibErr = 1

        # compute expected values.
        absMean = 0.5*(fluxPos - fluxNeg)*calib
        absDiff = (fluxNeg + fluxPos)*calib
        absMeanErr = 0.5*np.sqrt(2*(fluxErr*calib)**2
                                 + ((fluxPos - fluxNeg)*calibErr)**2)
        absDiffErr = np.sqrt(2*(fluxErr*calib)**2
                             + ((fluxPos + fluxNeg)*calibErr)**2)

        self.dataDict["ip_diffim_DipoleFluxNeg_instFlux"] = np.full(self.nRecords, fluxNeg)
        self.dataDict["ip_diffim_DipoleFluxNeg_instFluxErr"] = np.full(self.nRecords, fluxErr)
        self.dataDict["ip_diffim_DipoleFluxPos_instFlux"] = np.full(self.nRecords, fluxPos)
        self.dataDict["ip_diffim_DipoleFluxPos_instFluxErr"] = np.full(self.nRecords, fluxErr)
        self.dataDict["base_LocalPhotoCalib"] = np.full(self.nRecords, calib)
        self.dataDict["base_LocalPhotoCalibErr"] = np.full(self.nRecords,
                                                           calibErr)

        parq = self.simulateMultiParquet(self.dataDict)
        func = LocalDipoleMeanFlux("ip_diffim_DipoleFluxPos_instFlux",
                                   "ip_diffim_DipoleFluxNeg_instFlux",
                                   "ip_diffim_DipoleFluxPos_instFluxErr",
                                   "ip_diffim_DipoleFluxNeg_instFluxErr",
                                   "base_LocalPhotoCalib",
                                   "base_LocalPhotoCalibErr")
        val = self._funcVal(func, parq)
        self.assertTrue(np.allclose(val.values,
                                    absMean,
                                    atol=1e-13,
                                    rtol=0))

        func = LocalDipoleMeanFluxErr("ip_diffim_DipoleFluxPos_instFlux",
                                      "ip_diffim_DipoleFluxNeg_instFlux",
                                      "ip_diffim_DipoleFluxPos_instFluxErr",
                                      "ip_diffim_DipoleFluxNeg_instFluxErr",
                                      "base_LocalPhotoCalib",
                                      "base_LocalPhotoCalibErr")
        val = self._funcVal(func, parq)
        self.assertTrue(np.allclose(val.values,
                                    absMeanErr,
                                    atol=1e-13,
                                    rtol=0))

        func = LocalDipoleDiffFlux("ip_diffim_DipoleFluxPos_instFlux",
                                   "ip_diffim_DipoleFluxNeg_instFlux",
                                   "ip_diffim_DipoleFluxPos_instFluxErr",
                                   "ip_diffim_DipoleFluxNeg_instFluxErr",
                                   "base_LocalPhotoCalib",
                                   "base_LocalPhotoCalibErr")
        val = self._funcVal(func, parq)
        self.assertTrue(np.allclose(val.values,
                                    absDiff,
                                    atol=1e-13,
                                    rtol=0))

        func = LocalDipoleDiffFluxErr("ip_diffim_DipoleFluxPos_instFlux",
                                      "ip_diffim_DipoleFluxNeg_instFlux",
                                      "ip_diffim_DipoleFluxPos_instFluxErr",
                                      "ip_diffim_DipoleFluxNeg_instFluxErr",
                                      "base_LocalPhotoCalib",
                                      "base_LocalPhotoCalibErr")
        val = self._funcVal(func, parq)
        self.assertTrue(np.allclose(val.values,
                                    absDiffErr,
                                    atol=1e-13,
                                    rtol=0))

    def testConvertPixelToArcseconds(self):
        """Test calculations of the pixel scale and conversions of pixel to
        arcseconds.
        """
        dipoleSep = 10
        ixx = 10
        testPixelDeltas = np.random.uniform(-100, 100, size=(self.nRecords, 2))
        import lsst.afw.table as afwTable
        localWcsPlugin = measBase.EvaluateLocalWcsPlugin(
            None,
            "base_LocalWcs",
            afwTable.SourceTable.makeMinimalSchema(),
            None)
        for dec in np.linspace(-90, 90, 10):
            for x, y in zip(np.random.uniform(2 * 1109.99981456774, size=10),
                            np.random.uniform(2 * 560.018167811613, size=10)):
                center = geom.Point2D(x, y)
                wcs = self._makeWcs(dec)
                skyOrigin = wcs.pixelToSky(center)

                linAffMatrix = localWcsPlugin.makeLocalTransformMatrix(wcs,
                                                                       center)
                self.dataDict["dipoleSep"] = np.full(self.nRecords, dipoleSep)
                self.dataDict["ixx"] = np.full(self.nRecords, ixx)
                self.dataDict["slot_Centroid_x"] = np.full(self.nRecords, x)
                self.dataDict["slot_Centroid_y"] = np.full(self.nRecords, y)
                self.dataDict["someCentroid_x"] = x + testPixelDeltas[:, 0]
                self.dataDict["someCentroid_y"] = y + testPixelDeltas[:, 1]
                self.dataDict["base_LocalWcs_CDMatrix_1_1"] = np.full(self.nRecords,
                                                                      linAffMatrix[0, 0])
                self.dataDict["base_LocalWcs_CDMatrix_1_2"] = np.full(self.nRecords,
                                                                      linAffMatrix[0, 1])
                self.dataDict["base_LocalWcs_CDMatrix_2_1"] = np.full(self.nRecords,
                                                                      linAffMatrix[1, 0])
                self.dataDict["base_LocalWcs_CDMatrix_2_2"] = np.full(self.nRecords,
                                                                      linAffMatrix[1, 1])
                parq = self.simulateMultiParquet(self.dataDict)
                func = LocalWcs("base_LocalWcs_CDMatrix_1_1",
                                "base_LocalWcs_CDMatrix_1_2",
                                "base_LocalWcs_CDMatrix_2_1",
                                "base_LocalWcs_CDMatrix_2_2")
                df = parq.toDataFrame(columns={"dataset": "meas",
                                               "band": "g",
                                               "columns": ["dipoleSep",
                                                           "slot_Centroid_x",
                                                           "slot_Centroid_y",
                                                           "someCentroid_x",
                                                           "someCentroid_y",
                                                           "base_LocalWcs_CDMatrix_1_1",
                                                           "base_LocalWcs_CDMatrix_1_2",
                                                           "base_LocalWcs_CDMatrix_2_1",
                                                           "base_LocalWcs_CDMatrix_2_2"]})

                # Exercise the full set of functions in LocalWcs.
                sepRadians = func.getSkySeperationFromPixel(
                    df[("meas", "g", "someCentroid_x")] - df[("meas", "g", "slot_Centroid_x")],
                    df[("meas", "g", "someCentroid_y")] - df[("meas", "g", "slot_Centroid_y")],
                    0.0,
                    0.0,
                    df[("meas", "g", "base_LocalWcs_CDMatrix_1_1")],
                    df[("meas", "g", "base_LocalWcs_CDMatrix_1_2")],
                    df[("meas", "g", "base_LocalWcs_CDMatrix_2_1")],
                    df[("meas", "g", "base_LocalWcs_CDMatrix_2_2")])

                # Test functor values against afw SkyWcs computations.
                for centX, centY, sep in zip(testPixelDeltas[:, 0],
                                             testPixelDeltas[:, 1],
                                             sepRadians.values):
                    afwSepRadians = skyOrigin.separation(
                        wcs.pixelToSky(x + centX, y + centY)).asRadians()
                    self.assertAlmostEqual(1 - sep / afwSepRadians, 0, places=6)

                # Test the pixel scale computation.
                func = ComputePixelScale("base_LocalWcs_CDMatrix_1_1",
                                         "base_LocalWcs_CDMatrix_1_2",
                                         "base_LocalWcs_CDMatrix_2_1",
                                         "base_LocalWcs_CDMatrix_2_2")
                pixelScale = self._funcVal(func, parq)
                self.assertTrue(np.allclose(
                    wcs.getPixelScale(center).asArcseconds(),
                    pixelScale.values,
                    rtol=1e-8,
                    atol=0))

                # Test pixel -> arcsec conversion.
                func = ConvertPixelToArcseconds("dipoleSep",
                                                "base_LocalWcs_CDMatrix_1_1",
                                                "base_LocalWcs_CDMatrix_1_2",
                                                "base_LocalWcs_CDMatrix_2_1",
                                                "base_LocalWcs_CDMatrix_2_2")
                val = self._funcVal(func, parq)
                self.assertTrue(np.allclose(pixelScale.values * dipoleSep,
                                            val.values,
                                            atol=1e-16,
                                            rtol=1e-16))

                # Test pixel^2 -> arcsec^2 conversion.
                func = ConvertPixelSqToArcsecondsSq("ixx",
                                                    "base_LocalWcs_CDMatrix_1_1",
                                                    "base_LocalWcs_CDMatrix_1_2",
                                                    "base_LocalWcs_CDMatrix_2_1",
                                                    "base_LocalWcs_CDMatrix_2_2")
                val = self._funcVal(func, parq)
                self.assertTrue(np.allclose(pixelScale.values ** 2 * dipoleSep,
                                            val.values,
                                            atol=1e-16,
                                            rtol=1e-16))

    def _makeWcs(self, dec=53.1595451514076):
        """Create a wcs from real CFHT values.

        Returns
        -------
        wcs : `lsst.afw.geom`
            Created wcs.
        """
        metadata = dafBase.PropertySet()

        metadata.set("SIMPLE", "T")
        metadata.set("BITPIX", -32)
        metadata.set("NAXIS", 2)
        metadata.set("NAXIS1", 1024)
        metadata.set("NAXIS2", 1153)
        metadata.set("RADECSYS", 'FK5')
        metadata.set("EQUINOX", 2000.)

        metadata.setDouble("CRVAL1", 215.604025685476)
        metadata.setDouble("CRVAL2", dec)
        metadata.setDouble("CRPIX1", 1109.99981456774)
        metadata.setDouble("CRPIX2", 560.018167811613)
        metadata.set("CTYPE1", 'RA---SIN')
        metadata.set("CTYPE2", 'DEC--SIN')

        metadata.setDouble("CD1_1", 5.10808596133527E-05)
        metadata.setDouble("CD1_2", 1.85579539217196E-07)
        metadata.setDouble("CD2_2", -5.10281493481982E-05)
        metadata.setDouble("CD2_1", -8.27440751733828E-07)

        return afwGeom.makeSkyWcs(metadata)

    def testRatio(self):
        """Test the ratio functor where one of the functors should be junk.
        """
        self.dataDict["base_PsfFlux_instFlux"] = np.full(self.nRecords, 1000)
        self.dataDict["base_PsfFlux_instFluxErr"] = np.full(self.nRecords, 100)
        parq = self.simulateMultiParquet(self.dataDict)

        func = Ratio("base_PsfFlux_instFlux", "base_PsfFlux_instFluxErr")

        val = self._funcVal(func, parq)
        self.assertTrue(np.allclose(np.full(self.nRecords, 10),
                                    val.values,
                                    atol=1e-16,
                                    rtol=1e-16))

    def testHtm(self):
        """Test that HtmIndxes are created as expected.
        """
        parq = self.simulateMultiParquet(self.dataDict)
        func = HtmIndex20("coord_ra", "coord_dec")

        val = self._funcVal(func, parq)
        # Test that the HtmIds come out as the ra/dec in dataDict.
        self.assertTrue(np.all(np.equal(
            val.values,
            [14924528684992, 14924528689697, 14924528501716, 14924526434259,
             14924526433879])))


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
