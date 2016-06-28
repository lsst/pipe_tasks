from __future__ import absolute_import, division, print_function
#
# LSST Data Management System
# Copyright 2016 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
from itertools import izip
import os
import unittest

import numpy as np
from astropy.io import fits
from astropy.table import Table

import lsst.utils
from lsst.pipe.tasks.readCatalog import ReadFitsCatalogTask

# If you want to update the FITS table used for this test:
# - modify makeFitsTable to create the table as you want it
# - set SaveFitsTable = True
# - sun the test once to create the new file
# - set SaveFitsTable = False again
SaveFitsTable = False  # construct and save a new FITS table file?
TestDir = os.path.dirname(__file__)
FitsPath = os.path.join(TestDir, "data", "testReadFitsCatalog.fits")


def setup_module(module):
    lsst.utils.tests.init()


def makeFitsTable():
    """Create a fits file containing two tables"""
    hdu0 = fits.PrimaryHDU()

    # table for HDU 1
    cols1 = [
        fits.Column(name='name', format='10A', array=["object 1", "object 2"]),
        fits.Column(name='ra', format='E', array=[10, 5]),
        fits.Column(name='dec', format='E', array=[-5, 45]),
        fits.Column(name='counts', format='J', array=[1000, 2000]),
        fits.Column(name='flux', format='D', array=[1.1, 2.2]),
        fits.Column(name='resolved', format='L', array=[True, False]),
    ]
    hdu1 = fits.BinTableHDU.from_columns(fits.ColDefs(cols1))

    # table for HDU 2,
    cols2 = [
        fits.Column(name='name', format='10A', array=["object 3", "object 4"]),
        fits.Column(name='ra', format='E', array=[16, 3]),
        fits.Column(name='dec', format='E', array=[75, -34]),
        fits.Column(name='resolved', format='L', array=[False, True]),
        fits.Column(name='flux', format='D', array=[10.1, 20.2]),
        fits.Column(name='counts', format='J', array=[15000, 22000]),
        fits.Column(name='other', format='D', array=[11, 12]),
    ]
    hdu2 = fits.BinTableHDU.from_columns(fits.ColDefs(cols2))

    # add an image HDU to test that these are not treated as tables
    hdu3 = fits.ImageHDU(data=np.zeros([5, 5]))

    foo = fits.HDUList([hdu0, hdu1, hdu2, hdu3])
    return foo

if SaveFitsTable:
    print("Warning: writing a new FITS file; to stop this set SaveFitsTable = False")
    fitsTable = makeFitsTable()
    fitsTable.writeto(FitsPath, clobber=True)


class ReadFitsCatalogTaskTestCase(lsst.utils.tests.TestCase):
    """Test ReadFitsCatalogTask, a reader used by IngestIndexedReferenceTask"""
    def setUp(self):
        fitsTable = makeFitsTable()
        self.arr1 = fitsTable[1].data
        self.arr2 = fitsTable[2].data
        self.fitsTable = fitsTable

    def testHDU1DefaultNames(self):
        """Test the data in HDU 1, loading all columns without renaming
        """
        task = ReadFitsCatalogTask()
        table = task.run(FitsPath)
        self.assertTrue(np.array_equal(table, self.arr1))
        self.assertEqual(len(table), 2)

    def testHDU1GivenNames(self):
        """Test the data in HDU 1 with some column renaming

        All columns should be in the same order; those that are renamed should have
        their new name, and the rest should have their original name.
        """
        column_map = {
            "name": "source",
            "ra": "ra_deg",
            "dec": "dec_deg",
        }
        config = ReadFitsCatalogTask.ConfigClass()
        config.column_map = column_map
        self.assertEqual(config.hdu, 1)
        task = ReadFitsCatalogTask(config=config)
        arr = task.run(FitsPath)
        self.assertEqual(len(Table(arr).columns), len(Table(self.arr1).columns))
        for inname, outname in izip(self.arr1.dtype.names, arr.dtype.names):
            des_outname = column_map.get(inname, inname)
            self.assertEqual(outname, des_outname)
            self.assertTrue(np.array_equal(self.arr1[inname], arr[outname]))

    def testHDU2(self):
        """Test reading HDU 2 with original order"""
        config = ReadFitsCatalogTask.ConfigClass()
        config.hdu = 2
        task = ReadFitsCatalogTask(config=config)
        arr = task.run(FitsPath)
        self.assertTrue(np.array_equal(arr, self.arr2))

    def testBadPath(self):
        """Test that an invalid path causes an error"""
        task = ReadFitsCatalogTask()
        badPath = "does/not/exists.garbage"
        with self.assertRaises(IOError):
            task.run(badPath)

    def testBadColumnName(self):
        """Test that non-existent columns in column_map cause an error"""
        config = ReadFitsCatalogTask.ConfigClass()
        for badColNames in (
            ["name", "ra", "dec", "counts", "flux", "resolved", "other"],  # "other" only in hdu 2
            ["name", "ra", "dec", "counts", "flux", "invalidname"],
            ["invalid1"],
        ):
            config.column_map = dict((name, "col %s" % (i,)) for i, name in enumerate(badColNames))
            task = ReadFitsCatalogTask(config=config)
            with self.assertRaises(RuntimeError):
                task.run(FitsPath)

    def testBadHdu(self):
        """Test that non-existent HDUs cause an error"""
        for badHdu in [0, 3, 4]:
            config = ReadFitsCatalogTask.ConfigClass()
            config.hdu = badHdu
            task = ReadFitsCatalogTask(config=config)
            with self.assertRaises(Exception):
                task.run(FitsPath)


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
