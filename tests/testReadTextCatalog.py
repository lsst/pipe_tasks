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

import lsst.utils
from lsst.pipe.tasks.readCatalog import ReadTextCatalogTask

# If you want to update the FITS table used for this test:
# - modify makeFitsTable to create the table as you want it
# - set SaveTextCatalog = True
# - sun the test once to create the new file
# - set SaveTextCatalog = False again
SaveTextCatalog = False  # construct and save a new text table file?
TestDir = os.path.dirname(__file__)
TextPath = os.path.join(TestDir, "data", "testReadTextCatalog.csv")


def setup_module(module):
    lsst.utils.tests.init()


def makeCatalog():
    """Create an object catalog as a numpy structured array

    dtypes are chosen to match how the data is read back in, for ease in testing
    """
    dtype = [("name", "a8"), ("ra", "float64"), ("dec", "float64"),
             ("counts", "int64"), ("flux", "float64"), ("resolved", "int64")]
    data = [
        ("object 1", -5, 10, 1000, 1.1, True),
        ("object 2", 45,  5, 2000, 1.2, False),
    ]
    return np.array(data, dtype=dtype)

if SaveTextCatalog:
    print("Warning: writing a new text catalog file; to stop this set SaveTextCatalog = False")
    arr = makeCatalog()
    with open(TextPath, "w") as f:
        f.write(", ".join(arr.dtype.names))
        f.write("\n")
        for row in arr:
            f.write(", ".join(str(val) for val in row))
            f.write("\n")


class ReadTextCatalogTaskTestCase(lsst.utils.tests.TestCase):
    """Test ReadTextCatalogTask, a reader used by IngestIndexedReferenceTask"""
    def setUp(self):
        self.arr = makeCatalog()

    def testDefaultNames(self):
        """Test reading without renaming
        """
        task = ReadTextCatalogTask()
        arr = task.run(TextPath)
        self.assertTrue(np.array_equal(arr, self.arr))
        self.assertEqual(len(arr), 2)

    def testGivenNames(self):
        """Test reading with column names in the config
        """
        colnames = ("id", "ra_deg", "dec_deg", "total_counts", "total_flux", "is_resolved")
        config = ReadTextCatalogTask.ConfigClass()
        config.colnames = colnames
        config.header_lines = 1
        task = ReadTextCatalogTask(config=config)
        arr = task.run(TextPath)
        self.assertEqual(arr.dtype.names, colnames)
        self.assertEqual(len(arr), 2)
        for inname, outname in izip(self.arr.dtype.names, colnames):
            self.assertTrue(np.array_equal(self.arr[inname], arr[outname]))

    def testBadPath(self):
        """Test that an invalid path causes an error"""
        task = ReadTextCatalogTask()
        badPath = "does/not/exists.garbage"
        with self.assertRaises(IOError):
            task.run(badPath)

    def testTooFewColumnNames(self):
        """Test that too few names in config.colnames causes an error"""
        config = ReadTextCatalogTask.ConfigClass()
        for badColNames in (
            ["name", "ra", "dec", "counts", "flux"],
            ["name"],
        ):
            config.colnames = badColNames
            config.header_lines = 1
            task = ReadTextCatalogTask(config=config)
            with self.assertRaises(ValueError):
                task.run(TextPath)


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
