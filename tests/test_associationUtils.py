# This file is part of pipe_tasks.

# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import healpy as hp
import numpy as np
import unittest

from lsst.pipe.tasks.associationUtils import query_disc
import lsst.utils.tests


class TestAssociationUtils(lsst.utils.tests.TestCase):

    def test_queryDisc(self):
        """Test that doing a disc query of healpy using our wrapper works as
        expected.

        Most other calculations are very simple and not worth testing.
        """
        # Find nearest pixel to r=45, dec=45.
        nside = 128
        testRa = 225
        testDec = 45
        centerPixNumber = hp.ang2pix(nside, testRa, testDec, lonlat=True)
        ra, dec = hp.pix2ang(nside, centerPixNumber, lonlat=True)
        # Test that only one pixel is found.
        pixelReturn = query_disc(nside, ra, dec, np.radians(0.1))
        self.assertEqual(len(pixelReturn), 1)
        self.assertTrue(centerPixNumber in pixelReturn)
        # Test that nearby pixels are returned.
        pixelReturn = query_disc(nside, ra, dec, np.radians(1))
        self.assertEqual(len(pixelReturn), 17)
        self.assertTrue(centerPixNumber in pixelReturn)
        # Test excluding the central pixel
        pixelReturn = query_disc(
            nside, ra, dec, np.radians(1), np.radians(0.01))
        self.assertEqual(len(pixelReturn), 16)
        self.assertFalse(centerPixNumber in pixelReturn)


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
