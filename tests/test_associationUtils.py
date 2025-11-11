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

import hpgeom as hpg
import numpy as np
import unittest

from lsst.pipe.tasks.associationUtils import query_disc, obj_id_to_ss_object_id, ss_object_id_to_obj_id
import lsst.utils.tests


class TestAssociationUtils(lsst.utils.tests.TestCase):

    def test_queryDisc(self):
        """Test that doing a circle query of hpgeom using our wrapper works as
        expected.

        Most other calculations are very simple and not worth testing.
        """
        # Find nearest pixel to r=45, dec=45.
        nside = 128
        testRa = 225
        testDec = 45
        centerPixNumber = hpg.angle_to_pixel(nside, testRa, testDec, nest=False)
        ra, dec = hpg.pixel_to_angle(nside, centerPixNumber, nest=False)
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

    def test_conversions_between_obj_id_and_ss_object_id(self):
        """Convert between ssObjectIDs and MPC packed designations
        """
        allowed_strings = ['J95X00A', 'J95X01L', 'J95F13B', 'J98SA8Q', 'J98SC7V', 'J98SG2S'] \
            + ['K99AJ3Z', 'K08Aa0A', 'K07Tf8A', 'PLS2040', 'T1S3138', 'T2S1010', 'T3S4101'] \
            + ['       ', 'PJ48Q010', 'AAAAAAAA']
        for allowed_string in allowed_strings:
            returned_string = ss_object_id_to_obj_id(obj_id_to_ss_object_id(allowed_string))
            self.assertEqual(allowed_string, returned_string)

    def test_invalid_conversions_between_obj_id_and_ss_object_id(self):
        """Convert between ssObjectIDs and MPC packed designations
        """
        disallowed_strings = [''] + [ch for ch in 'ABCDEFGHIJKMNOPQRSTUVWXYZ0123456789 -'] \
            + ['A' * i for i in range(2, 7)] + ['Z' * i for i in range(2, 7)] \
            + ['Ā', '🔭', 'A' * 9, ' ' * 9, 'A' * 128]
        disallowed_ssObjectIDs = [-1, 1 << 64, 1 << 64 + 1, 2 << 65]
        for disallowed_ssObjectID in disallowed_ssObjectIDs:
            with self.assertRaises(ValueError):
                ss_object_id_to_obj_id(disallowed_ssObjectID)
        for disallowed_string in disallowed_strings:
            with self.assertRaises(ValueError):
                obj_id_to_ss_object_id(disallowed_string)


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
