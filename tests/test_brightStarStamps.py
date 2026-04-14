# This file is part of meas_algorithms.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import unittest
from typing import cast

import astropy.units as u
import lsst.images.tests
import lsst.utils.tests
import numpy as np
from lsst.images import Image, Mask, MaskedImage, MaskSchema
from lsst.pipe.tasks.brightStars import BrightStarStamp, BrightStarStampInfo, BrightStarStamps


class BrightStarStampsTestCase(lsst.utils.tests.TestCase):
    """Test BrightStarStamps."""

    def setUp(self):
        rng = np.random.Generator(np.random.MT19937(seed=5))
        stamp_size = (25, 25)

        # Generate simulated bright star stamps
        bright_star_stamps = []
        self.stamp_infos = []
        mask_schema = MaskSchema([])

        for i in range(3):
            stamp = MaskedImage(
                image=Image(rng.random(stamp_size).astype(np.float32)),
                mask=Mask(0, schema=mask_schema, shape=stamp_size),
                variance=Image(rng.random(stamp_size).astype(np.float32)),
            )

            stamp_info = BrightStarStampInfo(
                visit=100 + i,
                detector=200 + i,
                ref_id=1000 + i,
                ref_mag=10.0 + i,
                position_x=float(rng.random()),
                position_y=float(rng.random()),
                focal_plane_radius=float(rng.random()) * u.mm,
                focal_plane_angle=float(rng.random()) * u.rad,
            )
            self.stamp_infos.append(stamp_info)

            # Build a normalized 2-D Gaussian kernel image directly.
            yy, xx = np.mgrid[: stamp_size[0], : stamp_size[1]]
            cy = (stamp_size[0] - 1) / 2.0
            cx = (stamp_size[1] - 1) / 2.0
            sigma = 1.5
            kernel_array = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))
            kernel_array /= np.sum(kernel_array)
            psf_kernel_image = Image(kernel_array.astype(np.float64))

            bright_star_stamps.append(
                BrightStarStamp(
                    image=stamp.image,
                    mask=stamp.mask,
                    variance=stamp.variance,
                    psf_kernel_image=psf_kernel_image,
                    stamp_info=stamp_info,
                )
            )

        self.global_metadata = {"TEST_KEY": "TEST VALUE"}
        self.bright_star_stamps = BrightStarStamps(bright_star_stamps, self.global_metadata)

    def tearDown(self):
        del self.bright_star_stamps
        del self.stamp_infos
        del self.global_metadata

    def testBrightStarStamps(self):
        """Test that BrightStarStamps can be serialized and deserialized."""

        with lsst.images.tests.RoundtripFits(
            self, self.bright_star_stamps, storage_class="BrightStarStamps"
        ) as roundtrip:
            pass
        bright_star_stamps = roundtrip.result

        global_metadata = bright_star_stamps.metadata
        self.assertEqual(self.global_metadata["TEST_KEY"], global_metadata["TEST_KEY"])
        self.assertEqual(len(self.bright_star_stamps), len(bright_star_stamps))
        for input_info, input_stamp, output_stamp in zip(
            self.stamp_infos, self.bright_star_stamps, bright_star_stamps
        ):
            self.assertEqual(input_stamp.metadata, {})
            self.assertEqual(output_stamp.metadata, {})

            output_info = output_stamp.stamp_info
            self.assertEqual(input_info.visit, output_info.visit)
            self.assertEqual(input_info.detector, output_info.detector)
            self.assertEqual(input_info.ref_id, output_info.ref_id)
            self.assertAlmostEqual(input_info.ref_mag, output_info.ref_mag, places=10)
            self.assertAlmostEqual(input_info.position_x, output_info.position_x, places=10)
            self.assertAlmostEqual(input_info.position_y, output_info.position_y, places=10)

            self.assertIsNotNone(input_info.focal_plane_radius)
            self.assertIsNotNone(output_info.focal_plane_radius)
            input_radius = cast(u.Quantity, input_info.focal_plane_radius)
            output_radius = cast(u.Quantity, output_info.focal_plane_radius)
            self.assertAlmostEqual(input_radius.to_value(u.mm), output_radius.to_value(u.mm), places=10)

            self.assertIsNotNone(input_info.focal_plane_angle)
            self.assertIsNotNone(output_info.focal_plane_angle)
            input_angle = cast(u.Quantity, input_info.focal_plane_angle)
            output_angle = cast(u.Quantity, output_info.focal_plane_angle)
            self.assertAlmostEqual(input_angle.to_value(u.rad), output_angle.to_value(u.rad), places=10)

            np.testing.assert_allclose(
                input_stamp.image.array, output_stamp.image.array, rtol=0.0, atol=1e-10
            )
            np.testing.assert_array_equal(input_stamp.mask.array, output_stamp.mask.array)
            np.testing.assert_allclose(
                input_stamp.variance.array, output_stamp.variance.array, rtol=0.0, atol=1e-10
            )
            np.testing.assert_allclose(
                input_stamp.psf_kernel_image.array,
                output_stamp.psf_kernel_image.array,
                rtol=0.0,
                atol=1e-10,
            )


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
