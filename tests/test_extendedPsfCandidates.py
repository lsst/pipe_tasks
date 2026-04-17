# This file is part of pipe_tasks.
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
from lsst.pipe.tasks.extendedPsf import ExtendedPsfCandidate, ExtendedPsfCandidateInfo, ExtendedPsfCandidates


class ExtendedPsfCandidatesTestCase(lsst.utils.tests.TestCase):
    """Test ExtendedPsfCandidates."""

    def setUp(self):
        rng = np.random.Generator(np.random.MT19937(seed=5))
        cutout_size = (25, 25)

        # Generate simulated stars
        extended_psf_candidates = []
        self.star_infos = []
        mask_schema = MaskSchema([])

        for i in range(3):
            candidate_masked_image = MaskedImage(
                image=Image(rng.random(cutout_size).astype(np.float32)),
                mask=Mask(0, schema=mask_schema, shape=cutout_size),
                variance=Image(rng.random(cutout_size).astype(np.float32)),
            )

            star_info = ExtendedPsfCandidateInfo(
                visit=100 + i,
                detector=200 + i,
                ref_id=1000 + i,
                ref_mag=10.0 + i,
                position_x=float(rng.random()),
                position_y=float(rng.random()),
                focal_plane_radius=float(rng.random()) * u.mm,
                focal_plane_angle=float(rng.random()) * u.rad,
            )
            self.star_infos.append(star_info)

            # Build a normalized 2-D Gaussian kernel image directly.
            yy, xx = np.mgrid[: cutout_size[0], : cutout_size[1]]
            cy = (cutout_size[0] - 1) / 2.0
            cx = (cutout_size[1] - 1) / 2.0
            sigma = 1.5
            kernel_array = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))
            kernel_array /= np.sum(kernel_array)
            psf_kernel_image = Image(kernel_array.astype(np.float64))

            extended_psf_candidates.append(
                ExtendedPsfCandidate(
                    image=candidate_masked_image.image,
                    mask=candidate_masked_image.mask,
                    variance=candidate_masked_image.variance,
                    psf_kernel_image=psf_kernel_image,
                    star_info=star_info,
                )
            )

        self.global_metadata = {"TEST_KEY": "TEST VALUE"}
        self.extended_psf_candidates = ExtendedPsfCandidates(extended_psf_candidates, self.global_metadata)

    def tearDown(self):
        del self.extended_psf_candidates
        del self.star_infos
        del self.global_metadata

    def testExtendedPsfCandidates(self):
        """Test that ExtendedPsfCandidates can be serialized/deserialized."""

        with lsst.images.tests.RoundtripFits(
            self, self.extended_psf_candidates, storage_class="ExtendedPsfCandidates"
        ) as roundtrip:
            pass
        extended_psf_candidates = roundtrip.result

        global_metadata = extended_psf_candidates.metadata
        self.assertEqual(self.global_metadata["TEST_KEY"], global_metadata["TEST_KEY"])
        self.assertEqual(len(self.extended_psf_candidates), len(extended_psf_candidates))
        for input_info, input_candidate, output_candidate in zip(
            self.star_infos, self.extended_psf_candidates, extended_psf_candidates
        ):
            self.assertEqual(input_candidate.metadata, {})
            self.assertEqual(output_candidate.metadata, {})

            output_info = output_candidate.star_info
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
                input_candidate.image.array, output_candidate.image.array, rtol=0.0, atol=1e-10
            )
            np.testing.assert_array_equal(input_candidate.mask.array, output_candidate.mask.array)
            np.testing.assert_allclose(
                input_candidate.variance.array, output_candidate.variance.array, rtol=0.0, atol=1e-10
            )
            np.testing.assert_allclose(
                input_candidate.psf_kernel_image.array,
                output_candidate.psf_kernel_image.array,
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
