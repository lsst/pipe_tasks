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
import numpy as np
import tempfile

from lsst.afw import image as afwImage
from lsst.pipe.tasks import extended_psf
import lsst.utils.tests

np.random.seed(51778)


def make_extended_psf(n_extended_psf=1):
    e_psf_images = [afwImage.MaskedImageF(25, 25) for _ in range(n_extended_psf)]
    for e_psf_im in e_psf_images:
        e_psf_im.image.array += np.random.rand(25, 25)
        e_psf_im.mask.array += np.random.choice(3, size=(25, 25))
    return e_psf_images


class ExtendedPsfTestCase(lsst.utils.tests.TestCase):
    """Test ExtendedPsf.
    """
    def setUp(self):
        self.default_e_psf = make_extended_psf(1)[0]
        self.constant_e_psf = extended_psf.ExtendedPsf(self.default_e_psf)
        self.regions = ["NW", "SW", "E"]
        self.region_detectors = []
        for i in range(3):
            self.det = extended_psf.DetectorsInRegion()
            r0 = 10*i
            r1 = 10*(i+1)
            self.det.detectors = list(range(r0, r1))
            self.region_detectors.append(self.det)
        self.regional_e_psfs = make_extended_psf(3)

    def tearDown(self):
        del self.default_e_psf
        del self.regions
        del self.region_detectors
        del self.det
        del self.regional_e_psfs

    def test_constant_psf(self):
        # When calling a constant ExtendedPsf, the same PSF is returned whether
        # a detector ID is given or not.
        cons_psf0 = self.constant_e_psf()
        cons_psf1 = self.constant_e_psf(detector=11)
        self.assertMaskedImagesAlmostEqual(cons_psf0, self.default_e_psf)
        self.assertMaskedImagesAlmostEqual(cons_psf1, self.default_e_psf)

    def test_regional_psf_addition(self):
        # Start with either an empty instance, or one containing a default
        # extended PSF.
        starts_empty_e_psf = extended_psf.ExtendedPsf()
        with_default_e_psf = extended_psf.ExtendedPsf(self.default_e_psf)
        self.assertEqual(len(starts_empty_e_psf), 0)
        self.assertEqual(len(with_default_e_psf), 1)
        # Add a couple of regional PSFs.
        for j in range(2):
            starts_empty_e_psf.add_regional_extended_psf(self.regional_e_psfs[j], self.regions[j],
                                                         self.region_detectors[j])
            with_default_e_psf.add_regional_extended_psf(self.regional_e_psfs[j], self.regions[j],
                                                         self.region_detectors[j])
        self.assertEqual(len(starts_empty_e_psf), 2)
        self.assertEqual(len(with_default_e_psf), 3)
        # Ensure we recover the correct regional PSF.
        for j in range(2):
            for det in self.region_detectors[j].detectors:
                # Try it by calling the class directly.
                reg_psf0, reg_psf1 = starts_empty_e_psf(det), with_default_e_psf(det)
                self.assertMaskedImagesAlmostEqual(reg_psf0, self.regional_e_psfs[j])
                self.assertMaskedImagesAlmostEqual(reg_psf1, self.regional_e_psfs[j])
                # Try it by passing on a detector number to the
                # get_extended_psf method.
                reg_psf0 = starts_empty_e_psf.get_extended_psf(region_name=det)
                reg_psf1 = with_default_e_psf.get_extended_psf(region_name=det)
                self.assertMaskedImagesAlmostEqual(reg_psf0, self.regional_e_psfs[j])
                self.assertMaskedImagesAlmostEqual(reg_psf1, self.regional_e_psfs[j])
            # Try it by passing on a region name.
            reg_psf0 = starts_empty_e_psf.get_extended_psf(region_name=self.regions[j])
            reg_psf1 = with_default_e_psf.get_extended_psf(region_name=self.regions[j])
            self.assertMaskedImagesAlmostEqual(reg_psf0, self.regional_e_psfs[j])
            self.assertMaskedImagesAlmostEqual(reg_psf1, self.regional_e_psfs[j])
        # Ensure we recover the original default PSF.
        self.assertMaskedImagesAlmostEqual(with_default_e_psf(), self.default_e_psf)

    def test_IO(self):
        # Test IO with a constant extended PSF.
        with tempfile.NamedTemporaryFile() as f:
            self.constant_e_psf.writeFits(f.name)
            read_e_psf = extended_psf.ExtendedPsf.readFits(f.name)
            self.assertMaskedImagesAlmostEqual(self.constant_e_psf(), read_e_psf())
        # Test IO with per-region extended PSFs (with default).
        per_region_e_psf0 = extended_psf.ExtendedPsf(self.default_e_psf)
        for j in range(3):
            per_region_e_psf0.add_regional_extended_psf(self.regional_e_psfs[j], self.regions[j],
                                                        self.region_detectors[j])
        with tempfile.NamedTemporaryFile() as f:
            per_region_e_psf0.writeFits(f.name)
            read_e_psf0 = extended_psf.ExtendedPsf.readFits(f.name)
            self.assertEqual(per_region_e_psf0.detectors_focal_plane_regions,
                             read_e_psf0.detectors_focal_plane_regions)
            # Check default extended PSF.
            self.assertMaskedImagesAlmostEqual(per_region_e_psf0(), read_e_psf0())
            # And per-region extended PSFs.
            for j in range(3):
                for det in self.region_detectors[j].detectors:
                    reg_psf0, read_reg_psf0 = per_region_e_psf0(det), read_e_psf0(det)
                    self.assertMaskedImagesAlmostEqual(reg_psf0, read_reg_psf0)
        # Test IO with a single per-region extended PSF.
        per_region_e_psf1 = extended_psf.ExtendedPsf()
        per_region_e_psf1.add_regional_extended_psf(self.regional_e_psfs[1], self.regions[1],
                                                    self.region_detectors[1])
        with tempfile.NamedTemporaryFile() as f:
            per_region_e_psf1.writeFits(f.name)
            read_e_psf1 = extended_psf.ExtendedPsf.readFits(f.name)
            self.assertEqual(per_region_e_psf0.detectors_focal_plane_regions,
                             read_e_psf0.detectors_focal_plane_regions)
            for det in self.region_detectors[1].detectors:
                reg_psf1, read_reg_psf1 = per_region_e_psf1(det), read_e_psf1(det)
                self.assertMaskedImagesAlmostEqual(reg_psf1, read_reg_psf1)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
