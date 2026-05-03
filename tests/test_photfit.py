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
import lsst.utils.tests
from lsst.pipe.tasks.ssp.photfit import HG12_model, fitHG12

# Independently calculated test vectors for H=17.30, G12=0.42
# 15 observations spanning phase angles 0.8-60 degrees
H_TRUE = 17.30
G12_TRUE = 0.42

PHASE_ANGLE = np.array([
    0.8, 1.5, 2.3, 3.5, 5.0,
    7.5, 10.0, 13.0, 16.0, 20.0,
    25.0, 30.0, 38.0, 48.0, 60.0,
])
RDIST = np.array([
    2.33, 2.31, 2.29, 2.27, 2.25,
    2.23, 2.21, 2.20, 2.18, 2.16,
    2.14, 2.12, 2.10, 2.08, 2.06,
])
TDIST = np.array([
    1.42, 1.38, 1.34, 1.30, 1.26,
    1.23, 1.20, 1.18, 1.16, 1.15,
    1.14, 1.15, 1.18, 1.24, 1.33,
])
MAG_EXPECTED = np.array([
    20.03700074, 20.01538407, 19.99152984, 19.98201741, 19.97201618,
    20.00200255, 20.01813960, 20.07342182, 20.11401822, 20.19891977,
    20.30457878, 20.43928577, 20.69068120, 21.05570108, 21.53526150,
])
MAG_SIGMA = np.full_like(MAG_EXPECTED, 0.03)


class TestHG12Model(lsst.utils.tests.TestCase):

    def testHG12ModelEvaluation(self):
        """Verify HG12_model reproduces independently calculated
        apparent magnitudes.
        """
        phase_rad = np.deg2rad(PHASE_ANGLE)
        reduced_mag = HG12_model(phase_rad, [H_TRUE, G12_TRUE])
        dmag = 5.0 * np.log10(TDIST * RDIST)
        apparent = reduced_mag + dmag

        np.testing.assert_allclose(
            apparent, MAG_EXPECTED, atol=1e-6,
            err_msg="HG12_model does not reproduce expected mags",
        )

    def testFitHG12FreeG12(self):
        """Fit noiseless data with free G12 and verify parameter
        recovery.
        """
        result = fitHG12(
            MAG_EXPECTED, MAG_SIGMA, PHASE_ANGLE, TDIST, RDIST,
        )
        H_fit, G12_fit, H_err, G_err, HG_cov, chi2dof, nobs = result

        self.assertAlmostEqual(H_fit, H_TRUE, places=4)
        self.assertAlmostEqual(G12_fit, G12_TRUE, places=4)
        self.assertTrue(np.isfinite(H_err) and H_err > 0)
        self.assertTrue(np.isfinite(G_err) and G_err > 0)
        self.assertLess(chi2dof, 1e-6)
        self.assertEqual(nobs, 15)

    def testFitHG12FixedG12(self):
        """Fit with fixedG12 matching the true value."""
        result = fitHG12(
            MAG_EXPECTED, MAG_SIGMA, PHASE_ANGLE, TDIST, RDIST,
            fixedG12=G12_TRUE,
        )
        H_fit, G12_fit, H_err, G_err, HG_cov, chi2dof, nobs = result

        self.assertAlmostEqual(H_fit, H_TRUE, places=4)
        self.assertEqual(G12_fit, G12_TRUE)
        self.assertTrue(np.isfinite(H_err) and H_err > 0)
        self.assertTrue(np.isnan(G_err))
        self.assertTrue(np.isnan(HG_cov))
        self.assertLess(chi2dof, 1e-6)
        self.assertEqual(nobs, 15)

    def testFitHG12FixedG12Wrong(self):
        """Fit with wrong fixedG12 — should succeed but with worse
        chi2 and shifted H.
        """
        result_correct = fitHG12(
            MAG_EXPECTED, MAG_SIGMA, PHASE_ANGLE, TDIST, RDIST,
            fixedG12=G12_TRUE,
        )
        result_wrong = fitHG12(
            MAG_EXPECTED, MAG_SIGMA, PHASE_ANGLE, TDIST, RDIST,
            fixedG12=0.8,
        )
        chi2_correct = result_correct.chi2dof
        H_wrong = result_wrong.H
        chi2_wrong = result_wrong.chi2dof

        # Wrong G12 should produce a noticeably different H
        self.assertGreater(abs(H_wrong - H_TRUE), 0.005)
        # Wrong G12 should produce larger chi2
        self.assertGreater(chi2_wrong, chi2_correct)

    def testMagSigmaFloor(self):
        """Verify magSigmaFloor reduces chi2 and increases H_err."""
        rng = np.random.RandomState(42)
        noisy_mag = MAG_EXPECTED + rng.normal(0, 0.03, len(MAG_EXPECTED))

        result_no_floor = fitHG12(
            noisy_mag, MAG_SIGMA, PHASE_ANGLE, TDIST, RDIST,
            fixedG12=G12_TRUE,
        )
        result_with_floor = fitHG12(
            noisy_mag, MAG_SIGMA, PHASE_ANGLE, TDIST, RDIST,
            fixedG12=G12_TRUE, magSigmaFloor=0.05,
        )
        chi2_no_floor = result_no_floor.chi2dof
        chi2_with_floor = result_with_floor.chi2dof
        herr_no_floor = result_no_floor.H_err
        herr_with_floor = result_with_floor.H_err

        # Floor inflates errors → chi2 should decrease
        self.assertLess(chi2_with_floor, chi2_no_floor)
        # H_err should increase (or at least not decrease) with floor
        self.assertGreaterEqual(herr_with_floor, herr_no_floor * 0.99)

    def testFitFailureSingleObs(self):
        """Single observation should fail gracefully."""
        # Free fit with 1 obs and 2 params is underdetermined
        result = fitHG12(
            MAG_EXPECTED[:1], MAG_SIGMA[:1], PHASE_ANGLE[:1],
            TDIST[:1], RDIST[:1], fixedG12=0.5,
        )
        # With fixedG12, 1 obs and 1 param gives 0 DOF — fit may
        # succeed but let's just check it doesn't crash.
        self.assertEqual(len(result), 7)

        # With 0 observations, should return NaN
        result = fitHG12(
            np.array([]), np.array([]), np.array([]),
            np.array([]), np.array([]),
        )
        H, G, H_err, G_err, HG_cov, chi2dof, nobs = result
        self.assertTrue(np.isnan(H))
        self.assertEqual(nobs, 0)

    def testFitWithNaNMagnitudes(self):
        """NaN magnitudes (from negative fluxes) should be filtered
        out, not crash the fit.
        """
        # Insert NaN values into the test data
        mag = MAG_EXPECTED.copy()
        sigma = MAG_SIGMA.copy()
        phase = PHASE_ANGLE.copy()
        tdist = TDIST.copy()
        rdist = RDIST.copy()

        # Inject 3 NaN magnitudes
        mag_nan = np.concatenate([mag, [np.nan, np.nan, np.nan]])
        sig_nan = np.concatenate([sigma, [0.03, 0.03, 0.03]])
        pa_nan = np.concatenate([phase, [10.0, 20.0, 30.0]])
        td_nan = np.concatenate([tdist, [1.3, 1.3, 1.3]])
        rd_nan = np.concatenate([rdist, [2.2, 2.2, 2.2]])

        result = fitHG12(
            mag_nan, sig_nan, pa_nan, td_nan, rd_nan,
            fixedG12=G12_TRUE,
        )
        # Should succeed using the 15 valid observations
        self.assertAlmostEqual(result.H, H_TRUE, places=4)
        self.assertEqual(result.nobs, 15)

    def testFitAllNaN(self):
        """All-NaN input should return failure, not crash."""
        result = fitHG12(
            np.array([np.nan, np.nan]),
            np.array([0.03, 0.03]),
            np.array([10.0, 20.0]),
            np.array([1.3, 1.3]),
            np.array([2.2, 2.2]),
        )
        self.assertTrue(np.isnan(result.H))
        self.assertEqual(result.nobs, 0)

    def testDistanceCorrection(self):
        """Same (H, G12, phase) at different distances should yield
        the same H.
        """
        # Create a second set at 2x heliocentric distance
        rdist2 = RDIST * 2.0
        phase_rad = np.deg2rad(PHASE_ANGLE)
        reduced_mag = HG12_model(phase_rad, [H_TRUE, G12_TRUE])
        mag2 = reduced_mag + 5.0 * np.log10(TDIST * rdist2)
        sigma2 = np.full_like(mag2, 0.03)

        result = fitHG12(
            mag2, sigma2, PHASE_ANGLE, TDIST, rdist2,
        )
        self.assertAlmostEqual(result.H, H_TRUE, places=4)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
