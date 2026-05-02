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
import lsst.utils.tests
from lsst.pipe.tasks.ssp.moid import MOIDSolver, earth_orbit

# J2000.0 epoch as MJD
MJD_J2000 = 51544.5


class TestEarthOrbit(lsst.utils.tests.TestCase):

    def testEarthOrbitJ2000(self):
        """Verify J2000 values match JPL coefficients."""
        el = earth_orbit(MJD_J2000)
        self.assertAlmostEqual(el.a_AU, 1.00000261, places=8)
        self.assertAlmostEqual(el.e, 0.01671123, places=8)
        self.assertAlmostEqual(el.omega_deg, 102.93768193, places=5)
        self.assertEqual(el.Omega_deg, 0.0)
        self.assertAlmostEqual(el.inc_deg, 0.00005, places=6)

    def testEarthOrbitSecularEvolution(self):
        """Verify elements evolve at the expected rate."""
        el_j2000 = earth_orbit(MJD_J2000)
        # One century later
        el_later = earth_orbit(MJD_J2000 + 36525.0)

        # omega should increase by ~0.323 deg/century
        domega = el_later.omega_deg - el_j2000.omega_deg
        self.assertAlmostEqual(domega, 0.32327364, places=5)

        # eccentricity should decrease by ~0.0000439/century
        de = el_later.e - el_j2000.e
        self.assertAlmostEqual(de, -0.00004392, places=8)

        # semi-major axis should increase slightly
        da = el_later.a_AU - el_j2000.a_AU
        self.assertAlmostEqual(da, 0.00000562, places=8)

    def testEarthOrbitEclipticFrame(self):
        """Verify Omega=0 and inc~0 at all epochs."""
        for mjd in [40000.0, MJD_J2000, 60000.0, 80000.0]:
            el = earth_orbit(mjd)
            self.assertEqual(el.Omega_deg, 0.0)
            self.assertAlmostEqual(el.inc_deg, 0.00005, places=6)


class TestMOIDSolver(lsst.utils.tests.TestCase):

    def setUp(self):
        self.solver = MOIDSolver()

    def testIdenticalOrbits(self):
        """Two identical circular orbits should have MOID = 0."""
        el = (1.0, 0.0, 0.0, 0.0, 0.0)
        result = self.solver.compute(el, el)
        self.assertAlmostEqual(result.MOID_AU, 0.0, places=6)

    def testCoplanarCircular(self):
        """Coplanar circular orbits a=1, a=2 should have MOID=1.0."""
        el1 = (1.0, 0.0, 0.0, 0.0, 0.0)
        el2 = (2.0, 0.0, 0.0, 0.0, 0.0)
        result = self.solver.compute(el1, el2)
        self.assertAlmostEqual(result.MOID_AU, 1.0, places=4)

    def testCoplanarCrossing(self):
        """Eccentric orbit crossing a circular one should have
        MOID near zero.
        """
        # a=1 e=0 (circular) vs a=1.5 e=0.5
        # (perihelion=0.75, aphelion=2.25 — crosses the r=1 circle)
        el1 = (1.0, 0.0, 0.0, 0.0, 0.0)
        el2 = (1.5, 0.5, 0.0, 0.0, 0.0)
        result = self.solver.compute(el1, el2)
        self.assertLess(result.MOID_AU, 0.001)

    def testPerpendicularOrbit(self):
        """90-degree inclination orbit should have a finite,
        positive MOID.
        """
        earth = earth_orbit(60000.0)
        perp = (1.5, 0.3, 90.0, 0.0, 0.0)
        result = self.solver.compute(earth, perp)
        self.assertGreater(result.MOID_AU, 0.0)
        self.assertLess(result.MOID_AU, 2.0)

    def testKnownNEA(self):
        """Apophis-like elements should give MOID ~ 0.0002 AU."""
        # Approximate elements for (99942) Apophis
        apophis = (0.9224, 0.1914, 3.339, 204.43, 126.39)
        earth = earth_orbit(60000.0)
        result = self.solver.compute(earth, apophis)
        # MPC gives ~0.00025 AU; we allow some tolerance for
        # element epoch differences
        self.assertLess(result.MOID_AU, 0.005)
        self.assertGreater(result.MOID_AU, 0.0)

    def testDeltaVPositive(self):
        """DeltaV should always be positive."""
        earth = earth_orbit(60000.0)
        for el in [
            (1.5, 0.3, 10.0, 30.0, 45.0),
            (2.5, 0.1, 5.0, 120.0, 200.0),
            (0.9224, 0.1914, 3.339, 204.43, 126.39),
        ]:
            result = self.solver.compute(earth, el)
            self.assertGreater(result.DeltaV_kms, 0.0)

    def testResultFields(self):
        """MOIDResult should have all expected fields."""
        el1 = (1.0, 0.1, 5.0, 30.0, 45.0)
        el2 = (1.5, 0.2, 15.0, 60.0, 10.0)
        result = self.solver.compute(el1, el2)

        self.assertTrue(hasattr(result, 'MOID_AU'))
        self.assertTrue(hasattr(result, 'DeltaV_kms'))
        self.assertTrue(hasattr(result, 'EclipticLongitude_deg'))
        self.assertTrue(hasattr(result, 'TrueAnomaly1_deg'))
        self.assertTrue(hasattr(result, 'TrueAnomaly2_deg'))

        # Angles should be in [0, 360)
        self.assertGreaterEqual(result.EclipticLongitude_deg, 0.0)
        self.assertLess(result.EclipticLongitude_deg, 360.0)
        self.assertGreaterEqual(result.TrueAnomaly1_deg, 0.0)
        self.assertLess(result.TrueAnomaly1_deg, 360.0)
        self.assertGreaterEqual(result.TrueAnomaly2_deg, 0.0)
        self.assertLess(result.TrueAnomaly2_deg, 360.0)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
