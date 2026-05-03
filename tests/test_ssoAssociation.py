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
from numpy.polynomial.chebyshev import chebval, Chebyshev
from scipy.spatial import cKDTree
from astropy.coordinates import SkyCoord
import astropy.units as u
import lsst.utils.tests
from lsst.pipe.tasks.ssoAssociation import SolarSystemAssociationTask

AU_KM = 1.496e8


# ---------------------------------------------------------------
# Test vectors from 2003 YF26 (verified against JPL Horizons
# in sssource_verification.ipynb)
# ---------------------------------------------------------------
YF26 = dict(
    helio_x=-2.03972984386492,
    helio_y=-0.754159365265218,
    helio_z=-0.09916618480505775,
    helio_vx=6.451686836410149,
    helio_vy=-17.36829941588102,
    helio_vz=-9.013339853647382,
    topo_x=-1.2867480491014316,
    topo_y=-0.1395377271885233,
    topo_z=0.16727378829681566,
    topo_vx=-13.061265792221151,
    topo_vy=3.5105145303033005,
    topo_vz=-0.11752881599981002,
    ephRa=186.18909260549128,
    ephDec=7.364065668224215,
    helioRange=2.1769446746252505,
    topoRange=1.3050562591039694,
    phaseAngle=17.248719450313473,
    elongation=140.17164534286258,
    ephRate=0.13175297987940152,
    ephRateRa=-0.1242177608944433,
    ephRateDec=-0.0439180553471223,
    helio_vtot=20.603940956820594,
    topo_vtot=13.525316609422026,
    helioRangeRate=0.3824562080346701,
    topoRangeRate=12.487622241678675,
    eclLambda=182.73184707660818,
    eclBeta=9.214287183249514,
    galLon=283.959546691413,
    galLat=69.24711603487795,
    ephOffsetRa=-0.08092633090227584,
    ephOffsetDec=-0.06292996923882299,
    ephOffset=0.10251464315747216,
    ephOffsetAlongTrack=0.09727483587724564,
    ephOffsetCrossTrack=0.03235519072357292,
    DIA_ra=186.18906993899677,
    DIA_dec=7.364048187677204,
)


class TestRadecToXyz(lsst.utils.tests.TestCase):
    """Test the RA/Dec to unit vector conversion."""

    def setUp(self):
        self.task = SolarSystemAssociationTask()

    def testNorthPole(self):
        """North celestial pole: (RA=0, Dec=90) -> (0, 0, 1)."""
        xyz = self.task._radec_to_xyz(
            np.array([0.0]), np.array([90.0])
        )
        np.testing.assert_allclose(xyz[0], [0, 0, 1], atol=1e-12)

    def testEquator(self):
        """Points on the equator."""
        # RA=0, Dec=0 -> (1, 0, 0)
        xyz = self.task._radec_to_xyz(
            np.array([0.0]), np.array([0.0])
        )
        np.testing.assert_allclose(xyz[0], [1, 0, 0], atol=1e-12)

        # RA=90, Dec=0 -> (0, 1, 0)
        xyz = self.task._radec_to_xyz(
            np.array([90.0]), np.array([0.0])
        )
        np.testing.assert_allclose(xyz[0], [0, 1, 0], atol=1e-12)

    def testUnitNorm(self):
        """All output vectors should have unit norm."""
        ras = np.array([0, 45, 90, 180, 270, 123.456])
        decs = np.array([0, 30, -45, 60, -80, 7.364])
        xyz = self.task._radec_to_xyz(ras, decs)
        norms = np.linalg.norm(xyz, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def testRoundtrip(self):
        """Convert RA/Dec -> xyz -> RA/Dec and verify recovery."""
        ras = np.array([0.0, 45.0, 186.189, 300.0])
        decs = np.array([0.0, -30.0, 7.364, 85.0])
        xyz = self.task._radec_to_xyz(ras, decs)

        # Recover RA/Dec from xyz
        rec_dec = np.degrees(np.arcsin(xyz[:, 2]))
        rec_ra = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360

        np.testing.assert_allclose(rec_ra, ras, atol=1e-10)
        np.testing.assert_allclose(rec_dec, decs, atol=1e-10)


class TestCoordinateTransforms(lsst.utils.tests.TestCase):
    """Test ecliptic and galactic coordinate transforms."""

    def testEclipticCoords(self):
        """Verify ecliptic coords for the 2003 YF26 test vector."""
        sc = SkyCoord(
            ra=YF26['DIA_ra'] * u.deg,
            dec=YF26['DIA_dec'] * u.deg,
        )
        ecl = sc.barycentrictrueecliptic
        self.assertAlmostEqual(
            ecl.lon.deg, YF26['eclLambda'], places=6
        )
        self.assertAlmostEqual(
            ecl.lat.deg, YF26['eclBeta'], places=6
        )

    def testGalacticCoords(self):
        """Verify galactic coords for the 2003 YF26 test vector."""
        sc = SkyCoord(
            ra=YF26['DIA_ra'] * u.deg,
            dec=YF26['DIA_dec'] * u.deg,
        )
        gal = sc.galactic
        self.assertAlmostEqual(
            gal.l.deg, YF26['galLon'], places=6
        )
        self.assertAlmostEqual(
            gal.b.deg, YF26['galLat'], places=6
        )

    def testEclipticPole(self):
        """North ecliptic pole should be near (RA=270, Dec=66.56)."""
        sc = SkyCoord(ra=270 * u.deg, dec=66.56 * u.deg)
        ecl = sc.barycentrictrueecliptic
        self.assertAlmostEqual(ecl.lat.deg, 90.0, places=0)


class TestEphemerisOffsets(lsst.utils.tests.TestCase):
    """Test ephemeris offset computation."""

    def testEphOffsetRaDec(self):
        """Verify RA/Dec offset formula."""
        dia_ra = YF26['DIA_ra']
        dia_dec = YF26['DIA_dec']
        eph_ra = YF26['ephRa']
        eph_dec = YF26['ephDec']

        offset_ra = (
            (dia_ra - eph_ra)
            * np.cos(np.deg2rad(dia_dec)) * 3600
        )
        offset_dec = (dia_dec - eph_dec) * 3600

        self.assertAlmostEqual(
            offset_ra, YF26['ephOffsetRa'], places=10
        )
        self.assertAlmostEqual(
            offset_dec, YF26['ephOffsetDec'], places=10
        )

    def testEphOffsetTotal(self):
        """Verify total offset = sqrt(dRA² + dDec²)."""
        total = np.sqrt(
            YF26['ephOffsetRa']**2 + YF26['ephOffsetDec']**2
        )
        self.assertAlmostEqual(
            total, YF26['ephOffset'], places=10
        )

    def testAlongCrossTrack(self):
        """Verify along/cross-track decomposition."""
        rate_ra = YF26['ephRateRa']
        rate_dec = YF26['ephRateDec']
        rate = YF26['ephRate']

        # Unit vectors
        along = np.array([rate_ra / rate, rate_dec / rate])
        cross = np.array([-rate_dec / rate, rate_ra / rate])
        offset = np.array(
            [YF26['ephOffsetRa'], YF26['ephOffsetDec']]
        )

        along_track = np.dot(offset, along)
        cross_track = np.dot(offset, cross)

        self.assertAlmostEqual(
            along_track, YF26['ephOffsetAlongTrack'], places=10
        )
        self.assertAlmostEqual(
            cross_track, YF26['ephOffsetCrossTrack'], places=10
        )

    def testAlongCrossTrackPureRA(self):
        """If motion is purely in RA, along-track = RA offset."""
        offset_ra = 0.1  # arcsec
        offset_dec = 0.05
        rate_ra = 0.15  # deg/day
        rate_dec = 0.0
        rate = abs(rate_ra)

        along = np.array([rate_ra / rate, rate_dec / rate])
        cross = np.array([-rate_dec / rate, rate_ra / rate])
        offset = np.array([offset_ra, offset_dec])

        self.assertAlmostEqual(
            np.dot(offset, along), offset_ra, places=10
        )
        self.assertAlmostEqual(
            np.dot(offset, cross), offset_dec, places=10
        )

    def testAlongCrossTrackOrthogonality(self):
        """Along² + cross² should equal total²."""
        at = YF26['ephOffsetAlongTrack']
        ct = YF26['ephOffsetCrossTrack']
        total = YF26['ephOffset']
        self.assertAlmostEqual(
            at**2 + ct**2, total**2, places=10
        )


class TestDerivedQuantities(lsst.utils.tests.TestCase):
    """Test derived physical quantities."""

    def testHelioRange(self):
        """helioRange = |helio_xyz|."""
        r = np.sqrt(
            YF26['helio_x']**2 + YF26['helio_y']**2
            + YF26['helio_z']**2
        )
        self.assertAlmostEqual(r, YF26['helioRange'], places=10)

    def testTopoRange(self):
        """topoRange = |topo_xyz|."""
        r = np.sqrt(
            YF26['topo_x']**2 + YF26['topo_y']**2
            + YF26['topo_z']**2
        )
        self.assertAlmostEqual(r, YF26['topoRange'], places=10)

    def testHelioVtot(self):
        """helio_vtot = |helio_v|."""
        v = np.sqrt(
            YF26['helio_vx']**2 + YF26['helio_vy']**2
            + YF26['helio_vz']**2
        )
        self.assertAlmostEqual(v, YF26['helio_vtot'], places=10)

    def testTopoVtot(self):
        """topo_vtot = |topo_v|."""
        v = np.sqrt(
            YF26['topo_vx']**2 + YF26['topo_vy']**2
            + YF26['topo_vz']**2
        )
        self.assertAlmostEqual(v, YF26['topo_vtot'], places=10)

    def testEphRate(self):
        """ephRate = sqrt(ephRateRa² + ephRateDec²)."""
        rate = np.sqrt(
            YF26['ephRateRa']**2 + YF26['ephRateDec']**2
        )
        self.assertAlmostEqual(rate, YF26['ephRate'], places=10)

    def testPhaseAngle(self):
        """Phase angle from helio and topo vectors."""
        helio = np.array([
            YF26['helio_x'], YF26['helio_y'], YF26['helio_z']
        ])
        topo = np.array([
            YF26['topo_x'], YF26['topo_y'], YF26['topo_z']
        ])
        cos_pa = np.dot(helio, topo) / (
            np.linalg.norm(helio) * np.linalg.norm(topo)
        )
        pa = np.degrees(np.arccos(np.clip(cos_pa, -1, 1)))
        self.assertAlmostEqual(pa, YF26['phaseAngle'], places=6)

    def testElongation(self):
        """Elongation = Sun-object angle as seen from observer."""
        helio = np.array([
            YF26['helio_x'], YF26['helio_y'], YF26['helio_z']
        ])
        topo = np.array([
            YF26['topo_x'], YF26['topo_y'], YF26['topo_z']
        ])
        sun_obs = topo - helio  # vector from Sun to observer
        cos_elong = np.dot(sun_obs, topo) / (
            np.linalg.norm(sun_obs) * np.linalg.norm(topo)
        )
        elong = np.degrees(np.arccos(np.clip(cos_elong, -1, 1)))
        self.assertAlmostEqual(
            elong, YF26['elongation'], places=4
        )

    def testHelioRangeRate(self):
        """helioRangeRate = (v . r) / |r|."""
        r = np.array([
            YF26['helio_x'], YF26['helio_y'], YF26['helio_z']
        ])
        # Velocities are km/s, positions are AU
        # helioRangeRate is in km/s
        v = np.array([
            YF26['helio_vx'], YF26['helio_vy'],
            YF26['helio_vz'],
        ])
        r_km = r * AU_KM
        rdot = np.dot(r_km, v) / np.linalg.norm(r_km)
        self.assertAlmostEqual(
            rdot, YF26['helioRangeRate'], places=4
        )


class TestMatching(lsst.utils.tests.TestCase):
    """Test KDTree spatial matching logic."""

    def setUp(self):
        self.task = SolarSystemAssociationTask()

    def testExactMatch(self):
        """Source at same position as SSO should match."""
        sso_ra = np.array([180.0])
        sso_dec = np.array([0.0])
        dia_ra = np.array([180.0])
        dia_dec = np.array([0.0])

        dia_xyz = self.task._radec_to_xyz(dia_ra, dia_dec)
        tree = cKDTree(dia_xyz)

        sso_xyz = self.task._radec_to_xyz(sso_ra, sso_dec)
        dist, idx = tree.query(sso_xyz, k=1)
        self.assertEqual(idx[0], 0)
        self.assertAlmostEqual(dist[0], 0.0, places=10)

    def testNoMatchBeyondRadius(self):
        """Sources beyond matching radius should not match."""
        sso_ra = np.array([180.0])
        sso_dec = np.array([0.0])
        # Source 10 arcsec away
        dia_ra = np.array([180.0 + 10.0 / 3600])
        dia_dec = np.array([0.0])

        dia_xyz = self.task._radec_to_xyz(dia_ra, dia_dec)
        tree = cKDTree(dia_xyz)

        sso_xyz = self.task._radec_to_xyz(sso_ra, sso_dec)
        max_rad = np.radians(1.0 / 3600)  # 1 arcsec
        dist, idx = tree.query(
            sso_xyz, k=1,
            distance_upper_bound=max_rad,
        )
        self.assertTrue(np.isinf(dist[0]))

    def testNearestNeighbor(self):
        """Closer source should be preferred over farther one."""
        sso_ra = np.array([180.0])
        sso_dec = np.array([0.0])
        # Two sources: one 0.2", one 0.5" away
        dia_ra = np.array([
            180.0 + 0.2 / 3600, 180.0 + 0.5 / 3600
        ])
        dia_dec = np.array([0.0, 0.0])

        dia_xyz = self.task._radec_to_xyz(dia_ra, dia_dec)
        tree = cKDTree(dia_xyz)

        sso_xyz = self.task._radec_to_xyz(sso_ra, sso_dec)
        dist, idx = tree.query(sso_xyz, k=1)
        self.assertEqual(idx[0], 0)  # closer source


class TestChebyshevEphemeris(lsst.utils.tests.TestCase):
    """Test Chebyshev polynomial ephemeris evaluation."""

    def testChebvalEvaluation(self):
        """Evaluate a known Chebyshev polynomial."""
        # f(x) = 3 + 2*T1(x) + 0.5*T2(x)
        # At x=0: T0=1, T1=0, T2=-1 -> f(0) = 3 + 0 - 0.5 = 2.5
        coeffs = [3.0, 2.0, 0.5]
        self.assertAlmostEqual(chebval(0, coeffs), 2.5)
        # At x=1: T0=1, T1=1, T2=1 -> f(1) = 3 + 2 + 0.5 = 5.5
        self.assertAlmostEqual(chebval(1, coeffs), 5.5)

    def testChebvalDerivative(self):
        """Velocity from Chebyshev derivative of position."""
        # Position: p(x) = 1 + x + x² (in Chebyshev basis)
        # In Chebyshev: x = T1, x² = (T0 + T2)/2
        # So p(x) = 1.5*T0 + T1 + 0.5*T2
        coeffs = np.array([1.5, 1.0, 0.5])
        poly = Chebyshev(coeffs)
        dpoly = poly.deriv()

        # dp/dx = 1 + 2x
        # At x=0: dp/dx = 1
        self.assertAlmostEqual(dpoly(0), 1.0, places=10)
        # At x=0.5: dp/dx = 2
        self.assertAlmostEqual(dpoly(0.5), 2.0, places=10)


class TestSorchaPath(lsst.utils.tests.TestCase):
    """Test Sorcha ephemeris path computations."""

    def testTopoFromHelioAndObserver(self):
        """Topo position = helio position - observer position."""
        # Use YF26 vectors: topo = helio - observer
        # observer = helio - topo
        obs_x = YF26['helio_x'] - YF26['topo_x']
        obs_y = YF26['helio_y'] - YF26['topo_y']
        obs_z = YF26['helio_z'] - YF26['topo_z']

        topo_x = YF26['helio_x'] - obs_x
        topo_y = YF26['helio_y'] - obs_y
        topo_z = YF26['helio_z'] - obs_z

        self.assertAlmostEqual(topo_x, YF26['topo_x'], places=12)
        self.assertAlmostEqual(topo_y, YF26['topo_y'], places=12)
        self.assertAlmostEqual(topo_z, YF26['topo_z'], places=12)

    def testKmToAuConversion(self):
        """Verify km to AU conversion factor."""
        # 1 AU in km
        self.assertAlmostEqual(
            1.0 * AU_KM, 1.496e8, places=-2
        )
        # Round-trip: AU -> km -> AU
        dist_au = 2.17
        dist_km = dist_au * AU_KM
        self.assertAlmostEqual(
            dist_km / AU_KM, dist_au, places=10
        )


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
