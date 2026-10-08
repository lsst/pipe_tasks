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
from astropy.table import Table
import astropy.units as u
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.daf.base as dafBase
import lsst.geom
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
        max_arcsec = self.task.config.maxDistArcSeconds
        sso_ra = np.array([180.0])
        sso_dec = np.array([0.0])
        # Source well beyond the matching radius
        dia_ra = np.array([180.0 + 10 * max_arcsec / 3600])
        dia_dec = np.array([0.0])

        dia_xyz = self.task._radec_to_xyz(dia_ra, dia_dec)
        tree = cKDTree(dia_xyz)

        sso_xyz = self.task._radec_to_xyz(sso_ra, sso_dec)
        max_rad = np.radians(max_arcsec / 3600)
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


# ---------------------------------------------------------------
# Shutter-corrected epochs (``run(..., shutterTiming=...)``)
# ---------------------------------------------------------------
T_VISIT = 61000.25      # MJD TAI of visitInfo.date
T_MIN = 61000.0         # Chebyshev reference epoch (mpSky ``tmin``)
SEC_PER_DAY = 86400.0
# Tolerances: platforms differ at the ~1e-10 deg / bit level (Jenkins); the
# shifts tested move positions by >~1e-5 deg and >~10 km.
EPH_ATOL_DEG = 1e-8
POS_ATOL_AU = 1e-11     # 1.5 m
VEL_RTOL = 1e-10

# Objects: pixel position at the visit time, topocentric distance (au),
# sky rate (deg/day) and position angle of the motion (deg, E of N).
OBJECTS = [
    # x,      y,      rho,  rate, pa
    (300.0, 400.0, 1.3, 0.25, 30.0),
    (3800.0, 3600.0, 2.5, 0.10, 200.0),
    (2000.0, 2000.0, 0.05, 10.0, 75.0),   # fast-moving NEA
    (3500.0, 500.0, 0.8, 1.0, 300.0),
]

# ephRa, ephDec, topoRange and helio_vx of the mpSky OBJECTS, sorted by
# ssObjectId, from origin/main's code (before shutterTiming was added).
MAIN_MPSKY = np.array([
    (150.09795883127038, 9.911056565639797, 1.300000166899959, 28.43065939511504),
    (149.90051673635801, 10.08883186363795, 2.5000003199946863, 29.83129921288734),
    (150.00620316401947, 9.996846454592841, 0.05000049072273312, 22.904194542965843),
    (149.91778094808433, 9.916393993240405, 0.8000006524288672, 41.910656240016266),
])


def _makeWcs(ra=150.0, dec=10.0):
    """A TAN WCS with 0.2"/px for an LSSTCam-sized detector."""
    return afwGeom.makeSkyWcs(crpix=lsst.geom.Point2D(2036.0, 2000.0),
                              crval=lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees),
                              cdMatrix=afwGeom.makeCdMatrix(scale=0.2*lsst.geom.arcseconds))


def _unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])


def _makeMpSkyObjects(wcs):
    """mpSky-style ``ssObjects``: Chebyshev coefficients in (t - tmin)
    days of the observer and object positions (au).
    """
    tRef = T_VISIT - T_MIN
    obs = np.array([[0.9, 0.017, 1e-4], [-0.4, 0.0155, -2e-4], [-0.17, 0.0067, 5e-5]])

    objs, ras, decs = [], [], []
    for i, (x, y, rho, rate, pa) in enumerate(OBJECTS):
        # Direction and sky velocity (au/d) at the visit time.
        sp = wcs.pixelToSky(x, y)
        ra, dec = sp.getRa().asDegrees(), sp.getDec().asDegrees()
        e = _unit(ra, dec)
        east = np.array([-np.sin(np.radians(ra)), np.cos(np.radians(ra)), 0.0])
        north = np.cross(e, east)
        vel = rho*np.radians(rate)*(np.sin(np.radians(pa))*east + np.cos(np.radians(pa))*north)

        # Object = observer + rho e at the visit time, moving with vel.
        obj = obs.copy()
        obj[:, 0] += rho*e - vel*tRef
        obj[:, 1] += vel
        obj[:, 2] += 1e-6*(i + 1)

        objs.append(obj)
        ras.append(ra)
        decs.append(dec)

    n = len(OBJECTS)
    names = [f"2026 A{i:02d}" for i in range(n)]
    table = Table({
        'ObjID': names, 'ra': ras * u.deg, 'dec': decs * u.deg,
        'tmin': np.full(n, T_MIN) * u.d, 'tmax': np.full(n, T_MIN + 1) * u.d,
        'trailedSourceMagTrue': np.full(n, 20.0) * u.mag,
        'Err(arcsec)': np.full(n, 2.0) * u.arcsec,
        'ssObjectId': np.arange(1, n + 1, dtype=np.int64) * 1000,
        'MPCORB_unpacked_primary_provisional_designation': names,
    })
    for i, c in enumerate('xyz'):
        table[f'obj_{c}_poly'] = [obj[i] for obj in objs]
        table[f'obs_{c}_poly'] = [obs[i]]*n

    return table


def _expectedMpSkyRaDec(ssObjects, mjdTai):
    """Topocentric RA, Dec (deg) evaluated directly from the Chebyshev
    coefficients at ``mjdTai`` (scalar or per object).
    """
    vec = np.array([[chebval(t - T_MIN, row[f'obj_{c}_poly']) - chebval(t - T_MIN, row[f'obs_{c}_poly'])
                     for c in 'xyz']
                    for row, t in zip(ssObjects, np.broadcast_to(mjdTai, (len(ssObjects),)))])
    ra = np.degrees(np.arctan2(vec[:, 1], vec[:, 0])) % 360
    dec = np.degrees(np.arctan2(vec[:, 2], np.hypot(vec[:, 0], vec[:, 1])))
    return ra, dec


def _makeSorchaObjects(wcs, objects=OBJECTS):
    """Sorcha-style ``ssObjects``: RA/Dec and sky rates at the visit time,
    plus heliocentric state vectors.
    """
    n = len(objects)
    sky = [wcs.pixelToSky(x, y) for x, y, *_ in objects]
    km = 1.495978707e8
    t = Table({
        'ObjID': [f"S{i:03d}" for i in range(n)],
        'packed_desig': [f"K26A{i:02d}A" for i in range(n)],
        'FieldID': np.zeros(n, dtype=int),
        'fieldMJD_TAI': np.full(n, T_VISIT),
        'fieldJD_TDB': np.full(n, T_VISIT + 2400000.5),
        'RATrue_deg': [sp.getRa().asDegrees() for sp in sky] * u.deg,
        'DecTrue_deg': [sp.getDec().asDegrees() for sp in sky] * u.deg,
        'RARateCosDec_deg_day': [o[3]*np.sin(np.radians(o[4])) for o in objects],
        'DecRate_deg_day': [o[3]*np.cos(np.radians(o[4])) for o in objects],
        'RangeRate_LTC_km_s': np.full(n, 3.0),
        'phase_deg': np.full(n, 20.0),
        'Range_LTC_km': np.array([o[2] for o in objects])*km * u.km,
        'trailedSourceMagTrue': np.full(n, 21.0),
    })
    for i, c in enumerate('xyz'):
        t[f'Obj_Sun_{c}_LTC_km'] = np.full(n, (1.5 + 0.1*i)*km) * u.km
        t[f'Obj_Sun_v{c}_LTC_km_s'] = np.full(n, 10.0 + i) * u.km/u.s
        t[f'Obs_Sun_{c}_km'] = np.full(n, (0.9 - 0.3*i)*km) * u.km
        t[f'Obs_Sun_v{c}_km_s'] = np.full(n, 20.0 - i) * u.km/u.s

    return t


def _sorchaExpected(ssObjects, dtSec):
    """Sorcha positions moved linearly by ``dtSec`` (scalar or per object),
    in the order of `TestShutterTimingEpochs._ephByObject`.
    """
    dt = np.asarray(dtSec)/SEC_PER_DAY
    ra0, dec0 = np.array(ssObjects['RATrue_deg']), np.array(ssObjects['DecTrue_deg'])
    ra = (ra0 + np.array(ssObjects['RARateCosDec_deg_day'])*dt/np.cos(np.radians(dec0))) % 360
    dec = dec0 + np.array(ssObjects['DecRate_deg_day'])*dt

    order = np.argsort(ssObjects['ObjID'])
    return ra[order], dec[order]


def _makeDiaSources(ras, decs, offsetArcsec=0.05):
    """DiaSources a little off each predicted position, plus an unrelated one."""
    ras = np.append(np.asarray(ras, dtype=float), 150.01)
    decs = np.append(np.asarray(decs, dtype=float) + offsetArcsec/3600, 10.01)
    n = len(ras)
    return Table({
        'diaSourceId': np.arange(1, n + 1, dtype=np.int64), 'midpointMjdTai': np.full(n, T_VISIT),
        'ra': ras, 'dec': decs, 'extendedness': np.zeros(n), 'band': ['r']*n,
        'psfFlux': np.full(n, 1000.0), 'psfFluxErr': np.full(n, 10.0),
    })


class FakeShutterTiming:
    """Stand-in for `lsst.ip.isr.shutterTiming.ShutterTiming`: time
    ``T_VISIT + offset(x, y)`` seconds, NaN where ``unavailable(x, y)``.
    """

    def __init__(self, offset, unavailable=lambda x, y: False):
        self.offset = offset
        self.unavailable = unavailable

    def tMidMjdTai(self, x, y):
        """Times (MJD TAI) at pixel positions ``x``, ``y`` (arrays)."""
        t = T_VISIT + np.broadcast_to(self.offset(x, y), x.shape)/SEC_PER_DAY
        return np.where(self.unavailable(x, y), np.nan, t)


class TestShutterTimingEpochs(lsst.utils.tests.TestCase):
    """Per-object ephemeris epochs from a shutter timing."""

    def setUp(self):
        self.wcs = _makeWcs()
        self.bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(4072, 4000))
        self.visitInfo = afwImage.VisitInfo(date=dafBase.DateTime(T_VISIT, dafBase.DateTime.MJD,
                                                                  dafBase.DateTime.TAI))

    def _run(self, ssObjects, shutterTiming=None, wcs=None):
        """Run the task with DiaSources near the visit-time predictions."""
        if 'obs_x_poly' in ssObjects.columns:
            diaRaDec = _expectedMpSkyRaDec(ssObjects, T_VISIT)
        else:
            diaRaDec = (ssObjects['RATrue_deg'], ssObjects['DecTrue_deg'])

        return SolarSystemAssociationTask().run(_makeDiaSources(*diaRaDec), ssObjects.copy(), self.visitInfo,
                                                self.bbox, wcs or self.wcs, shutterTiming=shutterTiming)

    @staticmethod
    def _ephByObject(result):
        """Predicted RA, Dec of every object, sorted by ssObjectId."""
        tables = [result.associatedSsSources, result.unassociatedSsObjects]
        ids, ra, dec = (np.concatenate([t[c] for t in tables]) for c in ['ssObjectId', 'ephRa', 'ephDec'])
        order = np.argsort(ids)
        return ids[order], ra[order], dec[order]

    def testNoneIsUnchanged(self):
        """``shutterTiming=None`` reproduces origin/main (stored values, as
        bit-level results differ across platforms) and equals, bit for bit,
        a timing that is NaN everywhere.
        """
        # No timing: the stored main-branch values.
        sss = self._run(_makeMpSkyObjects(self.wcs)).associatedSsSources
        sss.sort('ssObjectId')
        for i, (col, atol) in enumerate([('ephRa', EPH_ATOL_DEG), ('ephDec', EPH_ATOL_DEG),
                                         ('topoRange', POS_ATOL_AU)]):
            np.testing.assert_allclose(sss[col], MAIN_MPSKY[:, i], rtol=0, atol=atol)
        np.testing.assert_allclose(sss['helio_vx'], MAIN_MPSKY[:, 3], rtol=VEL_RTOL)

        # A timing that is NaN everywhere: bit-identical to no timing.
        nowhere = FakeShutterTiming(lambda x, y: 0.3, unavailable=lambda x, y: True)
        for make in (_makeMpSkyObjects, _makeSorchaObjects):
            ssObjects = make(self.wcs)
            none, unav = self._run(ssObjects), self._run(ssObjects, nowhere)
            self.assertEqual(none.nAssociatedSsObjects, len(ssObjects))
            for name in ['ssoAssocDiaSources', 'unAssocDiaSources', 'associatedSsSources',
                         'unassociatedSsObjects']:
                a, b = getattr(none, name), getattr(unav, name)
                self.assertEqual(a.colnames, b.colnames)
                for c in a.colnames:
                    self.assertEqual(np.asarray(a[c]).tobytes(), np.asarray(b[c]).tobytes(), c)

    def testConstantOffset(self):
        """A constant Δt moves every prediction, and the state vectors, to
        the Chebyshev evaluated at t_visit + Δt.
        """
        dtSec = 0.37
        ssObjects = _makeMpSkyObjects(self.wcs)
        result = self._run(ssObjects, FakeShutterTiming(lambda x, y: dtSec))

        # Predicted positions.
        expRa, expDec = _expectedMpSkyRaDec(ssObjects, T_VISIT + dtSec/SEC_PER_DAY)
        _, ra, dec = self._ephByObject(result)
        np.testing.assert_allclose(ra, expRa, rtol=0, atol=EPH_ATOL_DEG)
        np.testing.assert_allclose(dec, expDec, rtol=0, atol=EPH_ATOL_DEG)

        # The 10 deg/day NEA moves along its track by rate × Δt (~0.15").
        visRa, visDec = _expectedMpSkyRaDec(ssObjects, T_VISIT)
        _, _, _, rate, pa = OBJECTS[2]
        move = rate*dtSec/SEC_PER_DAY
        np.testing.assert_allclose([(ra[2] - visRa[2])*np.cos(np.radians(visDec[2])), dec[2] - visDec[2]],
                                   move*np.array([np.sin(np.radians(pa)), np.cos(np.radians(pa))]),
                                   rtol=0, atol=1e-3*move)

        # State vectors.
        tRef = T_VISIT + dtSec/SEC_PER_DAY - T_MIN
        sss = result.associatedSsSources
        sss.sort('ssObjectId')
        self.assertEqual(len(sss), len(ssObjects))
        for k, row in enumerate(ssObjects):
            helio = np.array([chebval(tRef, row[f'obj_{c}_poly']) for c in 'xyz'])
            topo = helio - [chebval(tRef, row[f'obs_{c}_poly']) for c in 'xyz']
            np.testing.assert_allclose([sss[f'helio_{c}'][k] for c in 'xyz'], helio, rtol=0, atol=POS_ATOL_AU)
            np.testing.assert_allclose([sss[f'topo_{c}'][k] for c in 'xyz'], topo, rtol=0, atol=POS_ATOL_AU)
            self.assertAlmostEqual(sss['topoRange'][k], np.linalg.norm(topo), delta=POS_ATOL_AU)

        # Offsets are measured from the shifted prediction.
        np.testing.assert_allclose(sss['ephOffsetDec'], (sss['dec'] - sss['ephDec'])*3600,
                                   rtol=0, atol=1e-5)  # arcsec

    def testPositionDependent(self):
        """Each object uses the time of its own predicted pixel."""
        def offset(x, y):
            return 0.25*(x - 2036.0)/2036.0 - 0.1*y/4000.0

        ssObjects = _makeMpSkyObjects(self.wcs)
        result = self._run(ssObjects, FakeShutterTiming(offset))

        # The offsets differ between objects.
        x, y = self.wcs.skyToPixelArray(*_expectedMpSkyRaDec(ssObjects, T_VISIT), degrees=True)
        self.assertGreater(np.ptp(offset(x, y)), 0.2)

        expRa, expDec = _expectedMpSkyRaDec(ssObjects, T_VISIT + offset(x, y)/SEC_PER_DAY)
        _, ra, dec = self._ephByObject(result)
        np.testing.assert_allclose(ra, expRa, rtol=0, atol=EPH_ATOL_DEG)
        np.testing.assert_allclose(dec, expDec, rtol=0, atol=EPH_ATOL_DEG)

    def testUnavailableKeepsVisitTime(self):
        """Objects whose pixel time is NaN (UNAVAILABLE) keep the visit time."""
        ssObjects = _makeMpSkyObjects(self.wcs)
        result = self._run(ssObjects, FakeShutterTiming(lambda x, y: 0.4, lambda x, y: x > 3000.0))

        # Some objects, but not all, have no time.
        bad = np.array([o[0] > 3000.0 for o in OBJECTS])
        self.assertTrue(np.any(bad) and not np.all(bad))

        expRa, expDec = _expectedMpSkyRaDec(ssObjects, np.where(bad, T_VISIT, T_VISIT + 0.4/SEC_PER_DAY))
        _, ra, dec = self._ephByObject(result)
        np.testing.assert_allclose(ra, expRa, rtol=0, atol=EPH_ATOL_DEG)
        np.testing.assert_allclose(dec, expDec, rtol=0, atol=EPH_ATOL_DEG)

    def testSorchaLinearShift(self):
        """Sorcha predictions move linearly by rate × Δt."""
        ssObjects = _makeSorchaObjects(self.wcs)
        _, ra, dec = self._ephByObject(self._run(ssObjects, FakeShutterTiming(lambda x, y: 0.45)))
        expRa, expDec = _sorchaExpected(ssObjects, 0.45)
        np.testing.assert_allclose(ra, expRa, rtol=0, atol=EPH_ATOL_DEG)
        np.testing.assert_allclose(dec, expDec, rtol=0, atol=EPH_ATOL_DEG)

    def testSorchaRaWrap(self):
        """Linear shifts across RA = 0 stay in [0, 360)."""
        wcs = _makeWcs(0.0, 0.0)
        ssObjects = _makeSorchaObjects(wcs, OBJECTS[:2])
        ssObjects['RATrue_deg'] = [359.99998, 0.00002] * u.deg
        ssObjects['DecTrue_deg'] = [0.0, 0.0] * u.deg
        ssObjects['RARateCosDec_deg_day'] = [10.0, -10.0]   # east, west
        ssObjects['DecRate_deg_day'] = [0.0, 0.0]

        _, ra, dec = self._ephByObject(self._run(ssObjects, FakeShutterTiming(lambda x, y: 0.45), wcs=wcs))
        expRa, expDec = _sorchaExpected(ssObjects, 0.45)
        self.assertTrue(expRa[0] < 1.0 and expRa[1] > 359.0)   # both cross RA = 0
        np.testing.assert_allclose(ra, expRa, rtol=0, atol=EPH_ATOL_DEG)
        np.testing.assert_allclose(dec, expDec, rtol=0, atol=EPH_ATOL_DEG)

    def testSorchaFieldTime(self):
        """Sorcha predictions move from their own ``fieldMJD_TAI``, not from
        ``visitInfo.date``.
        """
        fieldOffsetSec = np.array([-2.0, -1.0, 0.5, 1.5])
        ssObjects = _makeSorchaObjects(self.wcs)
        ssObjects['fieldMJD_TAI'] = T_VISIT + fieldOffsetSec/SEC_PER_DAY

        _, ra, dec = self._ephByObject(self._run(ssObjects, FakeShutterTiming(lambda x, y: 0.45)))
        expRa, expDec = _sorchaExpected(ssObjects, 0.45 - fieldOffsetSec)
        np.testing.assert_allclose(ra, expRa, rtol=0, atol=EPH_ATOL_DEG)
        np.testing.assert_allclose(dec, expDec, rtol=0, atol=EPH_ATOL_DEG)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
