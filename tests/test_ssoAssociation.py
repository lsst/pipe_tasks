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
from lsst.ip.isr.shutterTiming import ShutterTimingStatus
import healpy as hp
import lsst.utils.tests
from lsst.pipe.tasks.ssoAssociation import SolarSystemAssociationConfig, SolarSystemAssociationTask

AU_KM = 1.496e8


# ---------------------------------------------------------------
# Synthetic inputs for SolarSystemAssociationTask.run
# ---------------------------------------------------------------
T_VISIT = 61000.25      # MJD TAI of visitInfo.date
T_MIN = 61000.0         # Chebyshev reference epoch (mpSky ``tmin``)
SEC_PER_DAY = 86400.0
# Agreement of predicted positions (deg).  Different platforms (e.g. macOS
# arm64 vs Linux x86) differ by ~1e-10 deg; the epoch shifts tested here move
# positions by >~1e-5 deg.
EPH_ATOL_DEG = 1e-8
# Same for state vectors: positions (au; 1e-11 au = 1.5 m) and velocities
# (relative).  The epoch shifts tested here move objects by >~10 km.
POS_ATOL_AU = 1e-11
VEL_RTOL = 1e-10


def _makeWcsAndBox():
    """A TAN WCS with 0.2"/px around (RA, Dec) = (150, 10) and an
    LSSTCam-sized detector.
    """
    bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(4072, 4000))
    wcs = afwGeom.makeSkyWcs(
        crpix=lsst.geom.Point2D(2036.0, 2000.0),
        crval=lsst.geom.SpherePoint(150.0, 10.0, lsst.geom.degrees),
        cdMatrix=afwGeom.makeCdMatrix(scale=0.2*lsst.geom.arcseconds),
    )
    return bbox, wcs


def _makeVisitInfo():
    return afwImage.VisitInfo(date=dafBase.DateTime(T_VISIT, dafBase.DateTime.MJD,
                                                    dafBase.DateTime.TAI))


def _unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])


# Objects: pixel position at the visit time, topocentric distance (au),
# sky rate (deg/day) and position angle of the motion (deg, E of N).
OBJECTS = [
    # x,      y,      rho,  rate, pa
    (300.0, 400.0, 1.3, 0.25, 30.0),
    (3800.0, 3600.0, 2.5, 0.10, 200.0),
    (2000.0, 2000.0, 0.05, 10.0, 75.0),   # fast-moving NEA
    (3500.0, 500.0, 0.8, 1.0, 300.0),
]


def _makeMpSkyObjects(wcs, objects=OBJECTS):
    """mpSky-style ``ssObjects``: Chebyshev coefficients in (t - tmin)
    days of the observer and object barycentric positions (au).
    """
    tRef = T_VISIT - T_MIN
    # Observer: Earth-like, with a small quadratic term.
    obs = np.array([[0.9, 0.017, 1e-4],
                    [-0.4, 0.0155, -2e-4],
                    [-0.17, 0.0067, 5e-5]])
    rows = []
    for i, (x, y, rho, rate, pa) in enumerate(objects):
        sp = wcs.pixelToSky(x, y)
        ra, dec = sp.getRa().asDegrees(), sp.getDec().asDegrees()
        e = _unit(ra, dec)
        east = np.array([-np.sin(np.radians(ra)), np.cos(np.radians(ra)), 0.0])
        north = np.cross(e, east)
        direction = np.sin(np.radians(pa))*east + np.cos(np.radians(pa))*north
        vel = rho*np.radians(rate)*direction           # au/day
        obj = obs.copy()
        obj[:, 0] += rho*e - vel*tRef
        obj[:, 1] += vel
        # Make the object's polynomial quadratic too.
        obj[:, 2] += 1e-6*(i + 1)
        rows.append((obj, ra, dec))
    n = len(objects)
    names = [f"2026 A{i:02d}" for i in range(n)]
    return Table({
        'ObjID': names,
        'ra': [r[1] for r in rows] * u.deg,
        'dec': [r[2] for r in rows] * u.deg,
        'tmin': np.full(n, T_MIN) * u.d,
        'tmax': np.full(n, T_MIN + 1) * u.d,
        'trailedSourceMagTrue': np.full(n, 20.0) * u.mag,
        'obj_x_poly': [r[0][0] for r in rows],
        'obj_y_poly': [r[0][1] for r in rows],
        'obj_z_poly': [r[0][2] for r in rows],
        'obs_x_poly': [obs[0] for _ in range(n)],
        'obs_y_poly': [obs[1] for _ in range(n)],
        'obs_z_poly': [obs[2] for _ in range(n)],
        'Err(arcsec)': np.full(n, 2.0) * u.arcsec,
        'ssObjectId': np.arange(1, n + 1, dtype=np.int64) * 1000,
        'MPCORB_unpacked_primary_provisional_designation': names,
    })


def _expectedMpSkyRaDec(ssObjects, mjdTai):
    """Topocentric RA, Dec (deg) of each object evaluated directly from its
    Chebyshev coefficients at ``mjdTai`` (scalar or per object).
    """
    mjdTai = np.broadcast_to(mjdTai, (len(ssObjects),))
    vec = []
    for row, t in zip(ssObjects, mjdTai):
        dt = t - T_MIN
        vec.append([chebval(dt, row[f'obj_{c}_poly']) - chebval(dt, row[f'obs_{c}_poly'])
                    for c in 'xyz'])
    vec = np.array(vec)
    ra = np.degrees(np.arctan2(vec[:, 1], vec[:, 0])) % 360
    dec = np.degrees(np.arctan2(vec[:, 2], np.hypot(vec[:, 0], vec[:, 1])))
    return ra, dec


def _makeSorchaObjects(wcs, objects=OBJECTS):
    """Sorcha-style ``ssObjects``: precomputed RA/Dec and sky rates at the
    visit time, plus heliocentric state vectors.
    """
    n = len(objects)
    ras, decs, raRates, decRates = [], [], [], []
    for x, y, rho, rate, pa in objects:
        sp = wcs.pixelToSky(x, y)
        ras.append(sp.getRa().asDegrees())
        decs.append(sp.getDec().asDegrees())
        raRates.append(rate*np.sin(np.radians(pa)))
        decRates.append(rate*np.cos(np.radians(pa)))
    km = 1.495978707e8
    t = Table({
        'ObjID': [f"S{i:03d}" for i in range(n)],
        'packed_desig': [f"K26A{i:02d}A" for i in range(n)],
        'FieldID': np.zeros(n, dtype=int),
        'fieldMJD_TAI': np.full(n, T_VISIT),
        'fieldJD_TDB': np.full(n, T_VISIT + 2400000.5),
        'RATrue_deg': ras * u.deg,
        'DecTrue_deg': decs * u.deg,
        'RARateCosDec_deg_day': raRates,
        'DecRate_deg_day': decRates,
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


def _makeDiaSources(ras, decs, offsetArcsec=0.05):
    """DiaSources a little off each predicted position, plus an unrelated
    one.
    """
    ras = np.append(np.asarray(ras, dtype=float), 150.01)
    decs = np.append(np.asarray(decs, dtype=float) + offsetArcsec/3600, 10.01)
    n = len(ras)
    return Table({
        'diaSourceId': np.arange(1, n + 1, dtype=np.int64),
        'midpointMjdTai': np.full(n, T_VISIT),
        'ra': ras,
        'dec': decs,
        'extendedness': np.zeros(n),
        'band': ['r']*n,
        'psfFlux': np.full(n, 1000.0),
        'psfFluxErr': np.full(n, 10.0),
    })


class FakeShutterTiming:
    """Stand-in for `lsst.ip.isr.shutterTiming.ShutterTiming`: per-pixel
    time ``T_VISIT + offset(x, y)`` seconds; UNAVAILABLE (NaN) where
    ``unavailable(x, y)`` is true.
    """

    def __init__(self, offset, unavailable=None, headerMidMjdTai=T_VISIT, focalPlaneMjdTai=T_VISIT,
                 detectorId=None):
        self.offset = offset
        self.unavailable = unavailable
        self.headerMidMjdTai = headerMidMjdTai
        self.focalPlaneMjdTai = focalPlaneMjdTai
        if detectorId is not None:
            self.detectorId = detectorId
        self.calls = []

    def _bad(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        bad = ~(np.isfinite(x) & np.isfinite(y))
        if self.unavailable is not None:
            bad |= self.unavailable(x, y)
        return bad

    def tMidMjdTai(self, x, y):
        self.calls.append((np.array(x, dtype=float), np.array(y, dtype=float)))
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        t = T_VISIT + np.broadcast_to(self.offset(x, y), x.shape)/SEC_PER_DAY
        return np.where(self._bad(x, y), np.nan, t)

    def sourceStatus(self, x, y):
        bad = self._bad(x, y)
        return np.where(bad, ShutterTimingStatus.UNAVAILABLE, ShutterTimingStatus.OK).astype(np.uint8)


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


def _assertTablesEqual(testCase, a, b):
    """Assert two astropy tables have identical columns, dtypes and bytes."""
    testCase.assertEqual(a.colnames, b.colnames)
    for c in a.colnames:
        testCase.assertEqual(a[c].dtype, b[c].dtype, c)
        testCase.assertEqual(np.asarray(a[c]).tobytes(), np.asarray(b[c]).tobytes(), c)


class TestShutterTimingEpochs(lsst.utils.tests.TestCase):
    """Per-object ephemeris epochs from a shutter timing
    (``SolarSystemAssociationTask.run(..., shutterTiming=...)``).
    """

    def setUp(self):
        self.bbox, self.wcs = _makeWcsAndBox()
        self.visitInfo = _makeVisitInfo()

    def _run(self, ssObjects, shutterTiming=..., diaRaDec=None, wcs=None, detector=None):
        if diaRaDec is None:
            if 'obs_x_poly' in ssObjects.columns:
                diaRaDec = _expectedMpSkyRaDec(ssObjects, T_VISIT)
            else:
                diaRaDec = (np.array(ssObjects['RATrue_deg']), np.array(ssObjects['DecTrue_deg']))
        task = SolarSystemAssociationTask()
        kwargs = {} if shutterTiming is ... else {'shutterTiming': shutterTiming}
        diaSources = _makeDiaSources(*diaRaDec)
        if detector is not None:
            diaSources['detector'] = np.full(len(diaSources), detector, dtype=np.int16)
        result = task.run(diaSources, ssObjects.copy(), self.visitInfo, self.bbox,
                          self.wcs if wcs is None else wcs, **kwargs)
        return task, result

    @staticmethod
    def _ephByObject(result):
        """Predicted RA, Dec of every object (associated or not), sorted by
        ssObjectId.
        """
        ids = np.concatenate([result.associatedSsSources['ssObjectId'],
                              result.unassociatedSsObjects['ssObjectId']])
        ra = np.concatenate([result.associatedSsSources['ephRa'], result.unassociatedSsObjects['ephRa']])
        dec = np.concatenate([result.associatedSsSources['ephDec'],
                              result.unassociatedSsObjects['ephDec']])
        order = np.argsort(ids)
        return ids[order], ra[order], dec[order]

    def _compareResults(self, a, b):
        self.assertEqual(a.nTotalSsObjects, b.nTotalSsObjects)
        self.assertEqual(a.nAssociatedSsObjects, b.nAssociatedSsObjects)
        for name in ['ssoAssocDiaSources', 'unAssocDiaSources', 'associatedSsSources',
                     'unassociatedSsObjects']:
            _assertTablesEqual(self, getattr(a, name), getattr(b, name))

    def testNoneIsUnchanged(self):
        """``shutterTiming=None`` gives the same outputs as omitting it and as
        a timing that is UNAVAILABLE everywhere, with every prediction
        bit-identical to the pre-shutter-timing evaluation at the visit time.
        """
        for make in (_makeMpSkyObjects, _makeSorchaObjects):
            ssObjects = make(self.wcs)
            _, default = self._run(ssObjects)
            task, none = self._run(ssObjects, None)
            self._compareResults(default, none)
            self.assertEqual(task.metadata['nSsoShutterEpochs'], 0)
            self.assertEqual(task.metadata['ssoShutterEpochMaxShift'], 0.0)
            unavailable = FakeShutterTiming(lambda x, y: 0.3, unavailable=lambda x, y: np.ones_like(x, bool))
            task, unav = self._run(ssObjects, unavailable)
            self._compareResults(none, unav)
            self.assertEqual(task.metadata['nSsoShutterEpochs'], 0)
            self.assertEqual(none.nAssociatedSsObjects, len(ssObjects))

        # The mpSky evaluation at visitInfo.date as written before shutter
        # timing was added, as an independent reference.
        ssObjects = _makeMpSkyObjects(self.wcs)
        _, none = self._run(ssObjects, None)
        ref_time = self.visitInfo.date.toAstropy().tai.mjd - ssObjects["tmin"].quantity.to_value(u.d)[0]
        obs = [np.array([chebval(ref_time, row['obs_x_poly']), chebval(ref_time, row['obs_y_poly']),
                         chebval(ref_time, row['obs_z_poly'])]) for row in ssObjects] * u.au
        obj = [np.array([chebval(ref_time, row['obj_x_poly']), chebval(ref_time, row['obj_y_poly']),
                         chebval(ref_time, row['obj_z_poly'])]) for row in ssObjects] * u.au
        objV = [np.array([chebval(ref_time, Chebyshev(row['obj_x_poly']).deriv().coef),
                          chebval(ref_time, Chebyshev(row['obj_y_poly']).deriv().coef),
                          chebval(ref_time, Chebyshev(row['obj_z_poly']).deriv().coef)])
                for row in ssObjects] * u.au/u.d
        ras, decs = np.vstack(hp.vec2ang(np.vstack(obj.to_value(u.au) - obs.to_value(u.au)), lonlat=True))
        ids, ra, dec = self._ephByObject(none)
        np.testing.assert_array_equal(ids, ssObjects['ssObjectId'])
        np.testing.assert_allclose(ra, ras, rtol=0, atol=EPH_ATOL_DEG)
        np.testing.assert_allclose(dec, decs, rtol=0, atol=EPH_ATOL_DEG)
        sss = none.associatedSsSources
        order = np.argsort(sss['ssObjectId'])
        np.testing.assert_allclose(sss['helio_x'][order], obj.to_value(u.au)[:, 0], rtol=0, atol=POS_ATOL_AU)
        np.testing.assert_allclose(sss['helio_vz'][order], objV.to_value(u.km/u.s)[:, 2], rtol=VEL_RTOL)

    def testConstantOffset(self):
        """A constant Δt moves every prediction to the Chebyshev evaluated at
        t_visit + Δt, and the state vectors with it.
        """
        dtSec = 0.37
        ssObjects = _makeMpSkyObjects(self.wcs)
        task, result = self._run(ssObjects, FakeShutterTiming(lambda x, y: dtSec))
        expRa, expDec = _expectedMpSkyRaDec(ssObjects, T_VISIT + dtSec/SEC_PER_DAY)
        ids, ra, dec = self._ephByObject(result)
        np.testing.assert_allclose(ra, expRa, rtol=0, atol=EPH_ATOL_DEG)
        np.testing.assert_allclose(dec, expDec, rtol=0, atol=EPH_ATOL_DEG)
        # Not at the visit time (the NEA moves ~0.15" in 0.37 s).
        visRa, visDec = _expectedMpSkyRaDec(ssObjects, T_VISIT)
        self.assertGreater(np.max(np.abs(ra - visRa)*3600), 0.05)

        tRef = T_VISIT + dtSec/SEC_PER_DAY - T_MIN
        sss = result.associatedSsSources
        self.assertEqual(len(sss), len(ssObjects))
        order = np.argsort(sss['ssObjectId'])
        for i, row in enumerate(ssObjects):
            helio = [chebval(tRef, row[f'obj_{c}_poly']) for c in 'xyz']
            topo = [chebval(tRef, row[f'obj_{c}_poly']) - chebval(tRef, row[f'obs_{c}_poly']) for c in 'xyz']
            k = order[i]
            np.testing.assert_allclose([sss['helio_x'][k], sss['helio_y'][k], sss['helio_z'][k]], helio,
                                       rtol=0, atol=POS_ATOL_AU)
            np.testing.assert_allclose([sss['topo_x'][k], sss['topo_y'][k], sss['topo_z'][k]], topo,
                                       rtol=0, atol=POS_ATOL_AU)
            self.assertAlmostEqual(sss['topoRange'][k], np.linalg.norm(topo), delta=POS_ATOL_AU)
            # Velocities are the Chebyshev derivatives at the same epoch.
            kmPerSecPerAuPerDay = (1*u.au/u.d).to_value(u.km/u.s)
            helioV = [chebval(tRef, Chebyshev(row[f'obj_{c}_poly']).deriv().coef)*kmPerSecPerAuPerDay
                      for c in 'xyz']
            topoV = [(chebval(tRef, Chebyshev(row[f'obj_{c}_poly']).deriv().coef)
                      - chebval(tRef, Chebyshev(row[f'obs_{c}_poly']).deriv().coef))*kmPerSecPerAuPerDay
                     for c in 'xyz']
            np.testing.assert_allclose([sss['helio_vx'][k], sss['helio_vy'][k], sss['helio_vz'][k]], helioV,
                                       rtol=VEL_RTOL, atol=0)
            np.testing.assert_allclose([sss['topo_vx'][k], sss['topo_vy'][k], sss['topo_vz'][k]], topoV,
                                       rtol=VEL_RTOL, atol=0)
            self.assertAlmostEqual(sss['helioRangeRate'][k], np.dot(helioV, helio)/np.linalg.norm(helio),
                                   delta=VEL_RTOL*np.linalg.norm(helioV))
        # Offsets are measured from the shifted prediction.
        np.testing.assert_allclose(sss['ephOffsetDec'], (sss['dec'] - sss['ephDec'])*3600, atol=1e-9)
        self.assertEqual(task.metadata['nSsoShutterEpochs'], len(ssObjects))
        self.assertAlmostEqual(task.metadata['ssoShutterEpochMaxShift'], dtSec, places=5)

    def testPositionDependent(self):
        """Each object uses the time of its own predicted pixel."""
        def offset(x, y):
            return 0.25*(x - 2036.0)/2036.0 - 0.1*y/4000.0

        timing = FakeShutterTiming(offset)
        ssObjects = _makeMpSkyObjects(self.wcs)
        task, result = self._run(ssObjects, timing)
        x, y = timing.calls[0]
        visRa, visDec = _expectedMpSkyRaDec(ssObjects, T_VISIT)
        expX, expY = self.wcs.skyToPixelArray(visRa, visDec, degrees=True)
        np.testing.assert_allclose(x, expX, rtol=0, atol=1e-6)
        np.testing.assert_allclose(y, expY, rtol=0, atol=1e-6)
        self.assertGreater(np.ptp(offset(x, y)), 0.2)
        epochs = T_VISIT + offset(x, y)/SEC_PER_DAY
        expRa, expDec = _expectedMpSkyRaDec(ssObjects, epochs)
        ids, ra, dec = self._ephByObject(result)
        np.testing.assert_allclose(ra, expRa, rtol=0, atol=EPH_ATOL_DEG)
        np.testing.assert_allclose(dec, expDec, rtol=0, atol=EPH_ATOL_DEG)
        self.assertAlmostEqual(task.metadata['ssoShutterEpochMaxShift'],
                               np.max(np.abs(offset(x, y))), places=5)

    def testUnavailableKeepsVisitTime(self):
        """Objects whose pixel is UNAVAILABLE (or NaN) keep the visit time."""
        def unavailable(x, y):
            return x > 3000.0

        ssObjects = _makeMpSkyObjects(self.wcs)
        task, result = self._run(ssObjects, FakeShutterTiming(lambda x, y: 0.4, unavailable))
        bad = np.array([o[0] > 3000.0 for o in OBJECTS])
        epochs = np.where(bad, T_VISIT, T_VISIT + 0.4/SEC_PER_DAY)
        expRa, expDec = _expectedMpSkyRaDec(ssObjects, epochs)
        ids, ra, dec = self._ephByObject(result)
        np.testing.assert_allclose(ra, expRa, rtol=0, atol=EPH_ATOL_DEG)
        np.testing.assert_allclose(dec, expDec, rtol=0, atol=EPH_ATOL_DEG)
        visRa, visDec = _expectedMpSkyRaDec(ssObjects, T_VISIT)
        np.testing.assert_allclose(ra[bad], visRa[bad], rtol=0, atol=EPH_ATOL_DEG)
        self.assertEqual(task.metadata['nSsoShutterEpochs'], np.count_nonzero(~bad))

        # A status of UNAVAILABLE with a finite time also keeps the visit time.
        class StatusOnly(FakeShutterTiming):
            def tMidMjdTai(self, x, y):
                return np.full(np.shape(x), T_VISIT + 0.4/SEC_PER_DAY)

        task, result2 = self._run(ssObjects, StatusOnly(lambda x, y: 0.4, unavailable))
        np.testing.assert_array_equal(self._ephByObject(result2)[1], ra)

    def testFastMoverAlongTrack(self):
        """A 10 deg/day NEA moves along its track by rate × Δt."""
        dtSec = 0.45
        ssObjects = _makeMpSkyObjects(self.wcs)
        _, none = self._run(ssObjects, None)
        _, shifted = self._run(ssObjects, FakeShutterTiming(lambda x, y: dtSec))
        k = 2  # the NEA
        x, y, rho, rate, pa = OBJECTS[k]
        ids, ra0, dec0 = self._ephByObject(none)
        _, ra1, dec1 = self._ephByObject(shifted)
        dRa = (ra1[k] - ra0[k])*np.cos(np.radians(dec0[k]))*3600
        dDec = (dec1[k] - dec0[k])*3600
        expected = rate*3600*dtSec/SEC_PER_DAY   # arcsec
        self.assertAlmostEqual(np.hypot(dRa, dDec), expected, delta=1e-3*expected)
        along = dRa*np.sin(np.radians(pa)) + dDec*np.cos(np.radians(pa))
        cross = -dRa*np.cos(np.radians(pa)) + dDec*np.sin(np.radians(pa))
        self.assertAlmostEqual(along, expected, delta=1e-3*expected)
        self.assertLess(abs(cross), 1e-3*expected)

    def testSorchaLinearShift(self):
        """Without polynomials the prediction moves by rate × Δt."""
        dtSec = 0.45
        ssObjects = _makeSorchaObjects(self.wcs)
        task, result = self._run(ssObjects, FakeShutterTiming(lambda x, y: dtSec))
        ids, ra, dec = self._ephByObject(result)
        order = np.argsort(ssObjects['ObjID'])
        dt = dtSec/SEC_PER_DAY
        ra0 = np.array(ssObjects['RATrue_deg'])
        dec0 = np.array(ssObjects['DecTrue_deg'])
        expRa = ra0 + np.array(ssObjects['RARateCosDec_deg_day'])*dt/np.cos(np.radians(dec0))
        expDec = dec0 + np.array(ssObjects['DecRate_deg_day'])*dt
        np.testing.assert_allclose(ra, expRa[order], rtol=0, atol=EPH_ATOL_DEG)
        np.testing.assert_allclose(dec, expDec[order], rtol=0, atol=EPH_ATOL_DEG)
        self.assertEqual(task.metadata['nSsoShutterEpochs'], len(ssObjects))

    def testSorchaRaWrap(self):
        """Linear shifts across RA = 0 stay in [0, 360) and move by
        rate × Δt on the sky.
        """
        dtSec = 0.45
        wcs = afwGeom.makeSkyWcs(
            crpix=lsst.geom.Point2D(2036.0, 2000.0),
            crval=lsst.geom.SpherePoint(0.0, 0.0, lsst.geom.degrees),
            cdMatrix=afwGeom.makeCdMatrix(scale=0.2*lsst.geom.arcseconds),
        )
        ssObjects = _makeSorchaObjects(wcs, OBJECTS[:2])
        rate = 10.0  # deg/day
        ssObjects['RATrue_deg'] = [359.99998, 0.00002] * u.deg
        ssObjects['DecTrue_deg'] = [0.0, 0.0] * u.deg
        ssObjects['RARateCosDec_deg_day'] = [rate, -rate]   # east, west
        ssObjects['DecRate_deg_day'] = [0.0, 0.0]
        ra0 = np.array(ssObjects['RATrue_deg'])
        dec0 = np.array(ssObjects['DecTrue_deg'])
        task, result = self._run(ssObjects, FakeShutterTiming(lambda x, y: dtSec), wcs=wcs,
                                 diaRaDec=(ra0, dec0))
        self.assertEqual(task.metadata['nSsoShutterEpochs'], 2)
        ids, ra, dec = self._ephByObject(result)
        order = np.argsort(ssObjects['ObjID'])
        ra0, dec0 = ra0[order], dec0[order]
        self.assertTrue(np.all((ra >= 0.0) & (ra < 360.0)), ra)
        # Both cross RA = 0.
        self.assertLess(ra[0], 1.0)
        self.assertGreater(ra[1], 359.0)
        sep = SkyCoord(ra0*u.deg, dec0*u.deg).separation(SkyCoord(ra*u.deg, dec*u.deg)).deg
        np.testing.assert_allclose(sep, rate*dtSec/SEC_PER_DAY, rtol=0, atol=EPH_ATOL_DEG)

    def _sorchaExpected(self, ssObjects, dtSec):
        """Sorcha positions moved linearly by ``dtSec`` (scalar or per
        object), ordered as `_ephByObject`.
        """
        order = np.argsort(ssObjects['ObjID'])
        dt = np.asarray(dtSec)/SEC_PER_DAY
        ra0 = np.array(ssObjects['RATrue_deg'])
        dec0 = np.array(ssObjects['DecTrue_deg'])
        expRa = ra0 + np.array(ssObjects['RARateCosDec_deg_day'])*dt/np.cos(np.radians(dec0))
        expDec = dec0 + np.array(ssObjects['DecRate_deg_day'])*dt
        return expRa[order], expDec[order]

    def testSorchaFieldTime(self):
        """Sorcha positions are moved from their own ``fieldMJD_TAI``, not
        from ``visitInfo.date``.
        """
        dtSec = 0.45
        fieldOffsetSec = np.array([-2.0, -1.0, 0.5, 1.5])
        ssObjects = _makeSorchaObjects(self.wcs)
        ssObjects['fieldMJD_TAI'] = T_VISIT + fieldOffsetSec/SEC_PER_DAY
        task, result = self._run(ssObjects, FakeShutterTiming(lambda x, y: dtSec))
        expRa, expDec = self._sorchaExpected(ssObjects, dtSec - fieldOffsetSec)
        ids, ra, dec = self._ephByObject(result)
        # MJD rounding limits the agreement to ~1e-11 deg; the field-time
        # offsets themselves move the NEA by ~1e-4 deg.
        np.testing.assert_allclose(ra, expRa, rtol=0, atol=EPH_ATOL_DEG)
        np.testing.assert_allclose(dec, expDec, rtol=0, atol=EPH_ATOL_DEG)
        # The shift metadata is still relative to visitInfo.date.
        self.assertAlmostEqual(task.metadata['ssoShutterEpochMaxShift'], dtSec, places=5)

    def testSorchaWithoutFieldTime(self):
        """Without ``fieldMJD_TAI`` Sorcha positions are moved from
        ``visitInfo.date``.
        """
        dtSec = 0.45
        ssObjects = _makeSorchaObjects(self.wcs)
        ssObjects.remove_column('fieldMJD_TAI')
        task, result = self._run(ssObjects, FakeShutterTiming(lambda x, y: dtSec))
        expRa, expDec = self._sorchaExpected(ssObjects, dtSec)
        ids, ra, dec = self._ephByObject(result)
        np.testing.assert_allclose(ra, expRa, rtol=0, atol=EPH_ATOL_DEG)
        np.testing.assert_allclose(dec, expDec, rtol=0, atol=EPH_ATOL_DEG)
        self.assertEqual(task.metadata['nSsoShutterEpochs'], len(ssObjects))

    def _assertIgnored(self, ssObjects, timing, detector=None):
        _, none = self._run(ssObjects, None, detector=detector)
        task, result = self._run(ssObjects, timing, detector=detector)
        self._compareResults(none, result)
        self.assertTrue(task.metadata['ssoShutterTimingRejected'])
        self.assertEqual(task.metadata['nSsoShutterEpochs'], 0)
        self.assertEqual(task.metadata['ssoShutterEpochMaxShift'], 0.0)
        self.assertEqual(timing.calls, [])

    def _assertUsed(self, ssObjects, timing, detector=None):
        task, _ = self._run(ssObjects, timing, detector=detector)
        self.assertFalse(task.metadata['ssoShutterTimingRejected'])
        self.assertEqual(task.metadata['nSsoShutterEpochs'], len(ssObjects))

    def testTimingOfAnotherExposureIgnored(self):
        """A timing whose header midpoint is not ``visitInfo.date`` is
        ignored with a warning.
        """
        for make in (_makeMpSkyObjects, _makeSorchaObjects):
            ssObjects = make(self.wcs)
            with self.assertLogs(level='WARNING') as cm:
                self._assertIgnored(ssObjects, FakeShutterTiming(lambda x, y: 0.3,
                                                                 headerMidMjdTai=T_VISIT + 35/SEC_PER_DAY))
            self.assertIn("header midpoint", "\n".join(cm.output))
            # Within the tolerance: used.
            self._assertUsed(ssObjects, FakeShutterTiming(lambda x, y: 0.3,
                                                          headerMidMjdTai=T_VISIT + 0.5/SEC_PER_DAY))

    def testFocalPlaneTimeCheck(self):
        """Without a header midpoint, the focal-plane time is checked within
        a window wide enough for late readouts.
        """
        ssObjects = _makeMpSkyObjects(self.wcs)
        # A late readout: visitInfo.date 600 s after the shutter time.
        self._assertUsed(ssObjects, FakeShutterTiming(lambda x, y: 0.3, headerMidMjdTai=np.nan,
                                                      focalPlaneMjdTai=T_VISIT - 600/SEC_PER_DAY))
        self._assertIgnored(ssObjects, FakeShutterTiming(lambda x, y: 0.3, headerMidMjdTai=np.nan,
                                                         focalPlaneMjdTai=T_VISIT - 3600/SEC_PER_DAY))
        # Configurable.
        task = SolarSystemAssociationTask(config=SolarSystemAssociationConfig(
            shutterTimingMaxFocalPlaneOffset=100.0))
        timing = FakeShutterTiming(lambda x, y: 0.3, headerMidMjdTai=np.nan,
                                   focalPlaneMjdTai=T_VISIT - 600/SEC_PER_DAY)
        self.assertIsNone(task._checkShutterTiming(timing, self.visitInfo, _makeDiaSources([150.0], [10.0])))
        self.assertTrue(task.metadata['ssoShutterTimingRejected'])
        # Neither time known (UNAVAILABLE without cards): not rejected.
        timing = FakeShutterTiming(lambda x, y: 0.3, headerMidMjdTai=np.nan, focalPlaneMjdTai=np.nan)
        self.assertIs(task._checkShutterTiming(timing, self.visitInfo, _makeDiaSources([150.0], [10.0])),
                      timing)

    def testTimingOfAnotherDetectorIgnored(self):
        """A timing for another detector than the DiaSources' is ignored."""
        ssObjects = _makeMpSkyObjects(self.wcs)
        with self.assertLogs(level='WARNING') as cm:
            self._assertIgnored(ssObjects, FakeShutterTiming(lambda x, y: 0.3, detectorId=94), detector=93)
        self.assertIn("detector 94", "\n".join(cm.output))
        self._assertUsed(ssObjects, FakeShutterTiming(lambda x, y: 0.3, detectorId=94), detector=94)
        # No detector column: not checked.
        self._assertUsed(ssObjects, FakeShutterTiming(lambda x, y: 0.3, detectorId=94))


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
