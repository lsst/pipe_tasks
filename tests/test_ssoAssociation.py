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

import copy
import json
import os
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
from lsst.pipe.tasks.associationUtils import obj_id_to_ss_object_id
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
# Shutter-corrected epochs (``run(..., shutterTiming=...)``), against
# JPL Horizons ephemerides (tests/data/ssoShutterEpochs/README.md)
# ---------------------------------------------------------------
HORIZONS_FIXTURE = os.path.join(os.path.dirname(__file__), "data", "ssoShutterEpochs", "horizons.json")
M_PER_AU = 1.495978707e11
MJD_ATOL = 2e-11       # 2 us; MJD 61212 has a 7e-12 d ulp
# Tolerances are SAFETY times the error expected from the physics (stored in
# the fixture: the Chebyshev fit residual for mpSky, the error of a linear
# move for Sorcha), plus a floor for the ~1 us float64 resolution of MJD
# times (a few cm, a few uas at these objects' speeds) and platform noise.
SAFETY = 3.0
FLOOR = dict(skyMas=0.01, helioM=0.1, topoM=0.1, rangeM=0.1, phaseDeg=1e-8,
             helioVelocityMmS=1.0, topoVelocityMmS=1.0)


def _loadHorizonsObjects():
    """Load the objects of the Horizons fixture (`list` [`dict`])."""
    with open(HORIZONS_FIXTURE) as f:
        return json.load(f)["objects"]


def _makeWcs(ra, dec):
    """Make a TAN WCS with 0.2"/px for an LSSTCam-sized detector, centred
    on ``ra``, ``dec`` (degrees).
    """
    return afwGeom.makeSkyWcs(crpix=lsst.geom.Point2D(2036.0, 2000.0),
                              crval=lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees),
                              cdMatrix=afwGeom.makeCdMatrix(scale=0.2*lsst.geom.arcseconds))


def _makeMpSkyObjects(obj):
    """Make the mpSky ``ssObjects`` of one fixture object.

    Parameters
    ----------
    obj : `dict`
        Fixture object.

    Returns
    -------
    ssObjects : `astropy.table.Table`
        One row: Chebyshev coefficients (au, days since ``tmin``) of the
        object and observer heliocentric positions.
    """
    mpSky = obj["mpSky"]
    table = Table({
        'ObjID': [obj["objId"]],
        'ra': [obj["sorcha"]["RATrue_deg"]] * u.deg, 'dec': [obj["sorcha"]["DecTrue_deg"]] * u.deg,
        'tmin': [mpSky["tmin"]] * u.d, 'tmax': [mpSky["tmax"]] * u.d,
        'trailedSourceMagTrue': [obj["sorcha"]["trailedSourceMagTrue"]] * u.mag,
        'Err(arcsec)': [2.0] * u.arcsec,
        'ssObjectId': [obj_id_to_ss_object_id(obj["packedDesig"])],
        'MPCORB_unpacked_primary_provisional_designation': [obj["unpackedDesig"]],
    })
    for i, c in enumerate('xyz'):
        table[f'obj_{c}_poly'] = [mpSky["objPoly"][i]]
        table[f'obs_{c}_poly'] = [mpSky["obsPoly"][i]]

    return table


def _makeSorchaObjects(obj):
    """Make the Sorcha ``ssObjects`` (one row) of one fixture object, with
    units on the angle, distance and velocity columns that carry them.
    """
    table = Table({name: [value] for name, value in obj["sorcha"].items()})
    for name in table.colnames:
        if name.endswith('_km'):
            table[name].unit = u.km
        elif name.endswith('_km_s'):
            table[name].unit = u.km/u.s
    for name in ('RATrue_deg', 'DecTrue_deg'):
        table[name].unit = u.deg

    return table


def _turnAboutPole(obj, angle):
    """Return a copy of a fixture object turned by ``angle`` (degrees) about
    the celestial pole, in its Sorcha row and truth: RA moves by ``angle``,
    while Dec, distances, rates and the phase angle stay.
    """
    obj = copy.deepcopy(obj)
    cos, sin = np.cos(np.radians(angle)), np.sin(np.radians(angle))

    def turn(x, y):
        return cos*x - sin*y, sin*x + cos*y

    sorcha, truth = obj["sorcha"], obj["truth"]
    sorcha["RATrue_deg"] = (sorcha["RATrue_deg"] + angle) % 360.0
    truth["ephRa"] = (truth["ephRa"] + angle) % 360.0
    for prefix, suffix in [('Obj_Sun_', '_LTC_km'), ('Obj_Sun_v', '_LTC_km_s'),
                           ('Obs_Sun_', '_km'), ('Obs_Sun_v', '_km_s')]:
        x, y = f'{prefix}x{suffix}', f'{prefix}y{suffix}'
        sorcha[x], sorcha[y] = turn(sorcha[x], sorcha[y])
    for name in ('helio', 'helioVelocity', 'topo', 'topoVelocity'):
        truth[name][0], truth[name][1] = turn(truth[name][0], truth[name][1])

    return obj


def _tolerances(expected):
    """Compute tolerances (same keys as ``FLOOR``) from the expected errors
    ``expected`` (`dict`, same keys and units).
    """
    return {key: SAFETY*expected[key] + floor for key, floor in FLOOR.items()}


def _makeDiaSources(ra, dec, offsetArcsec):
    """Make one DiaSource ``offsetArcsec`` north of ``ra``, ``dec``
    (degrees).
    """
    return Table({
        'diaSourceId': np.array([1], dtype=np.int64), 'midpointMjdTai': [0.0],
        'ra': [ra], 'dec': [dec + offsetArcsec/3600], 'extendedness': [0.0], 'band': ['r'],
        'psfFlux': [1000.0], 'psfFluxErr': [10.0],
    })


class FakeShutterTiming:
    """Stand-in for `lsst.ip.isr.shutterTiming.ShutterTiming`: the time
    ``midpointMjdTai`` everywhere, or NaN everywhere (an UNAVAILABLE
    detector) by default.
    """

    def __init__(self, midpointMjdTai=np.nan):
        self.time = midpointMjdTai

    def midpointMjdTai(self, x, y):
        """Return times (MJD TAI) at pixel positions ``x``, ``y``
        (arrays).
        """
        return np.full(np.shape(x), self.time)


class TestShutterTimingEpochs(lsst.utils.tests.TestCase):
    """Per-object ephemeris epochs from a shutter timing, compared with JPL
    Horizons at those epochs.
    """

    def setUp(self):
        self.objects = _loadHorizonsObjects()
        self.bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(4072, 4000))

    def _run(self, obj, ssObjects, shutterTiming=None, offsetArcsec=0.05):
        """Run the task on a detector centred on the object, with a
        DiaSource ``offsetArcsec`` north of its true position at the
        shutter-corrected time (beyond the matching radius, nothing
        associates).
        """
        truth = obj["truth"]
        visitInfo = afwImage.VisitInfo(date=dafBase.DateTime(obj["visitMjdTai"], dafBase.DateTime.MJD,
                                                             dafBase.DateTime.TAI))
        diaSources = _makeDiaSources(truth["ephRa"], truth["ephDec"], offsetArcsec)

        return SolarSystemAssociationTask().run(diaSources, ssObjects, visitInfo, self.bbox,
                                                _makeWcs(truth["ephRa"], truth["ephDec"]),
                                                shutterTiming=shutterTiming)

    def _assertMatchesTruth(self, obj, make, tolerance):
        """Check that the unassociated prediction of ``obj`` (``ssObjects``
        from ``make(obj)``) is the Horizons truth at its shutter-corrected
        time, within ``tolerance``, and that its offset from a DiaSource is
        measured from that prediction.
        """
        timing = FakeShutterTiming(obj["shutterMjdTai"])
        truth = obj["truth"]

        # Epoch.
        unassociated = self._run(obj, make(obj), timing, offsetArcsec=100.0).unassociatedSsObjects
        self.assertEqual(len(unassociated), 1)
        got = unassociated[0]
        self.assertAlmostEqual(got['midpointMjdTai'], obj["shutterMjdTai"], delta=MJD_ATOL)

        # Sky position (mas).
        cosDec = np.cos(np.radians(truth["ephDec"]))
        self.assertLess(np.hypot((got['ephRa'] - truth["ephRa"])*cosDec, got['ephDec'] - truth["ephDec"])
                        * 3.6e6, tolerance['skyMas'])

        # State vectors (au, km/s), range (au) and phase angle (deg).
        for column, key, toleranceKey, unit in [
                ('helio_{}', 'helio', 'helioM', M_PER_AU), ('topo_{}', 'topo', 'topoM', M_PER_AU),
                ('helio_v{}', 'helioVelocity', 'helioVelocityMmS', 1e6),
                ('topo_v{}', 'topoVelocity', 'topoVelocityMmS', 1e6)]:
            np.testing.assert_allclose([got[column.format(c)] for c in 'xyz'], truth[key], rtol=0,
                                       atol=tolerance[toleranceKey]/unit, err_msg=key)
        self.assertAlmostEqual(got['topoRange'], truth["topoRange"], delta=tolerance['rangeM']/M_PER_AU)
        self.assertAlmostEqual(got['phaseAngle'], truth["phaseAngle"], delta=tolerance['phaseDeg'])

        # Associated: the offset of a DiaSource 50 mas north of the truth is
        # measured from the moved prediction.
        sss = self._run(obj, make(obj), timing).associatedSsSources
        self.assertEqual(len(sss), 1)
        self.assertAlmostEqual(sss['ephOffsetRa'][0]*1e3, 0.0, delta=tolerance['skyMas'])
        self.assertAlmostEqual(sss['ephOffsetDec'][0]*1e3, 50.0, delta=tolerance['skyMas'])

    def testChebyshevShift(self):
        """Check that every predicted quantity of an mpSky object, evaluated
        again at its shutter-corrected time, is the Horizons truth there
        within the fit residual, and that unassociated predictions record
        that time.
        """
        for obj in self.objects:
            with self.subTest(obj["name"]):
                # Errors expected from the fit: topocentric quantities
                # combine the object and observer fits, and the phase angle
                # moves by the position error over the topocentric distance.
                fit = obj["fitResidual"]
                expected = dict(skyMas=fit["skyMas"], helioM=fit["positionM"], topoM=2*fit["positionM"],
                                rangeM=2*fit["positionM"], helioVelocityMmS=fit["velocityMmS"],
                                topoVelocityMmS=2*fit["velocityMmS"],
                                phaseDeg=np.degrees(2*fit["positionM"]/(obj["truth"]["topoRange"]*M_PER_AU)))
                self._assertMatchesTruth(obj, _makeMpSkyObjects, _tolerances(expected))

    def testSorchaShift(self):
        """Check that Sorcha predictions moved linearly from their own
        ``fieldMJD_TAI`` to the shutter-corrected time are the Horizons truth
        there within the error of a linear move, RA staying in [0, 360)
        across RA = 0, and that unassociated predictions record that time.
        """
        # 2025 WC4 turned to cross RA = 0 westward between field and
        # shutter-corrected times.
        fast = self.objects[-1]
        crossing = _turnAboutPole(fast, 0.4/3600 - fast["sorcha"]["RATrue_deg"])
        crossing["name"] += " turned to RA 0"
        self.assertTrue(crossing["sorcha"]["RATrue_deg"] < 1.0 and crossing["truth"]["ephRa"] > 359.0)

        for obj in self.objects + [crossing]:
            with self.subTest(obj["name"]):
                self._assertMatchesTruth(obj, _makeSorchaObjects, _tolerances(obj["sorchaLinearError"]))

    def testUnavailableKeepsVisitTime(self):
        """Check that a detector without corrected times gives exactly the
        result without timing, for both ephemeris sources, and that
        unassociated predictions record the ephemeris epoch.
        """
        for obj in self.objects:
            for make, epoch in [(_makeMpSkyObjects, obj["visitMjdTai"]),
                                (_makeSorchaObjects, obj["sorcha"]["fieldMJD_TAI"])]:
                with self.subTest(obj["name"], source=make.__name__):
                    for offsetArcsec in (0.05, 100.0):
                        none = self._run(obj, make(obj), offsetArcsec=offsetArcsec)
                        unav = self._run(obj, make(obj), FakeShutterTiming(), offsetArcsec=offsetArcsec)

                        for name in ['ssoAssocDiaSources', 'unAssocDiaSources', 'associatedSsSources',
                                     'unassociatedSsObjects']:
                            a, b = getattr(none, name), getattr(unav, name)
                            self.assertEqual(a.colnames, b.colnames)
                            for c in a.colnames:
                                self.assertEqual(np.asarray(a[c]).tobytes(), np.asarray(b[c]).tobytes(), c)

                    # With nothing associated, the prediction is at its epoch.
                    self.assertEqual(len(none.unassociatedSsObjects), 1)
                    self.assertAlmostEqual(none.unassociatedSsObjects['midpointMjdTai'][0], epoch,
                                           delta=MJD_ATOL)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
