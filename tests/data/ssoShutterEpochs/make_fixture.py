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

"""Make ``horizons.json``, the JPL Horizons fixture of the shutter-timing
tests in ``tests/test_ssoAssociation.py``.

Usage::

    python make_fixture.py [--cache DIR]

Horizons responses are cached in ``DIR`` (one JSON file per query), so a
rerun with the cache in place makes no network requests.  Queries are made
one at a time, at least 1.5 s apart.  See ``README.md`` for the contents.
"""

import argparse
import datetime
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
from numpy.polynomial import chebyshev
from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
QUERY_INTERVAL = 1.5
"""Minimum time between two Horizons requests (s)."""

AU_KM = 149597870.700
SEC_PER_DAY = 86400.0
SITE = EarthLocation.from_geodetic(lon=289.2506*u.deg, lat=-30.244623*u.deg, height=2693.92*u.m)
"""Rubin Observatory (MPC code X05), geodetic coordinates as Horizons
reports them for the site."""

WINDOW_HOURS = 3.0
"""The mpSky polynomials are fitted from 3 h before to 3 h after the visit."""
STEP_MINUTES = 10.0
CHEBYSHEV_DEGREE = 10
"""Degree of the mpSky polynomials; the fit reaches the float64 resolution of
the MJD times (~1 us, a few cm) from degree 10 on."""

OBJECTS = [
    dict(name="(1232) Cortusa", kind="main-belt asteroid", command="1232;",
         objId="1232", packedDesig="J31T02F", unpackedDesig="1931 TF2", H=10.31, G=0.15,
         visitUtc="2026-06-21T05:00:00", shutterOffset=-0.18, fieldOffset=2.5),
    dict(name="(17182) 1999 VU", kind="near-Earth asteroid", command="17182;",
         objId="17182", packedDesig="J99V00U", unpackedDesig="1999 VU", H=17.13, G=0.15,
         visitUtc="2026-06-21T08:00:00", shutterOffset=0.27, fieldOffset=-1.5),
    dict(name="2025 WC4", kind="near-Earth asteroid at close approach", command="DES=2025 WC4;",
         objId="2025 WC4", packedDesig="K25W04C", unpackedDesig="2025 WC4", H=20.33, G=0.15,
         visitUtc="2026-06-21T00:30:00", shutterOffset=-0.29, fieldOffset=3.0),
]
"""Objects of the fixture: ``shutterOffset`` is the shutter-corrected time
of the object's pixel and ``fieldOffset`` the Sorcha ``fieldMJD_TAI``, both
in seconds from the visit time."""


class Horizons:
    """Cached, serial client of the JPL Horizons API.

    Parameters
    ----------
    cacheDir : `str`
        Directory of the cached responses.
    """

    def __init__(self, cacheDir: str):
        self.cacheDir = cacheDir
        self.nQueries = 0
        self._last = 0.0

    def vectors(self, command: str, center: str, jdTdb: np.ndarray) -> np.ndarray:
        """Query geometric ICRF state vectors at TDB epochs.

        Parameters
        ----------
        command : `str`
            Horizons target (``COMMAND``).
        center : `str`
            Horizons origin (``CENTER``).
        jdTdb : `numpy.ndarray`
            Epochs (JD TDB).  Horizons keeps epochs as a double JD, so these
            are sent as the shortest decimal strings of the doubles and are
            evaluated exactly at those values.

        Returns
        -------
        states : `numpy.ndarray`, (N, 6)
            Positions (km) and velocities (km/s) at ``jdTdb``.
        """
        # Request a vector table in km and km/s: 16 significant digits are
        # micrometre resolution at 1e8 km.
        params = dict(
            format="json", COMMAND=f"'{command}'", CENTER=f"'{center}'", EPHEM_TYPE="VECTORS",
            TLIST=" ".join(f"'{float(t)!r}'" for t in jdTdb), TLIST_TYPE="JD", TIME_TYPE="TDB",
            REF_PLANE="FRAME", REF_SYSTEM="ICRF", VEC_CORR="NONE", OUT_UNITS="KM-S",
            VEC_TABLE="2", CSV_FORMAT="YES", VEC_LABELS="NO", OBJ_DATA="NO",
        )
        result = self._query(params)

        # Rows come sorted by time; match each epoch to its printed JD.
        body = result[result.index("$$SOE") + 5:result.index("$$EOE")]
        rows = [line.split(",") for line in body.strip().splitlines()]
        printedJd = np.array([float(row[0]) for row in rows])
        values = np.array([[float(v) for v in row[2:8]] for row in rows])
        index = np.argmin(np.abs(printedJd[np.newaxis, :] - jdTdb[:, np.newaxis]), axis=1)
        if len(rows) != len(jdTdb) or np.max(np.abs(printedJd[index] - jdTdb)) > 1e-9:
            raise RuntimeError(f"Horizons epochs do not match the request for {command}")

        return values[index]

    def _query(self, params: dict) -> str:
        """Return the ``result`` text of a query, from the cache if there."""
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        path = os.path.join(self.cacheDir, f"horizons-{key}.json")
        if not os.path.exists(path):
            time.sleep(max(0.0, self._last + QUERY_INTERVAL - time.time()))
            with urllib.request.urlopen(HORIZONS_URL + "?" + urllib.parse.urlencode(params),
                                        timeout=300) as response:
                data = json.loads(response.read())
            self._last = time.time()
            self.nQueries += 1
            data["request"] = params
            with open(path, "w") as f:
                json.dump(data, f)
        with open(path) as f:
            return json.load(f)["result"]


def raDec(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute RA, Dec (degrees) of the directions of (N, 3) vectors."""
    ra = np.degrees(np.arctan2(vector[:, 1], vector[:, 0])) % 360.0
    dec = np.degrees(np.arctan2(vector[:, 2], np.hypot(vector[:, 0], vector[:, 1])))
    return ra, dec


def skyRates(topo: np.ndarray, topoVelocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the sky rates (deg/day) of RA cos(Dec) and Dec.

    Parameters
    ----------
    topo : `numpy.ndarray`, (N, 3)
        Topocentric positions (km).
    topoVelocity : `numpy.ndarray`, (N, 3)
        Topocentric velocities (km/s).

    Returns
    -------
    raRateCosDec, decRate : `numpy.ndarray`
        Rates of the direction of ``topo`` (deg/day).
    """
    ra, dec = np.radians(raDec(topo))
    east = np.column_stack([-np.sin(ra), np.cos(ra), np.zeros_like(ra)])
    north = np.column_stack([-np.sin(dec)*np.cos(ra), -np.sin(dec)*np.sin(ra), np.cos(dec)])
    distance = np.linalg.norm(topo, axis=1)
    toDegDay = np.degrees(SEC_PER_DAY)/distance

    return (np.sum(east*topoVelocity, axis=1)*toDegDay,
            np.sum(north*topoVelocity, axis=1)*toDegDay)


def phaseAngle(helio: np.ndarray, topo: np.ndarray) -> np.ndarray:
    """Compute the phase angle (degrees) from heliocentric and topocentric
    positions, (N, 3) each.
    """
    cosPhase = np.sum(helio*topo, axis=1)/np.linalg.norm(helio, axis=1)/np.linalg.norm(topo, axis=1)
    return np.degrees(np.arccos(cosPhase))


def apparentMagnitude(H: float, G: float, helio: np.ndarray, topo: np.ndarray) -> float:
    """Compute the V magnitude in the H, G system (positions in km, (3,))."""
    alpha = np.radians(phaseAngle(helio[np.newaxis], topo[np.newaxis])[0])
    phi1 = np.exp(-3.33*np.tan(alpha/2)**0.63)
    phi2 = np.exp(-1.87*np.tan(alpha/2)**1.22)
    r, delta = np.linalg.norm(helio)/AU_KM, np.linalg.norm(topo)/AU_KM
    return H + 5*np.log10(r*delta) - 2.5*np.log10((1 - G)*phi1 + G*phi2)


def taiMjd(jdTdb: np.ndarray) -> np.ndarray:
    """Convert JD TDB at the Rubin site to MJD TAI."""
    return Time(jdTdb, format="jd", scale="tdb", location=SITE).tai.mjd


def makeObject(horizons: Horizons, spec: dict) -> dict:
    """Make the fixture entry of one object.

    Parameters
    ----------
    horizons : `Horizons`
        Horizons client.
    spec : `dict`
        Entry of ``OBJECTS``.

    Returns
    -------
    entry : `dict`
        Visit epoch, truth at the shutter-corrected time, mpSky polynomials,
        Sorcha row, fit residuals and Sorcha linear-shift errors.
    """
    # Epochs (JD TDB doubles): the fit grid, then visit, shutter and field.
    visit = Time(spec["visitUtc"], scale="utc", location=SITE).tdb.jd
    nGrid = int(round(2*WINDOW_HOURS*60/STEP_MINUTES)) + 1
    grid = visit - WINDOW_HOURS/24 + np.arange(nGrid)*STEP_MINUTES/1440
    special = visit + np.array([0.0, spec["shutterOffset"], spec["fieldOffset"]])/SEC_PER_DAY
    jdTdb = np.concatenate([grid, special])
    mjdTai = taiMjd(jdTdb)

    # Geometric state vectors: object and observer heliocentric (the
    # observer is minus the Sun seen from X05), and the object from X05 as a
    # check of the observer.
    helio = horizons.vectors(spec["command"], "500@10", jdTdb)
    observer = -horizons.vectors("10", "X05@399", jdTdb)
    topoCheck = horizons.vectors(spec["command"], "X05@399", jdTdb)
    topo = helio - observer
    observerCheckM = np.max(np.linalg.norm(topo[:, :3] - topoCheck[:, :3], axis=1))*1e3

    # mpSky polynomials: positions (au) in days since tmin (MJD TAI).
    tmin, tmax = mjdTai[0], mjdTai[nGrid - 1]
    t = mjdTai - tmin
    objPoly = chebyshev.chebfit(t[:nGrid], helio[:nGrid, :3]/AU_KM, CHEBYSHEV_DEGREE)
    obsPoly = chebyshev.chebfit(t[:nGrid], observer[:nGrid, :3]/AU_KM, CHEBYSHEV_DEGREE)

    # Fit residuals, on the grid and at the three out-of-grid epochs.
    def evaluate(poly, order=0):
        return chebyshev.chebval(t, chebyshev.chebder(poly, order)).T

    fitHelio, fitObserver = evaluate(objPoly)*AU_KM, evaluate(obsPoly)*AU_KM
    fitHelioV = evaluate(objPoly, 1)*AU_KM/SEC_PER_DAY
    fitObserverV = evaluate(obsPoly, 1)*AU_KM/SEC_PER_DAY
    fitRa, fitDec = raDec(fitHelio - fitObserver)
    ra, dec = raDec(topo[:, :3])
    skyMas = np.hypot((fitRa - ra)*np.cos(np.radians(dec)), fitDec - dec)*3.6e6
    residuals = dict(
        positionM=float(max(np.max(np.linalg.norm(fitHelio - helio[:, :3], axis=1)),
                            np.max(np.linalg.norm(fitObserver - observer[:, :3], axis=1)))*1e3),
        velocityMmS=float(max(np.max(np.linalg.norm(fitHelioV - helio[:, 3:], axis=1)),
                              np.max(np.linalg.norm(fitObserverV - observer[:, 3:], axis=1)))*1e6),
        skyMas=float(np.max(skyMas)),
    )

    # Truth at the shutter-corrected time (index -2).
    k = len(jdTdb) - 2
    truthRa, truthDec = raDec(topo[k:k + 1, :3])
    truth = dict(
        ephRa=float(truthRa[0]), ephDec=float(truthDec[0]),
        helio=(helio[k, :3]/AU_KM).tolist(), helioVelocity=helio[k, 3:].tolist(),
        topo=(topo[k, :3]/AU_KM).tolist(), topoVelocity=topo[k, 3:].tolist(),
        topoRange=float(np.linalg.norm(topo[k, :3])/AU_KM),
        phaseAngle=float(phaseAngle(helio[k:k + 1, :3], topo[k:k + 1, :3])[0]),
    )

    # Sorcha row at the field time (index -1), geometric.
    f = len(jdTdb) - 1
    fieldRa, fieldDec = raDec(topo[f:, :3])
    raRate, decRate = skyRates(topo[f:, :3], topo[f:, 3:])
    sorcha = dict(
        ObjID=spec["objId"], packed_desig=spec["packedDesig"], FieldID=0,
        fieldMJD_TAI=float(mjdTai[f]), fieldJD_TDB=float(jdTdb[f]),
        RATrue_deg=float(fieldRa[0]), DecTrue_deg=float(fieldDec[0]),
        RARateCosDec_deg_day=float(raRate[0]), DecRate_deg_day=float(decRate[0]),
        Range_LTC_km=float(np.linalg.norm(topo[f, :3])),
        RangeRate_LTC_km_s=float(np.dot(topo[f, :3], topo[f, 3:])/np.linalg.norm(topo[f, :3])),
        phase_deg=float(phaseAngle(helio[f:, :3], topo[f:, :3])[0]),
        trailedSourceMagTrue=round(float(apparentMagnitude(spec["H"], spec["G"], helio[f, :3],
                                                           topo[f, :3])), 3),
    )
    for i, c in enumerate("xyz"):
        sorcha[f"Obj_Sun_{c}_LTC_km"] = float(helio[f, i])
        sorcha[f"Obj_Sun_v{c}_LTC_km_s"] = float(helio[f, 3 + i])
        sorcha[f"Obs_Sun_{c}_km"] = float(observer[f, i])
        sorcha[f"Obs_Sun_v{c}_km_s"] = float(observer[f, 3 + i])

    # Expected error of moving the Sorcha row linearly to the shutter time:
    # half the second derivative times the step squared, from the fit.
    step = (mjdTai[k] - mjdTai[f])*SEC_PER_DAY
    linearError = sorchaLinearError(objPoly, obsPoly, t[f], step)

    return dict(
        name=spec["name"], kind=spec["kind"], objId=spec["objId"],
        unpackedDesig=spec["unpackedDesig"], packedDesig=spec["packedDesig"],
        visitMjdTai=float(mjdTai[-3]), shutterMjdTai=float(mjdTai[k]),
        mpSky=dict(tmin=float(tmin), tmax=float(tmax), objPoly=objPoly.T.tolist(),
                   obsPoly=obsPoly.T.tolist()),
        sorcha=sorcha, truth=truth, fitResidual=residuals, sorchaLinearError=linearError,
        info=dict(visitUtc=spec["visitUtc"], shutterOffsetSec=spec["shutterOffset"],
                  fieldOffsetSec=spec["fieldOffset"], visitJdTdb=float(jdTdb[-3]),
                  shutterJdTdb=float(jdTdb[k]), fieldJdTdb=float(jdTdb[f]),
                  skyRateDegDay=float(np.hypot(*skyRates(topo[k:k + 1, :3], topo[k:k + 1, 3:]))[0]),
                  observerCheckM=float(observerCheckM),
                  helioRangeAu=float(np.linalg.norm(helio[k, :3])/AU_KM)),
    )


def sorchaLinearError(objPoly: np.ndarray, obsPoly: np.ndarray, tField: float, step: float) -> dict:
    """Estimate the error of a linear move of a Sorcha row.

    Parameters
    ----------
    objPoly, obsPoly : `numpy.ndarray`
        Chebyshev coefficients of the object and observer heliocentric
        positions (au), in days since ``tmin``.
    tField : `float`
        Field time (days since ``tmin``).
    step : `float`
        Time of the move (s).

    Returns
    -------
    error : `dict`
        Half the second derivative times ``step`` squared, for the sky
        position (mas), the heliocentric object and observer positions and
        the topocentric position (m), the range (m) and the phase angle
        (deg); and the second derivative times ``step``, for the velocities
        of the object, heliocentric and topocentric (mm/s), which a linear
        move keeps.
    """
    # Quantities at field time +- h, h = 30 s; second derivatives by
    # central differences (error ~ h^2 f''''/12, negligible here).
    h = 30.0
    t = tField + np.array([-h, 0.0, h])/SEC_PER_DAY
    helio = chebyshev.chebval(t, objPoly).T*AU_KM*1e3
    observer = chebyshev.chebval(t, obsPoly).T*AU_KM*1e3
    topo = helio - observer
    ra, dec = raDec(topo)
    ra = np.unwrap(np.radians(ra))
    dec = np.radians(dec)

    def half(q):
        return 0.5*(q[0] - 2*q[1] + q[2])/h**2*step**2

    def velocityChange(q):
        return np.linalg.norm((q[0] - 2*q[1] + q[2])/h**2*step)*1e3

    sky = np.hypot(half(ra)*np.cos(dec[1]), half(dec))
    return dict(
        skyMas=float(np.degrees(sky)*3.6e6),
        helioM=float(np.linalg.norm(half(helio))),
        observerM=float(np.linalg.norm(half(observer))),
        topoM=float(np.linalg.norm(half(topo))),
        rangeM=float(abs(half(np.linalg.norm(topo, axis=1)))),
        phaseDeg=float(abs(half(phaseAngle(helio, topo)))),
        helioVelocityMmS=float(velocityChange(helio)),
        topoVelocityMmS=float(velocityChange(topo)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default="cache", help="Directory of cached Horizons responses.")
    args = parser.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    # Build every object from Horizons vectors.
    horizons = Horizons(args.cache)
    objects = [makeObject(horizons, spec) for spec in OBJECTS]
    print(f"Horizons queries made: {horizons.nQueries}")

    # Write compact JSON next to this script.
    fixture = dict(
        description="Geometric JPL Horizons state vectors (ICRF, heliocentric object and Rubin "
                    "X05 observer) for the shutter-timing tests of SolarSystemAssociationTask; "
                    "see README.md.",
        generated=datetime.date.today().isoformat(),
        units=dict(time="MJD TAI", positions="au", velocities="km/s", angles="deg",
                   chebyshev="positions in au, polynomial in days since tmin (MJD TAI)"),
        chebyshevDegree=CHEBYSHEV_DEGREE, windowHours=WINDOW_HOURS, stepMinutes=STEP_MINUTES,
        objects=objects,
    )
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "horizons.json")
    with open(path, "w") as f:
        json.dump(fixture, f, separators=(",", ":"))
        f.write("\n")

    # Summary.
    for obj in objects:
        print(json.dumps(dict(name=obj["name"], info=obj["info"], fit=obj["fitResidual"],
                              linear=obj["sorchaLinearError"]), indent=1))
    print(f"Wrote {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
