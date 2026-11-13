# Horizons fixture for the shutter-timing SSO tests

`horizons.json` holds JPL Horizons ephemerides of three real asteroids for
`TestShutterTimingEpochs` in `tests/test_ssoAssociation.py`. The tests move
each prediction of `SolarSystemAssociationTask` to a shutter-corrected time
and compare it with Horizons at that time. `make_fixture.py` writes the file;
pytest does not collect it.

## Objects

Each object has its own visit on the night of 2026-06-21 at Rubin (MPC code
X05), with the object above the horizon and the Sun well below it. The
elevations below are geometric and computed with astropy.

| Object | Kind | Visit (UTC) | Elevation | Distance (au) | r (au) | Rate (deg/day) | Phase (deg) |
|---|---|---|---|---|---|---|---|
| (1232) Cortusa = 1931 TF2 | main-belt asteroid near opposition | 05:00 | 70 deg | 1.748 | 2.740 | 0.197 | 5.7 |
| (17182) 1999 VU | numbered NEO, 2 days after its 0.176 au approach | 08:00 | 39 deg | 0.177 | 1.045 | 2.64 | 75.7 |
| 2025 WC4 | NEO at close approach (0.026 au, 2026-06-21 04:02 TDB) | 00:30 | 30 deg | 0.026 | 1.005 | 24.2 | 115.1 |

How the objects were chosen:

- **2025 WC4** came from the SBDB close-approach API (`cad.api`). The search
  covered approaches within 0.03 au between 2025-07 and 2026-09 with
  H < 24.5 and a geocentric rate of about 20-45 deg/day. 2025 WC4 is a PHA
  with a multi-opposition orbit (JPL#31, 143 observations, 2021-2026). Its
  approach falls in a Chilean winter night.
- **(17182)** is the only numbered NEO in `cad.api` within 0.3 au of the Earth
  between 2026-06-17 and 2026-06-25.
- **(1232) Cortusa** came from screening numbered MBAs with H < 10.5
  (`sbdb_query.api`) with two-body orbits. The screen looked for an object
  high in the sky at 05:00 UTC whose principal designation is an ordinary
  provisional one, so it packs to a valid 7-character `packed_desig`.

`packed_desig` holds the packed principal provisional designations:
`J31T02F`, `J99V00U` and `K25W04C`.

## Epochs

All times in the file are MJD TAI. The visit time `visitMjdTai` is the UTC
time above. The shutter-corrected time `shutterMjdTai` is the visit plus
`shutterOffsetSec`. The Sorcha row's `fieldMJD_TAI` is the visit plus
`fieldOffsetSec`. The Sorcha branch therefore moves its row by
`shutterOffsetSec - fieldOffsetSec`.

| Object | shutterOffsetSec | fieldOffsetSec | Sorcha move (s) |
|---|---|---|---|
| Cortusa | -0.18 | +2.5 | -2.68 |
| 1999 VU | +0.27 | -1.5 | +1.77 |
| 2025 WC4 | -0.29 | +3.0 | -3.29 |

Horizons is queried with `TLIST` in JD TDB. Horizons stores an epoch as a
double JD, which has a resolution of 2^-31 d (40 us). Each epoch is therefore
chosen as a JD TDB double and sent as the shortest decimal string that gives
that double, so Horizons evaluates exactly at the stated epoch.

Shorter decimal strings are not exact. A test showed that `+1e-9` d becomes
`+2^-30` d (80 us), and that `+1e-10` d does not change the epoch at all.

TDB is converted to TAI with astropy, at the X05 location.

## Horizons queries and conventions

Every query is a vector table with these settings:

```
EPHEM_TYPE=VECTORS  REF_PLANE=FRAME  REF_SYSTEM=ICRF  VEC_CORR=NONE
OUT_UNITS=KM-S  VEC_TABLE=2  CSV_FORMAT=YES  TIME_TYPE=TDB  TLIST_TYPE=JD
```

That means geometric states (no light-time or aberration correction), in ICRF
equatorial axes, in km and km/s. The km units give 16 significant digits, or
micrometres.

Three queries are made per object, each at the same 40 epochs. The 40 epochs
are a 37-point grid plus the visit, shutter and field times.

- The object from `CENTER='500@10'` (the Sun's centre), for its heliocentric
  state.
- The Sun (`COMMAND='10'`) from `CENTER='X05@399'`. The observer's
  heliocentric state is minus this vector.
- The object from `CENTER='X05@399'`, as a check. It agrees with the object
  minus the observer to 0.14 mm.

The targets are `COMMAND='1232;'`, `'17182;'` and `'DES=2025 WC4;'`. These
use JPL solutions with DE441 and the SB441-N16 small-body perturbers.

The task computes RA and Dec as the direction of (object - observer), so the
truth uses the same convention. The truth is the geometric topocentric
direction, not Horizons' astrometric RA and Dec. For 2025 WC4 the two differ
by the light-time correction, about 13 arcsec. An observer table from Horizons
confirms this.

## Contents of `horizons.json`

Per object:

- **`truth`**: the values at `shutterMjdTai`:
  - `ephRa` and `ephDec` (deg).
  - `helio` and `topo` positions (au).
  - `helioVelocity` and `topoVelocity` (km/s).
  - `topoRange` (au).
  - `phaseAngle` (deg), the angle between the heliocentric and topocentric
    vectors, as the task defines it.
- **`mpSky`**: `tmin` and `tmax` (MJD TAI), plus `objPoly` and `obsPoly`. Each
  polynomial has three Chebyshev coefficient arrays (x, y, z) of the
  heliocentric position in au. The polynomials are evaluated, like mpSky's,
  at the time in days since `tmin`.
  - The fit uses degree 10 over the 37-point grid, from 3 h before to 3 h
    after the visit, every 10 min.
  - Production mpSky fits one-day windows with degree 3 for objects and 7 for
    the observer. Those fits are far too coarse to test at the mas level.
  - From degree 10 up, the residual is set by the float64 resolution of the
    MJD sample times (~1 us, a few cm at 30 km/s). Up to degree 10 numpy
    gives no rank warning.
- **`fitResidual`**: the maximum error of the fit over all 40 epochs, including
  the three off-grid ones:

  | Object | Position (m) | Velocity (mm/s) | Sky (mas) |
  |---|---|---|---|
  | Cortusa | 0.031 | 0.75 | 8e-6 |
  | 1999 VU | 0.028 | 0.87 | 1e-4 |
  | 2025 WC4 | 0.035 | 0.88 | 1e-3 |

- **`sorcha`**: a Sorcha row at `fieldMJD_TAI`, built from the same geometric
  vectors:
  - RA and Dec, their rates, the range and range rate, and the phase angle.
  - The heliocentric object and observer states in km and km/s.
  - `fieldJD_TDB`.
  - `trailedSourceMagTrue`, the H, G magnitude.
- **`sorchaLinearError`**: the expected error of the Sorcha branch's linear
  move. It is computed by central differences on the fitted polynomials:
  - Half the second derivative times the move squared, for the sky position
    (mas), the heliocentric, observer and topocentric positions (m), the range
    (m) and the phase angle (deg).
  - The second derivative times the move, for the heliocentric and topocentric
    velocities (mm/s), which the move leaves unchanged.

  The largest values are for 2025 WC4: 0.0085 mas, 0.59 m in range and
  0.16 m in topocentric position. The topocentric velocity changes by
  0.05-0.1 m/s over the 1.8-3.3 s moves, mostly from the Earth's rotation.

- **`info`**: provenance only, not used by the tests. It holds the TDB epochs,
  the sky rate, the observer check and the heliocentric distance.

## Regenerating

Use the LSST stack environment (numpy and astropy):

```
python make_fixture.py --cache /path/to/cache
```

The script makes nine Horizons queries, one at a time and at least 1.5 s
apart. It caches each response as JSON in the `--cache` directory, so a rerun
with the cache in place makes no requests.

The cache for this file is
`/sdf/data/rubin/user/mjuric/shutter-timing/stack-port/validation/horizons-fixture/cache/`
(not in git). The file was generated on 2026-10-08, from the Horizons
solutions current on that date. A later run may give slightly different orbits,
and with them different values. The tests compare the task with whatever the
file holds.
