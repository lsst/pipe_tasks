from astropy.time import Time
import astropy.units as u

import jax
from jorbit import Particle

import numpy as np

jax.config.update("jax_enable_x64", True)


def _aux_compute_ephemerides(provID, ephTimes, mpcorb):

    kms = u.km / u.s

    # Ephemerides
    (packed, a, e, i, node, argperi, M, epoch, H, G) = mpcorb.query(
        "unpacked_primary_provisional_designation == @provID", engine="python"
    )["packed_primary_provisional_designation a e i node argperi mean_anomaly epoch_mjd h g".split()].iloc[0]

    # FIXME: hack until jorbit is fixed (https://github.com/ben-cassese/jorbit/issues/26)
    p = Particle.from_horizons(name=packed, time=Time(epoch, format="mjd", scale="tdb"))

    eph, xx, vv, obs = p.ephemeris(times=ephTimes, observer="rubin")

    xx, vv, obs = np.array(xx).T, np.array(vv).T, np.array(obs).T
    vv = (vv * u.au / u.day).to(kms).value  # convert from AU/day to km/s

    # rate of motion
    dt = 1.0 / (3600.0 + 24.0) * u.s
    ephTimes2 = ephTimes + dt
    eph2, _, _, _ = p.ephemeris(times=ephTimes2, observer="rubin")
    dlon, dlat = eph.spherical_offsets_to(eph2)  # small offsets on tangent plane
    mu_lon = (dlon / dt).to(u.deg / u.day)  # ≈ d(RA*cosDec)/dt
    mu_lat = (dlat / dt).to(u.deg / u.day)  # ≈ d(Dec)/dt
    mu = (eph.separation(eph2) / dt).to(u.deg / u.day)  # total rate of motion

    return eph, (H, G), xx, vv, obs, mu_lon, mu_lat, mu
