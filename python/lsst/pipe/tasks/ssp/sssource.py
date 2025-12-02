from astropy.coordinates import (
    get_body_barycentric_posvel,
    solar_system_ephemeris,
    SkyCoord,
    HeliocentricEclipticIAU76,
)
from astropy.time import Time
import astropy.units as u
from functools import partial
import numpy as np
import pandas as pd

from . import util, schema
from .photfit import hg_V_mag, phase_angle_deg
from .ephem import _aux_compute_ephemerides


def compute_sssource_entry(sss, assoc, mpcorb, dia):

    kms = u.km / u.s

    # extract only the subset of observations related to this object
    dia = dia.iloc[assoc["dia_index"]]

    # just verify we didn't screw up something
    assert np.all(dia["ssObjectId"] == dia["ssObjectId"].iloc[0])
    assert np.all(sss["ssObjectId"] == sss["ssObjectId"][0])
    assert len(dia) == len(sss)

    provID = sss["designation"][0]
    ephTimes = Time(dia["midpointMjdTai"].values, format="mjd", scale="tai")
    eph, (H, G), xx, vv, obs, mu_lon, mu_lat, mu = _aux_compute_ephemerides(provID, ephTimes, mpcorb)

    sss["ephRateRa"] = mu_lon.value
    sss["ephRateDec"] = mu_lat.value
    sss["ephRate"] = mu.value

    # location/velocity of the Sun
    with solar_system_ephemeris.set("de440"):
        pos, vel = get_body_barycentric_posvel("sun", ephTimes)
    hx, hy, hz = pos.x.to(u.au).value, pos.z.to(u.au).value, pos.z.to(u.au).value
    hvx, hvy, hvz = vel.x.to(kms).value, vel.y.to(kms).value, vel.z.to(kms).value

    # location/velocity of the observer
    robs, vobs = util.observatory_barycentric_posvel("X05", ephTimes)
    robs = robs.to(u.au).value
    vobs = vobs.to(u.km / u.s).value
    # r_obs_sun = np.sqrt((robs * robs).sum(axis=0))

    sss["ephRa"] = eph.ra.deg
    sss["ephDec"] = eph.dec.deg
    obsv = SkyCoord(ra=dia["ra"], dec=dia["dec"], unit="deg", frame="icrs")

    sss["ephOffsetDec"] = (dia["dec"].to_numpy() - sss["ephDec"]) * 3600
    sss["ephOffsetRa"] = (dia["ra"].to_numpy() - sss["ephRa"]) * np.cos(np.deg2rad(sss["ephDec"])) * 3600
    sss["ephOffset"] = eph.separation(obsv).arcsec

    # Compute heliocentric position components
    sss["helio_x"] = xx[0] - hx
    sss["helio_y"] = xx[1] - hy
    sss["helio_z"] = xx[2] - hz
    sss["helioRange"] = np.sqrt(sss["helio_x"] ** 2 + sss["helio_y"] ** 2 + sss["helio_z"] ** 2)

    # Compute heliocentric velocity components
    sss["helio_vx"] = vv[0] - hvx
    sss["helio_vy"] = vv[1] - hvy
    sss["helio_vz"] = vv[2] - hvz
    sss["helio_vtot"] = np.sqrt(sss["helio_vx"] ** 2 + sss["helio_vy"] ** 2 + sss["helio_vz"] ** 2)

    # Compute heliocentric radial velocity: dot product of velocity
    # and unit position vector
    sss["helioRangeRate"] = (
        sss["helio_vx"] * sss["helio_x"] + sss["helio_vy"] * sss["helio_y"] + sss["helio_vz"] * sss["helio_z"]
    ) / sss["helioRange"]

    # Compute topocentric position components
    sss["topo_x"] = xx[0] - obs[0]
    sss["topo_y"] = xx[1] - obs[1]
    sss["topo_z"] = xx[2] - obs[2]
    sss["topoRange"] = np.sqrt(sss["topo_x"] ** 2 + sss["topo_y"] ** 2 + sss["topo_z"] ** 2)

    # Compute topocentric velocity components
    sss["topo_vx"] = vv[0] - vobs[0]
    sss["topo_vy"] = vv[1] - vobs[1]
    sss["topo_vz"] = vv[2] - vobs[2]
    sss["topo_vtot"] = np.sqrt(sss["topo_vx"] ** 2 + sss["topo_vy"] ** 2 + sss["topo_vz"] ** 2)

    # Compute topocentric radial velocity: dot product of velocity
    # and unit position vector
    sss["topoRangeRate"] = (
        sss["topo_vx"] * sss["topo_x"] + sss["topo_vy"] * sss["topo_y"] + sss["topo_vz"] * sss["topo_z"]
    ) / sss["topoRange"]

    sss["phaseAngle"] = phase_deg = phase_angle_deg(xx, obs)

    sss["ephVmag"] = hg_V_mag(H, G, sss["helioRange"], sss["topoRange"], phase_deg)

    max_sep = np.max(sss["ephOffset"])
    med_sep = np.median(sss["ephOffset"])
    print(f"{provID}: max/median separation: {max_sep:.4f}, {med_sep:.4f} arcsec")


if __name__ == "__main__":
    input_dir = "./analysis/inputs"
    output_dir = "./analysis/outputs"

    dia = pd.read_parquet(
        f"{input_dir}/dia_sources.parquet", engine="pyarrow", dtype_backend="pyarrow"
    ).reset_index(drop=True)
    # DEBUG: while debugging, remove some indices and resort the array
    dia = dia.sample(frac=0.9, random_state=42).reset_index(drop=True)

    det = pd.read_parquet(
        f"{input_dir}/obs_sbn.parquet", engine="pyarrow", dtype_backend="pyarrow"
    ).reset_index()

    # DEBUG: cut this down to a much smaller table
    sampled_provids = det["provid"].drop_duplicates().sample(10, random_state=42)
    det = det[det["provid"].isin(sampled_provids)].reset_index()
    print(len(det))

    # FIXME: this will have to check if the ID's are IAU-style
    # (with string prefixes)
    det["obssubid"] = det["obssubid"].astype(int)
    det = det[
        [
            "trksub",
            "obssubid",
            "provid",
            "permid",
            "submission_id",
            "ra",
            "dec",
            "obstime",
            "designation_asterisk",
        ]
    ].copy()

    # verify types didn't get mangled somewhere along the way
    # from the database to here
    expect_dtypes = dict(
        trksub="string[pyarrow]",
        obssubid="int64",
        provid="string[pyarrow]",
        permid="string[pyarrow]",
        submission_id="string[pyarrow]",
        ra="double[pyarrow]",
        dec="double[pyarrow]",
        obstime="timestamp[us][pyarrow]",
        designation_asterisk="bool[pyarrow]",
    )

    for col in det.columns:
        assert det[col].dtype == expect_dtypes[col]

    # create the association side table
    assoc = (
        dia[["diaSourceId"]]
        .reset_index()
        .merge(det.add_prefix("mpc_"), left_on="diaSourceId", right_on="mpc_obssubid", how="inner")
    )
    assoc.rename(columns={"index": "dia_index"}, inplace=True)

    # verify all went well
    assert np.all(dia["diaSourceId"].iloc[assoc["dia_index"]].to_numpy() == assoc["diaSourceId"].to_numpy())

    # verify contents of the association table
    util.assoc_validate(dia, assoc)

    totalNumObs = len(assoc)

    numid = pd.read_parquet(
        f"{input_dir}/numbered_identifications.parquet",
        engine="pyarrow",
        columns=["permid", "unpacked_primary_provisional_designation"],
        dtype_backend="pyarrow",
    ).reset_index(drop=True)
    curid = pd.read_parquet(
        f"{input_dir}/current_identifications.parquet",
        engine="pyarrow",
        dtype_backend="pyarrow",
        columns=[
            "unpacked_primary_provisional_designation",
            "unpacked_secondary_provisional_designation",
            "packed_primary_provisional_designation",
        ],
    ).reset_index(drop=True)

    # First step: some numbered objects in `obs_sbn` don't have their
    # provID set. Restore it.
    df = assoc[["mpc_provid", "mpc_permid"]].merge(numid, left_on="mpc_permid", right_on="permid", how="left")
    assert len(df) == len(assoc)

    assoc["mpc_provid"] = assoc["mpc_provid"].where(
        assoc["mpc_provid"].notna(), df["unpacked_primary_provisional_designation"]
    )

    assert not assoc["mpc_provid"].isna().any()
    assert len(assoc) == totalNumObs

    # Second step: update provisional designations with the primary ones.

    df = assoc[["mpc_provid"]].merge(
        curid, left_on="mpc_provid", right_on="unpacked_secondary_provisional_designation", how="inner"
    )
    assoc["mpc_provid"] = df["unpacked_primary_provisional_designation"]
    assoc["mpc_packed"] = df["packed_primary_provisional_designation"]

    assert len(assoc) == totalNumObs

    # sort the association table by object
    assoc.sort_values(["mpc_provid"], inplace=True)

    # create the output array for SSSource
    sss = np.zeros(totalNumObs, dtype=schema.SSSourceDtype)
    sss.dtype.itemsize, len(sss), f"{sss.nbytes:,}"

    #
    # construct SSSource -- start with easily vectorizable columns
    #
    sss["diaSourceId"] = assoc["diaSourceId"].values
    sss["ssObjectId"] = util.packed_ascii_to_uint64_le(assoc["mpc_packed"])
    sss["designation"] = assoc["mpc_provid"]

    df = dia[["ra", "dec", "midpointMjdTai"]].iloc[assoc["dia_index"]]
    ra, dec, t = (
        df["ra"].to_numpy(),
        df["dec"].to_numpy(),
        Time(df["midpointMjdTai"].to_numpy(), format="mjd", scale="tai"),
    )

    sss["elongation"] = util.solar_elongation_ndarray(ra, dec, t)

    # FIXME: verify these coordinate transforms replicate IAU76 at JPL
    p = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=1 * u.au, frame="hcrs")
    ecl = p.transform_to(HeliocentricEclipticIAU76)
    p = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=1 * u.au, frame="icrs")
    gal = p.transform_to("galactic")

    sss["eclLambda"] = ecl.lon
    sss["eclBeta"] = ecl.lat
    sss["galLon"] = gal.l
    sss["galLat"] = gal.b

    mpcorb = pd.read_parquet(
        f"{input_dir}/mpc_orbits.parquet",
        engine="pyarrow",
        dtype_backend="pyarrow",
        columns=[
            "unpacked_primary_provisional_designation",
            "packed_primary_provisional_designation",
            "a",
            "q",
            "e",
            "i",
            "node",
            "argperi",
            "peri_time",
            "mean_anomaly",
            "epoch_mjd",
            "h",
            "g",
        ],
    ).reset_index(drop=True)

    util.group_by([sss, assoc], "ssObjectId", partial(compute_sssource_entry, mpcorb=mpcorb, dia=dia))

    totalNumObjects = np.unique(sss["ssObjectId"]).size
    print(f"{totalNumObjects:,} unique objects with {len(sss):,} total observations.")

    util.struct_to_parquet(sss, f"{output_dir}/sssource.parquet")
