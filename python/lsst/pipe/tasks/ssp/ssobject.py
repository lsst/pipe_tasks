import pandas as pd
import numpy as np
from . import photfit
from . import util
from . import schema
from .moid import MOIDSolver, earth_orbit_J2000
import argparse
import sys

# The only columns we need from DiaSource.

DIA_COLUMNS = [
    "diaSourceId", "midpointMjdTai", "ra", "dec", "extendedness",
    "band", "psfFlux", "psfFluxErr"
]


def nJy_to_mag(f_njy):
    """
    Convert flux density in nanoJanskys (nJy) to AB magnitude.

    Parameters
    ----------
    f_njy : float or array-like
        Flux density in nanoJanskys.

    Returns
    -------
    float or array-like
        AB magnitude corresponding to the input flux density.
    """
    return 31.4 - 2.5 * np.log10(f_njy)


def nJy_err_to_mag_err(f_njy, f_err_njy):
    """
    Convert flux error in nanoJanskys to magnitude error.

    Parameters
    ----------
    f_njy : float
        Flux in nanoJanskys.
    f_err_njy : float
        Flux error in nanoJanskys.

    Returns
    -------
    float
        Magnitude error.
    """
    return 1.085736 * (f_err_njy / f_njy)


def compute_ssobject_entry(row, sss):
    # just verify we didn't screw up something
    assert sss["ssObjectId"].nunique() == 1

    # Metadata columns
    row["ssObjectId"] = sss["ssObjectId"].iloc[0]
    row["firstObservationMjdTai"] = sss["dia_midpointMjdTai"].min()

    if "discoverySubmissionDate" in row.dtype.names:  # DP2 does not have this field
        # FIXME: here I arbitrarily guess we discover everything 7 days
        # after first obsv. we should really pull this out of the obs_sbn tbl.
        row["discoverySubmissionDate"] = row["firstObservationMjdTai"] + 7.0
    row["arc"] = np.ptp(sss["dia_midpointMjdTai"])
    row["designation"] = sss["designation"].iloc[0]

    # observation counts
    row["nObs"] = len(sss)

    # per band entries
    for band in "ugrizy":
        df = sss[sss["dia_band"] == band]

        # set defaults for this band (equivalents of NULL)
        row[f"{band}_Chi2"] = -1
        row[f"{band}_G12"] = np.nan
        row[f"{band}_G12Err"] = np.nan
        row[f"{band}_H"] = np.nan
        row[f"{band}_H_{band}_G12_Cov"] = np.nan
        row[f"{band}_HErr"] = np.nan
        row[f"{band}_nObsUsed"] = -1

        nBandObs = len(df)
        row[f"{band}_nObs"] = nBandObs
        if nBandObs > 0:
            paMin, paMax = df["phaseAngle"].min(), df["phaseAngle"].max()
            row[f"{band}_phaseAngleMin"] = paMin
            row[f"{band}_phaseAngleMax"] = paMax

            if nBandObs > 1:
                # do the absmag/slope fits, if there are at least two
                # data points
                H, G12, sigmaH, sigmaG12, covHG12, chi2dof, nobsv = photfit.fitHG12(
                    df["dia_psfMag"], df["dia_psfMagErr"], df["phaseAngle"], df["topoRange"], df["helioRange"]
                )
                nDof = 2
                # print(provID, band, H, G12, sigmaH, sigmaG12, covHG12,
                #       chi2dof, nobsv)

                # mark if the fit failed
                if np.isnan(G12):
                    row[f"{band}_slope_fit_failed"] = True
                    # FIXME: if fitting fails, we should revert to simple
                    # estimation of H using a fiducial G12 value, storing
                    # that G12 as well.

                row[f"{band}_Chi2"] = chi2dof * nDof
                row[f"{band}_G12"] = G12
                row[f"{band}_G12Err"] = sigmaG12
                row[f"{band}_H"] = H
                row[f"{band}_H_{band}_G12_Cov"] = covHG12
                row[f"{band}_HErr"] = sigmaH
                row[f"{band}_nObsUsed"] = nobsv

    # Extendedness
    row["extendednessMin"] = sss["dia_extendedness"].min()
    row["extendednessMax"] = sss["dia_extendedness"].max()
    row["extendednessMedian"] = sss["dia_extendedness"].median()


def compute_ssobject(sss, dia, mpcorb):
    """
    Compute solar system object properties by joining and processing
    SSSource, DiaSource, and MPC orbit data.

    This function takes a pre-grouped SSSource table, joins it with
    DiaSource data, computes per-object quantities, and calculates
    additional orbital parameters like Tisserand J and Minimum Orbit
    Intersection Distance (MOID) with Earth for matching objects.

    Parameters
    ----------
    sss : pandas.DataFrame
        SSSource table, pre-grouped by 'ssObjectId'. Must be sorted by
        'ssObjectId' for correct grouping. Contains columns like
        'ssObjectId', 'diaSourceId', etc.
    dia : pandas.DataFrame
        DiaSource table with columns prefixed as 'dia_' in the join.
        Must include 'dia_diaSourceId', 'dia_psfFlux', 'dia_psfFluxErr',
        etc.
    mpcorb : pandas.DataFrame
        MPC orbit data with columns like
        'unpacked_primary_provisional_designation', 'q', 'e', 'i',
        'node', 'argperi'.

    Returns
    -------
    numpy.ndarray
        Array of ssObject records with dtype schema.ssObjectDtype,
        containing computed properties for each unique ssObjectId,
        including magnitudes, orbital elements, Tisserand J, and
        MOID-related values.

    Raises
    ------
    AssertionError
        If 'sss' is not pre-grouped by 'ssObjectId', or if DiaSources
        are missing after join.

    Notes
    -----
    - The function assumes 'sss' is large and avoids internal
      sorting/copying for efficiency.
    - Tisserand J and MOID are computed only for objects matching
      designations in 'mpcorb'.
    - MOID computation uses a MOIDSolver for each matched object.
    """

    # assert that sss is pre-grouped by ssObjectId
    assert util.values_grouped(sss["ssObjectId"]), (
        "SSSource table must be pre-grouped by ssObjectId. "
        "An easy way to do this is to sort by ssObjectId before calling compute_ssobject(). "
        "The grouping is required for correct per-object computations, and since SSSource is "
        "typically large and we want to avoid copies, it's not done internally."
    )

    # Join the DiaSource parts we're interested in to our SSSource table
    num = len(sss)
    dia_tmp = dia[DIA_COLUMNS].add_prefix("dia_")  # FIXME: does this cause unnececessary copy?
    # FIXME: The diaSourceId should really be uint64. But Felis doesn't speak
    # uint64, but only knows about int64. Yet the pipeline produces uint64
    # diaSourceId in the dia_source dataset. So we have to cast here to int64
    # to make the join work (otherwise pyarrow tries to cast to float64, and
    # the whole thing gloriously explodes).
    dia_tmp["dia_diaSourceId"] = dia_tmp["dia_diaSourceId"].astype("int64[pyarrow]")
    sss = sss.merge(dia_tmp, left_on="diaSourceId", right_on="dia_diaSourceId", how="inner")
    assert num == len(sss), f"{num - len(sss)} DiaSources found missing."
    del sss["dia_diaSourceId"]
    del dia_tmp

    # add magnitude columns
    sss["dia_psfMag"] = nJy_to_mag(sss["dia_psfFlux"])
    sss["dia_psfMagErr"] = nJy_err_to_mag_err(sss["dia_psfFlux"], sss["dia_psfFluxErr"])

    # Pre-create the empty array
    totalNumObjects = np.unique(sss["ssObjectId"]).size
    obj = np.zeros(totalNumObjects, dtype=schema.SSObjectDtype)

    # compute per-object quantities
    util.group_by([sss], "ssObjectId", compute_ssobject_entry, out=obj)

    #
    # compute columns that can be efficiently computed in a vector fashon
    #
    # Tisserand J

    if mpcorb is not None:
        # inner join by provisional designation. We allow for some objects to be
        # missing from mpcorb (this should not happen often, but it did in DP1).
        # FIXME: at some point require that no objects are missing. I _think_ that
        # shouldn't happen in normal operations.
        oidx, midx = util.argjoin(
            obj["designation"].astype("U"),
            mpcorb["unpacked_primary_provisional_designation"].to_numpy().astype("U"),
        )
        assert np.all(
            mpcorb["unpacked_primary_provisional_designation"].take(midx)
            == obj["designation"][oidx].astype("U")
        )
        q, e, i, node, argperi = util.unpack(mpcorb["q e i node argperi".split()].take(midx))
        a = q / (1.0 - e)
        obj["tisserand_J"][oidx] = util.tisserand_jupiter(a, e, i)

        # MOID computation
        solver = MOIDSolver()
        for i, el_obj in enumerate(zip(a, e, i, node, argperi)):
            (moid, deltaV, eclon, trueEarth, trueObject) = solver.compute(earth_orbit_J2000(), el_obj)
            row = obj[oidx[i]]
            row["MOIDEarth"] = moid
            row["MOIDEarthDeltaV"] = deltaV
            row["MOIDEarthEclipticLongitude"] = eclon
            row["MOIDEarthTrueAnomaly"] = trueEarth
            row["MOIDEarthTrueAnomalyObject"] = trueObject

    return obj


def main():
    """
    CLI entry point for building SSObject table from SSSource,
    DiaSource, and MPC orbit data.
    """
    parser = argparse.ArgumentParser(
        description="Build SSObject table from SSSource, DiaSource, and MPC orbit Parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ssp-build-ssobject sssource.parquet dia_sources.parquet mpc_orbits.parquet --output ssobject.parquet
        """,
    )

    parser.add_argument("sssource_parquet", help="Path to SSSource Parquet file")
    parser.add_argument("diasource_parquet", help="Path to DiaSource Parquet file")
    parser.add_argument("mpcorb_parquet", help="Path to MPC orbits Parquet file")
    parser.add_argument("--output", "-o", required=True, help="Path to output SSObject Parquet file")
    parser.add_argument(
        "--reraise",
        action="store_true",
        help="Re-raise exceptions instead of exiting gracefully (for debugging)",
    )

    args = parser.parse_args()

    try:
        # Load SSSource
        print(f"Loading SSSource from {args.sssource_parquet}...")
        sss = pd.read_parquet(args.sssource_parquet, engine="pyarrow", dtype_backend="pyarrow").reset_index(
            drop=True
        )
        num = len(sss)
        print(f"Loaded {num:,} SSSource rows")

        # Load DiaSource with required columns
        dia_columns = [
            "diaSourceId",
            "midpointMjdTai",
            "ra",
            "dec",
            "extendedness",
            "band",
            "psfFlux",
            "psfFluxErr",
        ]
        print(f"Loading DiaSource from {args.diasource_parquet}...")
        dia = pd.read_parquet(
            args.diasource_parquet, engine="pyarrow", dtype_backend="pyarrow", columns=dia_columns
        ).reset_index(drop=True)
        print(f"Loaded {len(dia):,} DiaSource rows")

        # Ensure diaSourceId is uint64
        assert np.all(dia["diaSourceId"] >= 0)
        dia["diaSourceId"] = dia["diaSourceId"].astype("uint64[pyarrow]")

        # Load MPC orbits
        mpcorb_columns = [
            "unpacked_primary_provisional_designation",
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
        ]
        print(f"Loading MPC orbits from {args.mpcorb_parquet}...")
        mpcorb = pd.read_parquet(
            args.mpcorb_parquet, engine="pyarrow", dtype_backend="pyarrow", columns=mpcorb_columns
        ).reset_index(drop=True)
        print(f"Loaded {len(mpcorb):,} MPC orbit rows")

        # Compute SSObject
        print("Computing SSObject data...")
        obj = compute_ssobject(sss, dia, mpcorb)

        # Save result
        print(f"Saving {len(obj):,} SSObject rows to {args.output}...")
        util.struct_to_parquet(obj, args.output)

        print(f"Success! Created SSObject with {len(obj):,} objects")
        print(f"Row size: {obj.dtype.itemsize:,} bytes, Total size: {obj.nbytes:,} bytes")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.reraise:
            raise
        sys.exit(1)


if __name__ == "__main__":
    input_dir = "./analysis/inputs"
    output_dir = "./analysis/outputs"

    #
    # Loads
    #

    # load SSObject
    sss = pd.read_parquet(
        f"{output_dir}/sssource.parquet", engine="pyarrow", dtype_backend="pyarrow"
    ).reset_index(drop=True)
    num = len(sss)

    # load corresponding DiaSource
    dia = pd.read_parquet(
        f"{input_dir}/dia_sources.parquet", engine="pyarrow", dtype_backend="pyarrow", columns=DIA_COLUMNS
    ).reset_index(drop=True)

    # FIXME: I'm not sure why the datatype is int and not uint here.
    # Investigate upstream...
    assert np.all(dia["diaSourceId"] >= 0)
    dia["diaSourceId"] = dia["diaSourceId"].astype("uint64[pyarrow]")

    # Load mpcorb
    mpcorb = pd.read_parquet(
        f"{input_dir}/mpc_orbits.parquet",
        engine="pyarrow",
        dtype_backend="pyarrow",
        columns=[
            "unpacked_primary_provisional_designation",
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

    #
    # Business logic
    #
    obj = compute_ssobject(sss, dia, mpcorb)

    #
    # Save
    #
    util.struct_to_parquet(obj, f"{output_dir}/ssobject.parquet")

    print(f"row_length={obj.dtype.itemsize:,} bytes, rows={len(obj):,}, {obj.nbytes:,} bytes total")
