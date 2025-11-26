import numpy as np
import astropy
from astropy.time import Time
import pandas as pd
import astropy.units as u
from astropy.coordinates import get_sun, angular_separation
import numpy.ma as ma
from astropy.constants import R_earth
from astropy.coordinates import (
    EarthLocation,
    solar_system_ephemeris,
    get_body_barycentric_posvel,
)
from astroquery.mpc import MPC
from typing import Optional
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import pyarrow.compute as pc


def assoc_validate(dia, assoc):
    # verify coordinates and times match
    dia = dia[["ra", "dec", "midpointMjdTai"]].iloc[assoc["dia_index"].values]

    # verify coordinates match
    obs = astropy.coordinates.SkyCoord(ra=dia["ra"].values, dec=dia["dec"].values, unit="deg")
    mpc = astropy.coordinates.SkyCoord(ra=assoc["mpc_ra"].values, dec=assoc["mpc_dec"].values, unit="deg")
    sep = obs.separation(mpc)

    print("Separation diffeerence range (arcsec): ", sep.min().arcsec, sep.max().arcsec)
    assert sep.min().arcsec >= 0
    assert sep.max().arcsec <= 0.005  # FIXME: this should be further
    # tightened once we start submitting extra precision to the MPC

    # verify times match
    t_utc = Time(dia["midpointMjdTai"].to_numpy(), format="mjd", scale="tai").utc
    midpoint_utc = pd.to_datetime(t_utc.to_datetime())
    mpc_time = pd.to_datetime(assoc["mpc_obstime"])
    delta_sec = (mpc_time - midpoint_utc).dt.total_seconds()
    delta_sec

    print("Time diffeerence range (sec):          ", delta_sec.min(), delta_sec.max())

    # FIXME: this was relaxed as USDF replica's obstime datatype is borked
    # and rounds (or truncates?) the timestamps to 1 second. E-mailed Dan S.
    # to get it fixed.
    # assert abs(delta_sec).max() < 0.01
    assert abs(delta_sec).max() < 0.51

    print(f"All OK, {len(assoc):,} observations.")


def packed_ascii_to_uint64_le(mpc_packed):
    """
    Convert a pandas string[pyarrow] column of ASCII strings (<= 8 bytes)
    to little-endian uint64 by left-padding with ASCII spaces to 8 chars.
    """

    # Step 1: Convert pandas Series → real pyarrow.StringArray
    arr = pa.array(mpc_packed, type=pa.string())

    # Step 2: Left-pad to length 8 with ASCII spaces (works on older PyArrow!)
    # ascii_lpad takes (string_array, target_length, pad_char)
    padded = pc.ascii_lpad(arr, 8, " ")  # returns string array padded to 8 chars

    # Step 3: Convert padded string → binary
    bin_arr = pc.cast(padded, pa.binary())

    # Step 4: Slice each to exactly 8 bytes
    fixed = pc.binary_slice(bin_arr, 0, 8)

    # Step 5: Flatten chunks into a single contiguous array
    if isinstance(fixed, pa.ChunkedArray):
        fixed = fixed.combine_chunks()

    # Step 6: Extract contiguous values buffer
    buf = fixed.buffers()[2]

    # Step 7: Interpret every 8 bytes as a little-endian uint64
    return np.frombuffer(buf, dtype="<u8")


def solar_elongation_ndarray(ra_deg, dec_deg, t):
    """
    Very fast computation of solar elongation (ICRS great-circle separation)
    using astropy.coordinates.angular_separation.

    Parameters
    ----------
    ra_deg : ndarray
        Target RA in degrees (ICRS).
    dec_deg : ndarray
        Target Dec in degrees (ICRS).
    t : astropy.time.Time
        Observation times.

    Returns
    -------
    elong_deg : ndarray
        Solar elongation in degrees.
    """

    # Get Sun coordinates
    # FIXME: This is sloooooww af. Should probably extract it in a few points,
    # then fit a spline or piecewise poly.
    sun = get_sun(t).icrs  # cheap transformation to ICRS once per row

    # Extract Sun RA/Dec arrays (radian floats)
    sun_ra = sun.ra.radian
    sun_dec = sun.dec.radian

    # Convert input to radians
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    # Fast great-circle angular separation
    sep = angular_separation(ra, dec, sun_ra, sun_dec)

    # Convert to degrees for return
    return np.degrees(sep)


def group_by(arrs, key, func, out=None, check_grouped=True):
    """
    Group multiple NumPy arrays by arrs[0][key], assuming the key column
    is already grouped (equal keys are contiguous), but group blocks may
    appear in any order.

    If out is provided:
        func(row_view, *subarrs)
    If out is None:
        func(*subarrs)

    Parameters
    ----------
    arrs : list/tuple of ndarray
        Equal-length arrays. arrs[0] contains the grouping key.
    key : str
        Column name in arrs[0] to group by.
    func : callable
        Called either as func(row, *subarrs) or func(*subarrs).
    out : ndarray or None
        Optional preallocated structured output array.
    check_grouped : bool
        If True, verify that key values are grouped contiguously.

    Returns
    -------
    ndarray or dict
    """
    arr0 = arrs[0]
    keys = arr0[key]

    # ---------- Grouped-contiguous check ----------
    if check_grouped:
        seen = set()
        current = keys[0]
        seen.add(current)

        for i in range(1, len(keys)):
            k = keys[i]
            if k != current:
                # Key changed
                if k in seen:
                    raise ValueError(
                        f"Key '{key}' is not properly grouped. "
                        f"Value {k} reappears at index {i} "
                        f"after a different key was encountered."
                    )
                seen.add(k)
                current = k

    # ---------- Find true group boundaries ----------
    unique_keys, idx_start, counts = np.unique(keys, return_index=True, return_counts=True)
    idx_end = idx_start + counts
    n_groups = len(unique_keys)

    # ---------- Preallocated output path ----------
    if out is not None:
        if len(out) < n_groups:
            raise ValueError(f"Out array too small: need {n_groups}, have {len(out)}")

        for out_idx, (start, end) in enumerate(zip(idx_start, idx_end)):
            subarrs = tuple(a[start:end] for a in arrs)
            row = out[out_idx]  # writable structured scalar
            func(row, *subarrs)
            if out_idx % 100 == 0:
                print(f"[{datetime.now().isoformat()}] count={out_idx}")

        return out

    # ---------- Dict output path ----------
    results = {}
    for keyval, start, end in zip(unique_keys, idx_start, idx_end):
        subarrs = tuple(a[start:end] for a in arrs)
        results[keyval] = func(*subarrs)

    return results


def values_grouped(a: np.ndarray) -> bool:
    """
    Return True if each distinct value in 1D array `a`
    appears in a single contiguous block (all duplicates grouped).
    """
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError("a must be 1D")
    if a.size == 0:
        return True

    # 1) True where a new group starts: first element, or value != previous
    group_starts = np.concatenate(([True], a[1:] != a[:-1]))

    # 2) Values for each group (one per contiguous block)
    group_vals = a[group_starts]

    # 3) Check that no value appears in more than one group
    #    i.e., all group_vals are unique
    return np.unique(group_vals).size == group_vals.size


def earthlocation_from_obscode(obscode: str) -> EarthLocation:
    """
    Convert an MPC observatory code (e.g. 'X05') to an EarthLocation,
    using MPC.get_observatory_codes() columns:
      Code, Longitude, cos, sin, Name.
    """
    tbl = MPC.get_observatory_codes()
    row = tbl[tbl["Code"] == obscode]
    if len(row) != 1:
        raise ValueError(f"Unknown or ambiguous obscode {obscode!r}")
    row = row[0]

    # Handle missing ground positions (spacecraft, etc.)
    if ma.is_masked(row["Longitude"]) or ma.is_masked(row["cos"]) or ma.is_masked(row["sin"]):
        raise ValueError(f"Obscode {obscode!r} has no ground position (spacecraft?)")

    lon = (row["Longitude"] * u.deg).to(u.rad).value  # radians
    rho_cosphi = float(row["cos"])
    rho_sinphi = float(row["sin"])

    # Geocentric Cartesian coordinates in Earth radii
    x_er = rho_cosphi * np.cos(lon)
    y_er = rho_cosphi * np.sin(lon)
    z_er = rho_sinphi

    # Convert Earth radii -> meters
    x = (x_er * R_earth).to(u.m)
    y = (y_er * R_earth).to(u.m)
    z = (z_er * R_earth).to(u.m)

    return EarthLocation.from_geocentric(x, y, z)


def observatory_barycentric_posvel(obscode: str, obstime: Time):
    """
    Barycentric (ICRS) position and velocity of an observatory given an
    MPC obscode, using JPL DE440 for the Earth ephemeris.

    Returns
    -------
    r_bary : Quantity, shape (3, ...)
        Barycentric position in AU.
    v_bary : Quantity, shape (3, ...)
        Barycentric velocity in AU/day.
    """
    loc = earthlocation_from_obscode(obscode)

    # Geocentric position & velocity of the site in GCRS (Earth center)
    obsgeoloc, obsgeovel = loc.get_gcrs_posvel(obstime)

    # Earth barycentric pos/vel in ICRS using DE440
    with solar_system_ephemeris.set("de440"):
        earth_pos, earth_vel = get_body_barycentric_posvel("earth", obstime)

    # ---- convert everything to SI ----
    # Earth (ICRS, barycentric)
    r_earth_si = earth_pos.xyz.to(u.m)
    v_earth_si = earth_vel.xyz.to(u.m / u.s)

    # Site (GCRS, geocentric)
    r_site_geo_si = getattr(obsgeoloc, "xyz", obsgeoloc).to(u.m)
    v_site_geo_si = getattr(obsgeovel, "xyz", obsgeovel).to(u.m / u.s)

    # ---- barycentric site vectors in SI ----
    r_site_bary_si = r_earth_si + r_site_geo_si
    v_site_bary_si = v_earth_si + v_site_geo_si

    # ---- convert to AU, AU/day ----
    r_site_bary = r_site_bary_si.to(u.au)
    v_site_bary = v_site_bary_si.to(u.au / u.day)

    return r_site_bary, v_site_bary


#
# Serialization
#


def struct_to_parquet(
    arr: np.ndarray,
    path: str,
    *,
    chunk_size: Optional[int] = None,
    row_group_size: Optional[int] = None,
) -> None:
    """
    Write a large NumPy structured array to a Parquet file using PyArrow.

    Designed for dtypes like dia_dtype / orbit_dtype with up to ~1e8 rows.
    """

    if arr.dtype.names is None:
        raise TypeError("struct_to_parquet expects a structured NumPy array (dtype.names is None).")

    n_rows = len(arr)
    if n_rows == 0:
        return

    # Heuristic default chunk size
    if chunk_size is None:
        if n_rows <= 10_000_000:
            chunk_size = n_rows
        else:
            chunk_size = 1_000_000

    def _numpy_to_arrow_array(col: np.ndarray, name: str) -> pa.Array:
        """
        Convert a 1D NumPy column view to a PyArrow Array.

        - Object / unicode / bytes → Arrow string(), with padding stripped
          for fixed-width 'S'/'U' dtypes.
        - Numeric types → Arrow infers type from NumPy.
        """
        kind = col.dtype.kind

        if kind == "O":
            # Already Python objects (str); Arrow will handle them fine.
            return pa.array(col, type=pa.string())

        if kind == "S":
            # Fixed-width bytes, padded with b"\\x00".
            # Decode + strip trailing NULs.
            # NOTE: adjust encoding if you know it's not ASCII/UTF-8.
            decoded = np.char.decode(col, "utf-8", errors="ignore")
            stripped = np.char.rstrip(decoded, "\x00")
            return pa.array(stripped, type=pa.string())

        if kind == "U":
            # Fixed-width unicode, padded with U+0000.
            stripped = np.char.rstrip(col, "\x00")
            return pa.array(stripped, type=pa.string())

        # Numeric / bool dtypes: let Arrow infer
        return pa.array(col)

    def _chunk_to_table(chunk: np.ndarray) -> pa.Table:
        arrays = []
        fields = []

        for name in chunk.dtype.names:
            col = chunk[name]
            arr_arrow = _numpy_to_arrow_array(col, name)
            arrays.append(arr_arrow)
            fields.append(pa.field(name, arr_arrow.type))

        schema = pa.schema(fields)
        return pa.Table.from_arrays(arrays, schema=schema)

    writer: Optional[pq.ParquetWriter] = None
    try:
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            chunk = arr[start:end]
            table = _chunk_to_table(chunk)

            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table, row_group_size=row_group_size)
    finally:
        if writer is not None:
            writer.close()


# Jupiter's semimajor axis in AU (J2000-ish)
A_JUP = 5.2044


def tisserand_jupiter(a, e, inc_deg, a_j=A_JUP):
    """
    Compute Tisserand parameter with respect to Jupiter.

    Parameters
    ----------
    a : float or ndarray
        Semimajor axis of the small body [AU].
    e : float or ndarray
        Eccentricity.
    inc_deg : float or ndarray
        Inclination [degrees], typically to the ecliptic.
    a_j : float
        Semimajor axis of Jupiter [AU]. Default ~5.2044.

    Returns
    -------
    T_J : float or ndarray
        Tisserand parameter with respect to Jupiter.
    """
    inc_rad = np.deg2rad(inc_deg)
    return (a_j / a) + 2.0 * np.cos(inc_rad) * np.sqrt((a / a_j) * (1.0 - e**2))


def unpack(df, to_numpy=True):
    """
    Return all DataFrame columns as a tuple.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    to_numpy : bool, default True
        If True, return each column as a NumPy array.
        If False, return each column as a pandas Series.

    Returns
    -------
    tuple
        Tuple of columns in the original order.
    """
    if to_numpy:
        return tuple(df[col].to_numpy() for col in df.columns)
    else:
        return tuple(df[col] for col in df.columns)


def argjoin(a, v):
    """
    Perform an efficient inner join between two 1-D NumPy arrays, returning
    the index pairs that match by value.

    Parameters
    ----------
    a : ndarray
        The left-hand array to join on. Must be 1-dimensional.
    v : ndarray
        The right-hand array to join on. Must be 1-dimensional.

    Returns
    -------
    aidx : ndarray (int)
        Indices into `a` selecting the rows that participate in the join.
    vidx : ndarray (int)
        Indices into `v` selecting the corresponding matching rows.

        After the join:
            a[aidx] == v[vidx]
        is guaranteed to be true for all elements.

    Notes
    -----
    This function implements a pure NumPy equivalent of an SQL-style
    INNER JOIN on the key columns `a` and `v`.

    The algorithm:

    1. Sort `a` to produce a permutation `i` so that `a[i]` is sorted.
    2. Use `np.searchsorted(a[i], v)` to find, for each element of `v`,
       the candidate matching location in the sorted array.
    3. Map these positions back to the coordinates of the original array `a`
       using the permutation `i`.
    4. Filter out non-matches (values in `v` not present in `a`).
       The remaining pairs form the inner join.

    Complexity
    ----------
    Sorting:      O(len(a) log len(a))
    Searching:    O(len(v) log len(a))
    Total:        O(n log n)

    This is optimal for join-like operations on unsorted arrays in NumPy.

    Examples
    --------
    >>> a = np.array(["b", "a", "c", "b"])
    >>> v = np.array(["a", "b", "x", "b"])

    >>> aidx, vidx = argjoin(a, v)
    >>> a[aidx]
    array(['a', 'b', 'b'])
    >>> v[vidx]
    array(['a', 'b', 'b'])

    """
    # 1. Sort a, remembering the permutation
    i = np.argsort(a)
    ai = a[i]

    # 2. Locate each element of v within the sorted array
    idx = np.searchsorted(ai, v)

    # Clip to avoid out-of-range indices when v contains values > max(a)
    idx = np.clip(idx, 0, len(ai) - 1)

    # 3. Map positions in sorted array back to original array indices
    aidx_candidate = i[idx]

    # 4. Keep only true matches (this implements an INNER JOIN)
    mask = ai[idx] == v
    vidx = np.flatnonzero(mask)

    # Final matched indices in a
    aidx = aidx_candidate[vidx]

    assert np.all(a[aidx] == v[vidx])
    return aidx, vidx
