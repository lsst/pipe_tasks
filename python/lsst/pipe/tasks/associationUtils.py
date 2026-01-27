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

"""Utilities for interfacing with hpgeom. Originally implemented in
http://github.com/LSSTDESC/dia_pipe and then translated to hpgeom.
"""

__all__ = ["toIndex", "toRaDec", "eq2xyz", "eq2xyzVec", "convert_spherical",
           "convert_spherical_array", "query_disc", "obj_id_to_ss_object_id", "ss_object_id_to_obj_id"]

from astropy.time import Time
import hpgeom as hpg
import numpy as np
import numpy.typing as npt


def toIndex(nside, ra, dec):
    """Return healpix index given ra, dec in degrees

    Parameters
    ----------
    nside : `int`
        Power of 2 nside healpix resolution.
    ra : `float`
        RA in degrees.
    dec : `float`
        Declination in degrees

    Returns
    -------
    index : `int`
        Unique healpix pixel ID containing point RA, DEC at resolution nside.
    """
    return hpg.angle_to_pixel(nside, ra, dec, nest=False)


def toRaDec(nside, index):
    """Convert from healpix index to ra,dec in degrees

    Parameters
    ----------
    nside : `int`
        Resolution of healpixel "grid".
    index : `int`
        Index of the healpix pixel we want to find the location of.

    Returns
    -------
    pos : `numpy.ndarray`, (2,)
        RA and DEC of healpix pixel location in degrees.
    """
    ra, dec = hpg.pixel_to_angle(nside, index, nest=False)
    return np.dstack((ra, dec))[0]


def eq2xyz(ra, dec):
    """Convert from equatorial ra,dec in degrees to x,y,z on unit sphere.

    Parameters
    ----------
    ra : `float`
        RA in degrees.
    dec : `float`
        Declination in degrees

    Returns
    -------
    xyz : `numpy.ndarray`, (3,)
        Float xyz positions on the unit sphere.
    """
    phi = np.deg2rad(ra)
    theta = np.pi/2 - np.deg2rad(dec)
    sintheta = np.sin(theta)
    x = sintheta*np.cos(phi)
    y = sintheta*np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def eq2xyzVec(ra, dec):
    """Convert equatorial ra,dec in degrees to x,y,z on the unit sphere
    parameters

    Vectorized version of ``eq2xyz``

    Parameters
    ----------
    ra : array_like, (N,)
        Array of RA in degrees.
    dec : array_like, (N,)
        Declination in degrees

    Returns
    -------
    vec : `numpy.ndarray`, (N,3)
        Array of unitsphere 3-vectors.
    """
    ra = np.array(ra, dtype='f8', ndmin=1, copy=False)
    dec = np.array(dec, dtype='f8', ndmin=1, copy=False)
    if ra.size != dec.size:
        raise ValueError("ra,dec not same size: %s,%s" % (ra.size, dec.size))

    vec = eq2xyz(ra, dec)

    return vec


def convert_spherical(ra, dec):
    """Convert from ra,dec to spherical coordinates.

    Used in query_disc.

    Parameters
    ----------
    ra : `float`
        RA in radians.
    dec : `float`
        Declination in radians
    """
    return np.dstack([np.cos(dec*np.pi/180)*np.cos(ra*np.pi/180),
                      np.cos(dec*np.pi/180)*np.sin(ra*np.pi/180),
                      np.sin(dec*np.pi/180)])[0]


def convert_spherical_array(array):
    """Convert from and a array ra,dec to spherical coordinates.

    Used in query_disc

    Parameters
    ----------
    array : `numpy.ndarray`, (N, 2)
        (N, 2) Array of RA, DEC values.

    Returns
    -------
    vecs : `numpy.ndarray`, (N, 3)
        Vectors on the unit sphere
    """
    ra = array[:, 0]
    dec = array[:, 1]
    return convert_spherical(ra, dec)


def query_disc(nside, ra, dec, max_rad, min_rad=0):
    """Get the list of healpix indices within max_rad, min_rad given in radians
    around ra,dec given in degrees

    Parameters
    ----------
    nside : `int`
        Resolution of the healpixels to search/return.
    ra : `float`
        RA in degrees.
    dec : `float`
        Declination in degrees
    max_rad : `float`
        Max distance in radians to search nearby healpixels.
    min_rad : `float`, optional
        Minimum distance in radians to search healpixels. Default = 0.
    """
    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)

    max_rad_deg = np.rad2deg(max_rad)

    pixels = np.unique(
        [hpg.query_circle(nside, a, b, max_rad_deg, nest=False)
         for (a, b) in zip(ra, dec)])

    if min_rad > 0 and len(pixels) > 0:
        vec0 = convert_spherical(ra, dec)
        min_rad2 = min_rad**2
        vecs = convert_spherical_array(toRaDec(nside, pixels))
        dsq = np.sum((vecs - vec0)**2, axis=1)
        match = dsq > min_rad2
        pixels = pixels[match]

    return pixels


def obj_id_to_ss_object_id(objID):
    if ' ' in objID:
        objID = pack_provisional_designation(objID)
    return packed_obj_id_to_ss_object_id(objID)


def packed_obj_id_to_ss_object_id(objID):
    """Convert from Minor Planet Center packed provisional object ID to
    Rubin ssObjectID.

    Parameters
    ----------
    objID : `str`
        Minor Planet Center packed provisional designation for a small solar
        system object. Must be fewer than eight characters.

    Returns
    -------
    ssObjectID : `int`
        Rubin ssObjectID

    Raises
    ------
    ValueError
        Raised if either objID is shorter than 7 or longer than 8 characters or contains
        illegal objID characters
    """
    if len(objID) > 8:
        raise ValueError(f'objID longer than 8 characters: "{objID}"')
    if len(objID) < 7:
        raise ValueError(f'objID shorter than 7 characters: "{objID}"')
    if any([ord(c) > 255 for c in objID]):
        raise ValueError(f'{[c for c in objID if ord(c) > 255]} not legal objID characters (ascii [1, 255])')

    ssObjectID = ord(objID[0])
    for character in objID[1:]:
        ssObjectID <<= 8
        ssObjectID += ord(character)
    return ssObjectID


def ss_object_id_to_obj_id(ssObjectID, packed=False):
    """Convert from Rubin ssObjectID to Minor Planet Center packed provisional
    object ID.

    Parameters
    ----------
    ssObjectID : `int`
        Rubin ssObjectID

    Returns
    -------
    objID : `str`
        Minor Planet Center packed provisional designation.

    Raises
    ------
    """
    if ssObjectID < 0 or ssObjectID >= (1 << 64):
        raise ValueError(f'ssObjectID ({ssObjectID}) outside [0, 2^64 - 1].')

    objID = ''.join([chr((ssObjectID >> (8 * i)) % 256) for i in reversed(range(0, 8))])
    objID = objID.replace('\x00', '')
    if packed:
        return objID
    else:
        return unpack_provisional_designation(objID)

# All the below designation-related code are copied from B612's adam_core
# adam_core should eventually be added as an external dependency, and this
# should be replaced with imports on DM-53907


def pack_mpc_designation(designation: str) -> str:
    """
    B612 code to pack a unpacked MPC designation. For example,
    provisional designation 1998 SS162 will be packed to J98SG2S.
    Permanent designation 323 will be packed to 00323.

    TODO: add support for comet and natural satellite designations

    Parameters
    ----------
    designation : str
        MPC unpacked designation.

    Returns
    -------
    designation_pf : str
        MPC packed form designation.

    Raises
    ------
    ValueError : If designation cannot be packed.
    """
    # Lets see if its a numbered object
    try:
        return pack_numbered_designation(designation)
    except ValueError:
        pass

    # If its not numbered, maybe its a provisional designation
    try:
        return pack_provisional_designation(designation)
    except ValueError:
        pass

    # If its a survey designation, deal with it
    try:
        return pack_survey_designation(designation)
    except ValueError:
        pass

    err = (
        "Unpacked designation '{}' could not be packed.\n"
        "It could not be recognized as any of the following:\n"
        " - a numbered object (e.g. '3202', '203289', '3140113')\n"
        " - a provisional designation (e.g. '1998 SV127', '2008 AA360')\n"
        " - a survey designation (e.g. '2040 P-L', '3138 T-1')"
    )
    raise ValueError(err.format(designation))


BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BASE62_MAP = {BASE62[i]: i for i in range(len(BASE62))}


def _unpack_mpc_date(epoch_pf: str) -> Time:
    # Taken from Lynne Jones' SSO TOOLS.
    # See https://minorplanetcenter.net/iau/info/PackedDates.html
    # for MPC documentation on packed dates.
    # Examples:
    #    1998 Jan. 18.73     = J981I73
    #    2001 Oct. 22.138303 = K01AM138303
    epoch_pf = str(epoch_pf)
    year = int(epoch_pf[0], base=32) * 100 + int(epoch_pf[1:3])
    month = int(epoch_pf[3], base=32)
    day = int(epoch_pf[4], base=32)
    isot_string = "{:d}-{:02d}-{:02d}".format(year, month, day)

    if len(epoch_pf) > 5:
        fractional_day = float("." + epoch_pf[5:])
        hours = int((24 * fractional_day))
        minutes = int(60 * ((24 * fractional_day) - hours))
        seconds = 3600 * (24 * fractional_day - hours - minutes / 60)
        isot_string += "T{:02d}:{:02d}:{:09.6f}".format(hours, minutes, seconds)

    return Time(isot_string, format="isot", scale="tt")


def convert_mpc_packed_dates(pf_tt: npt.ArrayLike) -> Time:
    """
    Convert MPC packed form dates (in the TT time scale) to
    MJDs in TT. See: https://minorplanetcenter.net/iau/info/PackedDates.html
    for details on the packed date format.

    Parameters
    ----------
    pf_tt : `~numpy.ndarray` (N)
        MPC-style packed form epochs in the TT time scale.

    Returns
    -------
    mjd_tt : `~astropy.time.core.Time` (N)
        Epochs in TT MJDs.
    """
    isot_tt = []
    for epoch in pf_tt:
        isot_tt.append(_unpack_mpc_date(epoch))

    return Time(isot_tt)


def pack_numbered_designation(designation: str) -> str:
    """
    Pack a numbered MPC designation.

    Examples of numbered designations:
        Numbered      Packed
        3202          03202
        50000         50000
        100345        A0345
        360017        a0017
        203289        K3289
        620000        ~0000
        620061        ~000z
        3140113       ~AZaz
        15396335      ~zzzz

    Parameters
    ----------
    designation : str
        MPC numbered designation.

    Returns
    -------
    designation_pf : str
        MPC packed numbered designation.

    Raises
    ------
    ValueError : If the numbered designation cannot be packed.
        If the numbered designation is larger than 15396335.
    """
    number = int(designation)
    if number > 15396335:
        raise ValueError(
            "Numbered designation is too large. Maximum supported is 15396335."
        )

    if number <= 99999:
        return "{:05}".format(number)
    elif (number >= 100000) and (number <= 619999):
        bigpart, remainder = divmod(number, 10000)
        return f"{BASE62[bigpart]}{remainder:04}"
    else:
        x = number - 620000
        number_pf = []
        while x:
            number_pf.append(BASE62[int(x % 62)])
            x //= 62

        number_pf.reverse()
        return "~{}".format("".join(number_pf).zfill(4))


def pack_provisional_designation(designation: str) -> str:
    """
    Pack a provisional MPC designation.

    Examples of provisional designations:
        Provisional   Packed
        1995 XA       J95X00A
        1995 XL1      J95X01L
        1995 FB13     J95F13B
        1998 SQ108    J98SA8Q
        1998 SV127    J98SC7V
        1998 SS162    J98SG2S
        2099 AZ193    K99AJ3Z
        2008 AA360    K08Aa0A
        2007 TA418    K07Tf8A

    Parameters
    ----------
    designation : str
        MPC provisional designation.

    Returns
    -------
    designation_pf : str
        MPC packed provisional designation.

    Raises
    ------
    ValueError : If the provisional designation cannot be packed.
        The provisional designations is not at least 6 characters long.
        The first 4 characters of the provisional designation are not a year.
        The 5th character of the provisional designation is not a space.
        The provisional designation contains a hyphen.
        The half-month letter is I or Z.
    """
    if len(designation) < 6:
        raise ValueError(
            "Provisional designations should be at least 6 characters long."
        )
    if not designation[:3].isdecimal():
        raise ValueError(
            "Expected the first 4 characters of the provisional designation to be a year."
        )
    if designation[4] != " ":
        raise ValueError(
            "Expected the 5th character of the provisional designation to be a space."
        )
    if "-" in designation:
        raise ValueError("Provisional designations cannot contain a hyphen.")

    year = BASE62[int(designation[0:2])] + designation[2:4]
    letter1 = designation[5]
    letter2 = designation[6]
    cycle = designation[7:]

    if letter1 in {"I", "Z"}:
        raise ValueError("Half-month letters cannot be I or Z.")
    if letter1.isdecimal() or letter2.isdecimal():
        raise ValueError("Invalid provisional designation.")

    cycle_pf = "00"
    if len(cycle) > 0:
        cycle_int = int(cycle)
        if cycle_int <= 99:
            cycle_pf = str(cycle_int).zfill(2)
        else:
            cycle_pf = BASE62[cycle_int // 10] + str(cycle_int % 10)

    designation_pf = "{}{}{}{}".format(year, letter1, cycle_pf, letter2)
    return designation_pf


def pack_survey_designation(designation: str) -> str:
    """
    Pack a survey MPC designation.

    Examples of survey designations:
        Survey       Packed
        2040 P-L     PLS2040
        3138 T-1     T1S3138
        1010 T-2     T2S1010
        4101 T-3     T3S4101

    Parameters
    ----------
    designation : str
        MPC survey designation.

    Returns
    -------
    designation_pf : str
        MPC packed survey designation.

    Raises
    ------
    ValueError : If the survey designation cannot be packed.
        The survey designation does not start with P-L, T-1, T-2, or T-3.
    """
    number = designation[0:4]
    survey = designation[5:]

    if survey == "P-L":
        survey_pf = "PLS"

    elif survey[0:2] == "T-" and survey[2] in {"1", "2", "3"}:
        survey_pf = "T{}S".format(survey[2])

    else:
        raise ValueError("Survey designations must start with P-L, T-1, T-2, T-3.")

    designation_pf = "{}{}".format(survey_pf, number.zfill(4))
    return designation_pf


def unpack_numbered_designation(designation_pf: str) -> str:
    """
    Unpack a numbered MPC designation.

    Examples of numbered designations:
        Numbered      Unpacked
        03202         3202
        50000         50000
        A0345         100345
        a0017         360017
        K3289         203289
        ~0000         620000
        ~000z         620061
        ~AZaz         3140113
        ~zzzz         15396335

    Parameters
    ----------
    designation_pf : str
        MPC packed numbered designation.

    Returns
    -------
    designation : str
        MPC unpacked numbered designation.

    Raises
    ------
    ValueError : If the numbered designation cannot be unpacked.
        The packed numbered designation is not at least 4 characters long.
    """
    number = None
    # Numbered objects (1 - 99999)
    if designation_pf.isdecimal():
        number = int(designation_pf)

    # Numbered objects (620000+)
    elif designation_pf[0] == "~":
        number = 620000
        number_pf = designation_pf[1:]
        for i, c in enumerate(number_pf):
            power = len(number_pf) - (i + 1)
            number += BASE62_MAP[c] * (62**power)

    # Numbered objects (100000 - 619999)
    else:
        number = BASE62_MAP[designation_pf[0]] * 10000 + int(designation_pf[1:])

    if number is None:
        raise ValueError("Packed numbered designation could not be unpacked.")
    else:
        designation = str(number)

    return designation


def unpack_provisional_designation(designation_pf: str) -> str:
    """
    Unpack a provisional MPC designation.

    Examples of provisional designations:
        Provisional   Unpacked
        J95X00A       1995 XA
        J95X01L       1995 XL1
        J95F13B       1995 FB13
        J98SA8Q       1998 SQ108
        J98SC7V       1998 SV127
        J98SG2S       1998 SS162
        K99AJ3Z       2099 AZ193
        K08Aa0A       2008 AA360
        K07Tf8A       2007 TA418

    Parameters
    ----------
    designation_pf : str
        MPC packed provisional designation.

    Returns
    -------
    designation : str
        MPC unpacked provisional designation.

    Raises
    ------
    ValueError : If the provisional designation cannot be unpacked.
        The packed provisional designation is not 7 characters long.
        The packed provisional designation does not have a year.
    """
    if len(designation_pf) != 7:
        raise ValueError("Provisional designation must be 7 characters long.")
    if not designation_pf[1].isdecimal() or not designation_pf[2].isdecimal():
        raise ValueError("Provisional designation must have a year.")
    year = str(BASE62_MAP[designation_pf[0]] * 100 + int(designation_pf[1:3]))
    letter1 = designation_pf[3]
    letter2 = designation_pf[6]
    if letter1.isdecimal() or letter2.isdecimal():
        raise ValueError()
    cycle1 = designation_pf[4]
    cycle2 = designation_pf[5]

    number = int(BASE62_MAP[cycle1]) * 10 + BASE62_MAP[cycle2]
    if number == 0:
        number_str = ""
    else:
        number_str = str(number)

    designation = "{} {}{}{}".format(year, letter1, letter2, number_str)

    return designation


def unpack_survey_designation(designation_pf: str) -> str:
    """
    Unpack a survey MPC designation.

    Examples of survey designations:
        Survey       Packed
        PLS2040      2040 P-L
        T1S3138      3138 T-1
        T2S1010      1010 T-2
        T3S4101      4101 T-3

    Parameters
    ----------
    designation_pf : str
        MPC packed survey designation.

    Returns
    -------
    designation : str
        MPC unpacked survey designation.

    Raises
    ------
    ValueError : If the survey designation cannot be unpacked.
        The packed survey designation does not start with PLS, T1S, T2S, or T3S.
    """
    number = int(designation_pf[3:8])
    survey_pf = designation_pf[0:3]
    if survey_pf not in {"PLS", "T1S", "T2S", "T3S"}:
        raise ValueError(
            "Packed survey designation must start with PLS, T1S, T2S, or T3S."
        )

    if survey_pf == "PLS":
        survey = "P-L"

    if survey_pf[0] == "T" and survey_pf[2] == "S":
        survey = "T-{}".format(survey_pf[1])

    designation = "{} {}".format(number, survey)
    return designation


def unpack_mpc_designation(designation_pf: str) -> str:
    """
    Unpack a packed MPC designation. For example, provisional
    designation J98SG2S will be unpacked to 1998 SS162. Permanent
    designation 00323 will be unpacked to 323.

    TODO: add support for comet and natural satellite designations

    Parameters
    ----------
    designation_pf : str
        MPC packed form designation.

    Returns
    -------
    designation : str
        MPC unpacked designation.

    Raises
    ------
    ValueError : If designation_pf cannot be unpacked.
    """
    # Lets see if its a numbered object
    try:
        return unpack_numbered_designation(designation_pf)
    except ValueError:
        pass

    # Lets see if its a provisional designation
    try:
        return unpack_provisional_designation(designation_pf)
    except ValueError:
        pass

    # Lets see if its a survey designation
    try:
        return unpack_survey_designation(designation_pf)
    except ValueError:
        pass

    # At this point we haven't had any success so lets raise an error
    err = (
        "Packed form designation '{}' could not be unpacked.\n"
        "It could not be recognized as any of the following:\n"
        " - a numbered object (e.g. '03202', 'K3289', '~AZaz')\n"
        " - a provisional designation (e.g. 'J98SC7V', 'K08Aa0A')\n"
        " - a survey designation (e.g. 'PLS2040', 'T1S3138')"
    )
    raise ValueError(err.format(designation_pf))
