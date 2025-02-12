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

import hpgeom as hpg
import numpy as np


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


def obj_id_to_ss_object_id(objID, flags=0):
    """Convert from Minor Planet Center packed provisional object ID to
    Rubin ssObjectID.

    Parameters
    ----------
    objID : `str`
        Minor Planet Center packed provisional designation for a small solar
        system object. Must be fewer than eight characters.
    flags : `int`, optional
        Eight free bits to enable future decoupling between Minor Planet Center
        and Rubin. Zero by default, should not be changed unless we need to
        move away from a 1:1 mapping with the MPC. Must be within [0, 255].

    Returns
    -------
    ssObjectID : `int`
        Rubin ssObjectID

    Raises
    ------
    ValueError
        Raised if either objID is longer than 7 characters or flags is greater
        than 255 or less than 0.
    """
    if len(objID) > 7:
        raise ValueError(f'objID longer than 7 characters: "{objID}"')
    if len(objID) < 7:
        raise ValueError(f'objID shorter than 7 characters: "{objID}"')
    if flags < 0 or flags > 255:
        raise ValueError(f'Flags ({flags}) outside [0, 255].')
    if any([ord(c) > 255 for c in objID]):
        raise ValueError(f'{[c for c in objID if ord(c) > 255]} not legal objID characters (ascii [1, 255])')

    ssObjectID = flags
    for character in objID:
        ssObjectID <<= 8
        ssObjectID += ord(character)
    return ssObjectID


def ss_object_id_to_obj_id(ssObjectID):
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

    flags : `int`
        Rubin flags (not yet defined, but usable in case we decouple from MPC).

    Raises
    ------
    """
    if ssObjectID < 0 or ssObjectID >= (1 << 64):
        raise ValueError(f'ssObjectID ({ssObjectID}) outside [0, 2^64 - 1].')

    objID = ''.join([chr((ssObjectID >> (8 * i)) % 256) for i in reversed(range(0, 7))])
    return objID, ssObjectID >> (8 * 7) % 256
