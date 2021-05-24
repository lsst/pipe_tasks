import healpy as hp
import numpy as np


def toIndex(nside, ra, dec):
    """Return healpix index given ra,dec in degrees"""
    return hp.pixelfunc.ang2pix(nside, np.radians(-dec+90.), np.radians(ra))


def toRaDec(nside, index):
    """Convert from healpix index to ra,dec in degrees"""
    vec = hp.pix2ang(nside, index)
    dec = np.rad2deg(-vec[0])+90
    ra = np.rad2deg(vec[1])
    return np.dstack((ra, dec))[0]


def eq2xyz(ra, dec):
    """Convert from equatorial ra,dec in degrees to x,y,z on unit sphere"""
    phi = np.deg2rad(ra)
    theta = np.pi/2 - np.deg2rad(dec)
    sintheta = np.sin(theta)
    x = sintheta * np.cos(phi)
    y = sintheta * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def eq2vec(ra, dec):
    """Convert equatorial ra,dec in degrees to x,y,z on the unit sphere parameters"""
    ra = np.array(ra, dtype='f8', ndmin=1, copy=False)
    dec = np.array(dec, dtype='f8', ndmin=1, copy=False)
    if ra.size != dec.size:
        raise ValueError("ra,dec not same size: %s,%s" % (ra.size, dec.size))

    vec = eq2xyz(ra, dec)

    return vec


def convert_spherical(ra, dec):
    """Convert from ra,dec to spherical"""

    return np.dstack([np.cos(dec*np.pi/180.)*np.cos(ra*np.pi/180.),
                      np.cos(dec*np.pi/180.)*np.sin(ra*np.pi/180.),
                      np.sin(dec*np.pi/180.)])[0]


def convert_spherical_array(array):
    """Convert from ra,dec to spherical from array"""
    ra = array[:, 0]
    dec = array[:, 1]
    return convert_spherical(ra, dec)


def query_disc(nside, ra, dec, max_rad, min_rad=0):
    """
    Get the list of healpix indices within max_rad,min_rad given in radians
    around ra,dec given in degrees
    """
    if np.isscalar(ra):
        ra = np.array([ra])
        dec = np.array([dec])

    pixels = np.unique([hp.query_disc(nside, eq2vec(a, b), max_rad) for (a, b) in zip(ra, dec)])

    if min_rad > 0 and len(pixels) > 0:
        vec0 = convert_spherical(ra, dec)
        min_rad2 = min_rad**2
        vecs = convert_spherical_array(toRaDec(nside, pixels))
        dsq = np.sum((vecs-vec0)**2, axis=1)
        match = dsq > min_rad2
        pixels = pixels[match]

    return pixels
