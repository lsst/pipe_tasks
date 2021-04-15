#
# LSST Data Management System
# Copyright 2008-2021 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
from collections import defaultdict
import numbers
import numpy as np
import healpy as hp
import healsparse as hsp

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.geom
import lsst.afw.geom as afwGeom
from lsst.daf.butler import Formatter


__all__ = ["HealSparseInputMapTask", "HealSparseInputMapConfig",
           "HealSparseMapFormatter"]


class HealSparseMapFormatter(Formatter):
    """Interface for reading and writing healsparse.HealSparseMap files
    """
    unsupportedParameters = frozenset()
    supportedExtensions = frozenset({".hsp", ".fit", ".fits"})
    extension = '.hsp'

    def read(self, component=None):
        # Docstring inherited from Formatter.read.
        path = self.fileDescriptor.location.path

        if component == 'coverage':
            try:
                data = hsp.HealSparseCoverage.read(path)
            except (OSError, RuntimeError):
                raise ValueError(f"Unable to read healsparse map with URI {self.fileDescriptor.location.uri}")

            return data

        if self.fileDescriptor.parameters is None:
            pixels = None
            degrade_nside = None
        else:
            pixels = self.fileDescriptor.parameters.get('pixels', None)
            degrade_nside = self.fileDescriptor.parameters.get('degrade_nside', None)
        try:
            data = hsp.HealSparseMap.read(path, pixels=pixels, degrade_nside=degrade_nside)
        except (OSError, RuntimeError):
            raise ValueError(f"Unable to read healsparse map with URI {self.fileDescriptor.location.uri}")

        return data

    def write(self, inMemoryDataset):
        # Docstring inherited from Formatter.write.
        # Update the location with the formatter-preferred file extension
        self.fileDescriptor.location.updateExtension(self.extension)
        inMemoryDataset.write(self.fileDescriptor.location.path)


def _is_power_of_two(value):
    """Check that value is a power of two.

    Parameters
    ----------
    value : `int`
        Value to check.

    Returns
    -------
    is_power_of_two : `bool`
        True if value is a power of two; False otherwise, or
        if value is not an integer.
    """
    if not isinstance(value, numbers.Integral):
        return False

    # See https://stackoverflow.com/questions/57025836
    # Every power of 2 has exactly 1 bit set to 1; subtracting
    # 1 therefore flips every preceding bit.  If you and that
    # together with the original value it must be 0.
    return (value & (value - 1) == 0) and value != 0


class HealSparseInputMapConfig(pexConfig.Config):
    """Configuration parameters for HealSparseInputMapTask"""
    nside = pexConfig.Field(
        doc="Mapping healpix nside.  Must be power of 2.",
        dtype=int,
        default=32768,
        check=_is_power_of_two,
    )
    nside_coverage = pexConfig.Field(
        doc="HealSparse coverage map nside.  Must be power of 2.",
        dtype=int,
        default=256,
        check=_is_power_of_two,
    )
    bad_mask_min_coverage = pexConfig.Field(
        doc=("Minimum area fraction of a map healpixel pixel that must be "
             "covered by bad pixels to be removed from the input map. "
             "This is approximate."),
        dtype=float,
        default=0.5,
    )


class HealSparseInputMapTask(pipeBase.Task):
    """Task for making a HealSparse input map."""

    ConfigClass = HealSparseInputMapConfig
    _DefaultName = "healSparseInputMap"

    def __init__(self, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)

        self.ccd_input_map = None

    def build_ccd_input_map(self, bbox, wcs, ccds):
        """Build a map from ccd valid polygons or bounding boxes.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box for region to build input map.
        wcs : `lsst.afw.geom.SkyWcs`
            WCS object for region to build input map.
        ccds : `lsst.afw.table.ExposureCatalog`
            Exposure catalog with ccd data from coadd inputs.
        """
        self.ccd_input_map = hsp.HealSparseMap.make_empty(nside_coverage=self.config.nside_coverage,
                                                          nside_sparse=self.config.nside,
                                                          dtype=hsp.WIDE_MASK,
                                                          wide_mask_maxbits=len(ccds))
        self._wcs = wcs
        self._bbox = bbox
        self._ccds = ccds

        pixel_scale = wcs.getPixelScale().asArcseconds()
        hpix_area_arcsec2 = hp.nside2pixarea(self.config.nside, degrees=True)*(3600.**2.)
        self._min_bad = self.config.bad_mask_min_coverage*hpix_area_arcsec2/(pixel_scale**2.)

        metadata = {}
        self._bits_per_visit_ccd = {}
        self._bits_per_visit = defaultdict(list)
        for bit, ccd in enumerate(ccds):
            metadata[f'B{bit:04d}CCD'] = ccd['ccd']
            metadata[f'B{bit:04d}VIS'] = ccd['visit']
            metadata[f'B{bit:04d}WT'] = ccd['weight']

            self._bits_per_visit_ccd[(ccd['visit'], ccd['ccd'])] = bit
            self._bits_per_visit[ccd['visit']].append(bit)

            ccd_poly = ccd.getValidPolygon()
            if ccd_poly is None:
                ccd_poly = afwGeom.Polygon(lsst.geom.Box2D(ccd.getBBox()))
            # Detectors need to be rendered with their own wcs.
            ccd_poly_radec = self._pixels_to_radec(ccd.getWcs(), ccd_poly.convexHull().getVertices())

            # Create a ccd healsparse polygon
            poly = hsp.Polygon(ra=ccd_poly_radec[: -1, 0],
                               dec=ccd_poly_radec[: -1, 1],
                               value=[bit])
            self.ccd_input_map.set_bits_pix(poly.get_pixels(nside=self.ccd_input_map.nside_sparse),
                                            [bit])

        # Cut down to the overall bounding box with associated wcs.
        bbox_afw_poly = afwGeom.Polygon(lsst.geom.Box2D(bbox))
        bbox_poly_radec = self._pixels_to_radec(self._wcs,
                                                bbox_afw_poly.convexHull().getVertices())
        bbox_poly = hsp.Polygon(ra=bbox_poly_radec[: -1, 0], dec=bbox_poly_radec[: -1, 1],
                                value=np.arange(self.ccd_input_map.wide_mask_maxbits))
        bbox_poly_map = bbox_poly.get_map_like(self.ccd_input_map)
        self.ccd_input_map = hsp.and_intersection([self.ccd_input_map, bbox_poly_map])
        self.ccd_input_map.metadata = metadata

        # Create a temporary map to hold the count of bad pixels in each healpix pixel
        self._ccd_input_pixels = self.ccd_input_map.valid_pixels

        dtype = [(f'v{visit}', 'i4') for visit in self._bits_per_visit.keys()]

        cov = self.config.nside_coverage
        ns = self.config.nside
        self._ccd_input_bad_count_map = hsp.HealSparseMap.make_empty(nside_coverage=cov,
                                                                     nside_sparse=ns,
                                                                     dtype=dtype,
                                                                     primary=dtype[0][0])
        # Don't set input bad map if there are no ccds which overlap the bbox.
        if len(self._ccd_input_pixels) > 0:
            self._ccd_input_bad_count_map[self._ccd_input_pixels] = np.zeros(1, dtype=dtype)

    def mask_warp_bbox(self, bbox, visit, mask, bit_mask_value):
        """Mask a subregion from a visit.
        This must be run after build_ccd_input_map initializes
        the overall map.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box from region to mask.
        visit : `int`
            Visit number corresponding to warp with mask.
        mask : `lsst.afw.image.MaskX`
            Mask plane from warp exposure.
        bit_mask_value : `int`
            Bit mask to check for bad pixels.

        Raises
        ------
        RuntimeError : Raised if build_ccd_input_map was not run first.
        """
        if self.ccd_input_map is None:
            raise RuntimeError("Must run build_ccd_input_map before mask_warp_bbox")

        # Find the bad pixels and convert to healpix
        bad_pixels = np.where(mask.array & bit_mask_value)
        if len(bad_pixels[0]) == 0:
            # No bad pixels
            return

        # Bad pixels come from warps which use the overall wcs.
        bad_ra, bad_dec = self._wcs.pixelToSkyArray(bad_pixels[1].astype(np.float64),
                                                    bad_pixels[0].astype(np.float64),
                                                    degrees=True)
        bad_hpix = hp.ang2pix(self.config.nside, bad_ra, bad_dec,
                              lonlat=True, nest=True)

        # Count the number of bad image pixels in each healpix pixel
        min_bad_hpix = bad_hpix.min()
        bad_hpix_count = np.zeros(bad_hpix.max() - min_bad_hpix + 1, dtype=np.int32)
        np.add.at(bad_hpix_count, bad_hpix - min_bad_hpix, 1)

        # Add these to the accumulator map.
        # We need to make sure that the "primary" array has valid values for
        # this pixel to be registered in the accumulator map.
        pix_to_add, = np.where(bad_hpix_count > 0)
        count_map_arr = self._ccd_input_bad_count_map[min_bad_hpix + pix_to_add]
        primary = self._ccd_input_bad_count_map.primary
        count_map_arr[primary] = np.clip(count_map_arr[primary], 0, None)

        count_map_arr[f'v{visit}'] = np.clip(count_map_arr[f'v{visit}'], 0, None)
        count_map_arr[f'v{visit}'] += bad_hpix_count[pix_to_add]

        self._ccd_input_bad_count_map[min_bad_hpix + pix_to_add] = count_map_arr

    def finalize_ccd_input_map_mask(self):
        """Use accumulated mask information to finalize the masking of
        ccd_input_map.

        Raises
        ------
        RuntimeError : Raised if build_ccd_input_map was not run first.
        """
        if self.ccd_input_map is None:
            raise RuntimeError("Must run build_ccd_input_map before finalize_ccd_input_map_mask.")

        count_map_arr = self._ccd_input_bad_count_map[self._ccd_input_pixels]
        for visit in self._bits_per_visit:
            to_mask, = np.where(count_map_arr[f'v{visit}'] > self._min_bad)
            if to_mask.size == 0:
                continue
            self.ccd_input_map.clear_bits_pix(self._ccd_input_pixels[to_mask],
                                              self._bits_per_visit[visit])

        # Clear memory
        self._ccd_input_bad_count_map = None

    def _pixels_to_radec(self, wcs, pixels):
        """Convert pixels to ra/dec positions using a wcs.

        Parameters
        ----------
        wcs : `lsst.afw.geom.SkyWcs`
            WCS object.
        pixels : `list` [`lsst.geom.Point2D`]
            List of pixels to convert.

        Returns
        -------
        radec : `numpy.ndarray`
            Nx2 array of ra/dec positions associated with pixels.
        """
        sph_pts = wcs.pixelToSky(pixels)
        return np.array([(sph.getRa().asDegrees(), sph.getDec().asDegrees())
                         for sph in sph_pts])
