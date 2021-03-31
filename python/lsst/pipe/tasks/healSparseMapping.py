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
import numpy as np
import healpy as hp
import healsparse as hsp

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.geom
import lsst.afw.geom as afwGeom


__all__ = ["HealSparseInputMapTask", "HealSparseInputMapConfig"]


class HealSparseInputMapConfig(pexConfig.Config):
    """Configuration parameters for HealSparseInputMapTask"""
    nside = pexConfig.Field(
        doc="Mapping healpix nside.  Must be power of 2.",
        dtype=int,
        default=32768,
    )
    nsideCoverage = pexConfig.Field(
        doc="HealSparse coverage map nside.  Must be power of 2.",
        dtype=int,
        default=256,
    )
    badMaskMinCoverage = pexConfig.Field(
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

        self.patchInputMap = None

    def buildCcdInputMap(self, bbox, wcs, ccds):
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
        self.ccdInputMap = hsp.HealSparseMap.make_empty(nside_coverage=self.config.nsideCoverage,
                                                        nside_sparse=self.config.nside,
                                                        dtype=hsp.WIDE_MASK,
                                                        wide_mask_maxbits=len(ccds))
        self._wcs = wcs
        self._bbox = bbox
        self._ccds = ccds

        pixelScale = wcs.getPixelScale().asArcseconds()
        hpixAreaArcsec2 = hp.nside2pixarea(self.config.nside, degrees=True)*(3600.**2.)
        self._minBad = self.config.badMaskMinCoverage*hpixAreaArcsec2/(pixelScale**2.)

        metadata = {}
        self._bitsPerVisitCcd = {}
        self._bitsPerVisit = defaultdict(list)
        for bit, ccd in enumerate(ccds):
            metadata[f'B{bit:04d}CCD'] = ccd['ccd']
            metadata[f'B{bit:04d}VIS'] = ccd['visit']
            metadata[f'B{bit:04d}WT'] = ccd['weight']

            self._bitsPerVisitCcd[(ccd['visit'], ccd['ccd'])] = bit
            self._bitsPerVisit[ccd['visit']].append(bit)

            ccdPoly = ccd.getValidPolygon()
            if ccdPoly is None:
                ccdPoly = afwGeom.Polygon(lsst.geom.Box2D(ccd.getBBox()))
            ccdPolyRadec = self._pixels_to_radec(wcs, ccdPoly.convexHull().getVertices())

            # Create a ccd healsparse polygon
            poly = hsp.Polygon(ra=ccdPolyRadec[: -1, 0],
                               dec=ccdPolyRadec[: -1, 1],
                               value=[bit])
            self.ccdInputMap.set_bits_pix(poly.get_pixels(nside=self.ccdInputMap.nside_sparse),
                                          [bit])

        # Cut down to the overall bounding box
        bboxAfwPoly = afwGeom.Polygon(lsst.geom.Box2D(bbox))
        bboxPolyRadec = self._pixels_to_radec(wcs, bboxAfwPoly.convexHull().getVertices())
        bboxPoly = hsp.Polygon(ra=bboxPolyRadec[: -1, 0], dec=bboxPolyRadec[: -1, 1],
                               value=np.arange(self.ccdInputMap.wide_mask_maxbits))
        bboxPolyMap = bboxPoly.get_map_like(self.ccdInputMap)
        self.ccdInputMap = hsp.and_intersection([self.ccdInputMap, bboxPolyMap])
        self.ccdInputMap.metadata = metadata

        # Create a temporary map to hold the count of bad pixels in each healpix pixel
        self._ccdInputPixels = self.ccdInputMap.valid_pixels

        dtype = [(f'v{visit}', 'i4') for visit in self._bitsPerVisit.keys()]

        self._ccdInputBadCountMap = hsp.HealSparseMap.make_empty(nside_coverage=self.config.nsideCoverage,
                                                                 nside_sparse=self.config.nside,
                                                                 dtype=dtype,
                                                                 primary=dtype[0][0])

        self._ccdInputBadCountMap[self._ccdInputPixels] = np.zeros(1, dtype=dtype)

    def maskWarpBBox(self, bbox, visit, mask, bitMaskValue):
        """Mask a subregion from a visit.  This must be run after
        buildCcdInputMap initializes the overall map.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box from region to mask.
        visit : `int`
            Visit number corresponding to warp with mask.
        mask : `lsst.afw.image.MaskX`
            Mask plane from warp exposure.
        bitMaskValue : `int`
            Bit mask to check for bad pixels.
        """
        if self.ccdInputMap is None:
            raise RuntimeError("Must run buildCcdInputMap before maskWarpBBox")

        # Find the bad pixels and convert to healpix
        badPixels = np.where(mask.getArray() & bitMaskValue)
        if len(badPixels[0]) == 0:
            # No bad pixels
            return

        badRa, badDec = self._wcs.pixelToSkyArray(badPixels[1].astype(np.float64),
                                                  badPixels[0].astype(np.float64),
                                                  degrees=True)
        badHpix = hp.ang2pix(self.config.nside, badRa, badDec,
                             lonlat=True, nest=True)

        # Count the number of bad image pixels in each healpix pixel
        minBadHpix = badHpix.min()
        badHpixCount = np.zeros(badHpix.max() - minBadHpix + 1, dtype=np.int32)
        np.add.at(badHpixCount, badHpix - minBadHpix, 1)

        # Add these to the accumulator map.
        # We need to make sure that the "primary" array has valid values for
        # this pixel to be registered in the accumulator map.
        pixToAdd, = np.where(badHpixCount > 0)
        countMapArr = self._ccdInputBadCountMap[minBadHpix + pixToAdd]
        primary = self._ccdInputBadCountMap.primary
        countMapArr[primary] = np.clip(countMapArr[primary], 0, None)

        countMapArr[f'v{visit}'] = np.clip(countMapArr[f'v{visit}'], 0, None)
        countMapArr[f'v{visit}'] += badHpixCount[pixToAdd]

        self._ccdInputBadCountMap[minBadHpix + pixToAdd] = countMapArr

    def finalizeCcdInputMapMask(self):
        """Use accumulated mask information to finalize the masking of
        ccdInputMap.
        """
        countMapArr = self._ccdInputBadCountMap[self._ccdInputPixels]
        for visit in self._bitsPerVisit:
            toMask, = np.where(countMapArr[f'v{visit}'] > self._minBad)
            if toMask.size == 0:
                continue
            self.ccdInputMap.clear_bits_pix(self._ccdInputPixels[toMask],
                                            self._bitsPerVisit[visit])

        # Clear memory
        self._ccdInputBadCountMap = None

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
        sphPts = wcs.pixelToSky(pixels)
        return np.array([(sph.getRa().asDegrees(), sph.getDec().asDegrees())
                         for sph in sphPts])
