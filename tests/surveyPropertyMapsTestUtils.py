# This file is part of pipe_tasks.
#
# LSST Data Management System
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
# See COPYRIGHT file at the top of the source tree.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
"""Utilities for HealSparsePropertyMapTask and others."""
import numpy as np

import lsst.daf.butler
import lsst.geom as geom
from lsst.daf.base import DateTime
from lsst.afw.coord import Observatory
from lsst.pipe.tasks.postprocess import ConsolidateVisitSummaryTask
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
from lsst.afw.detection import GaussianPsf


__all__ = ['makeMockVisitSummary', 'MockVisitSummaryReference', 'MockCoaddReference',
           'MockInputMapReference']


def makeMockVisitSummary(visit,
                         ra_center=0.0,
                         dec_center=-45.0,
                         physical_filter='TEST-I',
                         band='i',
                         mjd=59234.7083333334,
                         psf_sigma=3.0,
                         zenith_distance=45.0,
                         zero_point=30.0,
                         sky_background=100.0,
                         sky_noise=10.0,
                         mean_var=100.0,
                         exposure_time=100.0,
                         detector_size=200,
                         pixel_scale=0.2):
    """Make a mock visit summary catalog.

    This will contain two square detectors with the same metadata,
    with a small (20 pixel) gap between the detectors.  There is no
    rotation, as each detector is simply offset in RA from the
    specified boresight.

    Parameters
    ----------
    visit : `int`
        Visit number.
    ra_center : `float`
        Right ascension of the center of the "camera" boresight (degrees).
    dec_center : `float`
        Declination of the center of the "camera" boresight (degrees).
    physical_filter : `str`
        Arbitrary name for the physical filter.
    band : `str`
        Name of the associated band.
    mjd : `float`
        Modified Julian Date.
    psf_sigma : `float`
        Sigma width of Gaussian psf.
    zenith_distance : `float`
        Distance from zenith of the visit (degrees).
    zero_point : `float`
        Constant zero point for the visit (magnitudes).
    sky_background : `float`
        Background level for the visit (counts).
    sky_noise : `float`
        Noise level for the background of the visit (counts).
    mean_var : `float`
        Mean of the variance plane of the visit (counts).
    exposure_time : `float`
        Exposure time of the visit (seconds).
    detector_size : `int`
        Size of each square detector in the visit (pixels).
    pixel_scale : `float`
        Size of the pixel in arcseconds per pixel.

    Returns
    -------
    visit_summary : `lsst.afw.table.ExposureCatalog`
    """
    # We are making a 2 detector "camera"
    n_detector = 2

    schema = ConsolidateVisitSummaryTask().schema
    visit_summary = afwTable.ExposureCatalog(schema)
    visit_summary.resize(n_detector)

    bbox = geom.Box2I(x=geom.IntervalI(min=0, max=detector_size - 1),
                      y=geom.IntervalI(min=0, max=detector_size - 1))

    for detector_id in range(n_detector):
        row = visit_summary[detector_id]

        row['id'] = detector_id
        row.setBBox(bbox)
        row['visit'] = visit
        row['physical_filter'] = physical_filter
        row['band'] = band
        row['zenithDistance'] = zenith_distance
        row['zeroPoint'] = zero_point
        row['skyBg'] = sky_background
        row['skyNoise'] = sky_noise
        row['meanVar'] = mean_var

        # Generate a photocalib
        instFluxMag0 = 10.**(zero_point/2.5)
        row.setPhotoCalib(afwImage.makePhotoCalibFromCalibZeroPoint(instFluxMag0))

        # Generate a WCS and set values accordingly
        crpix = geom.Point2D(detector_size/2., detector_size/2.)
        # Create a 20 pixel gap between the two detectors (each offset 10 pixels).
        if detector_id == 0:
            delta_ra = -1.0*((detector_size + 10)*pixel_scale/3600.)/np.cos(np.deg2rad(dec_center))
            delta_dec = 0.0
        elif detector_id == 1:
            delta_ra = ((detector_size + 10)*pixel_scale/3600.)/np.cos(np.deg2rad(dec_center))
            delta_dec = 0.0
        crval = geom.SpherePoint(ra_center + delta_ra, dec_center + delta_dec, geom.degrees)
        cd_matrix = afwGeom.makeCdMatrix(scale=pixel_scale*geom.arcseconds, orientation=0.0*geom.degrees)
        wcs = afwGeom.makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cd_matrix)
        row.setWcs(wcs)

        sph_pts = wcs.pixelToSky(geom.Box2D(bbox).getCorners())
        row['raCorners'] = np.array([float(sph.getRa().asDegrees()) for sph in sph_pts])
        row['decCorners'] = np.array([float(sph.getDec().asDegrees()) for sph in sph_pts])
        sph_pt = wcs.pixelToSky(bbox.getCenter())
        row['ra'] = sph_pt.getRa().asDegrees()
        row['decl'] = sph_pt.getDec().asDegrees()

        # Generate a visitInfo.
        # This does not need to be consistent with the zenith angle in the table,
        # it just needs to be valid and have sufficient information to compute
        # exposure time and parallactic angle.
        date = DateTime(date=mjd, system=DateTime.DateSystem.MJD)
        visit_info = afwImage.VisitInfo(exposureId=visit,
                                        exposureTime=exposure_time,
                                        date=date,
                                        darkTime=0.0,
                                        boresightRaDec=geom.SpherePoint(ra_center,
                                                                        dec_center,
                                                                        geom.degrees),
                                        era=45.1*geom.degrees,
                                        observatory=Observatory(
                                            11.1*geom.degrees,
                                            0.0*geom.degrees,
                                            0.333),
                                        boresightRotAngle=0.0*geom.degrees,
                                        rotType=afwImage.RotType.SKY)
        row.setVisitInfo(visit_info)

        # Generate a PSF and set values accordingly
        psf = GaussianPsf(15, 15, psf_sigma)
        row.setPsf(psf)
        psfAvgPos = psf.getAveragePosition()
        shape = psf.computeShape(psfAvgPos)
        row['psfSigma'] = psf.getSigma()
        row['psfIxx'] = shape.getIxx()
        row['psfIyy'] = shape.getIyy()
        row['psfIxy'] = shape.getIxy()
        row['psfArea'] = shape.getArea()

    return visit_summary


class MockVisitSummaryReference(lsst.daf.butler.DeferredDatasetHandle):
    """Very simple object that looks like a Gen3 data reference to
    a visit summary.

    Parameters
    ----------
    visit_summary : `lsst.afw.table.ExposureCatalog`
        Visit summary catalog.
    visit : `int`
        Visit number.
    """
    def __init__(self, visit_summary, visit):
        self.visit_summary = visit_summary
        self.visit = visit

    def get(self):
        """Retrieve the specified dataset using the API of the Gen3 Butler.

        Returns
        -------
        visit_summary : `lsst.afw.table.ExposureCatalog`
        """
        return self.visit_summary


class MockCoaddReference(lsst.daf.butler.DeferredDatasetHandle):
    """Very simple object that looks like a Gen3 data reference to
    a coadd.

    Parameters
    ----------
    exposure : `lsst.afw.image.Exposure`
        The exposure to be retrieved by the data reference.
    coaddName : `str`
        The type of coadd produced.  Typically "deep".
    patch : `int`
        Unique identifier for a subdivision of a tract.
    tract : `int`
        Unique identifier for a tract of a skyMap
    """
    def __init__(self, exposure, coaddName="deep", patch=0, tract=0):
        self.coaddName = coaddName
        self.exposure = exposure
        self.tract = tract
        self.patch = patch

    def get(self, component=None):
        """Retrieve the specified dataset using the API of the Gen 3 Butler.

        Parameters
        ----------
        component : `str`, optional
            If supplied, return the named metadata of the exposure.  Allowed
            components are "photoCalib" or "coaddInputs".

        Returns
        -------
        `lsst.afw.image.Exposure` ('component=None') or
        `lsst.afw.image.PhotoCalib` ('component="photoCalib") or
        `lsst.afw.image.CoaddInputs` ('component="coaddInputs")
        """
        if component == "photoCalib":
            return self.exposure.getPhotoCalib()
        elif component == "coaddInputs":
            return self.exposure.getInfo().getCoaddInputs()

        return self.exposure.clone()


class MockInputMapReference(lsst.daf.butler.DeferredDatasetHandle):
    """Very simple object that looks like a Gen3 data reference to
    an input map.

    Parameters
    ----------
    input_map : `healsparse.HealSparseMap`
        Bitwise input map.
    patch : `int`
        Patch number.
    tract : `int`
        Tract number.
    """
    def __init__(self, input_map, patch=0, tract=0):
        self.input_map = input_map
        self.tract = tract
        self.patch = patch

    def get(self):
        """
        Retrieve the specified dataset using the API of the Gen 3 Butler.

        Returns
        -------
        input_map : `healsparse.HealSparseMap`
        """
        return self.input_map
