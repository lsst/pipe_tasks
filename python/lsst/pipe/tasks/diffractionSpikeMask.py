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


__all__ = ["DiffractionSpikeMaskConfig", "DiffractionSpikeMaskTask"]


import math
import numpy as np
import astropy.units as u

import lsst.afw.geom as afwGeom
from lsst.afw.image import abMagErrFromFluxErr
import lsst.geom
from lsst.pex.config import Config, ConfigField, Field
from lsst.pipe.base import Struct, Task
from lsst.meas.algorithms import getRefFluxField, LoadReferenceObjectsConfig
import lsst.sphgeom
from .colorterms import ColortermLibrary


class DiffractionSpikeMaskConfig(Config):
    """Config for BrightStarMaskTask.
    """
    refObjLoader = ConfigField(dtype=LoadReferenceObjectsConfig,
                               doc="Configuration of reference object loader")
    applyColorTerms = Field(
        dtype=bool,
        default=False,
        doc=("Apply photometric color terms to reference stars?\n"
             "`True`: attempt to apply color terms; fail if color term data is "
             "not available for the specified reference catalog and filter.\n"
             "`False`: do not apply color terms."),
        optional=True,
    )
    colorterms = ConfigField(
        dtype=ColortermLibrary,
        doc="Library of photometric reference catalog name: color term dict"
            " (see also applyColorTerms).",
    )
    photoCatName = Field(
        dtype=str,
        optional=True,
        doc=("Name of photometric reference catalog; used to select a color"
             " term dict in colorterms. See also applyColorTerms."),
    )
    raKey = Field(
        dtype=str,
        default="coord_ra",
        doc="RA column name in the reference catalog.",
    )
    decKey = Field(
        dtype=str,
        default="coord_dec",
        doc="Declination column name in the reference catalog.",
    )
    angleMargin = Field(
        dtype=float,
        default=60.,
        doc="Margin outside the exposure bounding box to include bright "
            "sources. In arcseconds.",
    )
    magnitudeThreshold = Field(
        dtype=float,
        default=15,
        doc="Threshold magnitude for treating a star from the reference catalog"
            " as bright.",
    )
    diffractAngle = Field(
        dtype=float,
        default=45,
        doc="Angle in degrees of the location of diffraction spikes with "
            "respect to camera at 0 rotation angle.",
    )
    spikeAspectRatio = Field(
        dtype=float,
        default=10,
        doc="Ratio of the length of a diffraction spike to it's width in the"
            " core of the star.",
    )
    magSlope = Field(
        dtype=float,
        default=-0.12,
        doc="Slope of the fit for the log(spike length) as a function of"
            " magnitude.",
    )
    magOffset = Field(
        dtype=float,
        default=3.8,
        doc="Intercept of the fit for the log(spike length) as a function of"
            " magnitude.",
    )
    fallbackMagnitude = Field(
        dtype=float,
        default=12.,
        doc="Default magnitude to use for sources in the reference catalog"
            " with missing magnitudes that land in regions of saturated pixels.",
    )
    anyFilterMapsToThis = Field(
        dtype=str,
        default="phot_g_mean",
        doc="Fallback flux field in the reference catalog to use for sources"
        " that don't have measurements in the science image's band.",
    )
    spikeMask = Field(
        dtype=str,
        default="SPIKE",
        doc="Name of the mask plane indicating likely contamination from"
            " a diffraction spike.",
    )


class DiffractionSpikeMaskTask(Task):
    """Load a reference catalog to identify bright stars that are likely to be
    saturated and have visible diffraction spikes that need to be masked.

    Attributes
    ----------
    angles : `numpy.ndarray`
        Expected angles of diffraction spikes for bright sources.
    refObjLoader : `lsst.meas.algorithms.ReferenceObjectLoader`
        An instance of a reference object loader.
    """
    ConfigClass = DiffractionSpikeMaskConfig
    _DefaultName = "diffractionSpikeMask"

    def __init__(self, refObjLoader=None, **kwargs):
        Task.__init__(self, **kwargs)
        self.refObjLoader = refObjLoader

    def setRefObjLoader(self, refObjLoader):
        """Set the reference object loader for the task.

        Parameters
        ----------
        refObjLoader : `lsst.meas.algorithms.ReferenceObjectLoader`
            An instance of a reference object loader.
        """
        self.refObjLoader = refObjLoader

    def run(self, exposure):
        """Load reference objects and mask bright stars on an exposure.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Science exposure to set the SPIKE plane. Will be modified in place.

        Returns
        -------
        spikeCat : `lsst.afw.table.SourceCatalog`
            The entries from the reference catalog selected as stars with
            diffraction spikes.
        """
        if self.refObjLoader is None:
            raise RuntimeError("Running diffraction spike mask task with no refObjLoader set in"
                               " __init__ or setRefObjLoader")
        # First set the angles of the diffraction spikes from the exposure
        # metadata.
        self.set_diffraction_angle(exposure)
        filterLabel = exposure.getFilter()
        region = getRegion(exposure, lsst.sphgeom.Angle.fromDegrees(self.config.angleMargin/3600.))
        refCat = self.refObjLoader.loadRegion(region, filterLabel.bandLabel).refCat

        # Mask any sources with known or estimated magnitudes for the current
        # filter, including sources off of the image which may have diffraction
        # spikes that extend onto the image.
        magnitudes = self.extractMagnitudes(refCat, filterLabel).refMag
        radii = self.calculateReferenceRadius(magnitudes)
        bright = (magnitudes < self.config.magnitudeThreshold) & (radii > 1)

        nBright = np.count_nonzero(bright)

        mask = exposure.mask
        mask.addMaskPlane(self.config.spikeMask)
        if nBright > 0:
            xvals, yvals = exposure.wcs.skyToPixelArray(refCat[bright][self.config.raKey],
                                                        refCat[bright][self.config.decKey])
            self.log.info("Calculating mask for %d stars brighter than magnitude %f", nBright,
                          self.config.magnitudeThreshold)
            self.maskSources(xvals, yvals, radii[bright], mask)
        else:
            self.log.info("No bright stars found in the reference catalog; not masking diffraction spikes.")

        return refCat[bright].copy(deep=True)

    def maskSources(self, xvals, yvals, radii, mask):
        """Apply the SPIKE mask for a given set of coordinates. The mask plane
        will be modified in place.

        Parameters
        ----------
        xvals, yvals : `numpy.ndarray`
            Array of x- and y-values of bright sources to mask.
        radii : `numpy.ndarray`
            Array of radius values for each bright source.
        mask : `lsst.afw.image.Mask`
            The mask plane of the image to set the BRIGHT mask plane.
        """
        bbox = mask.getBBox()
        for x, y, r in zip(xvals, yvals, radii):
            maskSingle = self.makeSingleMask(x, y, r)
            singleBBox = maskSingle.getBBox()
            if bbox.overlaps(singleBBox):
                singleBBox = singleBBox.clippedTo(bbox)
                mask[singleBBox] |= maskSingle[singleBBox]

    def makeSingleMask(self, x, y, r):
        """Create a mask plane centered on a single source with the BRIGHT mask
        set. This mask does not have to be fully contained in the bounding box
        of the mask of the science image.

        Parameters
        ----------
        x, y : `float`
            Coordinates of the source to be masked.
        r : `float`
            Expected length of a diffraction spike for a source with a
            magnitude

        Returns
        -------
        mask : `lsst.afw.image.Mask`
            A mask plane centered on the single source being modeled.
        """
        polygons = []
        for angle in self.angles:
            # Model the diffraction spike as an isosceles triangle with a long
            # side along the spike and a short base. For efficiency, model the
            # diffraction spike in the opposite direction at the same time,
            # so the overall shape is a narrow diamond with equal length sides.
            xLong = math.cos(np.deg2rad(angle))*r
            yLong = math.sin(np.deg2rad(angle))*r
            xShort = -math.sin(np.deg2rad(angle))*r/self.config.spikeAspectRatio
            yShort = math.cos(np.deg2rad(angle))*r/self.config.spikeAspectRatio

            corners = [lsst.geom.Point2D(x + xLong, y + yLong),
                       lsst.geom.Point2D(x + xShort, y + yShort),
                       lsst.geom.Point2D(x - xLong, y - yLong),
                       lsst.geom.Point2D(x - xShort, y - yShort)]
            polygons.append(afwGeom.Polygon(corners))
        # Combine all of the polygons into a single region
        polygon = polygons[0]
        for poly in polygons[1:]:
            polygon = polygon.unionSingle(poly)
        bbox = lsst.geom.Box2I(polygon.getBBox())
        polyImage = polygon.createImage(bbox).array
        mask = lsst.afw.image.Mask(bbox)
        mask.array[polyImage > 0] = mask.getPlaneBitMask(self.config.spikeMask)
        return mask

    def set_diffraction_angle(self, exposure):
        """Calculate the angle of diffration spikes on the image given the
        camera rotation.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            The exposure to calculate the expected angle of diffraction spikes.
        """
        angle = (exposure.visitInfo.boresightParAngle.asDegrees()
                 - exposure.visitInfo.boresightRotAngle.asDegrees()
                 + self.config.diffractAngle)
        self.angles = np.array([angle, angle + 90]) % 360

    def calculateReferenceRadius(self, magnitudes):
        """Calculate the size of the region to mask for each bright star.

        Parameters
        ----------
        magnitudes : `numpy.ndarray`
            Magnitudes of the bright stars.

        Returns
        -------
        radii : `numpy.ndarray`
            Array of radius values for the given magnitudes.
        """
        # In pixels
        radii = [10**(self.config.magSlope*magnitude + self.config.magOffset) for magnitude in magnitudes]
        return np.array(radii)

    def extractMagnitudes(self, refCat, filterLabel):
        """Extract magnitude and magnitude error arrays from the given catalog.

        Parameters
        ----------
        refCat : `lsst.afw.table.SourceCatalog`
            The input reference catalog.
        filterLabel : `str`
            Label of filter being calibrated.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``refMag``
                Reference magnitude (`np.array`).
            ``refMagErr``
                Reference magnitude error (`np.array`).
            ``refFluxFieldList``
                A list of field names of the reference catalog used for fluxes (1 or 2 strings) (`list`).
        """
        refSchema = refCat.schema

        if self.config.applyColorTerms:
            self.log.info("Applying color terms for filter=%r, config.photoCatName=%s",
                          filterLabel.physicalLabel, self.config.photoCatName)
            colorterm = self.config.colorterms.getColorterm(filterLabel.physicalLabel,
                                                            self.config.photoCatName,
                                                            doRaise=True)

            refMagArr, refMagErrArr = colorterm.getCorrectedMagnitudes(refCat)
            fluxFieldList = [getRefFluxField(refSchema, filt) for filt in (colorterm.primary,
                                                                           colorterm.secondary)]
        else:
            self.log.info("Not applying color terms.")
            colorterm = None

            fluxFieldList = [getRefFluxField(refSchema, filterLabel.bandLabel)]
            fluxField = getRefFluxField(refSchema, filterLabel.bandLabel)
            fluxKey = refSchema.find(fluxField).key
            refFluxArr = np.array(refCat.get(fluxKey))

            try:
                fluxErrKey = refSchema.find(fluxField + "Err").key
                refFluxErrArr = np.array(refCat.get(fluxErrKey))
            except KeyError:
                # Reference catalogue may not have flux uncertainties; HACK DM-2308
                self.log.warning("Reference catalog does not have flux uncertainties for %s;"
                                 " using sqrt(flux).", fluxField)
                refFluxErrArr = np.sqrt(refFluxArr)

            refMagArr = u.Quantity(refFluxArr, u.nJy).to_value(u.ABmag)
            # HACK convert to Jy until we have a replacement for this (DM-16903)
            refMagErrArr = abMagErrFromFluxErr(refFluxErrArr*1e-9, refFluxArr*1e-9)
        # Not all sources in the reference catalog will have flux measurements
        # in the current band. Since we are only masking sources use a fallback
        # flux measurement in these cases to get an approximate magnitude to
        # check for bright sources and to set the size of the mask.
        badMagnitudes = np.isnan(refMagArr)
        if np.count_nonzero(badMagnitudes):
            fallbackFluxKey = refSchema.find(f"{self.config.anyFilterMapsToThis}_flux").key
            fallbackFluxArr = np.array(refCat.get(fallbackFluxKey))
            fallbackMagArr = u.Quantity(fallbackFluxArr[badMagnitudes], u.nJy).to_value(u.ABmag)
            refMagArr[badMagnitudes] = fallbackMagArr

        return Struct(
            refMag=refMagArr,
            refMagErr=refMagErrArr,
            refFluxFieldList=fluxFieldList,
        )


def getRegion(exposure, margin=None):
    """Calculate an enveloping region for an exposure.

    Parameters
    ----------
    exposure : `lsst.afw.image.Exposure`
        Exposure object with calibrated WCS.

    Returns
    -------
    region : `lsst.sphgeom.Region`
        Region enveloping an exposure.
    """
    # Bounding box needs to be a `Box2D` not a `Box2I` for `wcs.pixelToSky()`
    bbox = lsst.geom.Box2D(exposure.getBBox())
    wcs = exposure.getWcs()

    region = lsst.sphgeom.ConvexPolygon([pp.getVector() for pp in wcs.pixelToSky(bbox.getCorners())])
    if margin is not None:
        # This is an ad-hoc, approximate implementation. It should be good
        # enough for catalog loading, but is not a general-purpose solution.
        center = lsst.geom.SpherePoint(region.getCentroid())
        corners = [lsst.geom.SpherePoint(c) for c in region.getVertices()]
        # Approximate the region as a Euclidian square
        # geom.Angle(sphgeom.Angle) converter not pybind-wrapped???
        diagonal_margin = lsst.geom.Angle(margin.asRadians() * math.sqrt(2.0))
        padded = [c.offset(center.bearingTo(c), diagonal_margin) for c in corners]
        return lsst.sphgeom.ConvexPolygon.convexHull([c.getVector() for c in padded])

    return region


def computePsfWidthFromMoments(psf, angle=0.):
    """Calculate the width of an elliptical PSF along a given direction.

    Parameters
    ----------
    psf : `lsst.afw.detection.Psf`
        The point spread function of the image.
    angle : `float`, optional
        Rotation CCW from the +x axis to calculate the width of the PSF along.

    Returns
    -------
    fwhm : `float`
        Full width at half maximum of the fitted shape of the PSF along the
        given `angle`. In pixels.
    """
    psfShape = psf.computeShape(psf.getAveragePosition())
    c = np.cos(np.deg2rad(angle))
    s = np.sin(np.deg2rad(angle))
    sigma2 = c*c*psfShape.getIxx() + 2*c*s*psfShape.getIxy() + s*s*psfShape.getIyy()
    sigma = np.sqrt(max(sigma2, 0.0))   # rms width in pixels

    # 5) optional: Gaussian-equivalent FWHM along angle
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    return fwhm
