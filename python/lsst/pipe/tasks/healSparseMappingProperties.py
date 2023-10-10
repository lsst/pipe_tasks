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
import numpy as np
import healsparse as hsp

import lsst.pex.config as pexConfig
import lsst.geom
from lsst.afw.math import ChebyshevBoundedField, ChebyshevBoundedFieldControl


__all__ = ["BasePropertyMapConfig", "PropertyMapRegistry", "register_property_map",
           "PropertyMapMap", "BasePropertyMap", "ExposureTimePropertyMap",
           "PsfSizePropertyMap", "PsfE1PropertyMap", "PsfE2PropertyMap",
           "NExposurePropertyMap", "PsfMaglimPropertyMapConfig",
           "PsfMaglimPropertyMap", "SkyBackgroundPropertyMap", "SkyNoisePropertyMap",
           "DcrDraPropertyMap", "DcrDdecPropertyMap", "DcrE1PropertyMap",
           "DcrE2PropertyMap", "compute_approx_psf_size_and_shape"]


class BasePropertyMapConfig(pexConfig.Config):
    do_min = pexConfig.Field(dtype=bool, default=False,
                             doc="Compute map of property minima.")
    do_max = pexConfig.Field(dtype=bool, default=False,
                             doc="Compute map of property maxima.")
    do_mean = pexConfig.Field(dtype=bool, default=False,
                              doc="Compute map of property means.")
    do_weighted_mean = pexConfig.Field(dtype=bool, default=False,
                                       doc="Compute map of weighted property means.")
    do_sum = pexConfig.Field(dtype=bool, default=False,
                             doc="Compute map of property sums.")


class PropertyMapRegistry(pexConfig.Registry):
    """Class for property map registry.

    Notes
    -----
    This code is based on `lsst.meas.base.PluginRegistry`.
    """
    class Configurable:
        """Class used as the element in the property map registry.

        Parameters
        ----------
        name : `str`
            Name under which the property map is registered.
        PropertyMapClass : subclass of `BasePropertyMap`
        """
        def __init__(self, name, PropertyMapClass):
            self.name = name
            self.PropertyMapClass = PropertyMapClass

        @property
        def ConfigClass(self):
            return self.PropertyMapClass.ConfigClass

        def __call__(self, config):
            return (self.name, config, self.PropertyMapClass)

    def register(self, name, PropertyMapClass):
        """Register a property map class with the given name.

        Parameters
        ----------
        name : `str`
            The name of the property map.
        PropertyMapClass : subclass of `BasePropertyMap`
        """
        pexConfig.Registry.register(self, name, self.Configurable(name, PropertyMapClass))


def register_property_map(name):
    """A decorator to register a property map class in its base class's registry."""
    def decorate(PropertyMapClass):
        PropertyMapClass.registry.register(name, PropertyMapClass)
        return PropertyMapClass
    return decorate


def compute_approx_psf_size_and_shape(ccd_row, ra, dec, nx=20, ny=20, orderx=2, ordery=2):
    """Compute the approximate psf size and shape.

    This routine fits how the psf size and shape varies over a field by approximating
    with a Chebyshev bounded field.

    Parameters
    ----------
    ccd_row : `lsst.afw.table.ExposureRecord`
        Exposure metadata for a given detector exposure.
    ra : `np.ndarray`
        Right ascension of points to compute size and shape (degrees).
    dec : `np.ndarray`
        Declination of points to compute size and shape (degrees).
    nx : `int`, optional
        Number of sampling points in the x direction.
    ny : `int`, optional
        Number of sampling points in the y direction.
    orderx : `int`, optional
        Chebyshev polynomial order for fit in x direction.
    ordery : `int`, optional
        Chebyshev polynomial order for fit in y direction.

    Returns
    -------
    psf_array : `np.ndarray`
        Record array with "psf_size", "psf_e1", "psf_e2".
    """
    pts = [lsst.geom.SpherePoint(r*lsst.geom.degrees, d*lsst.geom.degrees) for
           r, d in zip(ra, dec)]
    pixels = ccd_row.getWcs().skyToPixel(pts)

    ctrl = ChebyshevBoundedFieldControl()
    ctrl.orderX = orderx
    ctrl.orderY = ordery
    ctrl.triangular = False

    bbox = ccd_row.getBBox()
    xSteps = np.linspace(bbox.getMinX(), bbox.getMaxX(), nx)
    ySteps = np.linspace(bbox.getMinY(), bbox.getMaxY(), ny)
    x = np.tile(xSteps, nx)
    y = np.repeat(ySteps, ny)

    psf_size = np.zeros(x.size)
    psf_e1 = np.zeros(x.size)
    psf_e2 = np.zeros(x.size)
    psf_area = np.zeros(x.size)

    psf = ccd_row.getPsf()
    for i in range(x.size):
        shape = psf.computeShape(lsst.geom.Point2D(x[i], y[i]))
        psf_size[i] = shape.getDeterminantRadius()
        ixx = shape.getIxx()
        iyy = shape.getIyy()
        ixy = shape.getIxy()

        psf_e1[i] = (ixx - iyy)/(ixx + iyy + 2.*psf_size[i]**2.)
        psf_e2[i] = (2.*ixy)/(ixx + iyy + 2.*psf_size[i]**2.)

        im = psf.computeKernelImage(lsst.geom.Point2D(x[i], y[i]))
        psf_area[i] = np.sum(im.array)/np.sum(im.array**2.)

    pixel_x = np.array([pix.getX() for pix in pixels])
    pixel_y = np.array([pix.getY() for pix in pixels])

    psf_array = np.zeros(pixel_x.size, dtype=[("psf_size", "f8"),
                                              ("psf_e1", "f8"),
                                              ("psf_e2", "f8"),
                                              ("psf_area", "f8")])

    # Protect against nans which can come in at the edges and masked regions.
    good = np.isfinite(psf_size)
    x = x[good]
    y = y[good]

    cheb_size = ChebyshevBoundedField.fit(lsst.geom.Box2I(bbox), x, y, psf_size[good], ctrl)
    psf_array["psf_size"] = cheb_size.evaluate(pixel_x, pixel_y)
    cheb_e1 = ChebyshevBoundedField.fit(lsst.geom.Box2I(bbox), x, y, psf_e1[good], ctrl)
    psf_array["psf_e1"] = cheb_e1.evaluate(pixel_x, pixel_y)
    cheb_e2 = ChebyshevBoundedField.fit(lsst.geom.Box2I(bbox), x, y, psf_e2[good], ctrl)
    psf_array["psf_e2"] = cheb_e2.evaluate(pixel_x, pixel_y)
    cheb_area = ChebyshevBoundedField.fit(lsst.geom.Box2I(bbox), x, y, psf_area[good], ctrl)
    psf_array["psf_area"] = cheb_area.evaluate(pixel_x, pixel_y)

    return psf_array


class PropertyMapMap(dict):
    """Map of property maps to be run for a given task.

    Notes
    -----
    Property maps are classes derived from `BasePropertyMap`
    """
    def __iter__(self):
        for property_map in self.values():
            if (property_map.config.do_min or property_map.config.do_max or property_map.config.do_mean
                    or property_map.config.do_weighted_mean or property_map.config.do_sum):
                yield property_map


class BasePropertyMap:
    """Base class for property maps.

    Parameters
    ----------
    config : `BasePropertyMapConfig`
        Property map configuration.
    name : `str`
        Property map name.
    """
    dtype = np.float64
    requires_psf = False

    ConfigClass = BasePropertyMapConfig

    registry = PropertyMapRegistry(BasePropertyMapConfig)

    def __init__(self, config, name):
        object.__init__(self)
        self.config = config
        self.name = name
        self.zeropoint = 0.0

    def initialize_tract_maps(self, nside_coverage, nside):
        """Initialize the tract maps.

        Parameters
        ----------
        nside_coverage : `int`
            Healpix nside of the healsparse coverage map.
        nside : `int`
            Healpix nside of the property map.
        """
        if self.config.do_min:
            self.min_map = hsp.HealSparseMap.make_empty(nside_coverage,
                                                        nside,
                                                        self.dtype)
        if self.config.do_max:
            self.max_map = hsp.HealSparseMap.make_empty(nside_coverage,
                                                        nside,
                                                        self.dtype)
        if self.config.do_mean:
            self.mean_map = hsp.HealSparseMap.make_empty(nside_coverage,
                                                         nside,
                                                         self.dtype)
        if self.config.do_weighted_mean:
            self.weighted_mean_map = hsp.HealSparseMap.make_empty(nside_coverage,
                                                                  nside,
                                                                  self.dtype)
        if self.config.do_sum:
            self.sum_map = hsp.HealSparseMap.make_empty(nside_coverage,
                                                        nside,
                                                        self.dtype)

    def initialize_values(self, n_pixels):
        """Initialize the value arrays for accumulation.

        Parameters
        ----------
        n_pixels : `int`
            Number of pixels in the map.
        """
        if self.config.do_min:
            self.min_values = np.zeros(n_pixels, dtype=self.dtype)
            # This works for float types, need check for integers...
            self.min_values[:] = np.nan
        if self.config.do_max:
            self.max_values = np.zeros(n_pixels, dtype=self.dtype)
            self.max_values[:] = np.nan
        if self.config.do_mean:
            self.mean_values = np.zeros(n_pixels, dtype=self.dtype)
        if self.config.do_weighted_mean:
            self.weighted_mean_values = np.zeros(n_pixels, dtype=self.dtype)
        if self.config.do_sum:
            self.sum_values = np.zeros(n_pixels, dtype=self.dtype)

    def accumulate_values(self, indices, ra, dec, weights, scalings, row,
                          psf_array=None):
        """Accumulate values from a row of a visitSummary table.

        Parameters
        ----------
        indices : `np.ndarray`
            Indices of values that should be accumulated.
        ra : `np.ndarray`
            Array of right ascension for indices
        dec : `np.ndarray`
            Array of declination for indices
        weights : `float` or `np.ndarray`
            Weight(s) for indices to be accumulated.
        scalings : `float` or `np.ndarray`
            Scaling values to coadd zeropoint.
        row : `lsst.afw.table.ExposureRecord`
            Row of a visitSummary ExposureCatalog.
        psf_array : `np.ndarray`, optional
            Array of approximate psf values matched to ra/dec.

        Raises
        ------
        ValueError : Raised if requires_psf is True and psf_array is None.
        """
        if self.requires_psf and psf_array is None:
            name = self.__class__.__name__
            raise ValueError(f"Cannot compute {name} without psf_array.")

        values = self._compute(row, ra, dec, scalings, psf_array=psf_array)
        if self.config.do_min:
            self.min_values[indices] = np.fmin(self.min_values[indices], values)
        if self.config.do_max:
            self.max_values[indices] = np.fmax(self.max_values[indices], values)
        if self.config.do_mean:
            self.mean_values[indices] += values
        if self.config.do_weighted_mean:
            self.weighted_mean_values[indices] += weights*values
        if self.config.do_sum:
            self.sum_values[indices] += values

    def finalize_mean_values(self, total_weights, total_inputs):
        """Finalize the accumulation of the mean and weighted mean.

        Parameters
        ----------
        total_weights : `np.ndarray`
            Total accumulated weights, for each value index.
        total_inputs : `np.ndarray`
            Total number of inputs, for each value index.
        """
        if self.config.do_mean:
            use, = np.where(total_inputs > 0)
            self.mean_values[use] /= total_inputs[use]
        if self.config.do_weighted_mean:
            use, = np.where(total_weights > 0.0)
            self.weighted_mean_values[use] /= total_weights[use]

        # And perform any necessary post-processing
        self._post_process(total_weights, total_inputs)

    def set_map_values(self, pixels):
        """Assign accumulated values to the maps.

        Parameters
        ----------
        pixels : `np.ndarray`
            Array of healpix pixels (nest scheme) to set in the map.
        """
        if self.config.do_min:
            self.min_map[pixels] = self.min_values
        if self.config.do_max:
            self.max_map[pixels] = self.max_values
        if self.config.do_mean:
            self.mean_map[pixels] = self.mean_values
        if self.config.do_weighted_mean:
            self.weighted_mean_map[pixels] = self.weighted_mean_values
        if self.config.do_sum:
            self.sum_map[pixels] = self.sum_values

    def _compute(self, row, ra, dec, scalings, psf_array=None):
        """Compute map value from a row in the visitSummary catalog.

        Parameters
        ----------
        row : `lsst.afw.table.ExposureRecord`
            Row of a visitSummary ExposureCatalog.
        ra : `np.ndarray`
            Array of right ascensions
        dec : `np.ndarray`
            Array of declinations
        scalings : `float` or `np.ndarray`
            Scaling values to coadd zeropoint.
        psf_array : `np.ndarray`, optional
            Array of approximate psf values matched to ra/dec.
        """
        raise NotImplementedError("All property maps must implement _compute()")

    def _post_process(self, total_weights, total_inputs):
        """Perform post-processing on values.

        Parameters
        ----------
        total_weights : `np.ndarray`
            Total accumulated weights, for each value index.
        total_inputs : `np.ndarray`
            Total number of inputs, for each value index.
        """
        # Override of this method is not required.
        pass


@register_property_map("exposure_time")
class ExposureTimePropertyMap(BasePropertyMap):
    """Exposure time property map."""

    def _compute(self, row, ra, dec, scalings, psf_array=None):
        return row.getVisitInfo().getExposureTime()


@register_property_map("psf_size")
class PsfSizePropertyMap(BasePropertyMap):
    """PSF size property map."""
    requires_psf = True

    def _compute(self, row, ra, dec, scalings, psf_array=None):
        return psf_array["psf_size"]


@register_property_map("psf_e1")
class PsfE1PropertyMap(BasePropertyMap):
    """PSF shape e1 property map."""
    requires_psf = True

    def _compute(self, row, ra, dec, scalings, psf_array=None):
        return psf_array["psf_e1"]


@register_property_map("psf_e2")
class PsfE2PropertyMap(BasePropertyMap):
    """PSF shape e2 property map."""
    requires_psf = True

    def _compute(self, row, ra, dec, scalings, psf_array=None):
        return psf_array["psf_e2"]


@register_property_map("n_exposure")
class NExposurePropertyMap(BasePropertyMap):
    """Number of exposures property map."""
    dtype = np.int32

    def _compute(self, row, ra, dec, scalings, psf_array=None):
        return 1


class PsfMaglimPropertyMapConfig(BasePropertyMapConfig):
    """Configuration for the PsfMaglim property map."""
    maglim_nsigma = pexConfig.Field(dtype=float, default=5.0,
                                    doc="Number of sigma to compute magnitude limit.")

    def validate(self):
        super().validate()
        if self.do_min or self.do_max or self.do_mean or self.do_sum:
            raise ValueError("Can only use do_weighted_mean with PsfMaglimPropertyMap")


@register_property_map("psf_maglim")
class PsfMaglimPropertyMap(BasePropertyMap):
    """PSF magnitude limit property map."""
    requires_psf = True

    ConfigClass = PsfMaglimPropertyMapConfig

    def _compute(self, row, ra, dec, scalings, psf_array=None):
        # Our values are the weighted mean of the psf area
        return psf_array["psf_area"]

    def _post_process(self, total_weights, total_inputs):
        psf_area = self.weighted_mean_values.copy()
        maglim = (self.zeropoint
                  - 2.5*np.log10(self.config.maglim_nsigma*np.sqrt(psf_area/total_weights)))
        self.weighted_mean_values[:] = maglim


@register_property_map("sky_background")
class SkyBackgroundPropertyMap(BasePropertyMap):
    """Sky background property map."""
    def _compute(self, row, ra, dec, scalings, psf_array=None):
        return scalings*row["skyBg"]


@register_property_map("sky_noise")
class SkyNoisePropertyMap(BasePropertyMap):
    """Sky noise property map."""
    def _compute(self, row, ra, dec, scalings, psf_array=None):
        return scalings*row["skyNoise"]


@register_property_map("dcr_dra")
class DcrDraPropertyMap(BasePropertyMap):
    """Effect of DCR on delta-RA property map."""
    def _compute(self, row, ra, dec, scalings, psf_array=None):
        par_angle = row.getVisitInfo().getBoresightParAngle().asRadians()
        return np.tan(np.deg2rad(row["zenithDistance"]))*np.sin(par_angle)


@register_property_map("dcr_ddec")
class DcrDdecPropertyMap(BasePropertyMap):
    """Effect of DCR on delta-Dec property map."""
    def _compute(self, row, ra, dec, scalings, psf_array=None):
        par_angle = row.getVisitInfo().getBoresightParAngle().asRadians()
        return np.tan(np.deg2rad(row["zenithDistance"]))*np.cos(par_angle)


@register_property_map("dcr_e1")
class DcrE1PropertyMap(BasePropertyMap):
    """Effect of DCR on psf shape e1 property map."""
    def _compute(self, row, ra, dec, scalings, psf_array=None):
        par_angle = row.getVisitInfo().getBoresightParAngle().asRadians()
        return (np.tan(np.deg2rad(row["zenithDistance"]))**2.)*np.cos(2.*par_angle)


@register_property_map("dcr_e2")
class DcrE2PropertyMap(BasePropertyMap):
    """Effect of DCR on psf shape e2 property map."""
    def _compute(self, row, ra, dec, scalings, psf_array=None):
        par_angle = row.getVisitInfo().getBoresightParAngle().asRadians()
        return (np.tan(np.deg2rad(row["zenithDistance"]))**2.)*np.sin(2.*par_angle)
