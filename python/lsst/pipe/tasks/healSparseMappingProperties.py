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
# import lsst.pipe.base as pipeBase


__all__ = ["BasePropertyMapConfig", "PropertyMapRegistry", "register",
           "PropertyMapMap", "BasePropertyMap", "ExposureTimePropertyMap",
           "PsfSizePropertyMap", "PsfE1PropertyMap", "PsfE2PropertyMap",
           "NExposurePropertyMap"]


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
    """
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


def register(name):
    """A decorator to register a property map class in its base class's registry."""
    def decorate(PropertyMapClass):
        PropertyMapClass.registry.register(name, PropertyMapClass)
        return PropertyMapClass
    return decorate


class PropertyMapMap(dict):
    """Map of property maps to be run for a given task.

    Notes
    -----
    Property maps are classes derived from `BasePropertyMap`
    """
    def __iter__(self):
        for property_map in self.values():
            if property_map.config.do_min or property_map.config.do_max \
               or property_map.config.do_mean or property_map.config.do_weighted_mean \
               or property_map.config.do_sum:
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

    ConfigClass = BasePropertyMapConfig

    registry = PropertyMapRegistry(BasePropertyMapConfig)

    def __init__(self, config, name):
        object.__init__(self)
        self.config = config
        self.name = name

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

    def accumulate_values(self, indices, ra, dec, weights, row):
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
        row : `lsst.afw.table.ExposureRecord`
            Row of a visitSummary ExposureCatalog.
        """
        values = self.compute(row, ra, dec)
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

    def compute(self, row, ra, dec):
        """Compute map value from a row in the visitSummary catalog.

        Parameters
        ---------
        row : `lsst.afw.table.ExposureRecord`
            Row of a visitSummary ExposureCatalog.
        ra : `np.ndarray`
            Array of right ascensions
        dec : `np.ndarray`
            Array of declinations
        """
        raise NotImplementedError("All property maps must implement compute()")


@register("exposure_time")
class ExposureTimePropertyMap(BasePropertyMap):
    def compute(self, row, ra, dec):
        return row.getVisitInfo().getExposureTime()


@register("psf_size")
class PsfSizePropertyMap(BasePropertyMap):
    def compute(self, row, ra, dec):
        return row['psfSigma']


@register("psf_e1")
class PsfE1PropertyMap(BasePropertyMap):
    def compute(self, row, ra, dec):
        ixx = row['psfIxx']
        iyy = row['psfIyy']
        size = row['psfSigma']
        return (ixx - iyy)/(ixx + iyy + 2.*size**2.)


@register("psf_e2")
class PsfE2PropertyMap(BasePropertyMap):
    def compute(self, row, ra, dec):
        ixx = row['psfIxx']
        iyy = row['psfIyy']
        ixy = row['psfIxy']
        size = row['psfSigma']
        return (2.*ixy)/(ixx + iyy + 2.*size**2.)


@register("n_exposure")
class NExposurePropertyMap(BasePropertyMap):
    dtype = np.int32

    def compute(self, row, ra, dec):
        return 1


# FIXME: add maglim map
