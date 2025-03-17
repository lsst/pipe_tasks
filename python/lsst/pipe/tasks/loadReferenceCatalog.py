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

"""Load a full reference catalog in numpy/table/dataframe format.

This task will load multi-band reference objects, apply a reference selector,
and apply color terms.
"""

__all__ = ['LoadReferenceCatalogConfig', 'LoadReferenceCatalogTask']

import numpy as np
from astropy import units

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.meas.algorithms import ReferenceSourceSelectorTask
from lsst.meas.algorithms import getRefFluxField
from lsst.pipe.tasks.colorterms import ColortermLibrary
from lsst.afw.image import abMagErrFromFluxErr
from lsst.meas.algorithms import ReferenceObjectLoader, LoadReferenceObjectsConfig


class LoadReferenceCatalogConfig(pexConfig.Config):
    """Config for LoadReferenceCatalogTask"""
    refObjLoader = pexConfig.ConfigField(
        dtype=LoadReferenceObjectsConfig,
        doc="Configuration for the reference object loader.",
    )
    doApplyColorTerms = pexConfig.Field(
        doc=("Apply photometric color terms to reference stars? "
             "Requires that colorterms be set to a ColorTermLibrary"),
        dtype=bool,
        default=True
    )
    colorterms = pexConfig.ConfigField(
        doc="Library of photometric reference catalog name to color term dict.",
        dtype=ColortermLibrary,
    )
    referenceSelector = pexConfig.ConfigurableField(
        target=ReferenceSourceSelectorTask,
        doc="Selection of reference sources",
    )
    doReferenceSelection = pexConfig.Field(
        doc="Run the reference selector on the reference catalog?",
        dtype=bool,
        default=True
    )

    def validate(self):
        super().validate()
        if self.doApplyColorTerms and len(self.colorterms.data) == 0:
            msg = "applyColorTerms=True requires the `colorterms` field be set to a ColortermLibrary."
            raise pexConfig.FieldValidationError(LoadReferenceCatalogConfig.colorterms, self, msg)


class LoadReferenceCatalogTask(pipeBase.Task):
    """Load multi-band reference objects from a reference catalog.

    Parameters
    ----------
    dataIds : iterable of `lsst.daf.butler.dataId`
        An iterable object of dataIds which point to reference catalogs
        in a Gen3 repository.  Required for Gen3.
    refCats : iterable of `lsst.daf.butler.DeferredDatasetHandle`
        An iterable object of dataset refs for reference catalogs in
        a Gen3 repository.
    name : `str`
        The name of the refcat that this object will load. This name is used
        for applying colorterms, for example.

    Raises
    ------
    RuntimeError if dataIds or refCats is None.
    """
    ConfigClass = LoadReferenceCatalogConfig
    _DefaultName = "loadReferenceCatalog"

    def __init__(self, *, dataIds, refCats, name, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)
        refConfig = self.config.refObjLoader
        self.refObjLoader = ReferenceObjectLoader(dataIds=dataIds,
                                                  refCats=refCats,
                                                  name=name,
                                                  config=refConfig,
                                                  log=self.log)

        if self.config.doReferenceSelection:
            self.makeSubtask('referenceSelector')
        self._fluxFilters = None
        self._fluxFields = None
        self._referenceFilter = None

    def getPixelBoxCatalog(self, bbox, wcs, filterList, epoch=None,
                           bboxToSpherePadding=None):
        """Get a multi-band reference catalog by specifying a bounding box and WCS.

        The catalog will be in `numpy.ndarray`, with positions proper-motion
        corrected to "epoch" (if specified, and if the reference catalog has
        proper motions); sources cut on a reference selector (if
        "config.doReferenceSelection = True"); and color-terms applied (if
        "config.doApplyColorTerms = True").

        The format of the reference catalog will be of the format:

        dtype = [('ra', 'np.float64'),
                 ('dec', 'np.float64'),
                 ('refMag', 'np.float32', (len(filterList), )),
                 ('refMagErr', 'np.float32', (len(filterList), ))]

        Reference magnitudes (AB) and errors will be NaN for non-detections
        in a given band.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Box which bounds a region in pixel space.
        wcs : `lsst.afw.geom.SkyWcs`
            Wcs object defining the pixel to sky (and reverse) transform for
            the supplied bbox.
        filterList : `List` [ `str` ]
            List of camera physicalFilter names to retrieve magnitudes.
        epoch : `astropy.time.Time`, optional
            Epoch to which to correct proper motion and parallax
            (if available), or `None` to not apply such corrections.
        bboxToSpherePadding : `int`, optional
            Padding to account for translating a set of corners into a
            spherical (convex) boundary that is certain to encompass the
            entire area covered by the bbox.

        Returns
        -------
        refCat : `numpy.ndarray`
            Reference catalog.
        """
        # Check if we have previously cached values for the fluxFields
        if self._fluxFilters is None or self._fluxFilters != filterList:
            center = wcs.pixelToSky(bbox.getCenter())
            self._determineFluxFields(center, filterList)

        skyBox = self.refObjLoader.loadPixelBox(bbox, wcs, self._referenceFilter,
                                                epoch=epoch,
                                                bboxToSpherePadding=bboxToSpherePadding)

        if not skyBox.refCat.isContiguous():
            refCat = skyBox.refCat.copy(deep=True)
        else:
            refCat = skyBox.refCat

        return self._formatCatalog(refCat, filterList)

    def getSkyCircleCatalog(self, center, radius, filterList, epoch=None,
                            catalogFormat='numpy'):
        """Get a multi-band reference catalog by specifying a center and radius.

        The catalog will be in `numpy.ndarray`, with positions proper-motion
        corrected to "epoch" (if specified, and if the reference catalog has
        proper motions); sources cut on a reference selector (if
        "config.doReferenceSelection = True"); and color-terms applied (if
        "config.doApplyColorTerms = True").

        The format of the reference catalog will be of the format:

        dtype = [('ra', 'np.float64'),
                 ('dec', 'np.float64'),
                 ('refMag', 'np.float32', (len(filterList), )),
                 ('refMagErr', 'np.float32', (len(filterList), ))]

        Reference magnitudes (AB) and errors will be NaN for non-detections
        in a given band.

        Parameters
        ----------
        center : `lsst.geom.SpherePoint`
            Point defining the center of the circular region.
        radius : `lsst.geom.Angle`
            Defines the angular radius of the circular region.
        filterList : `List` [ `str` ]
            List of camera physicalFilter names to retrieve magnitudes.
        epoch : `astropy.time.Time`, optional
            Epoch to which to correct proper motion and parallax
            (if available), or `None` to not apply such corrections.

        Returns
        -------
        refCat : `numpy.ndarray`
            Reference catalog.
        """
        # Check if we have previously cached values for the fluxFields
        if self._fluxFilters is None or self._fluxFilters != filterList:
            self._determineFluxFields(center, filterList)

        skyCircle = self.refObjLoader.loadSkyCircle(center, radius,
                                                    self._referenceFilter,
                                                    epoch=epoch)

        if not skyCircle.refCat.isContiguous():
            refCat = skyCircle.refCat.copy(deep=True)
        else:
            refCat = skyCircle.refCat

        return self._formatCatalog(refCat, filterList)

    def _formatCatalog(self, refCat, filterList):
        """Format a reference afw table into the final format.

        This method applies reference selections and color terms as specified
        by the config.

        Parameters
        ----------
        refCat : `lsst.afw.table.SourceCatalog`
            Reference catalog in afw format.
        filterList : `list` [`str`]
            List of camera physicalFilter names to apply color terms.

        Returns
        -------
        refCat : `numpy.ndarray`
            Reference catalog.
        """
        if self.config.doReferenceSelection:
            goodSources = self.referenceSelector.selectSources(refCat)
            selected = goodSources.selected
        else:
            selected = np.ones(len(refCat), dtype=bool)

        npRefCat = np.zeros(np.sum(selected), dtype=[('ra', 'f8'),
                                                     ('dec', 'f8'),
                                                     ('refMag', 'f4', (len(filterList), )),
                                                     ('refMagErr', 'f4', (len(filterList), ))])

        if npRefCat.size == 0:
            # Return an empty catalog if we don't have any selected sources.
            return npRefCat

        # Natively "coord_ra" and "coord_dec" are stored in radians.
        # Doing this as an array rather than by row with the coord access is
        # approximately 600x faster.
        npRefCat['ra'] = np.rad2deg(refCat['coord_ra'][selected])
        npRefCat['dec'] = np.rad2deg(refCat['coord_dec'][selected])

        # Default (unset) values are np.nan
        npRefCat['refMag'][:, :] = np.nan
        npRefCat['refMagErr'][:, :] = np.nan

        if self.config.doApplyColorTerms:
            refCatName = self.refObjLoader.name

            for i, (filterName, fluxField) in enumerate(zip(self._fluxFilters, self._fluxFields)):
                if fluxField is None:
                    # There is no matching reference band.
                    # This will leave the column filled with np.nans
                    continue
                self.log.debug("Applying color terms for filterName='%s'", filterName)

                colorterm = self.config.colorterms.getColorterm(filterName, refCatName, doRaise=True)

                refMag, refMagErr = colorterm.getCorrectedMagnitudes(refCat)

                # nan_to_num below replaces nans with 99, and this ensures
                # that we select magnitudes that both filter out nans and are
                # not very large (corresponding to very small fluxes), as "99"
                # is a common sentinel for illegal magnitudes in reference catalogs.
                good, = np.where((np.nan_to_num(refMag[selected], nan=99.0) < 90.0)
                                 & (np.nan_to_num(refMagErr[selected], nan=99.0) < 90.0)
                                 & (np.nan_to_num(refMagErr[selected]) > 0.0))

                npRefCat['refMag'][good, i] = refMag[selected][good]
                npRefCat['refMagErr'][good, i] = refMagErr[selected][good]
        else:
            # No color terms to apply
            for i, (filterName, fluxField) in enumerate(zip(self._fluxFilters, self._fluxFields)):
                # nan_to_num below replaces nans with zeros, and this ensures that
                # we select fluxes that both filter out nans and are positive.
                good, = np.where((np.nan_to_num(refCat[fluxField][selected]) > 0.0)
                                 & (np.nan_to_num(refCat[fluxField+'Err'][selected]) > 0.0))
                refMag = (refCat[fluxField][selected][good]*units.nJy).to_value(units.ABmag)
                refMagErr = abMagErrFromFluxErr(refCat[fluxField+'Err'][selected][good],
                                                refCat[fluxField][selected][good])
                npRefCat['refMag'][good, i] = refMag
                npRefCat['refMagErr'][good, i] = refMagErr

        return npRefCat

    def _determineFluxFields(self, center, filterList):
        """Determine the flux field names for a reference catalog.

        This method sets self._fluxFields, self._referenceFilter.

        Parameters
        ----------
        center : `lsst.geom.SpherePoint`
            The center around which to load test sources.
        filterList : `list` [`str`]
            List of camera physicalFilter names.
        """
        # Search for a good filter to use to load the reference catalog
        # via the refObjLoader task which requires a valid filterName
        foundReferenceFilter = False

        # Store the original config
        _config = self.refObjLoader.config

        configTemp = LoadReferenceObjectsConfig()
        configIntersection = {k: getattr(self.refObjLoader.config, k)
                              for k, v in self.refObjLoader.config.toDict().items()
                              if (k in configTemp.keys() and k != "connections")}
        # We must turn off the proper motion checking to find the refFilter.
        configIntersection['requireProperMotion'] = False
        configTemp.update(**configIntersection)

        self.refObjLoader.config = configTemp

        for filterName in filterList:
            if self.config.refObjLoader.anyFilterMapsToThis is not None:
                refFilterName = self.config.refObjLoader.anyFilterMapsToThis
            else:
                refFilterName = self.config.refObjLoader.filterMap.get(filterName)
            if refFilterName is None:
                continue
            try:
                results = self.refObjLoader.loadSchema(refFilterName)
                foundReferenceFilter = True
                self._referenceFilter = refFilterName
                break
            except RuntimeError as err:
                # This just means that the filterName wasn't listed
                # in the reference catalog.  This is okay.
                if 'not find flux' in err.args[0]:
                    # The filterName wasn't listed in the reference catalog.
                    # This is not a fatal failure (yet)
                    pass
                else:
                    raise err

        self.refObjLoader.config = _config

        if not foundReferenceFilter:
            raise RuntimeError("Could not find any valid flux field(s) %s" %
                               (", ".join(filterList)))

        # Record self._fluxFilters for checks on subsequent calls
        self._fluxFilters = filterList

        # Retrieve all the fluxField names
        self._fluxFields = []
        for filterName in filterList:
            fluxField = None

            if self.config.refObjLoader.anyFilterMapsToThis is not None:
                refFilterName = self.config.refObjLoader.anyFilterMapsToThis
            else:
                refFilterName = self.config.refObjLoader.filterMap.get(filterName)

            if refFilterName is not None:
                try:
                    fluxField = getRefFluxField(results.schema, filterName=refFilterName)
                except RuntimeError:
                    # This flux field isn't available.  Set to None
                    fluxField = None

            if fluxField is None:
                self.log.warning('No reference flux field for camera filter %s', filterName)

            self._fluxFields.append(fluxField)
