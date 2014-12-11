#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008-2013 LSST Corporation.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

import lsst.afw.geom

from lsst.pex.config import Config, Field, FieldValidationError
from lsst.pipe.base import Task, TaskError

__all__ = ("BaseReferencesTask", "CoaddSrcReferencesTask")

class BaseReferencesConfig(Config):
    removePatchOverlaps = Field(
        doc = "Only include reference sources for each patch that lie within the patch's inner bbox",
        dtype = bool,
        default = True
    )
    filter = Field(
        doc = "Bandpass for reference sources; None indicates chi-squared detections.",
        dtype = str,
        optional = True
    )

class BaseReferencesTask(Task):
    """Base class for forced photometry subtask that retrieves reference sources.

    BaseReferencesTask defines the required API for the references task, which includes:
      - getSchema(butler)
      - fetchInPatches(butler, tract, filter, patchList)
      - fetchInBox(self, butler, tract, filter, bbox, wcs)
      - the removePatchOverlaps config option

    It also provides the subset() method, which may be of use to derived classes when
    reimplementing fetchInBox.
    """

    ConfigClass = BaseReferencesConfig

    def __init__(self, butler=None, schema=None, **kwargs):
        """Initialize the task.

        BaseReferencesTask and its subclasses take two keyword arguments beyond the usual Task arguments:
         - schema: the Schema of the reference catalog
         - butler: a butler that will allow the task to load its Schema from disk.
        At least one of these arguments must be present; if both are, schema takes precedence.
        """
        Task.__init__(self, **kwargs)

    def getSchema(self, butler):
        """Return the schema for the reference sources.

        Must be available even before any data has been processed.
        """
        raise NotImplementedError("BaseReferencesTask is pure abstract, and cannot be used directly.")

    def getWcs(self, dataRef):
        """Return the WCS for reference sources.  The given dataRef must include the tract in its dataId.
        """
        raise NotImplementedError("BaseReferencesTask is pure abstract, and cannot be used directly.")

    def fetchInBox(self, dataRef, bbox, wcs):
        """Return reference sources that overlap a region defined by a pixel-coordinate bounding box
        and corresponding Wcs.

        @param[in] dataRef    ButlerDataRef; the implied data ID must contain the 'tract' key.
        @param[in] bbox       a afw.geom.Box2I or Box2D that defines the region in pixel coordinates
        @param[in] wcs        afw.image.Wcs that maps the bbox to sky coordinates

        @return an iterable of reference sources

        It is not required that the returned object be a SourceCatalog; it may be any Python iterable
        containing SourceRecords (including a lazy iterator).

        The returned set of sources should be complete and close to minimal.
        """
        raise NotImplementedError("BaseReferencesTask is pure abstract, and cannot be used directly.")

    def fetchInPatches(self, dataRef, patchList):
        """Return reference sources that overlap a region defined by one or more SkyMap patches.

        @param[in] dataRef    ButlerDataRef; the implied data ID must contain the 'tract' key.
        @param[in] patchList  list of skymap.PatchInfo instances for which to fetch reference sources

        @return an iterable of reference sources

        It is not required that the returned object be a SourceCatalog; it may be any Python sequence
        containing SourceRecords (including a lazy iterator).

        The returned set of sources should be complete and close to minimal.  If
        config.removePatchOverlaps is True, only sources within each patch's "inner" bounding box
        should be returned.
        """
        raise NotImplementedError("BaseReferencesTask is pure abstract, and cannot be used directly.")

    def subset(self, sources, bbox, wcs):
        """Filter sources to contain only those within the given box, defined in the coordinate system
        defined by the given Wcs.

        @param[in] sources     input iterable of SourceRecords
        @param[in] bbox        bounding box with which to filter reference sources (Box2I or Box2D)
        @param[in] wcs         afw.image.Wcs that defines the coordinate system of bbox

        @return an iterable of filtered reference sources

        This is not a part of the required BaseReferencesTask interface; it's a convenience function
        used in implementing fetchInBox that may be of use to subclasses.
        """
        boxD = lsst.afw.geom.Box2D(bbox)
        for source in sources:
            pixel = wcs.skyToPixel(source.getCoord())
            if boxD.contains(pixel):
                yield source

class CoaddSrcReferencesConfig(BaseReferencesTask.ConfigClass):
    coaddName = Field(
        doc = "Coadd name: typically one of deep or goodSeeing.",
        dtype = str,
        default = "deep",
    )

    def validate(self):
        if (self.coaddName == "chiSquared") != (self.filter is None):
            raise FieldValidationError(
                field=CoaddSrcReferencesConfig.coaddName,
                config=self,
                msg="filter may be None if and only if coaddName is chiSquared"
                )

class CoaddSrcReferencesTask(BaseReferencesTask):
    """A references task implementation that loads the coadd_src dataset directly from disk using
    the butler.
    """

    ConfigClass = CoaddSrcReferencesConfig

    def __init__(self, butler=None, schema=None, **kwargs):
        BaseReferencesTask.__init__(self, butler=butler, schema=schema, **kwargs)
        if schema is None:
            assert butler is not None, "No butler nor schema provided"
            schema = butler.get(self.config.coaddName + "Coadd_ref_schema", immediate=True).getSchema()
        self.schema = schema

    def getWcs(self, dataRef):
        """Return the WCS for reference sources.  The given dataRef must include the tract in its dataId.
        """
        skyMap = dataRef.get(self.config.coaddName + "Coadd_skyMap", immediate=True)
        return skyMap[dataRef.dataId["tract"]].getWcs()

    def fetchInPatches(self, dataRef, patchList):
        """An implementation of BaseReferencesTask.fetchInPatches that loads 'coadd_src' catalogs
        using the butler.

        The given dataRef must include the tract in its dataId.
        """
        dataset = self.config.coaddName + "Coadd_src"
        tract = dataRef.dataId["tract"]
        butler = dataRef.butlerSubset.butler
        for patch in patchList:
            dataId = {'tract': tract, 'patch': "%d,%d" % patch.getIndex()}
            if self.config.filter is not None:
                dataId['filter'] = self.config.filter

            if not butler.datasetExists(dataset, dataId):
                #raise TaskError("Reference %s doesn't exist" % (dataId,))
                self.log.warn("Reference %s doesn't exist" % (dataId,))
                continue
            self.log.info("Getting references in %s" % (dataId,))
            catalog = butler.get(dataset, dataId, immediate=True)
            if self.config.removePatchOverlaps:
                bbox = lsst.afw.geom.Box2D(patch.getInnerBBox())
                for source in catalog:
                    if bbox.contains(source.getCentroid()):
                        yield source
            else:
                for source in catalog:
                    yield source

    def fetchInBox(self, dataRef, bbox, wcs):
        """Return reference sources that overlap a region defined by a pixel-coordinate bounding box
        and corresponding Wcs.

        @param[in] dataRef    ButlerDataRef; the implied data ID must contain the 'tract' key.
        @param[in] bbox       a afw.geom.Box2I or Box2D that defines the region in pixel coordinates
        @param[in] wcs        afw.image.Wcs that maps the bbox to sky coordinates

        @return an iterable of reference sources
        """
        skyMap = dataRef.get(self.config.coaddName + "Coadd_skyMap", immediate=True)
        tract = skyMap[dataRef.dataId["tract"]]
        coordList = [wcs.pixelToSky(corner) for corner in lsst.afw.geom.Box2D(bbox).getCorners()]
        self.log.info("Getting references in region with corners %s [degrees]" %
                      ", ".join("(%s)" % coord.getPosition(lsst.afw.geom.degrees) for coord in coordList))
        patchList = tract.findPatchList(coordList)
        return self.subset(self.fetchInPatches(dataRef, patchList), bbox, wcs)
