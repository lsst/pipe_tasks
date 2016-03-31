from __future__ import absolute_import, division, print_function
#
# LSST Data Management System
#
# Copyright 2008-2016 AURA/LSST.
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
# see <https://www.lsstcorp.org/LegalNotices/>.
#

from lsst.meas.algorithms import getRefFluxField, LoadReferenceObjectsTask
from lsst.pipe.tasks.indexReferenceTask import IngestIndexedReferenceTask

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
__all__ = ["LoadIndexedReferenceObjectsTask"]


class LoadIndexedReferenceObjectConfig(pexConfig.Config):
    ingest_config_name = pexConfig.Field(
        dtype = str,
        default = 'IngestIndexedReferenceTask_config',
        doc = 'Name of the config dataset used to ingest the reference'
    )


class LoadIndexedReferenceObjectsTask(LoadReferenceObjectsTask):
    ConfigClass = LoadIndexedReferenceObjectConfig
    _DefaultName = 'LoadIndexedReferenceObjectsTask'

    def __init__(self, butler, ingest_factory=IngestIndexedReferenceTask, *args, **kwargs):
        LoadReferenceObjectsTask.__init__(self, *args, **kwargs)
        ingest_config = butler.get(self.config.ingest_config_name, immediate=True)
        ingester = ingest_factory(butler=butler, config=ingest_config)
        self.indexer = ingester.indexer
        self.make_data_id = ingester.make_data_id
        self.ref_dataset_name = ingester.config.ref_dataset_name
        self.butler = butler

    @pipeBase.timeMethod
    def loadSkyCircle(self, ctrCoord, radius, filterName=None):
        """!Load reference objects that overlap a circular sky region

        @param[in] ctrCoord  center of search region (an lsst.afw.geom.Coord)
        @param[in] radius  radius of search region (an lsst.afw.geom.Angle)
        @param[in] filterName  name of filter, or None for the default filter;
            used for flux values in case we have flux limits (which are not yet implemented)

        @return an lsst.pipe.base.Struct containing:
        - refCat a catalog of reference objects with the
            \link meas_algorithms_loadReferenceObjects_Schema standard schema \endlink
            as documented in LoadReferenceObjects, including photometric, resolved and variable;
            hasCentroid is False for all objects.  None if no ref objects available.
        - fluxField = name of flux field for specified filterName.  None if refCat is None.
        """
        id_list, boundary_mask = self.indexer.get_pixel_ids(ctrCoord, radius)
        shards = self.get_shards(id_list)
        refCat = self.butler.get(self.ref_dataset_name, dataId=self.make_data_id('master_schema'),
                                 immediate=True)
        fluxField = getRefFluxField(schema=refCat.schema, filterName=filterName)
        for shard, is_on_boundary in zip(shards, boundary_mask):
            if shard is None:
                continue
            if is_on_boundary:
                refCat.extend(self._trim_to_circle(shard, ctrCoord, radius))
            else:
                refCat.extend(shard)
        # return reference catalog
        return pipeBase.Struct(
            refCat = refCat,
            fluxField = fluxField,
        )

    def get_shards(self, id_list):
        """!Get all shards that touch a circular aperture

        @param[in] id_list  A list of integer pixel ids
        @param[out] a list of SourceCatalogs for each pixel, None if not data exists
        """
        shards = []
        for pixel_id in id_list:
            if self.butler.datasetExists(self.ref_dataset_name, dataId=self.make_data_id(pixel_id)):
                shards.append(self.butler.get(self.ref_dataset_name,
                              dataId=self.make_data_id(pixel_id), immediate=True))
        return shards

    def _trim_to_circle(self, catalog_shard, ctrCoord, radius):
        """!Trim a catalog to a circular aperture.

        @param[in] catalog_shard  SourceCatalog to be trimmed
        @param[in] ctrCoord  afw.Coord to compare each record to
        @param[in] radius  afwGeom.Angle indicating maximume separation
        @param[out] a SourceCatalog constructed from records that fall in the circular aperture
        """
        temp_cat = type(catalog_shard)(catalog_shard.schema)
        for record in catalog_shard:
            if record.getCoord().angularSeparation(ctrCoord) < radius:
                temp_cat.append(record)
        return temp_cat

