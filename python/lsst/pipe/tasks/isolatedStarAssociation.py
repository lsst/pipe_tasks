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

__all__ = ['IsolatedStarAssociationConnections',
           'IsolatedStarAssociationConfig',
           'IsolatedStarAssociationTask']

import numpy as np
import pandas as pd
from smatch.matcher import Matcher

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.skymap import BaseSkyMap
from lsst.meas.algorithms.sourceSelector import sourceSelectorRegistry


class IsolatedStarAssociationConnections(pipeBase.PipelineTaskConnections,
                                         dimensions=('instrument', 'tract', 'skymap',),
                                         defaultTemplates={}):
    source_table_visit = pipeBase.connectionTypes.Input(
        doc='Source table in parquet format, per visit',
        name='sourceTable_visit',
        storageClass='DataFrame',
        dimensions=('instrument', 'visit'),
        deferLoad=True,
        multiple=True,
    )
    skymap = pipeBase.connectionTypes.Input(
        doc="Input definition of geometry/bbox and projection/wcs for warped exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass='SkyMap',
        dimensions=('skymap',),
    )
    isolated_star_sources = pipeBase.connectionTypes.Output(
        doc='Catalog of individual sources for the isolated stars',
        name='isolated_star_sources',
        storageClass='DataFrame',
        dimensions=('instrument', 'tract', 'skymap'),
    )
    isolated_star_cat = pipeBase.connectionTypes.Output(
        doc='Catalog of isolated star positions',
        name='isolated_star_cat',
        storageClass='DataFrame',
        dimensions=('instrument', 'tract', 'skymap'),
    )


class IsolatedStarAssociationConfig(pipeBase.PipelineTaskConfig,
                                    pipelineConnections=IsolatedStarAssociationConnections):
    """Configuration for IsolatedStarAssociationTask."""

    inst_flux_field = pexConfig.Field(
        doc=('Full name of instFlux field to use for s/n selection and persistence. '
             'The associated flag will be implicity included in bad_flags. '
             'Note that this is expected to end in ``instFlux``.'),
        dtype=str,
        default='apFlux_12_0_instFlux',
    )
    match_radius = pexConfig.Field(
        doc='Match radius (arcseconds)',
        dtype=float,
        default=1.0,
    )
    isolation_radius = pexConfig.Field(
        doc=('Isolation radius (arcseconds).  Any stars with average centroids '
             'within this radius of another star will be rejected from the final '
             'catalog.  This radius should be at least 2x match_radius.'),
        dtype=float,
        default=2.0,
    )
    band_order = pexConfig.ListField(
        doc=(('Ordered list of bands to use for matching/storage. '
              'Any bands not listed will not be matched.')),
        dtype=str,
        default=['i', 'z', 'r', 'g', 'y', 'u'],
    )
    id_column = pexConfig.Field(
        doc='Name of column with source id.',
        dtype=str,
        default='sourceId',
    )
    ra_column = pexConfig.Field(
        doc='Name of column with right ascension.',
        dtype=str,
        default='ra',
    )
    dec_column = pexConfig.Field(
        doc='Name of column with declination.',
        dtype=str,
        default='dec',
    )
    physical_filter_column = pexConfig.Field(
        doc='Name of column with physical filter name',
        dtype=str,
        default='physical_filter',
    )
    band_column = pexConfig.Field(
        doc='Name of column with band name',
        dtype=str,
        default='band',
    )
    extra_columns = pexConfig.ListField(
        doc='Extra names of columns to read and persist (beyond instFlux and error).',
        dtype=str,
        default=['x',
                 'y',
                 'xErr',
                 'yErr',
                 'apFlux_17_0_instFlux',
                 'apFlux_17_0_instFluxErr',
                 'apFlux_17_0_flag',
                 'localBackground_instFlux',
                 'localBackground_flag',
                 'ixx',
                 'iyy',
                 'ixy',]
    )
    source_selector = sourceSelectorRegistry.makeField(
        doc='How to select sources.  Under normal usage this should not be changed.',
        default='science'
    )

    def setDefaults(self):
        super().setDefaults()

        source_selector = self.source_selector['science']
        source_selector.setDefaults()

        source_selector.doFlags = True
        source_selector.doUnresolved = True
        source_selector.doSignalToNoise = True
        source_selector.doIsolated = True
        source_selector.doRequireFiniteRaDec = True
        source_selector.doRequirePrimary = True

        source_selector.signalToNoise.minimum = 10.0
        source_selector.signalToNoise.maximum = 1000.0

        flux_flag_name = self.inst_flux_field.replace("instFlux", "flag")

        source_selector.flags.bad = ['pixelFlags_edge',
                                     'pixelFlags_interpolatedCenter',
                                     'pixelFlags_saturatedCenter',
                                     'pixelFlags_crCenter',
                                     'pixelFlags_bad',
                                     'pixelFlags_interpolated',
                                     'pixelFlags_saturated',
                                     'centroid_flag',
                                     flux_flag_name]

        source_selector.signalToNoise.fluxField = self.inst_flux_field
        source_selector.signalToNoise.errField = self.inst_flux_field + 'Err'

        source_selector.isolated.parentName = 'parentSourceId'
        source_selector.isolated.nChildName = 'deblend_nChild'

        source_selector.unresolved.maximum = 0.5
        source_selector.unresolved.name = 'extendedness'

        source_selector.requireFiniteRaDec.raColName = self.ra_column
        source_selector.requireFiniteRaDec.decColName = self.dec_column


class IsolatedStarAssociationTask(pipeBase.PipelineTask):
    """Associate sources into isolated star catalogs.
    """
    ConfigClass = IsolatedStarAssociationConfig
    _DefaultName = 'isolatedStarAssociation'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.makeSubtask('source_selector')
        # Only log warning and fatal errors from the source_selector
        self.source_selector.log.setLevel(self.source_selector.log.WARN)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        input_ref_dict = butlerQC.get(inputRefs)

        tract = butlerQC.quantum.dataId['tract']

        source_table_refs = input_ref_dict['source_table_visit']

        self.log.info('Running with %d source_table_visit dataRefs',
                      len(source_table_refs))

        source_table_ref_dict_temp = {source_table_ref.dataId['visit']: source_table_ref for
                                      source_table_ref in source_table_refs}

        bands = {source_table_ref.dataId['band'] for source_table_ref in source_table_refs}
        for band in bands:
            if band not in self.config.band_order:
                self.log.warning('Input data has data from band %s but that band is not '
                                 'configured for matching', band)

        # TODO: Sort by visit until DM-31701 is done and we have deterministic
        # dataset ordering.
        source_table_ref_dict = {visit: source_table_ref_dict_temp[visit] for
                                 visit in sorted(source_table_ref_dict_temp.keys())}

        struct = self.run(input_ref_dict['skymap'], tract, source_table_ref_dict)

        butlerQC.put(pd.DataFrame(struct.star_source_cat),
                     outputRefs.isolated_star_sources)
        butlerQC.put(pd.DataFrame(struct.star_cat),
                     outputRefs.isolated_star_cat)

    def run(self, skymap, tract, source_table_ref_dict):
        """Run the isolated star association task.

        Parameters
        ----------
        skymap : `lsst.skymap.SkyMap`
            Skymap object.
        tract : `int`
            Tract number.
        source_table_ref_dict : `dict`
            Dictionary of source_table refs.  Key is visit, value is dataref.

        Returns
        -------
        struct : `lsst.pipe.base.struct`
            Struct with outputs for persistence.
        """
        star_source_cat = self._make_all_star_sources(skymap[tract], source_table_ref_dict)

        primary_bands = self.config.band_order

        # Do primary matching
        primary_star_cat = self._match_primary_stars(primary_bands, star_source_cat)

        if len(primary_star_cat) == 0:
            return pipeBase.Struct(star_source_cat=np.zeros(0, star_source_cat.dtype),
                                   star_cat=np.zeros(0, primary_star_cat.dtype))

        # Remove neighbors
        primary_star_cat = self._remove_neighbors(primary_star_cat)

        if len(primary_star_cat) == 0:
            return pipeBase.Struct(star_source_cat=np.zeros(0, star_source_cat.dtype),
                                   star_cat=np.zeros(0, primary_star_cat.dtype))

        # Crop to inner tract region
        inner_tract_ids = skymap.findTractIdArray(primary_star_cat[self.config.ra_column],
                                                  primary_star_cat[self.config.dec_column],
                                                  degrees=True)
        use = (inner_tract_ids == tract)
        self.log.info('Total of %d isolated stars in inner tract.', use.sum())

        primary_star_cat = primary_star_cat[use]

        if len(primary_star_cat) == 0:
            return pipeBase.Struct(star_source_cat=np.zeros(0, star_source_cat.dtype),
                                   star_cat=np.zeros(0, primary_star_cat.dtype))

        # Set the unique ids.
        primary_star_cat['isolated_star_id'] = self._compute_unique_ids(skymap,
                                                                        tract,
                                                                        len(primary_star_cat))

        # Match to sources.
        star_source_cat, primary_star_cat = self._match_sources(primary_bands,
                                                                star_source_cat,
                                                                primary_star_cat)

        return pipeBase.Struct(star_source_cat=star_source_cat,
                               star_cat=primary_star_cat)

    def _make_all_star_sources(self, tract_info, source_table_ref_dict):
        """Make a catalog of all the star sources.

        Parameters
        ----------
        tract_info : `lsst.skymap.TractInfo`
            Information about the tract.
        source_table_ref_dict : `dict`
            Dictionary of source_table refs.  Key is visit, value is dataref.

        Returns
        -------
        star_source_cat : `np.ndarray`
            Catalog of star sources.
        """
        # Internally, we use a numpy recarray, they are by far the fastest
        # option in testing for relatively narrow tables.
        # (have not tested wide tables)
        all_columns, persist_columns = self._get_source_table_visit_column_names()
        poly = tract_info.outer_sky_polygon

        tables = []
        for visit in source_table_ref_dict:
            source_table_ref = source_table_ref_dict[visit]
            df = source_table_ref.get(parameters={'columns': all_columns})
            df.reset_index(inplace=True)

            goodSrc = self.source_selector.selectSources(df)

            table = df[persist_columns][goodSrc.selected].to_records()

            # Append columns that include the row in the source table
            # and the matched object index (to be filled later).
            table = np.lib.recfunctions.append_fields(table,
                                                      ['source_row',
                                                       'obj_index'],
                                                      [np.where(goodSrc.selected)[0],
                                                       np.zeros(goodSrc.selected.sum(), dtype=np.int32)],
                                                      dtypes=['i4', 'i4'],
                                                      usemask=False)

            # We cut to the outer tract polygon to ensure consistent matching
            # from tract to tract.
            tract_use = poly.contains(np.deg2rad(table[self.config.ra_column]),
                                      np.deg2rad(table[self.config.dec_column]))

            tables.append(table[tract_use])

        # Combine tables
        star_source_cat = np.concatenate(tables)

        return star_source_cat

    def _get_source_table_visit_column_names(self):
        """Get the list of sourceTable_visit columns from the config.

        Returns
        -------
        all_columns : `list` [`str`]
            All columns to read
        persist_columns : `list` [`str`]
            Columns to persist (excluding selection columns)
        """
        columns = [self.config.id_column,
                   'visit', 'detector',
                   self.config.ra_column, self.config.dec_column,
                   self.config.physical_filter_column, self.config.band_column,
                   self.config.inst_flux_field, self.config.inst_flux_field + 'Err']
        columns.extend(self.config.extra_columns)

        all_columns = columns.copy()
        if self.source_selector.config.doFlags:
            all_columns.extend(self.source_selector.config.flags.bad)
        if self.source_selector.config.doUnresolved:
            all_columns.append(self.source_selector.config.unresolved.name)
        if self.source_selector.config.doIsolated:
            all_columns.append(self.source_selector.config.isolated.parentName)
            all_columns.append(self.source_selector.config.isolated.nChildName)
        if self.source_selector.config.doRequirePrimary:
            all_columns.append(self.source_selector.config.requirePrimary.primaryColName)

        return all_columns, columns

    def _match_primary_stars(self, primary_bands, star_source_cat):
        """Match primary stars.

        Parameters
        ----------
        primary_bands : `list` [`str`]
            Ordered list of primary bands.
        star_source_cat : `np.ndarray`
            Catalog of star sources.

        Returns
        -------
        primary_star_cat : `np.ndarray`
            Catalog of primary star positions
        """
        ra_col = self.config.ra_column
        dec_col = self.config.dec_column

        dtype = self._get_primary_dtype(primary_bands)

        primary_star_cat = None
        for primary_band in primary_bands:
            use = (star_source_cat['band'] == primary_band)

            ra = star_source_cat[ra_col][use]
            dec = star_source_cat[dec_col][use]

            with Matcher(ra, dec) as matcher:
                try:
                    # New smatch API
                    idx = matcher.query_groups(self.config.match_radius/3600., min_match=1)
                except AttributeError:
                    # Old smatch API
                    idx = matcher.query_self(self.config.match_radius/3600., min_match=1)

            count = len(idx)

            if count == 0:
                self.log.info('Found 0 primary stars in %s band.', primary_band)
                continue

            band_cat = np.zeros(count, dtype=dtype)
            band_cat['primary_band'] = primary_band

            # If the tract cross ra=0 (that is, it has both low ra and high ra)
            # then we need to remap all ra values from [0, 360) to [-180, 180)
            # before doing any position averaging.
            remapped = False
            if ra.min() < 60.0 and ra.max() > 300.0:
                ra_temp = (ra + 180.0) % 360. - 180.
                remapped = True
            else:
                ra_temp = ra

            # Compute mean position for each primary star
            for i, row in enumerate(idx):
                row = np.array(row)
                band_cat[ra_col][i] = np.mean(ra_temp[row])
                band_cat[dec_col][i] = np.mean(dec[row])

            if remapped:
                # Remap ra back to [0, 360)
                band_cat[ra_col] %= 360.0

            # Match to previous band catalog(s), and remove duplicates.
            if primary_star_cat is None or len(primary_star_cat) == 0:
                primary_star_cat = band_cat
            else:
                with Matcher(band_cat[ra_col], band_cat[dec_col]) as matcher:
                    idx = matcher.query_radius(primary_star_cat[ra_col],
                                               primary_star_cat[dec_col],
                                               self.config.match_radius/3600.)
                # Any object with a match should be removed.
                match_indices = np.array([i for i in range(len(idx)) if len(idx[i]) > 0])
                if len(match_indices) > 0:
                    band_cat = np.delete(band_cat, match_indices)

                primary_star_cat = np.append(primary_star_cat, band_cat)
            self.log.info('Found %d primary stars in %s band.', len(band_cat), primary_band)

        # If everything was cut, we still want the correct datatype.
        if primary_star_cat is None:
            primary_star_cat = np.zeros(0, dtype=dtype)

        return primary_star_cat

    def _remove_neighbors(self, primary_star_cat):
        """Remove neighbors from the primary star catalog.

        Parameters
        ----------
        primary_star_cat : `np.ndarray`
            Primary star catalog.

        Returns
        -------
        primary_star_cat_cut : `np.ndarray`
            Primary star cat with neighbors removed.
        """
        ra_col = self.config.ra_column
        dec_col = self.config.dec_column

        with Matcher(primary_star_cat[ra_col], primary_star_cat[dec_col]) as matcher:
            # By setting min_match=2 objects that only match to themselves
            # will not be recorded.
            try:
                # New smatch API
                idx = matcher.query_groups(self.config.isolation_radius/3600., min_match=2)
            except AttributeError:
                # Old smatch API
                idx = matcher.query_self(self.config.isolation_radius/3600., min_match=2)

        try:
            neighbor_indices = np.concatenate(idx)
        except ValueError:
            neighbor_indices = np.zeros(0, dtype=int)

        if len(neighbor_indices) > 0:
            neighbored = np.unique(neighbor_indices)
            self.log.info('Cutting %d objects with close neighbors.', len(neighbored))
            primary_star_cat = np.delete(primary_star_cat, neighbored)

        return primary_star_cat

    def _match_sources(self, bands, star_source_cat, primary_star_cat):
        """Match individual sources to primary stars.

        Parameters
        ----------
        bands : `list` [`str`]
            List of bands.
        star_source_cat : `np.ndarray`
            Array of star sources.
        primary_star_cat : `np.ndarray`
            Array of primary stars.

        Returns
        -------
        star_source_cat_sorted : `np.ndarray`
            Sorted and cropped array of star sources.
        primary_star_cat : `np.ndarray`
            Catalog of isolated stars, with indexes to star_source_cat_cut.
        """
        ra_col = self.config.ra_column
        dec_col = self.config.dec_column

        # We match sources per-band because it allows us to have sorted
        # sources for easy retrieval of per-band matches.
        n_source_per_band_per_obj = np.zeros((len(bands),
                                              len(primary_star_cat)),
                                             dtype=np.int32)
        band_uses = []
        idxs = []
        with Matcher(primary_star_cat[ra_col], primary_star_cat[dec_col]) as matcher:
            for b, band in enumerate(bands):
                band_use, = np.where(star_source_cat['band'] == band)

                idx = matcher.query_radius(star_source_cat[ra_col][band_use],
                                           star_source_cat[dec_col][band_use],
                                           self.config.match_radius/3600.)
                n_source_per_band_per_obj[b, :] = np.array([len(row) for row in idx])
                idxs.append(idx)
                band_uses.append(band_use)

        n_source_per_obj = np.sum(n_source_per_band_per_obj, axis=0)

        primary_star_cat['nsource'] = n_source_per_obj
        primary_star_cat['source_cat_index'][1:] = np.cumsum(n_source_per_obj)[:-1]

        n_tot_source = primary_star_cat['source_cat_index'][-1] + primary_star_cat['nsource'][-1]

        # Temporary arrays until we crop/sort the source catalog
        source_index = np.zeros(n_tot_source, dtype=np.int32)
        obj_index = np.zeros(n_tot_source, dtype=np.int32)

        ctr = 0
        for i in range(len(primary_star_cat)):
            obj_index[ctr: ctr + n_source_per_obj[i]] = i
            for b in range(len(bands)):
                source_index[ctr: ctr + n_source_per_band_per_obj[b, i]] = band_uses[b][idxs[b][i]]
                ctr += n_source_per_band_per_obj[b, i]

        source_cat_index_band_offset = np.cumsum(n_source_per_band_per_obj, axis=0)

        for b, band in enumerate(bands):
            primary_star_cat[f'nsource_{band}'] = n_source_per_band_per_obj[b, :]
            if b == 0:
                # The first band listed is the same as the overall star
                primary_star_cat[f'source_cat_index_{band}'] = primary_star_cat['source_cat_index']
            else:
                # Other band indices are offset from the previous band
                primary_star_cat[f'source_cat_index_{band}'] = (primary_star_cat['source_cat_index']
                                                                + source_cat_index_band_offset[b - 1, :])

        star_source_cat = star_source_cat[source_index]
        star_source_cat['obj_index'] = obj_index

        return star_source_cat, primary_star_cat

    def _compute_unique_ids(self, skymap, tract, nstar):
        """Compute unique star ids.

        This is a simple hash of the tract and star to provide an
        id that is unique for a given processing.

        Parameters
        ----------
        skymap : `lsst.skymap.Skymap`
            Skymap object.
        tract : `int`
            Tract id number.
        nstar : `int`
            Number of stars.

        Returns
        -------
        ids : `np.ndarray`
            Array of unique star ids.
        """
        # The end of the id will be big enough to hold the tract number
        mult = 10**(int(np.log10(len(skymap))) + 1)

        return (np.arange(nstar) + 1)*mult + tract

    def _get_primary_dtype(self, primary_bands):
        """Get the numpy datatype for the primary star catalog.

        Parameters
        ----------
        primary_bands : `list` [`str`]
            List of primary bands.

        Returns
        -------
        dtype : `numpy.dtype`
            Datatype of the primary catalog.
        """
        max_len = max([len(primary_band) for primary_band in primary_bands])

        dtype = [('isolated_star_id', 'i8'),
                 (self.config.ra_column, 'f8'),
                 (self.config.dec_column, 'f8'),
                 ('primary_band', f'U{max_len}'),
                 ('source_cat_index', 'i4'),
                 ('nsource', 'i4')]

        for band in primary_bands:
            dtype.append((f'source_cat_index_{band}', 'i4'))
            dtype.append((f'nsource_{band}', 'i4'))

        return dtype
