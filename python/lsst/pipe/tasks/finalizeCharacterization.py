#
# LSST Data Management System
# Copyright 2008-2022 AURA/LSST.
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
import esutil
import pandas as pd

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
import lsst.afw.table as afwTable
import lsst.meas.algorithms as measAlg
import lsst.meas.extensions.psfex.psfexPsfDeterminer  # noqa: F401
from lsst.meas.algorithms import MeasureApCorrTask
from lsst.meas.base import SingleFrameMeasurementTask, ApplyApCorrTask
from lsst.meas.algorithms.sourceSelector import sourceSelectorRegistry

from .reserveIsolatedStars import ReserveIsolatedStarsTask

__all__ = ['FinalizeCharacterizationConnections',
           'FinalizeCharacterizationConfig',
           'FinalizeCharacterizationTask']


class FinalizeCharacterizationConnections(pipeBase.PipelineTaskConnections,
                                          dimensions=('instrument', 'visit',),
                                          defaultTemplates={}):
    src_schema = pipeBase.connectionTypes.InitInput(
        doc='Input schema used for src catalogs.',
        name='src_schema',
        storageClass='SourceCatalog',
    )
    srcs = pipeBase.connectionTypes.Input(
        doc='Source catalogs for the visit',
        name='src',
        storageClass='SourceCatalog',
        dimensions=('instrument', 'visit', 'detector'),
        deferLoad=True,
        multiple=True,
    )
    calexps = pipeBase.connectionTypes.Input(
        doc='Calexps for the visit',
        name='calexp',
        storageClass='ExposureF',
        dimensions=('instrument', 'visit', 'detector'),
        deferLoad=True,
        multiple=True,
    )
    isolated_star_cats = pipeBase.connectionTypes.Input(
        doc='Catalog of isolated stars',
        name='isolated_star_cat',
        storageClass='DataFrame',
        dimensions=('instrument', 'tract', 'skymap'),
        deferLoad=True,
        multiple=True,
    )
    isolated_star_sources = pipeBase.connectionTypes.Input(
        doc='Catalog of isolated star sources',
        name='isolated_star_sources',
        storageClass='DataFrame',
        dimensions=('instrument', 'tract', 'skymap'),
        deferLoad=True,
        multiple=True,
    )
    finalized_psf_cat = pipeBase.connectionTypes.Output(
        doc=('Per-visit finalized psf models.  These catalogs use detector id '
             'for the id and are sorted for fast lookups of a detector.'),
        name='finalized_psf_catalog',
        storageClass='ExposureCatalog',
        dimensions=('instrument', 'visit'),
    )
    finalized_ap_corr_map_cat = pipeBase.connectionTypes.Output(
        doc=('Per-visit finalized aperture correction maps.  These catalogs '
             'use detector id for the id and are sorted for fast lookups '
             'of a detector.'),
        name='finalized_ap_corr_map_catalog',
        storageClass='ExposureCatalog',
        dimensions=('instrument', 'visit'),
    )
    finalized_src_table = pipeBase.connectionTypes.Output(
        doc=('Per-visit catalog of measurements for psf/flag/etc.'),
        name='finalized_src_table',
        storageClass='DataFrame',
        dimensions=('instrument', 'visit'),
    )


class FinalizeCharacterizationConfig(pipeBase.PipelineTaskConfig,
                                     pipelineConnections=FinalizeCharacterizationConnections):
    """Configuration for FinalizeCharacterizationTask."""
    source_selector = sourceSelectorRegistry.makeField(
        doc="How to select sources",
        default="science"
    )
    id_column = pexConfig.Field(
        doc='Name of column in isolated_star_sources with source id.',
        dtype=str,
        default='sourceId',
    )
    reserve_selection = pexConfig.ConfigurableField(
        target=ReserveIsolatedStarsTask,
        doc='Task to select reserved stars',
    )
    make_psf_candidates = pexConfig.ConfigurableField(
        target=measAlg.MakePsfCandidatesTask,
        doc='Task to make psf candidates from selected stars.',
    )
    psf_determiner = measAlg.psfDeterminerRegistry.makeField(
        'PSF Determination algorithm',
        default='psfex'
    )
    measurement = pexConfig.ConfigurableField(
        target=SingleFrameMeasurementTask,
        doc='Measure sources for aperture corrections'
    )
    measure_ap_corr = pexConfig.ConfigurableField(
        target=MeasureApCorrTask,
        doc="Subtask to measure aperture corrections"
    )
    apply_ap_corr = pexConfig.ConfigurableField(
        target=ApplyApCorrTask,
        doc="Subtask to apply aperture corrections"
    )

    def setDefaults(self):
        source_selector = self.source_selector['science']
        source_selector.setDefaults()

        # We use the source selector only to select out flagged objects
        # and signal-to-noise.  Isolated, unresolved sources are handled
        # by the isolated star catalog.

        source_selector.doFlags = True
        source_selector.doSignalToNoise = True

        source_selector.signalToNoise.minimum = 20.0
        source_selector.signalToNoise.maximum = 1000.0

        source_selector.signalToNoise.fluxField = 'base_GaussianFlux_instFlux'
        source_selector.signalToNoise.errField = 'base_GaussianFlux_instFluxErr'

        source_selector.flags.bad = ['base_PixelFlags_flag_edge',
                                     'base_PixelFlags_flag_interpolatedCenter',
                                     'base_PixelFlags_flag_saturatedCenter',
                                     'base_PixelFlags_flag_crCenter',
                                     'base_PixelFlags_flag_bad',
                                     'base_PixelFlags_flag_interpolated',
                                     'base_PixelFlags_flag_saturated',
                                     'slot_Centroid_flag',
                                     'base_GaussianFlux_flag']

        self.measure_ap_corr.sourceSelector['flagged'].field = 'final_psf_used'

        # Turn off slot setting for measurement for centroid and shape
        # (for which we use the input src catalog measurements)
        self.measurement.slots.centroid = None
        self.measurement.slots.apFlux = None
        self.measurement.slots.calibFlux = None


class FinalizeCharacterizationTask(pipeBase.PipelineTask):
    """Run final characterization on exposures."""
    ConfigClass = FinalizeCharacterizationConfig
    _DefaultName = 'finalize_characterization'

    def __init__(self, initInputs=None, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)

        self.schema_mapper, self.schema = self._make_output_schema_mapper(
            initInputs['src_schema'].schema
        )

        self.makeSubtask('reserve_selection')
        self.makeSubtask('source_selector')
        self.makeSubtask('make_psf_candidates')
        self.makeSubtask('psf_determiner')
        self.makeSubtask('measurement', schema=self.schema)
        self.makeSubtask('measure_ap_corr', schema=self.schema)
        self.makeSubtask('apply_ap_corr', schema=self.schema)

        # Only log warning and fatal errors from the source_selector
        self.source_selector.log.setLevel(self.source_selector.log.WARN)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        input_ref_dict = butlerQC.get(inputRefs)

        band = butlerQC.quantum.dataId['band']
        visit = butlerQC.quantum.dataId['visit']

        src_dict_temp = {ref.dataId['detector']: ref
                         for ref in input_ref_dict['srcs']}
        calexp_dict_temp = {ref.dataId['detector']: ref
                            for ref in input_ref_dict['calexps']}
        isolated_star_cat_dict_temp = {ref.dataId['tract']: ref
                                       for ref in input_ref_dict['isolated_star_cats']}
        isolated_star_source_dict_temp = {ref.dataId['tract']: ref
                                          for ref in input_ref_dict['isolated_star_sources']}
        # TODO: Sort until DM-31701 is done and we have deterministic
        # dataset ordering.
        src_dict = {detector: src_dict_temp[detector] for
                    detector in sorted(src_dict_temp.keys())}
        calexp_dict = {detector: calexp_dict_temp[detector] for
                       detector in sorted(calexp_dict_temp.keys())}
        isolated_star_cat_dict = {tract: isolated_star_cat_dict_temp[tract] for
                                  tract in sorted(isolated_star_cat_dict_temp.keys())}
        isolated_star_source_dict = {tract: isolated_star_source_dict_temp[tract] for
                                     tract in sorted(isolated_star_source_dict_temp.keys())}

        struct = self.run(visit,
                          band,
                          isolated_star_cat_dict,
                          isolated_star_source_dict,
                          src_dict,
                          calexp_dict)

        butlerQC.put(struct.psf_cat,
                     outputRefs.finalized_psf_cat)
        butlerQC.put(struct.ap_corr_map_cat,
                     outputRefs.finalized_ap_corr_map_cat)
        butlerQC.put(pd.DataFrame(struct.output_table),
                     outputRefs.finalized_src_table)

    def run(self, visit, band, isolated_star_cat_dict, isolated_star_source_dict, src_dict, calexp_dict):
        """
        Run the FinalizeCharacterizationTask.

        Parameters
        ----------
        visit : `int`
            Visit number.  Used in the output catalogs.
        band : `str`
            Band name.  Used to select reserved stars.
        isolated_star_cat_dict : `dict`
            Per-tract dict of isolated star catalog references.
        isolated_star_source_dict : `dict`
            Per-tract dict of isolated star source catalog references.
        src_dict : `dict`
            Per-detector dict of src catalog references.
        calexp_dict : `dict`
            Per-detector dict of calibrated exposure references.

        Returns
        -------
        struct : `lsst.pipe.base.struct`
            Struct with outputs for persistence.
        """
        # We do not need the isolated star table at this time.
        _, isolated_source_table = self._concat_isolated_star_cats(
            band,
            isolated_star_cat_dict,
            isolated_star_source_dict
        )

        exposure_cat_schema = afwTable.ExposureTable.makeMinimalSchema()
        exposure_cat_schema.addField('visit', type='I', doc='Visit number')

        metadata = dafBase.PropertyList()
        metadata.add("COMMENT", "Catalog id is detector id, sorted.")
        metadata.add("COMMENT", "Only detectors with data have entries.")

        psf_cat = afwTable.ExposureCatalog(exposure_cat_schema)
        psf_cat.setMetadata(metadata)
        ap_corr_cat = afwTable.ExposureCatalog(exposure_cat_schema)
        ap_corr_cat.setMetadata(metadata)

        measured_src_tables = []

        for detector in src_dict:
            src = src_dict[detector].get()
            exposure = calexp_dict[detector].get()

            psf, ap_corr_map, measured_src = self._compute_psf_and_ap_corr_map(
                visit,
                detector,
                exposure,
                src,
                isolated_source_table
            )

            # And now we package it together...
            psf_record = psf_cat.addNew()
            psf_record['id'] = int(detector)
            psf_record['visit'] = visit
            if psf is not None:
                psf_record.setPsf(psf)

            ap_corr_record = ap_corr_cat.addNew()
            ap_corr_record['id'] = int(detector)
            ap_corr_record['visit'] = visit
            if ap_corr_map is not None:
                ap_corr_record.setApCorrMap(ap_corr_map)

            measured_src['visit'][:] = visit
            measured_src['detector'][:] = detector

            measured_src_tables.append(measured_src.asAstropy().as_array())

        measured_src_table = np.concatenate(measured_src_tables)

        return pipeBase.Struct(psf_cat=psf_cat,
                               ap_corr_map_cat=ap_corr_cat,
                               output_table=measured_src_table)

    def _make_output_schema_mapper(self, input_schema):
        """Make the schema mapper from the input schema to the output schema.

        Parameters
        ----------
        input_schema : `lsst.afw.table.Schema`
            Input schema.

        Returns
        -------
        mapper : `lsst.afw.table.SchemaMapper`
            Schema mapper
        output_schema : `lsst.afw.table.Schema`
            Output schema (with alias map)
        """
        mapper = afwTable.SchemaMapper(input_schema)
        mapper.addMinimalSchema(afwTable.SourceTable.makeMinimalSchema())
        mapper.addMapping(input_schema['slot_Centroid_x'].asKey())
        mapper.addMapping(input_schema['slot_Centroid_y'].asKey())

        aper_fields = input_schema.extract('base_CircularApertureFlux_*')
        for field, item in aper_fields.items():
            mapper.addMapping(item.key)

        # The following two may be redundant, but then the mapping is a no-op.
        apflux_fields = input_schema.extract('slot_ApFlux_*')
        for field, item in apflux_fields.items():
            mapper.addMapping(item.key)

        calibflux_fields = input_schema.extract('slot_CalibFlux_*')
        for field, item in calibflux_fields.items():
            mapper.addMapping(item.key)

        mapper.addMapping(
            input_schema[self.config.source_selector.active.signalToNoise.fluxField].asKey(),
            'final_psf_selection_flux')
        mapper.addMapping(
            input_schema[self.config.source_selector.active.signalToNoise.errField].asKey(),
            'final_psf_selection_flux_err')

        mapper.editOutputSchema().addField(
            'final_psf_candidate',
            type='Flag',
            doc=('set if the source was a candidate for PSF determination, '
                 'as determined from FinalizeCharacterizationTask.'),
        )
        mapper.editOutputSchema().addField(
            'final_psf_reserved',
            type='Flag',
            doc=('set if source was reserved from PSF determination by '
                 'FinalizeCharacterizationTask.'),
        )
        mapper.editOutputSchema().addField(
            'final_psf_used',
            type='Flag',
            doc=('set if source was used in the PSF determination by '
                 'FinalizeCharacterizationTask.'),
        )
        mapper.editOutputSchema().addField(
            'visit',
            type=np.int32,
            doc='Visit number for the sources.',
        )
        mapper.editOutputSchema().addField(
            'detector',
            type=np.int32,
            doc='Detector number for the sources.',
        )

        alias_map = input_schema.getAliasMap()
        alias_map_output = afwTable.AliasMap()
        alias_map_output.set('slot_Centroid', alias_map.get('slot_Centroid'))
        alias_map_output.set('slot_ApFlux', alias_map.get('slot_ApFlux'))
        alias_map_output.set('slot_CalibFlux', alias_map.get('slot_CalibFlux'))
        output_schema = mapper.getOutputSchema()
        output_schema.setAliasMap(alias_map_output)

        return mapper, output_schema

    def _make_selection_schema_mapper(self, input_schema):
        """Make the schema mapper from the input schema to the selection schema.

        Parameters
        ----------
        input_schema : `lsst.afw.table.Schema`
            Input schema.

        Returns
        -------
        mapper : `lsst.afw.table.SchemaMapper`
            Schema mapper
        selection_schema : `lsst.afw.table.Schema`
            Selection schema (with alias map)
        """
        mapper = afwTable.SchemaMapper(input_schema)
        fields = input_schema.extract('*')
        for field, item in fields.items():
            mapper.addMapping(item.key)

        mapper.editOutputSchema().addField(
            'final_psf_candidate',
            type='Flag',
            doc=('set if the source was a candidate for PSF determination, '
                 'as determined from FinalizeCharacterizationTask.'),
        )
        mapper.editOutputSchema().addField(
            'final_psf_reserved',
            type='Flag',
            doc=('set if source was reserved from PSF determination by '
                 'FinalizeCharacterizationTask.'),
        )
        mapper.editOutputSchema().addField(
            'final_psf_used',
            type='Flag',
            doc=('set if source was used in the PSF determination by '
                 'FinalizeCharacterizationTask.'),
        )

        selection_schema = mapper.getOutputSchema()
        selection_schema.setAliasMap(input_schema.getAliasMap())

        return mapper, selection_schema

    def _concat_isolated_star_cats(self, band, isolated_star_cat_dict, isolated_star_source_dict):
        """
        Concatenate isolated star catalogs and make reserve selection.

        Parameters
        ----------
        band : `str`
            Band name.  Used to select reserved stars.
        isolated_star_cat_dict : `dict`
            Per-tract dict of isolated star catalog references.
        isolated_star_source_dict : `dict`
            Per-tract dict of isolated star source catalog references.

        Returns
        -------
        isolated_table : `np.ndarray` (N,)
            Table of isolated stars, with indexes to isolated sources.
        isolated_source_table : `np.ndarray` (M,)
            Table of isolated sources, with indexes to isolated stars.
        """
        isolated_tables = []
        isolated_sources = []
        merge_cat_counter = 0
        merge_source_counter = 0

        for tract in isolated_star_cat_dict:
            df_cat = isolated_star_cat_dict[tract].get()
            table_cat = df_cat.to_records()

            df_source = isolated_star_source_dict[tract].get(
                parameters={'columns': [self.config.id_column,
                                        'obj_index']}
            )
            table_source = df_source.to_records()

            # Cut isolated star table to those observed in this band, and adjust indices
            use_band, = np.where(table_cat[f'nsource_{band}'] > 0)

            # With the following matching:
            #   table_source[b] <-> table_cat[use_band[a]]
            obj_index = table_source['obj_index'][:]
            a, b = esutil.numpy_util.match(use_band, obj_index)

            # Update indexes and cut to band-selected stars/sources
            table_source['obj_index'][b] = a
            _, index_new = np.unique(a, return_index=True)
            table_cat[f'source_cat_index_{band}'][use_band] = index_new

            table_source = table_source[b]
            table_cat = table_cat[use_band]

            # Add reserved flag column to tables
            table_cat = np.lib.recfunctions.append_fields(
                table_cat,
                'reserved',
                np.zeros(table_cat.size, dtype=bool),
                usemask=False
            )
            table_source = np.lib.recfunctions.append_fields(
                table_source,
                'reserved',
                np.zeros(table_source.size, dtype=bool),
                usemask=False
            )

            # Get reserve star flags
            table_cat['reserved'][:] = self.reserve_selection.run(
                f'{band}_{tract}',
                len(table_cat)
            )
            table_source['reserved'][:] = table_cat['reserved'][table_source['obj_index']]

            # Offset indices to account for tract merging
            table_cat[f'source_cat_index_{band}'] += merge_source_counter
            table_source['obj_index'] += merge_cat_counter

            isolated_tables.append(table_cat)
            isolated_sources.append(table_source)

            merge_cat_counter += len(table_cat)
            merge_source_counter += len(table_source)

        isolated_table = np.concatenate(isolated_tables)
        isolated_source_table = np.concatenate(isolated_sources)

        return isolated_table, isolated_source_table

    def _compute_psf_and_ap_corr_map(self, visit, detector, exposure, src, isolated_source_table):
        """Compute psf model and aperture correction map for a single exposure.

        Parameters
        ----------
        visit : `int`
            Visit number (for logging).
        detector : `int`
            Detector number (for logging).
        exposure : `lsst.afw.image.ExposureF`
        src : `lsst.afw.table.SourceCatalog`
        isolated_source_table : `np.ndarray`

        Returns
        -------
        psf : `lsst.meas.algorithms.ImagePsf`
            PSF Model
        ap_corr_map : `lsst.afw.image.ApCorrMap`
            Aperture correction map.
        measured_src : `lsst.afw.table.SourceCatalog`
            Updated source catalog with measurements, flags and aperture corrections.
        """
        # Apply source selector (s/n, flags, etc.)
        good_src = self.source_selector.selectSources(src)

        # Cut down input src to the selected sources
        selection_mapper, selection_schema = self._make_selection_schema_mapper(src.schema)

        selected_src = afwTable.SourceCatalog(selection_schema)
        selected_src.reserve(good_src.selected.sum())
        selected_src.extend(src[good_src.selected], mapper=selection_mapper)

        # Find the isolated sources and set flags
        matched_src, matched_iso = esutil.numpy_util.match(
            selected_src['id'],
            isolated_source_table[self.config.id_column]
        )

        matched_arr = np.zeros(len(selected_src), dtype=bool)
        matched_arr[matched_src] = True
        selected_src['final_psf_candidate'] = matched_arr

        reserved_arr = np.zeros(len(selected_src), dtype=bool)
        reserved_arr[matched_src] = isolated_source_table['reserved'][matched_iso]
        selected_src['final_psf_reserved'] = reserved_arr

        selected_src = selected_src[selected_src['final_psf_candidate']].copy(deep=True)

        # Make the measured source catalog as well, based on the selected catalog.
        measured_src = afwTable.SourceCatalog(self.schema)
        measured_src.reserve(len(selected_src))
        measured_src.extend(selected_src, mapper=self.schema_mapper)

        # We need to copy over the final_psf flags because they were not in the mapper
        measured_src['final_psf_candidate'] = selected_src['final_psf_candidate']
        measured_src['final_psf_reserved'] = selected_src['final_psf_reserved']

        # Select the psf candidates from the selection catalog
        try:
            psf_selection_result = self.make_psf_candidates.run(selected_src, exposure=exposure)
        except Exception as e:
            self.log.warn('Failed to make psf candidates for visit %d, detector %d: %s',
                          visit, detector, e)
            return None, None, measured_src

        psf_cand_cat = psf_selection_result.goodStarCat

        # Make list of psf candidates to send to the determiner
        # (omitting those marked as reserved)
        psf_determiner_list = [cand for cand, use
                               in zip(psf_selection_result.psfCandidates,
                                      ~psf_cand_cat['final_psf_reserved']) if use]
        flag_key = psf_cand_cat.schema['final_psf_used'].asKey()
        try:
            psf, cell_set = self.psf_determiner.determinePsf(exposure,
                                                             psf_determiner_list,
                                                             self.metadata,
                                                             flagKey=flag_key)
        except Exception as e:
            self.log.warn('Failed to determine psf for visit %d, detector %d: %s',
                          visit, detector, e)
            return None, None, measured_src

        # At this point, we need to transfer the psf used flag from the selection
        # catalog to the measurement catalog.
        matched_selected, matched_measured = esutil.numpy_util.match(
            selected_src['id'],
            measured_src['id']
        )
        measured_used = np.zeros(len(measured_src), dtype=bool)
        measured_used[matched_measured] = selected_src['final_psf_used'][matched_selected]
        measured_src['final_psf_used'] = measured_used

        # Next, we do the measurement on the psf stars ...
        try:
            self.measurement.run(measCat=measured_src, exposure=exposure)
        except Exception as e:
            self.log.warn('Failed to make measurements for visit %d, detector %d: %s',
                          visit, detector, e)
            return psf, None, measured_src

        # And finally the ap corr map.
        try:
            ap_corr_map = self.measure_ap_corr.run(exposure=exposure,
                                                   catalog=measured_src).apCorrMap
        except Exception as e:
            self.log.warn('Failed to compute aperture corrections for visit %d, detector %d: %s',
                          visit, detector, e)
            return psf, None, measured_src

        try:
            self.apply_ap_corr.run(catalog=measured_src, apCorrMap=ap_corr_map)
        except Exception as e:
            self.log.warn('Failed to apply aperture corrections for visit %d, detector %d: %s',
                          visit, detector, e)
            # Assume aperture correction map is bad, so return None.
            return psf, None, measured_src

        return psf, ap_corr_map, measured_src
