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
    finalizedSomethingCat = pipeBase.connectionTypes.Output(
        doc=('Per-visit catalog of measurements for psf/flag/etc.'),
        name='finalized_something_catalog',
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


class FinalizeCharacterizationTask(pipeBase.PipelineTask):
    """Run final characterization on exposures."""
    ConfigClass = FinalizeCharacterizationConfig
    _DefaultName = 'finalize_characterization'

    def __init__(self, initInputs=None, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)

        self.schema_mapper, self.schema = self._make_schema_mapper(initInputs['src_schema'].schema)

        self.makeSubtask('reserve_selection')
        self.makeSubtask('source_selector')
        self.makeSubtask('make_psf_candidates')
        self.makeSubtask('psf_determiner')
        self.makeSubtask('measurement', schema=self.schema)
        self.makeSubtask('measure_ap_corr', schema=self.schema)
        self.makeSubtask('apply_ap_corr', schema=self.schema)

        # Reset alias map to those that are in the input src_schema.
        alias_map = self.schema.getAliasMap()
        alias_map_input = initInputs['src_schema'].schema.getAliasMap()
        alias_map.set('slot_Centroid', alias_map_input.get('slot_Centroid'))
        alias_map.set('slot_Shape', alias_map_input.get('slot_Shape'))
        alias_map.erase('slot_PsfShape')
        self.schema.setAliasMap(alias_map)

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
        isolated_source_table, isolated_reserved = self._concat_isolated_star_cats(
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

        for detector in src_dict:
            src = src_dict[detector].get()
            exposure = calexp_dict[detector].get()

            psf, ap_corr_map = self._compute_psf_and_ap_corr_map(
                exposure,
                src,
                isolated_source_table,
                isolated_reserved
            )

            # And now we package it together...
            psf_record = psf_cat.addNew()
            psf_record['id'] = int(detector)
            psf_record['visit'] = visit
            psf_record.setPsf(psf)

            ap_corr_record = ap_corr_cat.addNew()
            ap_corr_record['id'] = int(detector)
            ap_corr_record['visit'] = visit
            ap_corr_record.setApCorrMap(ap_corr_map)

        return pipeBase.Struct(psf_cat=psf_cat,
                               ap_corr_map_cat=ap_corr_cat)

    def _make_schema_mapper(self, input_schema):
        """Make the schema mapper from the input schema.

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

        shape_fields = input_schema.extract('slot_Shape_*')
        for field, item in shape_fields.items():
            mapper.addMapping(item.key)

        aper_fields = input_schema.extract('base_CircularApertureFlux_*')
        for field, item in aper_fields.items():
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

        alias_map = input_schema.getAliasMap()
        alias_map_output = afwTable.AliasMap()
        alias_map_output.set('slot_Centroid', alias_map.get('slot_Centroid'))
        alias_map_output.set('slot_Shape', alias_map.get('slot_Shape'))
        output_schema = mapper.getOutputSchema()
        output_schema.setAliasMap(alias_map_output)

        return mapper, output_schema

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
        isolated_source_table : `np.ndarray` (N,)
            Table of isolated sources, with indexes to isolated stars.
        isolated_reserved : `np.ndarray` (M,)
            bool array of flags for reserved stars.
        """
        isolated_sources = []
        isolated_reserves = []
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
            # Update obj_index, include any offsets due to merging tracts
            a, b = esutil.numpy_util.match(use_band, table_source['obj_index'])
            table_source['obj_index'][b] = a + merge_cat_counter
            # Cut to observed
            table_cat = table_cat[use_band]
            # Update source index accounting for sources
            table_cat[f'source_cat_index_{band}'] += merge_source_counter

            # Get reserve star flags
            reserved = self.reserve_selection.run(tract, len(table_cat))

            isolated_reserves.append(reserved)
            isolated_sources.append(table_source)

            merge_cat_counter += len(table_cat)
            merge_source_counter += len(table_source)

        isolated_source_table = np.concatenate(isolated_sources)
        isolated_reserved = np.concatenate(isolated_reserves)

        return isolated_source_table, isolated_reserved

    def _compute_psf_and_ap_corr_map(self, exposure, src, isolated_source_table, isolated_reserved):
        """Compute psf model and aperture correction map for a single exposure.

        Parameters
        ----------
        exposure : `lsst.afw.image.ExposureF`
        src : `lsst.afw.table.SourceCatalog`
        isolated_source_table : `np.ndarray`
        isolated_reserved : NO

        Returns
        -------
        psf : `lsst.meas.algorithms.ImagePsf`
            PSF Model
        ap_corr_map : `lsst.afw.image.ApCorrMap`
            Aperture correction map.
        """
        # Apply source selector (s/n, flags, etc.)
        good_src = self.source_selector.selectSources(src)

        src_narrow = afwTable.SourceCatalog(self.schema)
        src_narrow.reserve(good_src.selected.sum())
        src_narrow.extend(src[good_src.selected], mapper=self.schema_mapper)

        matched_src, b = esutil.numpy_util.match(src_narrow['id'],
                                                 isolated_source_table[self.config.id_column])
        matched_isolated_obj = isolated_source_table['obj_index'][b]

        matched_arr = np.zeros(len(src_narrow), dtype=bool)
        matched_arr[matched_src] = True
        src_narrow['final_psf_candidate'] = matched_arr

        reserved_arr = np.zeros(len(src_narrow), dtype=bool)
        reserved_arr[matched_src] = isolated_reserved[matched_isolated_obj]
        src_narrow['final_psf_reserved'] = reserved_arr

        src_narrow = src_narrow[src_narrow['final_psf_candidate']].copy(deep=True)

        # Now we can make the psfs, etc.
        selection_result = self.make_psf_candidates.run(src_narrow, exposure=exposure)

        psf_cand_cat = selection_result.goodStarCat

        # Make list of psf candidates to send to the determiner (omitting those marked as reserved)
        psf_determiner_list = [cand for cand, use
                               in zip(selection_result.psfCandidates,
                                      ~psf_cand_cat['final_psf_reserved']) if use]

        psf, cell_set = self.psf_determiner.determinePsf(exposure,
                                                         psf_determiner_list,
                                                         self.metadata,
                                                         flagKey=self.schema['final_psf_used'].asKey())

        # Next, we do the measurement on the psf stars ...
        self.measurement.run(measCat=src_narrow, exposure=exposure)

        # And finally the ap corr map.
        ap_corr_map = self.measure_ap_corr.run(exposure=exposure,
                                               catalog=src_narrow).apCorrMap

        self.apply_ap_corr.run(catalog=src_narrow, apCorrMap=ap_corr_map)

        return psf, ap_corr_map
