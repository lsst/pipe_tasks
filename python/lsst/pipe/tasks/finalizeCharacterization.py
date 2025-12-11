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

"""Task to run a finalized image characterization, using additional data.
"""

__all__ = [
    'FinalizeCharacterizationConnections',
    'FinalizeCharacterizationConfig',
    'FinalizeCharacterizationTask',
    'FinalizeCharacterizationDetectorConnections',
    'FinalizeCharacterizationDetectorConfig',
    'FinalizeCharacterizationDetectorTask',
    'ConsolidateFinalizeCharacterizationDetectorConnections',
    'ConsolidateFinalizeCharacterizationDetectorConfig',
    'ConsolidateFinalizeCharacterizationDetectorTask',
]

import astropy.table
import astropy.units as u
import numpy as np
import esutil
from smatch.matcher import Matcher


import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
import lsst.afw.table as afwTable
import lsst.meas.algorithms as measAlg
import lsst.meas.extensions.piff.piffPsfDeterminer  # noqa: F401
from lsst.meas.algorithms import MeasureApCorrTask
from lsst.meas.base import SingleFrameMeasurementTask, ApplyApCorrTask
from lsst.meas.algorithms.sourceSelector import sourceSelectorRegistry

from .reserveIsolatedStars import ReserveIsolatedStarsTask
from lsst.obs.base.utils import TableVStack


class FinalizeCharacterizationConnectionsBase(
    pipeBase.PipelineTaskConnections,
    dimensions=('instrument', 'visit',),
    defaultTemplates={},
):
    src_schema = pipeBase.connectionTypes.InitInput(
        doc='Input schema used for src catalogs.',
        name='src_schema',
        storageClass='SourceCatalog',
    )
    isolated_star_cats = pipeBase.connectionTypes.Input(
        doc=('Catalog of isolated stars with average positions, number of associated '
             'sources, and indexes to the isolated_star_sources catalogs.'),
        name='isolated_star_presource_associations',
        storageClass='ArrowAstropy',
        dimensions=('instrument', 'tract', 'skymap'),
        deferLoad=True,
        multiple=True,
    )
    isolated_star_sources = pipeBase.connectionTypes.Input(
        doc=('Catalog of isolated star sources with sourceIds, and indexes to the '
             'isolated_star_cats catalogs.'),
        name='isolated_star_presources',
        storageClass='ArrowAstropy',
        dimensions=('instrument', 'tract', 'skymap'),
        deferLoad=True,
        multiple=True,
    )
    fgcm_standard_star = pipeBase.connectionTypes.Input(
        doc=('Catalog of fgcm for color corrections, and indexes to the '
             'isolated_star_cats catalogs.'),
        name='fgcm_standard_star',
        storageClass='ArrowAstropy',
        dimensions=('instrument', 'tract', 'skymap'),
        deferLoad=True,
        multiple=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if config is None:
            return None
        if not self.config.psf_determiner['piff'].useColor:
            self.inputs.remove("fgcm_standard_star")


class FinalizeCharacterizationConnections(
    FinalizeCharacterizationConnectionsBase,
    dimensions=('instrument', 'visit',),
    defaultTemplates={},
):
    srcs = pipeBase.connectionTypes.Input(
        doc='Source catalogs for the visit',
        name='src',
        storageClass='SourceCatalog',
        dimensions=('instrument', 'visit', 'detector'),
        deferLoad=True,
        multiple=True,
        deferGraphConstraint=True,
    )
    calexps = pipeBase.connectionTypes.Input(
        doc='Calexps for the visit',
        name='calexp',
        storageClass='ExposureF',
        dimensions=('instrument', 'visit', 'detector'),
        deferLoad=True,
        multiple=True,
    )
    finalized_psf_ap_corr_cat = pipeBase.connectionTypes.Output(
        doc=('Per-visit finalized psf models and aperture corrections.  This '
             'catalog uses detector id for the id and are sorted for fast '
             'lookups of a detector.'),
        name='finalized_psf_ap_corr_catalog',
        storageClass='ExposureCatalog',
        dimensions=('instrument', 'visit'),
    )
    finalized_src_table = pipeBase.connectionTypes.Output(
        doc=('Per-visit catalog of measurements for psf/flag/etc.'),
        name='finalized_src_table',
        storageClass='ArrowAstropy',
        dimensions=('instrument', 'visit'),
    )


class FinalizeCharacterizationDetectorConnections(
    FinalizeCharacterizationConnectionsBase,
    dimensions=('instrument', 'visit', 'detector',),
    defaultTemplates={},
):
    src = pipeBase.connectionTypes.Input(
        doc='Source catalog for the visit/detector.',
        name='src',
        storageClass='SourceCatalog',
        dimensions=('instrument', 'visit', 'detector'),
    )
    calexp = pipeBase.connectionTypes.Input(
        doc='Calibrated exposure for the visit/detector.',
        name='calexp',
        storageClass='ExposureF',
        dimensions=('instrument', 'visit', 'detector'),
    )
    finalized_psf_ap_corr_detector_cat = pipeBase.connectionTypes.Output(
        doc=('Per-visit/per-detector finalized psf models and aperture corrections.  This '
             'catalog uses detector id for the id.'),
        name='finalized_psf_ap_corr_detector_catalog',
        storageClass='ExposureCatalog',
        dimensions=('instrument', 'visit', 'detector'),
    )
    finalized_src_detector_table = pipeBase.connectionTypes.Output(
        doc=('Per-visit/per-detector catalog of measurements for psf/flag/etc.'),
        name='finalized_src_detector_table',
        storageClass='ArrowAstropy',
        dimensions=('instrument', 'visit', 'detector'),
    )


class FinalizeCharacterizationConfigBase(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=FinalizeCharacterizationConnectionsBase,
):
    """Configuration for FinalizeCharacterizationBaseTask."""
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
        default='piff'
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
        super().setDefaults()

        source_selector = self.source_selector['science']
        source_selector.setDefaults()

        # We use the source selector only to select out flagged objects
        # and signal-to-noise.  Isolated, unresolved sources are handled
        # by the isolated star catalog.

        source_selector.doFlags = True
        source_selector.doSignalToNoise = True
        source_selector.doFluxLimit = False
        source_selector.doUnresolved = False
        source_selector.doIsolated = False

        source_selector.signalToNoise.minimum = 50.0
        source_selector.signalToNoise.maximum = 1000.0

        source_selector.signalToNoise.fluxField = 'base_GaussianFlux_instFlux'
        source_selector.signalToNoise.errField = 'base_GaussianFlux_instFluxErr'

        source_selector.flags.bad = ['base_PixelFlags_flag_edge',
                                     'base_PixelFlags_flag_nodata',
                                     'base_PixelFlags_flag_interpolatedCenter',
                                     'base_PixelFlags_flag_saturatedCenter',
                                     'base_PixelFlags_flag_crCenter',
                                     'base_PixelFlags_flag_bad',
                                     'base_PixelFlags_flag_interpolated',
                                     'base_PixelFlags_flag_saturated',
                                     'slot_Centroid_flag',
                                     'base_GaussianFlux_flag']

        # Configure aperture correction to select only high s/n sources (that
        # were used in the psf modeling) to avoid background problems when
        # computing the aperture correction map.
        self.measure_ap_corr.sourceSelector = 'science'

        ap_selector = self.measure_ap_corr.sourceSelector['science']
        # We do not need to filter flags or unresolved because we have used
        # the filtered isolated stars as an input
        ap_selector.doFlags = False
        ap_selector.doUnresolved = False

        import lsst.meas.modelfit  # noqa: F401
        import lsst.meas.extensions.photometryKron  # noqa: F401
        import lsst.meas.extensions.convolved  # noqa: F401
        import lsst.meas.extensions.gaap  # noqa: F401
        import lsst.meas.extensions.shapeHSM  # noqa: F401

        # Set up measurement defaults
        self.measurement.plugins.names = [
            'base_FPPosition',
            'base_PsfFlux',
            'base_GaussianFlux',
            'modelfit_DoubleShapeletPsfApprox',
            'modelfit_CModel',
            'ext_photometryKron_KronFlux',
            'ext_convolved_ConvolvedFlux',
            'ext_gaap_GaapFlux',
            'ext_shapeHSM_HsmShapeRegauss',
            'ext_shapeHSM_HsmSourceMoments',
            'ext_shapeHSM_HsmPsfMoments',
            'ext_shapeHSM_HsmSourceMomentsRound',
            'ext_shapeHSM_HigherOrderMomentsSource',
            'ext_shapeHSM_HigherOrderMomentsPSF',
        ]
        self.measurement.slots.modelFlux = 'modelfit_CModel'
        self.measurement.plugins['ext_convolved_ConvolvedFlux'].seeing.append(8.0)
        self.measurement.plugins['ext_gaap_GaapFlux'].sigmas = [
            0.5,
            0.7,
            1.0,
            1.5,
            2.5,
            3.0
        ]
        self.measurement.plugins['ext_gaap_GaapFlux'].doPsfPhotometry = True
        self.measurement.slots.shape = 'ext_shapeHSM_HsmSourceMoments'
        self.measurement.slots.psfShape = 'ext_shapeHSM_HsmPsfMoments'
        self.measurement.plugins['ext_shapeHSM_HsmShapeRegauss'].deblendNChild = ""

        # TODO: Remove in DM-44658, streak masking to happen only in ip_diffim
        # Keep track of which footprints contain streaks
        self.measurement.plugins['base_PixelFlags'].masksFpAnywhere = ['STREAK']
        self.measurement.plugins['base_PixelFlags'].masksFpCenter = ['STREAK']

        # Turn off slot setting for measurement for centroid and shape
        # (for which we use the input src catalog measurements)
        self.measurement.slots.centroid = None
        self.measurement.slots.apFlux = None
        self.measurement.slots.calibFlux = None

        names = self.measurement.plugins['ext_convolved_ConvolvedFlux'].getAllResultNames()
        self.measure_ap_corr.allowFailure += names
        names = self.measurement.plugins["ext_gaap_GaapFlux"].getAllGaapResultNames()
        self.measure_ap_corr.allowFailure += names


class FinalizeCharacterizationConfig(
    FinalizeCharacterizationConfigBase,
    pipelineConnections=FinalizeCharacterizationConnections,
):
    pass


class FinalizeCharacterizationDetectorConfig(
    FinalizeCharacterizationConfigBase,
    pipelineConnections=FinalizeCharacterizationDetectorConnections,
):
    pass


class FinalizeCharacterizationTaskBase(pipeBase.PipelineTask):
    """Run final characterization on exposures."""
    ConfigClass = FinalizeCharacterizationConfigBase
    _DefaultName = 'finalize_characterization_base'

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
        self.isPsfDeterminerPiff = False
        if isinstance(self.psf_determiner, lsst.meas.extensions.piff.piffPsfDeterminer.PiffPsfDeterminerTask):
            self.isPsfDeterminerPiff = True

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

        # The aperture fields may be used by the psf determiner.
        aper_fields = input_schema.extract('base_CircularApertureFlux_*')
        for field, item in aper_fields.items():
            mapper.addMapping(item.key)

        # The following two may be redundant, but then the mapping is a no-op.
        # Note that the slot_CalibFlux mapping will copy over any
        # normalized compensated fluxes that are used for calibration.
        apflux_fields = input_schema.extract('slot_ApFlux_*')
        for field, item in apflux_fields.items():
            mapper.addMapping(item.key)

        calibflux_fields = input_schema.extract('slot_CalibFlux_*')
        for field, item in calibflux_fields.items():
            mapper.addMapping(item.key)

        mapper.addMapping(
            input_schema[self.config.source_selector.active.signalToNoise.fluxField].asKey(),
            'calib_psf_selection_flux')
        mapper.addMapping(
            input_schema[self.config.source_selector.active.signalToNoise.errField].asKey(),
            'calib_psf_selection_flux_err')

        output_schema = mapper.getOutputSchema()

        output_schema.addField(
            'calib_psf_candidate',
            type='Flag',
            doc=('set if the source was a candidate for PSF determination, '
                 'as determined from FinalizeCharacterizationTask.'),
        )
        output_schema.addField(
            'calib_psf_reserved',
            type='Flag',
            doc=('set if source was reserved from PSF determination by '
                 'FinalizeCharacterizationTask.'),
        )
        output_schema.addField(
            'calib_psf_used',
            type='Flag',
            doc=('set if source was used in the PSF determination by '
                 'FinalizeCharacterizationTask.'),
        )
        output_schema.addField(
            'visit',
            type=np.int64,
            doc='Visit number for the sources.',
        )
        output_schema.addField(
            'detector',
            type=np.int32,
            doc='Detector number for the sources.',
        )
        output_schema.addField(
            'psf_color_value',
            type=np.float32,
            doc="Color used in PSF fit."
        )
        output_schema.addField(
            'psf_color_type',
            type=str,
            size=10,
            doc="Color used in PSF fit."
        )
        output_schema.addField(
            'psf_max_value',
            type=np.float32,
            doc="Maximum value in the star image used to train PSF.",
            doReplace=True,
        )

        alias_map = input_schema.getAliasMap()
        alias_map_output = afwTable.AliasMap()
        alias_map_output.set('slot_Centroid', alias_map.get('slot_Centroid'))
        alias_map_output.set('slot_ApFlux', alias_map.get('slot_ApFlux'))
        alias_map_output.set('slot_CalibFlux', alias_map.get('slot_CalibFlux'))

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
        mapper.addMinimalSchema(input_schema)

        selection_schema = mapper.getOutputSchema()

        selection_schema.setAliasMap(input_schema.getAliasMap())

        selection_schema.addField(
            'psf_color_value',
            type=np.float32,
            doc="Color used in PSF fit."
        )
        selection_schema.addField(
            'psf_color_type',
            type=str,
            size=10,
            doc="Color used in PSF fit."
        )
        selection_schema.addField(
            'psf_max_value',
            type=np.float32,
            doc="Maximum value in the star image used to train PSF.",
            doReplace=True,
        )

        return mapper, selection_schema

    def concat_isolated_star_cats(self, band, isolated_star_cat_dict, isolated_star_source_dict):
        """
        Concatenate isolated star catalogs and make reserve selection.

        Parameters
        ----------
        band : `str`
            Band name.  Used to select reserved stars.
        isolated_star_cat_dict : `dict`
            Per-tract dict of isolated star catalog handles.
        isolated_star_source_dict : `dict`
            Per-tract dict of isolated star source catalog handles.

        Returns
        -------
        isolated_table : `astropy.table.Table` (N,)
            Table of isolated stars, with indexes to isolated sources.
            Returns None if there are no usable isolated catalogs.
        isolated_source_table : `astropy.table.Table` (M,)
            Table of isolated sources, with indexes to isolated stars.
            Returns None if there are no usable isolated catalogs.
        """
        isolated_tables = []
        isolated_sources = []
        merge_cat_counter = 0
        merge_source_counter = 0

        for tract in isolated_star_cat_dict:
            astropy_cat = isolated_star_cat_dict[tract].get()
            astropy_source = isolated_star_source_dict[tract].get(
                parameters={'columns': [self.config.id_column, 'obj_index']}
            )

            # Cut isolated star table to those observed in this band, and adjust indexes
            (use_band,) = (astropy_cat[f'nsource_{band}'] > 0).nonzero()

            if len(use_band) == 0:
                # There are no sources in this band in this tract.
                self.log.info("No sources found in %s band in tract %d.", band, tract)
                continue

            # With the following matching:
            #   table_source[b] <-> table_cat[use_band[a]]
            a, b = esutil.numpy_util.match(use_band, np.asarray(astropy_source['obj_index']))

            # Update indexes and cut to band-selected stars/sources
            astropy_source['obj_index'][b] = a
            _, index_new = np.unique(a, return_index=True)
            astropy_cat[f'source_cat_index_{band}'][use_band] = index_new

            # After the following cuts, the catalogs have the following properties:
            # - table_cat only contains isolated stars that have at least one source
            #   in ``band``.
            # - table_source only contains ``band`` sources.
            # - The slice table_cat["source_cat_index_{band}"]: table_cat["source_cat_index_{band}"]
            #                                                   + table_cat["nsource_{band}]
            #   applied to table_source will give all the sources associated with the star.
            # - For each source, table_source["obj_index"] points to the index of the associated
            #   isolated star.
            astropy_source = astropy_source[b]
            astropy_cat = astropy_cat[use_band]

            # Add reserved flag column to tables
            astropy_cat['reserved'] = False
            astropy_source['reserved'] = False

            # Get reserve star flags
            astropy_cat['reserved'][:] = self.reserve_selection.run(
                len(astropy_cat),
                extra=f'{band}_{tract}',
            )
            astropy_source['reserved'][:] = astropy_cat['reserved'][astropy_source['obj_index']]

            # Offset indexes to account for tract merging
            astropy_cat[f'source_cat_index_{band}'] += merge_source_counter
            astropy_source['obj_index'] += merge_cat_counter

            isolated_tables.append(astropy_cat)
            isolated_sources.append(astropy_source)

            merge_cat_counter += len(astropy_cat)
            merge_source_counter += len(astropy_source)

        if len(isolated_tables) > 0:
            isolated_table = astropy.table.vstack(isolated_tables, metadata_conflicts='silent')
            isolated_source_table = astropy.table.vstack(isolated_sources, metadata_conflicts='silent')
        else:
            isolated_table = None
            isolated_source_table = None

        return isolated_table, isolated_source_table

    def add_src_colors(self, srcCat, fgcmCat, band):

        if self.isPsfDeterminerPiff and fgcmCat is not None:

            raSrc = (srcCat['coord_ra'] * u.radian).to(u.degree).value
            decSrc = (srcCat['coord_dec'] * u.radian).to(u.degree).value

            with Matcher(raSrc, decSrc) as matcher:
                idx, idxSrcCat, idxColorCat, d = matcher.query_radius(
                    fgcmCat["ra"],
                    fgcmCat["dec"],
                    1. / 3600.0,
                    return_indices=True,
                )

            magStr1 = self.psf_determiner.config.color[band][0]
            magStr2 = self.psf_determiner.config.color[band][2]
            colors = fgcmCat[f'mag_{magStr1}'] - fgcmCat[f'mag_{magStr2}']

            for idSrc, idColor in zip(idxSrcCat, idxColorCat):
                srcCat[idSrc]['psf_color_value'] = colors[idColor]
                srcCat[idSrc]['psf_color_type'] = f"{magStr1}-{magStr2}"

    def compute_psf_and_ap_corr_map(self, visit, detector, exposure, src,
                                    isolated_source_table, fgcm_standard_star_cat):
        """Compute psf model and aperture correction map for a single exposure.

        Parameters
        ----------
        visit : `int`
            Visit number (for logging).
        detector : `int`
            Detector number (for logging).
        exposure : `lsst.afw.image.ExposureF`
        src : `lsst.afw.table.SourceCatalog`
        isolated_source_table : `np.ndarray` or `astropy.table.Table`
        fgcm_standard_star_cat : `np.ndarray`

        Returns
        -------
        psf : `lsst.meas.algorithms.ImagePsf`
            PSF Model
        ap_corr_map : `lsst.afw.image.ApCorrMap`
            Aperture correction map.
        measured_src : `lsst.afw.table.SourceCatalog`
            Updated source catalog with measurements, flags and aperture corrections.
        """
        # Extract footprints from the input src catalog for noise replacement.
        footprints = SingleFrameMeasurementTask.getFootprintsFromCatalog(src)

        # Apply source selector (s/n, flags, etc.)
        good_src = self.source_selector.selectSources(src)
        if sum(good_src.selected) == 0:
            self.log.warning('No good sources remain after cuts for visit %d, detector %d',
                             visit, detector)
            return None, None, None

        # Cut down input src to the selected sources
        # We use a separate schema/mapper here than for the output/measurement catalog because of
        # clashes between fields that were previously run and those that need to be rerun with
        # the new psf model.  This may be slightly inefficient but keeps input
        # and output values cleanly separated.
        selection_mapper, selection_schema = self._make_selection_schema_mapper(src.schema)

        selected_src = afwTable.SourceCatalog(selection_schema)
        selected_src.reserve(good_src.selected.sum())
        selected_src.extend(src[good_src.selected], mapper=selection_mapper)

        # The calib flags have been copied from the input table,
        # and we reset them here just to ensure they aren't propagated.
        selected_src['calib_psf_candidate'] = np.zeros(len(selected_src), dtype=bool)
        selected_src['calib_psf_used'] = np.zeros(len(selected_src), dtype=bool)
        selected_src['calib_psf_reserved'] = np.zeros(len(selected_src), dtype=bool)

        # Find the isolated sources and set flags
        matched_src, matched_iso = esutil.numpy_util.match(
            selected_src['id'],
            np.asarray(isolated_source_table[self.config.id_column]),
        )
        if len(matched_src) == 0:
            self.log.warning(
                "No candidates from matched isolate stars for visit=%s, detector=%s "
                "(this is probably the result of an earlier astrometry failure).",
                visit, detector,
            )
            return None, None, None

        matched_arr = np.zeros(len(selected_src), dtype=bool)
        matched_arr[matched_src] = True
        selected_src['calib_psf_candidate'] = matched_arr

        reserved_arr = np.zeros(len(selected_src), dtype=bool)
        reserved_arr[matched_src] = np.asarray(isolated_source_table['reserved'][matched_iso])
        selected_src['calib_psf_reserved'] = reserved_arr

        selected_src = selected_src[selected_src['calib_psf_candidate']].copy(deep=True)

        # Make the measured source catalog as well, based on the selected catalog.
        measured_src = afwTable.SourceCatalog(self.schema)
        measured_src.reserve(len(selected_src))
        measured_src.extend(selected_src, mapper=self.schema_mapper)

        # We need to copy over the calib_psf flags because they were not in the mapper
        measured_src['calib_psf_candidate'] = selected_src['calib_psf_candidate']
        measured_src['calib_psf_reserved'] = selected_src['calib_psf_reserved']
        if exposure.filter.hasBandLabel():
            band = exposure.filter.bandLabel
        else:
            band = None
        self.add_src_colors(selected_src, fgcm_standard_star_cat, band)
        self.add_src_colors(measured_src, fgcm_standard_star_cat, band)

        # Select the psf candidates from the selection catalog
        try:
            psf_selection_result = self.make_psf_candidates.run(selected_src, exposure=exposure)
            _ = self.make_psf_candidates.run(measured_src, exposure=exposure)
        except Exception as e:
            self.log.exception('Failed to make PSF candidates for visit %d, detector %d: %s',
                               visit, detector, e)
            return None, None, measured_src

        psf_cand_cat = psf_selection_result.goodStarCat

        # Make list of psf candidates to send to the determiner
        # (omitting those marked as reserved)
        psf_determiner_list = [cand for cand, use
                               in zip(psf_selection_result.psfCandidates,
                                      ~psf_cand_cat['calib_psf_reserved']) if use]
        flag_key = psf_cand_cat.schema['calib_psf_used'].asKey()

        try:
            psf, cell_set = self.psf_determiner.determinePsf(exposure,
                                                             psf_determiner_list,
                                                             self.metadata,
                                                             flagKey=flag_key)
        except Exception as e:
            self.log.exception('Failed to determine PSF for visit %d, detector %d: %s',
                               visit, detector, e)
            return None, None, measured_src
        # Verify that the PSF is usable by downstream tasks
        sigma = psf.computeShape(psf.getAveragePosition(), psf.getAverageColor()).getDeterminantRadius()
        if np.isnan(sigma):
            self.log.warning('Failed to determine psf for visit %d, detector %d: '
                             'Computed final PSF size is NAN.',
                             visit, detector)
            return None, None, measured_src

        # Set the psf in the exposure for measurement/aperture corrections.
        exposure.setPsf(psf)

        # At this point, we need to transfer the psf used flag from the selection
        # catalog to the measurement catalog.
        matched_selected, matched_measured = esutil.numpy_util.match(
            selected_src['id'],
            measured_src['id']
        )
        measured_used = np.zeros(len(measured_src), dtype=bool)
        measured_used[matched_measured] = selected_src['calib_psf_used'][matched_selected]
        measured_src['calib_psf_used'] = measured_used

        # Next, we do the measurement on all the psf candidate, used, and reserved stars.
        # We use the full footprint list from the input src catalog for noise replacement.
        try:
            self.measurement.run(measCat=measured_src, exposure=exposure, footprints=footprints)
        except Exception as e:
            self.log.warning('Failed to make measurements for visit %d, detector %d: %s',
                             visit, detector, e)
            return psf, None, measured_src

        # And finally the ap corr map.
        try:
            ap_corr_map = self.measure_ap_corr.run(exposure=exposure,
                                                   catalog=measured_src).apCorrMap
        except Exception as e:
            self.log.warning('Failed to compute aperture corrections for visit %d, detector %d: %s',
                             visit, detector, e)
            return psf, None, measured_src

        # Need to merge the original normalization aperture correction map.
        ap_corr_map_input = exposure.apCorrMap
        for key in ap_corr_map_input:
            if key not in ap_corr_map:
                ap_corr_map[key] = ap_corr_map_input[key]

        self.apply_ap_corr.run(catalog=measured_src, apCorrMap=ap_corr_map)

        return psf, ap_corr_map, measured_src


class FinalizeCharacterizationTask(FinalizeCharacterizationTaskBase):
    """Run final characterization on full visits."""
    ConfigClass = FinalizeCharacterizationConfig
    _DefaultName = 'finalize_characterization'

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        input_handle_dict = butlerQC.get(inputRefs)

        band = butlerQC.quantum.dataId['band']
        visit = butlerQC.quantum.dataId['visit']

        src_dict_temp = {handle.dataId['detector']: handle
                         for handle in input_handle_dict['srcs']}
        calexp_dict_temp = {handle.dataId['detector']: handle
                            for handle in input_handle_dict['calexps']}
        isolated_star_cat_dict_temp = {handle.dataId['tract']: handle
                                       for handle in input_handle_dict['isolated_star_cats']}
        isolated_star_source_dict_temp = {handle.dataId['tract']: handle
                                          for handle in input_handle_dict['isolated_star_sources']}

        src_dict = {detector: src_dict_temp[detector] for
                    detector in sorted(src_dict_temp.keys())}
        calexp_dict = {detector: calexp_dict_temp[detector] for
                       detector in sorted(calexp_dict_temp.keys())}
        isolated_star_cat_dict = {tract: isolated_star_cat_dict_temp[tract] for
                                  tract in sorted(isolated_star_cat_dict_temp.keys())}
        isolated_star_source_dict = {tract: isolated_star_source_dict_temp[tract] for
                                     tract in sorted(isolated_star_source_dict_temp.keys())}

        if self.config.psf_determiner['piff'].useColor:
            fgcm_standard_star_dict_temp = {handle.dataId['tract']: handle
                                            for handle in input_handle_dict['fgcm_standard_star']}
            fgcm_standard_star_dict = {tract: fgcm_standard_star_dict_temp[tract] for
                                       tract in sorted(fgcm_standard_star_dict_temp.keys())}
        else:
            fgcm_standard_star_dict = None

        struct = self.run(
            visit,
            band,
            isolated_star_cat_dict,
            isolated_star_source_dict,
            src_dict,
            calexp_dict,
            fgcm_standard_star_dict=fgcm_standard_star_dict,
        )

        butlerQC.put(struct.psf_ap_corr_cat, outputRefs.finalized_psf_ap_corr_cat)
        butlerQC.put(struct.output_table, outputRefs.finalized_src_table)

    def run(self, visit, band, isolated_star_cat_dict, isolated_star_source_dict,
            src_dict, calexp_dict, fgcm_standard_star_dict=None):
        """
        Run the FinalizeCharacterizationTask.

        Parameters
        ----------
        visit : `int`
            Visit number.  Used in the output catalogs.
        band : `str`
            Band name.  Used to select reserved stars.
        isolated_star_cat_dict : `dict`
            Per-tract dict of isolated star catalog handles.
        isolated_star_source_dict : `dict`
            Per-tract dict of isolated star source catalog handles.
        src_dict : `dict`
            Per-detector dict of src catalog handles.
        calexp_dict : `dict`
            Per-detector dict of calibrated exposure handles.
        fgcm_standard_star_dict : `dict`
            Per tract dict of fgcm isolated stars.

        Returns
        -------
        struct : `lsst.pipe.base.struct`
            Struct with outputs for persistence.

        Raises
        ------
        NoWorkFound
            Raised if the selector returns no good sources.
        """
        # Check if we have the same inputs for each of the
        # src_dict and calexp_dict.
        src_detectors = set(src_dict.keys())
        calexp_detectors = set(calexp_dict.keys())

        if src_detectors != calexp_detectors:
            detector_keys = sorted(src_detectors.intersection(calexp_detectors))
            self.log.warning(
                "Input src and calexp have mismatched detectors; "
                "running intersection of %d detectors.",
                len(detector_keys),
            )
        else:
            detector_keys = sorted(src_detectors)

        # We do not need the isolated star table in this task.
        # However, it is used in tests to confirm consistency of indexes.
        _, isolated_source_table = self.concat_isolated_star_cats(
            band,
            isolated_star_cat_dict,
            isolated_star_source_dict
        )

        if isolated_source_table is None:
            raise pipeBase.NoWorkFound(f'No good isolated sources found for any detectors in visit {visit}')

        exposure_cat_schema = afwTable.ExposureTable.makeMinimalSchema()
        exposure_cat_schema.addField('visit', type='L', doc='Visit number')

        metadata = dafBase.PropertyList()
        metadata.add("COMMENT", "Catalog id is detector id, sorted.")
        metadata.add("COMMENT", "Only detectors with data have entries.")

        psf_ap_corr_cat = afwTable.ExposureCatalog(exposure_cat_schema)
        psf_ap_corr_cat.setMetadata(metadata)

        measured_src_tables = []
        measured_src_table = None

        if fgcm_standard_star_dict is not None:

            fgcm_standard_star_cat = []

            for tract in fgcm_standard_star_dict:
                astropy_fgcm = fgcm_standard_star_dict[tract].get()
                table_fgcm = np.asarray(astropy_fgcm)
                fgcm_standard_star_cat.append(table_fgcm)

            fgcm_standard_star_cat = np.concatenate(fgcm_standard_star_cat)
        else:
            fgcm_standard_star_cat = None

        self.log.info("Running finalizeCharacterization on %d detectors.", len(detector_keys))
        for detector in detector_keys:
            self.log.info("Starting finalizeCharacterization on detector ID %d.", detector)
            src = src_dict[detector].get()
            exposure = calexp_dict[detector].get()

            psf, ap_corr_map, measured_src = self.compute_psf_and_ap_corr_map(
                visit,
                detector,
                exposure,
                src,
                isolated_source_table,
                fgcm_standard_star_cat,
            )

            # And now we package it together...
            if measured_src is not None:
                record = psf_ap_corr_cat.addNew()
                record['id'] = int(detector)
                record['visit'] = visit
                if psf is not None:
                    record.setPsf(psf)
                if ap_corr_map is not None:
                    record.setApCorrMap(ap_corr_map)

                measured_src['visit'][:] = visit
                measured_src['detector'][:] = detector

                measured_src_tables.append(measured_src.asAstropy())

        if len(measured_src_tables) > 0:
            measured_src_table = astropy.table.vstack(measured_src_tables, join_type='exact')

        if measured_src_table is None:
            raise pipeBase.NoWorkFound(f'No good sources found for any detectors in visit {visit}')

        return pipeBase.Struct(
            psf_ap_corr_cat=psf_ap_corr_cat,
            output_table=measured_src_table,
        )


class FinalizeCharacterizationDetectorTask(FinalizeCharacterizationTaskBase):
    """Run final characterization per detector."""
    ConfigClass = FinalizeCharacterizationDetectorConfig
    _DefaultName = 'finalize_characterization_detector'

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        input_handle_dict = butlerQC.get(inputRefs)

        band = butlerQC.quantum.dataId['band']
        visit = butlerQC.quantum.dataId['visit']
        detector = butlerQC.quantum.dataId['detector']

        isolated_star_cat_dict_temp = {handle.dataId['tract']: handle
                                       for handle in input_handle_dict['isolated_star_cats']}
        isolated_star_source_dict_temp = {handle.dataId['tract']: handle
                                          for handle in input_handle_dict['isolated_star_sources']}

        isolated_star_cat_dict = {tract: isolated_star_cat_dict_temp[tract] for
                                  tract in sorted(isolated_star_cat_dict_temp.keys())}
        isolated_star_source_dict = {tract: isolated_star_source_dict_temp[tract] for
                                     tract in sorted(isolated_star_source_dict_temp.keys())}

        if self.config.psf_determiner['piff'].useColor:
            fgcm_standard_star_dict_temp = {handle.dataId['tract']: handle
                                            for handle in input_handle_dict['fgcm_standard_star']}
            fgcm_standard_star_dict = {tract: fgcm_standard_star_dict_temp[tract] for
                                       tract in sorted(fgcm_standard_star_dict_temp.keys())}
        else:
            fgcm_standard_star_dict = None

        struct = self.run(
            visit,
            band,
            detector,
            isolated_star_cat_dict,
            isolated_star_source_dict,
            input_handle_dict['src'],
            input_handle_dict['calexp'],
            fgcm_standard_star_dict=fgcm_standard_star_dict,
        )

        butlerQC.put(
            struct.psf_ap_corr_cat,
            outputRefs.finalized_psf_ap_corr_detector_cat,
        )
        butlerQC.put(
            struct.output_table,
            outputRefs.finalized_src_detector_table,
        )

    def run(self, visit, band, detector, isolated_star_cat_dict, isolated_star_source_dict,
            src, exposure, fgcm_standard_star_dict=None):
        """
        Run the FinalizeCharacterizationDetectorTask.

        Parameters
        ----------
        visit : `int`
            Visit number.  Used in the output catalogs.
        band : `str`
            Band name.  Used to select reserved stars.
        detector : `int`
            Detector number.
        isolated_star_cat_dict : `dict`
            Per-tract dict of isolated star catalog handles.
        isolated_star_source_dict : `dict`
            Per-tract dict of isolated star source catalog handles.
        src : `lsst.afw.table.SourceCatalog`
            Src catalog.
        exposure : `lsst.afw.image.Exposure`
            Calexp exposure.
        fgcm_standard_star_dict : `dict`
            Per tract dict of fgcm isolated stars.

        Returns
        -------
        struct : `lsst.pipe.base.struct`
            Struct with outputs for persistence.

        Raises
        ------
        NoWorkFound
            Raised if the selector returns no good sources.
        """
        # We do not need the isolated star table in this task.
        # However, it is used in tests to confirm consistency of indexes.
        _, isolated_source_table = self.concat_isolated_star_cats(
            band,
            isolated_star_cat_dict,
            isolated_star_source_dict,
        )

        if isolated_source_table is None:
            raise pipeBase.NoWorkFound(f'No good isolated sources found for any detectors in visit {visit}')

        exposure_cat_schema = afwTable.ExposureTable.makeMinimalSchema()
        exposure_cat_schema.addField('visit', type='L', doc='Visit number')

        metadata = dafBase.PropertyList()
        metadata.add("COMMENT", "Catalog id is detector id, sorted.")
        metadata.add("COMMENT", "Only one detector with data has an entry.")

        psf_ap_corr_cat = afwTable.ExposureCatalog(exposure_cat_schema)
        psf_ap_corr_cat.setMetadata(metadata)

        self.log.info("Starting finalizeCharacterization on detector ID %d.", detector)

        if fgcm_standard_star_dict is not None:
            fgcm_standard_star_cat = []

            for tract in fgcm_standard_star_dict:
                astropy_fgcm = fgcm_standard_star_dict[tract].get()
                table_fgcm = np.asarray(astropy_fgcm)
                fgcm_standard_star_cat.append(table_fgcm)

            fgcm_standard_star_cat = np.concatenate(fgcm_standard_star_cat)
        else:
            fgcm_standard_star_cat = None

        psf, ap_corr_map, measured_src = self.compute_psf_and_ap_corr_map(
            visit,
            detector,
            exposure,
            src,
            isolated_source_table,
            fgcm_standard_star_cat,
        )

        # And now we package it together...
        measured_src_table = None
        if measured_src is not None:
            record = psf_ap_corr_cat.addNew()
            record['id'] = int(detector)
            record['visit'] = visit
            if psf is not None:
                record.setPsf(psf)
            if ap_corr_map is not None:
                record.setApCorrMap(ap_corr_map)

            measured_src['visit'][:] = visit
            measured_src['detector'][:] = detector

            measured_src_table = measured_src.asAstropy()

        if measured_src_table is None:
            raise pipeBase.NoWorkFound(f'No good sources found for visit {visit} / detector {detector}')

        return pipeBase.Struct(
            psf_ap_corr_cat=psf_ap_corr_cat,
            output_table=measured_src_table,
        )


class ConsolidateFinalizeCharacterizationDetectorConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=('instrument', 'visit',),
):
    finalized_psf_ap_corr_detector_cats = pipeBase.connectionTypes.Input(
        doc='Per-visit/per-detector finalized psf models and aperture corrections.',
        name='finalized_psf_ap_corr_detector_catalog',
        storageClass='ExposureCatalog',
        dimensions=('instrument', 'visit', 'detector'),
        multiple=True,
        deferLoad=True,
    )
    finalized_src_detector_tables = pipeBase.connectionTypes.Input(
        doc=('Per-visit/per-detector catalog of measurements for psf/flag/etc.'),
        name='finalized_src_detector_table',
        storageClass='ArrowAstropy',
        dimensions=('instrument', 'visit', 'detector'),
        multiple=True,
        deferLoad=True,
    )
    finalized_psf_ap_corr_cat = pipeBase.connectionTypes.Output(
        doc=('Per-visit finalized psf models and aperture corrections.  This '
             'catalog uses detector id for the id and are sorted for fast '
             'lookups of a detector.'),
        name='finalized_psf_ap_corr_catalog',
        storageClass='ExposureCatalog',
        dimensions=('instrument', 'visit'),
    )
    finalized_src_table = pipeBase.connectionTypes.Output(
        doc=('Per-visit catalog of measurements for psf/flag/etc.'),
        name='finalized_src_table',
        storageClass='ArrowAstropy',
        dimensions=('instrument', 'visit'),
    )


class ConsolidateFinalizeCharacterizationDetectorConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=ConsolidateFinalizeCharacterizationDetectorConnections,
):
    pass


class ConsolidateFinalizeCharacterizationDetectorTask(pipeBase.PipelineTask):
    """Consolidate per-detector finalize characterization catalogs."""
    ConfigClass = ConsolidateFinalizeCharacterizationDetectorConfig
    _DefaultName = 'consolidate_finalize_characterization_detector'

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        input_handle_dict = butlerQC.get(inputRefs)

        psf_ap_corr_detector_dict_temp = {
            handle.dataId['detector']: handle
            for handle in input_handle_dict['finalized_psf_ap_corr_detector_cats']
        }
        src_detector_table_dict_temp = {
            handle.dataId['detector']: handle
            for handle in input_handle_dict['finalized_src_detector_tables']
        }

        psf_ap_corr_detector_dict = {
            detector: psf_ap_corr_detector_dict_temp[detector]
            for detector in sorted(psf_ap_corr_detector_dict_temp.keys())
        }
        src_detector_table_dict = {
            detector: src_detector_table_dict_temp[detector]
            for detector in sorted(src_detector_table_dict_temp.keys())
        }

        result = self.run(
            psf_ap_corr_detector_dict=psf_ap_corr_detector_dict,
            src_detector_table_dict=src_detector_table_dict,
        )

        butlerQC.put(result.psf_ap_corr_cat, outputRefs.finalized_psf_ap_corr_cat)
        butlerQC.put(result.output_table, outputRefs.finalized_src_table)

    def run(self, *, psf_ap_corr_detector_dict, src_detector_table_dict):
        """
        Run the ConsolidateFinalizeCharacterizationDetectorTask.

        Parameters
        ----------
        psf_ap_corr_detector_dict : `dict` [`int`, `lsst.daf.butler.DeferredDatasetHandle`]
            Dictionary of input exposure catalogs, keyed by detector id.
        src_detector_table_dict : `dict` [`int`, `lsst.daf.butler.DeferredDatasetHandle`]
            Dictionary of input source tables, keyed by detector id.

        Returns
        -------
        result : `lsst.pipe.base.struct`
            Struct with the following outputs:
            ``psf_ap_corr_cat``: Consolidated exposure catalog
            ``src_table``: Consolidated source table.
        """
        if not len(psf_ap_corr_detector_dict):
            raise pipeBase.NoWorkFound("No inputs found.")

        if not np.all(
            np.asarray(psf_ap_corr_detector_dict.keys())
            == np.asarray(src_detector_table_dict.keys())
        ):
            raise ValueError(
                "Input psf_ap_corr_detector_dict and src_detector_table_dict must have the same keys",
            )

        psf_ap_corr_cat = None
        for detector_id, handle in psf_ap_corr_detector_dict.items():
            if psf_ap_corr_cat is None:
                psf_ap_corr_cat = handle.get()
            else:
                psf_ap_corr_cat.append(handle.get().find(detector_id))

        # Make sure it is a contiguous catalog.
        psf_ap_corr_cat = psf_ap_corr_cat.copy(deep=True)

        src_table = TableVStack.vstack_handles(src_detector_table_dict.values())

        return pipeBase.Struct(
            psf_ap_corr_cat=psf_ap_corr_cat,
            output_table=src_table,
        )
