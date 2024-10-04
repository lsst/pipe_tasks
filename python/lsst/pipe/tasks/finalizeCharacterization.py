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

__all__ = ['FinalizeCharacterizationConnections',
           'FinalizeCharacterizationConfig',
           'FinalizeCharacterizationTask']

import logging

import numpy as np
import esutil
import pandas as pd

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


_LOG = logging.getLogger(__name__)


class FinalizeCharacterizationConnections(pipeBase.PipelineTaskConnections,
                                          dimensions=('instrument', 'visit',),
                                          defaultTemplates={}):
    src_schema = pipeBase.connectionTypes.InitInput(
        doc='Input schema used for src catalogs.',
        name='initial_stars_schema',
        storageClass='SourceCatalog',
    )
    srcs = pipeBase.connectionTypes.Input(
        doc='Source catalogs for the visit',
        name='initial_stars_footprints_detector',
        storageClass='SourceCatalog',
        dimensions=('instrument', 'visit', 'detector'),
        deferLoad=True,
        multiple=True,
    )
    calexps = pipeBase.connectionTypes.Input(
        doc='Calexps for the visit',
        name='initial_pvi',
        storageClass='ExposureF',
        dimensions=('instrument', 'visit', 'detector'),
        deferLoad=True,
        multiple=True,
    )
    isolated_star_cats = pipeBase.connectionTypes.Input(
        doc=('Catalog of isolated stars with average positions, number of associated '
             'sources, and indexes to the isolated_star_sources catalogs.'),
        name='isolated_star_cat',
        storageClass='DataFrame',
        dimensions=('instrument', 'tract', 'skymap'),
        deferLoad=True,
        multiple=True,
    )
    isolated_star_sources = pipeBase.connectionTypes.Input(
        doc=('Catalog of isolated star sources with sourceIds, and indexes to the '
             'isolated_star_cats catalogs.'),
        name='isolated_star_sources',
        storageClass='DataFrame',
        dimensions=('instrument', 'tract', 'skymap'),
        deferLoad=True,
        multiple=True,
    )
    initial_photo_calibs = pipeBase.connectionTypes.Input(
        doc=("Initial photometric calibration that was already applied to "
             "calexps, to be removed prior to measurement in order to recover "
             "instrumental fluxes."),
        name="initial_photoCalib_detector",
        storageClass="PhotoCalib",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
        deferLoad=True,
        minimum=0,
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
        storageClass='DataFrame',
        dimensions=('instrument', 'visit'),
    )

    def adjustQuantum(self, inputs, outputs, label, data_id):
        if self.config.remove_initial_photo_calib and not inputs["initial_photo_calib"]:
            _LOG.warning(
                "Dropping %s quantum %s because initial photo calibs are needed and none were present "
                "this may be an upstream partial-outputs error covering an entire visit (which is why this "
                "is not an error), but it may mean that 'config.remove_initial_photo_calib' should be "
                "False.",
                label,
                data_id,
            )
            raise pipeBase.NoWorkFound("No initial photo calibs.")
        elif not self.config.remove_initial_photo_calib and inputs["initial_photo_calib"]:
            _LOG.warning(
                "Input collections have initial photo calib datasets but "
                "'config.remove_initial_photo_calib=False'.  This is either a very unusual collection "
                "search path or (more likely) a bad configuration.  Not that this config option should "
                "be true when using images produced by CalibrateImageTask.",
                label,
                data_id,
            )
        return super().adjustQuantum(inputs, outputs, label, data_id)


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
    remove_initial_photo_calib = pexConfig.Field(
        doc=("Expect an initial photo calib input to be present, and use it ",
             "to restore the image to instrumental units."),
        dtype=bool,
        default=True,
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
        input_handle_dict = butlerQC.get(inputRefs)

        band = butlerQC.quantum.dataId['band']
        visit = butlerQC.quantum.dataId['visit']

        src_dict_temp = {handle.dataId['detector']: handle
                         for handle in input_handle_dict['srcs']}
        calexp_dict_temp = {handle.dataId['detector']: handle
                            for handle in input_handle_dict['calexps']}
        initial_photo_calib_dict_temp = {handle.dataId['detector']: handle
                                         for handle in input_handle_dict['initial_photo_calibs']}
        isolated_star_cat_dict_temp = {handle.dataId['tract']: handle
                                       for handle in input_handle_dict['isolated_star_cats']}
        isolated_star_source_dict_temp = {handle.dataId['tract']: handle
                                          for handle in input_handle_dict['isolated_star_sources']}
        # TODO: Sort until DM-31701 is done and we have deterministic
        # dataset ordering.
        src_dict = {detector: src_dict_temp[detector] for
                    detector in sorted(src_dict_temp.keys())}
        calexp_dict = {detector: calexp_dict_temp[detector] for
                       detector in sorted(calexp_dict_temp.keys())}
        initial_photo_calib_dict = {detector: initial_photo_calib_dict_temp[detector]
                                    for detector in sorted(initial_photo_calib_dict_temp.keys())}
        isolated_star_cat_dict = {tract: isolated_star_cat_dict_temp[tract] for
                                  tract in sorted(isolated_star_cat_dict_temp.keys())}
        isolated_star_source_dict = {tract: isolated_star_source_dict_temp[tract] for
                                     tract in sorted(isolated_star_source_dict_temp.keys())}

        struct = self.run(visit,
                          band,
                          isolated_star_cat_dict,
                          isolated_star_source_dict,
                          src_dict,
                          calexp_dict,
                          initial_photo_calib_dict)

        butlerQC.put(struct.psf_ap_corr_cat,
                     outputRefs.finalized_psf_ap_corr_cat)
        butlerQC.put(pd.DataFrame(struct.output_table),
                     outputRefs.finalized_src_table)

    def run(
        self,
        visit,
        band,
        isolated_star_cat_dict,
        isolated_star_source_dict,
        src_dict,
        calexp_dict,
        initial_photo_calib_dict,
    ):
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
        initial_photo_calib_dict : `dict`
            Per-detector dict of initial photometric calibration handles

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
            isolated_star_source_dict
        )

        exposure_cat_schema = afwTable.ExposureTable.makeMinimalSchema()
        exposure_cat_schema.addField('visit', type='L', doc='Visit number')

        metadata = dafBase.PropertyList()
        metadata.add("COMMENT", "Catalog id is detector id, sorted.")
        metadata.add("COMMENT", "Only detectors with data have entries.")

        psf_ap_corr_cat = afwTable.ExposureCatalog(exposure_cat_schema)
        psf_ap_corr_cat.setMetadata(metadata)

        measured_src_tables = []
        measured_src_table = None

        for detector in src_dict:
            src = src_dict[detector].get()
            exposure = calexp_dict[detector].get()
            if detector in initial_photo_calib_dict:
                initial_photo_calib = initial_photo_calib_dict[detector].get()
            else:
                initial_photo_calib = None

            psf, ap_corr_map, measured_src = self.compute_psf_and_ap_corr_map(
                visit,
                detector,
                exposure,
                src,
                isolated_source_table,
                initial_photo_calib
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

                measured_src_tables.append(measured_src.asAstropy().as_array())

        if len(measured_src_tables) > 0:
            measured_src_table = np.concatenate(measured_src_tables)

        if measured_src_table is None:
            raise pipeBase.NoWorkFound(f'No good sources found for any detectors in visit {visit}')

        return pipeBase.Struct(psf_ap_corr_cat=psf_ap_corr_cat,
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

            # Cut isolated star table to those observed in this band, and adjust indexes
            (use_band,) = (table_cat[f'nsource_{band}'] > 0).nonzero()

            if len(use_band) == 0:
                # There are no sources in this band in this tract.
                self.log.info("No sources found in %s band in tract %d.", band, tract)
                continue

            # With the following matching:
            #   table_source[b] <-> table_cat[use_band[a]]
            obj_index = table_source['obj_index'][:]
            a, b = esutil.numpy_util.match(use_band, obj_index)

            # Update indexes and cut to band-selected stars/sources
            table_source['obj_index'][b] = a
            _, index_new = np.unique(a, return_index=True)
            table_cat[f'source_cat_index_{band}'][use_band] = index_new

            # After the following cuts, the catalogs have the following properties:
            # - table_cat only contains isolated stars that have at least one source
            #   in ``band``.
            # - table_source only contains ``band`` sources.
            # - The slice table_cat["source_cat_index_{band}"]: table_cat["source_cat_index_{band}"]
            #                                                   + table_cat["nsource_{band}]
            #   applied to table_source will give all the sources associated with the star.
            # - For each source, table_source["obj_index"] points to the index of the associated
            #   isolated star.
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
                len(table_cat),
                extra=f'{band}_{tract}',
            )
            table_source['reserved'][:] = table_cat['reserved'][table_source['obj_index']]

            # Offset indexes to account for tract merging
            table_cat[f'source_cat_index_{band}'] += merge_source_counter
            table_source['obj_index'] += merge_cat_counter

            isolated_tables.append(table_cat)
            isolated_sources.append(table_source)

            merge_cat_counter += len(table_cat)
            merge_source_counter += len(table_source)

        isolated_table = np.concatenate(isolated_tables)
        isolated_source_table = np.concatenate(isolated_sources)

        return isolated_table, isolated_source_table

    def compute_psf_and_ap_corr_map(
        self,
        visit,
        detector,
        exposure,
        src,
        isolated_source_table,
        initial_photo_calib,
    ):
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
        initial_photo_calib : `lsst.afw.image.PhotoCalib` or `None`
            Initial photometric calibration to remove from the image.

        Returns
        -------
        psf : `lsst.meas.algorithms.ImagePsf`
            PSF Model
        ap_corr_map : `lsst.afw.image.ApCorrMap`
            Aperture correction map.
        measured_src : `lsst.afw.table.SourceCatalog`
            Updated source catalog with measurements, flags and aperture corrections.
        """
        if self.config.remove_initial_photo_calib:
            if initial_photo_calib is None:
                self.log.warning("No initial photo calib found for visit %d, detector %d", visit, detector)
                return None, None, None
            if not initial_photo_calib._isConstant:
                # TODO DM-46720: remove this limitation and usage of private (why?!) property.
                raise NotImplementedError(
                    "removeInitialPhotoCalib=True can only work when the initialPhotoCalib is constant."
                )
            exposure.maskedImage /= initial_photo_calib.getCalibrationMean()

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
            isolated_source_table[self.config.id_column]
        )

        matched_arr = np.zeros(len(selected_src), dtype=bool)
        matched_arr[matched_src] = True
        selected_src['calib_psf_candidate'] = matched_arr

        reserved_arr = np.zeros(len(selected_src), dtype=bool)
        reserved_arr[matched_src] = isolated_source_table['reserved'][matched_iso]
        selected_src['calib_psf_reserved'] = reserved_arr

        selected_src = selected_src[selected_src['calib_psf_candidate']].copy(deep=True)

        # Make the measured source catalog as well, based on the selected catalog.
        measured_src = afwTable.SourceCatalog(self.schema)
        measured_src.reserve(len(selected_src))
        measured_src.extend(selected_src, mapper=self.schema_mapper)

        # We need to copy over the calib_psf flags because they were not in the mapper
        measured_src['calib_psf_candidate'] = selected_src['calib_psf_candidate']
        measured_src['calib_psf_reserved'] = selected_src['calib_psf_reserved']

        # Select the psf candidates from the selection catalog
        try:
            psf_selection_result = self.make_psf_candidates.run(selected_src, exposure=exposure)
        except Exception as e:
            self.log.warning('Failed to make psf candidates for visit %d, detector %d: %s',
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
            self.log.warning('Failed to determine psf for visit %d, detector %d: %s',
                             visit, detector, e)
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

        self.apply_ap_corr.run(catalog=measured_src, apCorrMap=ap_corr_map)

        return psf, ap_corr_map, measured_src
