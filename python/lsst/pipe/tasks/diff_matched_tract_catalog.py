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

__all__ = [
    'DiffMatchedTractCatalogConfig', 'DiffMatchedTractCatalogTask', 'MatchedCatalogFluxesConfig',
    'MatchType', 'MeasurementType', 'SourceType',
    'Statistic', 'Median', 'SigmaIQR', 'SigmaMAD', 'Percentile',
]

import lsst.afw.geom as afwGeom
from lsst.meas.astrom.matcher_probabilistic import (
    ComparableCatalog, ConvertCatalogCoordinatesConfig,
)
from lsst.meas.astrom.match_probabilistic_task import radec_to_xy
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.skymap import BaseSkyMap

from abc import ABCMeta, abstractmethod
from astropy.stats import mad_std
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import iqr
from typing import Dict, Sequence


def is_sequence_set(x: Sequence):
    return len(x) == len(set(x))


def is_percentile(x: str):
    return 0 <= Decimal(x) <= 100


DiffMatchedTractCatalogBaseTemplates = {
    "name_input_cat_ref": "truth_summary",
    "name_input_cat_target": "objectTable_tract",
    "name_skymap": BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
}


class DiffMatchedTractCatalogConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "skymap"),
    defaultTemplates=DiffMatchedTractCatalogBaseTemplates,
):
    cat_ref = cT.Input(
        doc="Reference object catalog to match from",
        name="{name_input_cat_ref}",
        storageClass="DataFrame",
        dimensions=("tract", "skymap"),
        deferLoad=True,
    )
    cat_target = cT.Input(
        doc="Target object catalog to match",
        name="{name_input_cat_target}",
        storageClass="DataFrame",
        dimensions=("tract", "skymap"),
        deferLoad=True,
    )
    skymap = cT.Input(
        doc="Input definition of geometry/bbox and projection/wcs for coadded exposures",
        name="{name_skymap}",
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    cat_match_ref = cT.Input(
        doc="Reference match catalog with indices of target matches",
        name="match_ref_{name_input_cat_ref}_{name_input_cat_target}",
        storageClass="DataFrame",
        dimensions=("tract", "skymap"),
        deferLoad=True,
    )
    cat_match_target = cT.Input(
        doc="Target match catalog with indices of references matches",
        name="match_target_{name_input_cat_ref}_{name_input_cat_target}",
        storageClass="DataFrame",
        dimensions=("tract", "skymap"),
        deferLoad=True,
    )
    columns_match_target = cT.Input(
        doc="Target match catalog columns",
        name="match_target_{name_input_cat_ref}_{name_input_cat_target}.columns",
        storageClass="DataFrameIndex",
        dimensions=("tract", "skymap"),
    )
    cat_matched = cT.Output(
        doc="Catalog with reference and target columns for matched sources only",
        name="matched_{name_input_cat_ref}_{name_input_cat_target}",
        storageClass="DataFrame",
        dimensions=("tract", "skymap"),
    )
    diff_matched = cT.Output(
        doc="Table with aggregated counts, difference and chi statistics",
        name="diff_matched_{name_input_cat_ref}_{name_input_cat_target}",
        storageClass="DataFrame",
        dimensions=("tract", "skymap"),
    )


class MatchedCatalogFluxesConfig(pexConfig.Config):
    column_ref_flux = pexConfig.Field(
        dtype=str,
        doc='Reference catalog flux column name',
    )
    columns_target_flux = pexConfig.ListField(
        dtype=str,
        listCheck=is_sequence_set,
        doc="List of target catalog flux column names",
    )
    columns_target_flux_err = pexConfig.ListField(
        dtype=str,
        listCheck=is_sequence_set,
        doc="List of target catalog flux error column names",
    )

    # this should be an orderedset
    @property
    def columns_in_ref(self) -> list[str]:
        return [self.column_ref_flux]

    # this should also be an orderedset
    @property
    def columns_in_target(self) -> list[str]:
        columns = [col for col in self.columns_target_flux]
        columns.extend(col for col in self.columns_target_flux_err if col not in columns)
        return columns


class DiffMatchedTractCatalogConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=DiffMatchedTractCatalogConnections,
):
    column_matched_prefix_ref = pexConfig.Field(
        dtype=str,
        default='refcat_',
        doc='The prefix for matched columns copied from the reference catalog',
    )
    column_ref_extended = pexConfig.Field(
        dtype=str,
        default='is_pointsource',
        doc='The boolean reference table column specifying if the target is extended',
    )
    column_ref_extended_inverted = pexConfig.Field(
        dtype=bool,
        default=True,
        doc='Whether column_ref_extended specifies if the object is compact, not extended',
    )
    column_target_extended = pexConfig.Field(
        dtype=str,
        default='refExtendedness',
        doc='The target table column estimating the extendedness of the object (0 <= x <= 1)',
    )

    @property
    def columns_in_ref(self) -> list[str]:
        columns_all = [self.coord_format.column_ref_coord1, self.coord_format.column_ref_coord2,
                       self.column_ref_extended]
        for column_lists in (
            (
                self.columns_ref_copy,
            ),
            (x.columns_in_ref for x in self.columns_flux.values()),
        ):
            for column_list in column_lists:
                columns_all.extend(column_list)

        return set(columns_all)

    @property
    def columns_in_target(self) -> list[str]:
        columns_all = [self.coord_format.column_target_coord1, self.coord_format.column_target_coord2,
                       self.column_target_extended]
        if self.coord_format.coords_ref_to_convert is not None:
            columns_all.extend(col for col in self.coord_format.coords_ref_to_convert.values()
                               if col not in columns_all)
        for column_lists in (
            (
                self.columns_target_coord_err,
                self.columns_target_select_false,
                self.columns_target_select_true,
                self.columns_target_copy,
            ),
            (x.columns_in_target for x in self.columns_flux.values()),
        ):
            for column_list in column_lists:
                columns_all.extend(col for col in column_list if col not in columns_all)
        return columns_all

    columns_flux = pexConfig.ConfigDictField(
        doc="Configs for flux columns for each band",
        keytype=str,
        itemtype=MatchedCatalogFluxesConfig,
    )
    columns_ref_copy = pexConfig.ListField[str](
        doc='Reference table columns to copy into cat_matched',
        default=[],
        listCheck=is_sequence_set,
    )
    columns_target_coord_err = pexConfig.ListField[str](
        doc='Target table coordinate columns with standard errors (sigma)',
        listCheck=lambda x: (len(x) == 2) and (x[0] != x[1]),
    )
    columns_target_copy = pexConfig.ListField[str](
        doc='Target table columns to copy into cat_matched',
        default=('patch',),
        listCheck=is_sequence_set,
    )
    columns_target_select_true = pexConfig.ListField[str](
        doc='Target table columns to require to be True for selecting sources',
        default=('detect_isPrimary',),
        listCheck=is_sequence_set,
    )
    columns_target_select_false = pexConfig.ListField[str](
        doc='Target table columns to require to be False for selecting sources',
        default=('merge_peak_sky',),
        listCheck=is_sequence_set,
    )
    coord_format = pexConfig.ConfigField[ConvertCatalogCoordinatesConfig](
        doc="Configuration for coordinate conversion",
    )
    extendedness_cut = pexConfig.Field[float](
        dtype=float,
        default=0.5,
        doc='Minimum extendedness for a measured source to be considered extended',
    )
    mag_num_bins = pexConfig.Field[int](
        doc='Number of magnitude bins',
        default=15,
    )
    mag_brightest_ref = pexConfig.Field[float](
        doc='Brightest magnitude cutoff for binning',
        default=15,
    )
    mag_ceiling_target = pexConfig.Field[float](
        doc='Ceiling (maximum/faint) magnitude for target sources',
        default=None,
        optional=True,
    )
    mag_faintest_ref = pexConfig.Field[float](
        doc='Faintest magnitude cutoff for binning',
        default=30,
    )
    mag_zeropoint_ref = pexConfig.Field[float](
        doc='Magnitude zeropoint for reference sources',
        default=31.4,
    )
    mag_zeropoint_target = pexConfig.Field[float](
        doc='Magnitude zeropoint for target sources',
        default=31.4,
    )
    percentiles = pexConfig.ListField[str](
        doc='Percentiles to compute for diff/chi values',
        # -2, -1, +1, +2 sigma percentiles for normal distribution
        default=('2.275', '15.866', '84.134', '97.725'),
        itemCheck=is_percentile,
        listCheck=is_sequence_set,
    )


@dataclass(frozen=True)
class MeasurementTypeInfo:
    doc: str
    name: str


class MeasurementType(Enum):
    DIFF = MeasurementTypeInfo(
        doc="difference (measured - reference)",
        name="diff",
    )
    CHI = MeasurementTypeInfo(
        doc="scaled difference (measured - reference)/error",
        name="chi",
    )


class Statistic(metaclass=ABCMeta):
    """A statistic that can be applied to a set of values.
    """
    @abstractmethod
    def doc(self) -> str:
        """A description of the statistic"""
        raise NotImplementedError('Subclasses must implement this method')

    @abstractmethod
    def name_short(self) -> str:
        """A short name for the statistic, e.g. for a table column name"""
        raise NotImplementedError('Subclasses must implement this method')

    @abstractmethod
    def value(self, values):
        """The value of the statistic for a set of input values.

        Parameters
        ----------
        values : `Collection` [`float`]
            A set of values to compute the statistic for.

        Returns
        -------
        statistic : `float`
            The value of the statistic.
        """
        raise NotImplementedError('Subclasses must implement this method')


class Median(Statistic):
    """The median of a set of values."""
    @classmethod
    def doc(cls) -> str:
        return "Median"

    @classmethod
    def name_short(cls) -> str:
        return "median"

    def value(self, values):
        return np.median(values)


class SigmaIQR(Statistic):
    """The re-scaled interquartile range (sigma equivalent)."""
    @classmethod
    def doc(cls) -> str:
        return "Interquartile range divided by ~1.349 (sigma-equivalent)"

    @classmethod
    def name_short(cls) -> str:
        return "sig_iqr"

    def value(self, values):
        return iqr(values, scale='normal')


class SigmaMAD(Statistic):
    """The re-scaled median absolute deviation (sigma equivalent)."""
    @classmethod
    def doc(cls) -> str:
        return "Median absolute deviation multiplied by ~1.4826 (sigma-equivalent)"

    @classmethod
    def name_short(cls) -> str:
        return "sig_mad"

    def value(self, values):
        return mad_std(values)


@dataclass(frozen=True)
class Percentile(Statistic):
    """An arbitrary percentile.

    Parameters
    ----------
    percentile : `float`
        A valid percentile (0 <= p <= 100).
    """
    percentile: float

    def doc(self) -> str:
        return "Median absolute deviation multiplied by ~1.4826 (sigma-equivalent)"

    def name_short(self) -> str:
        return f"pctl{f'{self.percentile/100:.5f}'[2:]}"

    def value(self, values):
        return np.percentile(values, self.percentile)

    def __post_init__(self):
        if not ((self.percentile >= 0) and (self.percentile <= 100)):
            raise ValueError(f'percentile={self.percentile} not >=0 and <= 100')


def _get_stat_name(*args):
    return '_'.join(args)


def _get_column_name(band, *args):
    return f"{band}_{_get_stat_name(*args)}"


def compute_stats(values_ref, values_target, errors_target, row, stats, suffixes, prefix, skip_diff=False):
    """Compute statistics on differences and store results in a row.

    Parameters
    ----------
    values_ref : `numpy.ndarray`, (N,)
        Reference values.
    values_target : `numpy.ndarray`, (N,)
        Measured values.
    errors_target : `numpy.ndarray`, (N,)
        Errors (standard deviations) on `values_target`.
    row : `numpy.ndarray`, (1, C)
        A numpy array with pre-assigned column names.
    stats : `Dict` [`str`, `Statistic`]
        A dict of `Statistic` values to measure, keyed by their column suffix.
    suffixes : `Dict` [`MeasurementType`, `str`]
        A dict of measurement type column suffixes, keyed by the measurement type.
    prefix : `str`
        A prefix for all column names (e.g. band).
    skip_diff : `bool`
        Whether to skip computing statistics on differences. Note that
        differences will still be computed for chi statistics.

    Returns
    -------
    row_with_stats : `numpy.ndarray`, (1, C)
        The original `row` with statistic values assigned.
    """
    n_ref = len(values_ref)
    if n_ref > 0:
        n_target = len(values_target)
        n_target_err = len(errors_target) if errors_target is not None else n_ref
        if (n_target != n_ref) or (n_target_err != n_ref):
            raise ValueError(f'lengths of values_ref={n_ref}, values_target={n_target}'
                             f', error_target={n_target_err} must match')

        do_chi = errors_target is not None
        diff = values_target - values_ref
        chi = diff/errors_target if do_chi else diff
        # Could make this configurable, but non-finite values/errors are not really usable
        valid = np.isfinite(chi)
        values_type = {} if skip_diff else {MeasurementType.DIFF: diff[valid]}
        if do_chi:
            values_type[MeasurementType.CHI] = chi[valid]

        for suffix_type, suffix in suffixes.items():
            values = values_type.get(suffix_type)
            if values is not None and len(values) > 0:
                for stat_name, stat in stats.items():
                    row[_get_stat_name(prefix, suffix, stat_name)] = stat.value(values)
    return row


@dataclass(frozen=True)
class SourceTypeInfo:
    is_extended: bool | None
    label: str


class SourceType(Enum):
    ALL = SourceTypeInfo(is_extended=None, label='all')
    RESOLVED = SourceTypeInfo(is_extended=True, label='resolved')
    UNRESOLVED = SourceTypeInfo(is_extended=False, label='unresolved')


class MatchType(Enum):
    ALL = 'all'
    MATCH_RIGHT = 'match_right'
    MATCH_WRONG = 'match_wrong'


def _get_columns(bands_columns: Dict, suffixes: Dict, suffixes_flux: Dict, suffixes_mag: Dict,
                 stats: Dict, target: ComparableCatalog, column_dist: str):
    """Get column names for a table of difference statistics.

    Parameters
    ----------
    bands_columns : `Dict` [`str`,`MatchedCatalogFluxesConfig`]
        Dict keyed by band of flux column configuration.
    suffixes, suffixes_flux, suffixes_mag : `Dict` [`MeasurementType`, `str`]
        Dict of suffixes for each `MeasurementType` type, for general columns (e.g.
        coordinates), fluxes and magnitudes, respectively.
    stats : `Dict` [`str`, `Statistic`]
        Dict of suffixes for each `Statistic` type.
    target : `ComparableCatalog`
        A target catalog with coordinate column names.
    column_dist : `str`
        The name of the distance column.

    Returns
    -------
    columns : `Dict` [`str`, `type`]
        Dictionary of column types keyed by name.
    n_models : `int`
        The number of models measurements will be made for.

    Notes
    -----
    Presently, models must be identical for each band.
    """
    # Initial columns
    columns = {
        "bin": int,
        "mag_min": float,
        "mag_max": float,
    }

    # pre-assign all of the columns with appropriate types
    n_models = 0

    bands = list(bands_columns.keys())
    n_bands = len(bands)

    for idx, (band, config_flux) in enumerate(bands_columns.items()):
        columns_suffix = [
            ('flux', suffixes_flux),
            ('mag', suffixes_mag),
        ]
        if idx == 0:
            n_models = len(config_flux.columns_target_flux)
        if (idx > 0) or (n_bands > 2):
            columns_suffix.append((f'color_{bands[idx - 1]}_m_{band}', suffixes))
        n_models_flux = len(config_flux.columns_target_flux)
        n_models_err = len(config_flux.columns_target_flux_err)

        # TODO: Do equivalent validation earlier, in the config
        if (n_models_flux != n_models) or (n_models_err != n_models):
            raise RuntimeError(f'{config_flux} len(columns_target_flux)={n_models_flux} and'
                               f' len(columns_target_flux_err)={n_models_err} must equal {n_models}')

        for sourcetype in SourceType:
            label = sourcetype.value.label
            # Totals would be redundant
            if sourcetype != SourceType.ALL:
                for item in (f'n_{itype}_{mtype.value}' for itype in ('ref', 'target')
                             for mtype in MatchType):
                    columns[_get_column_name(band, label, item)] = int

            for item in (target.column_coord1, target.column_coord2, column_dist):
                for suffix in suffixes.values():
                    for stat in stats.keys():
                        columns[_get_column_name(band, label, item, suffix, stat)] = float

            for item in config_flux.columns_target_flux:
                for prefix_item, suffixes_col in columns_suffix:
                    for suffix in suffixes_col.values():
                        for stat in stats.keys():
                            columns[_get_column_name(band, label, prefix_item, item, suffix, stat)] = float

    return columns, n_models


class DiffMatchedTractCatalogTask(pipeBase.PipelineTask):
    """Load subsets of matched catalogs and output a merged catalog of matched sources.
    """
    ConfigClass = DiffMatchedTractCatalogConfig
    _DefaultName = "DiffMatchedTractCatalog"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        skymap = inputs.pop("skymap")

        columns_match_target = ['match_row']
        if 'match_candidate' in inputs['columns_match_target']:
            columns_match_target.append('match_candidate')

        outputs = self.run(
            catalog_ref=inputs['cat_ref'].get(parameters={'columns': self.config.columns_in_ref}),
            catalog_target=inputs['cat_target'].get(parameters={'columns': self.config.columns_in_target}),
            catalog_match_ref=inputs['cat_match_ref'].get(
                parameters={'columns': ['match_candidate', 'match_row']},
            ),
            catalog_match_target=inputs['cat_match_target'].get(
                parameters={'columns': columns_match_target},
            ),
            wcs=skymap[butlerQC.quantum.dataId["tract"]].wcs,
        )
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        catalog_ref: pd.DataFrame,
        catalog_target: pd.DataFrame,
        catalog_match_ref: pd.DataFrame,
        catalog_match_target: pd.DataFrame,
        wcs: afwGeom.SkyWcs = None,
    ) -> pipeBase.Struct:
        """Load matched reference and target (measured) catalogs, measure summary statistics, and output
        a combined matched catalog with columns from both inputs.

        Parameters
        ----------
        catalog_ref : `pandas.DataFrame`
            A reference catalog to diff objects/sources from.
        catalog_target : `pandas.DataFrame`
            A target catalog to diff reference objects/sources to.
        catalog_match_ref : `pandas.DataFrame`
            A catalog with match indices of target sources and selection flags
            for each reference source.
        catalog_match_target : `pandas.DataFrame`
            A catalog with selection flags for each target source.
        wcs : `lsst.afw.image.SkyWcs`
            A coordinate system to convert catalog positions to sky coordinates,
            if necessary.

        Returns
        -------
        retStruct : `lsst.pipe.base.Struct`
            A struct with output_ref and output_target attribute containing the
            output matched catalogs.
        """
        config = self.config

        select_ref = catalog_match_ref['match_candidate'].values
        # Add additional selection criteria for target sources beyond those for matching
        # (not recommended, but can be done anyway)
        select_target = (catalog_match_target['match_candidate'].values
                         if 'match_candidate' in catalog_match_target.columns
                         else np.ones(len(catalog_match_target), dtype=bool))
        for column in config.columns_target_select_true:
            select_target &= catalog_target[column].values
        for column in config.columns_target_select_false:
            select_target &= ~catalog_target[column].values

        ref, target = config.coord_format.format_catalogs(
            catalog_ref=catalog_ref, catalog_target=catalog_target,
            select_ref=None, select_target=select_target, wcs=wcs, radec_to_xy_func=radec_to_xy,
            return_converted_columns=config.coord_format.coords_ref_to_convert is not None,
        )
        cat_ref = ref.catalog
        cat_target = target.catalog
        n_target = len(cat_target)

        match_row = catalog_match_ref['match_row'].values
        matched_ref = match_row >= 0
        matched_row = match_row[matched_ref]
        matched_target = np.zeros(n_target, dtype=bool)
        matched_target[matched_row] = True

        # Create a matched table, preserving the target catalog's named index (if it has one)
        cat_left = cat_target.iloc[matched_row]
        has_index_left = cat_left.index.name is not None
        cat_right = cat_ref[matched_ref].reset_index()
        cat_matched = pd.concat(objs=(cat_left.reset_index(drop=True), cat_right), axis=1, sort=False)
        if has_index_left:
            cat_matched.index = cat_left.index
        cat_matched.columns.values[len(cat_target.columns):] = [f'refcat_{col}' for col in cat_right.columns]

        # Add/compute distance columns
        coord1_target_err, coord2_target_err = config.columns_target_coord_err
        column_dist, column_dist_err = 'distance', 'distanceErr'
        dist = np.full(n_target, np.Inf)

        dist[matched_row] = np.hypot(
            target.coord1[matched_row] - ref.coord1[matched_ref],
            target.coord2[matched_row] - ref.coord2[matched_ref],
        )
        dist_err = np.full(n_target, np.Inf)
        dist_err[matched_row] = np.hypot(cat_target.iloc[matched_row][coord1_target_err].values,
                                         cat_target.iloc[matched_row][coord2_target_err].values)
        cat_target[column_dist], cat_target[column_dist_err] = dist, dist_err

        # Slightly smelly hack for when a column (like distance) is already relative to truth
        column_dummy = 'dummy'
        cat_ref[column_dummy] = np.zeros_like(ref.coord1)

        # Add a boolean column for whether a match is classified correctly
        extended_ref = cat_ref[config.column_ref_extended]
        if config.column_ref_extended_inverted:
            extended_ref = 1 - extended_ref

        extended_target = cat_target[config.column_target_extended].values >= config.extendedness_cut

        # Define difference/chi columns and statistics thereof
        suffixes = {MeasurementType.DIFF: 'diff', MeasurementType.CHI: 'chi'}
        # Skip diff for fluxes - covered by mags
        suffixes_flux = {MeasurementType.CHI: suffixes[MeasurementType.CHI]}
        # Skip chi for magnitudes, which have strange errors
        suffixes_mag = {MeasurementType.DIFF: suffixes[MeasurementType.DIFF]}
        stats = {stat.name_short(): stat() for stat in (Median, SigmaIQR, SigmaMAD)}

        for percentile in self.config.percentiles:
            stat = Percentile(percentile=float(Decimal(percentile)))
            stats[stat.name_short()] = stat

        # Get dict of column names
        columns, n_models = _get_columns(
            bands_columns=config.columns_flux,
            suffixes=suffixes,
            suffixes_flux=suffixes_flux,
            suffixes_mag=suffixes_mag,
            stats=stats,
            target=target,
            column_dist=column_dist,
        )

        # Setup numpy table
        n_bins = config.mag_num_bins
        data = np.zeros((n_bins,), dtype=[(key, value) for key, value in columns.items()])
        data['bin'] = np.arange(n_bins)

        # Setup bins
        bins_mag = np.linspace(start=config.mag_brightest_ref, stop=config.mag_faintest_ref,
                               num=n_bins + 1)
        data['mag_min'] = bins_mag[:-1]
        data['mag_max'] = bins_mag[1:]
        bins_mag = tuple((bins_mag[idx], bins_mag[idx + 1]) for idx in range(n_bins))

        # Define temporary columns for intermediate storage
        column_mag_temp = 'mag_temp'
        column_color_temp = 'color_temp'
        column_color_err_temp = 'colorErr_temp'
        flux_err_frac_prev = [None]*n_models
        mag_prev = [None]*n_models

        columns_target = {
            target.column_coord1: (
                ref.column_coord1, target.column_coord1, coord1_target_err, False,
            ),
            target.column_coord2: (
                ref.column_coord2, target.column_coord2, coord2_target_err, False,
            ),
            column_dist: (column_dummy, column_dist, column_dist_err, False),
        }

        # Cheat a little and do the first band last so that the color is
        # based on the last band
        band_fluxes = [(band, config_flux) for (band, config_flux) in config.columns_flux.items()]
        n_bands = len(band_fluxes)
        band_fluxes.append(band_fluxes[0])
        flux_err_frac_first = None
        mag_first = None
        mag_ref_first = None

        band_prev = None
        for idx_band, (band, config_flux) in enumerate(band_fluxes):
            if idx_band == n_bands:
                # These were already computed earlier
                mag_ref = mag_ref_first
                flux_err_frac = flux_err_frac_first
                mag_model = mag_first
            else:
                mag_ref = -2.5*np.log10(cat_ref[config_flux.column_ref_flux]) + config.mag_zeropoint_ref
                flux_err_frac = [None]*n_models
                mag_model = [None]*n_models

                if idx_band > 0:
                    cat_ref[column_color_temp] = cat_ref[column_mag_temp] - mag_ref

            cat_ref[column_mag_temp] = mag_ref

            select_ref_bins = [select_ref & (mag_ref > mag_lo) & (mag_ref < mag_hi)
                               for idx_bin, (mag_lo, mag_hi) in enumerate(bins_mag)]

            # Iterate over multiple models, compute their mags and colours (if there's a previous band)
            for idx_model in range(n_models):
                column_target_flux = config_flux.columns_target_flux[idx_model]
                column_target_flux_err = config_flux.columns_target_flux_err[idx_model]

                flux_target = cat_target[column_target_flux]
                mag_target = -2.5*np.log10(flux_target) + config.mag_zeropoint_target
                if config.mag_ceiling_target is not None:
                    mag_target[mag_target > config.mag_ceiling_target] = config.mag_ceiling_target
                mag_model[idx_model] = mag_target

                # These are needed for computing magnitude/color "errors" (which are a sketchy concept)
                flux_err_frac[idx_model] = cat_target[column_target_flux_err]/flux_target

                # Stop if idx == 0: The rest will be picked up at idx == n_bins
                if idx_band > 0:
                    # Keep these mags tabulated for convenience
                    column_mag_temp_model = f'{column_mag_temp}{idx_model}'
                    cat_target[column_mag_temp_model] = mag_target

                    columns_target[f'flux_{column_target_flux}'] = (
                        config_flux.column_ref_flux,
                        column_target_flux,
                        column_target_flux_err,
                        True,
                    )
                    # Note: magnitude errors are generally problematic and not worth aggregating
                    columns_target[f'mag_{column_target_flux}'] = (
                        column_mag_temp, column_mag_temp_model, None, False,
                    )

                    # No need for colors if this is the last band and there are only two bands
                    # (because it would just be the negative of the first color)
                    skip_color = (idx_band == n_bands) and (n_bands <= 2)
                    if not skip_color:
                        column_color_temp_model = f'{column_color_temp}{idx_model}'
                        column_color_err_temp_model = f'{column_color_err_temp}{idx_model}'

                        # e.g. if order is ugrizy, first color will be u - g
                        cat_target[column_color_temp_model] = mag_prev[idx_model] - mag_model[idx_model]

                        # Sum (in quadrature, and admittedly sketchy for faint fluxes) magnitude errors
                        cat_target[column_color_err_temp_model] = 2.5/np.log(10)*np.hypot(
                            flux_err_frac[idx_model], flux_err_frac_prev[idx_model])
                        columns_target[f'color_{band_prev}_m_{band}_{column_target_flux}'] = (
                            column_color_temp,
                            column_color_temp_model,
                            column_color_err_temp_model,
                            False,
                        )

                    for idx_bin, (mag_lo, mag_hi) in enumerate(bins_mag):
                        row = data[idx_bin]
                        # Reference sources only need to be counted once
                        if idx_model == 0:
                            select_ref_bin = select_ref_bins[idx_bin]
                        select_target_bin = select_target & (mag_target > mag_lo) & (mag_target < mag_hi)

                        for sourcetype in SourceType:
                            sourcetype_info = sourcetype.value
                            is_extended = sourcetype_info.is_extended
                            # Counts filtered by match selection and magnitude bin
                            select_ref_sub = select_ref_bin.copy()
                            select_target_sub = select_target_bin.copy()
                            if is_extended is not None:
                                is_extended_ref = (extended_ref == is_extended)
                                select_ref_sub &= is_extended_ref
                                if idx_model == 0:
                                    n_ref_sub = np.count_nonzero(select_ref_sub)
                                    row[_get_column_name(band, sourcetype_info.label, 'n_ref',
                                                         MatchType.ALL.value)] = n_ref_sub
                                select_target_sub &= (extended_target == is_extended)
                                n_target_sub = np.count_nonzero(select_target_sub)
                                row[_get_column_name(band, sourcetype_info.label, 'n_target',
                                                     MatchType.ALL.value)] = n_target_sub

                            # Filter matches by magnitude bin and true class
                            match_row_bin = match_row.copy()
                            match_row_bin[~select_ref_sub] = -1
                            match_good = match_row_bin >= 0

                            n_match = np.count_nonzero(match_good)

                            # Same for counts of matched target sources (for e.g. purity)

                            if n_match > 0:
                                rows_matched = match_row_bin[match_good]
                                subset_target = cat_target.iloc[rows_matched]
                                if (is_extended is not None) and (idx_model == 0):
                                    right_type = extended_target[rows_matched] == is_extended
                                    n_total = len(right_type)
                                    n_right = np.count_nonzero(right_type)
                                    row[_get_column_name(band, sourcetype_info.label, 'n_ref',
                                                         MatchType.MATCH_RIGHT.value)] = n_right
                                    row[_get_column_name(
                                        band, sourcetype_info.label, 'n_ref', MatchType.MATCH_WRONG.value,
                                    )] = n_total - n_right

                                # compute stats for this bin, for all columns
                                for column, (column_ref, column_target, column_err_target, skip_diff) \
                                        in columns_target.items():
                                    values_ref = cat_ref[column_ref][match_good].values
                                    errors_target = (
                                        subset_target[column_err_target].values
                                        if column_err_target is not None
                                        else None
                                    )
                                    compute_stats(
                                        values_ref,
                                        subset_target[column_target].values,
                                        errors_target,
                                        row,
                                        stats,
                                        suffixes,
                                        prefix=f'{band}_{sourcetype_info.label}_{column}',
                                        skip_diff=skip_diff,
                                    )

                            # Count matched target sources with *measured* mags within bin
                            # Used for e.g. purity calculation
                            # Should be merged with above code if there's ever a need for
                            # measuring stats on this source selection
                            select_target_sub &= matched_target

                            if is_extended is not None and (np.count_nonzero(select_target_sub) > 0):
                                n_total = np.count_nonzero(select_target_sub)
                                right_type = np.zeros(n_target, dtype=bool)
                                right_type[match_row[matched_ref & is_extended_ref]] = True
                                right_type &= select_target_sub
                                n_right = np.count_nonzero(right_type)
                                row[_get_column_name(band, sourcetype_info.label, 'n_target',
                                                     MatchType.MATCH_RIGHT.value)] = n_right
                                row[_get_column_name(band, sourcetype_info.label, 'n_target',
                                                     MatchType.MATCH_WRONG.value)] = n_total - n_right

                    # delete the flux/color columns since they change with each band
                    for prefix in ('flux', 'mag'):
                        del columns_target[f'{prefix}_{column_target_flux}']
                    if not skip_color:
                        del columns_target[f'color_{band_prev}_m_{band}_{column_target_flux}']

            # keep values needed for colors
            flux_err_frac_prev = flux_err_frac
            mag_prev = mag_model
            band_prev = band
            if idx_band == 0:
                flux_err_frac_first = flux_err_frac
                mag_first = mag_model
                mag_ref_first = mag_ref

        retStruct = pipeBase.Struct(cat_matched=cat_matched, diff_matched=pd.DataFrame(data))
        return retStruct
