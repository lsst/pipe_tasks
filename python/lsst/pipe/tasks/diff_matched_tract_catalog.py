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
]

import lsst.afw.geom as afwGeom
from lsst.meas.astrom.matcher_probabilistic import ConvertCatalogCoordinatesConfig
from lsst.meas.astrom.match_probabilistic_task import radec_to_xy
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.skymap import BaseSkyMap
from lsst.daf.butler import DatasetProvenance

import astropy.table
import astropy.units as u
import numpy as np
from smatch.matcher import sphdist
from typing import Sequence


def is_sequence_set(x: Sequence):
    return len(x) == len(set(x))


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
        storageClass="ArrowAstropy",
        dimensions=("tract", "skymap"),
        deferLoad=True,
    )
    cat_target = cT.Input(
        doc="Target object catalog to match",
        name="{name_input_cat_target}",
        storageClass="ArrowAstropy",
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
        storageClass="ArrowAstropy",
        dimensions=("tract", "skymap"),
        deferLoad=True,
    )
    cat_match_target = cT.Input(
        doc="Target match catalog with indices of references matches",
        name="match_target_{name_input_cat_ref}_{name_input_cat_target}",
        storageClass="ArrowAstropy",
        dimensions=("tract", "skymap"),
        deferLoad=True,
    )
    columns_match_target = cT.Input(
        doc="Target match catalog columns",
        name="match_target_{name_input_cat_ref}_{name_input_cat_target}.columns",
        storageClass="ArrowColumnList",
        dimensions=("tract", "skymap"),
    )
    cat_matched = cT.Output(
        doc="Catalog with reference and target columns for joined sources",
        name="matched_{name_input_cat_ref}_{name_input_cat_target}",
        storageClass="ArrowAstropy",
        dimensions=("tract", "skymap"),
    )

    def __init__(self, *, config=None):
        if config.refcat_sharding_type != "tract":
            if config.refcat_sharding_type == "none":
                old = self.cat_ref
                del self.cat_ref
                self.cat_ref = cT.Input(
                    doc=old.doc,
                    name=old.name,
                    storageClass=old.storageClass,
                    dimensions=(),
                    deferLoad=old.deferLoad,
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
    column_match_candidate_ref = pexConfig.Field[str](
        default='match_candidate',
        doc='The column name for the boolean field identifying reference objects'
            ' that were used for matching',
        optional=True,
    )
    column_match_candidate_target = pexConfig.Field[str](
        default='match_candidate',
        doc='The column name for the boolean field identifying target objects'
            ' that were used for matching',
        optional=True,
    )
    column_matched_prefix_ref = pexConfig.Field[str](
        default='refcat_',
        doc='The prefix for matched columns copied from the reference catalog',
    )
    column_matched_prefix_target = pexConfig.Field[str](
        default='',
        doc='The prefix for matched columns copied from the target catalog',
    )
    include_unmatched = pexConfig.Field[bool](
        default=False,
        doc='Whether to include unmatched rows in the matched table',
    )
    filter_on_match_candidate = pexConfig.Field[bool](
        default=False,
        doc='Whether to use provided column_match_candidate_[ref/target] to'
            ' exclude rows from the output table. If False, any provided'
            ' columns will be copied instead.'
    )
    prefix_best_coord = pexConfig.Field[str](
        default=None,
        doc="A string prefix for ra/dec coordinate columns generated from the reference coordinate if "
            "available, and target otherwise. Ignored if None or include_unmatched is False.",
        optional=True,
    )

    @property
    def columns_in_ref(self) -> list[str]:
        columns_all = [self.coord_format.column_ref_coord1, self.coord_format.column_ref_coord2]
        for column_lists in (
            (
                self.columns_ref_copy,
            ),
            (x.columns_in_ref for x in self.columns_flux.values()),
        ):
            for column_list in column_lists:
                columns_all.extend(column_list)

        return list({column: None for column in columns_all}.keys())

    @property
    def columns_in_target(self) -> list[str]:
        columns_all = [self.coord_format.column_target_coord1, self.coord_format.column_target_coord2]
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
        default={},
    )
    columns_ref_mag_to_nJy = pexConfig.DictField[str, str](
        doc='Reference table AB mag columns to convert to nJy flux columns with new names',
        default={},
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
    columns_target_mag_to_nJy = pexConfig.DictField[str, str](
        doc='Target table AB mag columns to convert to nJy flux columns with new names',
        default={},
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
    refcat_sharding_type = pexConfig.ChoiceField[str](
        doc="The type of sharding (spatial splitting) for the reference catalog",
        allowed={"tract": "Tract-based shards", "none": "No sharding at all"},
        default="tract",
    )

    def validate(self):
        super().validate()

        errors = []

        for columns_mag, columns_in, name_columns_copy in (
            (self.columns_ref_mag_to_nJy, self.columns_in_ref, "columns_ref_copy"),
            (self.columns_target_mag_to_nJy, self.columns_in_target, "columns_target_copy"),
        ):
            columns_copy = getattr(self, name_columns_copy)
            for column_old, column_new in columns_mag.items():
                if column_old not in columns_in:
                    errors.append(
                        f"{column_old=} key in self.columns_mag_to_nJy not found in {columns_in=}; did you"
                        f" forget to add it to self.{name_columns_copy}={columns_copy}?"
                    )
                if column_new in columns_copy:
                    errors.append(
                        f"{column_new=} value found in self.{name_columns_copy}={columns_copy}"
                        f" this will cause a collision. Please choose a different name."
                    )
        if errors:
            raise ValueError("\n".join(errors))


class DiffMatchedTractCatalogTask(pipeBase.PipelineTask):
    """Load subsets of matched catalogs and output a merged catalog of matched sources.
    """
    ConfigClass = DiffMatchedTractCatalogConfig
    _DefaultName = "DiffMatchedTractCatalog"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        skymap = inputs.pop("skymap")

        columns_match_ref = ['match_row']
        if (column := self.config.column_match_candidate_ref) is not None:
            columns_match_ref.append(column)

        columns_match_target = ['match_row']
        if (column := self.config.column_match_candidate_target) is not None and (
            column in inputs['columns_match_target']
        ):
            columns_match_target.append(column)

        outputs = self.run(
            catalog_ref=inputs['cat_ref'].get(parameters={'columns': self.config.columns_in_ref}),
            catalog_target=inputs['cat_target'].get(parameters={'columns': self.config.columns_in_target}),
            catalog_match_ref=inputs['cat_match_ref'].get(parameters={'columns': columns_match_ref}),
            catalog_match_target=inputs['cat_match_target'].get(parameters={'columns': columns_match_target}),
            wcs=skymap[butlerQC.quantum.dataId["tract"]].wcs,
        )
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        catalog_ref: astropy.table.Table,
        catalog_target: astropy.table.Table,
        catalog_match_ref: astropy.table.Table,
        catalog_match_target: astropy.table.Table,
        wcs: afwGeom.SkyWcs = None,
    ) -> pipeBase.Struct:
        """Load matched reference and target (measured) catalogs, measure summary statistics, and output
        a combined matched catalog with columns from both inputs.

        Parameters
        ----------
        catalog_ref : `astropy.table.Table`
            A reference catalog to diff objects/sources from.
        catalog_target : `astropy.table.Table`
            A target catalog to diff reference objects/sources to.
        catalog_match_ref : `astropy.table.Table`
            A catalog with match indices of target sources and selection flags
            for each reference source.
        catalog_match_target : `astropy.table.Table`
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
        # Would be nice if this could refer directly to ConfigClass
        config: DiffMatchedTractCatalogConfig = self.config

        # Strip any provenance from tables before merging to prevent
        # warnings from conflicts being issued by astropy.utils.merge during
        # vstack or hstack calls.
        DatasetProvenance.strip_provenance_from_flat_dict(catalog_ref.meta)
        DatasetProvenance.strip_provenance_from_flat_dict(catalog_target.meta)
        DatasetProvenance.strip_provenance_from_flat_dict(catalog_match_ref.meta)
        DatasetProvenance.strip_provenance_from_flat_dict(catalog_match_target.meta)

        select_ref = catalog_match_ref['match_candidate']
        # Add additional selection criteria for target sources beyond those for matching
        # (not recommended, but can be done anyway)
        # It would be nice to make this a Selector but those are
        # only available in analysis_tools for now
        select_target = (
            catalog_match_target['match_candidate']
            if 'match_candidate' in catalog_match_target.columns
            else np.ones(len(catalog_match_target), dtype=bool)
        )
        for column in config.columns_target_select_true:
            select_target &= catalog_target[column]
        for column in config.columns_target_select_false:
            select_target &= ~catalog_target[column]

        ref, target = config.coord_format.format_catalogs(
            catalog_ref=catalog_ref, catalog_target=catalog_target,
            select_ref=None, select_target=select_target, wcs=wcs, radec_to_xy_func=radec_to_xy,
        )
        cat_ref = ref.catalog
        cat_target = target.catalog
        n_target = len(cat_target)

        if not config.filter_on_match_candidate:
            for cat_add, cat_match, column in (
                (cat_ref, catalog_match_ref, config.column_match_candidate_ref),
                (cat_target, catalog_match_target, config.column_match_candidate_target),
            ):
                if column is not None:
                    cat_add[column] = cat_match[column]

        match_row = catalog_match_ref['match_row']
        matched_ref = match_row >= 0
        matched_row = match_row[matched_ref]
        matched_target = np.zeros(n_target, dtype=bool)
        matched_target[matched_row] = True

        # Add/compute distance columns
        coord1_target_err, coord2_target_err = config.columns_target_coord_err
        column_dist, column_dist_err = 'match_distance', 'match_distanceErr'
        dist = np.full(n_target, np.nan)

        target_match_c1, target_match_c2 = (coord[matched_row] for coord in (target.coord1, target.coord2))
        target_ref_c1, target_ref_c2 = (coord[matched_ref] for coord in (ref.coord1, ref.coord2))

        dist_err = np.full(n_target, np.nan)
        dist[matched_row] = sphdist(
            target_match_c1, target_match_c2, target_ref_c1, target_ref_c2
        ) if config.coord_format.coords_spherical else np.hypot(
            target_match_c1 - target_ref_c1, target_match_c2 - target_ref_c2,
        )
        cat_target_matched = cat_target[matched_row]
        # This will convert a masked array to an array filled with nans
        # wherever there are bad values (otherwise sphdist can raise)
        c1_err, c2_err = (
            np.ma.getdata(cat_target_matched[c_err]) for c_err in (coord1_target_err, coord2_target_err)
        )
        # Should probably explicitly add cosine terms if ref has errors too
        dist_err[matched_row] = sphdist(
            target_match_c1, target_match_c2, target_match_c1 + c1_err, target_match_c2 + c2_err
        ) if config.coord_format.coords_spherical else np.hypot(c1_err, c2_err)
        cat_target[column_dist], cat_target[column_dist_err] = dist, dist_err

        # Create a matched table, preserving the target catalog's named index (if it has one)
        cat_left = cat_target[matched_row]
        cat_right = cat_ref[matched_ref]
        if config.column_matched_prefix_target:
            cat_left.rename_columns(
                list(cat_left.columns),
                new_names=[f'{config.column_matched_prefix_target}{col}' for col in cat_left.columns],
            )
        if config.column_matched_prefix_ref:
            cat_right.rename_columns(
                list(cat_right.columns),
                new_names=[f'{config.column_matched_prefix_ref}{col}' for col in cat_right.columns],
            )
        cat_matched = astropy.table.hstack((cat_left, cat_right))

        if config.include_unmatched:
            # Create an unmatched table with the same schema as the matched one
            # ... but only for objects with no matches (for completeness/purity)
            # and that were selected for matching (or inclusion via config)
            cat_right = astropy.table.Table(
                cat_ref[~matched_ref & select_ref]
            )
            cat_right.rename_columns(
                cat_right.colnames,
                [f"{config.column_matched_prefix_ref}{col}" for col in cat_right.colnames],
            )
            match_row_target = catalog_match_target['match_row']
            cat_left = cat_target[~(match_row_target >= 0) & select_target]
            cat_left.rename_columns(
                cat_left.colnames,
                [f"{config.column_matched_prefix_target}{col}" for col in cat_left.colnames],
            )
            # This may be slower than pandas but will, for example, create
            # masked columns for booleans, which pandas does not support.
            # See https://github.com/pandas-dev/pandas/issues/46662
            cat_unmatched = astropy.table.vstack([cat_left, cat_right])

        for columns_convert_base, prefix in (
            (config.columns_ref_mag_to_nJy, config.column_matched_prefix_ref),
            (config.columns_target_mag_to_nJy, ""),
        ):
            if columns_convert_base:
                columns_convert = {
                    f"{prefix}{k}": f"{prefix}{v}" for k, v in columns_convert_base.items()
                } if prefix else columns_convert_base
                to_convert = [cat_matched]
                if config.include_unmatched:
                    to_convert.append(cat_unmatched)
                for cat_convert in to_convert:
                    cat_convert.rename_columns(
                        tuple(columns_convert.keys()),
                        tuple(columns_convert.values()),
                    )
                    for column_flux in columns_convert.values():
                        cat_convert[column_flux] = u.ABmag.to(u.nJy, cat_convert[column_flux])

        if config.include_unmatched:
            # This is probably less efficient than just doing an outer join originally; worth checking
            cat_matched = astropy.table.vstack([cat_matched, cat_unmatched])
            if (prefix_coord := config.prefix_best_coord) is not None:
                columns_coord_best = (
                    f"{prefix_coord}{col_coord}" for col_coord in (
                        ("ra", "dec") if config.coord_format.coords_spherical else ("coord1", "coord2")
                    )
                )
                for column_coord_best, column_coord_ref, column_coord_target in zip(
                    columns_coord_best,
                    (config.coord_format.column_ref_coord1, config.coord_format.column_ref_coord2),
                    (config.coord_format.column_target_coord1, config.coord_format.column_target_coord2),
                ):
                    column_full_ref = f'{config.column_matched_prefix_ref}{column_coord_ref}'
                    column_full_target = f'{config.column_matched_prefix_target}{column_coord_target}'
                    values = cat_matched[column_full_ref]
                    unit = values.unit
                    values_bad = np.ma.masked_invalid(values).mask
                    # Cast to an unmasked array - there will be no bad values
                    values = np.array(values)
                    values[values_bad] = cat_matched[column_full_target][values_bad]
                    cat_matched[column_coord_best] = values
                    cat_matched[column_coord_best].unit = unit
                    cat_matched[column_coord_best].description = (
                        f"Best {columns_coord_best} value from {column_full_ref} if available"
                        f" else {column_full_target}"
                    )

        retStruct = pipeBase.Struct(cat_matched=cat_matched)
        return retStruct
