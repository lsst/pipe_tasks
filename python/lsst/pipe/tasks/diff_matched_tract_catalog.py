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

import numpy as np
import pandas as pd
from typing import Set


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
        doc="Reference matched catalog with indices of target matches",
        name="match_ref_{name_input_cat_ref}_{name_input_cat_target}",
        storageClass="DataFrame",
        dimensions=("tract", "skymap"),
        deferLoad=True,
    )
    cat_match_target = cT.Input(
        doc="Target matched catalog with indices of references matches",
        name="match_target_{name_input_cat_ref}_{name_input_cat_target}",
        storageClass="DataFrame",
        dimensions=("tract", "skymap"),
        deferLoad=True,
    )
    cat_matched = cT.Output(
        doc="Catalog with reference and target columns for matched sources only",
        name="matched_{name_input_cat_ref}_{name_input_cat_target}",
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
        listCheck=lambda x: len(set(x)) == len(x),
        doc="List of target catalog flux column names",
    )
    columns_target_flux_err = pexConfig.ListField(
        dtype=str,
        listCheck=lambda x: len(set(x)) == len(x),
        doc="List of target catalog flux error column names",
    )

    @property
    def columns_in_ref(self) -> Set[str]:
        return {self.column_ref_flux}

    @property
    def columns_in_target(self) -> Set[str]:
        return set(self.columns_target_flux).union(set(self.columns_target_flux_err))


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
    def columns_in_ref(self) -> Set[str]:
        columns_all = [self.coord_format.column_ref_coord1, self.coord_format.column_ref_coord2,
                       self.column_ref_extended]
        for columns_list in (
            (
                self.columns_ref_copy,
            ),
            (x.columns_in_ref for x in self.columns_flux.values()),
        ):
            for columns in columns_list:
                columns_all.extend(columns)

        return set(columns_all)

    @property
    def columns_in_target(self) -> Set[str]:
        columns_all = [self.coord_format.column_target_coord1, self.coord_format.column_target_coord2,
                       self.column_target_extended]
        if self.coord_format.coords_ref_to_convert is not None:
            columns_all.extend(self.coord_format.coords_ref_to_convert.values())
        for columns_list in (
            (
                self.columns_target_coord_err,
                self.columns_target_select_false,
                self.columns_target_select_true,
                self.columns_target_copy,
            ),
            (x.columns_in_target for x in self.columns_flux.values()),
        ):
            for columns in columns_list:
                columns_all.extend(columns)
        return set(columns_all)

    columns_flux = pexConfig.ConfigDictField(
        keytype=str,
        itemtype=MatchedCatalogFluxesConfig,
        doc="Configs for flux columns for each band",
    )
    columns_ref_copy = pexConfig.ListField(
        dtype=str,
        default=set(),
        doc='Reference table columns to copy to copy into cat_matched',
    )
    columns_target_coord_err = pexConfig.ListField(
        dtype=str,
        listCheck=lambda x: (len(x) == 2) and (x[0] != x[1]),
        doc='Target table coordinate columns with standard errors (sigma)',
    )
    columns_target_copy = pexConfig.ListField(
        dtype=str,
        default=('patch',),
        doc='Target table columns to copy to copy into cat_matched',
    )
    columns_target_select_true = pexConfig.ListField(
        dtype=str,
        default=('detect_isPrimary',),
        doc='Target table columns to require to be True for selecting sources',
    )
    columns_target_select_false = pexConfig.ListField(
        dtype=str,
        default=('merge_peak_sky',),
        doc='Target table columns to require to be False for selecting sources',
    )
    coord_format = pexConfig.ConfigField(
        dtype=ConvertCatalogCoordinatesConfig,
        doc="Configuration for coordinate conversion",
    )


class DiffMatchedTractCatalogTask(pipeBase.PipelineTask):
    """Load subsets of matched catalogs and output a merged catalog of matched sources.
    """
    ConfigClass = DiffMatchedTractCatalogConfig
    _DefaultName = "DiffMatchedTractCatalog"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        skymap = inputs.pop("skymap")

        outputs = self.run(
            catalog_ref=inputs['cat_ref'].get(parameters={'columns': self.config.columns_in_ref}),
            catalog_target=inputs['cat_target'].get(parameters={'columns': self.config.columns_in_target}),
            catalog_match_ref=inputs['cat_match_ref'].get(
                parameters={'columns': ['match_candidate', 'match_row']},
            ),
            catalog_match_target=inputs['cat_match_target'].get(
                parameters={'columns': ['match_row']},
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
        """Load matched reference and target (measured) catalogs, measure summary statistics (TBD) and output
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
        cat_matched = pd.concat((cat_left.reset_index(drop=True), cat_right), 1)
        if has_index_left:
            cat_matched.index = cat_left.index
        cat_matched.columns.values[len(cat_target.columns):] = [f'refcat_{col}' for col in cat_right.columns]

        retStruct = pipeBase.Struct(cat_matched=cat_matched)
        return retStruct
