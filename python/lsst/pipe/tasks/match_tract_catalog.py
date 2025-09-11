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
    'MatchTractCatalogSubConfig', 'MatchTractCatalogSubTask',
    'MatchTractCatalogConfig', 'MatchTractCatalogTask'
]

from abc import ABC, abstractmethod
from typing import Tuple, Set

import astropy.table
import numpy as np
import pandas as pd

import lsst.afw.geom as afwGeom
import lsst.geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.obs.base.utils import TableVStack
from lsst.skymap import BaseSkyMap

from .diff_matched_tract_catalog import DiffMatchedTractCatalogTaskBase

MatchTractCatalogBaseTemplates = {
    "name_input_cat_ref": "truth_summary",
    "name_input_cat_target": "objectTable_tract",
}


class MatchTractCatalogConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "skymap"),
    defaultTemplates=MatchTractCatalogBaseTemplates,
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
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    cat_output_matched = cT.Output(
        doc="Target matched catalog with indices of reference matches",
        name="matched_{name_input_cat_ref}_{name_input_cat_target}",
        storageClass="ArrowAstropy",
        dimensions=("tract", "skymap"),
    )
    cat_output_ref = cT.Output(
        doc="Reference matched catalog with indices of target matches",
        name="match_ref_{name_input_cat_ref}_{name_input_cat_target}",
        storageClass="ArrowAstropy",
        dimensions=("tract", "skymap"),
    )
    cat_output_target = cT.Output(
        doc="Target matched catalog with indices of reference matches",
        name="match_target_{name_input_cat_ref}_{name_input_cat_target}",
        storageClass="ArrowAstropy",
        dimensions=("tract", "skymap"),
    )

    def __init__(self, *, config=None):
        if not config.output_matched_catalog:
            del self.cat_output_matched
        if config.match_multiple_target:
            # allConnections will change during iteration
            for name, connection in tuple(self.allConnections.items()):
                if connection.storageClass == "SkyMap":
                    continue
                kwargs = {"deferLoad": connection.deferLoad} if hasattr(connection, "deferLoad") else {}
                delattr(self, name)
                setattr(
                    self,
                    name,
                    type(connection)(
                        doc=connection.doc,
                        name=connection.name,
                        storageClass=connection.storageClass,
                        dimensions=connection.dimensions,
                        multiple=True,
                        **kwargs
                    ),
                )
            self.dimensions = {"skymap"}
        if config.refcat_sharding_type != "tract":
            if config.refcat_sharding_type == "none":
                old = self.cat_ref
                self.cat_ref = cT.Input(
                    doc=old.doc,
                    name=old.name,
                    storageClass=old.storageClass,
                    dimensions=(),
                    deferLoad=old.deferLoad,
                )
            else:
                raise NotImplementedError(f"{config.refcat_sharding_type=} not implemented")
        if config.target_sharding_type != "tract":
            if config.target_sharding_type == "none":
                old = self.cat_target
                self.cat_target = cT.Input(
                    doc=old.doc,
                    name=old.name,
                    storageClass=old.storageClass,
                    dimensions=(),
                    deferLoad=old.deferLoad,
                )
            else:
                raise NotImplementedError(f"{config.target_sharding_type=} not implemented")


class MatchTractCatalogSubConfig(pexConfig.Config):
    """Config class for the MatchTractCatalogSubTask to define methods returning
    values that depend on multiple config settings.
    """
    @property
    @abstractmethod
    def columns_in_ref(self) -> Set[str]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def columns_in_target(self) -> Set[str]:
        raise NotImplementedError()


class MatchTractCatalogSubTask(pipeBase.Task, ABC):
    """An abstract interface for subtasks of MatchTractCatalogTask to match
    two tract object catalogs.

    Parameters
    ----------
    **kwargs
        Additional arguments to be passed to the `lsst.pipe.base.Task`
        constructor.
    """
    ConfigClass = MatchTractCatalogSubConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def run(
        self,
        catalog_ref: pd.DataFrame | astropy.table.Table,
        catalog_target: pd.DataFrame | astropy.table.Table,
        wcs: afwGeom.SkyWcs = None,
    ) -> pipeBase.Struct:
        """Match sources in a reference tract catalog with a target catalog.

        Parameters
        ----------
        catalog_ref : `pandas.DataFrame` | `astropy.table.Table`
            A reference catalog to match objects/sources from.
        catalog_target : `pandas.DataFrame` | `astropy.table.Table`
            A target catalog to match reference objects/sources to.
        wcs : `lsst.afw.image.SkyWcs`
            A coordinate system to convert catalog positions to sky coordinates.

        Returns
        -------
        retStruct : `lsst.pipe.base.Struct`
            A struct with output_ref and output_target attribute containing the
            output matched catalogs.
        """
        raise NotImplementedError()


class MatchTractCatalogConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=MatchTractCatalogConnections,
):
    """Configure a MatchTractCatalogTask, including a configurable matching subtask.
    """
    coord_unit = pexConfig.Field[str](
        doc="lsst.geom unit of the coordinate columns. Only used to determine"
            " the tract for rows in non-tract-sharded catalogs "
            " without a tract column.",
        optional=True,
    )
    diff_matched_catalog = pexConfig.ConfigurableField(
        target=DiffMatchedTractCatalogTaskBase,
        doc="Task to make a matched catalog out of the match index tables",
    )
    match_multiple_target = pexConfig.Field[bool](
        doc="Whether to match multiple target tract catalogs",
        default=False,
    )
    match_tract_catalog = pexConfig.ConfigurableField(
        target=MatchTractCatalogSubTask,
        doc="Task to match sources in a reference tract catalog with a target catalog",
    )
    output_matched_catalog = pexConfig.Field[bool](
        doc="Whether to run the diff_matched_catalog task and write a matched catalog,"
            " not just the catalogs of match indices",
        default=False,
    )
    refcat_sharding_type = pexConfig.ChoiceField[str](
        doc="The type of sharding (spatial splitting) for the reference catalog",
        allowed={"tract": "Tract-based shards", "none": "No sharding at all"},
        default="tract",
    )
    target_sharding_type = pexConfig.ChoiceField[str](
        doc="The type of sharding (spatial splitting) for the target catalog",
        allowed={"tract": "Tract-based shards", "none": "No sharding at all"},
        default="tract",
    )

    def get_columns_in(self) -> Tuple[Set, Set]:
        """Get the set of input columns required for matching.

        Returns
        -------
        columns_ref : `set` [`str`]
            The set of required input catalog column names.
        columns_target : `set` [`str`]
            The set of required target catalog column names.
        """
        try:
            columns_ref, columns_target = (set(self.match_tract_catalog.columns_in_ref),
                                           set(self.match_tract_catalog.columns_in_target))
        except AttributeError as err:
            raise RuntimeError(f'{__class__}.match_tract_catalog must have columns_in_ref and'
                               f' columns_in_target attributes: {err}') from None
        if self.output_matched_catalog:
            config_diff = self.diff_matched_catalog.value
            columns_ref.update(config_diff.columns_in_ref)
            columns_target.update(config_diff.columns_in_target)
        return columns_ref, columns_target


class MatchTractCatalogTask(pipeBase.PipelineTask):
    """Match sources in a reference tract catalog with those in a target catalog.
    """
    ConfigClass = MatchTractCatalogConfig
    _DefaultName = "MatchTractCatalog"

    def __init__(self, initInputs, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)
        self.makeSubtask("match_tract_catalog")
        self.makeSubtask("diff_matched_catalog")

    _astropy_u_to_lsst_geom = {
        "arcmin": "arcminutes",
        "deg": "degrees",
        "rad": "radians",
    }

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        columns_ref, columns_target = self.config.get_columns_in()
        skymap = inputs.pop("skymap")
        is_refcat_per_tract = self.config.refcat_sharding_type == 'tract'
        is_target_per_tract = self.config.target_sharding_type == 'tract'

        if self.config.match_multiple_target:
            if is_target_per_tract:
                names_columns = [("cat_target", columns_target)]
                catalogs = {}
            else:
                catalogs = {"cat_target": inputs["cat_target"].get(parameters={'columns': columns_target})}
                names_columns = []
            if is_refcat_per_tract:
                names_columns.append(("cat_ref", columns_ref))
            else:
                catalogs["cat_ref"] = inputs["cat_ref"].get(parameters={'columns': columns_ref})

            for name, columns in names_columns:
                extra_values = {}
                handles = []
                for idx, (tract, handle) in enumerate(sorted(
                    (inputRef.dataId["tract"], inputHandle)
                    for inputRef, inputHandle in zip(getattr(inputRefs, name), inputs[name], strict=True)
                )):
                    handles.append(handle)
                    extra_values[idx] = {"tract": tract}
                catalogs[name] = TableVStack.vstack_handles(
                    handles,
                    extra_values=extra_values,
                    kwargs_get={"parameters": {"columns": columns}}
                )
            catalog_ref, catalog_target = catalogs["cat_ref"], catalogs["cat_target"]
        else:
            catalog_ref, catalog_target = (
                inputs[name].get(parameters={'columns': columns})
                for name, columns in (('cat_ref', columns_ref), ('cat_target', columns_target))
            )
        if self.config.match_multiple_target:
            self._add_tract_column_to_catalogs(catalog_ref, catalog_target, skymap)

        outputs = self.run(
            catalog_ref=catalog_ref,
            catalog_target=catalog_target,
            wcs=None if self.config.match_multiple_target else skymap[butlerQC.quantum.dataId["tract"]].wcs,
        )
        if self.config.output_matched_catalog:
            outputs_new = self.diff_matched_catalog.run(
                catalog_ref=catalog_ref,
                catalog_target=catalog_target,
                catalog_match_ref=outputs.cat_output_ref,
                catalog_match_target=outputs.cat_output_target,
            )
            outputs = pipeBase.Struct(**outputs.getDict(), cat_output_matched=outputs_new.cat_matched)
        if self.config.match_multiple_target:
            outputs_new = {}

            for name, catalog_in in (("cat_output_ref", catalog_ref), ("cat_output_target", catalog_target)):
                catalogs_out = []
                for outputRef in getattr(outputRefs, name):
                    tract = outputRef.dataId["tract"]
                    catalog_out = getattr(outputs, name)
                    catalogs_out.append(catalog_out[catalog_in["tract"] == tract])
                outputs_new[name] = catalogs_out
            if self.config.output_matched_catalog:
                cat_matched = outputs.cat_output_matched
                catalogs_out = []
                tract_target = cat_matched[
                    f"{self.config.diff_matched_catalog.value.column_matched_prefix_target}tract"
                ]
                masked_target = tract_target.mask == True  # noqa: E712

                tract_ref = cat_matched[
                    f"{self.config.diff_matched_catalog.value.column_matched_prefix_ref}tract"
                ]

                for outputRef in outputRefs.cat_output_matched:
                    tract = outputRef.dataId["tract"]
                    within = np.array(tract_target == tract)
                    within |= (masked_target & np.array(tract_ref == tract))
                    catalogs_out.append(cat_matched[within])
                outputs_new["cat_output_matched"] = catalogs_out

            outputs = pipeBase.Struct(
                **outputs_new,
                **{k: v for k, v in outputs.getDict().items() if k not in outputs_new},
            )

        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        catalog_ref: pd.DataFrame | astropy.table.Table,
        catalog_target: pd.DataFrame | astropy.table.Table,
        wcs: afwGeom.SkyWcs = None,
    ) -> pipeBase.Struct:
        """Match sources in a reference tract catalog with a target catalog.

        Parameters
        ----------
        catalog_ref : `pandas.DataFrame` | `astropy.table.Table`
            A reference catalog to match objects/sources from.
        catalog_target : `pandas.DataFrame` | `astropy.table.Table`
            A target catalog to match reference objects/sources to.
        wcs : `lsst.afw.image.SkyWcs`
            A coordinate system to convert catalog positions to sky coordinates,
            if necessary.

        Returns
        -------
        retStruct : `lsst.pipe.base.Struct`
            A struct with output_ref and output_target attribute containing the
            output matched catalogs.
        """
        output = self.match_tract_catalog.run(catalog_ref, catalog_target, wcs=wcs)
        if output.exceptions:
            self.log.warning('Exceptions: %s', output.exceptions)
        retStruct = pipeBase.Struct(cat_output_ref=output.cat_output_ref,
                                    cat_output_target=output.cat_output_target)
        return retStruct

    def _add_tract_column_to_catalogs(self, catalog_ref, catalog_target, skymap):
        errors = []
        if compute_tract_target := ("tract" not in catalog_target.colnames):
            if self.config.target_sharding_type != "none":
                errors.append(
                    f"Target catalog has no tract column with {self.config.target_sharding_type=} != 'none'"
                )
        if compute_tract_ref := ("tract" not in catalog_ref.colnames):
            if self.config.refcat_sharding_type != "none":
                errors.append(
                    f"Ref catalog has no tract column with {self.config.refcat_sharding_type=} != 'none'"
                )
        if errors:
            raise RuntimeError("; ".join(errors))
        unit_dict = self._astropy_u_to_lsst_geom.copy()
        if compute_tract_target or compute_tract_ref:
            if not self.config.diff_matched_catalog.coord_format.coords_spherical:
                raise RuntimeError(
                    f"Can't compute tract columns with unless"
                    f" {self.config.diff_matched_catalog.coord_format.coords_spherical} == True"
                )
            ref_c, target_c = self.config.diff_matched_catalog.coord_format.format_catalogs(
                catalog_ref=catalog_ref, catalog_target=catalog_target,
            )
            SpherePoint = lsst.geom.SpherePoint
            cats_add = []
            if compute_tract_target:
                cats_add.append((target_c, catalog_target, "target"))
            if compute_tract_ref:
                cats_add.append((ref_c, catalog_ref, "ref"))
            for cat_c, catalog_add, name_c in cats_add:
                unit = getattr(catalog_add[cat_c.column_coord1], "unit", self.config.coord_unit)
                if unit is None:
                    raise RuntimeError(
                        f"Must specify coord_unit since {name_c} column={cat_c.column_coord1}"
                        f" has no units"
                    )
                unit = getattr(lsst.geom, unit_dict[str(unit)])
                coords = [SpherePoint(ra, dec, unit) for ra, dec in zip(cat_c.coord1, cat_c.coord2)]
                catalog_add["tract"] = np.array([skymap.findTract(coord).getId() for coord in coords])
