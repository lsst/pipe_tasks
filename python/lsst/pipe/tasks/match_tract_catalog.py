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
import pandas as pd

import lsst.afw.geom as afwGeom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.obs.base.utils import TableVStack
from lsst.skymap import BaseSkyMap


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
        if config.match_multiple_target:
            if config.refcat_sharding_type != "tract":
                raise ValueError(
                    f"{config.match_multiple_target} only supported for refcat_sharding_type=tract,"
                    f" not {config.refcat_sharding_type=}"
                )
            connections = {
                name: getattr(self, name)
                for name in ("cat_ref", "cat_target", "cat_output_ref", "cat_output_target")
            }
            del self.cat_ref, self.cat_target, self.cat_output_ref, self.cat_output_target
            for name, connection in connections.items():
                kwargs = {"deferLoad": connection.deferLoad} if hasattr(connection, "deferLoad") else {}
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
            self.dimensions = ("skymap",)
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
    match_multiple_target = pexConfig.Field[bool](
        doc="Whether to match multiple target tract catalogs",
        default=False,
    )
    match_tract_catalog = pexConfig.ConfigurableField(
        target=MatchTractCatalogSubTask,
        doc="Task to match sources in a reference tract catalog with a target catalog",
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
            columns_ref, columns_target = (self.match_tract_catalog.columns_in_ref,
                                           self.match_tract_catalog.columns_in_target)
        except AttributeError as err:
            raise RuntimeError(f'{__class__}.match_tract_catalog must have columns_in_ref and'
                               f' columns_in_target attributes: {err}') from None
        return set(columns_ref), set(columns_target)


class MatchTractCatalogTask(pipeBase.PipelineTask):
    """Match sources in a reference tract catalog with those in a target catalog.
    """
    ConfigClass = MatchTractCatalogConfig
    _DefaultName = "MatchTractCatalog"

    def __init__(self, initInputs, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)
        self.makeSubtask("match_tract_catalog")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        columns_ref, columns_target = self.config.get_columns_in()
        skymap = inputs.pop("skymap")
        is_refcat_per_tract = self.config.refcat_sharding_type == 'tract'

        if self.config.match_multiple_target:
            names = ['cat_target']
            if is_refcat_per_tract:
                names.append('cat_ref')
            catalogs = {}
            for name in names:
                handles = []
                extra_values = {}
                for tract, handle in sorted(
                    (inputRef.dataId["tract"], inputHandle)
                    for inputRef, inputHandle in zip(getattr(inputRefs, name), inputs[name], strict=True)
                ):
                    handles.append(handle)
                    extra_values[handle] = {"tract": tract}
                catalogs[name] = TableVStack.vstack_handles(handles, extra_values=extra_values)
            catalog_ref, catalog_target = catalogs['cat_ref'], catalogs['cat_target']
        else:
            catalog_ref, catalog_target = (
                inputs[name].get(parameters={'columns': columns})
                for (name, columns) in (('cat_ref', columns_ref), ('cat_target', columns_target))
            )

        outputs = self.run(
            catalog_ref=catalog_ref,
            catalog_target=catalog_target,
            wcs=None if self.config.match_multiple_target else skymap[butlerQC.quantum.dataId["tract"]].wcs,
        )
        if self.config.match_multiple_target:
            outputs_new = {}
            for name, catalog_in in (("cat_output_ref", catalog_ref), ("cat_output_target", catalog_target)):
                catalogs_out = []
                for outputRef in getattr(outputRefs, name):
                    tract = outputRef.dataId["tract"]
                    catalog_out = getattr(outputs, name)
                    catalogs_out.append(catalog_out[catalog_in["tract"] == tract])
                outputs_new[name] = catalogs_out
            outputs = pipeBase.Struct(
                cat_output_ref=outputs_new["cat_output_ref"],
                cat_output_target=outputs_new["cat_output_target"],
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
