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
    'MatchTractCatalogProbabilisticConfig', 'MatchTractCatalogProbabilisticTask',
]

import lsst.afw.geom as afwGeom
from lsst.meas.astrom.match_probabilistic_task import MatchProbabilisticTask
from lsst.meas.astrom.matcher_probabilistic import MatchProbabilisticConfig
import lsst.pipe.base as pipeBase

from .match_tract_catalog import MatchTractCatalogSubConfig, MatchTractCatalogSubTask

import astropy.table
import pandas as pd
from typing import Set


class MatchTractCatalogProbabilisticConfig(MatchProbabilisticConfig, MatchTractCatalogSubConfig):
    """Config class for the MatchTractCatalogSubTask to define methods returning
    values that depend on multiple config settings.
    """

    @property
    def columns_in_ref(self) -> Set[str]:
        return super().columns_in_ref

    @property
    def columns_in_target(self) -> Set[str]:
        return super().columns_in_target


class MatchTractCatalogProbabilisticTask(MatchProbabilisticTask, MatchTractCatalogSubTask):
    """An abstract interface for subtasks of MatchTractCatalogTask to match
    two tract object catalogs.

    Parameters
    ----------
    **kwargs
        Additional arguments to be passed to the `lsst.pipe.base.Task`
        constructor.
    """
    ConfigClass = MatchTractCatalogProbabilisticConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
            Only needed if `config.coords_ref_to_convert` is used to convert
            reference catalog sky coordinates to pixel positions.

        Returns
        -------
        retStruct : `lsst.pipe.base.Struct`
            A struct with output_ref and output_target attribute containing the
            output matched catalogs.
        """
        return super().run(
            catalog_ref=catalog_ref,
            catalog_target=catalog_target,
            wcs=wcs,
        )
