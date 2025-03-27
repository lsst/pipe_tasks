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

from __future__ import annotations

__all__ = ("SplitPrimaryTask",)

import dataclasses
from typing import ClassVar

import numpy as np
import astropy.table

from lsst.pex.config import Field, ListField
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConnections,
    PipelineTaskConfig,
    Struct,
)
import lsst.pipe.base.connectionTypes as cT


class SplitPrimaryConnections(PipelineTaskConnections, dimensions=[]):
    """Connections for SplitPrimaryTask.

    Notes
    -----
    Task and connection dimensions are set by the
    `SplitPrimaryConfig.dimensions` field.

    As this task is expected to be configured to run in several different
    pipeline contexts, all connection dataset type names should be explicitly
    configured in each; the default values are just placeholders.
    """

    full = cT.Input(
        "full",
        storageClass="ArrowAstropy",
        dimensions=[],
        doc=(
            "Input table with both primary and non-primary objects/sources "
            "and a column that distinguishes between them."
        ),
    )

    primary = cT.Output(
        "primary",
        storageClass="ArrowAstropy",
        dimensions=[],
        doc="Output table holding only primary objects/sources.",
    )

    nonprimary = cT.Output(
        "nonprimary",
        storageClass="ArrowAstropy",
        dimensions=[],
        doc="Output table holding only nonprimary objects/sources.",
    )

    def __init__(self, *, config: SplitPrimaryConfig = None):
        self.dimensions = set(config.dimensions)
        self.full = dataclasses.replace(self.full, dimensions=set(config.dimensions))
        self.primary = dataclasses.replace(self.primary, dimensions=set(config.dimensions))
        self.nonprimary = dataclasses.replace(self.nonprimary, dimensions=set(config.dimensions))


class SplitPrimaryConfig(
    PipelineTaskConfig, pipelineConnections=SplitPrimaryConnections
):
    dimensions = ListField[str](
        "Dimensions of this task and its inputs and outputs.",
        dtype=str,
        default=[],
    )
    primary_flag_column = Field[str](
        "Name of the column that distinguishes between primary (True) "
        "and non-primary (False) in the input catalog.",
        dtype=str,
        default="detect_isPrimary",
    )
    discard_primary_columns = ListField[str](
        "Additional columns to discard from the primary-only table (in addition to primary_flag_column). "
        "Configured columns that are not present in the input table are ignored.",
        dtype=str,
        default=[
            "detect_isPatchInner",
            "detect_isTractInner",
            "detect_isDeblendedSource",
            "sky_object",
            "merge_peak_sky",
        ],
    )
    discard_nonprimary_columns = ListField[str](
        "Additional columns to drop from the nonprimary-only table (in addition to primary_flag_column). "
        "Configured columns that are not present in the input table are ignored.",
        dtype=str,
        default=[],
    )


class SplitPrimaryTask(PipelineTask):
    """A task that splits its input table into "primary" and "nonprimary"
    row-subset tables based on the value of a boolean column, dropping that
    column and optionally others from the two outputs.
    """

    ConfigClass: ClassVar[type[PipelineTaskConfig]] = SplitPrimaryConfig

    _DefaultName: ClassVar[str] = "splitPrimary"

    def run(self, *, full: astropy.table.Table) -> Struct:  # type: ignore
        """Run the task.

        Parameters
        ----------
        full : `astropy.table.Table`
            Table to split into row subsets.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Structure with two attributes:

            - ``primary`` (`astropy.table.Table`) table with rows where the
              `SplitPrimaryConfig.primary_flag_column` is `True`.

            - ``nonprimary`` (`astropy.table.Table`) table with rows where the
              `SplitPrimaryConfig.primary_flag_column` is `False`.
        """
        primary_mask = full[self.config.primary_flag_column]
        primary = full[primary_mask]
        del primary[self.config.primary_flag_column]
        for name in self.config.discard_primary_columns:
            if name in primary.colnames:
                del primary[name]
        nonprimary = full[np.logical_not(primary_mask)]
        del nonprimary[self.config.primary_flag_column]
        for name in self.config.discard_nonprimary_columns:
            if name in nonprimary.colnames:
                del nonprimary[name]
        self.log.info(
            "Split %s rows into %s primary rows and %s nonprimary rows.",
            len(full),
            len(primary),
            len(nonprimary),
        )
        return Struct(primary=primary, nonprimary=nonprimary)
