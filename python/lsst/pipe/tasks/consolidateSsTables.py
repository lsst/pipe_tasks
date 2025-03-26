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

__all__ = ["ConsolidateSsTablesConfig", "ConsolidateSsTablesTask", "ConsolidateSsTablesConnections"]

import astropy.table as tb
import numpy as np
import warnings

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pipe.tasks.postprocess import TableVStack
from lsst.utils.timer import timeMethod

warnings.filterwarnings('ignore')


class ConsolidateSsTablesConnections(pipeBase.PipelineTaskConnections,
                                     dimensions=("skymap",),
                                     defaultTemplates={"coaddName": "goodSeeing",
                                                       "fakesType": ""}):
    inputCatalogs = cT.Input(
        doc="associated ssSources from all tract-patches.",
        name="{fakesType}{coaddName}Diff_assocSsSrcTable",
        storageClass="ArrowAstropy",
        dimensions=("tract", "patch"),
        multiple=True,
        deferLoad=True,
    )
    ssSourceTable = cT.Output(
        doc="",
        name="{fakesType}{coaddName}Diff_ssSrcTable",
        storageClass="ArrowAstropy",
        dimensions=(),
    )
    ssObjectTable = cT.Output(
        doc="",
        name="{fakesType}{coaddName}Diff_ssObjTable",
        storageClass="ArrowAstropy",
        dimensions=(),
    )


class ConsolidateSsTablesConfig(pipeBase.PipelineTaskConfig,
                                pipelineConnections=ConsolidateSsTablesConnections):

    """Config for ConsolidateSsTablesTask"""


class ConsolidateSsTablesTask(pipeBase.PipelineTask):
    """ Consolidate per-patch ssSource tables into a single table.
        Create ssObject table
        TODO (DM-49451): Fit per-object parameters
        TODO (DM-49453): Generate MPCORB table.
    """

    ConfigClass = ConsolidateSsTablesConfig
    _DefaultName = "consolidateSsTables"

    @timeMethod
    def run(self, inputCatalogs):
        """Concatenate per-patch ssSource tables.
            Generate ssObject table.

        Parameters
        ----------
        inputCatalogs `list` of `lsst.daf.butler.DeferredDatasetHandle`:
            All per-patch ssSource Tables

        Returns
        -------
        output : `lsst.pipe.base.Struct`
            Results struct with attributes:

            ``sSourceTable``
                Table of ssSources.
                (`astropy.table.Table`)
            ``ssObjectTable``
                Table of ssObjects
                (`astropy.table.Table`).
        """
        self.log.info("Concatenating %s per-patch ssSource Tables",
                      len(inputCatalogs))
        ssSourceTable = TableVStack.vstack_handles(inputCatalogs)
        ssObjectTable = tb.Table()
        ssObjectTable['ssObjectId'], ssObjectTable['numObs'] = np.unique(ssSourceTable['ssObjectId'],
                                                                         return_counts=True)
        ssObjectTable['discoverySubmissionDate'] = np.nan

        return pipeBase.Struct(
            ssSourceTable=ssSourceTable,
            ssObjectTable=ssObjectTable,
        )
