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

"""Pipeline for computing DiaObject summary/light curve values.
"""

__all__ = ["DrpDiaCalculationPipeTask",
           "DrpDiaCalculationPipeConfig",
           "DrpDiaCalculationPipeConnections"]

import pandas as pd

from lsst.meas.base import DiaObjectCalculationTask
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase


class DrpDiaCalculationPipeConnections(pipeBase.PipelineTaskConnections,
                                       dimensions=("tract", "patch", "skymap"),
                                       defaultTemplates={"coaddName": "deep",
                                                         "fakesType": ""}):
    assocDiaSourceTable = pipeBase.connectionTypes.Input(
        doc="Catalog of DiaSources covering the patch and associated with a "
            "DiaObject.",
        name="{fakesType}{coaddName}Diff_assocDiaSrcTable",
        storageClass="DataFrame",
        dimensions=("tract", "patch"),
    )
    diaObjectTable = pipeBase.connectionTypes.Input(
        doc="Catalog of DiaObjects created from spatially associating "
            "DiaSources.",
        name="{fakesType}{coaddName}Diff_diaObjTable",
        storageClass="DataFrame",
        dimensions=("tract", "patch"),
    )
    fullDiaObjectTable = pipeBase.connectionTypes.Output(
        doc="Catalog of DiaObjects created from spatially associating "
            "DiaSources.",
        name="{fakesType}{coaddName}Diff_fullDiaObjTable",
        storageClass="DataFrame",
        dimensions=("tract", "patch"),
    )


class DrpDiaCalculationPipeConfig(
        pipeBase.PipelineTaskConfig,
        pipelineConnections=DrpDiaCalculationPipeConnections):
    filterNames = pexConfig.ListField(
        dtype=str,
        default=['u', 'g', 'r', 'i', 'z', 'y'],
        doc="List of filters to attempt to calculate DiaObject summary "
            "values."
    )
    diaCalculation = pexConfig.ConfigurableField(
        target=DiaObjectCalculationTask,
        doc="Task to compute summary statistics for DiaObjects.",
    )

    def setDefaults(self):
        super().setDefaults()
        self.diaCalculation.plugins = ["ap_meanPosition",
                                       "ap_diaObjectFlag",
                                       "ap_meanFlux",
                                       "ap_sigmaFlux",
                                       "ap_minMaxFlux",
                                       "ap_maxSlopeFlux",
                                       "ap_meanErrFlux",
                                       "ap_meanTotFlux"]


class DrpDiaCalculationPipeTask(pipeBase.PipelineTask):
    """Driver pipeline for loading DiaSource catalogs in a patch/tract
    region and associating them.
    """
    ConfigClass = DrpDiaCalculationPipeConfig
    _DefaultName = "drpDiaCalculation"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("diaCalculation")

    def run(self, assocDiaSourceTable, diaObjectTable):
        """Compute summary statistics over the input set of DiaSources and
        store summary statistics into the associated DiaObjects.

        Parameters
        ----------
        assocDiaSourceTable : `pandas.DataFrame`
            Set of DiaSources spatially associated into the DiaObjects in
            ``diaObjectTable``.
        diaObjectTable : `pandas.DataFrame`
            DiaObjects created from associating the sources in
            ``assocDiaSourceTable``. All ids in the catalog must have a
            corresponding DiaSource in the input catalog.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Struct containing

            ``fullDiaObjectTable``
                DiaObjects with computed summary statistics based on their
                associated DiaSource light curves. (`pandas.DataFrame`).
        """
        # Return empty dataFrame if no DiaObjects in this patch.
        if len(diaObjectTable) <= 0 or len(assocDiaSourceTable) <= 0:
            return pipeBase.Struct(fullDiaObjectTable=pd.DataFrame())
        result = self.diaCalculation.run(
            diaObjectTable,
            assocDiaSourceTable,
            diaObjectTable.index.to_numpy(),
            self.config.filterNames)
        return pipeBase.Struct(fullDiaObjectTable=result.updatedDiaObjects)
