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
import astropy.units as u
import numpy as np
import warnings

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.obs.base.utils import TableVStack
from lsst.utils.timer import timeMethod
from . import ssp

warnings.filterwarnings("ignore")


class ConsolidateSsTablesConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("skymap",),
    defaultTemplates={"coaddName": "goodSeeing", "fakesType": ""},
):
    inputCatalogs = cT.Input(
        doc="associated ssSources from all tract-patches.",
        name="{fakesType}{coaddName}Diff_assocSsSrcTable",
        storageClass="ArrowAstropy",
        dimensions=("tract", "patch"),
        multiple=True,
        deferLoad=True,
    )
    mpcorb = cT.Input(
        doc="Minor Planet Center orbit table used for association",
        name="mpcorb",
        storageClass="DataFrame",
        dimensions={},
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


class ConsolidateSsTablesConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=ConsolidateSsTablesConnections
):
    """Config for ConsolidateSsTablesTask"""


class ConsolidateSsTablesTask(pipeBase.PipelineTask):
    """Consolidate per-patch ssSource tables into a single table.
    Create ssObject table
    TODO (DM-49451): Fit per-object parameters
    TODO (DM-49453): Generate MPCORB table.
    """

    ConfigClass = ConsolidateSsTablesConfig
    _DefaultName = "consolidateSsTables"

    @timeMethod
    def run(self, inputCatalogs, mpcorb):
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
        self.log.info("Concatenating %s per-patch ssSource Tables", len(inputCatalogs))
        ssSourceTable = TableVStack.vstack_handles(inputCatalogs)
        self.log.info(
            f"Done. {len(ssSourceTable)} observations, {np.unique(ssSourceTable['ssObjectId']).size} objects."
        )

        # Compatibility for pre RFC-1138 ss_source_associated tables
        if "heliocentricDist" in ssSourceTable.colnames:
            arr = ssSourceTable["ssObjectId"] + 0x20000000_00000000  # leading whitespace
            arr_s8 = arr.byteswap().view(arr.dtype.newbyteorder()).view("S8")
            ssSourceTable["designation"] = np.char.lstrip(arr_s8)
            # Distances → convert km → AU
            au_in_km = (1 * u.au).to(u.km).value
            ssSourceTable.rename_column("topocentricDist", "topoRange")
            ssSourceTable["topoRange"] /= au_in_km
            ssSourceTable.rename_column("heliocentricDist", "helioRange")
            ssSourceTable["helioRange"] /= au_in_km

            ssSourceTable.rename_column("heliocentricX", "helio_x")
            ssSourceTable["helio_x"] /= au_in_km
            ssSourceTable.rename_column("heliocentricY", "helio_y")
            ssSourceTable["helio_y"] /= au_in_km
            ssSourceTable.rename_column("heliocentricZ", "helio_z")
            ssSourceTable["helio_z"] /= au_in_km

            ssSourceTable.rename_column("topocentricX", "topo_x")
            ssSourceTable["topo_x"] /= au_in_km
            ssSourceTable.rename_column("topocentricY", "topo_y")
            ssSourceTable["topo_y"] /= au_in_km
            ssSourceTable.rename_column("topocentricZ", "topo_z")
            ssSourceTable["topo_z"] /= au_in_km

            # Velocities (no unit change, just renaming)
            ssSourceTable.rename_column("heliocentricVX", "helio_vx")
            ssSourceTable.rename_column("heliocentricVY", "helio_vy")
            ssSourceTable.rename_column("heliocentricVZ", "helio_vz")

            ssSourceTable.rename_column("topocentricVX", "topo_vx")
            ssSourceTable.rename_column("topocentricVY", "topo_vy")
            ssSourceTable.rename_column("topocentricVZ", "topo_vz")

            # the rest
            ssSourceTable.rename_column("residualRa", "ephOffsetRa")
            ssSourceTable.rename_column("residualDec", "ephOffsetDec")
            ssSourceTable.rename_column("eclipticLambda", "eclLambda")
            ssSourceTable.rename_column("eclipticBeta", "eclBeta")
            ssSourceTable.rename_column("galacticL", "galLon")
            ssSourceTable.rename_column("galacticB", "galLat")

            # if we're loading the old-style catalog, we require packed_primary_provisional_designation
            # to be in the mpcorb schema (and we'll pretend that it's actually unpacked)
            if mpcorb is not None:
                mpcorb["unpacked_primary_provisional_designation"] = mpcorb[
                    "packed_primary_provisional_designation"
                ]

        if mpcorb is not None:
            self.log.info(f"mpcorb loaded ({len(mpcorb)} objects, {len(mpcorb.columns)} columns)")
        else:
            self.log.info("mpcorb not loaded.")

        # extract the DiaSource subset and remove it from ssSourceTable
        diaSource = tb.Table()
        for c in ssp.ssobject.DIA_COLUMNS:
            src = c if c == "diaSourceId" else f"DIA_{c}"
            diaSource[c] = ssSourceTable[src]
            if src != "diaSourceId":
                del ssSourceTable[src]

        # build the SSObject table
        ssSourceTable.sort("ssObjectId")
        mpcorb = mpcorb.to_pandas() if isinstance(mpcorb, tb.Table) else mpcorb
        ssObjectTable = ssp.ssobject.compute_ssobject(
            ssSourceTable.to_pandas(), diaSource.to_pandas(), mpcorb
        )
        ssObjectTable = tb.Table(ssObjectTable)

        return pipeBase.Struct(
            ssSourceTable=ssSourceTable,
            ssObjectTable=ssObjectTable,
        )
