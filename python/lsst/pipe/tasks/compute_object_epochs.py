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

from astropy.table import Table
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np


class ComputeObjectEpochsConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "skymap"),
):
    objectCat = pipeBase.connectionTypes.Input(
        doc="Multiband catalog of positions in each patch.",
        name="deepCoadd_obj",
        storageClass="ArrowAstropy",
        dimensions=["skymap", "tract", "patch"],
        multiple=True,
        deferLoad=True,
        deferGraphConstraint=True,
    )

    epochMap = pipeBase.connectionTypes.Input(
        doc="Healsparse map of mean epoch of objectCat in each band.",
        name="deepCoadd_epoch_map_mean",
        storageClass="HealSparseMap",
        dimensions=("skymap", "tract", "band"),
        multiple=True,
        deferLoad=True,
    )

    objectEpochs = pipeBase.connectionTypes.Output(
        doc="Catalog of epochs for objectCat objects.",
        name="object_epoch",
        storageClass="ArrowAstropy",
        dimensions=["skymap", "tract", "patch"],
        multiple=True,
    )


class ComputeObjectEpochsConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=ComputeObjectEpochsConnections,
):
    bands = pexConfig.ListField(
        doc=("Bands in objectCat to be combined with `objectCat_selectors` to build objectCat column names."),
        dtype=str,
        default=["u", "g", "r", "i", "z", "y"],
    )


class ComputeObjectEpochsTask(pipeBase.PipelineTask):
    """Collect mean epochs for the observations that went into each objepipeBase.connectionTypes.

    TODO: DM-46202, Remove this task once the object epochs are available
    elsewhere.
    """

    ConfigClass = ComputeObjectEpochsConfig
    _DefaultName = "computeObjectEpochs"

    def getEpochs(self, cat, epochMapDict):
        """Get mean epoch of the visits corresponding to object position.

        Parameters
        ----------
        cat : `astropy.table.Table`
            Catalog containing object positions.
        epochMapDict: `dict` [`DeferredDatasetHandle`]
            Dictionary of handles for healsparse maps containing the mean epoch
            for positions in the reference catalog.

        Returns
        -------
        epochDf = `astropy.table.Table`
            Catalog with mean epoch of visits at each object position.
        """
        allEpochs = {}
        for band in self.config.bands:
            epochs = np.ones(len(cat)) * np.nan
            col_ra, col_dec = (str(("meas", band, f"coord_{coord}")) for coord in ("ra", "dec"))
            if col_ra in cat.columns and col_dec in cat.columns:
                ra, dec = cat[col_ra], cat[col_dec]
                validPositions = np.isfinite(ra) & np.isfinite(dec)
                if validPositions.any():
                    bandEpochs = epochMapDict[band].get_values_pos(
                        ra[validPositions], dec[validPositions]
                    )
                    epochsValid = epochMapDict[band].get_values_pos(
                        ra[validPositions], dec[validPositions], valid_mask=True
                    )
                    bandEpochs[~epochsValid] = np.nan
                    epochs[validPositions] = bandEpochs
            allEpochs[f"{band}_epoch"] = epochs
        allEpochs["objectId"] = cat["id"]

        epochTable = Table(allEpochs)
        return epochTable

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        columns = [("meas", band, f"coord_{coord}") for band in self.config.bands for coord in ["ra", "dec"]]

        inputs["epochMap"] = {ref.dataId["band"]: ref.get() for ref in inputs["epochMap"]}

        outputEpochRefs = {outputRef.dataId["patch"]: outputRef for outputRef in outputRefs.objectEpochs}
        for objectCatRef in inputs["objectCat"]:
            patch = objectCatRef.dataId["patch"]
            columns_avail = objectCatRef.get(component="columns")
            columns_patch = [column for column in columns if str(column) in columns_avail]
            objectCat = objectCatRef.get(parameters={"columns": columns_patch})
            epochs = self.getEpochs(objectCat, inputs["epochMap"])
            butlerQC.put(epochs, outputEpochRefs[patch])
