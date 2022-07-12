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

import pandas as pd

import lsst.utils
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

__all__ = ["MakeTrackletsConfig", "MakeTrackletsTask"]


class MakeTrackletsTaskConnections(pipeBase.PipelineTaskConnections,
                                   dimensions=(""),
                                   defaultTemplates={"": "",}):

    detections = pipeBase.connectionsType.Input(
        doc="Catalog of detections",
        dimensions=(""),
        storageClass="DataFrame",
        name="detections_catalog"
    )

    # This shouldn't be necessary since we're using a single obvseratory
    # observatory_list = pipeBase.connectionsType.Input()

    earth_ephemerides = pipeBase.connectionsType.Input(
        doc="recent Earth ephemerides",
        dimensions=(""),
        storageClass="DataFrame",
        name="earth_ephemerides"
    )

    # Should be able to use an existing stack datasetType
    exposures = pipeBase.connectionsType.Input(
        doc="Exposure information",
        dimensions=(""),
        storageClass="DataFrame",
        name="exposure_info"
    )

    tracklets = pipeBase.connectionsType.Output(
        doc="tracklets",
        dimensions=(""),
        storageClass="DataFrame",
        name="tracklets"
    )


class MakeTrackletsConfig(pipeBase.PipelineTaskConfig,
                          pipelineConnections=MakeTrackletsTaskConnections):
    """Config for MakeTrackletsTask
    """
    mintrkpts = pexConfig.Field(
        dtype=int,
        doc="...",
        default=2
    )


class MakeTrackletsTask(pipeBase.PipelineTask):
    """Make tracklets from set of detections of moving objects.
    """

    _DefaultName = "makeTrackletsTask"
    ConfigClass = MakeTrackletsTask

    def run(self, detections, earth_ephemerides, exposures):
        """Run hela.makeTracklets
        """

        hela_config = self.makeConfig()

        outputs = hela.makeTracklets(config, detections, earth_ephemerides, exposures)

        df_imgs = pd.DataFrame(outs[0])
        df_pairs = pd.DataFrame(outs[1])

        resultStruct = pipeBase.Struct(tracklets=df_pairs)
        return resultStruct

    def makeConfig(self):
        hela_config = hela.MakeTrackletsConfig()
        # Finish this later
