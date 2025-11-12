# This file is part of trailNet.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__all__ = ["MergeTrailMLTablesTask", "MergeTrailMLTablesConfig",
           "MergeTrailMLTablesConnections"]
           
import lsst.geom
import lsst.pex.config
import lsst.pipe.base
from lsst.utils.timer import timeMethod
import numpy as np
import astropy.table as at

import os
import importlib.util

class MergeTrailMLTablesConnections(lsst.pipe.base.PipelineTaskConnections,
                             dimensions=("instrument",)):
    """Configuration for MergeTrailMLTablesTask
    """

    classification_table = lsst.pipe.base.connectionTypes.Input(
        doc="Catalog of trailing classification for each visit, "
            "per detector image.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ArrowAstropy",
        name="trail_labels_detector",
        multiple=True,
    )

    classifications = lsst.pipe.base.connectionTypes.Output(
        doc="Catalog of trailing classification for all visits and detectors, "
            "merged into a single table.",
        dimensions=("instrument",),
        storageClass="ArrowAstropy",
        name="trail_labels",
    )

# create a config class for MergeTrailMLTablesTask with no parameters.
class MergeTrailMLTablesConfig(lsst.pipe.base.PipelineTaskConfig,
pipelineConnections=MergeTrailMLTablesConnections):
    """Configuration for MergeTrailMLTablesTask
    """ 
    pass

class MergeTrailMLTablesTask(lsst.pipe.base.PipelineTask):
    """A Task to merge the astropy tables output by TrailMLTask into a single table.
    """

    _DefaultName = "mergeTrailMLTables"
    ConfigClass = MergeTrailMLTablesConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """Execute the task on a quantum.
        
        Parameters
        ----------
        butlerQC : `lsst.daf.butler.QuantumContext`
            Quantum context for butler interactions.
        inputRefs : `lsst.pipe.base.InputQuantizedConnection`
            Input data references.
        outputRefs : `lsst.pipe.base.OutputQuantizedConnection`
            Output data references.
        """
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(inputs['classification_table'])
        butlerQC.put(outputs, outputRefs)

    @timeMethod
    def run(self, classification_table):
        """Merge the astropy tables output by TrailMLTask into a single table.

        Aggregates classification tables from all detectors and visits for the
        given instrument.

        Parameters
        ----------
        classification_table : `list` of `astropy.table.Table`
            A list of astropy tables containing the classifications from TrailMLTask,
            one per (visit, detector) combination.
        
        Returns
        -------
        result : `lsst.pipe.base.Struct`
            A struct containing the merged astropy table with all classifications.
        """
        if not classification_table:
            raise ValueError("No classification tables provided to merge.")
        
        combined_table = at.vstack(classification_table)

        return lsst.pipe.base.Struct(classifications=combined_table)
