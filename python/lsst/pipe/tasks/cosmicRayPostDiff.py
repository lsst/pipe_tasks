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

__all__ = ["CosmicRayPostDiffConfig", "CosmicRayPostDiffTask"]

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.utils.timer import timeMethod
from .repair import RepairTask


class CosmicRayPostDiffConnections(pipeBase.PipelineTaskConnections,
                                   dimensions=("instrument", "visit", "detector"),
                                   defaultTemplates={"coaddName": "deep",
                                                     "fakesType": ""}):
    exposure = cT.Input(
        doc="Input difference image",
        name="{fakesType}{coaddName}Diff_differenceExp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
    )
    repaired = cT.Output(
        doc="Output subtracted image after CR run.",
        name="{fakesType}{coaddName}Diff_repairedExp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
    )

    def adjustQuantum(self, inputs, outputs, label, dataId):
        # Docstring inherited from PipelineTaskConnections
        try:
            return super().adjustQuantum(inputs, outputs, label, dataId)
        except pipeBase.ScalarError as err:
            raise pipeBase.ScalarError(
                "CosmicRayPostDiffTask can at present only be run on visits that are associated with "
                "exactly one exposure.  Either this is not a valid exposure for this pipeline, or the "
                "snap-combination step you probably want hasn't been configured to run between ISR and "
                "this task (as of this writing, that would be because it hasn't been implemented yet)."
            ) from err


class CosmicRayPostDiffConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=CosmicRayPostDiffConnections):

    """Config for CosmicRayPostDiffTask"""
    repair = pexConfig.ConfigurableField(
        target=RepairTask,
        doc="Remove cosmic rays",
    )

    def setDefaults(self):
        super().setDefaults()


class CosmicRayPostDiffTask(pipeBase.PipelineTask):
    """ Detect and repair cosmic rays on an image difference.
        Given an output image from image differencing:
          - detect and repair cosmic rays
          - At the moment this task does NOT recompute the PSF or re-do source detection.
        To invoke the task, Call the `run` method.
    """

    ConfigClass = CosmicRayPostDiffConfig
    _DefaultName = "cosmicRayPostDiff"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def __init__(self, **kwargs):
        """Construct a CosmicRayPostDiffTask"""
        super().__init__(**kwargs)
        self.makeSubtask("repair")

    @timeMethod
    def run(self, exposure):
        """Run cosmic ray detection and repair on imageDifference outputs

        Parameters
        ----------
        exposure `lsst.afw.image.Exposure`:
            The following changes are made to the exposure,
            - Blank cosmic ray mask planes, run CR detection and repair, update CR mask plane

        Returns
        -------
        `lsst.pipe.base.Struct`:
             Struct containing the input image with the CR mask plane first blanked,
             and then cosmic rays detected, and the mask is updated accordingly.
             The PSF model is NOT changed.
        """

        self.repair.run(exposure=exposure)

        return pipeBase.Struct(
            repaired=exposure
        )
