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
#
"""Read preprocessed bright stars and stack them to build an extended
PSF model."""

__all__ = ["MeasureExtendedPsfTask"]

import traceback
import sys

from lsst.pipe import base as pipeBase
from lsst.pipe.tasks.assembleCoadd import AssembleCoaddTask
import lsst.pex.config as pexConfig


class MeasureExtendedPsfConfig(pexConfig.Config):
    """Configuration parameters for MeasureExtendedPsfTask
    """
    subregionSize = pexConfig.ListField(
        dtype=int,
        doc="Size, in pixels, of the subregions over which the stacking be "
            "iteratively performed.",
        default=(20, 20)
    )


class ExtendedPsfTaskRunner(pipeBase.TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [(parsedCmd.butler, parsedCmd.selectId.idList)]

    def __call__(self, parsedCmd):
        butler, selectDataList = parsedCmd
        task = self.TaskClass(config=self.config, log=self.log)
        if self.doRaise:
            results = task.runDataRef(butler, selectDataList)
        else:
            try:
                results = task.runDataRef(butler, selectDataList)
            except Exception as e:
                task.log.fatal("Failed: %s" % e)
                if not isinstance(e, pipeBase.TaskError):
                    traceback.print_exc(file=sys.stderr)
        # task.writeMetadata(butler)
        if self.doReturnResults:
            return results


class MeasureExtendedPsfTask(pipeBase.CmdLineTask):
    """TODO: write docstring
    """
    ConfigClass = MeasureExtendedPsfConfig
    RunnerClass = ExtendedPsfTaskRunner
    _DefaultName = "measureExtendedPsf"

    def __init__(self, initInputs=None, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)
        self.subBBoxes = AssembleCoaddTask._subBBoxIter(None, self.config.subregionSize)

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser.
        """
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--selectId", "brightStarStamps",
                               help="data ID, e.g. --selectId visit=6789 ccd=0..9",
                               ContainerClass=pipeBase.DataIdContainer)
        return parser

    def runDataRef(self, butler, selectDataList=None):
        return 1
