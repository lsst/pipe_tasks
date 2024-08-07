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

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.utils.timer import timeMethod


class ExampleSigmaClippedStatsConfig(pexConfig.Config):
    """Configuration for ExampleSigmaClippedStatsTask
    """
    badMaskPlanes = pexConfig.ListField(
        dtype=str,
        doc="Mask planes that, if set, indicate the associated pixel should "
        "not be included when the calculating statistics.",
        default=("EDGE",),
    )
    numSigmaClip = pexConfig.Field(
        doc="number of sigmas at which to clip data",
        dtype=float,
        default=3.0,
    )
    numIter = pexConfig.Field(
        doc="number of iterations of sigma clipping",
        dtype=int,
        default=2,
    )


class ExampleSigmaClippedStatsTask(pipeBase.Task):
    """Example task to compute sigma-clipped mean and standard deviation of an image.

    This is a simple example task designed to be run as a subtask by
    ExampleCmdLineTask. See also ExampleSimpleStatsTask as a variant that is
    even simpler.

    Notes
    -----
    The init method may compute anything that that does not require data.
    In this case we create a statistics control object using the config
    (which cannot change once the task is created).
    """
    ConfigClass = ExampleSigmaClippedStatsConfig
    _DefaultName = "exampleSigmaClippedStats"

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)

        self._badPixelMask = afwImage.Mask.getPlaneBitMask(self.config.badMaskPlanes)

        self._statsControl = afwMath.StatisticsControl()
        self._statsControl.setNumSigmaClip(self.config.numSigmaClip)
        self._statsControl.setNumIter(self.config.numIter)
        self._statsControl.setAndMask(self._badPixelMask)

    @timeMethod
    def run(self, maskedImage):
        """Compute and return statistics for a masked image.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Masked image to compute statistics on.

        Returns
        -------
        stats : `lsst.pipe.base.Struct`
            Statistics as a struct with attributes:

            ``mean``
                Mean of image plane (`float`).
            ``meanErr``
                Uncertainty in mean (`float`).
            ``stdDev``
                Standard deviation of image plane (`float`).
            ``stdDevErr``
                Uncertainty in standard deviation (`float`).
        """
        statObj = afwMath.makeStatistics(maskedImage, afwMath.MEANCLIP | afwMath.STDEVCLIP | afwMath.ERRORS,
                                         self._statsControl)
        mean, meanErr = statObj.getResult(afwMath.MEANCLIP)
        stdDev, stdDevErr = statObj.getResult(afwMath.STDEVCLIP)
        self.log.info("clipped mean=%0.2f; meanErr=%0.2f; stdDev=%0.2f; stdDevErr=%0.2f",
                      mean, meanErr, stdDev, stdDevErr)
        return pipeBase.Struct(
            mean=mean,
            meanErr=meanErr,
            stdDev=stdDev,
            stdDevErr=stdDevErr,
        )


class ExampleSimpleStatsTask(pipeBase.Task):
    """Example task to compute mean and standard deviation of an image.

    This was designed to be run as a subtask by ExampleCmdLineTask.
    It is about as simple as a task can be; it has no configuration parameters
    and requires no special initialization. See also
    ExampleSigmaClippedStatsTask as a variant that is slightly more
    complicated.

    The main method is ExampleSimpleTask.run "run".

    pipeTasks_ExampleSimpleStatsTask_Config    Configuration parameters

    This task has no configuration parameters.

    pipeTasks_ExampleSimpleStatsTask_Debug     Debug variables

    This task has no debug variables.
    """
    # Even a task with no configuration requires setting ConfigClass
    ConfigClass = pexConfig.Config
    # Having a default name simplifies construction of the task, since the
    # parent task need not specify a name. Note: having a default name is
    # required for command-line tasks.
    # The name can be simple and need not be unique (except for multiple
    # subtasks that will be run by a parent task at the same time).
    _DefaultName = "exampleSimpleStats"

    # The `lsst.utils.timer.timeMethod` decorator measures how long a task
    # method takes to run, and the resources needed to run it. The information
    # is recorded in the task's `metadata` field.
    # Most command-line tasks (not including the example below) save metadata
    # for the task and all of its subtasks whenver the task is run.
    @timeMethod
    def run(self, maskedImage):
        """Compute and return statistics for a masked image.

        Parameters
        ----------
        maskedImage : `lsst.afw.MaskedImage`
            Masked image to compute statistics on.

        Returns
        -------
        stats : `lsst.pipe.base.Struct`
            Statistics as a struct with attributes:

            ``mean``
                Mean of image plane (`float`).
            ``meanErr``
                Uncertainty in mean (`float`).
            ``stdDev``
                Standard deviation of image plane (`float`).
            ``stdDevErr``
                Uncertainty in standard deviation (`float`).
        """
        self._statsControl = afwMath.StatisticsControl()
        statObj = afwMath.makeStatistics(maskedImage, afwMath.MEAN | afwMath.STDEV | afwMath.ERRORS,
                                         self._statsControl)
        mean, meanErr = statObj.getResult(afwMath.MEAN)
        stdDev, stdDevErr = statObj.getResult(afwMath.STDEV)
        self.log.info("simple mean=%0.2f; meanErr=%0.2f; stdDev=%0.2f; stdDevErr=%0.2f",
                      mean, meanErr, stdDev, stdDevErr)

        return pipeBase.Struct(
            mean=mean,
            meanErr=meanErr,
            stdDev=stdDev,
            stdDevErr=stdDevErr,
        )
