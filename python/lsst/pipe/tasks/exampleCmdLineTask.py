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

import lsst.afw.display as afwDisplay
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.utils.timer import timeMethod
from .exampleStatsTasks import ExampleSigmaClippedStatsTask

__all__ = ["ExampleCmdLineConfig", "ExampleCmdLineTask"]

# The following block adds links to this task from the Task Documentation page.
# This works even for task(s) that are not in lsst.pipe.tasks.
## \addtogroup LSST_task_documentation
## \{
## \page page_ExampleTask ExampleTask
## \ref ExampleCmdLineTask "ExampleCmdLineTask"
##      An example intended to show how to write a command-line task.
## \}


class ExampleCmdLineConfig(pexConfig.Config):
    """!Configuration for ExampleCmdLineTask
    """
    stats = pexConfig.ConfigurableField(
        doc="Subtask to compute statistics of an image",
        target=ExampleSigmaClippedStatsTask,
    )
    doFail = pexConfig.Field(
        doc="Raise an lsst.base.TaskError exception when processing each image? "
            "This allows one to see the effects of the --doraise command line flag",
        dtype=bool,
        default=False,
    )


class ExampleCmdLineTask(pipeBase.CmdLineTask):
    r"""!
    Example command-line task that computes simple statistics on an image

    \section pipeTasks_ExampleCmdLineTask_Contents Contents

     - \ref pipeTasks_ExampleCmdLineTask_Purpose
     - \ref pipeTasks_ExampleCmdLineTask_Config
     - \ref pipeTasks_ExampleCmdLineTask_Debug
     - \ref pipeTasks_ExampleCmdLineTask_Example

    \section pipeTasks_ExampleCmdLineTask_Purpose Description

    \copybrief ExampleCmdLineTask

    This task was written as an example for the documents \ref pipeTasks_writeTask
    and \ref pipeTasks_writeCmdLineTask.
    The task reads in a "calexp" (a calibrated science \ref lsst::afw::image::Exposure "exposure"),
    computes statistics on the image plane, and logs and returns the statistics.
    In addition, if debugging is enabled, it displays the image in current display backend.

    The image statistics are computed using a subtask, in order to show how to call subtasks and how to
    \ref pipeBase_argumentParser_retargetSubtasks "retarget" (replace) them with variant subtasks.

    The main method is \ref ExampleCmdLineTask.runDataRef "runDataRef".

    \section pipeTasks_ExampleCmdLineTask_Config    Configuration parameters

    See \ref ExampleCmdLineConfig

    \section pipeTasks_ExampleCmdLineTask_Debug     Debug variables

    This task supports the following debug variables:
    <dl>
        <dt>`display`
        <dd>If True then display the exposure in current display backend
    </dl>

    To enable debugging, see \ref baseDebug.

    \section pipeTasks_ExampleCmdLineTask_Example A complete example of using ExampleCmdLineTask

    This code is in examples/exampleCmdLineTask.py, and can be run as follows:
    \code
    examples/exampleCmdLineTask.py $OBS_TEST_DIR/data/input --id
    # that will process all data; you can also try any combination of these flags:
    --id filter=g
    --config doFail=True --doraise
    --show config data
    \endcode
    """
    ConfigClass = ExampleCmdLineConfig
    _DefaultName = "exampleTask"

    def __init__(self, *args, **kwargs):
        """Construct an ExampleCmdLineTask

        Call the parent class constructor and make the "stats" subtask from the config field of the same name.
        """
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)
        self.makeSubtask("stats")

    @timeMethod
    def runDataRef(self, dataRef):
        """!Compute a few statistics on the image plane of an exposure

        @param dataRef: data reference for a calibrated science exposure ("calexp")
        @return a pipeBase Struct containing:
        - mean: mean of image plane
        - meanErr: uncertainty in mean
        - stdDev: standard deviation of image plane
        - stdDevErr: uncertainty in standard deviation
        """
        self.log.info("Processing data ID %s", dataRef.dataId)
        if self.config.doFail:
            raise pipeBase.TaskError("Raising TaskError by request (config.doFail=True)")

        # Unpersist the raw exposure pointed to by the data reference
        rawExp = dataRef.get("raw")
        maskedImage = rawExp.getMaskedImage()

        # Support extra debug output.
        # -
        import lsstDebug
        display = lsstDebug.Info(__name__).display
        if display:
            frame = 1
            disp = afwDisplay.Display(frame=frame)
            disp.mtv(rawExp, title="exposure")

        # return the pipe_base Struct that is returned by self.stats.run
        return self.stats.run(maskedImage)

    def _getConfigName(self):
        """!Get the name prefix for the task config's dataset type, or None to prevent persisting the config

        This override returns None to avoid persisting metadata for this trivial task.

        However, if the method returns a name, then the full name of the dataset type will be <name>_config.
        The default CmdLineTask._getConfigName returns _DefaultName,
        which for this task would result in a dataset name of "exampleTask_config".

        Normally you can use the default CmdLineTask._getConfigName, but here are two reasons
        why you might want to override it:
        - If you do not want your task to write its config, then have the override return None.
          That is done for this example task, because I didn't want to clutter up the
          repository with config information for a trivial task.
        - If the default name would not be unique. An example is
          \ref lsst.pipe.tasks.makeSkyMap.MakeSkyMapTask "MakeSkyMapTask": it makes a
          \ref lsst.skymap.SkyMap "sky map" (sky pixelization for a coadd)
          for any of several different types of coadd, such as deep or goodSeeing.
          As such, the name of the persisted config must include the coadd type in order to be unique.

        Normally if you override _getConfigName then you override _getMetadataName to match.
        """
        return None

    def _getMetadataName(self):
        """!Get the name prefix for the task metadata's dataset type, or None to prevent persisting metadata

        This override returns None to avoid persisting metadata for this trivial task.

        However, if the method returns a name, then the full name of the dataset type will be <name>_metadata.
        The default CmdLineTask._getConfigName returns _DefaultName,
        which for this task would result in a dataset name of "exampleTask_metadata".

        See the description of _getConfigName for reasons to override this method.
        """
        return None
