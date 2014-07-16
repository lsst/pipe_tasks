#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2014 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
from lsst.afw.display.ds9 import mtv
from lsst.afw.image import MaskU
import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

# The following block adds links to these tasks from the Task Documentation page.
# This works even for task(s) that are not in lsst.pipe.tasks.
## \addtogroup LSST_task_documentation
## \{
## \page pipeTasks_exampleTask
## \ref ExampleCmdLineTask "ExampleCmdLineTask"
##      An example intended to show how to write a command-line task.
## <br>
## \ref ExampleSigmaClippedStatsTask "ExampleSigmaClippedStatsTask"
##      A simple example subtask that computes sigma-clipped statistics of an image
## <br>
## \ref ExampleSimpleStatsTask "ExampleSimpleStatsTask"
##      A very simple example subtask that computes statistics of an image.
## \}

#------------------------- ExampleSigmaClippedStatsTask -------------------------#

class ExampleSigmaClippedStatsConfig(pexConfig.Config):
    """!Configuration for ExampleSigmaClippedStatsTask
    """
    badMaskPlanes = pexConfig.ListField(
        dtype = str,
        doc = "Mask planes that, if set, the associated pixel should not be included in the coaddTempExp.",
        default = ("EDGE",),
    )
    numSigmaClip = pexConfig.Field(
        doc = "number of sigmas at which to clip data",
        dtype = float,
        default = 3.0,
    )
    numIter = pexConfig.Field(
        doc = "number of iterations of sigma clipping",
        dtype = int,
        default = 2,
    )


class ExampleSigmaClippedStatsTask(pipeBase.Task):
    """!Example task to compute sigma-clipped mean and standard deviation of an image

    \section pipeTasks_ExampleSigmaClippedStatsTask_Contents Contents

     - \ref pipeTasks_ExampleSigmaClippedStatsTask_Purpose 
     - \ref pipeTasks_ExampleSigmaClippedStatsTask_Config
     - \ref pipeTasks_ExampleSigmaClippedStatsTask_Debug
     - \ref pipeTasks_ExampleSigmaClippedStatsTask_Example

    \section pipeTasks_ExampleSigmaClippedStatsTask_Purpose Description

    \copybrief ExampleSigmaClippedStatsTask

    This is a simple example task designed to be run as a subtask by ExampleCmdLineTask.
    See also ExampleSimpleStatsTask as a variant that is even simpler.
    
    The main method is \ref ExampleSigmaClippedStatsTask.run "run".

    \section pipeTasks_ExampleSigmaClippedStatsTask_Config  Configuration parameters

    See \ref ExampleSigmaClippedStatsConfig

    \section pipeTasks_ExampleSigmaClippedStatsTask_Debug   Debug variables

    This task has no debug variables.

    \section pipeTasks_ExampleSigmaClippedStatsTask_Example A complete example of using ExampleSigmaClippedStatsTask

    This code is in \link examples/exampleStatsTask.py\endlink (this one example runs both
    ExampleSigmaClippedStatsTask and ExampleSimpleStatsTask), and can be run as:
    \code
    examples/exampleStatsTask.py [fitsFile]
    \endcode
    """
    ConfigClass = ExampleSigmaClippedStatsConfig
    _DefaultName = "exampleSigmaClippedStats"

    def __init__(self, *args, **kwargs):
        """!Construct an ExampleSigmaClippedStatsTask

        The init method may compute anything that that does not require data.
        In this case we create a statistics control object using the config
        (which cannot change once the task is created).
        """
        pipeBase.Task.__init__(self, *args, **kwargs)

        self._badPixelMask = MaskU.getPlaneBitMask(self.config.badMaskPlanes)

        self._statsControl = afwMath.StatisticsControl()
        self._statsControl.setNumSigmaClip(3.0)
        self._statsControl.setNumIter(2)
        self._statsControl.setAndMask(self._badPixelMask)
        # end init (marker for Doxygen)

    # start run (marker for Doxygen)
    @pipeBase.timeMethod
    def run(self, maskedImage):
        """!Compute and return statistics for a masked image

        @param[in] maskedImage: masked image (an lsst::afw::MaskedImage)
        @return a pipeBase Struct containing:
        - mean: mean of image plane
        - meanErr: uncertainty in mean
        - stdDev: standard deviation of image plane
        - stdDevErr: uncertainty in standard deviation
        """
        statObj = afwMath.makeStatistics(maskedImage,
            afwMath.MEANCLIP | afwMath.STDEVCLIP, self._statsControl)
        mean, meanErr = statObj.getResult(afwMath.MEANCLIP)
        stdDev, stdDevErr = statObj.getResult(afwMath.STDEVCLIP)
        self.log.info("clipped mean=%0.2f; meanErr=%0.2f; stdDev=%0.2f; stdDevErr=%0.2f" % \
            (mean, meanErr, stdDev, stdDevErr))
        return pipeBase.Struct(
            mean = mean,
            meanErr = meanErr,
            stdDev = stdDev,
            stdDevErr = stdDevErr,
        )
        # end run (marker for Doxygen)

#------------------------- ExampleSimpleStatsTask -------------------------#

class ExampleSimpleStatsTask(pipeBase.Task):
    """!Example task to compute mean and standard deviation of an image

    \section pipeTasks_ExampleSimpleStatsTask_Contents Contents

     - \ref pipeTasks_ExampleSimpleStatsTask_Purpose
     - \ref pipeTasks_ExampleSimpleStatsTask_Config
     - \ref pipeTasks_ExampleSimpleStatsTask_Debug
     - \ref pipeTasks_ExampleSimpleStatsTask_Example

    \section pipeTasks_ExampleSimpleStatsTask_Purpose Description

    \copybrief ExampleSimpleStatsTask

    This was designed to be run as a subtask by ExampleCmdLineTask.
    It is about as simple as a task can be; it has no configuration parameters and requires no special
    initialization. See also ExampleSigmaClippedStatsTask as a variant that is slightly more complicated.
    
    The main method is \ref ExampleSimpleTask.run "run".

    \section pipeTasks_ExampleSimpleStatsTask_Config    Configuration parameters

    This task has no configuration parameters.

    \section pipeTasks_ExampleSimpleStatsTask_Debug     Debug variables

    This task has no debug variables.

    \section pipeTasks_ExampleSimpleStatsTask_Example A complete example of using ExampleSimpleStatsTask

    This code is in \link examples/exampleStatsTask.py\endlink (this one example runs both
    ExampleSigmaClippedStatsTask and ExampleSimpleStatsTask), and can be run as:
    \code
    examples/exampleStatsTask.py [fitsFile]
    \endcode
    """
    ### Even a task with no configuration requires setting ConfigClass
    ConfigClass = pexConfig.Config
    ### Having a default name simplifies construction of the task, since the parent task
    ### need not specify a name. Note: having a default name is required for command-line tasks.
    ### The name can be simple and need not be unique (except for multiple subtasks that will
    ### be run by a parent task at the same time).
    _DefaultName = "exampleSimpleStats"

    # The `lsst.pipe.timeMethod` decorator measures how long a task method takes to run,
    # and the resources needed to run it. The information is recorded in the task's `metadata` field.
    # Most command-line tasks (not including the example below) save metadata for the task
    # and all of its subtasks whenver the task is run.
    @pipeBase.timeMethod
    def run(self, maskedImage):
        """!Compute and return statistics for a masked image

        @param[in] maskedImage: masked image (an lsst::afw::MaskedImage)
        @return a pipeBase Struct containing:
        - mean: mean of image plane
        - meanErr: uncertainty in mean
        - stdDev: standard deviation of image plane
        - stdDevErr: uncertainty in standard deviation
        """
        self._statsControl = afwMath.StatisticsControl()
        statObj = afwMath.makeStatistics(maskedImage, afwMath.MEAN | afwMath.STDEV, self._statsControl)
        mean, meanErr = statObj.getResult(afwMath.MEAN)
        stdDev, stdDevErr = statObj.getResult(afwMath.STDEV)
        self.log.info("simple mean=%0.2f; meanErr=%0.2f; stdDev=%0.2f; stdDevErr=%0.2f" % \
            (mean, meanErr, stdDev, stdDevErr))

        return pipeBase.Struct(
            mean = mean,
            meanErr = meanErr,
            stdDev = stdDev,
            stdDevErr = stdDevErr,
        )

#------------------------- ExampleCmdLineTask -------------------------#

class ExampleCmdLineConfig(pexConfig.Config):
    """!Configuration for ExampleCmdLineTask
    """
    stats = pexConfig.ConfigurableField(
        doc = "Subtask to compute statistics of an image",
        target = ExampleSigmaClippedStatsTask,
    )
    raiseEveryN = pexConfig.Field(
        doc = "Raise an lsst.base.TaskError exception when processing every raiseEveryN'th image; " \
            + "if 0 then never raise. " \
            + "This allows one to see the effects of the --doraise command line flag",
        dtype = int,
        default = 0,
    )
    # end ExampleCmdLineConfig (marker for Doxygen)

class ExampleCmdLineTask(pipeBase.CmdLineTask):
    """!Example command-line task that computes simple statistics on an image

    \section pipeTasks_ExampleCmdLineTask_Contents Contents

     - \ref pipeTasks_ExampleCmdLineTask_Purpose
     - \ref pipeTasks_ExampleCmdLineTask_Config
     - \ref pipeTasks_ExampleCmdLineTask_Debug
     - \ref pipeTasks_ExampleCmdLineTask_Example

    \section pipeTasks_ExampleCmdLineTask_Purpose Description

    \copybrief ExampleCmdLineTask

    This example task was written for the document \ref pipeTasks_writeCmdLineTask, which describes
    the task in great detail.
    The task reads in a "calexp" (a calibrated science \ref lsst::afw::image::Exposure "exposure"),
    computes statistics on the image plane, and logs and returns the statistics.
    In addition, if debugging is enabled, it displays the image in ds9.

    The image statistics are computed using a subtask, in order to show how to call subtasks and how to
    "retarget" (replace) them with variant subtasks. See \ref pipeTasks_writeCmdLineTask_retargetingSubtasks
    for more information.

    The main method is \ref ExampleCmdLineTask.run "run".

    \section pipeTasks_ExampleCmdLineTask_Config    Configuration parameters

    See \ref ExampleCmdLineConfig

    \section pipeTasks_ExampleCmdLineTask_Debug     Debug variables

    The \link lsst.pipe.base.cmdLineTask.CmdLineTask command line task\endlink interface supports a
    flag \c -d to import \b debug.py from your \c PYTHONPATH; see \ref baseDebug for more about \b debug.py files.

    The available variables in ExampleCmdLineTask are:
    <dl>
        <dt>`display`
        <dd>If True then display the calepx in ds9
    </dl>

    \section pipeTasks_ExampleCmdLineTask_Example A complete example of using ExampleCmdLineTask

    This code is in \link examples/exampleCmdLineTask.py\endlink, and can be run as _e.g._
    \code
    examples/exampleCmdLineTask.py <path_to_data_repo> --id <data_id>
    # The following will work on an NCSA lsst* computer:
    examples/exampleCmdLineTask.py /lsst8/krughoff/diffim_data/sparse_diffim_output_v7_2 --id visit=6866601
    # also try these flags:
    --config raiseEveryN=2 --doraise
    --show config data
    \endcode

    <hr>
    To investigate the \ref pipeTasks_ExampleCmdLineTask_Debug, put something like
    \code{.py}
        import lsstDebug
        def DebugInfo(name):
            di = lsstDebug.getInfo(name)    # N.b. lsstDebug.Info(name) would call us recursively
            if name == "lsst.pipe.tasks.exampleTask":
                di.display = 1
            return di
        lsstDebug.Info = DebugInfo
    \endcode
    into your debug.py file and run exampleTask.py with the `--debug` flag.
    """
    ConfigClass = ExampleCmdLineConfig
    _DefaultName = "exampleTask"
    _imageNum = 0   # number of images processed; used to implement raiseEveryN
    # end class variables (marker for Doxygen)

    def __init__(self, *args, **kwargs):
        """Construct an ExampleCmdLineTask

        Call the parent class constructor and make the `stats` subtask from the config field of the same name.
        """
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)
        self.makeSubtask("stats")
        # end init (marker for Doxygen)
    
    # start run (marker for Doxygen)
    @pipeBase.timeMethod
    def run(self, dataRef):
        """!Compute a few statistics on the image plane of an exposure
        
        @param dataRef: data reference for a calibrated science exposure ("calexp")
        @return a pipeBase Struct containing:
        - mean: mean of image plane
        - meanErr: uncertainty in mean
        - stdDev: standard deviation of image plane
        - stdDevErr: uncertainty in standard deviation
        """
        ExampleCmdLineTask._imageNum += 1
        self.log.info("Processing image number=%d: id=%s" % (self._imageNum, dataRef.dataId))
        if self.config.raiseEveryN > 0 and self._imageNum % self.config.raiseEveryN == 0:
            raise pipeBase.TaskError(
                "As requested, raising TaskError for image number %d (raiseEveryN=%d)" % \
                (self._imageNum, self.config.raiseEveryN))

        # Unpersist the data. In this case the data reference will retrieve a "calexp" by default,
        # so the the string "calexp" is optiona, but the same data reference can be used
        # to retrieve other dataset types that use the same data ID, so it is nice to be explicit
        calExp = dataRef.get("calexp")
        maskedImage = calExp.getMaskedImage()

        # Support extra debug output. To trigger debug output the user must do two things:
        # Add the following to a file named debug.py somewhere on your $PYTHONPATH:
        # \code
        # import lsstDebug
        # def DebugInfo(name):
        #     di = lsstDebug.getInfo(name)        # N.b. lsstDebug.Info(name) would call us recursively
        #     if name == "lsst.pipe.tasks.exampleCmdLineTask":
        #         di.display = 1
        #
        #     return di
        #
        # lsstDebug.Info = DebugInfo
        # \endcode
        # into your debug.py file and run this task with the \c --debug flag.
        # - 
        import lsstDebug
        display = lsstDebug.Info(__name__).display
        if display:
            frame = 1
            mtv(calExp, frame=frame, title="photocal")

        return self.stats.run(maskedImage)
        # end run (marker for Doxygen)

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
