#
# LSST Data Management System
# Copyright 2008-2015 AURA/LSST.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
from lsst.ip.isr import IsrTask
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from .calibrate import CalibrateTask
from .characterizeImage import CharacterizeImageTask

class ProcessCcdConfig(pexConfig.Config):
    """Config for ProcessCcd"""
    isr = pexConfig.ConfigurableField(
        target = IsrTask,
        doc = """Task to perform instrumental signature removal or load a post-ISR image; ISR consists of:
            - assemble raw amplifier images into an exposure with image, variance and mask planes
            - perform bias subtraction, flat fielding, etc.
            - mask known bad pixels
            - provide a preliminary WCS
            """,
    )
    charImage = pexConfig.ConfigurableField(
        target = CharacterizeImageTask,
        doc = """Task to characterize a science exposure:
            - detect sources, usually at high S/N
            - estimate the background, which is subtracted from the image and returned as field "background"
            - estimate a PSF model, which is added to the exposure
            - interpolate over defects and cosmic rays, updating the image, variance and mask planes
            """,
    )
    doCalibrate = pexConfig.Field(
        dtype = bool,
        default = True,
        doc = "Perform calibration?",
    )
    calibrate = pexConfig.ConfigurableField(
        target = CalibrateTask,
        doc = """Task to perform astrometric and photometric calibration:
            - refine the WCS in the exposure
            - refine the Calib photometric calibration object in the exposure
            - detect sources, usually at low S/N
            """,
    )


## \addtogroup LSST_task_documentation
## \{
## \page ProcessCcdTask
## \ref ProcessCcdTask_ "ProcessCcdTask"
## \copybrief ProcessCcdTask
## \}

class ProcessCcdTask(pipeBase.CmdLineTask):
    """!Assemble raw data, detect and measure sources and fit background, PSF, WCS and zero-point

    @anchor ProcessCcdTask_
    
    @section pipe_tasks_processCcd_Contents  Contents

     - @ref pipe_tasks_processCcd_Purpose
     - @ref pipe_tasks_processCcd_Initialize
     - @ref pipe_tasks_processCcd_IO
     - @ref pipe_tasks_processCcd_Config
     - @ref pipe_tasks_processCcd_Debug
     - @ref pipe_tasks_processCcd_Example

    @section pipe_tasks_processCcd_Purpose  Description

    Perform the following operations:
    - Use `IsrTask` to unpersist raw data and assemble it into a post-ISR exposure
    - Use `CharacterizeImageTask` to model and subtract the background and model the PSF
    - Use `CalibrateTask` to detect, deblend and single-frame-measure sources
        and refine the WCS and fit photometric zero-point

    @section pipe_tasks_processCcd_Initialize  Task initialisation

    @copydoc \_\_init\_\_

    @section pipe_tasks_processCcd_IO  Invoking the Task

    This task is primarily designed to be run from the command line.

    The main method is `run`, which takes a single butler data reference for the raw input data.

    @section pipe_tasks_processCcd_Config  Configuration parameters

    See @ref ProcessCcdConfig

    @section pipe_tasks_processCcd_Debug  Debug variables

    ProcessCcdTask has no debug output, but its subtasks do.

    @section pipe_tasks_processCcd_Example   A complete example of using ProcessCcdTask

    The following command will process all visits in obs_test's data repository:
    `processCcd.py $OBS_TEST_DIR/data/input --output processCcdOut --id`
    """
    ConfigClass = ProcessCcdConfig
    _DefaultName = "processCcd"

    def __init__(self, **kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.makeSubtask("isr")
        self.makeSubtask("charImage")
        self.makeSubtask("calibrate")

    @pipeBase.timeMethod
    def run(self, sensorRef):
        """Process one CCD

        The sequence of operations is:
        - remove instrument signature
        - characterize image to estimate PSF and background
        - calibrate astrometry and photometry

        @param sensorRef: butler data reference for raw data

        @return pipe_base Struct containing these fields:
        - charRes: object returned by image characterization task; an lsst.pipe.base.Struct
            that will include "background" and "sourceCat" fields
        - calibRes: object returned by calibration task: an lsst.pipe.base.Struct
            that will include "background" and "sourceCat" fields
        - exposure: final exposure (an lsst.afw.image.ExposureF)
        - background: final background model (an lsst.afw.math.BackgroundList)
        """
        self.log.info("Processing %s" % (sensorRef.dataId))

        exposure = self.isr.runDataRef(sensorRef).exposure

        charRes = self.charImage.run(
            dataRef = sensorRef,
            exposure = exposure,
            doUnpersist = False,
        )
        exposure = charRes.exposure

        if self.config.doCalibrate:
            calibRes = self.calibrate.run(
                dataRef = sensorRef,
                exposure = charRes.exposure,
                background = charRes.background,
                doUnpersist = False,
            )

        return pipeBase.Struct(
            charRes = charRes,
            calibRes = calibRes if self.config.doCalibrate else None,
            exposure = exposure,
            background = calibRes.background if self.config.doCalibrate else charRes.background,
        )
