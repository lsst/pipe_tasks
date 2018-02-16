#
# LSST Data Management System
# Copyright 2008-2016 AURA/LSST.
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
from __future__ import absolute_import, division, print_function
from lsst.ip.isr import IsrTask
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from .calibrate import CalibrateTask
from .characterizeImage import CharacterizeImageTask

__all__ = ["ProcessCcdConfig", "ProcessCcdTask"]


class ProcessCcdConfig(pexConfig.Config):
    """Config for ProcessCcd"""
    isr = pexConfig.ConfigurableField(
        target=IsrTask,
        doc="""Task to perform instrumental signature removal or load a post-ISR image; ISR consists of:
            - assemble raw amplifier images into an exposure with image, variance and mask planes
            - perform bias subtraction, flat fielding, etc.
            - mask known bad pixels
            - provide a preliminary WCS
            """,
    )
    charImage = pexConfig.ConfigurableField(
        target=CharacterizeImageTask,
        doc="""Task to characterize a science exposure:
            - detect sources, usually at high S/N
            - estimate the background, which is subtracted from the image and returned as field "background"
            - estimate a PSF model, which is added to the exposure
            - interpolate over defects and cosmic rays, updating the image, variance and mask planes
            """,
    )
    doCalibrate = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Perform calibration?",
    )
    calibrate = pexConfig.ConfigurableField(
        target=CalibrateTask,
        doc="""Task to perform astrometric and photometric calibration:
            - refine the WCS in the exposure
            - refine the Calib photometric calibration object in the exposure
            - detect sources, usually at low S/N
            """,
    )

    def setDefaults(self):
        self.charImage.doWriteExposure = False
        self.charImage.detection.doTempLocalBackground = False
        self.calibrate.detection.doTempLocalBackground = False
        self.calibrate.deblend.maxFootprintSize = 2000

## \addtogroup LSST_task_documentation
## \{
## \page ProcessCcdTask
## \ref ProcessCcdTask_ "ProcessCcdTask"
## \copybrief ProcessCcdTask
## \}


class ProcessCcdTask(pipeBase.CmdLineTask):
    """!Assemble raw data, fit the PSF, detect and measure, and fit WCS and zero-point

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
    - Call isr to unpersist raw data and assemble it into a post-ISR exposure
    - Call charImage subtract background, fit a PSF model, repair cosmic rays,
        detect and measure bright sources, and measure aperture correction
    - Call calibrate to perform deep detection, deblending and single-frame measurement,
        refine the WCS and fit the photometric zero-point

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

    The following commands will process all raw data in obs_test's data repository.
    Note: be sure to specify an `--output` that does not already exist:

        setup obs_test
        setup pipe_tasks
        processCcd.py $OBS_TEST_DIR/data/input --output processCcdOut --id

    The data is read from the small repository in the `obs_test` package and written `./processCcdOut`
    (or whatever output you specified). Specifying `--id` with no values processes all data.
    Add the option `--help` to see more options.
    """
    ConfigClass = ProcessCcdConfig
    RunnerClass = pipeBase.ButlerInitializedTaskRunner
    _DefaultName = "processCcd"

    def __init__(self, butler=None, psfRefObjLoader=None, astromRefObjLoader=None, photoRefObjLoader=None,
                 **kwargs):
        """!
        @param[in] butler  The butler is passed to the refObjLoader constructor in case it is
            needed.  Ignored if the refObjLoader argument provides a loader directly.
        @param[in] psfRefObjLoader  An instance of LoadReferenceObjectsTasks that supplies an
            external reference catalog for image characterization.  An example of when this would
            be used is when a CatalogStarSelector is used.  May be None if the desired loader can
            be constructed from the butler argument or all steps requiring a catalog are disabled.
        @param[in] astromRefObjLoader  An instance of LoadReferenceObjectsTasks that supplies an
            external reference catalog for astrometric calibration.  May be None if the desired
            loader can be constructed from the butler argument or all steps requiring a reference
            catalog are disabled.
        @param[in] photoRefObjLoader  An instance of LoadReferenceObjectsTasks that supplies an
            external reference catalog for photometric calibration.  May be None if the desired
            loader can be constructed from the butler argument or all steps requiring a reference
            catalog are disabled.
        @param[in,out] kwargs  other keyword arguments for lsst.pipe.base.CmdLineTask
        """
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.makeSubtask("isr")
        self.makeSubtask("charImage", butler=butler, refObjLoader=psfRefObjLoader)
        self.makeSubtask("calibrate", butler=butler, icSourceSchema=self.charImage.schema,
                         astromRefObjLoader=astromRefObjLoader, photoRefObjLoader=photoRefObjLoader)

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
            dataRef=sensorRef,
            exposure=exposure,
            doUnpersist=False,
        )
        exposure = charRes.exposure

        if self.config.doCalibrate:
            calibRes = self.calibrate.run(
                dataRef=sensorRef,
                exposure=charRes.exposure,
                background=charRes.background,
                doUnpersist=False,
                icSourceCat=charRes.sourceCat,
            )

        return pipeBase.Struct(
            charRes=charRes,
            calibRes=calibRes if self.config.doCalibrate else None,
            exposure=exposure,
            background=calibRes.background if self.config.doCalibrate else charRes.background,
        )

    @classmethod
    def _makeArgumentParser(cls):
        """!Create and return an argument parser

        @param[in] cls      the class object
        @return the argument parser for this task.

        This override is used to delay making the data ref list until the dataset type is known;
        this is done in @ref parseAndRun.
        """
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id",
                               datasetType=pipeBase.ConfigDatasetType(name="isr.datasetType"),
                               help="data IDs, e.g. --id visit=12345 ccd=1,2^0,3")
        return parser
