#
# LSST Data Management System
# Copyright 2008-2017 AURA/LSST.
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
from .characterizeImage import CharacterizeImageTask
from .donutFit import DonutFitTask

__all__ = ["ProcessDonutConfig", "ProcessDonutTask"]


class ProcessDonutConfig(pexConfig.Config):
    """Config for ProcessDonut"""
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
    donutFit = pexConfig.ConfigurableField(
        target=DonutFitTask,
        doc="""Task to select and fit donuts:
            - Selects sources that look like donuts
            - Fit a wavefront forward model to donut images
            """,
    )

    def setDefaults(self):
        self.charImage.installSimplePsf.width = 61
        self.charImage.installSimplePsf.fwhm = 20.0
        self.charImage.detection.thresholdValue = 1.5
        self.charImage.doMeasurePsf = False
        self.charImage.doApCorr = False
        self.charImage.detection.doTempLocalBackground = False

## \addtogroup LSST_task_documentation
## \{
## \page ProcessDonutTask
## \ref ProcessDonutTask_ "ProcessDonutTask"
## \copybrief ProcessDonutTask
## \}


class ProcessDonutTask(pipeBase.CmdLineTask):
    """!Assemble raw data, detect and fit donuts

    @anchor ProcessDonutTask_

    @section pipe_tasks_processDonut_Contents  Contents

     - @ref pipe_tasks_processDonut_Purpose
     - @ref pipe_tasks_processDonut_Initialize
     - @ref pipe_tasks_processDonut_IO
     - @ref pipe_tasks_processDonut_Config
     - @ref pipe_tasks_processDonut_Debug

    @section pipe_tasks_processDonut_Purpose  Description

    Perform the following operations:
    - Call isr to unpersist raw data and assemble it into a post-ISR exposure
    - Call charImage subtract background, repair cosmic rays, and detect and measure bright sources
    - Call donutFit to fit Zernike wavefront models to donut images

    @section pipe_tasks_processDonut_Initialize  Task initialisation

    @copydoc \_\_init\_\_

    @section pipe_tasks_processDonut_IO  Invoking the Task

    This task is primarily designed to be run from the command line.

    The main method is `run`, which takes a single butler data reference for the raw input data.

    @section pipe_tasks_processDonut_Config  Configuration parameters

    See @ref processDonutConfig

    @section pipe_tasks_processDonut_Debug  Debug variables

    processDonutTask has no debug output, but its subtasks do.

    Add the option `--help` to see more options.
    """
    ConfigClass = ProcessDonutConfig
    RunnerClass = pipeBase.ButlerInitializedTaskRunner
    _DefaultName = "processDonut"

    def __init__(self, butler=None, psfRefObjLoader=None, **kwargs):
        """!
        @param[in] butler  The butler is passed to the refObjLoader constructor in case it is
            needed.  Ignored if the refObjLoader argument provides a loader directly.
        @param[in] psfRefObjLoader  An instance of LoadReferenceObjectsTasks that supplies an
            external reference catalog for image characterization.  An example of when this would
            be used is when a CatalogStarSelector is used.  May be None if the desired loader can
            be constructed from the butler argument or all steps requiring a catalog are disabled.
        @param[in,out] kwargs  other keyword arguments for lsst.pipe.base.CmdLineTask
        """
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.makeSubtask("isr")
        self.makeSubtask("charImage", butler=butler, refObjLoader=psfRefObjLoader)
        self.makeSubtask("donutFit", schema=self.charImage.schema)

    @pipeBase.timeMethod
    def run(self, sensorRef):
        """Process donuts on one CCD

        The sequence of operations is:
        - remove instrument signature
        - characterize image to estimate background and do detection
        - fit donuts

        @param sensorRef: butler data reference for raw data

        @return pipe_base Struct containing these fields:
        - donutCat : object returned by donut fitting task
        """
        self.log.info("Processing %s" % (sensorRef.dataId))

        exposure = self.isr.runDataRef(sensorRef).exposure

        charRes = self.charImage.run(
            dataRef=sensorRef,
            exposure=exposure,
            doUnpersist=False,
        )
        exposure = charRes.exposure

        donutRes = self.donutFit.run(exposure, charRes.sourceCat)

        sensorRef.put(donutRes.donutCat, "donutCat")

        return donutRes

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

    # Override/kill config/schema/meta/version output for the moment
    def writeConfig(self, butler, clobber=False, doBackup=True):
        pass
    def writeSchemas(self, butler, clobber=False, doBackup=True):
        pass
    def writeMetadata(self, dataRef):
        pass
    def writePackageVersions(self, butler, clobber=False, doBackup=True, dataset="packages"):
        pass
