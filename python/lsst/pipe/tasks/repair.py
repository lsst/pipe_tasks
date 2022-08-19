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

import lsst.pex.config as pexConfig
import lsst.afw.math as afwMath
import lsst.afw.detection as afwDet
import lsst.meas.algorithms as measAlg
import lsst.pipe.base as pipeBase
from lsstDebug import getDebugFrame
import lsst.afw.display as afwDisplay
from lsst.pipe.tasks.interpImage import InterpImageTask
from lsst.utils.timer import timeMethod


class RepairConfig(pexConfig.Config):
    doInterpolate = pexConfig.Field(
        dtype=bool,
        doc="Interpolate over defects? (ignored unless you provide a list of defects)",
        default=True,
    )
    doCosmicRay = pexConfig.Field(
        dtype=bool,
        doc="Find and mask out cosmic rays?",
        default=True,
    )
    cosmicray = pexConfig.ConfigField(
        dtype=measAlg.FindCosmicRaysConfig,
        doc="Options for finding and masking cosmic rays",
    )
    interp = pexConfig.ConfigurableField(
        target=InterpImageTask,
        doc="Interpolate over bad image pixels",
    )

    def setDefaults(self):
        self.interp.useFallbackValueAtEdge = True
        self.interp.fallbackValueType = "MEANCLIP"
        self.interp.negativeFallbackAllowed = True

## @addtogroup LSST_task_documentation
## @{
## @page page_RepairTask RepairTask
## @ref RepairTask_ "RepairTask"
## @copybrief RepairTask
## @}


class RepairTask(pipeBase.Task):
    r"""!
    @anchor RepairTask_

    @brief Interpolate over defects in an exposure and handle cosmic rays

    @section pipe_tasks_repair_Contents Contents

     - @ref pipe_tasks_repair_Purpose
     - @ref pipe_tasks_repair_Initialize
     - @ref pipe_tasks_repair_IO
     - @ref pipe_tasks_repair_Config
     - @ref pipe_tasks_repair_Debug
     - @ref pipe_tasks_repair_Example

    @section pipe_tasks_repair_Purpose Description

    @copybrief RepairTask

    This task operates on an lsst.afw.image.Exposure in place to interpolate over a set of
    lsst.meas.algorithms.Defect objects.
    It will also, optionally, find and interpolate any cosmic rays in the lsst.afw.image.Exposure.

    @section pipe_tasks_repair_Initialize Task initialization

    See: lsst.pipe.base.task.Task.__init__

    @section pipe_tasks_repair_IO Inputs/Outputs to the run method

    @copydoc run

    @section pipe_tasks_repair_Config Configuration parameters

    See @ref RepairConfig

    @section pipe_tasks_repair_Debug Debug variables

    The command line task interface supports a
    flag @c -d to import @b debug.py from your @c PYTHONPATH; see <a
    href="https://developer.lsst.io/stack/debug.html">Debugging Tasks with lsstDebug</a> for more
    about @b debug.py files.

    The available variables in RepairTask are:
    <DL>
      <DT> @c display
      <DD> A dictionary containing debug point names as keys with frame number as value. Valid keys are:
        <DL>
          <DT> repair.before
          <DD> display image before any repair is done
          <DT> repair.after
          <DD> display image after cosmic ray and defect correction
        </DL>
      <DT> @c displayCR
      <DD> If True, display the exposure on display's frame 1 and overlay bounding boxes around detects CRs.
    </DL>
    @section pipe_tasks_repair_Example A complete example of using RepairTask

    This code is in runRepair.py in the examples directory, and can be run as @em e.g.
    @code
    examples/runRepair.py
    @endcode
    @dontinclude runRepair.py
    Import the task.  There are other imports.  Read the source file for more info.
    @skipline RepairTask

    For this example, we manufacture a test image to run on.

    First, create a pure Poisson noise image and a Psf to go with it.  The mask plane
    and variance can be constructed at the same time.
    @skip poisson
    @until mask

    Inject some cosmic rays and generate the Exposure.  Exposures are MaskedImages (image + mask + variance)
    with other metadata (e.g. Psf and Wcs objects).
    @skip some CRs
    @until setPsf

    Defects are represented as bad columns of random lengths.  A defect list must be constructed to pass
    on to the RepairTask.
    @bug This is addressed in <a href="https://jira.lsstcorp.org/browse/DM-963"> DM-963</a>

    @skip addDefects
    @until push_back

    Finally, the exposure can be repaired.  Create an instance of the task and run it.  The exposure
    is modified in place.
    @skip RepairTask
    @until repair.run

    <HR>
    To investigate the @ref pipe_tasks_repair_Debug, put something like
    @code{.py}
    import lsstDebug
    def DebugInfo(name):
        di = lsstDebug.getInfo(name)        # N.b. lsstDebug.Info(name) would call us recursively
        if name == "lsst.pipe.tasks.repair":
            di.display = {'repair.before':2, 'repair.after':3}
            di.displayCR = True
        return di

    lsstDebug.Info = DebugInfo
    @endcode
    into your debug.py file and run runRepair.py with the @c --debug flag.


    Conversion notes:
        Display code should be updated once we settle on a standard way of controlling what is displayed.
    """
    ConfigClass = RepairConfig
    _DefaultName = "repair"

    def __init__(self, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)
        if self.config.doInterpolate:
            self.makeSubtask("interp")

    @timeMethod
    def run(self, exposure, defects=None, keepCRs=None):
        """!Repair an Exposure's defects and cosmic rays

        Parameters
        ----------
        exposure : `Unknown`
            lsst.afw.image.Exposure to process.  Exposure must have a valid Psf.
                                  Modified in place.
        Parameters
        ----------
        defects : `Unknown`
            an lsst.meas.algorithms.DefectListT object.  If None, do no
                                  defect correction.
        Parameters
        ----------
        keepCRs : `Unknown`
            don't interpolate over the CR pixels (defer to RepairConfig if None)

        @throws AssertionError with the following strings:

        <DL>
          <DT> No exposure provided
          <DD> The object provided as exposure evaluates to False
          <DT> No PSF provided
          <DD> The Exposure has no associated Psf
        </DL>
        """
        assert exposure, "No exposure provided"
        psf = exposure.getPsf()
        assert psf, "No PSF provided"

        frame = getDebugFrame(self._display, "repair.before")
        if frame:
            afwDisplay.Display(frame).mtv(exposure)

        if defects is not None and self.config.doInterpolate:
            self.interp.run(exposure, defects=defects)

        if self.config.doCosmicRay:
            self.cosmicRay(exposure, keepCRs=keepCRs)

        frame = getDebugFrame(self._display, "repair.after")
        if frame:
            afwDisplay.Display(frame).mtv(exposure)

    def cosmicRay(self, exposure, keepCRs=None):
        """Mask cosmic rays

        Parameters
        ----------
        exposure : `Unknown`
            Exposure to process
        Parameters
        ----------
        keepCRs : `Unknown`
            Don't interpolate over the CR pixels (defer to pex_config if None)
        """
        import lsstDebug
        display = lsstDebug.Info(__name__).display
        displayCR = lsstDebug.Info(__name__).displayCR

        assert exposure, "No exposure provided"
        psf = exposure.getPsf()
        assert psf, "No psf provided"

        # Blow away old mask
        try:
            mask = exposure.getMaskedImage().getMask()
            crBit = mask.getMaskPlane("CR")
            mask.clearMaskPlane(crBit)
        except Exception:
            pass

        exposure0 = exposure            # initial value of exposure
        binSize = self.config.cosmicray.background.binSize
        nx, ny = exposure.getWidth()/binSize, exposure.getHeight()/binSize
        # Treat constant background as a special case to avoid the extra complexity in calling
        # measAlg.SubtractBackgroundTask().
        if nx*ny <= 1:
            medianBg = afwMath.makeStatistics(exposure.getMaskedImage(), afwMath.MEDIAN).getValue()
            modelBg = None
        else:
            # make a deep copy of the exposure before subtracting its background,
            # because this routine is only allowed to modify the exposure by setting mask planes
            # and interpolating over defects, not changing the background level
            exposure = exposure.Factory(exposure, True)
            subtractBackgroundTask = measAlg.SubtractBackgroundTask(config=self.config.cosmicray.background)
            modelBg = subtractBackgroundTask.run(exposure).background
            medianBg = 0.0

        if keepCRs is None:
            keepCRs = self.config.cosmicray.keepCRs
        try:
            crs = measAlg.findCosmicRays(exposure.getMaskedImage(), psf, medianBg,
                                         pexConfig.makePropertySet(self.config.cosmicray), keepCRs)
            if modelBg:
                # Add back background image
                img = exposure.getMaskedImage()
                img += modelBg.getImageF()
                del img
                # Replace original image with CR subtracted image
                exposure0.setMaskedImage(exposure.getMaskedImage())

        except Exception:
            if display:
                afwDisplay.Display().mtv(exposure0, title="Failed CR")
            raise

        num = 0
        if crs is not None:
            mask = exposure0.getMaskedImage().getMask()
            crBit = mask.getPlaneBitMask("CR")
            afwDet.setMaskFromFootprintList(mask, crs, crBit)
            num = len(crs)

            if display and displayCR:
                disp = afwDisplay.Display()
                disp.incrDefaultFrame()
                disp.mtv(exposure0, title="Post-CR")

                with disp.Buffering():
                    for cr in crs:
                        afwDisplay.utils.drawBBox(cr.getBBox(), borderWidth=0.55)

        self.log.info("Identified %s cosmic rays.", num)
