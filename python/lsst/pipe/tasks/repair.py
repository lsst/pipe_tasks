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

__all__ = ["RepairConfig", "RepairTask", "TooManyCosmicRays"]

import lsst.pex.config as pexConfig
import lsst.afw.math as afwMath
import lsst.afw.detection as afwDet
import lsst.meas.algorithms as measAlg
import lsst.pipe.base as pipeBase
from lsstDebug import getDebugFrame
import lsst.afw.display as afwDisplay
from lsst.pipe.tasks.interpImage import InterpImageTask
from lsst.utils.timer import timeMethod
from lsst.pex.exceptions import LengthError


class TooManyCosmicRays(pipeBase.AlgorithmError):
    """Raised if the cosmic ray task fails with too many cosmics.

    Parameters
    ----------
    maxCosmicRays : `int`
        Maximum number of cosmic rays allowed.
    """
    def __init__(self, maxCosmicRays, **kwargs):
        msg = f"Cosmic ray task found more than {maxCosmicRays} cosmics."
        self.msg = msg
        self._metadata = kwargs
        super().__init__(msg, **kwargs)
        self._metadata["maxCosmicRays"] = maxCosmicRays

    def __str__(self):
        # Exception doesn't handle **kwargs, so we need a custom str.
        return f"{self.msg}: {self.metadata}"

    @property
    def metadata(self):
        for key, value in self._metadata.items():
            if not (isinstance(value, int) or isinstance(value, float) or isinstance(value, str)):
                raise TypeError(f"{key} is of type {type(value)}, but only (int, float, str) are allowed.")
        return self._metadata


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


class RepairTask(pipeBase.Task):
    """Repair an exposures defects and cosmic rays via interpolation.

    This task operates on an lsst.afw.image.Exposure in place to
    interpolate over a set of `~lsst.meas.algorithms.Defect` objects.

    It will also, optionally, find and interpolate any cosmic rays in the lsst.afw.image.Exposure.

    Notes
    -----
    Debugging:
    The available debug variables in RepairTask are:

    display :
        A dictionary containing debug point names as keys with frame number as value. Valid keys are:
    repair.before :
        display image before any repair is done
    repair.after :
        display image after cosmic ray and defect correction
    displayCR :
        If True, display the exposure on ds9's frame 1 and overlay bounding boxes around detects CRs.

    To investigate the pipe_tasks_repair_Debug, put something like

    .. code-block :: none

        import lsstDebug
        def DebugInfo(name):
            di = lsstDebug.getInfo(name)        # N.b. lsstDebug.Info(name) would call us recursively
            if name == "lsst.pipe.tasks.repair":
                di.display = {'repair.before':2, 'repair.after':3}
                di.displayCR = True
            return di

    lsstDebug.Info = DebugInfo
    into your debug.py file and run runRepair.py with the --debug flag.

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
        """Repair an Exposure's defects and cosmic rays.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure must have a valid Psf.
            Modified in place.
        defects : `lsst.meas.algorithms.DefectListT` or `None`, optional
            If `None`, do no defect correction.
        keepCRs : `Unknown` or `None`, optional
            Don't interpolate over the CR pixels (defer to ``RepairConfig`` if `None`).

        Raises
        ------
        AssertionError
            Raised if any of the following occur:
            - No exposure provided.
            - The object provided as exposure evaluates to False.
            - No PSF provided.
            - The Exposure has no associated Psf.
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
        """Mask cosmic rays.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to process.
        keepCRs : `Unknown` or `None`, optional
            Don't interpolate over the CR pixels (defer to ``pex_config`` if `None`).
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
            try:
                crs = measAlg.findCosmicRays(
                    exposure.getMaskedImage(),
                    psf,
                    medianBg,
                    pexConfig.makePropertySet(self.config.cosmicray),
                    keepCRs,
                )
            except LengthError:
                raise TooManyCosmicRays(self.config.cosmicray.nCrPixelMax) from None

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

        text = "kept" if keepCRs else "interpolated over"
        self.log.info("Identified and %s %s cosmic rays.", text, num)
        self.metadata["cosmic_ray_count"] = num
