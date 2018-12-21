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
import lsst.pex.config as pexConfig
import lsst.afw.math as afwMath
import lsst.afw.detection as afwDet
import lsst.meas.algorithms as measAlg
import lsst.pipe.base as pipeBase
from lsstDebug import getDebugFrame
from lsst.afw.display import getDisplay
from lsst.pipe.tasks.interpImage import InterpImageTask

__all__ = ['RepairConfig', 'RepairTask']


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
    """Repair image defects and find and repair cosmic rays in an exposure.
    
    This task operates on an exposure in place to interpolate over a list of
    lsst.meas.algorithms.Defect objects. It will also, optionally, find and
    interpolate any cosmic rays.
    """
    ConfigClass = RepairConfig
    _DefaultName = "repair"

    def __init__(self, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)
        if self.config.doInterpolate:
            self.makeSubtask("interp")

    @pipeBase.timeMethod
    def run(self, exposure, defects=None, keepCRs=None):
        """Repair an exposure's defects and cosmic rays.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure with a valid Psf, which will be modified in place.
        defects : `list` of `lsst.meas.algorithms.Defect`, optional
            If None, do no defect correction.
        keepCRs : `bool`, optional
            If True, don't interpolate over the CR pixels (keep them).
            If False, interpolate over the CR pixels (don't keep them).
            If None, defer to the setting in RepairConfig.cosmicray.
        """
        assert exposure, "No exposure provided"
        psf = exposure.getPsf()
        assert psf, "No PSF provided"

        frame = getDebugFrame(self._display, "repair.before")
        if frame:
            getDisplay(frame).mtv(exposure)

        if defects is not None and self.config.doInterpolate:
            self.interp.run(exposure, defects=defects)

        if self.config.doCosmicRay:
            self.cosmicRay(exposure, keepCRs=keepCRs)

        frame = getDebugFrame(self._display, "repair.after")
        if frame:
            getDisplay(frame).mtv(exposure)

    def cosmicRay(self, exposure, keepCRs=None):
        """Mask cosmic rays.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to process
        keepCRs : `bool`, optional
            If True, don't interpolate over the CR pixels (keep them).
            If False, interpolate over the CR pixels (don't keep them).
            If None, defer to the setting in RepairConfig.cosmicray.
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
                                         pexConfig.makePolicy(self.config.cosmicray), keepCRs)
            if modelBg:
                # Add back background image
                img = exposure.getMaskedImage()
                img += modelBg.getImageF()
                del img
                # Replace original image with CR subtracted image
                exposure0.setMaskedImage(exposure.getMaskedImage())

        except Exception:
            if display:
                import lsst.afw.display.ds9 as ds9
                ds9.mtv(exposure0, title="Failed CR")
            raise

        num = 0
        if crs is not None:
            mask = exposure0.getMaskedImage().getMask()
            crBit = mask.getPlaneBitMask("CR")
            afwDet.setMaskFromFootprintList(mask, crs, crBit)
            num = len(crs)

            if display and displayCR:
                import lsst.afw.display.ds9 as ds9
                import lsst.afw.display.utils as displayUtils

                ds9.incrDefaultFrame()
                ds9.mtv(exposure0, title="Post-CR")

                with ds9.Buffering():
                    for cr in crs:
                        displayUtils.drawBBox(cr.getBBox(), borderWidth=0.55)

        self.log.info("Identified %s cosmic rays." % (num,))
