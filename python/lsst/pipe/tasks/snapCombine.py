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

__all__ = ["InitialPsfConfig", "SnapCombineConfig", "SnapCombineTask"]

import numpy as num
import lsst.pex.config as pexConfig
import lsst.daf.base as dafBase
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pipe.base as pipeBase
from lsstDebug import getDebugFrame
from lsst.afw.display import getDisplay
from lsst.coadd.utils import addToCoadd, setCoaddEdgeBits
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.base import SingleFrameMeasurementTask
import lsst.meas.algorithms as measAlg
from lsst.utils.timer import timeMethod

from .repair import RepairTask


class InitialPsfConfig(pexConfig.Config):
    """Describes the initial PSF used for detection and measurement before we do PSF determination."""

    model = pexConfig.ChoiceField(
        dtype=str,
        doc="PSF model to install in snaps.",
        default="SingleGaussian",
        allowed={
            "SingleGaussian": "Single Gaussian model",
            "DoubleGaussian": "Double Gaussian model",
        },
    )
    fwhm = pexConfig.Field(
        dtype=float,
        doc="FWHM of PSF model (arcsec)",
        default=1.0,
    )
    size = pexConfig.Field(
        dtype=int,
        doc="Size of PSF model (pixels)",
        default=15,
    )


class SnapCombineConfig(pexConfig.Config):
    doRepair = pexConfig.Field(
        dtype=bool,
        doc="Repair images (CR reject and interpolate) before combining",
        default=True,
    )
    repairPsfFwhm = pexConfig.Field(
        dtype=float,
        doc="Psf FWHM (pixels) used to detect CRs",
        default=2.5,
    )
    doDiffIm = pexConfig.Field(
        dtype=bool,
        doc="Perform difference imaging before combining",
        default=False,
    )
    doMeasurement = pexConfig.Field(
        dtype=bool,
        doc="Measure difference sources (ignored if doDiffIm false)",
        default=True,
    )
    badMaskPlanes = pexConfig.ListField(
        dtype=str,
        doc="Mask planes that, if set, the associated pixels are not included in the combined exposure; "
        "DETECTED excludes cosmic rays",
        default=("DETECTED",),
    )
    averageKeys = pexConfig.ListField(
        dtype=str,
        doc="List of float metadata keys to average when combining snaps, e.g. float positions and dates; "
        "non-float data must be handled by overriding the fixMetadata method",
        optional=True,
    )
    sumKeys = pexConfig.ListField(
        dtype=str,
        doc="List of float or int metadata keys to sum when combining snaps, e.g. exposure time; "
        "non-float, non-int data must be handled by overriding the fixMetadata method",
        optional=True,
    )
    initialPsf = pexConfig.ConfigField(
        dtype=InitialPsfConfig,
        doc="Task to generate an initial PSF before repair."
    )
    repair = pexConfig.ConfigurableField(
        target=RepairTask,
        doc="Task to repair individual snaps before combining."
    )
    detection = pexConfig.ConfigurableField(
        target=SourceDetectionTask,
        doc="Task to perform detection on a difference image, if doDiffIm is True."
    )
    measurement = pexConfig.ConfigurableField(
        target=SingleFrameMeasurementTask,
        doc="Task to perform measurement on difference image, if doDiffIm is True."
    )

    def setDefaults(self):
        self.detection.thresholdPolarity = "both"

    def validate(self):
        if self.detection.thresholdPolarity != "both":
            raise ValueError("detection.thresholdPolarity must be 'both' for SnapCombineTask")


class SnapCombineTask(pipeBase.Task):
    """Combine two snaps into a single visit image.

    Notes
    -----
    Debugging:
    The `~lsst.base.lsstDebug` variables in SnapCombineTask are:

    display
    A dictionary containing debug point names as keys with frame number as value. Valid keys are:

    .. code-block:: none

        repair0
        Display the first snap after repairing.
        repair1
        Display the second snap after repairing.
    """

    ConfigClass = SnapCombineConfig
    _DefaultName = "snapCombine"

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("repair")
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        if self.config.doDiffIm:
            self.makeSubtask("detection", schema=self.schema)
            if self.config.doMeasurement:
                self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)

    @timeMethod
    def run(self, snap0, snap1, defects=None):
        """Combine two snaps, returning the combined image and possibly doing
        simple image subtraction on the two.

        Parameters
        ----------
        snap0 : `lsst.afw.image.Exposure`
            Snapshot exposure 0.
        snap1 : `lsst.afw.image.Exposure`
            Snapshot exposure 1.
        defects : `list` or `None`, optional
            Defect list (for repair task).

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``exposure``
                Snap-combined exposure.
            ``sources``
                Detected sources, or `None` if detection not performed.
        """
        # initialize optional outputs
        sources = None

        if self.config.doRepair:
            psf = self.makeInitialPsf(snap0, fwhmPix=self.config.repairPsfFwhm)
            snap0.setPsf(psf)
            snap1.setPsf(psf)
            self.log.info("Repairing snap image 1")
            self.repair.run(snap0, defects=defects, keepCRs=False)
            self.log.info("Repairing snap image 2")
            self.repair.run(snap1, defects=defects, keepCRs=False)

            repair0frame = getDebugFrame(self._display, "repair0")
            if repair0frame:
                getDisplay(repair0frame).mtv(snap0)
            repair1frame = getDebugFrame(self._display, "repair1")
            if repair1frame:
                getDisplay(repair1frame).mtv(snap1)

        if self.config.doDiffIm:
            diffExp = afwImage.ExposureF(snap0, True)
            diffMi = diffExp.maskedImage
            diffMi -= snap1.maskedImage

            psf = self.makeInitialPsf(snap0)
            diffExp.setPsf(psf)
            table = afwTable.SourceTable.make(self.schema)
            table.setMetadata(self.algMetadata)
            detRet = self.detection.run(table, diffExp)
            sources = detRet.sources
            if self.config.doMeasurement:
                self.measurement.measure(diffExp, sources)

            mask0 = snap0.mask
            mask1 = snap1.mask
            detRet.positive.setMask(mask0, "DETECTED")
            detRet.negative.setMask(mask1, "DETECTED")

            maskD = diffExp.mask
            detRet.positive.setMask(maskD, "DETECTED")
            detRet.negative.setMask(maskD, "DETECTED_NEGATIVE")

        combinedExp = self.addSnaps(snap0, snap1)

        return pipeBase.Struct(
            exposure=combinedExp,
            sources=sources,
        )

    def addSnaps(self, snap0, snap1):
        """Add two snap exposures together, returning a new exposure.

        Parameters
        ----------
        snap0 : `lsst.afw.image.Exposure`
            Snap exposure 0.
        snap1 : `lsst.afw.image.Exposure`
            Snap exposure 1.

        Returns
        -------
        combinedExp : `lsst.afw.image.Exposure`
            Combined exposure.
        """
        combinedExp = snap0.Factory(snap0, True)
        combinedMi = combinedExp.maskedImage
        combinedMi.set(0)

        weightMap = combinedMi.image.Factory(combinedMi.getBBox())
        weight = 1.0
        badPixelMask = afwImage.Mask.getPlaneBitMask(self.config.badMaskPlanes)
        addToCoadd(combinedMi, weightMap, snap0.maskedImage, badPixelMask, weight)
        addToCoadd(combinedMi, weightMap, snap1.maskedImage, badPixelMask, weight)

        # pre-scaling the weight map instead of post-scaling the combinedMi saves a bit of time
        # because the weight map is a simple Image instead of a MaskedImage
        weightMap *= 0.5  # so result is sum of both images, instead of average
        combinedMi /= weightMap
        setCoaddEdgeBits(combinedMi.getMask(), weightMap)

        # note: none of the inputs has a valid PhotoCalib object, so that is not touched
        # Filter was already copied

        combinedMetadata = combinedExp.getMetadata()
        metadata0 = snap0.getMetadata()
        metadata1 = snap1.getMetadata()
        self.fixMetadata(combinedMetadata, metadata0, metadata1)

        return combinedExp

    def fixMetadata(self, combinedMetadata, metadata0, metadata1):
        """Fix the metadata of the combined exposure (in place).

        This implementation handles items specified by config.averageKeys and config.sumKeys,
        which have data type restrictions. To handle other data types (such as sexagesimal
        positions and ISO dates) you must supplement this method with your own code.

        Parameters
        ----------
        combinedMetadata : `lsst.daf.base.PropertySet`
            Metadata of combined exposure;
            on input this is a deep copy of metadata0 (a PropertySet).
        metadata0 : `lsst.daf.base.PropertySet`
            Metadata of snap0 (a PropertySet).
        metadata1 : `lsst.daf.base.PropertySet`
            Metadata of snap1 (a PropertySet).

        Notes
        -----
        The inputs are presently PropertySets due to ticket #2542. However, in some sense
        they are just PropertyLists that are missing some methods. In particular: comments and order
        are preserved if you alter an existing value with set(key, value).
        """
        keyDoAvgList = []
        if self.config.averageKeys:
            keyDoAvgList += [(key, 1) for key in self.config.averageKeys]
        if self.config.sumKeys:
            keyDoAvgList += [(key, 0) for key in self.config.sumKeys]
        for key, doAvg in keyDoAvgList:
            opStr = "average" if doAvg else "sum"
            try:
                val0 = metadata0.getScalar(key)
                val1 = metadata1.getScalar(key)
            except Exception:
                self.log.warning("Could not %s metadata %r: missing from one or both exposures", opStr, key)
                continue

            try:
                combinedVal = val0 + val1
                if doAvg:
                    combinedVal /= 2.0
            except Exception:
                self.log.warning("Could not %s metadata %r: value %r and/or %r not numeric",
                                 opStr, key, val0, val1)
                continue

            combinedMetadata.set(key, combinedVal)

    def makeInitialPsf(self, exposure, fwhmPix=None):
        """Initialise the detection procedure by setting the PSF and WCS.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to process.

        Returns
        -------
        psf : `Unknown`
            PSF, WCS

        AssertionError
            Raised if any of the following occur:
            - No exposure provided.
            - No wcs in exposure.
        """
        assert exposure, "No exposure provided"
        wcs = exposure.getWcs()
        assert wcs, "No wcs in exposure"

        if fwhmPix is None:
            fwhmPix = self.config.initialPsf.fwhm / wcs.getPixelScale().asArcseconds()

        size = self.config.initialPsf.size
        model = self.config.initialPsf.model
        self.log.info("Initial snap PSF fwhm=%s pixels; size=%s pixels", fwhmPix, size)
        psfCls = getattr(measAlg, model + "Psf")
        psf = psfCls(size, size, fwhmPix/(2.0*num.sqrt(2*num.log(2.0))))
        return psf
