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

__all__ = ["CharacterizeImageConfig", "CharacterizeImageTask"]

import numpy as np

from lsstDebug import getDebugFrame
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
import lsst.pipe.base.connectionTypes as cT
from lsst.afw.math import BackgroundList
from lsst.afw.table import SourceTable
from lsst.meas.algorithms import (
    SubtractBackgroundTask,
    SourceDetectionTask,
    MeasureApCorrTask,
    MeasureApCorrError,
    MaskStreaksTask,
    NormalizedCalibrationFluxTask,
)
from lsst.meas.algorithms.installGaussianPsf import InstallGaussianPsfTask
from lsst.meas.astrom import displayAstrometry
from lsst.meas.base import (
    SingleFrameMeasurementTask,
    ApplyApCorrTask,
    CatalogCalculationTask,
    IdGenerator,
    DetectorVisitIdGeneratorConfig,
)
from lsst.meas.deblender import SourceDeblendTask
import lsst.meas.extensions.shapeHSM  # noqa: F401 needed for default shape plugin
from .measurePsf import MeasurePsfTask
from .repair import RepairTask
from .computeExposureSummaryStats import ComputeExposureSummaryStatsTask
from lsst.pex.exceptions import LengthError
from lsst.utils.timer import timeMethod


class CharacterizeImageConnections(pipeBase.PipelineTaskConnections,
                                   dimensions=("instrument", "visit", "detector")):
    exposure = cT.Input(
        doc="Input exposure data",
        name="postISRCCD",
        storageClass="Exposure",
        dimensions=["instrument", "exposure", "detector"],
    )
    characterized = cT.Output(
        doc="Output characterized data.",
        name="icExp",
        storageClass="ExposureF",
        dimensions=["instrument", "visit", "detector"],
    )
    sourceCat = cT.Output(
        doc="Output source catalog.",
        name="icSrc",
        storageClass="SourceCatalog",
        dimensions=["instrument", "visit", "detector"],
    )
    backgroundModel = cT.Output(
        doc="Output background model.",
        name="icExpBackground",
        storageClass="Background",
        dimensions=["instrument", "visit", "detector"],
    )
    outputSchema = cT.InitOutput(
        doc="Schema of the catalog produced by CharacterizeImage",
        name="icSrc_schema",
        storageClass="SourceCatalog",
    )

    def adjustQuantum(self, inputs, outputs, label, dataId):
        # Docstring inherited from PipelineTaskConnections
        try:
            return super().adjustQuantum(inputs, outputs, label, dataId)
        except pipeBase.ScalarError as err:
            raise pipeBase.ScalarError(
                "CharacterizeImageTask can at present only be run on visits that are associated with "
                "exactly one exposure.  Either this is not a valid exposure for this pipeline, or the "
                "snap-combination step you probably want hasn't been configured to run between ISR and "
                "this task (as of this writing, that would be because it hasn't been implemented yet)."
            ) from err


class CharacterizeImageConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=CharacterizeImageConnections):
    """Config for CharacterizeImageTask."""

    doMeasurePsf = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Measure PSF? If False then for all subsequent operations use either existing PSF "
            "model when present, or install simple PSF model when not (see installSimplePsf "
            "config options)"
    )
    doWrite = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Persist results?",
    )
    doWriteExposure = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Write icExp and icExpBackground in addition to icSrc? Ignored if doWrite False.",
    )
    psfIterations = pexConfig.RangeField(
        dtype=int,
        default=2,
        min=1,
        doc="Number of iterations of detect sources, measure sources, "
            "estimate PSF. If useSimplePsf is True then 2 should be plenty; "
            "otherwise more may be wanted.",
    )
    maxUnNormPsfEllipticityPerBand = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        default={
            "u": 3.8,
            "g": 3.8,
            "r": 3.8,
            "i": 3.8,
            "z": 3.8,
            "y": 3.8,
        },
        doc="Maximum unnormalized ellipticity (defined as hypot(Ixx - Iyy, 2Ixy)) of the PSF model "
            "deemed good enough for further consideration.  Values above this threshold raise "
            "UnprocessableDataError.",
    )
    maxUnNormPsfEllipticityFallback = pexConfig.Field(
        dtype=float,
        default=3.8,
        doc="Fallback maximum unnormalized ellipticity (defined as hypot(Ixx - Iyy, 2Ixy)) of the "
            "PSF model deemed good enough for further consideration if the current band is not in "
            "the config.maxUnNormPsfEllipticityPerBand dict.  Values above this threshold "
            "raise UnprocessableDataError.",
    )
    maxPsfEllipticityPerBand = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        default={
            "u": 0.33,
            "g": 0.32,
            "r": 0.35,
            "i": 0.35,
            "z": 0.37,
            "y": 0.32,
        },
        doc="Value of the PSF model ellipticity deemed good enough for further consideration, "
            "regardless of the value of the unnormalized PSF model ellipticity. Values above "
            "this threshold raise UnprocessableDataError.",
    )
    maxPsfEllipticityFallback = pexConfig.Field(
        dtype=float,
        default=0.35,
        doc="Fallback maximum ellipticity of the PSF model deemed good enough for further "
            "consideration if the current band is not in the config.maxPsfEllipticityPerBand "
            "dict.  Values above this threshold raise UnprocessableDataError.",
    )

    background = pexConfig.ConfigurableField(
        target=SubtractBackgroundTask,
        doc="Configuration for initial background estimation",
    )
    detection = pexConfig.ConfigurableField(
        target=SourceDetectionTask,
        doc="Detect sources"
    )
    doDeblend = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Run deblender input exposure"
    )
    deblend = pexConfig.ConfigurableField(
        target=SourceDeblendTask,
        doc="Split blended source into their components"
    )
    measurement = pexConfig.ConfigurableField(
        target=SingleFrameMeasurementTask,
        doc="Measure sources"
    )
    doNormalizedCalibration = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Use normalized calibration flux (e.g. compensated tophats)?",
    )
    normalizedCalibrationFlux = pexConfig.ConfigurableField(
        target=NormalizedCalibrationFluxTask,
        doc="Task to normalize the calibration flux (e.g. compensated tophats).",
    )
    doApCorr = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Run subtasks to measure and apply aperture corrections"
    )
    measureApCorr = pexConfig.ConfigurableField(
        target=MeasureApCorrTask,
        doc="Subtask to measure aperture corrections"
    )
    applyApCorr = pexConfig.ConfigurableField(
        target=ApplyApCorrTask,
        doc="Subtask to apply aperture corrections"
    )
    # If doApCorr is False, and the exposure does not have apcorrections already applied, the
    # active plugins in catalogCalculation almost certainly should not contain the characterization plugin
    catalogCalculation = pexConfig.ConfigurableField(
        target=CatalogCalculationTask,
        doc="Subtask to run catalogCalculation plugins on catalog"
    )
    doComputeSummaryStats = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Run subtask to measure exposure summary statistics",
        deprecated=("This subtask has been moved to CalibrateTask "
                    "with DM-30701.")
    )
    computeSummaryStats = pexConfig.ConfigurableField(
        target=ComputeExposureSummaryStatsTask,
        doc="Subtask to run computeSummaryStats on exposure",
        deprecated=("This subtask has been moved to CalibrateTask "
                    "with DM-30701.")
    )
    useSimplePsf = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Replace the existing PSF model with a simplified version that has the same sigma "
        "at the start of each PSF determination iteration? Doing so makes PSF determination "
        "converge more robustly and quickly.",
    )
    installSimplePsf = pexConfig.ConfigurableField(
        target=InstallGaussianPsfTask,
        doc="Install a simple PSF model",
    )
    measurePsf = pexConfig.ConfigurableField(
        target=MeasurePsfTask,
        doc="Measure PSF",
    )
    repair = pexConfig.ConfigurableField(
        target=RepairTask,
        doc="Remove cosmic rays",
    )
    requireCrForPsf = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Require cosmic ray detection and masking to run successfully before measuring the PSF."
    )
    checkUnitsParseStrict = pexConfig.Field(
        doc="Strictness of Astropy unit compatibility check, can be 'raise', 'warn' or 'silent'",
        dtype=str,
        default="raise",
    )
    doMaskStreaks = pexConfig.Field(
        doc="Mask streaks",
        default=False,
        dtype=bool,
        deprecated=("This subtask has been moved to detectAndMeasureTask in "
                    "ip_diffim with DM-43370 and will be removed in DM-44658.")
    )
    maskStreaks = pexConfig.ConfigurableField(
        target=MaskStreaksTask,
        doc="Subtask for masking streaks. Only used if doMaskStreaks is True. "
            "Adds a mask plane to an exposure, with the mask plane name set by streakMaskName.",
        deprecated=("This subtask has been moved to detectAndMeasureTask in "
                    "ip_diffim with DM-43370 and will be removed in DM-44658.")
    )
    idGenerator = DetectorVisitIdGeneratorConfig.make_field()

    def setDefaults(self):
        super().setDefaults()
        # Just detect bright stars.
        # The thresholdValue sets the minimum flux in a pixel to be included in the
        # footprint, while peaks are only detected when they are above
        # thresholdValue * includeThresholdMultiplier. The low thresholdValue
        # ensures that the footprints are large enough for the noise replacer
        # to mask out faint undetected neighbors that are not to be measured.
        self.detection.thresholdValue = 5.0
        self.detection.includeThresholdMultiplier = 10.0
        # do not deblend, as it makes a mess
        self.doDeblend = False
        # measure and apply aperture correction; note: measuring and applying aperture
        # correction are disabled until the final measurement, after PSF is measured
        self.doApCorr = True
        # During characterization, we don't have full source measurement information,
        # so must do the aperture correction with only psf stars, combined with the
        # default signal-to-noise cuts in MeasureApCorrTask.
        selector = self.measureApCorr.sourceSelector["science"]
        selector.doUnresolved = False
        selector.flags.good = ["calib_psf_used"]
        selector.flags.bad = []

        # minimal set of measurements needed to determine PSF
        self.measurement.plugins.names = [
            "base_PixelFlags",
            "base_SdssCentroid",
            "ext_shapeHSM_HsmSourceMoments",
            "base_GaussianFlux",
            "base_PsfFlux",
            "base_CircularApertureFlux",
            "base_CompensatedTophatFlux",
        ]
        self.measurement.slots.shape = "ext_shapeHSM_HsmSourceMoments"
        self.measurement.algorithms["base_CompensatedTophatFlux"].apertures = [12]

    def validate(self):
        if self.doApCorr and not self.measurePsf:
            raise RuntimeError("Must measure PSF to measure aperture correction, "
                               "because flags determined by PSF measurement are used to identify "
                               "sources used to measure aperture correction")


class CharacterizeImageTask(pipeBase.PipelineTask):
    """Measure bright sources and use this to estimate background and PSF of
    an exposure.

    Given an exposure with defects repaired (masked and interpolated over,
    e.g. as output by `~lsst.ip.isr.IsrTask`):
    - detect and measure bright sources
    - repair cosmic rays
    - measure and subtract background
    - measure PSF

    Parameters
    ----------
    schema : `lsst.afw.table.Schema`, optional
        Initial schema for icSrc catalog.
    **kwargs
        Additional keyword arguments.

    Notes
    -----
    Debugging:
    CharacterizeImageTask has a debug dictionary with the following keys:

    frame
        int: if specified, the frame of first debug image displayed (defaults to 1)
    repair_iter
        bool; if True display image after each repair in the measure PSF loop
    background_iter
        bool; if True display image after each background subtraction in the measure PSF loop
    measure_iter
        bool; if True display image and sources at the end of each iteration of the measure PSF loop
        See `~lsst.meas.astrom.displayAstrometry` for the meaning of the various symbols.
    psf
        bool; if True display image and sources after PSF is measured;
        this will be identical to the final image displayed by measure_iter if measure_iter is true
    repair
        bool; if True display image and sources after final repair
    measure
        bool; if True display image and sources after final measurement
    """

    ConfigClass = CharacterizeImageConfig
    _DefaultName = "characterizeImage"

    def __init__(self, schema=None, **kwargs):
        super().__init__(**kwargs)

        if schema is None:
            schema = SourceTable.makeMinimalSchema()
        self.schema = schema
        self.makeSubtask("background")
        self.makeSubtask("installSimplePsf")
        self.makeSubtask("repair")
        # TODO: DM-44658, streak masking to happen only in ip_diffim
        if self.config.doMaskStreaks:
            self.makeSubtask("maskStreaks")
        self.makeSubtask("measurePsf", schema=self.schema)
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask('detection', schema=self.schema)
        if self.config.doDeblend:
            self.makeSubtask("deblend", schema=self.schema)
        self.makeSubtask('measurement', schema=self.schema, algMetadata=self.algMetadata)
        if self.config.doNormalizedCalibration:
            self.makeSubtask('normalizedCalibrationFlux', schema=self.schema)
        if self.config.doApCorr:
            self.makeSubtask('measureApCorr', schema=self.schema)
            self.makeSubtask('applyApCorr', schema=self.schema)
        self.makeSubtask('catalogCalculation', schema=self.schema)
        self._initialFrame = getDebugFrame(self._display, "frame") or 1
        self._frame = self._initialFrame
        self.schema.checkUnits(parse_strict=self.config.checkUnitsParseStrict)
        afwTable.CoordKey.addErrorFields(self.schema)
        self.outputSchema = afwTable.SourceCatalog(self.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        if 'idGenerator' not in inputs.keys():
            inputs['idGenerator'] = self.config.idGenerator.apply(butlerQC.quantum.dataId)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    @timeMethod
    def run(self, exposure, background=None, idGenerator=None):
        """Characterize a science image.

        Peforms the following operations:
        - Iterate the following config.psfIterations times, or once if config.doMeasurePsf false:
            - detect and measure sources and estimate PSF (see detectMeasureAndEstimatePsf for details)
        - interpolate over cosmic rays
        - perform final measurement

        Parameters
        ----------
        exposure : `lsst.afw.image.ExposureF`
            Exposure to characterize.
        background : `lsst.afw.math.BackgroundList`, optional
            Initial model of background already subtracted from exposure.
        idGenerator : `lsst.meas.base.IdGenerator`, optional
            Object that generates source IDs and provides RNG seeds.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``exposure``
               Characterized exposure (`lsst.afw.image.ExposureF`).
            ``sourceCat``
               Detected sources (`lsst.afw.table.SourceCatalog`).
            ``background``
               Model of subtracted background (`lsst.afw.math.BackgroundList`).
            ``psfCellSet``
               Spatial cells of PSF candidates (`lsst.afw.math.SpatialCellSet`).
            ``characterized``
               Another reference to ``exposure`` for compatibility.
            ``backgroundModel``
               Another reference to ``background`` for compatibility.

        Raises
        ------
        RuntimeError
            Raised if PSF sigma is NaN.
        UnprocessableDataError
            Raised if the unnormalized model PSF ellipticity is greater than
            maxUnNormPsfEllipticity or the model PSF ellipticity is greater
            than maxPsfEllipticity.
        """
        self._frame = self._initialFrame  # reset debug display frame

        if not self.config.doMeasurePsf and not exposure.hasPsf():
            self.log.info("CharacterizeImageTask initialized with 'simple' PSF.")
            self.installSimplePsf.run(exposure=exposure)

        if idGenerator is None:
            idGenerator = IdGenerator()

        # subtract an initial estimate of background level
        background = self.background.run(exposure).background

        psfIterations = self.config.psfIterations if self.config.doMeasurePsf else 1
        for i in range(psfIterations):
            dmeRes = self.detectMeasureAndEstimatePsf(
                exposure=exposure,
                idGenerator=idGenerator,
                background=background,
            )
            psf = dmeRes.exposure.getPsf()
            # Just need a rough estimate; average positions are fine
            psfAvgPos = psf.getAveragePosition()
            psfShape = psf.computeShape(psfAvgPos)
            psfSigma = psfShape.getDeterminantRadius()
            psfE1 = (psfShape.getIxx() - psfShape.getIyy())/(psfShape.getIxx() + psfShape.getIyy())
            psfE2 = 2.0*psfShape.getIxy()/(psfShape.getIxx() + psfShape.getIyy())
            psfE = np.sqrt(psfE1**2.0 + psfE2**2.0)
            unNormPsfE = np.hypot(psfShape.getIxx() - psfShape.getIyy(), 2.0*psfShape.getIxy())

            psfDimensions = psf.computeImage(psfAvgPos).getDimensions()
            medBackground = np.median(dmeRes.background.getImage().getArray())
            self.log.info(
                "iter %s: PSF sigma=%0.4f, psfE=%.3f, unNormPsfE=%.2f, dimensions=%s, "
                "median background=%0.2f",
                i + 1, psfSigma, psfE, unNormPsfE, psfDimensions, medBackground)
            if np.isnan(psfSigma):
                raise RuntimeError("PSF sigma is NaN, cannot continue PSF determination.")
        band = exposure.filter.bandLabel
        if band in self.config.maxUnNormPsfEllipticityPerBand:
            maxUnNormPsfEllipticity = self.config.maxUnNormPsfEllipticityPerBand[band]
        else:
            maxUnNormPsfEllipticity = self.config.maxUnNormPsfEllipticityFallback
            self.log.warning(
                f"Band {band} was not included in self.config.maxUnNormPsfEllipticityPerBand. "
                f"Setting maxUnNormPsfEllipticity to fallback value of {maxUnNormPsfEllipticity}."
            )
        if band in self.config.maxPsfEllipticityPerBand:
            maxPsfEllipticity = self.config.maxPsfEllipticityPerBand[band]
        else:
            maxPsfEllipticity = self.config.maxPsfEllipticityFallback
            self.log.warning(
                f"Band {band} was not included in self.config.maxPsfEllipticityPerBand. "
                f"Setting maxUnNormPsfEllipticity to fallback value of {maxPsfEllipticity:.2f}."
            )

        if unNormPsfE > maxUnNormPsfEllipticity or psfE > maxPsfEllipticity:
            raise pipeBase.UnprocessableDataError(
                "Either the unnormalized model PSF ellipticity is greater than the maximum allowed "
                f"for band {band} ({maxUnNormPsfEllipticity:.2f}) or the model PSF ellipticity "
                f"is greater than the maximum allowed ({maxPsfEllipticity:.2f}) "
                f"(unNormPsfE={unNormPsfE:.2f}, psfE={psfE:.2f})"
            )

        self.display("psf", exposure=dmeRes.exposure, sourceCat=dmeRes.sourceCat)

        # perform final repair with final PSF
        self.repair.run(exposure=dmeRes.exposure)
        self.display("repair", exposure=dmeRes.exposure, sourceCat=dmeRes.sourceCat)

        # mask streaks
        # TODO: Remove in DM-44658, streak masking to happen only in ip_diffim
        if self.config.doMaskStreaks:
            _ = self.maskStreaks.run(dmeRes.exposure)

        # perform final measurement with final PSF, including measuring and applying aperture correction,
        # if wanted
        self.measurement.run(measCat=dmeRes.sourceCat, exposure=dmeRes.exposure,
                             exposureId=idGenerator.catalog_id)

        if self.config.doNormalizedCalibration:
            normApCorrMap = self.normalizedCalibrationFlux.run(
                exposure=dmeRes.exposure,
                catalog=dmeRes.sourceCat,
            ).ap_corr_map
            dmeRes.exposure.info.setApCorrMap(normApCorrMap)
        else:
            normApCorrMap = None

        if self.config.doApCorr:
            # This aperture correction is relative to slot_CalibFlux_instFlux
            # which is now set to the normalized calibration flux if that
            # has been run.
            try:
                apCorrMap = self.measureApCorr.run(
                    exposure=dmeRes.exposure,
                    catalog=dmeRes.sourceCat,
                ).apCorrMap
            except MeasureApCorrError:
                # We have failed to get a valid aperture correction map.
                # Proceed with processing, and image will be filtered
                # downstream.
                dmeRes.exposure.info.setApCorrMap(None)
            else:
                # Need to merge the aperture correction map from the normalization.
                if normApCorrMap:
                    for key in normApCorrMap:
                        apCorrMap[key] = normApCorrMap[key]
                dmeRes.exposure.info.setApCorrMap(apCorrMap)
                self.applyApCorr.run(catalog=dmeRes.sourceCat, apCorrMap=exposure.getInfo().getApCorrMap())

        self.catalogCalculation.run(dmeRes.sourceCat)

        self.display("measure", exposure=dmeRes.exposure, sourceCat=dmeRes.sourceCat)

        return pipeBase.Struct(
            exposure=dmeRes.exposure,
            sourceCat=dmeRes.sourceCat,
            background=dmeRes.background,
            psfCellSet=dmeRes.psfCellSet,

            characterized=dmeRes.exposure,
            backgroundModel=dmeRes.background
        )

    @timeMethod
    def detectMeasureAndEstimatePsf(self, exposure, idGenerator, background):
        """Perform one iteration of detect, measure, and estimate PSF.

        Performs the following operations:

        - if config.doMeasurePsf or not exposure.hasPsf():

            - install a simple PSF model (replacing the existing one, if need be)

        - interpolate over cosmic rays with keepCRs=True
        - estimate background and subtract it from the exposure
        - detect, deblend and measure sources, and subtract a refined background model;
        - if config.doMeasurePsf:
            - measure PSF

        Parameters
        ----------
        exposure : `lsst.afw.image.ExposureF`
            Exposure to characterize.
        idGenerator : `lsst.meas.base.IdGenerator`
            Object that generates source IDs and provides RNG seeds.
        background : `lsst.afw.math.BackgroundList`, optional
            Initial model of background already subtracted from exposure.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``exposure``
               Characterized exposure (`lsst.afw.image.ExposureF`).
            ``sourceCat``
               Detected sources (`lsst.afw.table.SourceCatalog`).
            ``background``
               Model of subtracted background (`lsst.afw.math.BackgroundList`).
            ``psfCellSet``
               Spatial cells of PSF candidates (`lsst.afw.math.SpatialCellSet`).

        Raises
        ------
        LengthError
            Raised if there are too many CR pixels.
        """
        # install a simple PSF model, if needed or wanted
        if not exposure.hasPsf() or (self.config.doMeasurePsf and self.config.useSimplePsf):
            self.log.info("PSF estimation initialized with 'simple' PSF")
            self.installSimplePsf.run(exposure=exposure)

        # run repair, but do not interpolate over cosmic rays (do that elsewhere, with the final PSF model)
        if self.config.requireCrForPsf:
            self.repair.run(exposure=exposure, keepCRs=True)
        else:
            try:
                self.repair.run(exposure=exposure, keepCRs=True)
            except LengthError:
                self.log.warning("Skipping cosmic ray detection: Too many CR pixels (max %0.f)",
                                 self.config.repair.cosmicray.nCrPixelMax)

        self.display("repair_iter", exposure=exposure)

        if background is None:
            background = BackgroundList()

        sourceIdFactory = idGenerator.make_table_id_factory()
        table = SourceTable.make(self.schema, sourceIdFactory)
        table.setMetadata(self.algMetadata)

        detRes = self.detection.run(table=table, exposure=exposure, doSmooth=True)
        sourceCat = detRes.sources
        if detRes.background:
            for bg in detRes.background:
                background.append(bg)

        if self.config.doDeblend:
            self.deblend.run(exposure=exposure, sources=sourceCat)
            # We need the output catalog to be contiguous for further processing.
            if not sourceCat.isContiguous():
                sourceCat = sourceCat.copy(deep=True)

        self.measurement.run(measCat=sourceCat, exposure=exposure, exposureId=idGenerator.catalog_id)

        measPsfRes = pipeBase.Struct(cellSet=None)
        if self.config.doMeasurePsf:
            measPsfRes = self.measurePsf.run(exposure=exposure, sources=sourceCat,
                                             expId=idGenerator.catalog_id)
        self.display("measure_iter", exposure=exposure, sourceCat=sourceCat)

        return pipeBase.Struct(
            exposure=exposure,
            sourceCat=sourceCat,
            background=background,
            psfCellSet=measPsfRes.cellSet,
        )

    def display(self, itemName, exposure, sourceCat=None):
        """Display exposure and sources on next frame (for debugging).

        Parameters
        ----------
        itemName : `str`
            Name of item in ``debugInfo``.
        exposure : `lsst.afw.image.ExposureF`
            Exposure to display.
        sourceCat : `lsst.afw.table.SourceCatalog`, optional
            Catalog of sources detected on the exposure.
        """
        val = getDebugFrame(self._display, itemName)
        if not val:
            return

        displayAstrometry(exposure=exposure, sourceCat=sourceCat, frame=self._frame, pause=False)
        self._frame += 1
