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

import numpy as np
import warnings

from lsstDebug import getDebugFrame
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
import lsst.pipe.base.connectionTypes as cT
from lsst.afw.math import BackgroundList
from lsst.afw.table import SourceTable, SourceCatalog
from lsst.meas.algorithms import SubtractBackgroundTask, SourceDetectionTask, MeasureApCorrTask
from lsst.meas.algorithms.installGaussianPsf import InstallGaussianPsfTask
from lsst.meas.astrom import RefMatchTask, displayAstrometry
from lsst.meas.algorithms import LoadReferenceObjectsConfig
from lsst.obs.base import ExposureIdInfo
from lsst.meas.base import SingleFrameMeasurementTask, ApplyApCorrTask, CatalogCalculationTask
from lsst.meas.deblender import SourceDeblendTask
import lsst.meas.extensions.shapeHSM  # noqa: F401 needed for default shape plugin
from .measurePsf import MeasurePsfTask
from .repair import RepairTask
from .computeExposureSummaryStats import ComputeExposureSummaryStatsTask
from lsst.pex.exceptions import LengthError
from lsst.utils.timer import timeMethod

__all__ = ["CharacterizeImageConfig", "CharacterizeImageTask"]


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
    refObjLoader = pexConfig.ConfigField(
        dtype=LoadReferenceObjectsConfig,
        deprecated="This field does nothing. Will be removed after v24 (see DM-34768).",
        doc="reference object loader",
    )
    ref_match = pexConfig.ConfigurableField(
        target=RefMatchTask,
        deprecated="This field was never usable. Will be removed after v24 (see DM-34768).",
        doc="Task to load and match reference objects. Only used if measurePsf can use matches. "
        "Warning: matching will only work well if the initial WCS is accurate enough "
        "to give good matches (roughly: good to 3 arcsec across the CCD).",
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

    def setDefaults(self):
        super().setDefaults()
        # just detect bright stars; includeThresholdMultipler=10 seems large,
        # but these are the values we have been using
        self.detection.thresholdValue = 5.0
        self.detection.includeThresholdMultiplier = 10.0
        self.detection.doTempLocalBackground = False
        # do not deblend, as it makes a mess
        self.doDeblend = False
        # measure and apply aperture correction; note: measuring and applying aperture
        # correction are disabled until the final measurement, after PSF is measured
        self.doApCorr = True
        # minimal set of measurements needed to determine PSF
        self.measurement.plugins.names = [
            "base_PixelFlags",
            "base_SdssCentroid",
            "ext_shapeHSM_HsmSourceMoments",
            "base_GaussianFlux",
            "base_PsfFlux",
            "base_CircularApertureFlux",
        ]
        self.measurement.slots.shape = "ext_shapeHSM_HsmSourceMoments"

    def validate(self):
        if self.doApCorr and not self.measurePsf:
            raise RuntimeError("Must measure PSF to measure aperture correction, "
                               "because flags determined by PSF measurement are used to identify "
                               "sources used to measure aperture correction")


class CharacterizeImageTask(pipeBase.PipelineTask):
    """Measure bright sources and use this to estimate background and PSF of an exposure.

    Parameters
    ----------
    butler : `None`
        Compatibility parameter. Should always be `None`.
    refObjLoader : `lsst.meas.algorithms.ReferenceObjectLoader`, optional
        Reference object loader if using a catalog-based star-selector.
    schema : `lsst.afw.table.Schema`, optional
        Initial schema for icSrc catalog.
    **kwargs
        Additional keyword arguments.
    """

    ConfigClass = CharacterizeImageConfig
    _DefaultName = "characterizeImage"

    def __init__(self, butler=None, refObjLoader=None, schema=None, **kwargs):
        super().__init__(**kwargs)

        if butler is not None:
            warnings.warn("The 'butler' parameter is no longer used and can be safely removed.",
                          category=FutureWarning, stacklevel=2)
            butler = None

        if schema is None:
            schema = SourceTable.makeMinimalSchema()
        self.schema = schema
        self.makeSubtask("background")
        self.makeSubtask("installSimplePsf")
        self.makeSubtask("repair")
        self.makeSubtask("measurePsf", schema=self.schema)
        # TODO DM-34769: remove this `if` block
        if self.config.doMeasurePsf and self.measurePsf.usesMatches:
            self.makeSubtask("ref_match", refObjLoader=refObjLoader)
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask('detection', schema=self.schema)
        if self.config.doDeblend:
            self.makeSubtask("deblend", schema=self.schema)
        self.makeSubtask('measurement', schema=self.schema, algMetadata=self.algMetadata)
        if self.config.doApCorr:
            self.makeSubtask('measureApCorr', schema=self.schema)
            self.makeSubtask('applyApCorr', schema=self.schema)
        self.makeSubtask('catalogCalculation', schema=self.schema)
        self._initialFrame = getDebugFrame(self._display, "frame") or 1
        self._frame = self._initialFrame
        self.schema.checkUnits(parse_strict=self.config.checkUnitsParseStrict)
        self.outputSchema = afwTable.SourceCatalog(self.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        if 'exposureIdInfo' not in inputs.keys():
            inputs['exposureIdInfo'] = ExposureIdInfo.fromDataId(butlerQC.quantum.dataId, "visit_detector")
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    @timeMethod
    def run(self, exposure, exposureIdInfo=None, background=None):
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
        exposureIdInfo : `lsst.obs.baseExposureIdInfo`, optional
            Exposure ID info. If not provided, returned SourceCatalog IDs will not
            be globally unique.
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
            ``characterized``
               Another reference to ``exposure`` for compatibility.
            ``backgroundModel``
               Another reference to ``background`` for compatibility.

        Raises
        ------
        RuntimeError
            Raised if PSF sigma is NaN.
        """
        self._frame = self._initialFrame  # reset debug display frame

        if not self.config.doMeasurePsf and not exposure.hasPsf():
            self.log.info("CharacterizeImageTask initialized with 'simple' PSF.")
            self.installSimplePsf.run(exposure=exposure)

        if exposureIdInfo is None:
            exposureIdInfo = ExposureIdInfo()

        # subtract an initial estimate of background level
        background = self.background.run(exposure).background

        psfIterations = self.config.psfIterations if self.config.doMeasurePsf else 1
        for i in range(psfIterations):
            dmeRes = self.detectMeasureAndEstimatePsf(
                exposure=exposure,
                exposureIdInfo=exposureIdInfo,
                background=background,
            )

            psf = dmeRes.exposure.getPsf()
            # Just need a rough estimate; average positions are fine
            psfAvgPos = psf.getAveragePosition()
            psfSigma = psf.computeShape(psfAvgPos).getDeterminantRadius()
            psfDimensions = psf.computeImage(psfAvgPos).getDimensions()
            medBackground = np.median(dmeRes.background.getImage().getArray())
            self.log.info("iter %s; PSF sigma=%0.2f, dimensions=%s; median background=%0.2f",
                          i + 1, psfSigma, psfDimensions, medBackground)
            if np.isnan(psfSigma):
                raise RuntimeError("PSF sigma is NaN, cannot continue PSF determination.")

        self.display("psf", exposure=dmeRes.exposure, sourceCat=dmeRes.sourceCat)

        # perform final repair with final PSF
        self.repair.run(exposure=dmeRes.exposure)
        self.display("repair", exposure=dmeRes.exposure, sourceCat=dmeRes.sourceCat)

        # perform final measurement with final PSF, including measuring and applying aperture correction,
        # if wanted
        self.measurement.run(measCat=dmeRes.sourceCat, exposure=dmeRes.exposure,
                             exposureId=exposureIdInfo.expId)
        if self.config.doApCorr:
            apCorrMap = self.measureApCorr.run(exposure=dmeRes.exposure, catalog=dmeRes.sourceCat).apCorrMap
            dmeRes.exposure.getInfo().setApCorrMap(apCorrMap)
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
    def detectMeasureAndEstimatePsf(self, exposure, exposureIdInfo, background):
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
        exposureIdInfo : `lsst.obs.baseExposureIdInfo`
            Exposure ID info.
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

        sourceIdFactory = exposureIdInfo.makeSourceIdFactory()
        table = SourceTable.make(self.schema, sourceIdFactory)
        table.setMetadata(self.algMetadata)

        detRes = self.detection.run(table=table, exposure=exposure, doSmooth=True)
        sourceCat = detRes.sources
        if detRes.fpSets.background:
            for bg in detRes.fpSets.background:
                background.append(bg)

        if self.config.doDeblend:
            self.deblend.run(exposure=exposure, sources=sourceCat)

        self.measurement.run(measCat=sourceCat, exposure=exposure, exposureId=exposureIdInfo.expId)

        measPsfRes = pipeBase.Struct(cellSet=None)
        if self.config.doMeasurePsf:
            # TODO DM-34769: remove this `if` block, and the `matches` kwarg from measurePsf.run below.
            if self.measurePsf.usesMatches:
                matches = self.ref_match.loadAndMatch(exposure=exposure, sourceCat=sourceCat).matches
            else:
                matches = None
            measPsfRes = self.measurePsf.run(exposure=exposure, sources=sourceCat, matches=matches,
                                             expId=exposureIdInfo.expId)
        self.display("measure_iter", exposure=exposure, sourceCat=sourceCat)

        return pipeBase.Struct(
            exposure=exposure,
            sourceCat=sourceCat,
            background=background,
            psfCellSet=measPsfRes.cellSet,
        )

    def getSchemaCatalogs(self):
        """Return a dict of empty catalogs for each catalog dataset produced by this task.
        """
        sourceCat = SourceCatalog(self.schema)
        sourceCat.getTable().setMetadata(self.algMetadata)
        return {"icSrc": sourceCat}

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
