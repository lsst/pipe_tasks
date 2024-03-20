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

__all__ = ["CalibrateConfig", "CalibrateTask"]

import math
import numpy as np

from lsstDebug import getDebugFrame
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.afw.table as afwTable
from lsst.meas.astrom import AstrometryTask, displayAstrometry, denormalizeMatches
from lsst.meas.algorithms import LoadReferenceObjectsConfig, SkyObjectsTask
import lsst.daf.base as dafBase
from lsst.afw.math import BackgroundList
from lsst.afw.table import SourceTable
from lsst.meas.algorithms import SourceDetectionTask, ReferenceObjectLoader, SetPrimaryFlagsTask
from lsst.meas.base import (SingleFrameMeasurementTask,
                            ApplyApCorrTask,
                            CatalogCalculationTask,
                            IdGenerator,
                            DetectorVisitIdGeneratorConfig)
from lsst.meas.deblender import SourceDeblendTask
from lsst.utils.timer import timeMethod
from .photoCal import PhotoCalTask
from .computeExposureSummaryStats import ComputeExposureSummaryStatsTask


class _EmptyTargetTask(pipeBase.PipelineTask):
    """
    This is a placeholder target for CreateSummaryMetrics and must be retargeted at runtime.
    CreateSummaryMetrics should target an analysis tool task, but that would, at the time
    of writing, result in a circular import.

    As a result, this class should not be used for anything else.
    """
    ConfigClass = pipeBase.PipelineTaskConfig

    def __init__(self, **kwargs) -> None:
        raise NotImplementedError(
            "doCreateSummaryMetrics is set to True, in which case "
            "createSummaryMetrics must be retargeted."
        )


class CalibrateConnections(pipeBase.PipelineTaskConnections, dimensions=("instrument", "visit", "detector"),
                           defaultTemplates={}):

    icSourceSchema = cT.InitInput(
        doc="Schema produced by characterize image task, used to initialize this task",
        name="icSrc_schema",
        storageClass="SourceCatalog",
    )

    outputSchema = cT.InitOutput(
        doc="Schema after CalibrateTask has been initialized",
        name="src_schema",
        storageClass="SourceCatalog",
    )

    exposure = cT.Input(
        doc="Input image to calibrate",
        name="icExp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
    )

    background = cT.Input(
        doc="Backgrounds determined by characterize task",
        name="icExpBackground",
        storageClass="Background",
        dimensions=("instrument", "visit", "detector"),
    )

    icSourceCat = cT.Input(
        doc="Source catalog created by characterize task",
        name="icSrc",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector"),
    )

    astromRefCat = cT.PrerequisiteInput(
        doc="Reference catalog to use for astrometry",
        name="gaia_dr3_20230707",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        deferLoad=True,
        multiple=True,
    )

    photoRefCat = cT.PrerequisiteInput(
        doc="Reference catalog to use for photometric calibration",
        name="ps1_pv3_3pi_20170110",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        deferLoad=True,
        multiple=True
    )

    outputExposure = cT.Output(
        doc="Exposure after running calibration task",
        name="calexp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
    )

    outputCat = cT.Output(
        doc="Source catalog produced in calibrate task",
        name="src",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector"),
    )

    outputBackground = cT.Output(
        doc="Background models estimated in calibration task",
        name="calexpBackground",
        storageClass="Background",
        dimensions=("instrument", "visit", "detector"),
    )

    outputSummaryMetrics = cT.Output(
        doc="Summary metrics created by the calibration task",
        name="calexpSummary_metrics",
        storageClass="MetricMeasurementBundle",
        dimensions=("instrument", "visit", "detector"),
    )

    matches = cT.Output(
        doc="Source/refObj matches from the astrometry solver",
        name="srcMatch",
        storageClass="Catalog",
        dimensions=("instrument", "visit", "detector"),
    )

    matchesDenormalized = cT.Output(
        doc="Denormalized matches from astrometry solver",
        name="srcMatchFull",
        storageClass="Catalog",
        dimensions=("instrument", "visit", "detector"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config.doAstrometry is False:
            self.prerequisiteInputs.remove("astromRefCat")
        if config.doPhotoCal is False:
            self.prerequisiteInputs.remove("photoRefCat")

        if config.doWriteMatches is False or config.doAstrometry is False:
            self.outputs.remove("matches")
        if config.doWriteMatchesDenormalized is False or config.doAstrometry is False:
            self.outputs.remove("matchesDenormalized")

        if config.doCreateSummaryMetrics is False:
            self.outputs.remove("outputSummaryMetrics")


class CalibrateConfig(pipeBase.PipelineTaskConfig, pipelineConnections=CalibrateConnections):
    """Config for CalibrateTask."""

    doWrite = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Save calibration results?",
    )
    doWriteHeavyFootprintsInSources = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Include HeavyFootprint data in source table? If false then heavy "
            "footprints are saved as normal footprints, which saves some space"
    )
    doWriteMatches = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Write reference matches (ignored if doWrite or doAstrometry false)?",
    )
    doWriteMatchesDenormalized = pexConfig.Field(
        dtype=bool,
        default=True,
        doc=("Write reference matches in denormalized format? "
             "This format uses more disk space, but is more convenient to "
             "read for debugging. Ignored if doWriteMatches=False or doWrite=False."),
    )
    doAstrometry = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Perform astrometric calibration?",
    )
    astromRefObjLoader = pexConfig.ConfigField(
        dtype=LoadReferenceObjectsConfig,
        doc="reference object loader for astrometric calibration",
    )
    photoRefObjLoader = pexConfig.ConfigField(
        dtype=LoadReferenceObjectsConfig,
        doc="reference object loader for photometric calibration",
    )
    astrometry = pexConfig.ConfigurableField(
        target=AstrometryTask,
        doc="Perform astrometric calibration to refine the WCS",
    )
    requireAstrometry = pexConfig.Field(
        dtype=bool,
        default=True,
        doc=("Raise an exception if astrometry fails? Ignored if doAstrometry "
             "false."),
    )
    doPhotoCal = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Perform phometric calibration?",
    )
    requirePhotoCal = pexConfig.Field(
        dtype=bool,
        default=True,
        doc=("Raise an exception if photoCal fails? Ignored if doPhotoCal "
             "false."),
    )
    photoCal = pexConfig.ConfigurableField(
        target=PhotoCalTask,
        doc="Perform photometric calibration",
    )
    icSourceFieldsToCopy = pexConfig.ListField(
        dtype=str,
        default=("calib_psf_candidate", "calib_psf_used", "calib_psf_reserved"),
        doc=("Fields to copy from the icSource catalog to the output catalog "
             "for matching sources Any missing fields will trigger a "
             "RuntimeError exception. Ignored if icSourceCat is not provided.")
    )
    matchRadiusPix = pexConfig.Field(
        dtype=float,
        default=3,
        doc=("Match radius for matching icSourceCat objects to sourceCat "
             "objects (pixels)"),
    )
    checkUnitsParseStrict = pexConfig.Field(
        doc=("Strictness of Astropy unit compatibility check, can be 'raise', "
             "'warn' or 'silent'"),
        dtype=str,
        default="raise",
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
        doc="Split blended sources into their components"
    )
    doSkySources = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Generate sky sources?",
    )
    skySources = pexConfig.ConfigurableField(
        target=SkyObjectsTask,
        doc="Generate sky sources",
    )
    measurement = pexConfig.ConfigurableField(
        target=SingleFrameMeasurementTask,
        doc="Measure sources"
    )
    postCalibrationMeasurement = pexConfig.ConfigurableField(
        target=SingleFrameMeasurementTask,
        doc="Second round of measurement for plugins that need to be run after photocal"
    )
    setPrimaryFlags = pexConfig.ConfigurableField(
        target=SetPrimaryFlagsTask,
        doc=("Set flags for primary source classification in single frame "
             "processing. True if sources are not sky sources and not a parent.")
    )
    doApCorr = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Run subtask to apply aperture correction"
    )
    applyApCorr = pexConfig.ConfigurableField(
        target=ApplyApCorrTask,
        doc="Subtask to apply aperture corrections"
    )
    # If doApCorr is False, and the exposure does not have apcorrections
    # already applied, the active plugins in catalogCalculation almost
    # certainly should not contain the characterization plugin
    catalogCalculation = pexConfig.ConfigurableField(
        target=CatalogCalculationTask,
        doc="Subtask to run catalogCalculation plugins on catalog"
    )
    doComputeSummaryStats = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Run subtask to measure exposure summary statistics?"
    )
    computeSummaryStats = pexConfig.ConfigurableField(
        target=ComputeExposureSummaryStatsTask,
        doc="Subtask to run computeSummaryStats on exposure"
    )
    doCreateSummaryMetrics = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Run the subtask to create summary metrics, and then write those metrics."
    )
    createSummaryMetrics = pexConfig.ConfigurableField(
        target=_EmptyTargetTask,
        doc="Subtask to create metrics from the summary stats. This must be retargeted, likely to an"
        "analysis_tools task such as CalexpSummaryMetrics."
    )
    doWriteExposure = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Write the calexp? If fakes have been added then we do not want to write out the calexp as a "
            "normal calexp but as a fakes_calexp."
    )
    idGenerator = DetectorVisitIdGeneratorConfig.make_field()

    def setDefaults(self):
        super().setDefaults()
        self.postCalibrationMeasurement.plugins.names = ["base_LocalPhotoCalib", "base_LocalWcs"]
        self.postCalibrationMeasurement.doReplaceWithNoise = False
        for key in self.postCalibrationMeasurement.slots:
            setattr(self.postCalibrationMeasurement.slots, key, None)
        self.astromRefObjLoader.anyFilterMapsToThis = "phot_g_mean"
        # The photoRefCat connection is the name to use for the colorterms.
        self.photoCal.photoCatName = self.connections.photoRefCat

        # Keep track of which footprints contain streaks
        self.measurement.plugins['base_PixelFlags'].masksFpAnywhere = ['STREAK']
        self.measurement.plugins['base_PixelFlags'].masksFpCenter = ['STREAK']


class CalibrateTask(pipeBase.PipelineTask):
    """Calibrate an exposure: measure sources and perform astrometric and
    photometric calibration.

    Given an exposure with a good PSF model and aperture correction map(e.g. as
    provided by `~lsst.pipe.tasks.characterizeImage.CharacterizeImageTask`),
    perform the following operations:
    - Run detection and measurement
    - Run astrometry subtask to fit an improved WCS
    - Run photoCal subtask to fit the exposure's photometric zero-point

    Parameters
    ----------
    butler : `None`
        Compatibility parameter. Should always be `None`.
    astromRefObjLoader : `lsst.meas.algorithms.ReferenceObjectLoader`, optional
        Unused in gen3: must be `None`.
    photoRefObjLoader : `lsst.meas.algorithms.ReferenceObjectLoader`, optional
        Unused in gen3: must be `None`.
    icSourceSchema : `lsst.afw.table.Schema`, optional
        Schema for the icSource catalog.
    initInputs : `dict`, optional
        Dictionary that can contain a key ``icSourceSchema`` containing the
        input schema. If present will override the value of ``icSourceSchema``.

    Raises
    ------
    RuntimeError
        Raised if any of the following occur:
        - isSourceCat is missing fields specified in icSourceFieldsToCopy.
        - PipelineTask form of this task is initialized with reference object
          loaders.

    Notes
    -----
    Quantities set in exposure Metadata:

    MAGZERO_RMS
        MAGZERO's RMS == sigma reported by photoCal task
    MAGZERO_NOBJ
        Number of stars used == ngood reported by photoCal task
    COLORTERM1
        ?? (always 0.0)
    COLORTERM2
        ?? (always 0.0)
    COLORTERM3
        ?? (always 0.0)

    Debugging:
    CalibrateTask has a debug dictionary containing one key:

    calibrate
        frame (an int; <= 0 to not display) in which to display the exposure,
        sources and matches. See @ref lsst.meas.astrom.displayAstrometry for
        the meaning of the various symbols.
    """

    ConfigClass = CalibrateConfig
    _DefaultName = "calibrate"

    def __init__(self, astromRefObjLoader=None,
                 photoRefObjLoader=None, icSourceSchema=None,
                 initInputs=None, **kwargs):
        super().__init__(**kwargs)

        if initInputs is not None:
            icSourceSchema = initInputs['icSourceSchema'].schema

        if icSourceSchema is not None:
            # use a schema mapper to avoid copying each field separately
            self.schemaMapper = afwTable.SchemaMapper(icSourceSchema)
            minimumSchema = afwTable.SourceTable.makeMinimalSchema()
            self.schemaMapper.addMinimalSchema(minimumSchema, False)

            # Add fields to copy from an icSource catalog
            # and a field to indicate that the source matched a source in that
            # catalog. If any fields are missing then raise an exception, but
            # first find all missing fields in order to make the error message
            # more useful.
            self.calibSourceKey = self.schemaMapper.addOutputField(
                afwTable.Field["Flag"]("calib_detected",
                                       "Source was detected as an icSource"))
            missingFieldNames = []
            for fieldName in self.config.icSourceFieldsToCopy:
                try:
                    schemaItem = icSourceSchema.find(fieldName)
                except Exception:
                    missingFieldNames.append(fieldName)
                else:
                    # field found; if addMapping fails then raise an exception
                    self.schemaMapper.addMapping(schemaItem.getKey())

            if missingFieldNames:
                raise RuntimeError("isSourceCat is missing fields {} "
                                   "specified in icSourceFieldsToCopy"
                                   .format(missingFieldNames))

            # produce a temporary schema to pass to the subtasks; finalize it
            # later
            self.schema = self.schemaMapper.editOutputSchema()
        else:
            self.schemaMapper = None
            self.schema = afwTable.SourceTable.makeMinimalSchema()
        afwTable.CoordKey.addErrorFields(self.schema)
        self.makeSubtask('detection', schema=self.schema)

        self.algMetadata = dafBase.PropertyList()

        if self.config.doDeblend:
            self.makeSubtask("deblend", schema=self.schema)
        if self.config.doSkySources:
            self.makeSubtask("skySources")
            self.skySourceKey = self.schema.addField("sky_source", type="Flag", doc="Sky objects.")
        self.makeSubtask('measurement', schema=self.schema,
                         algMetadata=self.algMetadata)
        self.makeSubtask('postCalibrationMeasurement', schema=self.schema,
                         algMetadata=self.algMetadata)
        self.makeSubtask("setPrimaryFlags", schema=self.schema, isSingleFrame=True)
        if self.config.doApCorr:
            self.makeSubtask('applyApCorr', schema=self.schema)
        self.makeSubtask('catalogCalculation', schema=self.schema)

        if self.config.doAstrometry:
            self.makeSubtask("astrometry", refObjLoader=astromRefObjLoader,
                             schema=self.schema)
        if self.config.doPhotoCal:
            self.makeSubtask("photoCal", refObjLoader=photoRefObjLoader,
                             schema=self.schema)
        if self.config.doComputeSummaryStats:
            self.makeSubtask('computeSummaryStats')
            if self.config.doCreateSummaryMetrics:
                self.makeSubtask('createSummaryMetrics')

        if initInputs is not None and (astromRefObjLoader is not None or photoRefObjLoader is not None):
            raise RuntimeError("PipelineTask form of this task should not be initialized with "
                               "reference object loaders.")

        if self.schemaMapper is not None:
            # finalize the schema
            self.schema = self.schemaMapper.getOutputSchema()
        self.schema.checkUnits(parse_strict=self.config.checkUnitsParseStrict)

        sourceCatSchema = afwTable.SourceCatalog(self.schema)
        sourceCatSchema.getTable().setMetadata(self.algMetadata)
        self.outputSchema = sourceCatSchema

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        inputs['idGenerator'] = self.config.idGenerator.apply(butlerQC.quantum.dataId)

        if self.config.doAstrometry:
            refObjLoader = ReferenceObjectLoader(dataIds=[ref.datasetRef.dataId
                                                          for ref in inputRefs.astromRefCat],
                                                 refCats=inputs.pop('astromRefCat'),
                                                 name=self.config.connections.astromRefCat,
                                                 config=self.config.astromRefObjLoader, log=self.log)
            self.astrometry.setRefObjLoader(refObjLoader)

        if self.config.doPhotoCal:
            photoRefObjLoader = ReferenceObjectLoader(dataIds=[ref.datasetRef.dataId
                                                      for ref in inputRefs.photoRefCat],
                                                      refCats=inputs.pop('photoRefCat'),
                                                      name=self.config.connections.photoRefCat,
                                                      config=self.config.photoRefObjLoader,
                                                      log=self.log)
            self.photoCal.match.setRefObjLoader(photoRefObjLoader)

        outputs = self.run(**inputs)

        if self.config.doWriteMatches and self.config.doAstrometry:
            if outputs.astromMatches is not None:
                normalizedMatches = afwTable.packMatches(outputs.astromMatches)
                normalizedMatches.table.setMetadata(outputs.matchMeta)
                if self.config.doWriteMatchesDenormalized:
                    denormMatches = denormalizeMatches(outputs.astromMatches, outputs.matchMeta)
                    outputs.matchesDenormalized = denormMatches
                outputs.matches = normalizedMatches
            else:
                del outputRefs.matches
                if self.config.doWriteMatchesDenormalized:
                    del outputRefs.matchesDenormalized
        butlerQC.put(outputs, outputRefs)

    @timeMethod
    def run(self, exposure, background=None,
            icSourceCat=None, idGenerator=None):
        """Calibrate an exposure.

        Parameters
        ----------
        exposure : `lsst.afw.image.ExposureF`
            Exposure to calibrate.
        background : `lsst.afw.math.BackgroundList`, optional
            Initial model of background already subtracted from exposure.
        icSourceCat : `lsst.afw.image.SourceCatalog`, optional
            SourceCatalog from CharacterizeImageTask from which we can copy
            some fields.
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
            ``outputBackground``
               Model of subtracted background (`lsst.afw.math.BackgroundList`).
            ``astromMatches``
               List of source/ref matches from astrometry solver.
            ``matchMeta``
               Metadata from astrometry matches.
            ``outputExposure``
               Another reference to ``exposure`` for compatibility.
            ``outputCat``
               Another reference to ``sourceCat`` for compatibility.
        """
        # detect, deblend and measure sources
        if idGenerator is None:
            idGenerator = IdGenerator()

        if background is None:
            background = BackgroundList()
        table = SourceTable.make(self.schema, idGenerator.make_table_id_factory())
        table.setMetadata(self.algMetadata)

        detRes = self.detection.run(table=table, exposure=exposure,
                                    doSmooth=True)
        sourceCat = detRes.sources
        if detRes.background:
            for bg in detRes.background:
                background.append(bg)
        if self.config.doSkySources:
            skySourceFootprints = self.skySources.run(mask=exposure.mask, seed=idGenerator.catalog_id)
            if skySourceFootprints:
                for foot in skySourceFootprints:
                    s = sourceCat.addNew()
                    s.setFootprint(foot)
                    s.set(self.skySourceKey, True)
        if self.config.doDeblend:
            self.deblend.run(exposure=exposure, sources=sourceCat)
        if not sourceCat.isContiguous():
            sourceCat = sourceCat.copy(deep=True)
        self.measurement.run(
            measCat=sourceCat,
            exposure=exposure,
            exposureId=idGenerator.catalog_id,
        )
        if self.config.doApCorr:
            apCorrMap = exposure.getInfo().getApCorrMap()
            if apCorrMap is None:
                self.log.warning("Image does not have valid aperture correction map for %r; "
                                 "skipping aperture correction", idGenerator)
            else:
                self.applyApCorr.run(
                    catalog=sourceCat,
                    apCorrMap=apCorrMap,
                )
        self.catalogCalculation.run(sourceCat)

        self.setPrimaryFlags.run(sourceCat)

        if icSourceCat is not None and \
           len(self.config.icSourceFieldsToCopy) > 0:
            self.copyIcSourceFields(icSourceCat=icSourceCat,
                                    sourceCat=sourceCat)

        # TODO DM-11568: this contiguous check-and-copy could go away if we
        # reserve enough space during SourceDetection and/or SourceDeblend.
        # NOTE: sourceSelectors require contiguous catalogs, so ensure
        # contiguity now, so views are preserved from here on.
        if not sourceCat.isContiguous():
            sourceCat = sourceCat.copy(deep=True)

        # perform astrometry calibration:
        # fit an improved WCS and update the exposure's WCS in place
        astromMatches = None
        matchMeta = None
        if self.config.doAstrometry:
            astromRes = self.astrometry.run(
                exposure=exposure,
                sourceCat=sourceCat,
            )
            astromMatches = astromRes.matches
            matchMeta = astromRes.matchMeta
            if exposure.getWcs() is None:
                if self.config.requireAstrometry:
                    raise RuntimeError(f"WCS fit failed for {idGenerator} and requireAstrometry "
                                       "is True.")
                else:
                    self.log.warning("Unable to perform astrometric calibration for %r but "
                                     "requireAstrometry is False: attempting to proceed...",
                                     idGenerator)

        # compute photometric calibration
        if self.config.doPhotoCal:
            if np.all(np.isnan(sourceCat["coord_ra"])) or np.all(np.isnan(sourceCat["coord_dec"])):
                if self.config.requirePhotoCal:
                    raise RuntimeError(f"Astrometry failed for {idGenerator}, so cannot do "
                                       "photoCal, but requirePhotoCal is True.")
                self.log.warning("Astrometry failed for %r, so cannot do photoCal. requirePhotoCal "
                                 "is False, so skipping photometric calibration and setting photoCalib "
                                 "to None.  Attempting to proceed...", idGenerator)
                exposure.setPhotoCalib(None)
                self.setMetadata(exposure=exposure, photoRes=None)
            else:
                try:
                    photoRes = self.photoCal.run(
                        exposure, sourceCat=sourceCat, expId=idGenerator.catalog_id
                    )
                    exposure.setPhotoCalib(photoRes.photoCalib)
                    # TODO: reword this to phrase it in terms of the
                    # calibration factor?
                    self.log.info("Photometric zero-point: %f",
                                  photoRes.photoCalib.instFluxToMagnitude(1.0))
                    self.setMetadata(exposure=exposure, photoRes=photoRes)
                except Exception as e:
                    if self.config.requirePhotoCal:
                        raise
                    self.log.warning("Unable to perform photometric calibration "
                                     "(%s): attempting to proceed", e)
                    self.setMetadata(exposure=exposure, photoRes=None)

        self.postCalibrationMeasurement.run(
            measCat=sourceCat,
            exposure=exposure,
            exposureId=idGenerator.catalog_id,
        )

        summaryMetrics = None
        if self.config.doComputeSummaryStats:
            summary = self.computeSummaryStats.run(exposure=exposure,
                                                   sources=sourceCat,
                                                   background=background)
            exposure.getInfo().setSummaryStats(summary)
            if self.config.doCreateSummaryMetrics:
                summaryMetrics = self.createSummaryMetrics.run(data=summary.__dict__).metrics

        frame = getDebugFrame(self._display, "calibrate")
        if frame:
            displayAstrometry(
                sourceCat=sourceCat,
                exposure=exposure,
                matches=astromMatches,
                frame=frame,
                pause=False,
            )

        return pipeBase.Struct(
            sourceCat=sourceCat,
            astromMatches=astromMatches,
            matchMeta=matchMeta,
            outputExposure=exposure,
            outputCat=sourceCat,
            outputBackground=background,
            outputSummaryMetrics=summaryMetrics
        )

    def setMetadata(self, exposure, photoRes=None):
        """Set task and exposure metadata.

        Logs a warning continues if needed data is missing.

        Parameters
        ----------
        exposure : `lsst.afw.image.ExposureF`
            Exposure to set metadata on.
        photoRes : `lsst.pipe.base.Struct`, optional
            Result of running photoCal task.
        """
        if photoRes is None:
            return

        metadata = exposure.getMetadata()

        # convert zero-point to (mag/sec/adu) for task MAGZERO metadata
        try:
            exposureTime = exposure.getInfo().getVisitInfo().getExposureTime()
            magZero = photoRes.zp - 2.5*math.log10(exposureTime)
        except Exception:
            self.log.warning("Could not set normalized MAGZERO in header: no "
                             "exposure time")
            magZero = math.nan

        try:
            metadata.set('MAGZERO', magZero)
            metadata.set('MAGZERO_RMS', photoRes.sigma)
            metadata.set('MAGZERO_NOBJ', photoRes.ngood)
            metadata.set('COLORTERM1', 0.0)
            metadata.set('COLORTERM2', 0.0)
            metadata.set('COLORTERM3', 0.0)
        except Exception as e:
            self.log.warning("Could not set exposure metadata: %s", e)

    def copyIcSourceFields(self, icSourceCat, sourceCat):
        """Match sources in an icSourceCat and a sourceCat and copy fields.

        The fields copied are those specified by
        ``config.icSourceFieldsToCopy``.

        Parameters
        ----------
        icSourceCat : `lsst.afw.table.SourceCatalog`
            Catalog from which to copy fields.
        sourceCat : `lsst.afw.table.SourceCatalog`
            Catalog to which to copy fields.

        Raises
        ------
        RuntimeError
            Raised if any of the following occur:
            - icSourceSchema and icSourceKeys are not specified.
            - icSourceCat and sourceCat are not specified.
            - icSourceFieldsToCopy is empty.
        """
        if self.schemaMapper is None:
            raise RuntimeError("To copy icSource fields you must specify "
                               "icSourceSchema and icSourceKeys when "
                               "constructing this task")
        if icSourceCat is None or sourceCat is None:
            raise RuntimeError("icSourceCat and sourceCat must both be "
                               "specified")
        if len(self.config.icSourceFieldsToCopy) == 0:
            self.log.warning("copyIcSourceFields doing nothing because "
                             "icSourceFieldsToCopy is empty")
            return

        mc = afwTable.MatchControl()
        mc.findOnlyClosest = False  # return all matched objects
        matches = afwTable.matchXy(icSourceCat, sourceCat,
                                   self.config.matchRadiusPix, mc)
        if self.config.doDeblend:
            deblendKey = sourceCat.schema["deblend_nChild"].asKey()
            # if deblended, keep children
            matches = [m for m in matches if m[1].get(deblendKey) == 0]

        # Because we had to allow multiple matches to handle parents, we now
        # need to prune to the best matches
        # closest matches as a dict of icSourceCat source ID:
        # (icSourceCat source, sourceCat source, distance in pixels)
        bestMatches = {}
        for m0, m1, d in matches:
            id0 = m0.getId()
            match = bestMatches.get(id0)
            if match is None or d <= match[2]:
                bestMatches[id0] = (m0, m1, d)
        matches = list(bestMatches.values())

        # Check that no sourceCat sources are listed twice (we already know
        # that each match has a unique icSourceCat source ID, due to using
        # that ID as the key in bestMatches)
        numMatches = len(matches)
        numUniqueSources = len(set(m[1].getId() for m in matches))
        if numUniqueSources != numMatches:
            self.log.warning("%d icSourceCat sources matched only %d sourceCat "
                             "sources", numMatches, numUniqueSources)

        self.log.info("Copying flags from icSourceCat to sourceCat for "
                      "%d sources", numMatches)

        # For each match: set the calibSourceKey flag and copy the desired
        # fields
        for icSrc, src, d in matches:
            src.setFlag(self.calibSourceKey, True)
            # src.assign copies the footprint from icSrc, which we don't want
            # (DM-407)
            # so set icSrc's footprint to src's footprint before src.assign,
            # then restore it
            icSrcFootprint = icSrc.getFootprint()
            try:
                icSrc.setFootprint(src.getFootprint())
                src.assign(icSrc, self.schemaMapper)
            finally:
                icSrc.setFootprint(icSrcFootprint)
