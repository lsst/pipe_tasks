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

__all__ = [
    "CalibrateImageTask",
    "CalibrateImageConfig",
    # "NoCalibrateImageTask",
    # "NoCalibrateImageConfig",
    "NoPsfStarsToStarsMatchError",
]

import numpy as np

import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.meas.algorithms
import lsst.meas.algorithms.installGaussianPsf
import lsst.meas.algorithms.measureApCorr
import lsst.meas.algorithms.setPrimaryFlags
import lsst.meas.base
import lsst.meas.astrom
import lsst.meas.deblender
import lsst.meas.extensions.shapeHSM
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import connectionTypes
from lsst.utils.timer import timeMethod

from . import measurePsf, repair, photoCal, computeExposureSummaryStats, snapCombine


class NoPsfStarsToStarsMatchError(pipeBase.AlgorithmError):
    """Raised when there are no matches between the psf_stars and stars
    catalogs.
    """
    def __init__(self, *, n_psf_stars, n_stars):
        msg = (f"No psf stars out of {n_psf_stars} matched {n_stars} calib stars."
               " Downstream processes probably won't have useful stars in this case."
               " Is `star_source_selector` too strict or is this a bad image?")
        super().__init__(msg)
        self.n_psf_stars = n_psf_stars
        self.n_stars = n_stars

    @property
    def metadata(self):
        return {"n_psf_stars": self.n_psf_stars,
                "n_stars": self.n_stars
                }


class CalibrateImageConfigBase(pexConfig.Config):
    """Base class for CalibrateImageConfig and NoCalibrateImageConfig."""
    # To generate catalog ids consistently across subtasks.
    id_generator = lsst.meas.base.DetectorVisitIdGeneratorConfig.make_field()

    snap_combine = pexConfig.ConfigurableField(
        target=snapCombine.SnapCombineTask,
        doc="Task to combine two snaps to make one exposure.",
    )

    # subtasks used during psf characterization
    install_simple_psf = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.installGaussianPsf.InstallGaussianPsfTask,
        doc="Task to install a simple PSF model into the input exposure to use "
            "when detecting bright sources for PSF estimation.",
    )
    psf_repair = pexConfig.ConfigurableField(
        target=repair.RepairTask,
        doc="Task to repair cosmic rays on the exposure before PSF determination.",
    )
    psf_subtract_background = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.SubtractBackgroundTask,
        doc="Task to perform intial background subtraction, before first detection pass.",
    )
    psf_detection = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.SourceDetectionTask,
        doc="Task to detect sources for PSF determination."
    )
    psf_source_measurement = pexConfig.ConfigurableField(
        target=lsst.meas.base.SingleFrameMeasurementTask,
        doc="Task to measure sources to be used for psf estimation."
    )
    psf_measure_psf = pexConfig.ConfigurableField(
        target=measurePsf.MeasurePsfTask,
        doc="Task to measure the psf on bright sources."
    )
    psf_normalized_calibration_flux = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.NormalizedCalibrationFluxTask,
        doc="Task to normalize the calibration flux (e.g. compensated tophats) "
            "for the bright stars used for psf estimation.",
    )

    # TODO DM-39203: we can remove aperture correction from this task once we are
    # using the shape-based star/galaxy code.
    measure_aperture_correction = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.measureApCorr.MeasureApCorrTask,
        doc="Task to compute the aperture correction from the bright stars."
    )

    # subtasks used during star measurement
    star_detection = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.SourceDetectionTask,
        doc="Task to detect stars to return in the output catalog."
    )
    star_sky_sources = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.SkyObjectsTask,
        doc="Task to generate sky sources ('empty' regions where there are no detections).",
    )
    star_deblend = pexConfig.ConfigurableField(
        target=lsst.meas.deblender.SourceDeblendTask,
        doc="Split blended sources into their components."
    )
    star_measurement = pexConfig.ConfigurableField(
        target=lsst.meas.base.SingleFrameMeasurementTask,
        doc="Task to measure stars to return in the output catalog."
    )
    star_normalized_calibration_flux = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.NormalizedCalibrationFluxTask,
        doc="Task to apply the normalization for calibration fluxes (e.g. compensated tophats) "
            "for the final output star catalog.",
    )
    star_apply_aperture_correction = pexConfig.ConfigurableField(
        target=lsst.meas.base.ApplyApCorrTask,
        doc="Task to apply aperture corrections to the selected stars."
    )
    star_catalog_calculation = pexConfig.ConfigurableField(
        target=lsst.meas.base.CatalogCalculationTask,
        doc="Task to compute extendedness values on the star catalog, "
            "for the star selector to remove extended sources."
    )
    star_set_primary_flags = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.setPrimaryFlags.SetPrimaryFlagsTask,
        doc="Task to add isPrimary to the catalog."
    )
    star_selector = lsst.meas.algorithms.sourceSelectorRegistry.makeField(
        default="science",
        doc="Task to select reliable stars to use for calibration."
    )

    compute_summary_stats = pexConfig.ConfigurableField(
        target=computeExposureSummaryStats.ComputeExposureSummaryStatsTask,
        doc="Task to to compute summary statistics on the calibrated exposure."
    )

    def setDefaults(self):
        super().setDefaults()

        # Use a very broad PSF here, to throughly reject CRs.
        # TODO investigation: a large initial psf guess may make stars look
        # like CRs for very good seeing images.
        self.install_simple_psf.fwhm = 4

        # S/N>=50 sources for PSF determination, but detection to S/N=5.
        # The thresholdValue sets the minimum flux in a pixel to be included in the
        # footprint, while peaks are only detected when they are above
        # thresholdValue * includeThresholdMultiplier. The low thresholdValue
        # ensures that the footprints are large enough for the noise replacer
        # to mask out faint undetected neighbors that are not to be measured.
        self.psf_detection.thresholdValue = 5.0
        self.psf_detection.includeThresholdMultiplier = 10.0
        # TODO investigation: Probably want False here, but that may require
        # tweaking the background spatial scale, to make it small enough to
        # prevent extra peaks in the wings of bright objects.
        self.psf_detection.doTempLocalBackground = False
        # NOTE: we do want reEstimateBackground=True in psf_detection, so that
        # each measurement step is done with the best background available.

        # Minimal measurement plugins for PSF determination.
        # TODO DM-39203: We can drop GaussianFlux and PsfFlux, if we use
        # shapeHSM/moments for star/galaxy separation.
        # TODO DM-39203: we can remove aperture correction from this task once
        # we are using the shape-based star/galaxy code.
        self.psf_source_measurement.plugins = ["base_PixelFlags",
                                               "base_SdssCentroid",
                                               "ext_shapeHSM_HsmSourceMoments",
                                               "base_CircularApertureFlux",
                                               "base_GaussianFlux",
                                               "base_PsfFlux",
                                               "base_CompensatedTophatFlux",
                                               ]
        self.psf_source_measurement.slots.shape = "ext_shapeHSM_HsmSourceMoments"
        # Only measure apertures we need for PSF measurement.
        self.psf_source_measurement.plugins["base_CircularApertureFlux"].radii = [12.0]
        self.psf_source_measurement.plugins["base_CompensatedTophatFlux"].apertures = [12]
        # TODO DM-40843: Remove this line once this is the psfex default.
        self.psf_measure_psf.psfDeterminer["psfex"].photometricFluxField = \
            "base_CircularApertureFlux_12_0_instFlux"

        # No extendeness information available: we need the aperture
        # corrections to determine that.
        self.measure_aperture_correction.sourceSelector["science"].doUnresolved = False
        self.measure_aperture_correction.sourceSelector["science"].flags.good = ["calib_psf_used"]
        self.measure_aperture_correction.sourceSelector["science"].flags.bad = []

        # Detection for good S/N for astrometry/photometry and other
        # downstream tasks; detection mask to S/N>=5, but S/N>=10 peaks.
        self.star_detection.thresholdValue = 5.0
        self.star_detection.includeThresholdMultiplier = 2.0
        self.star_measurement.plugins = ["base_PixelFlags",
                                         "base_SdssCentroid",
                                         "ext_shapeHSM_HsmSourceMoments",
                                         'ext_shapeHSM_HsmPsfMoments',
                                         "base_GaussianFlux",
                                         "base_PsfFlux",
                                         "base_CircularApertureFlux",
                                         "base_ClassificationSizeExtendedness",
                                         "base_CompensatedTophatFlux",
                                         ]
        self.star_measurement.slots.psfShape = "ext_shapeHSM_HsmPsfMoments"
        self.star_measurement.slots.shape = "ext_shapeHSM_HsmSourceMoments"
        # Only measure the apertures we need for star selection.
        self.star_measurement.plugins["base_CircularApertureFlux"].radii = [12.0]
        self.star_measurement.plugins["base_CompensatedTophatFlux"].apertures = [12]

        # We measure and apply the normalization aperture correction with the
        # psf_normalized_calibration_flux task, and we only apply the normalization
        # aperture correction for the full list of stars.
        self.star_normalized_calibration_flux.do_measure_ap_corr = False

        # Select stars with reliable measurements and no bad flags.
        self.star_selector["science"].doFlags = True
        self.star_selector["science"].doUnresolved = True
        self.star_selector["science"].doSignalToNoise = True
        self.star_selector["science"].signalToNoise.minimum = 10.0
        # Keep sky sources in the output catalog, even though they aren't
        # wanted for calibration.
        self.star_selector["science"].doSkySources = True

        # TODO: These should both be changed to calib_psf_used with DM-41640.
        self.compute_summary_stats.starSelection = "calib_photometry_used"
        self.compute_summary_stats.starSelector.flags.good = ["calib_photometry_used"]

    def validate(self):
        super().validate()

        # Ensure that the normalization calibration flux tasks
        # are configured correctly.
        if not self.psf_normalized_calibration_flux.do_measure_ap_corr:
            msg = ("psf_normalized_calibration_flux task must be configured with do_measure_ap_corr=True "
                   "or else the normalization and calibration flux will not be properly measured.")
            raise pexConfig.FieldValidationError(
                CalibrateImageConfig.psf_normalized_calibration_flux, self, msg,
            )
        if self.star_normalized_calibration_flux.do_measure_ap_corr:
            msg = ("star_normalized_calibration_flux task must be configured with do_measure_ap_corr=False "
                   "to apply the previously measured normalization to the full catalog of calibration "
                   "fluxes.")
            raise pexConfig.FieldValidationError(
                CalibrateImageConfig.star_normalized_calibration_flux, self, msg,
            )


class CalibrateImageConnectionsBase(pipeBase.PipelineTaskConnections,
                                    dimensions=("instrument", "visit", "detector")):
    exposures = connectionTypes.Input(
        doc="Exposure (or two snaps) to be calibrated, and detected and measured on.",
        name="postISRCCD",
        storageClass="Exposure",
        multiple=True,  # to handle 1 exposure or 2 snaps
        dimensions=["instrument", "exposure", "detector"],
    )

    # outputs
    initial_stars_schema = connectionTypes.InitOutput(
        doc="Schema of the output initial stars catalog.",
        name="initial_stars_schema",
        storageClass="SourceCatalog",
    )

    # TODO DM-38732: We want some kind of flag on Exposures/Catalogs to make
    # it obvious which components had failed to be computed/persisted.
    exposure = connectionTypes.Output(
        doc="Photometrically calibrated, background-subtracted exposure with fitted calibrations and "
            "summary statistics. To recover the original exposure, first add the background "
            "(`initial_pvi_background`), and then uncalibrate (divide by `initial_photoCalib_detector`).",
        name="initial_pvi",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
    )
    stars = connectionTypes.Output(
        doc="Catalog of unresolved sources detected on the calibrated exposure.",
        name="initial_stars_detector",
        storageClass="ArrowAstropy",
        dimensions=["instrument", "visit", "detector"],
    )
    stars_footprints = connectionTypes.Output(
        doc="Catalog of unresolved sources detected on the calibrated exposure; "
            "includes source footprints.",
        name="initial_stars_footprints_detector",
        storageClass="SourceCatalog",
        dimensions=["instrument", "visit", "detector"],
    )

    # Optional outputs
    psf_stars_footprints = connectionTypes.Output(
        doc="Catalog of bright unresolved sources detected on the exposure used for PSF determination; "
            "includes source footprints.",
        name="initial_psf_stars_footprints_detector",
        storageClass="SourceCatalog",
        dimensions=["instrument", "visit", "detector"],
    )
    psf_stars = connectionTypes.Output(
        doc="Catalog of bright unresolved sources detected on the exposure used for PSF determination.",
        name="initial_psf_stars_detector",
        storageClass="ArrowAstropy",
        dimensions=["instrument", "visit", "detector"],
    )


class CalibrateImageConnections(CalibrateImageConnectionsBase):
    astrometry_ref_cat = connectionTypes.PrerequisiteInput(
        doc="Reference catalog to use for astrometric calibration.",
        name="gaia_dr3_20230707",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        deferLoad=True,
        multiple=True,
    )
    photometry_ref_cat = connectionTypes.PrerequisiteInput(
        doc="Reference catalog to use for photometric calibration.",
        name="ps1_pv3_3pi_20170110",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        deferLoad=True,
        multiple=True
    )

    # Outputs
    applied_photo_calib = connectionTypes.Output(
        doc="Photometric calibration that was applied to exposure.",
        name="initial_photoCalib_detector",
        storageClass="PhotoCalib",
        dimensions=("instrument", "visit", "detector"),
    )

    background = connectionTypes.Output(
        doc="Background models estimated during calibration task; calibrated to be in nJy units.",
        name="initial_pvi_background",
        storageClass="Background",
        dimensions=("instrument", "visit", "detector"),
    )

    # Optional outputs
    astrometry_matches = connectionTypes.Output(
        doc="Source to reference catalog matches from the astrometry solver.",
        name="initial_astrometry_match_detector",
        storageClass="Catalog",
        dimensions=("instrument", "visit", "detector"),
    )
    photometry_matches = connectionTypes.Output(
        doc="Source to reference catalog matches from the photometry solver.",
        name="initial_photometry_match_detector",
        storageClass="Catalog",
        dimensions=("instrument", "visit", "detector"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if config.optional_outputs is None or "psf_stars" not in config.optional_outputs:
            del self.psf_stars
        if config.optional_outputs is None or "psf_stars_footprints" not in config.optional_outputs:
            del self.psf_stars_footprints
        if config.optional_outputs is None or "astrometry_matches" not in config.optional_outputs:
            del self.astrometry_matches
        if config.optional_outputs is None or "photometry_matches" not in config.optional_outputs:
            del self.photometry_matches


class CalibrateImageConfig(
    CalibrateImageConfigBase,
    pipeBase.PipelineTaskConfig,
    pipelineConnections=CalibrateImageConnections
):
    optional_outputs = pexConfig.ListField(
        doc="Which optional outputs to save (as their connection name)?"
            " If None, do not output any of these datasets.",
        dtype=str,
        # TODO: note somewhere to disable this for benchmarking, but should
        # we always have it on for production runs?
        default=["psf_stars", "psf_stars_footprints", "astrometry_matches", "photometry_matches"],
        optional=True
    )

    # final calibrations and statistics
    astrometry = pexConfig.ConfigurableField(
        target=lsst.meas.astrom.AstrometryTask,
        doc="Task to perform astrometric calibration to fit a WCS.",
    )
    astrometry_ref_loader = pexConfig.ConfigField(
        dtype=lsst.meas.algorithms.LoadReferenceObjectsConfig,
        doc="Configuration of reference object loader for astrometric fit.",
    )
    photometry = pexConfig.ConfigurableField(
        target=photoCal.PhotoCalTask,
        doc="Task to perform photometric calibration to fit a PhotoCalib.",
    )
    photometry_ref_loader = pexConfig.ConfigField(
        dtype=lsst.meas.algorithms.LoadReferenceObjectsConfig,
        doc="Configuration of reference object loader for photometric fit.",
    )

    def setDefaults(self):
        super().setDefaults()

        # Use the affine WCS fitter (assumes we have a good camera geometry).
        self.astrometry.wcsFitter.retarget(lsst.meas.astrom.FitAffineWcsTask)
        # phot_g_mean is the primary Gaia band for all input bands.
        self.astrometry_ref_loader.anyFilterMapsToThis = "phot_g_mean"

        # Only reject sky sources; we already selected good stars.
        self.astrometry.sourceSelector["science"].doFlags = True
        self.astrometry.sourceSelector["science"].flags.bad = ["sky_source"]
        self.photometry.match.sourceSelection.doFlags = True
        self.photometry.match.sourceSelection.flags.bad = ["sky_source"]
        # Unset the (otherwise reasonable, but we've already made the
        # selections we want above) selection settings in PhotoCalTask.
        self.photometry.match.sourceSelection.doRequirePrimary = False
        self.photometry.match.sourceSelection.doUnresolved = False


class CalibrateImageTask(pipeBase.PipelineTask):
    """Compute the PSF, aperture corrections, astrometric and photometric
    calibrations, and summary statistics for a single science exposure, and
    produce a catalog of brighter stars that were used to calibrate it.

    Parameters
    ----------
    initial_stars_schema : `lsst.afw.table.Schema`
        Schema of the initial_stars output catalog.
    """
    _DefaultName = "calibrateImage"
    ConfigClass = CalibrateImageConfig

    def __init__(self, initial_stars_schema=None, **kwargs):
        super().__init__(**kwargs)

        self.makeSubtask("snap_combine")

        # PSF determination subtasks
        self.makeSubtask("install_simple_psf")
        self.makeSubtask("psf_repair")
        self.makeSubtask("psf_subtract_background")
        self.psf_schema = afwTable.SourceTable.makeMinimalSchema()
        self.makeSubtask("psf_detection", schema=self.psf_schema)
        self.makeSubtask("psf_source_measurement", schema=self.psf_schema)
        self.makeSubtask("psf_measure_psf", schema=self.psf_schema)
        self.makeSubtask("psf_normalized_calibration_flux", schema=self.psf_schema)

        self.makeSubtask("measure_aperture_correction", schema=self.psf_schema)

        # star measurement subtasks
        if initial_stars_schema is None:
            initial_stars_schema = afwTable.SourceTable.makeMinimalSchema()

        # These fields let us track which sources were used for psf and
        # aperture correction calculations.
        self.psf_fields = ("calib_psf_candidate", "calib_psf_used", "calib_psf_reserved",
                           # TODO DM-39203: these can be removed once apcorr is gone.
                           "apcorr_slot_CalibFlux_used", "apcorr_base_GaussianFlux_used",
                           "apcorr_base_PsfFlux_used")
        for field in self.psf_fields:
            item = self.psf_schema.find(field)
            initial_stars_schema.addField(item.getField())

        afwTable.CoordKey.addErrorFields(initial_stars_schema)
        self.makeSubtask("star_detection", schema=initial_stars_schema)
        self.makeSubtask("star_sky_sources", schema=initial_stars_schema)
        self.makeSubtask("star_deblend", schema=initial_stars_schema)
        self.makeSubtask("star_measurement", schema=initial_stars_schema)
        self.makeSubtask("star_normalized_calibration_flux", schema=initial_stars_schema)

        self.makeSubtask("star_apply_aperture_correction", schema=initial_stars_schema)
        self.makeSubtask("star_catalog_calculation", schema=initial_stars_schema)
        self.makeSubtask("star_set_primary_flags", schema=initial_stars_schema, isSingleFrame=True)
        self.makeSubtask("star_selector")

        self.makeSubtask("astrometry", schema=initial_stars_schema)
        self.makeSubtask("photometry", schema=initial_stars_schema)

        self.makeSubtask("compute_summary_stats")

        # For the butler to persist it.
        self.initial_stars_schema = afwTable.SourceCatalog(initial_stars_schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        exposures = inputs.pop("exposures")

        id_generator = self.config.id_generator.apply(butlerQC.quantum.dataId)

        astrometry_loader = lsst.meas.algorithms.ReferenceObjectLoader(
            dataIds=[ref.datasetRef.dataId for ref in inputRefs.astrometry_ref_cat],
            refCats=inputs.pop("astrometry_ref_cat"),
            name=self.config.connections.astrometry_ref_cat,
            config=self.config.astrometry_ref_loader, log=self.log)
        self.astrometry.setRefObjLoader(astrometry_loader)

        photometry_loader = lsst.meas.algorithms.ReferenceObjectLoader(
            dataIds=[ref.datasetRef.dataId for ref in inputRefs.photometry_ref_cat],
            refCats=inputs.pop("photometry_ref_cat"),
            name=self.config.connections.photometry_ref_cat,
            config=self.config.photometry_ref_loader, log=self.log)
        self.photometry.match.setRefObjLoader(photometry_loader)

        # This should not happen with a properly configured execution context.
        assert not inputs, "runQuantum got more inputs than expected"

        # Specify the fields that `annotate` needs below, to ensure they
        # exist, even as None.
        result = pipeBase.Struct(exposure=None,
                                 stars_footprints=None,
                                 psf_stars_footprints=None,
                                 )
        try:
            self.run(exposures=exposures, result=result, id_generator=id_generator)
        except pipeBase.AlgorithmError as e:
            error = pipeBase.AnnotatedPartialOutputsError.annotate(
                e,
                self,
                result.exposure,
                result.psf_stars_footprints,
                result.stars_footprints,
                log=self.log
            )
            butlerQC.put(result, outputRefs)
            raise error from e

        butlerQC.put(result, outputRefs)

    @timeMethod
    def run(self, *, exposures, id_generator=None, result=None):
        """Find stars and perform psf measurement, then do a deeper detection
        and measurement and calibrate astrometry and photometry from that.

        Parameters
        ----------
        exposures : `lsst.afw.image.Exposure` or `list` [`lsst.afw.image.Exposure`]
            Post-ISR exposure(s), with an initial WCS, VisitInfo, and Filter.
            Modified in-place during processing if only one is passed.
            If two exposures are passed, treat them as snaps and combine
            before doing further processing.
        id_generator : `lsst.meas.base.IdGenerator`, optional
            Object that generates source IDs and provides random seeds.
        result : `lsst.pipe.base.Struct`, optional
            Result struct that is modified to allow saving of partial outputs
            for some failure conditions. If the task completes successfully,
            this is also returned.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``exposure``
                Calibrated exposure, with pixels in nJy units.
                (`lsst.afw.image.Exposure`)
            ``stars``
                Stars that were used to calibrate the exposure, with
                calibrated fluxes and magnitudes.
                (`astropy.table.Table`)
            ``stars_footprints``
                Footprints of stars that were used to calibrate the exposure.
                (`lsst.afw.table.SourceCatalog`)
            ``psf_stars``
                Stars that were used to determine the image PSF.
                (`astropy.table.Table`)
            ``psf_stars_footprints``
                Footprints of stars that were used to determine the image PSF.
                (`lsst.afw.table.SourceCatalog`)
            ``background``
                Background that was fit to the exposure when detecting
                ``stars``. (`lsst.afw.math.BackgroundList`)
            ``applied_photo_calib``
                Photometric calibration that was fit to the star catalog and
                applied to the exposure. (`lsst.afw.image.PhotoCalib`)
            ``astrometry_matches``
                Reference catalog stars matches used in the astrometric fit.
                (`list` [`lsst.afw.table.ReferenceMatch`] or `lsst.afw.table.BaseCatalog`)
            ``photometry_matches``
                Reference catalog stars matches used in the photometric fit.
                (`list` [`lsst.afw.table.ReferenceMatch`] or `lsst.afw.table.BaseCatalog`)
        """
        if result is None:
            result = pipeBase.Struct()
        if id_generator is None:
            id_generator = lsst.meas.base.IdGenerator()

        result.exposure = self.snap_combine.run(exposures).exposure

        result.psf_stars_footprints, result.background, candidates = self._compute_psf(result.exposure,
                                                                                       id_generator)
        self._measure_aperture_correction(result.exposure, result.psf_stars_footprints)

        result.psf_stars = result.psf_stars_footprints.asAstropy()

        result.stars_footprints = self._find_stars(result.exposure, result.background, id_generator)
        self._match_psf_stars(result.psf_stars_footprints, result.stars_footprints)
        result.stars = result.stars_footprints.asAstropy()

        astrometry_matches, astrometry_meta = self._fit_astrometry(result.exposure, result.stars_footprints)
        if self.config.optional_outputs is not None and "astrometry_matches" in self.config.optional_outputs:
            result.astrometry_matches = lsst.meas.astrom.denormalizeMatches(astrometry_matches,
                                                                            astrometry_meta)

        result.stars_footprints, photometry_matches, \
            photometry_meta, result.applied_photo_calib = self._fit_photometry(result.exposure,
                                                                               result.stars_footprints,
                                                                               result.background)
        # fit_photometry returns a new catalog, so we need a new astropy table view.
        result.stars = result.stars_footprints.asAstropy()
        if self.config.optional_outputs is not None and "photometry_matches" in self.config.optional_outputs:
            result.photometry_matches = lsst.meas.astrom.denormalizeMatches(photometry_matches,
                                                                            photometry_meta)

        self._summarize(result.exposure, result.stars_footprints, result.background)

        return result

    def _compute_psf(self, exposure, id_generator):
        """Find bright sources detected on an exposure and fit a PSF model to
        them, repairing likely cosmic rays before detection.

        Repair, detect, measure, and compute PSF twice, to ensure the PSF
        model does not include contributions from cosmic rays.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to detect and measure bright stars on.
        id_generator : `lsst.meas.base.IdGenerator`, optional
            Object that generates source IDs and provides random seeds.

        Returns
        -------
        sources : `lsst.afw.table.SourceCatalog`
            Catalog of detected bright sources.
        background : `lsst.afw.math.BackgroundList`
            Background that was fit to the exposure during detection.
        cell_set : `lsst.afw.math.SpatialCellSet`
            PSF candidates returned by the psf determiner.
        """
        def log_psf(msg):
            """Log the parameters of the psf and background, with a prepended
            message.
            """
            position = exposure.psf.getAveragePosition()
            sigma = exposure.psf.computeShape(position).getDeterminantRadius()
            dimensions = exposure.psf.computeImage(position).getDimensions()
            median_background = np.median(background.getImage().array)
            self.log.info("%s sigma=%0.4f, dimensions=%s; median background=%0.2f",
                          msg, sigma, dimensions, median_background)

        self.log.info("First pass detection with Guassian PSF FWHM=%s pixels",
                      self.config.install_simple_psf.fwhm)
        self.install_simple_psf.run(exposure=exposure)

        background = self.psf_subtract_background.run(exposure=exposure).background
        log_psf("Initial PSF:")
        self.psf_repair.run(exposure=exposure, keepCRs=True)

        table = afwTable.SourceTable.make(self.psf_schema, id_generator.make_table_id_factory())
        # Re-estimate the background during this detection step, so that
        # measurement uses the most accurate background-subtraction.
        detections = self.psf_detection.run(table=table, exposure=exposure, background=background)
        self.psf_source_measurement.run(detections.sources, exposure)
        psf_result = self.psf_measure_psf.run(exposure=exposure, sources=detections.sources)
        # Replace the initial PSF with something simpler for the second
        # repair/detect/measure/measure_psf step: this can help it converge.
        self.install_simple_psf.run(exposure=exposure)

        log_psf("Rerunning with simple PSF:")
        # TODO investigation: Should we only re-run repair here, to use the
        # new PSF? Maybe we *do* need to re-run measurement with PsfFlux, to
        # use the fitted PSF?
        # TODO investigation: do we need a separate measurement task here
        # for the post-psf_measure_psf step, since we only want to do PsfFlux
        # and GaussianFlux *after* we have a PSF? Maybe that's not relevant
        # once DM-39203 is merged?
        self.psf_repair.run(exposure=exposure, keepCRs=True)
        # Re-estimate the background during this detection step, so that
        # measurement uses the most accurate background-subtraction.
        detections = self.psf_detection.run(table=table, exposure=exposure, background=background)
        self.psf_source_measurement.run(detections.sources, exposure)
        psf_result = self.psf_measure_psf.run(exposure=exposure, sources=detections.sources)

        log_psf("Final PSF:")

        # Final repair with final PSF, removing cosmic rays this time.
        self.psf_repair.run(exposure=exposure)
        # Final measurement with the CRs removed.
        self.psf_source_measurement.run(detections.sources, exposure)

        # PSF is set on exposure; candidates are returned to use for
        # calibration flux normalization and aperture corrections.
        return detections.sources, background, psf_result.cellSet

    def _measure_aperture_correction(self, exposure, bright_sources):
        """Measure and set the ApCorrMap on the Exposure, using
        previously-measured bright sources.

        This function first normalizes the calibration flux and then
        the full set of aperture corrections are measured relative
        to this normalized calibration flux.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to set the ApCorrMap on.
        bright_sources : `lsst.afw.table.SourceCatalog`
            Catalog of detected bright sources; modified to include columns
            necessary for point source determination for the aperture correction
            calculation.
        """
        norm_ap_corr_map = self.psf_normalized_calibration_flux.run(
            exposure=exposure,
            catalog=bright_sources,
        ).ap_corr_map

        ap_corr_map = self.measure_aperture_correction.run(exposure, bright_sources).apCorrMap

        # Need to merge the aperture correction map from the normalization.
        for key in norm_ap_corr_map:
            ap_corr_map[key] = norm_ap_corr_map[key]

        exposure.info.setApCorrMap(ap_corr_map)

    def _find_stars(self, exposure, background, id_generator):
        """Detect stars on an exposure that has a PSF model, and measure their
        PSF, circular aperture, compensated gaussian fluxes.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to detect and measure stars on.
        background : `lsst.afw.math.BackgroundList`
            Background that was fit to the exposure during detection;
            modified in-place during subsequent detection.
        id_generator : `lsst.meas.base.IdGenerator`
            Object that generates source IDs and provides random seeds.

        Returns
        -------
        stars : `SourceCatalog`
            Sources that are very likely to be stars, with a limited set of
            measurements performed on them.
        """
        table = afwTable.SourceTable.make(self.initial_stars_schema.schema,
                                          id_generator.make_table_id_factory())
        # Re-estimate the background during this detection step, so that
        # measurement uses the most accurate background-subtraction.
        detections = self.star_detection.run(table=table, exposure=exposure, background=background)
        sources = detections.sources
        self.star_sky_sources.run(exposure.mask, id_generator.catalog_id, sources)

        # TODO investigation: Could this deblender throw away blends of non-PSF sources?
        self.star_deblend.run(exposure=exposure, sources=sources)
        # The deblender may not produce a contiguous catalog; ensure
        # contiguity for subsequent tasks.
        if not sources.isContiguous():
            sources = sources.copy(deep=True)

        # Measure everything, and use those results to select only stars.
        self.star_measurement.run(sources, exposure)
        # Run the normalization calibration flux task to apply the
        # normalization correction to create normalized
        # calibration fluxes.
        self.star_normalized_calibration_flux.run(exposure=exposure, catalog=sources)
        self.star_apply_aperture_correction.run(sources, exposure.apCorrMap)
        self.star_catalog_calculation.run(sources)
        self.star_set_primary_flags.run(sources)

        result = self.star_selector.run(sources)
        # The star selector may not produce a contiguous catalog.
        if not result.sourceCat.isContiguous():
            return result.sourceCat.copy(deep=True)
        else:
            return result.sourceCat

    def _match_psf_stars(self, psf_stars, stars):
        """Match calibration stars to psf stars, to identify which were psf
        candidates, and which were used or reserved during psf measurement.

        Parameters
        ----------
        psf_stars : `lsst.afw.table.SourceCatalog`
            PSF candidate stars that were sent to the psf determiner. Used to
            populate psf-related flag fields.
        stars : `lsst.afw.table.SourceCatalog`
            Stars that will be used for calibration; psf-related fields will
            be updated in-place.

        Notes
        -----
        This code was adapted from CalibrateTask.copyIcSourceFields().
        """
        control = afwTable.MatchControl()
        # Return all matched objects, to separate blends.
        control.findOnlyClosest = False
        matches = afwTable.matchXy(psf_stars, stars, 3.0, control)
        deblend_key = stars.schema["deblend_nChild"].asKey()
        matches = [m for m in matches if m[1].get(deblend_key) == 0]

        # Because we had to allow multiple matches to handle parents, we now
        # need to prune to the best (closest) matches.
        # Closest matches is a dict of psf_stars source ID to Match record
        # (psf_stars source, sourceCat source, distance in pixels).
        best = {}
        for match_psf, match_stars, d in matches:
            match = best.get(match_psf.getId())
            if match is None or d <= match[2]:
                best[match_psf.getId()] = (match_psf, match_stars, d)
        matches = list(best.values())
        # We'll use this to construct index arrays into each catalog.
        ids = np.array([(match_psf.getId(), match_stars.getId()) for match_psf, match_stars, d in matches]).T

        if (n_matches := len(matches)) == 0:
            raise NoPsfStarsToStarsMatchError(n_psf_stars=len(psf_stars), n_stars=len(stars))

        self.log.info("%d psf stars out of %d matched %d calib stars", n_matches, len(psf_stars), len(stars))

        # Check that no stars sources are listed twice; we already know
        # that each match has a unique psf_stars id, due to using as the key
        # in best above.
        n_unique = len(set(m[1].getId() for m in matches))
        if n_unique != n_matches:
            self.log.warning("%d psf_stars matched only %d stars", n_matches, n_unique)

        # The indices of the IDs, so we can update the flag fields as arrays.
        idx_psf_stars = np.searchsorted(psf_stars["id"], ids[0])
        idx_stars = np.searchsorted(stars["id"], ids[1])
        for field in self.psf_fields:
            result = np.zeros(len(stars), dtype=bool)
            result[idx_stars] = psf_stars[field][idx_psf_stars]
            stars[field] = result

    def _fit_astrometry(self, exposure, stars):
        """Fit an astrometric model to the data and return the reference
        matches used in the fit, and the fitted WCS.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure that is being fit, to get PSF and other metadata from.
            Modified to add the fitted skyWcs.
        stars : `SourceCatalog`
            Good stars selected for use in calibration, with RA/Dec coordinates
            computed from the pixel positions and fitted WCS.

        Returns
        -------
        matches : `list` [`lsst.afw.table.ReferenceMatch`]
            Reference/stars matches used in the fit.
        """
        result = self.astrometry.run(stars, exposure)
        return result.matches, result.matchMeta

    def _fit_photometry(self, exposure, stars, background):
        """Fit a photometric model to the data and return the reference
        matches used in the fit, and the fitted PhotoCalib.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure that is being fit, to get PSF and other metadata from.
            Modified to be in nanojanksy units, with an assigned photoCalib
            identically 1.
        stars : `lsst.afw.table.SourceCatalog`
            Good stars selected for use in calibration.

        Returns
        -------
        calibrated_stars : `lsst.afw.table.SourceCatalog`
            Star catalog with flux/magnitude columns computed from the fitted
            photoCalib.
        matches : `list` [`lsst.afw.table.ReferenceMatch`]
            Reference/stars matches used in the fit.
        photoCalib : `lsst.afw.image.PhotoCalib`
            Photometric calibration that was fit to the star catalog.
        """
        result = self.photometry.run(exposure, stars)
        calibrated_stars = result.photoCalib.calibrateCatalog(stars)
        exposure.maskedImage = result.photoCalib.calibrateImage(exposure.maskedImage)
        identity = afwImage.PhotoCalib(1.0,
                                       result.photoCalib.getCalibrationErr(),
                                       bbox=exposure.getBBox())
        exposure.setPhotoCalib(identity)

        assert result.photoCalib._isConstant, \
            "Background calibration assumes a constant PhotoCalib; PhotoCalTask should always return that."
        for bg in background:
            # The statsImage is a view, but we can't assign to a function call in python.
            binned_image = bg[0].getStatsImage()
            binned_image *= result.photoCalib.getCalibrationMean()

        return calibrated_stars, result.matches, result.matchMeta, result.photoCalib

    def _summarize(self, exposure, stars, background):
        """Compute summary statistics on the exposure and update in-place the
        calibrations attached to it.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure that was calibrated, to get PSF and other metadata from.
            Modified to contain the computed summary statistics.
        stars : `SourceCatalog`
            Good stars selected used in calibration.
        background : `lsst.afw.math.BackgroundList`
            Background that was fit to the exposure during detection of the
            above stars.
        """
        # TODO investigation: because this takes the photoCalib from the
        # exposure, photometric summary values may be "incorrect" (i.e. they
        # will reflect the ==1 nJy calibration on the exposure, not the
        # applied calibration). This needs to be checked.
        summary = self.compute_summary_stats.run(exposure, stars, background)
        exposure.info.setSummaryStats(summary)
