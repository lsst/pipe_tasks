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

import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.meas.algorithms
import lsst.meas.algorithms.installGaussianPsf
import lsst.meas.algorithms.measureApCorr
from lsst.meas.algorithms import sourceSelector
import lsst.meas.astrom
import lsst.meas.deblender
import lsst.meas.extensions.shapeHSM
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import connectionTypes
from lsst.utils.timer import timeMethod

from . import measurePsf, repair, setPrimaryFlags, photoCal, computeExposureSummaryStats


class CalibrateImageConnections(pipeBase.PipelineTaskConnections,
                                dimensions=("instrument", "visit", "detector")):

    astrometry_ref_cat = connectionTypes.PrerequisiteInput(
        doc="Reference catalog to use for astrometric calibration.",
        name="gaia_dr2_20200414",
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

    exposure = connectionTypes.Input(
        doc="Exposure to be calibrated, and detected and measured on.",
        name="postISRCCD",
        storageClass="Exposure",
        dimensions=["instrument", "exposure", "detector"],
    )

    # outputs
    initial_stars_schema = connectionTypes.InitOutput(
        doc="Schema of the output initial stars catalog.",
        name="initial_stars_schema",
        storageClass="SourceCatalog",
    )

    # TODO: We want some kind of flag on Exposures/Catalogs to make it obvious
    # which components had failed to be computed/persisted
    output_exposure = connectionTypes.Output(
        doc="Photometrically calibrated exposure with fitted calibrations and summary statistics.",
        name="initial_pvi",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
    )
    # TODO DM-40061: persist a parquet version of this!
    stars = connectionTypes.Output(
        doc="Catalog of unresolved sources detected on the calibrated exposure; "
            "includes source footprints.",
        name="initial_stars_footprints_detector",
        storageClass="SourceCatalog",
        dimensions=["instrument", "visit", "detector"],
    )
    applied_photo_calib = connectionTypes.Output(
        doc="Photometric calibration that was applied to exposure.",
        name="initial_photoCalib_detector",
        storageClass="PhotoCalib",
        dimensions=("instrument", "visit", "detector"),
    )
    background = connectionTypes.Output(
        doc="Background models estimated during calibration task.",
        name="initial_pvi_background",
        storageClass="Background",
        dimensions=("instrument", "visit", "detector"),
    )

    # Optional outputs

    # TODO: We need to decide on what intermediate outputs we want to save,
    # and which to save by default.
    # TODO DM-40061: persist a parquet version of this!
    psf_stars = connectionTypes.Output(
        doc="Catalog of bright unresolved sources detected on the exposure used for PSF determination; "
            "includes source footprints.",
        name="initial_psf_stars_footprints",
        storageClass="SourceCatalog",
        dimensions=["instrument", "visit", "detector"],
    )
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
        if not config.optional_outputs:
            self.outputs.remove("psf_stars")
            self.outputs.remove("astrometry_matches")
            self.outputs.remove("photometry_matches")


class CalibrateImageConfig(pipeBase.PipelineTaskConfig, pipelineConnections=CalibrateImageConnections):
    optional_outputs = pexConfig.ListField(
        doc="Which optional outputs to save (as their connection name)?",
        dtype=str,
        # TODO: note somewhere to disable this for benchmarking, but should
        # we always have it on for production runs?
        default=["psf_stars", "astrometry_matches", "photometry_matches"],
        optional=True
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
    star_deblend = pexConfig.ConfigurableField(
        target=lsst.meas.deblender.SourceDeblendTask,
        doc="Split blended sources into their components"
    )
    star_measurement = pexConfig.ConfigurableField(
        target=lsst.meas.base.SingleFrameMeasurementTask,
        doc="Task to measure stars to return in the output catalog."
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
        target=setPrimaryFlags.SetPrimaryFlagsTask,
        doc="Task to add isPrimary to the catalog."
    )
    star_selector = lsst.meas.algorithms.sourceSelectorRegistry.makeField(
        default="science",
        doc="Task to select isolated stars to use for calibration."
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

        # Only use high S/N sources for PSF determination.
        self.psf_detection.thresholdValue = 50.0
        self.psf_detection.thresholdType = "pixel_stdev"
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
                                               ]
        self.psf_source_measurement.slots.shape = "ext_shapeHSM_HsmSourceMoments"
        # Only measure apertures we need for PSF measurement.
        # TODO DM-40064: psfex has a hard-coded value of 9 in a psfex-config
        # file: make that configurable and/or change it to 12 to be consistent
        # with our other uses?
        # https://github.com/lsst/meas_extensions_psfex/blob/main/config/default-lsst.psfex#L14
        self.psf_source_measurement.plugins["base_CircularApertureFlux"].radii = [9.0, 12.0]

        self.psf_measure_psf.starSelector["objectSize"].doFluxLimit = False
        self.psf_measure_psf.starSelector["objectSize"].doSignalToNoiseLimit = True

        # No extendeness information available: we need the aperture
        # corrections to determine that.
        self.measure_aperture_correction.sourceSelector["science"].doUnresolved = False
        self.measure_aperture_correction.sourceSelector["science"].flags.good = ["calib_psf_used"]
        self.measure_aperture_correction.sourceSelector["science"].flags.bad = []

        # TODO investigation: how faint do we have to detect, to be able to
        # deblend, etc? We may need star_selector to have a separate value,
        # and do initial detection at S/N>5.0?
        # Detection for good S/N for astrometry/photometry and other
        # downstream tasks.
        self.star_detection.thresholdValue = 10.0
        self.star_detection.thresholdType = "pixel_stdev"
        self.star_measurement.plugins = ["base_PixelFlags",
                                         "base_SdssCentroid",
                                         "ext_shapeHSM_HsmSourceMoments",
                                         'ext_shapeHSM_HsmPsfMoments',
                                         "base_GaussianFlux",
                                         "base_PsfFlux",
                                         "base_CircularApertureFlux",
                                         ]
        self.star_measurement.slots.psfShape = "ext_shapeHSM_HsmPsfMoments"
        self.star_measurement.slots.shape = "ext_shapeHSM_HsmSourceMoments"
        # Only measure the apertures we need for star selection.
        self.star_measurement.plugins["base_CircularApertureFlux"].radii = [12.0]
        # Restrict footprint area to prevent memory blowup on huge footprints.
        self.star_deblend.maxFootprintArea = 10000

        # Select isolated stars with reliable measurements and no bad flags.
        self.star_selector["science"].doFlags = True
        self.star_selector["science"].doUnresolved = True
        self.star_selector["science"].doSignalToNoise = True
        self.star_selector["science"].doIsolated = True
        self.star_selector["science"].signalToNoise.minimum = 10.0

        # Use the affine WCS fitter (assumes we have a good camera geometry).
        self.astrometry.wcsFitter.retarget(lsst.meas.astrom.FitAffineWcsTask)
        # phot_g_mean is the primary Gaia band for all input bands.
        self.astrometry_ref_loader.anyFilterMapsToThis = "phot_g_mean"

        # Reject magnitude outliers (TODO DM-39796: should be task default)
        self.astrometry.doMagnitudeOutlierRejection = True

        # Do not subselect during fitting; we already selected good stars.
        self.astrometry.sourceSelector = "null"
        self.photometry.match.sourceSelection.retarget(sourceSelector.NullSourceSelectorTask)

        # All sources should be good for PSF summary statistics.
        self.compute_summary_stats.starSelection = "calib_photometry_used"


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

        # PSF determination subtasks
        self.makeSubtask("install_simple_psf")
        self.makeSubtask("psf_repair")
        self.makeSubtask("psf_subtract_background")
        self.psf_schema = afwTable.SourceTable.makeMinimalSchema()
        self.makeSubtask("psf_detection", schema=self.psf_schema)
        self.makeSubtask("psf_source_measurement", schema=self.psf_schema)
        self.makeSubtask("psf_measure_psf", schema=self.psf_schema)

        self.makeSubtask("measure_aperture_correction", schema=self.psf_schema)

        # star measurement subtasks
        if initial_stars_schema is None:
            initial_stars_schema = afwTable.SourceTable.makeMinimalSchema()
        afwTable.CoordKey.addErrorFields(initial_stars_schema)
        self.makeSubtask("star_detection", schema=initial_stars_schema)
        self.makeSubtask("star_deblend", schema=initial_stars_schema)
        self.makeSubtask("star_measurement", schema=initial_stars_schema)
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

        outputs = self.run(**inputs)

        butlerQC.put(outputs, outputRefs)

    @timeMethod
    def run(self, *, exposure):
        """Find stars and perform psf measurement, then do a deeper detection
        and measurement and calibrate astrometry and photometry from that.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Post-ISR exposure, with an initial WCS, VisitInfo, and Filter.
            Modified in-place during processing.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``output_exposure``
                Calibrated exposure, with pixels in nJy units.
                (`lsst.afw.image.Exposure`)
            ``stars``
                Stars that were used to calibrate the exposure, with
                calibrated fluxes and magnitudes.
                (`lsst.afw.table.SourceCatalog`)
            ``psf_stars``
                Stars that were used to determine the image PSF.
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
        psf_stars, background, candidates = self._compute_psf(exposure)

        self._measure_aperture_correction(exposure, psf_stars)

        stars = self._find_stars(exposure, background)

        astrometry_matches, astrometry_meta = self._fit_astrometry(exposure, stars)
        stars, photometry_matches, photometry_meta, photo_calib = self._fit_photometry(exposure, stars)

        self._summarize(exposure, stars, background)

        if self.config.optional_outputs:
            astrometry_matches = lsst.meas.astrom.denormalizeMatches(astrometry_matches, astrometry_meta)
            photometry_matches = lsst.meas.astrom.denormalizeMatches(photometry_matches, photometry_meta)

        return pipeBase.Struct(output_exposure=exposure,
                               stars=stars,
                               psf_stars=psf_stars,
                               background=background,
                               applied_photo_calib=photo_calib,
                               astrometry_matches=astrometry_matches,
                               photometry_matches=photometry_matches)

    def _compute_psf(self, exposure, guess_psf=True):
        """Find bright sources detected on an exposure and fit a PSF model to
        them, repairing likely cosmic rays before detection.

        Repair, detect, measure, and compute PSF twice, to ensure the PSF
        model does not include contributions from cosmic rays.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to detect and measure bright stars on.

        Returns
        -------
        sources : `lsst.afw.table.SourceCatalog`
            Catalog of detected bright sources.
        background : `lsst.afw.math.BackgroundList`
            Background that was fit to the exposure during detection.
        cell_set : `lsst.afw.math.SpatialCellSet`
            PSF candidates returned by the psf determiner.
        """
        self.log.info("First pass detection with Guassian PSF FWHM=%s", self.config.install_simple_psf.fwhm)
        self.install_simple_psf.run(exposure=exposure)

        background = self.psf_subtract_background.run(exposure=exposure).background
        self.psf_repair.run(exposure=exposure, keepCRs=True)

        table = afwTable.SourceTable.make(self.psf_schema)
        # Re-estimate the background during this detection step, so that
        # measurement uses the most accurate background-subtraction.
        detections = self.psf_detection.run(table=table, exposure=exposure, background=background)
        self.psf_source_measurement.run(detections.sources, exposure)
        psf_result = self.psf_measure_psf.run(exposure=exposure, sources=detections.sources)
        # Replace the initial PSF with something simpler for the second
        # repair/detect/measure/measure_psf step: this can help it converge.
        self.install_simple_psf.run(exposure=exposure)

        self.log.info("Re-running repair, detection, and PSF measurement using new simple PSF.")
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

        # PSF is set on exposure; only return candidates for optional saving.
        return detections.sources, background, psf_result.cellSet

    def _measure_aperture_correction(self, exposure, bright_sources):
        """Measure and set the ApCorrMap on the Exposure, using
        previously-measured bright sources.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to set the ApCorrMap on.
        bright_sources : `lsst.afw.table.SourceCatalog`
            Catalog of detected bright sources; modified to include columns
            necessary for point source determination for the aperture correction
            calculation.
        """
        result = self.measure_aperture_correction.run(exposure, bright_sources)
        exposure.setApCorrMap(result.apCorrMap)

    def _find_stars(self, exposure, background):
        """Detect stars on an exposure that has a PSF model, and measure their
        PSF, circular aperture, compensated gaussian fluxes.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to set the ApCorrMap on.
        background : `lsst.afw.math.BackgroundList`
            Background that was fit to the exposure during detection;
            modified in-place during subsequent detection.

        Returns
        -------
        stars : `SourceCatalog`
            Sources that are very likely to be stars, with a limited set of
            measurements performed on them.
        """
        table = afwTable.SourceTable.make(self.initial_stars_schema.schema)
        # Re-estimate the background during this detection step, so that
        # measurement uses the most accurate background-subtraction.
        detections = self.star_detection.run(table=table, exposure=exposure, background=background)
        sources = detections.sources
        # TODO investigation: Could this deblender throw away blends of non-PSF sources?
        self.star_deblend.run(exposure=exposure, sources=sources)
        # The deblender may not produce a contiguous catalog; ensure
        # contiguity for subsequent tasks.
        if not sources.isContiguous():
            sources = sources.copy(deep=True)

        # Measure everything, and use those results to select only stars.
        self.star_measurement.run(sources, exposure)
        self.star_apply_aperture_correction.run(sources, exposure.info.getApCorrMap())
        self.star_catalog_calculation.run(sources)
        self.star_set_primary_flags.run(sources)

        result = self.star_selector.run(sources)
        # The star selector may not produce a contiguous catalog.
        if not result.sourceCat.isContiguous():
            return result.sourceCat.copy(deep=True)
        else:
            return result.sourceCat

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

    def _fit_photometry(self, exposure, stars):
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
