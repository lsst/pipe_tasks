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
    "SingleBandMeasurementDriverConfig",
    "SingleBandMeasurementDriverTask",
    "MultiBandMeasurementDriverConfig",
    "MultiBandMeasurementDriverTask",
]

import logging
from abc import ABCMeta, abstractmethod

import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.meas.algorithms as measAlgorithms
import lsst.meas.base as measBase
import lsst.meas.deblender as measDeblender
import lsst.meas.extensions.scarlet as scarlet
from lsst.pex.config import Config, ConfigurableField, Field
import lsst.pipe.base as pipeBase
import numpy as np

logging.basicConfig(level=logging.INFO)


class MeasurementDriverBaseConfig(Config):
    """Base configuration for measurement driver tasks.

    This class provides foundational configuration for its subclasses to handle
    single-band and multi-band data. It defines the detection, deblending,
    measurement, aperture correction, and catalog calculation subtasks, which
    are intended to be executed in sequence by the driver tasks.
    """

    doScaleVariance = Field[bool](doc="Scale variance plane using empirical noise?", default=False)

    scaleVariance = ConfigurableField(
        target=measAlgorithms.ScaleVarianceTask, doc="Subtask to rescale variance plane"
    )

    doDetect = Field[bool](doc="Run the source detection algorithm?", default=True)

    detection = ConfigurableField(
        target=measAlgorithms.SourceDetectionTask, doc="Subtask to detect sources in the image."
    )

    doDeblend = Field[bool](doc="Run the source deblending algorithm?", default=True)
    # N.B. The 'deblend' configurable field should be defined in subclasses.

    doMeasure = Field[bool](doc="Run the source measurement algorithm?", default=True)

    measurement = ConfigurableField(
        target=measBase.SingleFrameMeasurementTask,
        doc="Subtask to measure sources and populate the output catalog",
    )

    psfCache = Field[int](doc="Size of psfCache", default=100)

    checkUnitsParseStrict = Field[str](
        doc="Strictness of Astropy unit compatibility check, can be 'raise', 'warn' or 'silent'",
        default="raise",
    )

    doApCorr = Field[bool](
        doc="Apply aperture corrections? If yes, your image must have an aperture correction map",
        default=False,
    )

    applyApCorr = ConfigurableField(
        doc="Subtask to apply aperture corrections",
        target=measBase.ApplyApCorrTask,
    )

    doRunCatalogCalculation = Field[bool](doc="Run catalogCalculation task?", default=False)

    catalogCalculation = ConfigurableField(
        target=measBase.CatalogCalculationTask, doc="Subtask to run catalogCalculation plugins on catalog"
    )


class MeasurementDriverBaseTask(pipeBase.Task, metaclass=ABCMeta):
    """Base class for the mid-level driver running detection, deblending,
    measurement algorithms, apperture correction, and catalog calculation in
    one go.

    This driver simplifies the process of applying a small set of measurement
    algorithms to images by abstracting away Schema and table boilerplate.
    Also, users don't need to Butlerize their input data. It is particularly
    suited for simple use cases, such as processing images without
    neighbor-noise-replacement or extensive configuration.

    Parameters
    ----------
    schema :
        Schema used to create the output `~lsst.afw.table.SourceCatalog`,
        modified in place with fields that will be written by this task.
    peakSchema :
        Schema of Footprint Peaks that will be passed to the deblender.
    **kwargs :
        Additional kwargs to pass to lsst.pipe.base.Task.__init__()

    Notes
    -----
    Subclasses (e.g. single-band vs multi-band) override how inputs are built
    or validated, but rely on this base for the pipeline logic.
    """

    ConfigClass = MeasurementDriverBaseConfig
    _DefaultName = "measurementDriverBase"
    _Deblender = ""

    def __init__(self, schema: afwTable.Schema = None, peakSchema: afwTable.Schema = None, **kwargs: dict):
        super().__init__(**kwargs)

        # Schema for the output catalog.
        self.schema = schema

        # Schema for deblender peaks.
        self.peakSchema = peakSchema

        # Placeholders for subclasses to populate.
        self.scaleVariance: measAlgorithms.ScaleVarianceTask
        self.detection: measAlgorithms.SourceDetectionTask
        self.deblend: measDeblender.SourceDeblendTask | scarlet.ScarletDeblendTask
        self.measure: measBase.SingleFrameMeasurementTask
        self.applyApCorr: measBase.ApplyApCorrTask
        self.catalogCalculation: measBase.CatalogCalculationTask
        self.exposure: afwImage.Exposure
        self.catalog: afwTable.SourceCatalog
        self.idGenerator: measBase.IdGenerator

    def _initializeSchema(self):
        """Initialize the Schema to be used for constructing the subtasks.

        Might seem a bit clunky, but this workaround is necessary to ensure
        that the Schema is consistent across all subtasks.
        """
        if self.catalog is None:
            if self.schema is None:
                # Create a minimal Schema that will be extended by tasks.
                self.schema = afwTable.SourceTable.makeMinimalSchema()

                # Add coordinate error fields to avoid missing field issues.
                afwTable.CoordKey.addErrorFields(self.schema)
        else:
            # Since a catalog is provided, use its Schema as the base.
            catalogSchema = self.catalog.schema

            # Create a SchemaMapper that maps from catalogSchema to a new one
            # it will create.
            self.mapper = afwTable.SchemaMapper(catalogSchema)

            # Add everything from catalogSchema to output Schema.
            self.mapper.addMinimalSchema(catalogSchema, True)

            # Get the output Schema from the SchemaMapper and assign it as the
            # Schema to be used for constructing the subtasks.
            self.schema = self.mapper.getOutputSchema()

    def _makeSubtasks(self):
        """Construct subtasks based on the configuration and the Schema."""
        if self.config.doScaleVariance and not hasattr(self, "scaleVariance"):
            self.makeSubtask("scaleVariance")

        if self.config.doDetect and not hasattr(self, "detection"):
            self.makeSubtask("detection", schema=self.schema)

        if self.config.doDeblend and not hasattr(self, "deblend"):
            self.makeSubtask("deblend", schema=self.schema, peakSchema=self.peakSchema)

        if self.config.doMeasure and not hasattr(self, "measurement"):
            self.makeSubtask("measurement", schema=self.schema)

        if self.config.doApCorr and not hasattr(self, "applyApCorr"):
            self.makeSubtask("applyApCorr", schema=self.schema)

        if self.config.doRunCatalogCalculation and not hasattr(self, "catalogCalculation"):
            self.makeSubtask("catalogCalculation", schema=self.schema)

        # Check that all units in the Schema are valid Astropy unit strings.
        self.schema.checkUnits(parse_strict=self.config.checkUnitsParseStrict)

    def _updateCatalogSchema(self):
        """Update the Schema of the provided catalog to incorporate changes
        made by the configured subtasks.
        """
        # Create an empty catalog with the Schema required by the subtasks that
        # are configured to run.
        newCatalog = afwTable.SourceCatalog(self.schema)

        # Transfer all records from the original catalog to the new catalog,
        # using the SchemaMapper to copy values.
        newCatalog.extend(self.catalog, mapper=self.mapper)

        # Replace the original catalog with the updated one, preserving the
        # records while applying the updated Schema.
        self.catalog = newCatalog

    @abstractmethod
    def run(self) -> afwTable.SourceCatalog:
        """Abstract method to run detection, deblending, measurement, aperture
        correction, and catalog calculation on a given exposure.

        Returns
        -------
        catalog :
            The source catalog with all requested measurements.
        """

        # Set up the Schema before creating subtasks.
        self._initializeSchema()

        # Create subtasks, passing the same Schema to each subtask's
        # constructor if need be.
        self._makeSubtasks()

        # Adjust the catalog Schema to align with changes made by the subtasks.
        if self.catalog is not None:
            self._updateCatalogSchema()

        # Generate catalog IDs consistently across subtasks.
        if self.idGenerator is None:
            self.idGenerator = measBase.IdGenerator()

        # Set psfcache.
        self.exposure.getPsf().setCacheCapacity(self.config.psfCache)

        # Scale variance plane.
        if self.config.doScaleVariance:
            varScale = self.scaleVariance.run(self.exposure.maskedImage)
            self.exposure.getMetadata().add("VARIANCE_SCALE", varScale)

        if self.config.doDetect:
            if self.catalog is None:
                # Create an empty source table with the known Schema into which
                # detected sources will be placed next.
                self.table = afwTable.SourceTable.make(self.schema, self.idGenerator.make_table_id_factory())
            else:
                raise RuntimeError(
                    "An input catalog was given to bypass detection, but detection is still on."
                )
        else:
            if self.catalog is None:
                raise RuntimeError("Cannot run without detection if no catalog is provided.")
            else:
                self.log.info("Using detections from provided catalog; skipping detection")

        # Detect sources in the image and populate the catalog.
        if self.config.doDetect:
            self._detectSources()

        # Deblend detected sources and update the catalog.
        if self.config.doDeblend:
            self.log.info(f"Deblending using '{self._Deblender}' on {len(self.catalog)} detection footprints")
            self._deblendSources()
            # The deblender may not produce a contiguous catalog; ensure
            # contiguity for the subsequent task.
            if not self.catalog.isContiguous():
                self.log.info("Catalog is not contiguous; making it contiguous")
                self.catalog = self.catalog.copy(deep=True)
        else:
            self.log.info("Deblending is disabled; skipping deblending")

        # Measure properties of detected/deblended sources.
        if self.config.doMeasure:
            self._measureSources()

        # Apply aperture corrections to the catalog.
        if self.config.doApCorr:
            self._applyApCorr()

        # Ensure contiguity again.
        if not self.catalog.isContiguous():
            self.catalog = self.catalog.copy(deep=True)

        # Run catalogCalculation on the catalog.
        if self.config.doRunCatalogCalculation:
            self._runCatalogCalculation()

        self.log.info(
            f"Run complete; output catalog has {self.catalog.schema.getFieldCount()} "
            f"fields and {len(self.catalog)} records"
        )

        return self.catalog

    def _detectSources(self):
        """Run the detection subtask to identify sources in the image."""
        self.log.info(f"Running detection on a {self.exposure.width}x{self.exposure.height} pixel exposure")
        self.catalog = self.detection.run(self.table, self.exposure).sources

    @abstractmethod
    def _deblendSources(self):
        """Run the deblending subtask to separate blended sources. Subclasses
        must implement this method to handle task-specific deblending logic.
        """
        raise NotImplementedError(f"{self.__class__.__name__} has not implemented '_deblendSources'.")

    def _measureSources(self):
        """Run the measurement subtask to compute properties of sources."""
        deblendedInfo = "and deblended" if self.config.doDeblend else "(not deblended)"
        self.log.info(f"Measuring {len(self.catalog)} detected {deblendedInfo} sources")
        self.measurement.run(
            measCat=self.catalog, exposure=self.exposure, exposureId=self.idGenerator.catalog_id
        )

    def _applyApCorr(self):
        """Apply aperture corrections to the catalog."""
        apCorrMap = self.exposure.getInfo().getApCorrMap()
        if apCorrMap is None:
            self.log.warning(
                "Image does not have valid aperture correction map for catalog id "
                f"{self.idGenerator.catalog_id}; skipping aperture correction"
            )
        else:
            self.log.info("Applying aperture corrections to the catalog")
            self.applyApCorr.run(catalog=self.catalog, apCorrMap=apCorrMap)

    def _runCatalogCalculation(self):
        """Run the catalogCalculation subtask to compute properties of sources."""
        self.log.info(f"Running catalogCalculation on {len(self.catalog)} sources")
        self.catalogCalculation.run(self.catalog)


class SingleBandMeasurementDriverConfig(MeasurementDriverBaseConfig):
    """Configuration for the single-band measurement driver task."""

    deblend = ConfigurableField(target=measDeblender.SourceDeblendTask, doc="Deblender for single-band data.")


class SingleBandMeasurementDriverTask(MeasurementDriverBaseTask):
    """Mid-level driver for processing single-band data.

    Provides an additional interface for handling raw image data that is
    specific to single-band processing.

    Examples
    --------
    Here is an example of how to use this class to run variance scaling,
    detection, deblending, and measurement on a single-band exposure:
    >>> from lsst.pipe.tasks.measurementDriver import (
    ...     SingleBandMeasurementDriverConfig,
    ...     SingleBandMeasurementDriverTask,
    ... )
    >>> import lsst.meas.extensions.shapeHSM  # To register its plugins
    >>> config = SingleBandMeasurementDriverConfig()
    >>> config.doScaleVariance = True
    >>> config.doDetect = True
    >>> config.doDeblend = True
    >>> config.doMeasure = True
    >>> config.scaleVariance.background.binSize = 64
    >>> config.detection.thresholdValue = 5.5
    >>> config.deblend.tinyFootprintSize = 3
    >>> config.measurement.plugins.names |= [
    ...     "base_SdssCentroid",
    ...     "base_SdssShape",
    ...     "ext_shapeHSM_HsmSourceMoments",
    ... ]
    >>> config.measurement.slots.psfFlux = None
    >>> config.measurement.doReplaceWithNoise = False
    >>> exposure = butler.get("deepCoadd", dataId=...)
    >>> driver = SingleBandMeasurementDriverTask(config=config)
    >>> catalog = driver.run(exposure)
    >>> catalog.writeFits("meas_catalog.fits")
    """

    ConfigClass = SingleBandMeasurementDriverConfig
    _DefaultName = "singleBandMeasurementDriver"
    _Deblender = "meas_deblender"

    def run(
        self,
        exposure: afwImage.Exposure,
        catalog: afwTable.SourceCatalog = None,
        idGenerator: measBase.IdGenerator = None,
    ) -> afwTable.SourceCatalog:
        """Process a single-band exposure.

        Parameters
        ----------
        exposure :
            The exposure on which to detect, deblend, and measure sources.
        catalog : optional
            Catalog to be extended by the driver task. If not provided, a new
            catalog will be created either from the user-provided Schema or a
            minimal Schema. It will then be populated with detected sources.
        idGenerator : optional
            Object that generates source IDs and provides random seeds.

        Returns
        -------
        catalog :
            Catalog containing the measured sources.
        """
        self.exposure = exposure
        self.catalog = catalog
        self.idGenerator = idGenerator
        return super().run()

    def runFromImage(
        self,
        image: afwImage.MaskedImage | afwImage.Image | np.ndarray,
        mask: afwImage.Mask | np.ndarray = None,
        variance: afwImage.Image | np.ndarray = None,
        wcs: afwGeom.SkyWcs = None,
        psf: afwDetection.Psf | np.ndarray = None,
        photoCalib: afwImage.PhotoCalib = None,
        idGenerator: measBase.IdGenerator = None,
    ) -> afwTable.SourceCatalog:
        """Convert image data to an `Exposure`, then run it through the
        configured subtasks.

        Parameters
        ----------
        image :
            Input image data. Will be converted into an `Exposure` before
            processing.
        mask : optional
            Mask data for the image. Used if ``image`` is a bare `array` or
            `Image`.
        variance : optional
            Variance plane data for the image.
        wcs : optional
            World Coordinate System to associate with the exposure that will
            be created from ``image``.
        psf : optional
            PSF model for the exposure.
        photoCalib : optional
            Photometric calibration model for the exposure.
        idGenerator : optional
            Generator for unique source IDs.

        Returns
        -------
        catalog :
            Final catalog of measured sources.
        """
        # Convert raw image data into an Exposure.
        if isinstance(image, np.ndarray):
            image = afwImage.makeImageFromArray(image)
        if isinstance(mask, np.ndarray):
            mask = afwImage.makeMaskFromArray(mask)
        if isinstance(variance, np.ndarray):
            variance = afwImage.makeImageFromArray(variance)
        if isinstance(image, afwImage.Image):
            image = afwImage.makeMaskedImage(image, mask, variance)

        # By now, the input should already be - or have been converted to - a
        # MaskedImage.
        if isinstance(image, afwImage.MaskedImage):
            exposure = afwImage.makeExposure(image, wcs)
        else:
            raise TypeError(f"Unsupported 'image' type: {type(image)}")

        if psf is not None:
            if isinstance(psf, np.ndarray):
                # Create a FixedKernel using the array.
                psf /= psf.sum()
                kernel = afwMath.FixedKernel(afwImage.makeImageFromArray(psf))
                # Create a KernelPsf using the kernel.
                psf = afwDetection.KernelPsf(kernel)
            elif not isinstance(psf, afwDetection.Psf):
                raise TypeError(f"Unsupported 'psf' type: {type(psf)}")
            exposure.setPsf(psf)

        if photoCalib is not None:
            exposure.setPhotoCalib(photoCalib)

        return self.run(exposure, idGenerator=idGenerator)

    def _deblendSources(self):
        self.deblend.run(exposure=self.exposure, sources=self.catalog)


class MultiBandMeasurementDriverConfig(MeasurementDriverBaseConfig):
    """Configuration for the multi-band measurement driver task."""

    deblend = ConfigurableField(
        target=scarlet.ScarletDeblendTask, doc="Scarlet deblender for multi-band data."
    )

    doConserveFlux = Field[bool](
        doc="Whether to use the deblender models as templates to re-distribute the flux from "
        "the 'exposure' (True), or to perform measurements on the deblender model footprints.",
        default=False,
    )

    doStripHeavyFootprints = Field[bool](
        doc="Whether to strip heavy footprints from the output catalog before saving to disk. "
        "This is usually done when using scarlet models to save disk space.",
        default=True,
    )


class MultiBandMeasurementDriverTask(MeasurementDriverBaseTask):
    """Mid-level driver for processing multi-band data.

    Provides functionality for handling a singe-band exposure and a list of
    single-band exposures in addition to a standard multi-band exposure.

    Examples
    --------
    Here is an example of how to use this class to run variance scaling,
    detection, deblending, and measurement on a multi-band exposure:
    >>> from lsst.afw.image import MultibandExposure
    >>> from lsst.pipe.tasks.measurementDriver import (
    ...     MultiBandMeasurementDriverConfig,
    ...     MultiBandMeasurementDriverTask,
    ... )
    >>> import lsst.meas.extensions.shapeHSM  # To register its plugins
    >>> config = MultiBandMeasurementDriverConfig()
    >>> config.doScaleVariance = True
    >>> config.doDetect = True
    >>> config.doDeblend = True
    >>> config.doMeasure = True
    >>> config.scaleVariance.background.binSize = 64
    >>> config.detection.thresholdValue = 5.5
    >>> config.deblend.minSNR = 42.0
    >>> config.deblend.maxIter = 20
    >>> config.measurement.plugins.names |= [
    ...     "base_SdssCentroid",
    ...     "base_SdssShape",
    ...     "ext_shapeHSM_HsmSourceMoments",
    ... ]
    >>> config.measurement.slots.psfFlux = None
    >>> config.measurement.doReplaceWithNoise = False
    >>> mExposure = MultibandExposure.fromButler(
    ...     butler, ["g", "r", "i"], "deepCoadd_calexp", ...
    ... )
    >>> driver = MultiBandMeasurementDriverTask(config=config)
    >>> catalog = driver.run(mExposure, "r")
    >>> catalog.writeFits("meas_catalog.fits")
    """

    ConfigClass = MultiBandMeasurementDriverConfig
    _DefaultName = "multiBandMeasurementDriver"
    _Deblender = "scarlet"

    def run(
        self,
        mExposure: afwImage.MultibandExposure | list[afwImage.Exposure],
        band: str | None = None,
        bands: list[str] | None = None,
        catalog: afwTable.SourceCatalog = None,
        idGenerator: measBase.IdGenerator = None,
    ) -> afwTable.SourceCatalog:
        """
        Process a multi-band exposure or a list of exposures.

        Parameters
        ----------
        mExposure :
            Multi-band data. May be a single `MultibandExposure` or a list of
            exposures associated with different bands in which case ``bands``
            must be provided.
        band : optional
            Reference band to use for detection and measurement.
        bands : optional
            List of bands associated with the exposures in ``mExposure``. Only
            required if ``mExposure`` is a list of single-band exposures.
        catalog : optional
            Catalog to be extended by the driver task. If not provided, a new
            catalog will be created and populated.
        idGenerator : optional
            Generator for unique source IDs.

        Returns
        -------
        catalog :
            Catalog containing the measured sources.
        """

        # Basic sanity checks to ensure the inputs are consistent.
        if (band is None) != (bands is None):
            raise ValueError("'band' and 'bands' must be provided together or not at all.")
        if band is not None and bands is not None:
            if band not in bands:
                raise ValueError(f"Reference band '{band}' is not in the list of bands: {bands}")

        # Store the reference band for later use.
        self.band = band

        # Convert mExposure to a MultibandExposure object if not already in
        # that form. Save the result as an instance attribute for later use.
        self.mExposure = self._buildMultibandExposure(mExposure, bands)

        if self.band not in self.mExposure:
            raise ValueError(f"Requested band '{band}' is not present in the multiband exposure.")

        # We use a reference band for band-specific tasks like detection and
        # measurement.
        self.exposure = self.mExposure[self.band]
        self.log.info(f"Using '{self.band}' band as the reference band for band-specific tasks")

        self.catalog = catalog
        self.idGenerator = idGenerator

        return super().run()

    def _deblendSources(self):
        self.catalog, modelData = self.deblend.run(mExposure=self.mExposure, mergedSources=self.catalog)

        # The footprints need to be updated for the subsequent measurement.
        if self.config.doConserveFlux:
            imageForRedistribution = self.exposure
        else:
            imageForRedistribution = None
        scarlet.io.updateCatalogFootprints(
            modelData=modelData,
            catalog=self.catalog,
            band=self.band,
            imageForRedistribution=imageForRedistribution,
            removeScarletData=True,
            updateFluxColumns=True,
        )

        # Strip HeavyFootprints to save space on disk.
        if self.config.doStripHeavyFootprints:
            sources = self.catalog
            for source in sources[sources["parent"] != 0]:
                source.setFootprint(None)

    def _buildMultibandExposure(
        self, mExposure: afwImage.MultibandExposure | list[afwImage.Exposure], bands: list[str] | None
    ) -> afwImage.MultibandExposure:
        """Convert a single-band exposure or a list of single-band exposures to
        a `MultibandExposure` if not already of that type.

        No conversion is done if the input is already a `MultibandExposure`.

        Parameters
        ----------
        mExposure :
            Input multi-band data.
        bands : optional
            List of bands associated with the exposures in ``mExposure``. Only
            required if ``mExposure`` is a list of single-band exposures.

        Returns
        -------
        mExposure :
            Converted multi-band exposure.
        """
        if isinstance(mExposure, afwImage.MultibandExposure):
            if bands is not None:
                self.log.warn("Ignoring 'bands' argument; using bands from the input `MultibandExposure`")
            return mExposure
        elif isinstance(mExposure, list):
            if bands is None:
                raise ValueError("List of bands must be provided if 'mExposure' is a list")
            if len(bands) != len(mExposure):
                raise ValueError("Number of bands and exposures must match.")
            return afwImage.MultibandExposure.fromExposures(bands, mExposure)
        elif isinstance(mExposure, afwImage.Exposure):
            # N.B. Scarlet is designed to leverage multiband information to
            # differentiate overlapping sources based on their spectral and
            # spatial profiles. However, it can also run on a single band and
            # often give better results than 'meas_deblender'.
            self.log.debug("Using 'scarlet' deblender for single-band processing; make sure it was intended!")
            if self.band is None:
                self.band = "N/A"  # Placeholder for single-band deblending
            if bands is None:
                bands = [self.band]
            # We need to have a multiband exposure to satisfy scarlet
            # function's signature, even when using a single band.
            return afwImage.MultibandExposure.fromExposures(bands, [mExposure])
        else:
            raise TypeError(f"Unsupported 'mExposure' type: {type(mExposure)}")
