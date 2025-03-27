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

import copy
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
import lsst.pipe.base as pipeBase
import lsst.scarlet.lite as scl
import numpy as np
from lsst.pex.config import Config, ConfigurableField, Field

logging.basicConfig(level=logging.INFO)


class MeasurementDriverBaseConfig(Config):
    """Base configuration for measurement driver tasks.

    This class provides foundational configuration for its subclasses to handle
    single-band and multi-band data. It defines variance scaling, detection,
    deblending, measurement, aperture correction, and catalog calculation
    subtasks, which are intended to be executed in sequence by the driver task.
    """

    doScaleVariance = Field[bool](doc="Scale variance plane using empirical noise?", default=False)

    scaleVariance = ConfigurableField(
        doc="Subtask to rescale variance plane", target=measAlgorithms.ScaleVarianceTask
    )

    doDetect = Field[bool](doc="Run the source detection algorithm?", default=True)

    detection = ConfigurableField(
        doc="Subtask to detect sources in the image", target=measAlgorithms.SourceDetectionTask
    )

    doDeblend = Field[bool](doc="Run the source deblending algorithm?", default=True)
    # N.B. The 'deblend' configurable field should be defined in subclasses.

    doMeasure = Field[bool](doc="Run the source measurement algorithm?", default=True)

    measurement = ConfigurableField(
        doc="Subtask to measure sources and populate the output catalog",
        target=measBase.SingleFrameMeasurementTask,
    )

    psfCache = Field[int](
        doc="Maximum number of PSFs to cache, preventing repeated PSF evaluations at the same "
        "point across different measurement plugins. Defaults to -1, which auto-sizes the cache "
        "based on the plugin count.",
        default=-1,
    )

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
        doc="Subtask to run catalogCalculation plugins on catalog", target=measBase.CatalogCalculationTask
    )

    doOptions = [
        "doScaleVariance",
        "doDetect",
        "doDeblend",
        "doMeasure",
        "doApCorr",
        "doRunCatalogCalculation",
    ]

    def validate(self):
        """Ensure that at least one processing step is enabled."""
        super().validate()

        if not any(getattr(self, opt) for opt in self.doOptions):
            raise ValueError(f"At least one of these options must be enabled: {self.doOptions}")


class MeasurementDriverBaseTask(pipeBase.Task, metaclass=ABCMeta):
    """Base class for the mid-level driver running variance scaling, detection,
    deblending, measurement, apperture correction, and catalog calculation in
    one go.

    Users don't need to Butlerize their input data, which is a significant
    advantage for quick data exploration and testing. This driver simplifies
    the process of applying measurement algorithms to images by abstracting
    away low-level implementation details such as Schema and table boilerplate.
    It's a convenient way to process images into catalogs with a user-friendly
    interface for non-developers while allowing extensive configuration and
    integration into unit tests for developers. It also considerably improves
    how demos and workflows are showcased in Jupyter notebooks.

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
    Subclasses (e.g., single-band vs. multi-band) share most methods and config
    options but differ in handling and validating inputs by overriding the base
    config class and any methods that require their own logic.
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
        self.mapper: afwTable.SchemaMapper
        self.scaleVariance: measAlgorithms.ScaleVarianceTask
        self.detection: measAlgorithms.SourceDetectionTask
        self.deblend: measDeblender.SourceDeblendTask | scarlet.ScarletDeblendTask
        self.measurement: measBase.SingleFrameMeasurementTask
        self.applyApCorr: measBase.ApplyApCorrTask
        self.catalogCalculation: measBase.CatalogCalculationTask

        # Store the initial Schema to use for reinitialization if necessary.
        self.initSchema: afwTable.Schema
        # To safeguard against user tampering and ensure predictable behavior,
        # the following attribute can only be modified within the class using
        # a controlled setter.
        super().__setattr__("initSchema", copy.deepcopy(schema))

    def __setattr__(self, name, value):
        """Prevent external modifications of the initial Schema."""
        if name == "initSchema":
            raise AttributeError(f"Cannot modify {name} directly")
        super().__setattr__(name, value)

    @abstractmethod
    def run(self, *args, **kwargs) -> pipeBase.Struct:
        """Run the measurement driver task. Subclasses must implement this
        method using their own logic to handle single-band or multi-band data.
        """
        raise NotImplementedError("This is not implemented on the base class")

    def _ensureValidInputs(
        self,
        catalog: afwTable.SourceCatalog | None,
    ):
        """Perform validation and adjustments of inputs without heavy
        computation.

        Parameters
        ----------
        catalog :
            Catalog to be extended by the driver task.
        """
        # Validate the configuration before proceeding.
        self.config.validate()

        if self.config.doDetect:
            if catalog is not None:
                raise RuntimeError(
                    "An input catalog was given to bypass detection, but 'doDetect' is still on"
                )
        else:
            if catalog is None:
                raise RuntimeError("Cannot run without detection if no 'catalog' is provided")

    def _initializeSchema(self, catalog: afwTable.SourceCatalog = None):
        """Initialize the Schema to be used for constructing the subtasks.

        Though it may seem clunky, this workaround is necessary to ensure
        Schema consistency across all subtasks.

        Parameters
        ----------
        catalog :
            Catalog from which to extract the Schema. If not provided, the
            user-provided Schema and if that is also not provided during
            initialization, a minimal Schema will be used.
        """
        # If the Schema has been modified (either by subtasks or externally by
        # the user), reset it to the initial state before creating subtasks.
        # This would be neccessary when running the same driver task multiple
        # times with different configs/inputs.
        if self.schema != self.initSchema:
            self.schema = copy.deepcopy(self.initSchema)

        if catalog is None:
            if self.schema is None:
                # Create a minimal Schema that will be extended by tasks.
                self.schema = afwTable.SourceTable.makeMinimalSchema()

                # Add coordinate error fields to avoid missing field issues.
                self._addCoordErrorFieldsIfMissing(self.schema)
        else:
            if self.schema is not None:
                self.log.warning(
                    "Both a catalog and a Schema were provided; using the Schema from the catalog only"
                )

            # Since a catalog is provided, use its Schema as the base.
            catalogSchema = catalog.schema

            # Ensure that the Schema has coordinate error fields.
            self._addCoordErrorFieldsIfMissing(catalogSchema)

            # Create a SchemaMapper that maps from catalogSchema to a new one
            # it will create.
            self.mapper = afwTable.SchemaMapper(catalogSchema)

            # Add everything from catalogSchema to output Schema.
            self.mapper.addMinimalSchema(catalogSchema, True)

            # Get the output Schema from the SchemaMapper and assign it as the
            # Schema to be used for constructing the subtasks.
            self.schema = self.mapper.getOutputSchema()

    def _addCoordErrorFieldsIfMissing(self, schema: afwTable.Schema):
        """Add coordinate error fields to the schema in-place if they are not
        already present.

        Parameters
        ----------
        schema :
            Schema to be checked for coordinate error fields.
        """
        if not any(
            errorField in schema.getNames()
            for errorField in ("coord_raErr", "coord_decErr", "coord_ra_dec_Cov")
        ):
            afwTable.CoordKey.addErrorFields(schema)

    def _makeSubtasks(self):
        """Construct subtasks based on the configuration and the Schema."""
        if self.schema is None and any(
            getattr(self.config, attr) for attr in self.config.doOptions if attr != "doScaleVariance"
        ):
            raise RuntimeError(
                "Cannot create requested subtasks without a Schema; "
                "ensure one is provided explicitly or via a catalog"
            )

        if self.config.doScaleVariance:
            self.makeSubtask("scaleVariance")

        if self.config.doDetect:
            self.makeSubtask("detection", schema=self.schema)

        if self.config.doDeblend:
            self.makeSubtask("deblend", schema=self.schema, peakSchema=self.peakSchema)

        if self.config.doMeasure:
            self.makeSubtask("measurement", schema=self.schema)

        if self.config.doApCorr:
            self.makeSubtask("applyApCorr", schema=self.schema)

        if self.config.doRunCatalogCalculation:
            self.makeSubtask("catalogCalculation", schema=self.schema)

    def _prepareSchemaAndSubtasks(
        self, catalog: afwTable.SourceCatalog | None
    ) -> afwTable.SourceCatalog | None:
        """Ensure subtasks are properly initialized according to the
        configuration and the provided catalog.

        Parameters
        ----------
        catalog :
            Optional catalog to be used for initializing the Schema and the
            subtasks.

        Returns
        -------
        catalog :
            Updated catalog to be passed to the subtasks, if it was provided.
        """
        # Set up the Schema before creating subtasks.
        self._initializeSchema(catalog)

        # Create subtasks, passing the same Schema to each subtask's
        # constructor that requires it.
        self._makeSubtasks()

        # Check that all units in the Schema are valid Astropy unit strings.
        self.schema.checkUnits(parse_strict=self.config.checkUnitsParseStrict)

        # Adjust the catalog Schema to align with changes made by the subtasks.
        if catalog:
            catalog = self._updateCatalogSchema(catalog)

        return catalog

    def _scaleVariance(self, exposure: afwImage.Exposure, band: str = "a single"):
        """Scale the variance plane of an exposure to match the observed
        variance.

        Parameters
        ----------
        exposure :
            Exposure on which to run the variance scaling algorithm.
        band :
            Band associated with the exposure. Used for logging.
        """
        self.log.info(f"Scaling variance plane for {band} band")
        varScale = self.scaleVariance.run(exposure.maskedImage)
        exposure.getMetadata().add("VARIANCE_SCALE", varScale)

    def _toContiguous(
        self, catalog: afwTable.SourceCatalog | dict[str, afwTable.SourceCatalog]
    ) -> afwTable.SourceCatalog | dict[str, afwTable.SourceCatalog]:
        """Make a catalog or catalogs contiguous if they are not already.

        Parameters
        ----------
        catalog :
            Catalog or dictionary of catalogs with bands as keys to be made
            contiguous.

        Returns
        -------
        catalog :
            Contiguous catalog or dictionary of contiguous catalogs.
        """
        if isinstance(catalog, dict):
            for band, cat in catalog.items():
                if not cat.isContiguous():
                    self.log.info(f"{band}-band catalog is not contiguous; making it contiguous")
                    catalog[band] = cat.copy(deep=True)
        else:
            if not catalog.isContiguous():
                self.log.info("Catalog is not contiguous; making it contiguous")
                catalog = catalog.copy(deep=True)
        return catalog

    def _updateCatalogSchema(self, catalog: afwTable.SourceCatalog) -> afwTable.SourceCatalog:
        """Update the Schema of the provided catalog to incorporate changes
        made by the configured subtasks.

        Parameters
        ----------
        catalog :
            Catalog to be updated with the Schema changes.

        Returns
        -------
        updatedCatalog :
            Catalog with the updated Schema.
        """
        # Create an empty catalog with the Schema required by the subtasks that
        # are configured to run.
        updatedCatalog = afwTable.SourceCatalog(self.schema)

        # Transfer all records from the original catalog to the new catalog,
        # using the SchemaMapper to copy values.
        updatedCatalog.extend(catalog, mapper=self.mapper)

        # Return the updated catalog, preserving the records while applying the
        # updated Schema.
        return updatedCatalog

    def _detectSources(
        self, exposure: afwImage.Exposure | afwImage.MultibandExposure, idGenerator: measBase.IdGenerator
    ) -> tuple[afwTable.SourceCatalog, afwMath.BackgroundList]:
        """Run the detection subtask to identify sources in the image.

        Parameters
        ----------
        exposure :
            Exposure on which to run the detection algorithm.
        idGenerator :
            Generator for unique source IDs.

        Returns
        -------
        catalog :
            A catalog containing detected sources.
        backgroundList :
            A list of background models obtained from the detection process,
            if available.
        """
        self.log.info(f"Running detection on a {exposure.width}x{exposure.height} pixel exposure")

        # Create an empty source table with the known Schema into which
        # detected sources will be placed next.
        table = afwTable.SourceTable.make(self.schema, idGenerator.make_table_id_factory())

        # Run the detection task on the exposure and make a source catalog.
        detections = self.detection.run(table, exposure)
        catalog = detections.sources
        backgroundList = afwMath.BackgroundList()

        # Get the background model from the detection task, if available.
        if hasattr(detections, "background") and detections.background:
            for bg in detections.background:
                backgroundList.append(bg)

        return catalog, backgroundList

    @abstractmethod
    def _deblendSources(self, *args, **kwargs):
        """Run the deblending subtask to separate blended sources. Subclasses
        must implement this method to handle task-specific deblending logic.
        """
        raise NotImplementedError("This is not implemented on the base class")

    def _measureSources(
        self,
        exposure: afwImage.Exposure,
        catalog: afwTable.SourceCatalog,
        idGenerator: measBase.IdGenerator,
    ):
        """Run the measurement subtask to compute properties of sources.

        Parameters
        ----------
        exposure :
            Exposure on which to run the measurement algorithm.
        catalog :
            Catalog containing sources on which to run the measurement subtask.
        idGenerator :
            Generator for unique source IDs.
        """
        self.measurement.run(measCat=catalog, exposure=exposure, exposureId=idGenerator.catalog_id)

    def _applyApCorr(
        self, exposure: afwImage.Exposure, catalog: afwTable.SourceCatalog, idGenerator: measBase.IdGenerator
    ):
        """Apply aperture corrections to the catalog.

        Parameters
        ----------
        exposure :
            Exposure on which to apply aperture corrections.
        catalog :
            Catalog to be corrected using the aperture correction map from
            the exposure.
        idGenerator :
            Generator for unique source IDs.
        """
        apCorrMap = exposure.getInfo().getApCorrMap()
        if apCorrMap is None:
            self.log.warning(
                "Image does not have valid aperture correction map for catalog id "
                f"{idGenerator.catalog_id}; skipping aperture correction"
            )
        else:
            self.applyApCorr.run(catalog=catalog, apCorrMap=apCorrMap)

    def _runCatalogCalculation(self, catalog: afwTable.SourceCatalog):
        """Run the catalog calculation plugins on the catalog.

        Parameters
        ----------
        catalog :
            Catalog to be processed by the catalog calculation subtask.
        """
        self.catalogCalculation.run(catalog)

    def _processCatalog(
        self,
        exposure: afwImage.Exposure,
        catalog: afwTable.SourceCatalog,
        idGenerator: measBase.IdGenerator,
        band: str = "a single",
    ) -> afwTable.SourceCatalog:
        """Process a catalog through measurement, aperture correction, and
        catalog calculation subtasks.

        Parameters
        ----------
        exposure :
            Exposure associated with the catalog.
        catalog :
            Catalog to be processed by the subtasks.
        idGenerator :
            Generator for unique source IDs.
        band :
            Band associated with the exposure and catalog. Used for logging.
        """
        # Set the PSF cache capacity to cache repeated PSF evaluations at the
        # same point coming from different measurement plugins.
        if self.config.psfCache > 0:
            # Set a hard limit on the number of PSFs to cache.
            exposure.psf.setCacheCapacity(self.config.psfCache)
        else:
            # Auto-size the cache based on the number of measurement
            # plugins. We assume each algorithm tries to evaluate the PSF
            # twice, which is more than enough since many don't evaluate it
            # at all, and there's no *good* reason for any algorithm to
            # evaluate it more than once.
            # (Adopted from drp_tasks/ForcedPhotCoaddTask)
            exposure.psf.setCacheCapacity(2 * len(self.config.measurement.plugins.names))

        # Measure properties of sources in the catalog.
        if self.config.doMeasure:
            self.log.info(f"Measuring {len(catalog)} sources in {band} band")
            self._measureSources(exposure, catalog, idGenerator)

        # Ensure contiguity again.
        catalog = self._toContiguous(catalog)

        # Apply aperture corrections to the catalog.
        if self.config.doApCorr:
            self.log.info(f"Applying aperture corrections to {band} band")
            self._applyApCorr(exposure, catalog, idGenerator)

        # Run catalogCalculation on the catalog.
        if self.config.doRunCatalogCalculation:
            self.log.info(f"Running catalog calculation on {band} band")
            self._runCatalogCalculation(catalog)

        self.log.info(
            f"Finished processing for {band} band; output catalog has {catalog.schema.getFieldCount()} "
            f"fields and {len(catalog)} records"
        )

        return catalog


class SingleBandMeasurementDriverConfig(MeasurementDriverBaseConfig):
    """Configuration for the single-band measurement driver task."""

    deblend = ConfigurableField(target=measDeblender.SourceDeblendTask, doc="Deblender for single-band data.")


class SingleBandMeasurementDriverTask(MeasurementDriverBaseTask):
    """Mid-level driver for processing single-band data.

    Offers a helper method for direct handling of raw image data in addition to
    the standard single-band exposure.

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
    >>> results = driver.run(exposure)
    >>> results.catalog.writeFits("meas_catalog.fits")

    Alternatively, if an exposure is not available, the driver can also process
    raw image data:

    >>> image = ...
    >>> mask = ...
    >>> variance = ...
    >>> wcs = ...
    >>> psf = ...
    >>> photoCalib = ...
    >>> results = driver.runFromImage(
    ...     image, mask, variance, wcs, psf, photoCalib
    ... )
    >>> results.catalog.writeFits("meas_catalog.fits")
    """

    ConfigClass = SingleBandMeasurementDriverConfig
    _DefaultName = "singleBandMeasurementDriver"
    _Deblender = "meas_deblender"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.deblend: measDeblender.SourceDeblendTask

    def run(
        self,
        exposure: afwImage.Exposure,
        catalog: afwTable.SourceCatalog | None = None,
        idGenerator: measBase.IdGenerator | None = None,
    ) -> pipeBase.Struct:
        """Process a single-band exposure through the configured subtasks and
        return the results as a struct.

        Parameters
        ----------
        exposure :
            The exposure on which to run the driver task.
        catalog :
            Catalog to be extended by the driver task. If not provided, an
            empty catalog will be created and populated.
        idGenerator :
            Object that generates source IDs and provides random seeds.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``catalog``
                Catalog containing the measured sources
                (`~lsst.afw.table.SourceCatalog`).
            ``backgroundList``
                List of backgrounds (`list[~lsst.afw.math.Background]`). Only
                populated if detection is enabled.
        """

        # Validate inputs before proceeding.
        self._ensureValidInputs(catalog)

        # Prepare the Schema and subtasks for processing.
        catalog = self._prepareSchemaAndSubtasks(catalog)

        # Generate catalog IDs consistently across subtasks.
        if idGenerator is None:
            idGenerator = measBase.IdGenerator()

        # Scale the variance plane. If enabled, this should be done before
        # detection.
        if self.config.doScaleVariance:
            self._scaleVariance(exposure)

        # Detect sources in the image and populate the catalog.
        if self.config.doDetect:
            catalog, backgroundList = self._detectSources(exposure, idGenerator)
        else:
            self.log.info("Skipping detection; using detections from the provided catalog")
            backgroundList = None

        # Deblend detected sources and update the catalog.
        if self.config.doDeblend:
            catalog = self._deblendSources(exposure, catalog)
        else:
            self.log.info("Skipping deblending")

        # Process catalog through measurement, aperture correction, and catalog
        # calculation subtasks.
        catalog = self._processCatalog(exposure, catalog, idGenerator)

        return pipeBase.Struct(catalog=catalog, backgroundList=backgroundList)

    def runFromImage(
        self,
        image: afwImage.MaskedImage | afwImage.Image | np.ndarray,
        mask: afwImage.Mask | np.ndarray = None,
        variance: afwImage.Image | np.ndarray = None,
        wcs: afwGeom.SkyWcs = None,
        psf: afwDetection.Psf | np.ndarray = None,
        photoCalib: afwImage.PhotoCalib = None,
        catalog: afwTable.SourceCatalog = None,
        idGenerator: measBase.IdGenerator = None,
    ) -> pipeBase.Struct:
        """Convert image data to an `Exposure`, then run it through the
        configured subtasks.

        Parameters
        ----------
        image :
            Input image data. Will be converted into an `Exposure` before
            processing.
        mask :
            Mask data for the image. Used if ``image`` is a bare `array` or
            `Image`.
        variance :
            Variance plane data for the image.
        wcs :
            World Coordinate System to associate with the exposure that will
            be created from ``image``.
        psf :
            PSF model for the exposure.
        photoCalib :
            Photometric calibration model for the exposure.
        catalog :
            Catalog to be extended by the driver task. If not provided, a new
            catalog will be created during detection and populated.
        idGenerator :
            Generator for unique source IDs.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``catalog``
                Catalog containing the measured sources
                (`~lsst.afw.table.SourceCatalog`).
            ``backgroundList``
                List of backgrounds (`list[~lsst.afw.math.Background]`).
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

        return self.run(exposure, catalog=catalog, idGenerator=idGenerator)

    def _deblendSources(
        self, exposure: afwImage.Exposure, catalog: afwTable.SourceCatalog
    ) -> afwTable.SourceCatalog:
        """Run single-band deblending given an exposure and a catalog.

        Parameters
        ----------
        exposure :
            Exposure on which to run the deblending algorithm.
        catalog :
            Catalog containing sources to be deblended.
        """
        self.log.info(f"Deblending using '{self._Deblender}' on {len(catalog)} detection footprints")
        self.deblend.run(exposure=exposure, sources=catalog)
        # The deblender may not produce contiguous catalogs; ensure
        # contiguity for the subsequent subtasks.
        return self._toContiguous(catalog)


class MultiBandMeasurementDriverConfig(MeasurementDriverBaseConfig):
    """Configuration for the multi-band measurement driver task."""

    deblend = ConfigurableField(
        target=scarlet.ScarletDeblendTask, doc="Scarlet deblender for multi-band data"
    )

    doConserveFlux = Field[bool](
        doc="Whether to use the deblender models as templates to re-distribute the flux from "
        "the 'exposure' (True), or to perform measurements on the deblender model footprints.",
        default=False,
    )

    measureOnlyInRefBand = Field[bool](
        doc="If True, all measurements downstream of deblending run only in the reference band that "
        "was used for detection; otherwise, they are performed in all available bands, generating a "
        "catalog for each. Regardless of this setting, deblending still uses all available bands.",
        default=False,
    )

    removeScarletData = Field[bool](
        doc="Whether or not to remove `ScarletBlendData` for each blend in order to save memory. "
        "If set to True, some sources may end up with missing footprints in catalogs other than the "
        "reference-band catalog, leading to failures in subsequent measurements that require footprints. "
        "For example, keep this False if `measureOnlyInRefBand` is set to False and "
        "`measurement.doReplaceWithNoise` to True, in order to make the footprints available in "
        "non-reference bands in addition to the reference band.",
        default=False,
    )

    updateFluxColumns = Field[bool](
        doc="Whether or not to update the `deblend_*` columns in the catalog. This should only be "
        "True when the input catalog schema already contains those columns.",
        default=True,
    )


class MultiBandMeasurementDriverTask(MeasurementDriverBaseTask):
    """Mid-level driver for processing multi-band data.

    The default behavior is to run detection on the reference band, use all
    available bands for deblending, and then process everything downstream
    separately for each band making per-band catalogs unless configured
    otherwise. This subclass provides functionality for handling a singe-band
    exposure and a list of single-band exposures in addition to a standard
    multi-band exposure.

    Examples
    --------
    Here is an example of how to use this class to run variance scaling,
    detection, deblending, measurement, and aperture correction on a multi-band
    exposure:

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
    >>> config.doApCorr = True
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
    >>> config.applyApCorr.doFlagApCorrFailures = False
    >>> mExposure = MultibandExposure.fromButler(
    ...     butler, ["g", "r", "i"], "deepCoadd_calexp", ...
    ... )
    >>> driver = MultiBandMeasurementDriverTask(config=config)
    >>> results = driver.run(mExposure, "r")
    >>> for band, catalog in results.catalogs.items():
    ...     catalog.writeFits(f"meas_catalog_{band}.fits")
    """

    ConfigClass = MultiBandMeasurementDriverConfig
    _DefaultName = "multiBandMeasurementDriver"
    _Deblender = "scarlet"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.deblend: scarlet.ScarletDeblendTask

        # Placeholder for the model data produced by the deblender. Caching
        # this data has proven be useful for debugging.
        self.modelData: scl.io.ScarletModelData

    def run(
        self,
        mExposure: afwImage.MultibandExposure | list[afwImage.Exposure] | afwImage.Exposure,
        refBand: str | None = None,
        bands: list[str] | None = None,
        catalog: afwTable.SourceCatalog = None,
        idGenerator: measBase.IdGenerator = None,
    ) -> pipeBase.Struct:
        """Process an exposure through the configured subtasks while using
        multi-band information for deblending.

        Parameters
        ----------
        mExposure :
            Multi-band data. May be a `MultibandExposure`, a single-band
            exposure (i.e., `Exposure`), or a list of single-band exposures
            associated with different bands in which case ``bands`` must be
            provided. If a single-band exposure is given, it will be treated as
            a `MultibandExposure` that contains only that one band.
        refBand :
            Reference band to use for detection. Not required for single-band
            exposures. If `measureOnlyInRefBand` is enabled while detection is
            disabled and a catalog of detected sources is provided, this
            should specify the band the sources were detected on (or the band
            you want to use to perform measurements on exclusively). If
            `measureOnlyInRefBand` is disabled instead in the latter scenario,
            ``refBand`` does not need to be provided.
        bands :
            List of bands associated with the exposures in ``mExposure``. Only
            required if ``mExposure`` is a list of single-band exposures. If
            provided for a multi-band exposure, it will be used to only process
            that subset of bands from the available ones in the exposure.
        catalog :
            Catalog to be extended by the driver task. If not provided, a new
            catalog will be created and populated.
        idGenerator :
            Generator for unique source IDs.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``catalogs``
                Dictionary of catalogs containing the measured sources with
                bands as keys (`dict[str, ~lsst.afw.table.SourceCatalog]`). If
                `measureOnlyInRefBand` is enabled or deblending is disabled,
                this will only contain the reference-band catalog; otherwise,
                it will contain a catalog for each band.
            ``backgroundList``
                List of backgrounds (`list[~lsst.afw.math.Background]`). Will
                be None if detection is disabled.
            ``modelData``
                Multiband scarlet models produced during deblending
                (`~lsst.scarlet.lite.io.ScarletModelData`). Will be None if
                deblending is disabled.
        """

        # Validate inputs and adjust them as necessary.
        mExposure, refBand, bands = self._ensureValidInputs(mExposure, refBand, bands, catalog)

        # Prepare the Schema and subtasks for processing.
        catalog = self._prepareSchemaAndSubtasks(catalog)

        # Generate catalog IDs consistently across subtasks.
        if idGenerator is None:
            idGenerator = measBase.IdGenerator()

        # Scale the variance plane. If enabled, this should be done before
        # detection.
        if self.config.doScaleVariance:
            # Here, we iterate over references to the exposures, not copies.
            for band in mExposure.bands:
                self._scaleVariance(mExposure[band], band=f"'{band}'")

        # Detect sources in the reference band and populate the catalog.
        if self.config.doDetect:
            catalog, backgroundList = self._detectSources(mExposure[refBand], idGenerator)
        else:
            self.log.info("Skipping detection; using detections from provided catalog")
            backgroundList = None

        # Deblend detected sources and update the catalog(s).
        if self.config.doDeblend:
            catalogs, self.modelData = self._deblendSources(mExposure, catalog, refBand=refBand)
        else:
            self.log.warning(
                "Skipping deblending; proceeding with the provided catalog in the reference band"
            )
            catalogs = {refBand: catalog}
            self.modelData = None

        # Process catalog(s) through measurement, aperture correction, and
        # catalog calculation subtasks.
        for band, catalog in catalogs.items():
            exposure = mExposure[band]
            self._processCatalog(exposure, catalog, idGenerator, band=f"'{band}'")

        return pipeBase.Struct(catalogs=catalogs, backgroundList=backgroundList, modelData=self.modelData)

    def _ensureValidInputs(
        self,
        mExposure: afwImage.MultibandExposure | list[afwImage.Exposure] | afwImage.Exposure,
        refBand: str | None,
        bands: list[str] | None,
        catalog: afwTable.SourceCatalog | None = None,
    ) -> tuple[afwImage.MultibandExposure, str, list[str] | None]:
        """Perform validation and adjustments of inputs without heavy
        computation.

        Parameters
        ----------
        mExposure :
            Multi-band data to be processed by the driver task.
        refBand :
            Reference band to use for detection or measurements.
        bands :
            List of bands associated with the exposures in ``mExposure``.
        catalog :
            Catalog to be extended by the driver task.

        Returns
        -------
        mExposure :
            Multi-band exposure to be processed by the driver task.
        refBand :
            Reference band to use for detection or measurements.
        bands :
            List of bands associated with the exposures in ``mExposure``.
        """

        # Perform basic checks that are shared with all driver tasks.
        super()._ensureValidInputs(catalog)

        # Multi-band-specific validation and adjustments.
        if isinstance(mExposure, afwImage.MultibandExposure):
            if bands is not None:
                if any(b not in mExposure.bands for b in bands):
                    raise ValueError(
                        "Some bands in the 'bands' list are not present in the input multi-band exposure"
                    )
                self.log.info(
                    f"Using bands {bands} out of the available {mExposure.bands} in the multi-band exposure"
                )
        elif isinstance(mExposure, list):
            if bands is None:
                raise ValueError("The 'bands' list must be provided if 'mExposure' is a list")
            if len(bands) != len(mExposure):
                raise ValueError("Number of bands and exposures must match")
        elif isinstance(mExposure, afwImage.Exposure):
            if bands is not None and len(bands) != 1:
                raise ValueError(
                    "The 'bands' list, if provided, must only contain a single band "
                    "if a single-band exposure is given"
                )
            if bands is None and refBand is None:
                refBand = "unknown"  # Placeholder for single-band deblending
                bands = [refBand]
            elif bands is None and refBand is not None:
                bands = [refBand]
            elif bands is not None and refBand is None:
                refBand = bands[0]
        else:
            raise TypeError(f"Unsupported 'mExposure' type: {type(mExposure)}")

        # Convert mExposure to a MultibandExposure object with the bands
        # provided.
        mExposure = self._buildMultibandExposure(mExposure, bands)

        if len(mExposure.bands) == 1:
            # N.B. Scarlet is designed to leverage multi-band information to
            # differentiate overlapping sources based on their spectral and
            # spatial profiles. However, it can also run on a single band and
            # often give better results than 'meas_deblender'.
            self.log.info(f"Running '{self._Deblender}' in single-band mode; make sure it was intended!")
            if refBand is None:
                refBand = mExposure.bands[0]
                self.log.info(
                    "No reference band provided for single-band data; "
                    f"using the only available band ('{refBand}') as the reference band"
                )
        else:
            if catalog is None:
                if self.config.measureOnlyInRefBand:
                    measInfo = "and everything downstream of deblending"
                else:
                    measInfo = (
                        "while subtasks downstream of deblending will be run in each of "
                        f"the {mExposure.bands} bands"
                    )
                self.log.info(f"Using '{refBand}' as the reference band for detection {measInfo}")

        # Final sanity checks after all the adjustments above.
        if refBand is None:
            raise ValueError("Reference band must be provided for multi-band data")

        if refBand not in mExposure.bands:
            raise ValueError(f"Requested band '{refBand}' is not present in the multi-band exposure")

        if bands is not None and refBand not in bands:
            raise ValueError(f"Reference band '{refBand}' is not in the list of 'bands' provided: {bands}")

        return mExposure, refBand, bands

    def _deblendSources(
        self, mExposure: afwImage.MultibandExposure, catalog: afwTable.SourceCatalog, refBand: str
    ) -> tuple[dict[str, afwTable.SourceCatalog], scl.io.ScarletModelData]:
        """Run multi-band deblending given a multi-band exposure and a catalog.

        Parameters
        ----------
        mExposure :
            Multi-band exposure on which to run the deblending algorithm.
        catalog :
            Catalog containing sources to be deblended.
        refBand :
            Reference band used for detection or the band to use for
            measurements if `measureOnlyInRefBand` is enabled.

        Returns
        -------
        catalogs :
            Dictionary of catalogs containing the deblended sources. If
            `measureOnlyInRefBand` is enabled, this will only contain the
            reference-band catalog; otherwise, it will contain a catalog for
            each band.
        modelData :
            Multiband scarlet models produced during deblending.
        """
        self.log.info(f"Deblending using '{self._Deblender}' on {len(catalog)} detection footprints")

        # Run the deblender on the multi-band exposure.
        catalog, modelData = self.deblend.run(mExposure, catalog)

        # Determine which bands to process post-deblending.
        bands = [refBand] if self.config.measureOnlyInRefBand else mExposure.bands

        catalogs = {band: catalog.copy(deep=True) for band in bands}
        for band in bands:
            # The footprints need to be updated for the subsequent measurement.
            imageForRedistribution = mExposure[band] if self.config.doConserveFlux else None
            scarlet.io.updateCatalogFootprints(
                modelData=modelData,
                catalog=catalogs[band],  # In-place modification
                band=band,
                imageForRedistribution=imageForRedistribution,
                removeScarletData=self.config.removeScarletData,
                updateFluxColumns=self.config.updateFluxColumns,
            )

        return self._toContiguous(catalogs), modelData

    def _buildMultibandExposure(
        self,
        mExposureData: afwImage.MultibandExposure | list[afwImage.Exposure] | afwImage.Exposure,
        bands: list[str] | None,
    ) -> afwImage.MultibandExposure:
        """Convert a single-band exposure or a list of single-band exposures to
        a `MultibandExposure` if not already of that type.

        No conversion will be done if ``mExposureData`` is already a
        `MultibandExposure` except it will be subsetted to the bands provided.

        Parameters
        ----------
        mExposureData :
            Input multi-band data.
        bands :
            List of bands associated with the exposures in ``mExposure``. Only
            required if ``mExposure`` is a list of single-band exposures. If
            provided while ``mExposureData`` is a ``MultibandExposure``, it
            will be used to select a specific subset of bands from the
            available ones.

        Returns
        -------
        mExposure :
            Converted multi-band exposure.
        """
        if isinstance(mExposureData, afwImage.MultibandExposure):
            if bands and not set(bands).issubset(mExposureData.bands):
                raise ValueError(
                    f"Requested bands {bands} are not a subset of available bands: {mExposureData.bands}"
                )
            return mExposureData[bands,] if bands and len(bands) > 1 else mExposureData
        elif isinstance(mExposureData, list):
            mExposure = afwImage.MultibandExposure.fromExposures(bands, mExposureData)
        elif isinstance(mExposureData, afwImage.Exposure):
            # We still need to build a multi-band exposure to satisfy scarlet
            # function's signature, even when using a single band.
            mExposure = afwImage.MultibandExposure.fromExposures(bands, [mExposureData])

        # Attach the WCS from each input exposure to the corresponding band of
        # the multi-band exposure; otherwise, their WCS will be None,
        # potentially causing issues downstream. Note that afwImage does not do
        # this when constructing a MultibandExposure from exposures.
        for band, exposure in zip(bands, mExposureData):
            mExposure[band].setWcs(exposure.getWcs())

        return mExposure
