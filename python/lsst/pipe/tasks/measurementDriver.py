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

import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.meas.algorithms as measAlgorithms
import lsst.meas.base as measBase
import lsst.meas.deblender as measDeblender
import lsst.meas.extensions.scarlet as scarlet
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np

logging.basicConfig(level=logging.INFO)


class MeasurementDriverBaseConfig(pexConfig.Config):
    """Base configuration for measurement driver tasks.

    This class provides foundational configuration for its subclasses to handle
    single-band and multi-band data. It defines the detection, deblending,
    and measurement subtasks, which are intended to be executed in sequence
    by the driver tasks.
    """

    idGenerator = measBase.DetectorVisitIdGeneratorConfig.make_field(
        doc="Configuration for generating catalog IDs from data IDs consistently across subtasks."
    )

    detection = pexConfig.ConfigurableField(
        target=measAlgorithms.SourceDetectionTask, doc="Subtask to detect sources in the image."
    )

    deblender = pexConfig.ChoiceField[str](
        doc="Which deblender to use?",
        default="meas_deblender",
        allowed={
            "meas_deblender": "Deblend using meas_deblender (only single-band)",
            "scarlet": "Deblend using scarlet (single- or multi-band)",
        },
    )

    deblend = pexConfig.ConfigurableField(
        target=measDeblender.SourceDeblendTask, doc="Subtask to split blended sources into components."
    )

    measurement = pexConfig.ConfigurableField(
        target=measBase.SingleFrameMeasurementTask,
        doc="Subtask to measure sources and populate the output catalog",
    )

    def __setattr__(self, key, value):
        """Intercept changes to 'deblender' and retarget subtask if needed."""
        super().__setattr__(key, value)

        # This is to ensure the deblend target is set correctly whenever the
        # deblender is changed. This is required because `setDefaults` is not
        # automatically invoked during reconfiguration.
        if key == "deblender":
            self._retargetDeblend()

    def setDefaults(self):
        super().setDefaults()
        self._retargetDeblend()

    def _retargetDeblend(self):
        if self.deblender == "scarlet":
            self.deblend.retarget(scarlet.ScarletDeblendTask)
        elif self.deblender == "meas_deblender":
            self.deblend.retarget(measDeblender.SourceDeblendTask)

    def validate(self):
        super().validate()
        targetMap = {
            "scarlet": scarlet.ScarletDeblendTask,
            "meas_deblender": measDeblender.SourceDeblendTask,
        }

        # Ensure the deblend target aligns with the selected deblender.
        if self.deblend.target != (expected := targetMap.get(self.deblender)):
            raise ValueError(
                f"Invalid target for '{self.deblender}': expected {expected}, got {self.deblend.target}"
            )


class MeasurementDriverBaseTask(pipeBase.Task):
    """Base class for the mid-level driver running detection, deblending
    (optional), and measurement algorithms in one go.

    This driver simplifies the process of applying a small set of measurement
    algorithms to images by abstracting away schema and table boilerplate. It
    is particularly suited for simple use cases, such as processing images
    without neighbor-noise-replacement or extensive configuration.

    Parameters
    ----------
    schema :
        Schema used to create the output `~lsst.afw.table.SourceCatalog`,
        modified in place with fields that will be written by this task.
    **kwargs :
        Additional kwargs to pass to lsst.pipe.base.Task.__init__()

    Notes
    -----
    Subclasses (e.g. single-band vs multi-band) override how inputs are built
    or validated, but rely on this base for the pipeline logic.
    """

    ConfigClass = MeasurementDriverBaseConfig
    _DefaultName = "measurementDriverBase"

    def __init__(self, schema: afwTable.Schema = None, **kwargs: dict):
        super().__init__(**kwargs)

        if schema is None:
            # Create a minimal schema that will be extended by tasks.
            self.schema = afwTable.SourceTable.makeMinimalSchema()
        else:
            self.schema = schema

        # Add coordinate error fields to avoid missing field issues in the
        # schema.
        afwTable.CoordKey.addErrorFields(self.schema)

        # Standard subtasks to run in sequence.
        self.subtaskNames = ["detection", "deblend", "measurement"]

    def makeSubtasks(self):
        """Construct subtasks based on the current configuration."""
        for name in self.subtaskNames:
            if not hasattr(self, name):
                self.makeSubtask(name, schema=self.schema)

    def run(
        self, exposure: afwImage.Exposure, idGenerator: measBase.IdGenerator = None
    ) -> afwTable.SourceCatalog:
        """Run detection, optional deblending, and measurement on a given
        image.

        Parameters
        ----------
        exposure :
            The exposure on which to detect, deblend and measure sources.
        idGenerator : optional
            Object that generates source IDs and provides random seeds.

        Returns
        -------
        catalog :
            The source catalog with all requested measurements.
        """
        # Make the `deblend` subtask only if it is enabled.
        if self.config.deblender is None:
            self.subtasks.remove("deblend")

        # Validate the configuration.
        self.config.validate()

        # Ensure this method picks up the current subtask config.
        self.makeSubtasks()
        # N.B. subtasks must be created here to handle reconfigurations, such
        # as retargeting the `deblend` subtask, because the `makeSubtask`
        # method locks in its config just before creating the subtask. If the
        # subtask was already made in __init__ using the initial config, it
        # cannot be retargeted now because retargeting happens at the config
        # level, not the subtask level.

        if idGenerator is None:
            idGenerator = measBase.IdGenerator()

        self.exposure = exposure

        # Create an empty source table with the known schema into which
        # detections will be placed next.
        self.catalog = afwTable.SourceTable.make(self.schema, idGenerator.make_table_id_factory())

        # Step 1: Detect sources in the image and populate the catalog.
        self._detectSources()

        # Step 2: If enabled, deblend detected sources and update the catalog.
        if self.config.deblender:
            self._deblendSources()
        else:
            self.log.info("Deblending is disabled; skipping deblending")

        # Step 3: Measure properties of detected/deblended sources.
        self._measureSources()

        return self.catalog

    def _detectSources(self):
        """Run the detection subtask to identify sources in the image."""
        self.log.info(f"Running detection on a {self.exposure.width}x{self.exposure.height} pixel exposure")
        self.catalog = self.detection.run(self.catalog, self.exposure).sources

    def _deblendSources(self):
        """Run the deblending subtask to separate blended sources."""
        self.log.info(
            f"Deblending using '{self.config.deblender}' on {len(self.catalog)} detection footprints"
        )
        if self.config.deblender == "meas_deblender":
            self.deblend.run(exposure=self.exposure, sources=self.catalog)
        elif self.config.deblender == "scarlet":
            if not isinstance(self.exposure, afwImage.MultibandExposure):
                # We need to have a multiband exposure to satisfy scarlet
                # function's signature, even when using a single band.
                self.band = "N/A"  # Placeholder for single-band deblending
                self.mExposure = afwImage.MultibandExposure.fromExposures([self.band], [self.exposure])
            self.catalog, modelData = self.deblend.run(mExposure=self.mExposure, mergedSources=self.catalog)
            # The footprints need to be updated for the subsequent measurement.
            scarlet.io.updateCatalogFootprints(
                modelData=modelData,
                catalog=self.catalog,
                band=self.band,
                imageForRedistribution=None,
                removeScarletData=True,
                updateFluxColumns=True,
            )
        # The deblender may not produce a contiguous catalog; ensure contiguity
        # for the subsequent task.
        if not self.catalog.isContiguous():
            self.log.info("Catalog is not contiguous; making it contiguous")
            self.catalog = self.catalog.copy(deep=True)

    def _measureSources(self):
        """Run the measurement subtask to compute properties of sources."""
        isDeblended = "and deblended" if self.config.deblender else "(not deblended)"
        self.log.info(f"Measuring {len(self.catalog)} detected {isDeblended} sources")
        self.measurement.run(self.catalog, self.exposure)
        self.log.info(
            f"Measurement complete - output catalog has " f"{self.catalog.schema.getFieldCount()} fields"
        )


class SingleBandMeasurementDriverConfig(MeasurementDriverBaseConfig):
    """Configuration for single-band measurement driver tasks.

    No additional parameters specific to single-band processing is added.
    """

    pass


class SingleBandMeasurementDriverTask(MeasurementDriverBaseTask):
    """Mid-level driver for processing single-band data.

    Provides an additional interface for handling raw image data that is
    specific to single-band scenarios.

    Examples
    --------
    Here is an example of how to use this class to run detection, deblending,
    and measurement on a single-band exposure:
    >>> from lsst.pipe.tasks.measurementDriver import (
    ...     SingleBandMeasurementDriverConfig,
    ...     SingleBandMeasurementDriverTask,
    ... )
    >>> import lsst.meas.extensions.shapeHSM  # To register its plugins
    >>> config = SingleBandMeasurementDriverConfig()
    >>> config.detection.thresholdValue = 5.5
    >>> config.deblender = "meas_deblender"
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

    _DefaultName = "singleBandMeasurementDriver"
    ConfigClass = SingleBandMeasurementDriverConfig

    def run(self, *args, **kwargs):
        if self.config.deblender == "scarlet":
            # N.B. scarlet is designed to leverage multiband information to
            # differentiate overlapping sources based on their spectral and
            # spatial profiles. However, it can also run on a single band and
            # often give better results than 'meas_deblender'.
            self.log.debug("Using 'scarlet' deblender for single-band processing; make sure it was intended")
        return super().run(*args, **kwargs)

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
            Mask data for the image. Used if 'image' is a bare `array` or
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
        # Convert raw image data into an Exposure
        # exposure = self._makeExposureFromImage(image, mask, variance, wcs, psf, photoCalib)
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


class MultiBandMeasurementDriverConfig(MeasurementDriverBaseConfig):
    """Configuration for multi-band measurement driver tasks.

    Adds a validation check to ensure the 'scarlet' deblender is used.
    """

    def validate(self):
        super().validate()
        if self.deblender != "scarlet":
            raise ValueError(
                f"Multi-band deblending requires the 'scarlet' deblender, but got '{self.deblender}'."
            )


class MultiBandMeasurementDriverTask(MeasurementDriverBaseTask):
    """Mid-level driver for processing multi-band data.

    Provides functionality for handling a list of single-band exposures in
    addition to a multi-band exposure.

    Examples
    --------
    Here is an example of how to use this class to run detection, deblending,
    and measurement on a multi-band exposure:
    >>> from lsst.afw.image import MultibandExposure
    >>> from lsst.pipe.tasks.measurementDriver import (
    ...     MultiBandMeasurementDriverConfig,
    ...     MultiBandMeasurementDriverTask,
    ... )
    >>> import lsst.meas.extensions.shapeHSM  # To register its plugins
    >>> config = MultiBandMeasurementDriverConfig()
    >>> config.detection.thresholdValue = 5.5
    >>> config.deblender = "scarlet"
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

    def run(
        self,
        mExposure: afwImage.MultibandExposure | list[afwImage.Exposure],
        band: str,
        bands: list[str] | None = None,
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
        band :
            Reference band to use for detection and measurement.
        bands : optional
            List of bands associated with the exposures in ``mExposure``. Only
            required if ``mExposure`` is a list of single-band exposures.
        idGenerator : optional
            Generator for unique source IDs.

        Returns
        -------
        catalog :
            Catalog containing the measured sources.
        """
        # Store the reference band for later use.
        self.band = band

        # Convert list of exposures to a MultibandExposure if needed. Save the
        # result as an instance attribute for later use.
        self.mExposure = self._buildMultibandExposure(mExposure, bands)

        if self.band not in self.mExposure:
            raise ValueError(f"Requested band '{band}' is not present in the multiband exposure.")

        # Use the reference band for detection and measurement.
        exposure = self.mExposure[self.band]
        self.log.info(f"Using '{self.band}' band as the reference band for detection and measurement")

        return super().run(exposure, idGenerator=idGenerator)

    def _buildMultibandExposure(
        self, exposure: afwImage.MultibandExposure | list[afwImage.Exposure], bands: list[str] | None
    ) -> afwImage.MultibandExposure:
        """
        Convert a list of single-band exposures to a MultibandExposure if needed.

        Parameters
        ----------
        exposure :
            Input multi-band data.
        bands : optional
            List of bands associated with the exposures in ``exposure``. Only
            required if ``exposure`` is a list of single-band exposures.

        Returns
        -------
        mbExposure :
            Converted multi-band exposure.
        """
        if isinstance(exposure, afwImage.MultibandExposure):
            if bands is not None:
                self.log.warn("Ignoring 'bands' argument; using bands from the input MultibandExposure")
            return exposure
        elif isinstance(exposure, list):
            if bands is None:
                raise ValueError("List of bands must be provided if 'exposure' is a list")
            if len(bands) != len(exposure):
                raise ValueError("Number of bands and exposures must match.")
            return afwImage.MultibandExposure.fromExposures(bands, exposure)
        else:
            raise TypeError("'exposure' must be a MultibandExposure or a list of single-band Exposures.")
