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

__all__ = ["MeasurementDriverConfig", "MeasurementDriverTask"]

import logging

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.meas.algorithms as measAlgorithms
import lsst.meas.base as measBase
import lsst.meas.deblender as measDeblender
import lsst.meas.extensions.scarlet as scarlet
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np

logging.basicConfig(level=logging.INFO)


class MeasurementDriverConfig(pexConfig.Config):
    """Configuration parameters for `MeasurementDriverTask`."""

    # To generate catalog ids consistently across subtasks.
    id_generator = measBase.DetectorVisitIdGeneratorConfig.make_field()

    detection = pexConfig.ConfigurableField(
        target=measAlgorithms.SourceDetectionTask,
        doc="Task to detect sources to return in the output catalog.",
    )

    deblender = pexConfig.ChoiceField[str](
        doc="The deblender to use.",
        default="meas_deblender",
        allowed={"meas_deblender": "Deblend using meas_deblender", "scarlet": "Deblend using scarlet"},
    )

    deblend = pexConfig.ConfigurableField(
        target=measDeblender.SourceDeblendTask, doc="Split blended sources into their components."
    )

    measurement = pexConfig.ConfigurableField(
        target=measBase.SingleFrameMeasurementTask,
        doc="Task to measure sources to return in the output catalog.",
    )

    def __setattr__(self, key, value):
        """Intercept attribute setting to trigger setDefaults when relevant
        fields change.
        """
        super().__setattr__(key, value)

        # This is to ensure the deblend target is set correctly whenever the
        # deblender is changed. This is required because `setDefaults` is not
        # automatically invoked during reconfiguration.
        if key == "deblender":
            self.setDefaults()

    def validate(self):
        super().validate()

        # Ensure the deblend target aligns with the selected deblender.
        if self.deblender == "scarlet":
            assert self.deblend.target == scarlet.ScarletDeblendTask
        elif self.deblender == "meas_deblender":
            assert self.deblend.target == measDeblender.SourceDeblendTask
        elif self.deblender is not None:
            raise ValueError(f"Invalid deblender value: {self.deblender}")

    def setDefaults(self):
        super().setDefaults()
        if self.deblender == "scarlet":
            self.deblend.retarget(scarlet.ScarletDeblendTask)
        elif self.deblender == "meas_deblender":
            self.deblend.retarget(measDeblender.SourceDeblendTask)


class MeasurementDriverTask(pipeBase.Task):
    """A mid-level driver for running detection, deblending (optional), and
    measurement algorithms in one go.

    This driver simplifies the process of applying a small set of measurement
    algorithms to images by abstracting away schema and table boilerplate. It
    is particularly suited for simple use cases, such as processing images
    without neighbor-noise-replacement or extensive configuration.

    Designed to streamline the measurement framework, this class integrates
    detection, deblending (if enabled), and measurement into a single workflow.

    Parameters
    ----------
    schema : `~lsst.afw.table.Schema`
        Schema used to create the output `~lsst.afw.table.SourceCatalog`,
        modified in place with fields that will be written by this task.
    **kwargs : `dict`
        Additional kwargs to pass to lsst.pipe.base.Task.__init__()

    Examples
    --------
    Here is an example of how to use this class to run detection, deblending,
    and measurement on a given exposure:
    >>> from lsst.pipe.tasks.measurementDriver import MeasurementDriverTask
    >>> import lsst.meas.extensions.shapeHSM  # To register its plugins
    >>> config = MeasurementDriverTask().ConfigClass()
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
    >>> driver = MeasurementDriverTask(config=config)
    >>> catalog = driver.run(exposure)
    >>> catalog.writeFits("meas_catalog.fits")
    """

    ConfigClass = MeasurementDriverConfig
    _DefaultName = "measurementDriver"

    def __init__(self, schema=None, **kwargs):
        super().__init__(**kwargs)

        if schema is None:
            # Create a minimal schema that will be extended by tasks.
            self.schema = afwTable.SourceTable.makeMinimalSchema()
        else:
            self.schema = schema

        # Add coordinate error fields to the schema (this is to avoid errors
        # such as: "Field with name 'coord_raErr' not found with type 'F'").
        afwTable.CoordKey.addErrorFields(self.schema)

        self.subtasks = ["detection", "deblend", "measurement"]

    def make_subtasks(self):
        """Create subtasks based on the current configuration."""
        for name in self.subtasks:
            self.makeSubtask(name, schema=self.schema)

    def run(
        self,
        image,
        bands=None,
        band=None,
        mask=None,
        variance=None,
        psf=None,
        wcs=None,
        photo_calib=None,
        id_generator=None,
    ):
        """Run detection, optional deblending, and measurement on a given
        image.

        Parameters
        ----------
        image: `~lsst.afw.image.Exposure` or `~lsst.afw.image.MaskedImage` or
            `~lsst.afw.image.Image` or `np.ndarray` or
            `~lsst.afw.image.MultibandExposure` or
            `list` of `~lsst.afw.image.Exposure`
            The image on which to detect, deblend and measure sources. If
            provided as a multiband exposure, or a list of `Exposure` objects,
            it can be taken advantage of by the 'scarlet' deblender. When using
            a list of `Exposure` objects, the ``bands`` parameter must also be
            provided.
        bands: `str` or `list` of `str`, optional
            The bands of the input image. Required if ``image`` is provided as
            a list of `Exposure` objects. Example: ["g", "r", "i", "z", "y"]
            or "grizy".
        band: `str`, optional
            The target band of the image to use for detection and measurement.
            Required when ``image`` is provided as a `MultibandExposure`, or a
            list of `Exposure` objects.
        mask: `~lsst.afw.image.Mask`, optional
            The mask for the input image. Only used if ``image`` is provided
            as an afw `Image` or a numpy `ndarray`.
        variance: `~lsst.afw.image.Image`, optional
            The variance image for the input image. Only used if ``image`` is
            provided as an afw `Image` or a numpy `ndarray`.
        psf: `~lsst.afw.detection.Psf`, optional
            The PSF model for the input image. Will be ignored if ``image`` is
            provided as an `Exposure`, `MultibandExposure`, or a list of
            `Exposure` objects.
        wcs: `~lsst.afw.image.Wcs`, optional
            The World Coordinate System (WCS) model for the input image. Will
            be ignored if ``image`` is provided as an `Exposure`,
            `MultibandExposure`, or a list of `Exposure` objects.
        photo_calib : `~lsst.afw.image.PhotoCalib`, optional
            Photometric calibration model for the input image. Will be ignored
            if ``image`` is provided as an `Exposure`, `MultibandExposure`, or
            a list of `Exposure` objects.
        id_generator : `~lsst.meas.base.IdGenerator`, optional
            Object that generates source IDs and provides random seeds.

        Returns
        -------
        catalog : `~lsst.afw.table.SourceCatalog`
            The source catalog with all requested measurements.
        """

        # Only make the `deblend` subtask if it is enabled.
        if self.config.deblender is None:
            self.subtasks.remove("deblend")

        # Validate the configuration before running the task.
        self.config.validate()

        # This guarantees the `run` method picks up the current subtask config.
        self.make_subtasks()
        # N.B. subtasks must be created here to handle reconfigurations, such
        # as retargeting the `deblend` subtask, because the `makeSubtask`
        # method locks in its config just before creating the subtask. If the
        # subtask was already made in __init__ using the initial config, it
        # cannot be retargeted now because retargeting happens at the config
        # level, not the subtask level.

        if id_generator is None:
            id_generator = measBase.IdGenerator()

        if isinstance(image, afwImage.MultibandExposure) or isinstance(image, list):
            if self.config.deblender != "scarlet":
                self.log.debug(
                    "Supplied a multiband exposure, or a list of exposures, while the deblender is set to "
                    f"'{self.config.deblender}'. A single exposure corresponding to target `band` will be "
                    "used for everything."
                )
            if band is None:
                raise ValueError(
                    "The target `band` must be provided when using multiband exposures or a list of "
                    "exposures."
                )
            if isinstance(image, list):
                if not all(isinstance(im, afwImage.Exposure) for im in image):
                    raise ValueError("All elements in the `image` list must be `Exposure` objects.")
                if bands is None:
                    raise ValueError(
                        "The `bands` parameter must be provided if `image` is a list of `Exposure` objects."
                    )
                if not isinstance(bands, (str, list)) or (
                    isinstance(bands, list) and not all(isinstance(b, str) for b in bands)
                ):
                    raise TypeError(
                        "The `bands` parameter must be a string or a list of strings if provided."
                    )
                if len(bands) != len(image):
                    raise ValueError(
                        "The number of bands must match the number of `Exposure` objects in the list."
                    )
        else:
            if band is None:
                band = "N/A"  # Just a placeholder for single-band deblending
            else:
                self.log.warn("The target `band` is not required when the input image is not multiband.")
            if bands is not None:
                self.log.warn(
                    "The `bands` parameter will be ignored because the input image is not multiband."
                )

        if self.config.deblender == "scarlet":
            if not isinstance(image, (afwImage.MultibandExposure, list, afwImage.Exposure)):
                raise ValueError(
                    "The `image` parameter must be a `MultibandExposure`, a list of `Exposure` "
                    "objects, or a single `Exposure` when the deblender is set to 'scarlet'."
                )
            if isinstance(image, afwImage.Exposure):
                # N.B. scarlet is designed to leverage multiband information to
                # differentiate overlapping sources based on their spectral and
                # spatial profiles. However, it can also run on a single band
                # and still give better results than 'meas_deblender'.
                self.log.debug(
                    "Supplied a single-band exposure, while the deblender is set to 'scarlet'."
                    "Make sure it was intended."
                )

        # Start with some image conversions if needed.
        if isinstance(image, np.ndarray):
            image = afwImage.makeImageFromArray(image)
        if isinstance(mask, np.ndarray):
            mask = afwImage.makeMaskFromArray(mask)
        if isinstance(variance, np.ndarray):
            variance = afwImage.makeImageFromArray(variance)
        if isinstance(image, afwImage.Image):
            image = afwImage.makeMaskedImage(image, mask, variance)

        # Avoid type checker errors by being explicit from here on.
        exposure: afwImage.Exposure

        # Make sure we have an `Exposure` object to work with (potentially
        # along with a `MultiBandExposure` for scarlet deblending).
        if isinstance(image, afwImage.Exposure):
            exposure = image
        elif isinstance(image, afwImage.MaskedImage):
            exposure = afwImage.makeExposure(image, wcs)
            if psf is not None:
                exposure.setPsf(psf)
            if photo_calib is not None:
                exposure.setPhotoCalib(photo_calib)
        elif isinstance(image, list):
            # Construct a multiband exposure for scarlet deblending.
            exposures = afwImage.MultibandExposure.fromExposures(bands, image)
            # Select the exposure of the desired band, which will be used for
            # detection and measurement.
            exposure = exposures[band]
        elif isinstance(image, afwImage.MultibandExposure):
            exposures = image
            exposure = exposures[band]
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Create a source table into which detections will be placed.
        table = afwTable.SourceTable.make(self.schema, id_generator.make_table_id_factory())

        # Detect sources and get a source catalog.
        self.log.info(f"Running detection on a {exposure.width}x{exposure.height} pixel image")
        detections = self.detection.run(table, exposure)
        catalog = detections.sources

        # Deblend sources into their components and update the catalog.
        if self.config.deblender is None:
            self.log.info("Deblending is disabled; skipping deblending")
        else:
            self.log.info(
                f"Running deblending via '{self.config.deblender}' on {len(catalog)} detection footprints"
            )
            if self.config.deblender == "meas_deblender":
                self.deblend.run(exposure=exposure, sources=catalog)
            elif self.config.deblender == "scarlet":
                if not isinstance(image, (afwImage.MultibandExposure, list)):
                    # We need to have a multiband exposure to satisfy scarlet
                    # function's signature, even when using a single band.
                    exposures = afwImage.MultibandExposure.fromExposures([band], [exposure])
                catalog, model_data = self.deblend.run(mExposure=exposures, mergedSources=catalog)
                # The footprints need to be updated for the subsequent
                # measurement.
                scarlet.io.updateCatalogFootprints(
                    modelData=model_data,
                    catalog=catalog,
                    band=band,
                    imageForRedistribution=exposure,
                    removeScarletData=True,
                    updateFluxColumns=True,
                )

        # The deblender may not produce a contiguous catalog; ensure contiguity
        # for the subsequent task.
        if not catalog.isContiguous():
            self.log.info("Catalog is not contiguous; making it contiguous")
            catalog = catalog.copy(deep=True)

        # Measure requested quantities on sources.
        self.measurement.run(catalog, exposure)
        self.log.info(
            f"Measured {len(catalog)} sources and stored them in the output "
            f"catalog containing {catalog.schema.getFieldCount()} fields"
        )

        return catalog
