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

"""Extract bright star cutouts; normalize and warp, optionally fit the PSF."""

__all__ = ["BrightStarCutoutConnections", "BrightStarCutoutConfig", "BrightStarCutoutTask"]

from typing import Any, Iterable, cast

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS, FOCAL_PLANE
from lsst.afw.detection import Footprint, FootprintSet, Threshold
from lsst.afw.geom import SkyWcs, SpanSet, makeModifiedWcs
from lsst.afw.geom.transformFactory import makeTransform
from lsst.afw.image import ExposureF, ImageD, ImageF, MaskedImageF
from lsst.afw.math import BackgroundList, FixedKernel, WarpingControl, warpImage
from lsst.daf.butler import DataCoordinate
from lsst.geom import (
    AffineTransform,
    Box2I,
    Extent2D,
    Extent2I,
    Point2D,
    Point2I,
    SpherePoint,
    arcseconds,
    floor,
    radians,
    Angle,
)
from lsst.meas.algorithms import (
    BrightStarStamp,
    BrightStarStamps,
    KernelPsf,
    LoadReferenceObjectsConfig,
    ReferenceObjectLoader,
    WarpedPsf,
)
from lsst.pex.config import ChoiceField, ConfigField, Field, ListField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input, Output, PrerequisiteInput
from lsst.utils.timer import timeMethod
from copy import deepcopy
import math


NEIGHBOR_MASK_PLANE = "NEIGHBOR"


class BrightStarCutoutConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),
):
    """Connections for BrightStarCutoutTask."""

    refCat = PrerequisiteInput(
        name="gaia_dr3_20230707",
        storageClass="SimpleCatalog",
        doc="Reference catalog that contains bright star positions.",
        dimensions=("skypix",),
        multiple=True,
        deferLoad=True,
    )
    inputExposure = Input(
        name="calexp",
        storageClass="ExposureF",
        doc="Background-subtracted input exposure from which to extract bright star stamp cutouts.",
        dimensions=("visit", "detector"),
    )
    inputBackground = Input(
        name="calexpBackground",
        storageClass="Background",
        doc="Background model for the input exposure, to be added back on during processing.",
        dimensions=("visit", "detector"),
    )
    extendedPsf = Input(
        name="extendedPsf2",
        storageClass="ImageF",
        doc="Extended PSF model, built from stacking bright star cutouts.",
        dimensions=("band",),
    )
    brightStarStamps = Output(
        name="brightStarStamps",
        storageClass="BrightStarStamps",
        doc="Set of preprocessed postage stamp cutouts, each centered on a single bright star.",
        dimensions=("visit", "detector"),
    )

    def __init__(self, *, config: "BrightStarCutoutConfig | None" = None):
        super().__init__(config=config)
        assert config is not None
        if not config.use_extended_psf:
            self.inputs.remove("extendedPsf")


class BrightStarCutoutConfig(
    PipelineTaskConfig,
    pipelineConnections=BrightStarCutoutConnections,
):
    """Configuration parameters for BrightStarCutoutTask."""

    # Star selection
    mag_range = ListField[float](
        doc="Magnitude range in Gaia G. Cutouts will be made for all stars in this range.",
        default=[0, 18],
    )
    exclude_arcsec_radius = Field[float](
        doc="Stars with a star in the range ``exclude_mag_range`` mag in ``exclude_arcsec_radius`` are not "
        "used.",
        default=5,
    )
    exclude_mag_range = ListField[float](
        doc="Stars with a star in the range ``exclude_mag_range`` mag in ``exclude_arcsec_radius`` are not "
        "used.",
        default=[0, 20],
    )
    min_area_fraction = Field[float](
        doc="Minimum fraction of the stamp area, post-masking, that must remain for a cutout to be retained.",
        default=0.1,
    )
    bad_mask_planes = ListField[str](
        doc="Mask planes that identify excluded pixels for the calculation of ``min_area_fraction`` and, "
        "optionally, fitting of the PSF.",
        default=[
            "BAD",
            "CR",
            "CROSSTALK",
            "EDGE",
            "NO_DATA",
            "SAT",
            "SUSPECT",
            "UNMASKEDNAN",
            NEIGHBOR_MASK_PLANE,
        ],
    )
    stamp_size = ListField[int](
        doc="Size of the stamps to be extracted, in pixels.",
        default=(251, 251),
    )
    stamp_size_padding = Field[float](
        doc="Multiplicative factor applied to the cutout stamp size, to guard against post-warp data loss.",
        default=1.1,
    )
    warping_kernel_name = ChoiceField[str](
        doc="Warping kernel.",
        default="lanczos5",
        allowed={
            "bilinear": "bilinear interpolation",
            "lanczos3": "Lanczos kernel of order 3",
            "lanczos4": "Lanczos kernel of order 4",
            "lanczos5": "Lanczos kernel of order 5",
        },
    )
    mask_warping_kernel_name = ChoiceField[str](
        doc="Warping kernel for mask.",
        default="bilinear",
        allowed={
            "bilinear": "bilinear interpolation",
            "lanczos3": "Lanczos kernel of order 3",
            "lanczos4": "Lanczos kernel of order 4",
            "lanczos5": "Lanczos kernel of order 5",
        },
    )
    off_frame_mag_lim = Field[float](
        doc="Stars fainter than this limit are only included if they appear within the frame boundaries.",
        default=15.0,
    )

    # PSF Fitting
    use_extended_psf = Field[bool](
        doc="Use the extended PSF model to normalize bright star cutouts.",
        default=False,
    )
    do_fit_psf = Field[bool](
        doc="Fit a scaled PSF and a pedestal to each bright star cutout.",
        default=True,
    )
    use_median_variance = Field[bool](
        doc="Use the median of the variance plane for PSF fitting.",
        default=False,
    )
    psf_masked_flux_frac_threshold = Field[float](
        doc="Maximum allowed fraction of masked PSF flux for PSF fitting to occur.",
        default=0.97,
    )
    fit_iterations = Field[int](
        doc="Number of iterations over pedestal-gradient and scaling fit.",
        default=5,
    )

    # Misc

    load_reference_objects_config = ConfigField[LoadReferenceObjectsConfig](
        doc="Reference object loader for astrometric calibration.",
    )


class BrightStarCutoutTask(PipelineTask):
    """Extract bright star cutouts; normalize and warp to the same pixel grid.

    The BrightStarCutoutTask is used to extract, process, and store small image
    cutouts (or "postage stamps") around bright stars.
    This task essentially consists of three principal steps.
    First, it identifies bright stars within an exposure using a reference
    catalog and extracts a stamp around each.
    Second, it shifts and warps each stamp to remove optical distortions and
    sample all stars on the same pixel grid.
    Finally, it optionally fits a PSF plus plane flux model to the cutout.
    This final fitting procedure may be used to normalize each bright star
    stamp prior to stacking when producing extended PSF models.
    """

    ConfigClass = BrightStarCutoutConfig
    _DefaultName = "brightStarCutout"
    config: BrightStarCutoutConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        stamp_size = Extent2D(*self.config.stamp_size.list())
        stamp_radius = floor(stamp_size / 2)
        self.stamp_bbox = Box2I(corner=Point2I(0, 0), dimensions=Extent2I(1, 1)).dilatedBy(stamp_radius)
        padded_stamp_size = stamp_size * self.config.stamp_size_padding
        self.padded_stamp_radius = floor(padded_stamp_size / 2)
        self.padded_stamp_bbox = Box2I(corner=Point2I(0, 0), dimensions=Extent2I(1, 1)).dilatedBy(
            self.padded_stamp_radius
        )
        self.model_scale = 1

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        inputs["dataId"] = butlerQC.quantum.dataId
        refObjLoader = ReferenceObjectLoader(
            dataIds=[ref.datasetRef.dataId for ref in inputRefs.refCat],
            refCats=inputs.pop("refCat"),
            name=self.config.connections.refCat,
            config=self.config.load_reference_objects_config,
        )
        extendedPsf = inputs.pop("extendedPsf", None)
        output = self.run(**inputs, extendedPsf=extendedPsf, refObjLoader=refObjLoader)
        # Only ingest Stamp if it exists; prevents ingesting an empty FITS file
        if output:
            butlerQC.put(output, outputRefs)

    @timeMethod
    def run(
        self,
        inputExposure: ExposureF,
        inputBackground: BackgroundList,
        extendedPsf: ImageF | None,
        refObjLoader: ReferenceObjectLoader,
        dataId: dict[str, Any] | DataCoordinate,
    ):
        """Identify bright stars within an exposure using a reference catalog,
        extract stamps around each, warp/shift stamps onto a common frame and
        then optionally fit a PSF plus plane model.

        Parameters
        ----------
        inputExposure : `~lsst.afw.image.ExposureF`
            The background-subtracted image to extract bright star stamps.
        inputBackground : `~lsst.afw.math.BackgroundList`
            The background model associated with the input exposure.
        extendedPsf: `~lsst.afw.image.ImageF`
            The extended PSF model from previous iteration(s).
        refObjLoader : `~lsst.meas.algorithms.ReferenceObjectLoader`, optional
            Loader to find objects within a reference catalog.
        dataId : `dict` or `~lsst.daf.butler.DataCoordinate`
            The dataId of the exposure that bright stars are extracted from.
            Both 'visit' and 'detector' will be persisted in the output data.

        Returns
        -------
        brightStarResults : `~lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``brightStarStamps``
                (`~lsst.meas.algorithms.brightStarStamps.BrightStarStamps`)
        """
        wcs = inputExposure.getWcs()
        bbox = inputExposure.getBBox()
        warping_control = WarpingControl(
            self.config.warping_kernel_name, self.config.mask_warping_kernel_name
        )

        ref_cat_bright = self._get_ref_cat_bright(refObjLoader, wcs, bbox)
        zip_ra_dec = zip(ref_cat_bright["coord_ra"] * radians, ref_cat_bright["coord_dec"] * radians)
        sphere_points = [SpherePoint(ra, dec) for ra, dec in zip_ra_dec]
        pix_coords = wcs.skyToPixel(sphere_points)

        # Restore original subtracted background
        inputMI = inputExposure.getMaskedImage()
        inputMI += inputBackground.getImage()

        # Set up NEIGHBOR mask plane; associate footprints with stars
        inputExposure.mask.addMaskPlane(NEIGHBOR_MASK_PLANE)
        all_footprints, associations = self._associate_footprints(inputExposure, pix_coords, plane="DETECTED")

        # TODO: If we eventually have better PhotoCalibs (eg FGCM), apply here
        inputMI = inputExposure.getPhotoCalib().calibrateImage(inputMI, False)

        # Set up transform
        detector = inputExposure.detector
        pixel_scale = wcs.getPixelScale().asArcseconds() * arcseconds
        pix_to_focal_plane_tan = detector.getTransform(PIXELS, FIELD_ANGLE).then(
            makeTransform(AffineTransform.makeScaling(1 / pixel_scale.asRadians()))
        )

        # Loop over each bright star
        stamps, good_fracs, stamps_fit_psf_results = [], [], []
        for star_index, (obj, pix_coord) in enumerate(zip(ref_cat_bright, pix_coords)):  # type: ignore
            # Excluding faint stars that are not within the frame.
            if obj["mag"] > self.config.off_frame_mag_lim and not self.star_in_frame(pix_coord, bbox):
                continue
            footprint_index = associations.get(star_index, None)
            stampMI = MaskedImageF(self.padded_stamp_bbox)

            # Set NEIGHBOR footprints in the mask plane
            if footprint_index:
                neighbor_footprints = [fp for i, fp in enumerate(all_footprints) if i != footprint_index]
                self._set_footprints(inputMI, neighbor_footprints, NEIGHBOR_MASK_PLANE)
            else:
                self._set_footprints(inputMI, all_footprints, NEIGHBOR_MASK_PLANE)

            # Define linear shifting to recenter stamps
            coord_focal_plane_tan = pix_to_focal_plane_tan.applyForward(pix_coord)  # center of warped star
            shift = makeTransform(AffineTransform(Point2D(0, 0) - coord_focal_plane_tan))
            angle = np.arctan2(coord_focal_plane_tan.getY(), coord_focal_plane_tan.getX()) * radians
            rotation = makeTransform(AffineTransform.makeRotation(-angle))
            pix_to_polar = pix_to_focal_plane_tan.then(shift).then(rotation)

            # Apply the warp to the star stamp (in-place)
            warpImage(stampMI, inputMI, pix_to_polar, warping_control)

            # Trim to the base stamp size, check mask coverage, update metadata
            stampMI = stampMI[self.stamp_bbox]
            bad_mask_bit_mask = stampMI.mask.getPlaneBitMask(self.config.bad_mask_planes)
            good_frac = np.sum(stampMI.mask.array & bad_mask_bit_mask == 0) / stampMI.mask.array.size
            good_fracs.append(good_frac)
            if good_frac < self.config.min_area_fraction:
                continue

            # Fit a scaled PSF and a pedestal to each bright star cutout
            psf = WarpedPsf(inputExposure.getPsf(), pix_to_polar, warping_control)
            constant_psf = KernelPsf(FixedKernel(psf.computeKernelImage(Point2D(0, 0))))
            if self.config.use_extended_psf:
                psf_image = deepcopy(extendedPsf)  # Assumed to be warped, center at [0,0]
            else:
                psf_image = constant_psf.computeKernelImage(constant_psf.getAveragePosition())
                # TODO: maybe we want to generate a smaller psf in case the following happens?
                # The following could happen for when the user chooses small stamp_size ~(50, 50)
                if (
                    psf_image.array.shape[0] > stampMI.image.array.shape[0]
                    or psf_image.array.shape[1] > stampMI.image.array.shape[1]
                ):
                    continue
            # Computing an scale factor that brings the model to the similar level of the star.
            self.estimate_model_scale_value(stampMI, psf_image)
            psf_image.array *= self.model_scale  # ####### model scale correction ########

            fit_psf_results = {}

            if self.config.do_fit_psf:
                fit_psf_results = self._fit_psf(stampMI, psf_image)
            stamps_fit_psf_results.append(fit_psf_results)

            # Save the stamp if the PSF fit was successful or no fit requested
            if fit_psf_results or not self.config.do_fit_psf:
                distance_mm, theta_angle = self.star_location_on_focal(pix_coord, detector)

                stamp = BrightStarStamp(
                    stamp_im=stampMI,
                    psf=constant_psf,
                    wcs=makeModifiedWcs(pix_to_polar, wcs, False),
                    visit=cast(int, dataId["visit"]),
                    detector=cast(int, dataId["detector"]),
                    ref_id=obj["id"],
                    ref_mag=obj["mag"],
                    position=pix_coord,
                    focal_plane_radius=distance_mm,
                    focal_plane_angle=theta_angle,  # TODO: add the lsst.geom.Angle here
                    scale=fit_psf_results.get("scale", None),
                    scale_err=fit_psf_results.get("scale_err", None),
                    pedestal=fit_psf_results.get("pedestal", None),
                    pedestal_err=fit_psf_results.get("pedestal_err", None),
                    pedestal_scale_cov=fit_psf_results.get("pedestal_scale_cov", None),
                    gradient_x=fit_psf_results.get("x_gradient", None),
                    gradient_y=fit_psf_results.get("y_gradient", None),
                    curvature_x=fit_psf_results.get("curvature_x", None),
                    curvature_y=fit_psf_results.get("curvature_y", None),
                    cross_tilt=fit_psf_results.get("cross_tilt", None),
                    global_reduced_chi_squared=fit_psf_results.get("global_reduced_chi_squared", None),
                    global_degrees_of_freedom=fit_psf_results.get("global_degrees_of_freedom", None),
                    psf_reduced_chi_squared=fit_psf_results.get("psf_reduced_chi_squared", None),
                    psf_degrees_of_freedom=fit_psf_results.get("psf_degrees_of_freedom", None),
                    psf_masked_flux_fraction=fit_psf_results.get("psf_masked_flux_frac", None),
                )
                stamps.append(stamp)

        self.log.info(
            "Extracted %i bright star stamp%s. "
            "Excluded %i star%s: insufficient area (%i), PSF fit failure (%i).",
            len(stamps),
            "" if len(stamps) == 1 else "s",
            len(ref_cat_bright) - len(stamps),
            "" if len(ref_cat_bright) - len(stamps) == 1 else "s",
            np.sum(np.array(good_fracs) < self.config.min_area_fraction),
            (
                np.sum(np.isnan([x.get("pedestal", np.nan) for x in stamps_fit_psf_results]))
                if self.config.do_fit_psf
                else 0
            ),
        )
        brightStarStamps = BrightStarStamps(stamps)
        return Struct(brightStarStamps=brightStarStamps)

    def star_location_on_focal(self, pix_coord, detector):
        """
        Calculates the radial coordinates of a star on the focal plane.

        Transforms the given pixel coordinates to the focal plane and computes
        the radial distance and angle relative to the optical axis.

        Args:
            pix_coord: `~lsst.geom.Point2D` or tuple
                The (x, y) coordinates of the star on the
                detector in pixels.
            detector (Detector): `~lsst.afw.cameraGeom.Detector`
                The detector object capable of transforming coordinates
                from PIXELS to FOCAL_PLANE.

        Returns:
            tuple: A tuple containing:
                - distance_mm (float): The radial distance from the center in millimeters.
                - theta_angle (Angle): The azimuthal angle of the star on the focal plane.
        """
        star_focal_plane_coords = detector.transform(pix_coord, PIXELS, FOCAL_PLANE)
        star_x_fp = star_focal_plane_coords.getX()
        star_y_fp = star_focal_plane_coords.getY()
        distance_mm = np.sqrt(star_x_fp ** 2 + star_y_fp ** 2)
        theta_rad = math.atan2(star_y_fp, star_x_fp)
        theta_angle = Angle(theta_rad, radians)
        return distance_mm, theta_angle

    def star_in_frame(self, pix_coord, inputExposureBBox):
        """
        Checks if a star's pixel coordinates lie within the exposure boundaries.

        Args:
            pix_coord: `~lsst.geom.Point2D` or tuple
                The (x, y) pixel coordinates of the star.
            inputExposureBBox : `~lsst.geom.Box2I`
                Bounding box of the exposure.

        Returns:
            bool: True if the coordinates are within the frame limits, False otherwise.
        """
        if (
            pix_coord[0] < 0
            or pix_coord[1] < 0
            or pix_coord[0] > inputExposureBBox.getDimensions()[0]
            or pix_coord[1] > inputExposureBBox.getDimensions()[1]
        ):
            return False
        return True

    def _get_ref_cat_bright(self, refObjLoader: ReferenceObjectLoader, wcs: SkyWcs, bbox: Box2I) -> Table:
        """Get a bright star subset of the reference catalog.

        Trim the reference catalog to only those objects within the exposure
        bounding box dilated by half the bright star stamp size.
        This ensures all stars that overlap the exposure are included.

        Parameters
        ----------
        refObjLoader : `~lsst.meas.algorithms.ReferenceObjectLoader`
            Loader to find objects within a reference catalog.
        wcs : `~lsst.afw.geom.SkyWcs`
            World coordinate system.
        bbox : `~lsst.geom.Box2I`
            Bounding box of the exposure.

        Returns
        -------
        ref_cat_bright : `~astropy.table.Table`
            Bright star subset of the reference catalog.
        """
        dilated_bbox = bbox.dilatedBy(self.padded_stamp_radius)
        within_exposure = refObjLoader.loadPixelBox(dilated_bbox, wcs, filterName="phot_g_mean")
        ref_cat_full = within_exposure.refCat
        flux_field: str = within_exposure.fluxField

        prox_flux_range = sorted(((self.config.exclude_mag_range * u.ABmag).to(u.nJy)).to_value())
        bright_flux_range = sorted(((self.config.mag_range * u.ABmag).to(u.nJy)).to_value())

        subset_stars = (ref_cat_full[flux_field] > np.min((prox_flux_range[0], bright_flux_range[0]))) & (
            ref_cat_full[flux_field] < np.max((prox_flux_range[1], bright_flux_range[1]))
        )
        ref_cat_subset = Table(
            ref_cat_full.extract("id", "coord_ra", "coord_dec", flux_field, where=subset_stars)
        )

        prox_stars = (ref_cat_subset[flux_field] >= prox_flux_range[0]) & (
            ref_cat_subset[flux_field] <= prox_flux_range[1]
        )
        bright_stars = (ref_cat_subset[flux_field] >= bright_flux_range[0]) & (
            ref_cat_subset[flux_field] <= bright_flux_range[1]
        )

        coords = SkyCoord(ref_cat_subset["coord_ra"], ref_cat_subset["coord_dec"], unit="rad")
        exclude_arcsec_radius = self.config.exclude_arcsec_radius * u.arcsec  # type: ignore
        ref_cat_bright_isolated = []
        for coord in cast(Iterable[SkyCoord], coords[bright_stars]):
            neighbors = coords[prox_stars]
            seps = coord.separation(neighbors).to(u.arcsec)
            too_close = (seps > 0) & (seps <= exclude_arcsec_radius)  # not self matched
            ref_cat_bright_isolated.append(not too_close.any())

        ref_cat_bright = cast(Table, ref_cat_subset[bright_stars][ref_cat_bright_isolated])

        flux_nanojansky = ref_cat_bright[flux_field][:] * u.nJy  # type: ignore
        ref_cat_bright["mag"] = flux_nanojansky.to(u.ABmag).to_value()  # AB magnitudes

        self.log.info(
            "Identified %i of %i star%s which satisfy: frame overlap; in the range %s mag; no neighboring "
            "stars within %s arcsec.",
            len(ref_cat_bright),
            len(ref_cat_full),
            "" if len(ref_cat_full) == 1 else "s",
            self.config.mag_range,
            self.config.exclude_arcsec_radius,
        )

        return ref_cat_bright

    def _associate_footprints(
        self, inputExposure: ExposureF, pix_coords: list[Point2D], plane: str
    ) -> tuple[list[Footprint], dict[int, int]]:
        """Associate footprints from a given mask plane with specific objects.

        Footprints from the given mask plane are associated with objects at the
        coordinates provided, where possible.

        Parameters
        ----------
        inputExposure : `~lsst.afw.image.ExposureF`
            The input exposure with a mask plane.
        pix_coords : `list` [`~lsst.geom.Point2D`]
            The pixel coordinates of the objects.
        plane : `str`
            The mask plane used to identify masked pixels.

        Returns
        -------
        footprints : `list` [`~lsst.afw.detection.Footprint`]
            The footprints from the input exposure.
        associations : `dict`[int, int]
            Association indices between objects (key) and footprints (value).
        """
        det_threshold = Threshold(inputExposure.mask.getPlaneBitMask(plane), Threshold.BITMASK)
        footprintSet = FootprintSet(inputExposure.mask, det_threshold)
        footprints = footprintSet.getFootprints()
        associations = {}
        for star_index, pix_coord in enumerate(pix_coords):
            for footprint_index, footprint in enumerate(footprints):
                if footprint.contains(Point2I(pix_coord)):
                    associations[star_index] = footprint_index
                    break
        self.log.debug(
            "Associated %i of %i star%s to one each of the %i %s footprint%s.",
            len(associations),
            len(pix_coords),
            "" if len(pix_coords) == 1 else "s",
            len(footprints),
            plane,
            "" if len(footprints) == 1 else "s",
        )
        return footprints, associations

    def _set_footprints(self, inputExposure: ExposureF, footprints: list, mask_plane: str):
        """Set footprints in a given mask plane.

        Parameters
        ----------
        inputExposure : `~lsst.afw.image.ExposureF`
            The input exposure to modify.
        footprints : `list` [`~lsst.afw.detection.Footprint`]
            The footprints to set in the mask plane.
        mask_plane : `str`
            The mask plane to set the footprints in.

        Notes
        -----
        This method modifies the ``inputExposure`` object in-place.
        """
        det_threshold = Threshold(inputExposure.mask.getPlaneBitMask(mask_plane), Threshold.BITMASK)
        det_threshold_value = int(det_threshold.getValue())
        footprint_set = FootprintSet(inputExposure.mask, det_threshold)

        # Wipe any existing footprints in the mask plane
        inputExposure.mask.clearMaskPlane(int(np.log2(det_threshold_value)))

        # Set the footprints in the mask plane
        footprint_set.setFootprints(footprints)
        footprint_set.setMask(inputExposure.mask, mask_plane)

    def _fit_psf(self, stampMI: MaskedImageF, psf_image: ImageD | ImageF) -> dict[str, Any]:
        """Fit a scaled PSF and a pedestal to each bright star cutout.

        Parameters
        ----------
        stampMI : `~lsst.afw.image.MaskedImageF`
            The masked image of the bright star cutout.
        psf_image : `~lsst.afw.image.ImageD` | `~lsst.afw.image.ImageF`
            The PSF model to fit.

        Returns
        -------
        fit_psf_results : `dict`[`str`, `float`]
            The result of the PSF fitting, with keys:

            ``scale`` : `float`
                The scale factor.
            ``scale_err`` : `float`
                The error on the scale factor.
            ``pedestal`` : `float`
                The pedestal value.
            ``pedestal_err`` : `float`
                The error on the pedestal value.
            ``pedestal_scale_cov`` : `float`
                The covariance between the pedestal and scale factor.
            ``x_gradient`` : `float`
                The gradient in the x-direction.
            ``y_gradient`` : `float`
                The gradient in the y-direction.
            ``global_reduced_chi_squared`` : `float`
                The global reduced chi-squared goodness-of-fit.
            ``global_degrees_of_freedom`` : `int`
                The global number of degrees of freedom.
            ``psf_reduced_chi_squared`` : `float`
                The PSF BBox reduced chi-squared goodness-of-fit.
            ``psf_degrees_of_freedom`` : `int`
                The PSF BBox number of degrees of freedom.
            ``psf_masked_flux_frac`` : `float`
                The fraction of the PSF image flux masked by bad pixels.
        """
        bad_mask_bit_mask = stampMI.mask.getPlaneBitMask(self.config.bad_mask_planes)

        # Calculate the fraction of the PSF image flux masked by bad pixels
        psf_masked_pixels = ImageF(psf_image.getBBox())
        psf_masked_pixels.array[:, :] = (stampMI.mask[psf_image.getBBox()].array & bad_mask_bit_mask).astype(
            bool
        )
        psf_masked_flux_frac = (
            np.dot(psf_image.array.flat, psf_masked_pixels.array.flat).astype(np.float64)
            / psf_image.array.sum()
        )
        if psf_masked_flux_frac > self.config.psf_masked_flux_frac_threshold:
            return {}  # Handle cases where the PSF image is mostly masked

        # Generating good spans for gradient-pedestal fitting (including the star DETECTED mask).
        gradient_good_spans = self.generate_gradient_spans(stampMI, bad_mask_bit_mask)
        variance_data = gradient_good_spans.flatten(stampMI.variance.array, stampMI.getXY0())
        if self.config.use_median_variance:
            variance_data = np.median(variance_data)
        sigma_data = np.sqrt(variance_data)

        for i in range(self.config.fit_iterations):
            # Gradient-pedestal fitting:
            if i:
                # if i > 0, there should be scale factor from the previous fit iteration. Therefore, we can
                # remove the star using the scale factor.
                stamp = self.remove_star(stampMI, scale, padded_psf_image)  # noqa: F821
            else:
                stamp = deepcopy(stampMI.image.array)

            image_data_gr = gradient_good_spans.flatten(stamp, stampMI.getXY0()) / sigma_data  # B
            n_data = len(image_data_gr)

            xy = gradient_good_spans.indices()
            y = xy[0, :]
            x = xy[1, :]
            coefficient_matrix = np.ones((n_data, 6), dtype=float)  # A
            coefficient_matrix[:, 0] /= sigma_data
            coefficient_matrix[:, 1] = y / sigma_data
            coefficient_matrix[:, 2] = x / sigma_data
            coefficient_matrix[:, 3] = y ** 2 / sigma_data
            coefficient_matrix[:, 4] = x ** 2 / sigma_data
            coefficient_matrix[:, 5] = x * y / sigma_data
            # scikit might have a fitting tool

            try:
                gr_solutions, gr_sum_squared_residuals, *_ = np.linalg.lstsq(
                    coefficient_matrix, image_data_gr, rcond=None
                )
                covariance_matrix = np.linalg.inv(
                    np.dot(coefficient_matrix.transpose(), coefficient_matrix)
                )  # C
            except np.linalg.LinAlgError:
                return {}  # Handle singular matrix errors
            if gr_sum_squared_residuals.size == 0:
                return {}  # Handle cases where sum of the squared residuals are empty

            pedestal = gr_solutions[0]
            pedestal_err = np.sqrt(covariance_matrix[0, 0])
            pedestal_scale_cov = None
            x_gradient = gr_solutions[2]
            y_gradient = gr_solutions[1]
            x_curvature = gr_solutions[4]
            y_curvature = gr_solutions[3]
            cross_tilt = gr_solutions[5]

            # Scale fitting:
            updatedStampMI = deepcopy(stampMI)
            self._removePedestalAndGradient(
                updatedStampMI, pedestal, x_gradient, y_gradient, x_curvature, y_curvature, cross_tilt
            )

            # Create a padded version of the input constant PSF image
            padded_psf_image = ImageF(updatedStampMI.getBBox())
            padded_psf_image[psf_image.getBBox()] = psf_image.convertF()

            # Generating a mask plane while considering bad pixels in the psf model.
            mask = self.add_psf_mask(padded_psf_image, updatedStampMI)
            # Create consistently masked data
            scale_good_spans = self.generate_good_spans(mask, updatedStampMI.getBBox(), bad_mask_bit_mask)

            variance_data_scale = scale_good_spans.flatten(stampMI.variance.array, stampMI.getXY0())
            if self.config.use_median_variance:
                variance_data_scale = np.median(variance_data_scale)
            sigma_data_scale = np.sqrt(variance_data_scale)

            image_data = scale_good_spans.flatten(updatedStampMI.image.array, updatedStampMI.getXY0())
            psf_data = scale_good_spans.flatten(padded_psf_image.array, padded_psf_image.getXY0())

            image_data /= sigma_data_scale
            psf_data /= sigma_data_scale
            scale_coefficient_matrix = psf_data.reshape(psf_data.shape[0], 1)
            try:
                scale_solution, scale_sum_squared_residuals, *_ = np.linalg.lstsq(
                    scale_coefficient_matrix, image_data, rcond=None
                )
            except np.linalg.LinAlgError:
                return {}  # Handle singular matrix errors
            if scale_sum_squared_residuals.size == 0:
                return {}  # Handle cases where sum of the squared residuals are empty
            scale = scale_solution[0]
            if scale <= 0:
                return {}  # Handle cases where the PSF scale fit has failed

        scale *= self.model_scale  # ####### model scale correction ########
        n_data = len(image_data)

        scale_covariance_matrix = np.linalg.inv(
            np.dot(scale_coefficient_matrix.transpose(), scale_coefficient_matrix)
        )  # C
        scale_err = scale_covariance_matrix[0].astype(float)[0]

        # Calculate global (whole image) reduced chi-squared (scale fit is assumed as the main fitting
        # process here.)
        global_chi_squared = np.sum(scale_sum_squared_residuals)
        global_degrees_of_freedom = n_data - 1
        global_reduced_chi_squared = np.float64(global_chi_squared / global_degrees_of_freedom)

        # Calculate PSF BBox reduced chi-squared
        psf_bbox_scale_good_spans = scale_good_spans.clippedTo(psf_image.getBBox())
        psf_bbox_scale_good_spans_x, psf_bbox_scale_good_spans_y = psf_bbox_scale_good_spans.indices()
        psf_bbox_data = psf_bbox_scale_good_spans.flatten(stampMI.image.array, stampMI.getXY0())
        padded_psf_image.array /= self.model_scale  # ####### model scale correction ########
        psf_bbox_model = (
            psf_bbox_scale_good_spans.flatten(padded_psf_image.array, stampMI.getXY0()) * scale
            + pedestal
            + psf_bbox_scale_good_spans_x * x_gradient
            + psf_bbox_scale_good_spans_y * y_gradient
        )
        psf_bbox_residuals = (psf_bbox_data - psf_bbox_model) ** 2  # / psfBBoxVariance
        psf_bbox_chi_squared = np.sum(psf_bbox_residuals)
        psf_bbox_degrees_of_freedom = len(psf_bbox_data) - 1
        psf_bbox_reduced_chi_squared = psf_bbox_chi_squared / psf_bbox_degrees_of_freedom

        return dict(
            scale=scale,
            scale_err=scale_err,
            pedestal=pedestal,
            pedestal_err=pedestal_err,
            x_gradient=x_gradient,
            y_gradient=y_gradient,
            curvature_x=x_curvature,
            curvature_y=y_curvature,
            cross_tilt=cross_tilt,
            pedestal_scale_cov=pedestal_scale_cov,
            global_reduced_chi_squared=global_reduced_chi_squared,
            global_degrees_of_freedom=global_degrees_of_freedom,
            psf_reduced_chi_squared=psf_bbox_reduced_chi_squared,
            psf_degrees_of_freedom=psf_bbox_degrees_of_freedom,
            psf_masked_flux_frac=psf_masked_flux_frac,
        )

    def add_psf_mask(self, psf_image, stampMI, maskZeros=True):
        """
        Creates a new mask by adding PSF bad pixels to an existing stamp mask.

        This method identifies "bad" pixels in the PSF image (NaNs and
        optionally zeros/non-positives) and adds them to a deep copy
        of the input stamp's mask.

        Args:
            psf_image : `~lsst.afw.image.ImageD` | `~lsst.afw.image.ImageF`
                The PSF image object.
            stampMI: `~lsst.afw.image.MaskedImageF`
                The masked image of the bright star cutout.
            maskZeros (bool, optional): If True (default), mask pixels
                where the PSF is <= 0. If False, only mask pixels < 0.

        Returns:
            Any: A new mask object (deep copy) with the PSF mask planes added.
        """
        cond = np.isnan(psf_image.array)
        if maskZeros:
            cond |= psf_image.array <= 0
        else:
            cond |= psf_image.array < 0
        mask = deepcopy(stampMI.mask)
        mask.array[cond] = np.bitwise_or(mask.array[cond], 1)
        return mask

    def _removePedestalAndGradient(
        self, stampMI, pedestal, x_gradient, y_gradient, x_curvature, y_curvature, cross_tilt
    ):
        """Apply fitted pedestal and gradients to a single bright star stamp."""
        stamp_bbox = stampMI.getBBox()
        x_grid, y_grid = np.meshgrid(stamp_bbox.getX().arange(), stamp_bbox.getY().arange())
        x_plane = ImageF((x_grid * x_gradient).astype(np.float32), xy0=stampMI.getXY0())
        y_plane = ImageF((y_grid * y_gradient).astype(np.float32), xy0=stampMI.getXY0())
        x_curve = ImageF((x_grid ** 2 * x_curvature).astype(np.float32), xy0=stampMI.getXY0())
        y_curve = ImageF((y_grid ** 2 * y_curvature).astype(np.float32), xy0=stampMI.getXY0())
        cross_tilt = ImageF((x_grid * y_grid * cross_tilt).astype(np.float32), xy0=stampMI.getXY0())
        stampMI -= pedestal
        stampMI -= x_plane
        stampMI -= y_plane
        stampMI -= x_curve
        stampMI -= y_curve
        stampMI -= cross_tilt

    def remove_star(self, stampMI, scale, psf_image):
        """
        Subtracts a scaled PSF model from a star image.

        This performs a simple subtraction: `image - (psf * scale)`.

        Args:
            stampMI: `~lsst.afw.image.MaskedImageF`
                The masked image of the bright star cutout.
            scale (float): The scaling factor to apply to the PSF.
            psf_image: `~lsst.afw.image.ImageD` | `~lsst.afw.image.ImageF`
                The PSF image object.

        Returns:
            np.ndarray: A new 2D numpy array containing the star-subtracted
            image.
        """
        star_removed_cutout = stampMI.image.array - psf_image.array * scale
        return star_removed_cutout

    def estimate_model_scale_value(self, stampMI, psf_image):
        """
        Computes the scaling factor of the given model against a star.

        Args:
            stampMI : `~lsst.afw.image.MaskedImageF`
                The masked image of the bright star cutout.
            psf_image : `~lsst.afw.image.ImageD` | `~lsst.afw.image.ImageF`
                The given PSF model.
        """
        cond = stampMI.mask.array == 0
        self.star_median = np.median(stampMI.image.array[cond]).astype(np.float64)

        psf_positives = psf_image.array > 0

        image_array = stampMI.image.array - self.star_median
        image_array_positives = image_array > 0
        self.model_scale = np.nanmean(image_array[image_array_positives]) / np.nanmean(
            psf_image.array[psf_positives]
        )

    def generate_gradient_spans(self, stampMI, bad_mask_bit_mask):
        """
        Generates spans of "good" pixels for gradient fitting.

        This method creates a combined bitmask by OR-ing the provided
        `bad_mask_bit_mask` with the "DETECTED" plane from the stamp's mask.
        It then calls `self.generate_good_spans` to find all pixel spans
        not covered by this combined mask.

        Args:
            stampMI: `~lsst.afw.image.MaskedImageF`
                The masked image of the bright star cutout.
            bad_mask_bit_mask (int): A bitmask representing planes to be
                considered "bad" for gradient fitting.

        Returns:
            gradient_good_spans: A SpanSet object containing the "good" spans.
        """
        bit_mask_detected = stampMI.mask.getPlaneBitMask("DETECTED")
        gradient_bit_mask = np.bitwise_or(bad_mask_bit_mask, bit_mask_detected)

        gradient_good_spans = self.generate_good_spans(stampMI.mask, stampMI.getBBox(), gradient_bit_mask)
        return gradient_good_spans

    def generate_good_spans(self, mask, bBox, bad_bit_mask):
        """
        Generates a SpanSet of "good" pixels from a mask.

        This method identifies all spans within a given bounding box (`bBox`)
        that are *not* flagged by the `bad_bit_mask` in the provided `mask`.

        Args:
            mask (lsst.afw.image.MaskedImageF.mask): The mask object (e.g., `stampMI.mask`).
            bBox (lsst.geom.Box2I): The bounding box of the image (e.g., `stampMI.getBBox()`).
            bad_bit_mask (int): The combined bitmask of planes to exclude.

        Returns:
            good_spans: A SpanSet object representing all "good" spans.
        """
        bad_spans = SpanSet.fromMask(mask, bad_bit_mask)
        good_spans = SpanSet(bBox).intersectNot(bad_spans)
        return good_spans
