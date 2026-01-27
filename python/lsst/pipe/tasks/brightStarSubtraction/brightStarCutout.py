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

import math
from copy import deepcopy
from typing import Any, Iterable, cast

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE, PIXELS, Detector
from lsst.afw.detection import Footprint, FootprintSet, Threshold, footprintsToNumpy
from lsst.afw.geom import SkyWcs, SpanSet, makeModifiedWcs
from lsst.afw.geom.transformFactory import makeTransform
from lsst.afw.image import ExposureF, ImageD, ImageF, MaskedImageF
from lsst.afw.math import BackgroundList, FixedKernel, WarpingControl, warpImage
from lsst.afw.table import SourceCatalog
from lsst.daf.base import PropertyList
from lsst.daf.butler import DataCoordinate
from lsst.geom import (
    AffineTransform,
    Angle,
    Box2I,
    Extent2D,
    Extent2I,
    Point2D,
    Point2I,
    SpherePoint,
    arcseconds,
    floor,
    radians,
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

NEIGHBOR_MASK_PLANE = "NEIGHBOR"


class BrightStarCutoutConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),
):
    """Connections for BrightStarCutoutTask."""

    ref_cat = PrerequisiteInput(
        name="gaia_dr3_20230707",
        storageClass="SimpleCatalog",
        doc="Reference catalog that contains bright star positions.",
        dimensions=("skypix",),
        multiple=True,
        deferLoad=True,
    )
    input_exposure = Input(
        name="visit_image",
        storageClass="ExposureF",
        doc="Background-subtracted input exposure from which to extract bright star stamp cutouts.",
        dimensions=("visit", "detector"),
    )
    input_background = Input(
        name="visit_image_background",
        storageClass="Background",
        doc="Background model for the input exposure, to be added back on during processing.",
        dimensions=("visit", "detector"),
    )
    input_source_catalog = Input(
        name="source_footprints",
        storageClass="SourceCatalog",
        doc="Footprints",
        dimensions=("visit", "detector"),
    )
    extended_psf = Input(
        name="extended_psf",
        storageClass="MaskedImageF",
        doc="Extended PSF model, built from stacking bright star cutouts.",
        dimensions=("band",),
    )
    bright_star_stamps = Output(
        name="bright_star_stamps",
        storageClass="BrightStarStamps",
        doc="Set of preprocessed postage stamp cutouts, each centered on a single bright star.",
        dimensions=("visit", "detector"),
    )

    def __init__(self, *, config: "BrightStarCutoutConfig | None" = None):
        super().__init__(config=config)
        assert config is not None
        if not config.use_extended_psf:
            self.inputs.remove("extended_psf")


class BrightStarCutoutConfig(
    PipelineTaskConfig,
    pipelineConnections=BrightStarCutoutConnections,
):
    """Configuration parameters for BrightStarCutoutTask."""

    # Star selection
    mag_range = ListField[float](
        doc="Magnitude range in Gaia G. Cutouts will be made for all stars in this range.",
        default=[10, 18],
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
    min_focal_plane_radius = Field[float](
        doc="Minimum distance to focal plane center in mm. Stars with a focal plane radius smaller than "
        "this will be omitted.",
        default=-1.0,
    )
    max_focal_plane_radius = Field[float](
        doc="Maximum distance to focal plane center in mm. Stars with a focal plane radius greater than "
        "this will be omitted.",
        default=2000.0,
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
        stamp_size_padded = stamp_size * self.config.stamp_size_padding
        self.stamp_radius_padded = floor(stamp_size_padded / 2)
        self.stamp_bbox_padded = Box2I(corner=Point2I(0, 0), dimensions=Extent2I(1, 1)).dilatedBy(
            self.stamp_radius_padded
        )
        self.model_scale = 1

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        inputs["data_id"] = butlerQC.quantum.dataId
        ref_obj_loader = ReferenceObjectLoader(
            dataIds=[ref.datasetRef.dataId for ref in inputRefs.ref_cat],
            refCats=inputs.pop("ref_cat"),
            name=self.config.connections.ref_cat,
            config=self.config.load_reference_objects_config,
        )
        extended_psf = inputs.pop("extended_psf", None)
        output = self.run(**inputs, extended_psf=extended_psf, ref_obj_loader=ref_obj_loader)
        # Only ingest Stamp if it exists; prevents ingesting an empty FITS file
        if output:
            butlerQC.put(output, outputRefs)

    @timeMethod
    def run(
        self,
        input_exposure: ExposureF,
        input_background: BackgroundList,
        input_source_catalog: SourceCatalog,
        extended_psf: ImageF | None,
        ref_obj_loader: ReferenceObjectLoader,
        data_id: dict[str, Any] | DataCoordinate,
    ):
        """Identify bright stars within an exposure using a reference catalog,
        extract stamps around each, warp/shift stamps onto a common frame and
        then optionally fit a PSF plus plane model.

        Parameters
        ----------
        input_exposure : `~lsst.afw.image.ExposureF`
            The background-subtracted image to extract bright star stamps.
        input_background : `~lsst.afw.math.BackgroundList`
            The background model associated with the input exposure.
        input_source_catalog : `~lsst.afw.table.SourceCatalog`
            The source catalog containing footprints on the input exposure.
        extended_psf: `~lsst.afw.image.ImageF`, optional
            The extended PSF model from previous iteration(s).
        ref_obj_loader : `~lsst.meas.algorithms.ReferenceObjectLoader`
            Loader to find objects within a reference catalog.
        data_id : `dict` or `~lsst.daf.butler.DataCoordinate`
            The data_id of the exposure that bright stars are extracted from.
            Both 'visit' and 'detector' will be persisted in the output data.

        Returns
        -------
        brightStarResults : `~lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``bright_star_stamps``
                (`~lsst.meas.algorithms.brightStarStamps.BrightStarStamps`)
        """
        wcs = input_exposure.getWcs()
        bbox = input_exposure.getBBox()
        warp_control = WarpingControl(self.config.warping_kernel_name, self.config.mask_warping_kernel_name)
        detector = input_exposure.detector
        input_MI = input_exposure.getMaskedImage()
        input_MI += input_background.getImage()
        # TODO: If we eventually have better PhotoCalibs (eg FGCM), apply here
        input_MI = input_exposure.getPhotoCalib().calibrateImage(input_MI, False)

        bright_stars = self._get_bright_stars(ref_obj_loader, wcs, bbox, detector)

        input_MI.mask.addMaskPlane(NEIGHBOR_MASK_PLANE)
        footprints = footprintsToNumpy(input_source_catalog, input_MI.getBBox(), asBool=False)

        pixel_scale = wcs.getPixelScale().asArcseconds() * arcseconds
        pixels_to_boresight_pseudopixels = detector.getTransform(PIXELS, FIELD_ANGLE).then(
            makeTransform(AffineTransform.makeScaling(1 / pixel_scale.asRadians()))
        )

        stamps = []  # , stamps_fit_psf_results = [], [], []
        for bright_star in bright_stars:
            pix_coord = Point2D(bright_star["pixel_x"], bright_star["pixel_y"])

            # Set mask NEIGHBOR plane for all sources except the current one
            neighbor_bit_mask = input_MI.mask.getPlaneBitMask(NEIGHBOR_MASK_PLANE)
            input_MI.mask.clearMaskPlane(int(np.log2(neighbor_bit_mask)))
            src_id = footprints[int(pix_coord.y), int(pix_coord.x)]
            neighbor_mask = (footprints != 0) & (footprints != src_id)
            input_MI.mask.array[neighbor_mask] |= neighbor_bit_mask

            stamp_MI = MaskedImageF(self.stamp_bbox_padded)

            # Define linear shifting and rotation to recenter and align stamps
            boresight_pseudopixel_coord = pixels_to_boresight_pseudopixels.applyForward(pix_coord)
            shift = makeTransform(AffineTransform(Point2D(0, 0) - boresight_pseudopixel_coord))
            rotation = makeTransform(AffineTransform.makeRotation(-bright_star["theta_radians"] * radians))
            pixels_to_stamp_frame = pixels_to_boresight_pseudopixels.then(shift).then(rotation)

            # Apply the warp to the star stamp (in-place) and trim to BBox
            warpImage(stamp_MI, input_MI, pixels_to_stamp_frame, warp_control)
            stamp_MI = stamp_MI[self.stamp_bbox]

            # Check mask coverage, update metadata
            bad_bit_mask = stamp_MI.mask.getPlaneBitMask(self.config.bad_mask_planes)
            good_frac = np.sum(stamp_MI.mask.array & bad_bit_mask == 0) / stamp_MI.mask.array.size
            if good_frac < self.config.min_area_fraction:
                continue

            warped_psf = WarpedPsf(input_exposure.getPsf(), pixels_to_stamp_frame, warp_control)
            stamp_psf = KernelPsf(FixedKernel(warped_psf.computeKernelImage(Point2D(0, 0))))
            stamp_wcs = makeModifiedWcs(pixels_to_stamp_frame, wcs, False)

            # # Fit a scaled PSF and a pedestal to each bright star cutout
            # psf = WarpedPsf(input_exposure.getPsf(), pixels_to_stamp_frame, warp_control)
            # stamp_psf = KernelPsf(FixedKernel(warped_psf.computeKernelImage(Point2D(0, 0))))
            # if self.config.use_extended_psf:
            #     psf_image = deepcopy(extended_psf.image)  # Assumed to be warped, center at [0,0]
            # else:
            #     psf_image = stamp_psf.computeKernelImage(stamp_psf.getAveragePosition())
            #     # TODO: maybe we want to generate a smaller psf in case the following happens?
            #     # The following could happen for when the user chooses small stamp_size ~(50, 50)
            #     if (
            #         psf_image.array.shape[0] > stamp_MI.image.array.shape[0]
            #         or psf_image.array.shape[1] > stamp_MI.image.array.shape[1]
            #     ):
            #         continue
            # # Computing an scale factor that brings the model to the similar level of the star.
            # self.estimate_model_scale_value(stamp_MI, psf_image)
            # psf_image.array *= self.model_scale  # ####### model scale correction ########

            # fit_psf_results = {}

            # if self.config.do_fit_psf:
            #     fit_psf_results = self._fit_psf(stamp_MI, psf_image)
            # stamps_fit_psf_results.append(fit_psf_results)

            stamp = BrightStarStamp(
                stamp_im=stamp_MI,
                psf=stamp_psf,
                wcs=stamp_wcs,
                visit=cast(int, data_id["visit"]),
                detector=cast(int, data_id["detector"]),
                ref_id=bright_star["id"],
                ref_mag=bright_star["mag"],
                position=pix_coord,
                focal_plane_radius=bright_star["radius_mm"],
                focal_plane_angle=Angle(bright_star["theta_radians"], radians),
                scale=None,
                scale_err=None,
                pedestal=None,
                pedestal_err=None,
                pedestal_scale_cov=None,
                gradient_x=None,
                gradient_y=None,
                curvature_x=None,
                curvature_y=None,
                curvature_xy=None,
                global_reduced_chi_squared=None,
                global_degrees_of_freedom=None,
                psf_reduced_chi_squared=None,
                psf_degrees_of_freedom=None,
                psf_masked_flux_fraction=None,
            )
            stamps.append(stamp)

            # # Save the stamp if the PSF fit was successful or no fit requested
            # if fit_psf_results or not self.config.do_fit_psf:
            #     stamp = BrightStarStamp(
            #         stamp_im=stamp_MI,
            #         psf=stamp_psf,
            #         wcs=makeModifiedWcs(pixels_to_stamp_frame, wcs, False),
            #         visit=cast(int, data_id["visit"]),
            #         detector=cast(int, data_id["detector"]),
            #         ref_id=obj["id"],
            #         ref_mag=obj["mag"],
            #         position=pix_coord,
            #         focal_plane_radius=distance_mm,
            #         focal_plane_angle=theta_angle,  # TODO: add the lsst.geom.Angle here
            #         scale=fit_psf_results.get("scale", None),
            #         scale_err=fit_psf_results.get("scale_err", None),
            #         pedestal=fit_psf_results.get("pedestal", None),
            #         pedestal_err=fit_psf_results.get("pedestal_err", None),
            #         pedestal_scale_cov=fit_psf_results.get("pedestal_scale_cov", None),
            #         gradient_x=fit_psf_results.get("x_gradient", None),
            #         gradient_y=fit_psf_results.get("y_gradient", None),
            #         curvature_x=fit_psf_results.get("curvature_x", None),
            #         curvature_y=fit_psf_results.get("curvature_y", None),
            #         curvature_xy=fit_psf_results.get("curvature_xy", None),
            #         global_reduced_chi_squared=fit_psf_results.get("global_reduced_chi_squared", None),
            #         global_degrees_of_freedom=fit_psf_results.get("global_degrees_of_freedom", None),
            #         psf_reduced_chi_squared=fit_psf_results.get("psf_reduced_chi_squared", None),
            #         psf_degrees_of_freedom=fit_psf_results.get("psf_degrees_of_freedom", None),
            #         psf_masked_flux_fraction=fit_psf_results.get("psf_masked_flux_frac", None),
            #     )
            #     stamps.append(stamp)

        num_excluded = len(bright_stars) - len(stamps)
        percent_excluded = 100.0 * num_excluded / len(bright_stars) if bright_stars else 0.0
        self.log.info(
            "Extracted %i bright star stamp%s. Excluded %i star%s (%.1f%%) due to masked area fraction > %s.",
            len(stamps),
            "" if len(stamps) == 1 else "s",
            num_excluded,
            "" if num_excluded == 1 else "s",
            percent_excluded,
            self.config.min_area_fraction,
        )

        focal_plane_radii = [stamp.focal_plane_radius for stamp in stamps]
        metadata = PropertyList()
        metadata["FOCAL_PLANE_RADIUS_MIN"] = np.min(focal_plane_radii)
        metadata["FOCAL_PLANE_RADIUS_MAX"] = np.max(focal_plane_radii)
        bright_star_stamps = BrightStarStamps(stamps, metadata=metadata)
        return Struct(bright_star_stamps=bright_star_stamps)

    def _get_bright_stars(
        self,
        ref_obj_loader: ReferenceObjectLoader,
        wcs: SkyWcs,
        bbox: Box2I,
        detector: Detector,
    ) -> Table:
        """Get a table of bright stars from the reference catalog.

        Trim the reference catalog to only those objects within the exposure
        bounding box dilated by half the bright star stamp size.
        This ensures all stars that overlap the exposure are included.
        Additional filtering is applied to select only isolated stars
        within the specified magnitude range.
        Value-added columns are also computed, including pixel coordinates
        and focal plane radial coordinates.

        Parameters
        ----------
        ref_obj_loader : `~lsst.meas.algorithms.ReferenceObjectLoader`
            Loader to find objects within a reference catalog.
        wcs : `~lsst.afw.geom.SkyWcs`
            World coordinate system.
        bbox : `~lsst.geom.Box2I`
            Bounding box of the exposure.
        detector : `~lsst.afw.cameraGeom.Detector`
            The detector object, used to transform from PIXELS to FOCAL_PLANE.

        Returns
        -------
        bright_stars : `~astropy.table.Table`
            Table of bright stars within the exposure.
        """
        within_region = ref_obj_loader.loadPixelBox(bbox, wcs, "phot_g_mean")
        ref_cat_full = within_region.refCat
        flux_field: str = within_region.fluxField
        exclude_arcsec_radius = self.config.exclude_arcsec_radius * u.arcsec

        flux_range_neighbor = sorted(((self.config.exclude_mag_range * u.ABmag).to(u.nJy)).to_value())
        flux_range_candidate = sorted(((self.config.mag_range * u.ABmag).to(u.nJy)).to_value())

        flux_min = np.min((flux_range_neighbor[0], flux_range_candidate[0]))
        flux_max = np.max((flux_range_neighbor[1], flux_range_candidate[1]))
        stars_subset = (ref_cat_full[flux_field] > flux_min) & (ref_cat_full[flux_field] < flux_max)
        ref_cat_subset_columns = ("id", "coord_ra", "coord_dec", flux_field)
        ref_cat_subset = Table(ref_cat_full.extract(*ref_cat_subset_columns, where=stars_subset))
        flux_subset = ref_cat_subset[flux_field]

        is_neighbor = (flux_subset >= flux_range_neighbor[0]) & (flux_subset <= flux_range_neighbor[1])
        is_candidate = (flux_subset >= flux_range_candidate[0]) & (flux_subset <= flux_range_candidate[1])

        coords = SkyCoord(ref_cat_subset["coord_ra"], ref_cat_subset["coord_dec"], unit="rad")
        coords_neighbor = coords[is_neighbor]
        coords_candidate = coords[is_candidate]

        if len(coords_neighbor) == 0:
            is_candidate_isolated = np.ones(len(coords_candidate), dtype=bool)
        else:
            seps = coords_candidate[:, None].separation(coords_neighbor[None, :]).to(u.arcsec)
            too_close = (seps > 0) & (seps <= exclude_arcsec_radius)  # not self matched
            is_candidate_isolated = ~too_close.any(axis=1)

        bright_stars = ref_cat_subset[is_candidate][is_candidate_isolated]

        flux_nanojansky = bright_stars[flux_field][:] * u.nJy
        bright_stars["mag"] = flux_nanojansky.to(u.ABmag).to_value()  # AB magnitudes

        zip_ra_dec = zip(bright_stars["coord_ra"] * radians, bright_stars["coord_dec"] * radians)
        sphere_points = [SpherePoint(ra, dec) for ra, dec in zip_ra_dec]
        pixel_coords = wcs.skyToPixel(sphere_points)
        bright_stars["pixel_x"] = [pixel_coord.x for pixel_coord in pixel_coords]
        bright_stars["pixel_y"] = [pixel_coord.y for pixel_coord in pixel_coords]

        mm_coords = detector.transform(pixel_coords, PIXELS, FOCAL_PLANE)
        mm_coords_x = np.array([mm_coord.x for mm_coord in mm_coords])
        mm_coords_y = np.array([mm_coord.y for mm_coord in mm_coords])
        radius_mm = np.sqrt(mm_coords_x**2 + mm_coords_y**2)
        theta_radians = np.arctan2(mm_coords_y, mm_coords_x)
        bright_stars["radius_mm"] = radius_mm
        bright_stars["theta_radians"] = theta_radians

        within_bbox = bright_stars["pixel_x"] >= bbox.getMinX()
        within_bbox &= bright_stars["pixel_x"] <= bbox.getMaxX()
        within_bbox &= bright_stars["pixel_y"] >= bbox.getMinY()
        within_bbox &= bright_stars["pixel_y"] <= bbox.getMaxY()

        within_radii = bright_stars["radius_mm"] >= self.config.min_focal_plane_radius
        within_radii &= bright_stars["radius_mm"] <= self.config.max_focal_plane_radius

        bright_stars = bright_stars[within_bbox & within_radii]

        self.log.info(
            "Identified %i of %i reference catalog star%s that are in the field of view, are in the range %s "
            "mag, and that have no neighboring stars within %s arcsec.",
            len(bright_stars),
            len(ref_cat_full),
            "" if len(ref_cat_full) == 1 else "s",
            self.config.mag_range,
            self.config.exclude_arcsec_radius,
        )

        return bright_stars

    def _fit_psf(self, stamp_MI: MaskedImageF, psf_image: ImageD | ImageF) -> dict[str, Any]:
        """Fit a scaled PSF and a pedestal to each bright star cutout.

        Parameters
        ----------
        stamp_MI : `~lsst.afw.image.MaskedImageF`
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
        bad_bit_mask = stamp_MI.mask.getPlaneBitMask(self.config.bad_mask_planes)

        # Calculate the fraction of the PSF image flux masked by bad pixels
        psf_masked_pixels = ImageF(psf_image.getBBox())
        psf_masked_pixels.array[:, :] = (stamp_MI.mask[psf_image.getBBox()].array & bad_bit_mask).astype(bool)
        psf_masked_flux_frac = (
            np.dot(psf_image.array.flat, psf_masked_pixels.array.flat).astype(np.float64)
            / psf_image.array.sum()
        )
        if psf_masked_flux_frac > self.config.psf_masked_flux_frac_threshold:
            return {}  # Handle cases where the PSF image is mostly masked

        # Generating good spans for gradient-pedestal fitting (including the star DETECTED mask).
        gradient_good_spans = self.generate_gradient_spans(stamp_MI, bad_bit_mask)
        variance_data = gradient_good_spans.flatten(stamp_MI.variance.array, stamp_MI.getXY0())
        if self.config.use_median_variance:
            variance_data = np.median(variance_data)
        sigma_data = np.sqrt(variance_data)

        for i in range(self.config.fit_iterations):
            # Gradient-pedestal fitting:
            if i:
                # if i > 0, there should be scale factor from the previous fit iteration. Therefore, we can
                # remove the star using the scale factor.
                stamp = self.remove_star(stamp_MI, scale, padded_psf_image)  # noqa: F821
            else:
                stamp = deepcopy(stamp_MI.image.array)

            image_data_gr = gradient_good_spans.flatten(stamp, stamp_MI.getXY0()) / sigma_data  # B
            n_data = len(image_data_gr)

            xy = gradient_good_spans.indices()
            y = xy[0, :]
            x = xy[1, :]
            coefficient_matrix = np.ones((n_data, 6), dtype=float)  # A
            coefficient_matrix[:, 0] /= sigma_data
            coefficient_matrix[:, 1] = y / sigma_data
            coefficient_matrix[:, 2] = x / sigma_data
            coefficient_matrix[:, 3] = y**2 / sigma_data
            coefficient_matrix[:, 4] = x**2 / sigma_data
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
            curvature_xy = gr_solutions[5]

            # Scale fitting:
            updatedStampMI = deepcopy(stamp_MI)
            self._removePedestalAndGradient(
                updatedStampMI, pedestal, x_gradient, y_gradient, x_curvature, y_curvature, curvature_xy
            )

            # Create a padded version of the input constant PSF image
            padded_psf_image = ImageF(updatedStampMI.getBBox())
            padded_psf_image[psf_image.getBBox()] = psf_image.convertF()

            # Generating a mask plane while considering bad pixels in the psf model.
            mask = self.add_psf_mask(padded_psf_image, updatedStampMI)
            # Create consistently masked data
            scale_good_spans = self.generate_good_spans(mask, updatedStampMI.getBBox(), bad_bit_mask)

            variance_data_scale = scale_good_spans.flatten(stamp_MI.variance.array, stamp_MI.getXY0())
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
        psf_bbox_data = psf_bbox_scale_good_spans.flatten(stamp_MI.image.array, stamp_MI.getXY0())
        padded_psf_image.array /= self.model_scale  # ####### model scale correction ########
        psf_bbox_model = (
            psf_bbox_scale_good_spans.flatten(padded_psf_image.array, stamp_MI.getXY0()) * scale
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
            curvature_xy=curvature_xy,
            pedestal_scale_cov=pedestal_scale_cov,
            global_reduced_chi_squared=global_reduced_chi_squared,
            global_degrees_of_freedom=global_degrees_of_freedom,
            psf_reduced_chi_squared=psf_bbox_reduced_chi_squared,
            psf_degrees_of_freedom=psf_bbox_degrees_of_freedom,
            psf_masked_flux_frac=psf_masked_flux_frac,
        )

    def add_psf_mask(self, psf_image, stamp_MI, maskZeros=True):
        """
        Creates a new mask by adding PSF bad pixels to an existing stamp mask.

        This method identifies "bad" pixels in the PSF image (NaNs and
        optionally zeros/non-positives) and adds them to a deep copy
        of the input stamp's mask.

        Args:
            psf_image : `~lsst.afw.image.ImageD` | `~lsst.afw.image.ImageF`
                The PSF image object.
            stamp_MI: `~lsst.afw.image.MaskedImageF`
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
        mask = deepcopy(stamp_MI.mask)
        mask.array[cond] = np.bitwise_or(mask.array[cond], 1)
        return mask

    def _removePedestalAndGradient(
        self, stamp_MI, pedestal, x_gradient, y_gradient, x_curvature, y_curvature, curvature_xy
    ):
        """Apply fitted pedestal and gradients to a single bright star stamp."""
        stamp_bbox = stamp_MI.getBBox()
        x_grid, y_grid = np.meshgrid(stamp_bbox.getX().arange(), stamp_bbox.getY().arange())
        x_plane = ImageF((x_grid * x_gradient).astype(np.float32), xy0=stamp_MI.getXY0())
        y_plane = ImageF((y_grid * y_gradient).astype(np.float32), xy0=stamp_MI.getXY0())
        x_curve = ImageF((x_grid**2 * x_curvature).astype(np.float32), xy0=stamp_MI.getXY0())
        y_curve = ImageF((y_grid**2 * y_curvature).astype(np.float32), xy0=stamp_MI.getXY0())
        curvature_xy = ImageF((x_grid * y_grid * curvature_xy).astype(np.float32), xy0=stamp_MI.getXY0())
        stamp_MI -= pedestal
        stamp_MI -= x_plane
        stamp_MI -= y_plane
        stamp_MI -= x_curve
        stamp_MI -= y_curve
        stamp_MI -= curvature_xy

    def remove_star(self, stamp_MI, scale, psf_image):
        """
        Subtracts a scaled PSF model from a star image.

        This performs a simple subtraction: `image - (psf * scale)`.

        Args:
            stamp_MI: `~lsst.afw.image.MaskedImageF`
                The masked image of the bright star cutout.
            scale (float): The scaling factor to apply to the PSF.
            psf_image: `~lsst.afw.image.ImageD` | `~lsst.afw.image.ImageF`
                The PSF image object.

        Returns:
            np.ndarray: A new 2D numpy array containing the star-subtracted
            image.
        """
        star_removed_cutout = stamp_MI.image.array - psf_image.array * scale
        return star_removed_cutout

    def estimate_model_scale_value(self, stamp_MI, psf_image):
        """
        Computes the scaling factor of the given model against a star.

        Args:
            stamp_MI : `~lsst.afw.image.MaskedImageF`
                The masked image of the bright star cutout.
            psf_image : `~lsst.afw.image.ImageD` | `~lsst.afw.image.ImageF`
                The given PSF model.
        """
        cond = stamp_MI.mask.array == 0
        self.star_median = np.median(stamp_MI.image.array[cond]).astype(np.float64)

        psf_positives = psf_image.array > 0

        image_array = stamp_MI.image.array - self.star_median
        image_array_positives = image_array > 0
        self.model_scale = np.nanmean(image_array[image_array_positives]) / np.nanmean(
            psf_image.array[psf_positives]
        )

    def generate_gradient_spans(self, stamp_MI, bad_bit_mask):
        """
        Generates spans of "good" pixels for gradient fitting.

        This method creates a combined bitmask by OR-ing the provided
        `bad_bit_mask` with the "DETECTED" plane from the stamp's mask.
        It then calls `self.generate_good_spans` to find all pixel spans
        not covered by this combined mask.

        Args:
            stamp_MI: `~lsst.afw.image.MaskedImageF`
                The masked image of the bright star cutout.
            bad_bit_mask (int): A bitmask representing planes to be
                considered "bad" for gradient fitting.

        Returns:
            gradient_good_spans: A SpanSet object containing the "good" spans.
        """
        bit_mask_detected = stamp_MI.mask.getPlaneBitMask("DETECTED")
        gradient_bit_mask = np.bitwise_or(bad_bit_mask, bit_mask_detected)

        gradient_good_spans = self.generate_good_spans(stamp_MI.mask, stamp_MI.getBBox(), gradient_bit_mask)
        return gradient_good_spans

    def generate_good_spans(self, mask, bBox, bad_bit_mask):
        """
        Generates a SpanSet of "good" pixels from a mask.

        This method identifies all spans within a given bounding box (`bBox`)
        that are *not* flagged by the `bad_bit_mask` in the provided `mask`.

        Args:
            mask (lsst.afw.image.MaskedImageF.mask): The mask object (e.g., `stamp_MI.mask`).
            bBox (lsst.geom.Box2I): The bounding box of the image (e.g., `stamp_MI.getBBox()`).
            bad_bit_mask (int): The combined bitmask of planes to exclude.

        Returns:
            good_spans: A SpanSet object representing all "good" spans.
        """
        bad_spans = SpanSet.fromMask(mask, bad_bit_mask)
        good_spans = SpanSet(bBox).intersectNot(bad_spans)
        return good_spans
