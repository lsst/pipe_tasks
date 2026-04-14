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

__all__ = ["BrightStarCutoutConnections", "BrightStarCutoutConfig", "BrightStarCutoutTask"]

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE, PIXELS
from lsst.afw.detection import footprintsToNumpy
from lsst.afw.geom import makeModifiedWcs
from lsst.afw.geom.transformFactory import makeTransform
from lsst.afw.image import ExposureF, MaskedImageF
from lsst.afw.math import BackgroundList, WarpingControl, warpImage
from lsst.afw.table import SourceCatalog
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
)
from lsst.images import GeneralFrame, Image, Mask, Projection
from lsst.meas.algorithms import (
    LoadReferenceObjectsConfig,
    ReferenceObjectLoader,
    WarpedPsf,
)
from lsst.pex.config import ChoiceField, ConfigField, Field, ListField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input, Output, PrerequisiteInput
from lsst.utils.timer import timeMethod

from .brightStarStamps import BrightStarStamp, BrightStarStampInfo, BrightStarStamps

NEIGHBOR_MASK_PLANE = "NEIGHBOR"


class BrightStarCutoutConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),
):
    """Connections for BrightStarCutoutTask."""

    ref_cat = PrerequisiteInput(
        name="the_monster_20250219",
        storageClass="SimpleCatalog",
        doc="Reference catalog that contains bright star positions.",
        dimensions=("skypix",),
        multiple=True,
        deferLoad=True,
    )
    input_exposure = Input(
        name="preliminary_visit_image",
        storageClass="ExposureF",
        doc="Background-subtracted input exposure from which to extract bright star stamp cutouts.",
        dimensions=("visit", "detector"),
    )
    input_background = Input(
        name="preliminary_visit_image_background",
        storageClass="Background",
        doc="Background model for the input exposure, to be added back on during processing.",
        dimensions=("visit", "detector"),
    )
    input_source_catalog = Input(
        name="single_visit_star_footprints",
        storageClass="SourceCatalog",
        doc="Source catalog containing footprints on the input exposure, used to mask neighboring sources.",
        dimensions=("visit", "detector"),
    )
    bright_star_stamps = Output(
        name="bright_star_stamps",
        storageClass="BrightStarStamps",
        doc="Set of preprocessed postage stamp cutouts, each centered on a single bright star.",
        dimensions=("visit", "detector"),
    )


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
        doc="No postage stamp will be generated for stars with a neighboring star in the range "
        "``exclude_mag_range`` mag within ``exclude_arcsec_radius`` arcseconds.",
        default=5,
    )
    exclude_mag_range = ListField[float](
        doc="No postage stamp will be generated for stars with a neighboring star in the range "
        "``exclude_mag_range`` mag within ``exclude_arcsec_radius`` arcseconds.",
        default=[0, 20],
    )
    min_area_fraction = Field[float](
        doc="Minimum fraction of the stamp area, post-masking, that must remain for a cutout to be retained.",
        default=0.1,
    )
    bad_mask_planes = ListField[str](
        doc="Mask planes that identify excluded pixels for the calculation of ``min_area_fraction``.",
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
    min_focal_plane_radius = Field[float](
        doc="Minimum distance to the center of the focal plane, in mm. "
        "Stars with a focal plane radius smaller than this will be omitted.",
        default=0.0,
    )
    max_focal_plane_radius = Field[float](
        doc="Maximum distance to the center of the focal plane, in mm. "
        "Stars with a focal plane radius larger than this will be omitted.",
        default=np.inf,
    )

    # Stamp geometry
    stamp_size = ListField[int](
        doc="Size of the stamps to be extracted, in pixels.",
        default=[251, 251],
    )
    warping_kernel_name = ChoiceField[str](
        doc="Warping kernel for image data warping.",
        default="lanczos5",
        allowed={
            "bilinear": "bilinear interpolation",
            "lanczos3": "Lanczos kernel of order 3",
            "lanczos4": "Lanczos kernel of order 4",
            "lanczos5": "Lanczos kernel of order 5",
        },
    )
    mask_warping_kernel_name = ChoiceField[str](
        doc="Warping kernel for mask warping. Typically a more conservative kernel (e.g. with less ringing) "
        "is desirable for warping masks than for warping image data.",
        default="bilinear",
        allowed={
            "bilinear": "bilinear interpolation",
            "lanczos3": "Lanczos kernel of order 3",
            "lanczos4": "Lanczos kernel of order 4",
            "lanczos5": "Lanczos kernel of order 5",
        },
    )

    # Misc
    load_reference_objects_config = ConfigField[LoadReferenceObjectsConfig](
        doc="Reference object loader for astrometric calibration.",
    )
    ref_cat_filter_name = Field[str](
        doc="Name of the filter in the reference catalog to use for star selection. ",
        default="phot_g_mean",
    )


class BrightStarCutoutTask(PipelineTask):
    """Extract bright star cutouts, and warp to the same pixel grid.

    The BrightStarCutoutTask is used to extract, process, and store small image
    cutouts (or "postage stamps") around bright stars.
    This task essentially consists of two principal steps.
    First, it identifies bright stars within an exposure using a reference
    catalog and extracts a stamp around each.
    Second, it shifts and warps each stamp to remove optical distortions and
    sample all stars on the same pixel grid.
    """

    ConfigClass = BrightStarCutoutConfig
    _DefaultName = "brightStarCutout"
    config: BrightStarCutoutConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        ref_obj_loader = ReferenceObjectLoader(
            dataIds=[ref.datasetRef.dataId for ref in inputRefs.ref_cat],
            refCats=inputs.pop("ref_cat"),
            name=self.config.connections.ref_cat,
            config=self.config.load_reference_objects_config,
        )
        output = self.run(**inputs, ref_obj_loader=ref_obj_loader)
        # Only ingest Stamp if it exists; prevents ingesting an empty FITS file
        if output:
            butlerQC.put(output, outputRefs)

    @timeMethod
    def run(
        self,
        input_exposure: ExposureF,
        input_background: BackgroundList,
        input_source_catalog: SourceCatalog,
        ref_obj_loader: ReferenceObjectLoader,
    ):
        """Identify bright stars within an exposure using a reference catalog,
        extract stamps around each and warp/shift stamps onto a common frame.

        Parameters
        ----------
        input_exposure : `~lsst.afw.image.ExposureF`
            The background-subtracted image to extract bright star stamps.
        input_background : `~lsst.afw.math.BackgroundList`
            The background model associated with the input exposure.
        input_source_catalog : `~lsst.afw.table.SourceCatalog`
            The source catalog containing footprints on the input exposure.
        ref_obj_loader : `~lsst.meas.algorithms.ReferenceObjectLoader`
            Loader to find objects within a reference catalog.

        Returns
        -------
        bright_star_stamps : `~lsst.meas.algorithms.BrightStarStamps`
            A set of postage stamp cutouts, each centered on a bright star.
        """
        bright_stars = self._get_bright_stars(ref_obj_loader, input_exposure)

        bright_star_stamps = self._get_bright_star_stamps(
            input_exposure,
            input_background,
            input_source_catalog,
            bright_stars,
        )

        return Struct(bright_star_stamps=bright_star_stamps)

    def _get_bright_stars(
        self,
        ref_obj_loader: ReferenceObjectLoader,
        input_exposure: ExposureF,
    ) -> Table:
        """Get a table of bright stars from the reference catalog.

        Trim the reference catalog to only those objects within the exposure
        bounding box.
        Then, select bright stars based on the specified magnitude range,
        isolation criteria, and optionally focal plane radius criteria.
        Finally, add columns with pixel coordinates and focal plane coordinates
        for each bright star.

        Parameters
        ----------
        ref_obj_loader : `~lsst.meas.algorithms.ReferenceObjectLoader`
            Loader to find objects within a reference catalog.
        input_exposure : `~lsst.afw.image.ExposureF`
            The exposure for which bright stars are being selected.

        Returns
        -------
        bright_stars : `~astropy.table.Table`
            Table of bright stars within the exposure.
        """
        bbox = input_exposure.getBBox()
        wcs = input_exposure.getWcs()
        detector = input_exposure.detector

        # Load all ref cat stars within the padded exposure bounding box
        within_region = ref_obj_loader.loadPixelBox(bbox, wcs, self.config.ref_cat_filter_name)
        ref_cat_full = within_region.refCat
        flux_field: str = within_region.fluxField
        exclude_arcsec_radius = self.config.exclude_arcsec_radius * u.arcsec

        # Convert mag ranges to flux in nJy for comparison with ref cat fluxes
        flux_range_candidate = sorted(((self.config.mag_range * u.ABmag).to(u.nJy)).to_value())
        flux_range_neighbor = sorted(((self.config.exclude_mag_range * u.ABmag).to(u.nJy)).to_value())

        # Create a subset of ref cat stars that includes all stars that could
        # potentially be either a candidate or a neighbor based on flux
        flux_min = np.min((flux_range_candidate[0], flux_range_neighbor[0]))
        flux_max = np.max((flux_range_candidate[1], flux_range_neighbor[1]))
        stars_subset = (ref_cat_full[flux_field] >= flux_min) & (ref_cat_full[flux_field] <= flux_max)
        ref_cat_subset_columns = ("id", "coord_ra", "coord_dec", flux_field)
        ref_cat_subset = Table(ref_cat_full.extract(*ref_cat_subset_columns, where=stars_subset))
        flux_subset = ref_cat_subset[flux_field]

        # Identify candidate bright stars and their neighbors based on flux
        is_candidate = (flux_subset >= flux_range_candidate[0]) & (flux_subset <= flux_range_candidate[1])
        is_neighbor = (flux_subset >= flux_range_neighbor[0]) & (flux_subset <= flux_range_neighbor[1])

        # Trim star coordinates to candidate and neighbor subsets
        coords = SkyCoord(ref_cat_subset["coord_ra"], ref_cat_subset["coord_dec"], unit="rad")
        coords_candidate = coords[is_candidate]
        coords_neighbor = coords[is_neighbor]

        # Identify candidate bright stars that have no contaminant neighbors
        is_candidate_isolated = np.ones(len(coords_candidate), dtype=bool)
        if len(coords_neighbor) > 0:
            _, indices_candidate, angular_separation, _ = coords_candidate.search_around_sky(
                coords_neighbor, exclude_arcsec_radius
            )
            indices_candidate = indices_candidate[angular_separation > 0 * u.arcsec]  # Exclude self-matches
            is_candidate_isolated[indices_candidate] = False

        # Trim ref cat subset to isolated bright stars; add ancillary data
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
        angle_radians = np.arctan2(mm_coords_y, mm_coords_x)
        bright_stars["radius_mm"] = radius_mm
        bright_stars["angle_radians"] = angle_radians

        # Trim bright star catalog to those within the exposure bounding box,
        # and optionally within a range of focal plane radii
        within_bbox = bright_stars["pixel_x"] >= bbox.getMinX()
        within_bbox &= bright_stars["pixel_x"] <= bbox.getMaxX()
        within_bbox &= bright_stars["pixel_y"] >= bbox.getMinY()
        within_bbox &= bright_stars["pixel_y"] <= bbox.getMaxY()
        within_radii = bright_stars["radius_mm"] >= self.config.min_focal_plane_radius
        within_radii &= bright_stars["radius_mm"] <= self.config.max_focal_plane_radius
        bright_stars = bright_stars[within_bbox & within_radii]

        self.log.info(
            "Identified %i reference star%s in the field of view after applying magnitude and isolation "
            "cuts.",
            len(bright_stars),
            "s" if len(bright_stars) != 1 else "",
        )

        return bright_stars

    def _get_bright_star_stamps(
        self,
        input_exposure: ExposureF,
        input_background: BackgroundList | None,
        footprints: SourceCatalog | np.ndarray,
        bright_stars: Table,
    ) -> BrightStarStamps | None:
        """Extract and warp bright star stamps.

        For each bright star, extract a stamp from the input exposure centered
        on the star's pixel coordinates.
        Then, shift and warp the stamp to recenter on the star and align each
        to the same orientation.
        Finally, check the fraction of the stamp area that is masked
        (e.g. due to neighboring sources or bad pixels), and only retain stamps
        with sufficient unmasked area.

        Parameters
        ----------
        input_exposure : `~lsst.afw.image.ExposureF`
            The science image to extract bright star stamps.
        input_background : `~lsst.afw.math.BackgroundList` | None
            The background model associated with the input exposure.
            If provided, this will be added back on to the input image.
        footprints : `~lsst.afw.table.SourceCatalog` | `numpy.ndarray`
            The source catalog containing footprints on the input exposure, or
            a 2D numpy array with the same dimensions as the input exposure
            where each pixel value corresponds to the source footprint ID.
        bright_stars : `~astropy.table.Table`
            Table of bright stars for which to extract stamps.

        Returns
        -------
        bright_star_stamps : `~lsst.meas.algorithms.BrightStarStamps` | None
            A set of postage stamp cutouts, each centered on a bright star.
            If no bright star stamps are retained post-masking, returns `None`.
        """
        warp_control = WarpingControl(self.config.warping_kernel_name, self.config.mask_warping_kernel_name)
        bbox = input_exposure.getBBox()

        # Prepare data: add bg back on, and convert to nJy
        input_MI = input_exposure.getMaskedImage()
        if input_background is not None:
            input_MI += input_background.getImage()
        input_MI = input_exposure.photoCalib.calibrateImage(input_MI)  # to nJy

        # Generate unique footprint IDs for NEIGHBOR masking
        input_MI.mask.addMaskPlane(NEIGHBOR_MASK_PLANE)
        if isinstance(footprints, SourceCatalog):
            footprints = footprintsToNumpy(footprints, bbox, asBool=False)

        # Establish pixel-to-boresight-pseudopixel transform
        pixel_scale = input_exposure.wcs.getPixelScale(bbox.getCenter()).asArcseconds() * arcseconds
        pixels_to_boresight_pseudopixels = input_exposure.detector.getTransform(PIXELS, FIELD_ANGLE).then(
            makeTransform(AffineTransform.makeScaling(1 / pixel_scale.asRadians()))
        )

        # Stamp bounding boxes
        stamp_radius = floor(Extent2D(*self.config.stamp_size) / 2)
        stamp_bbox = Box2I(Point2I(0, 0), Extent2I(1, 1)).dilatedBy(stamp_radius)  # always odd, centered 0,0
        stamp_radius_padded = floor((Extent2D(*self.config.stamp_size) * 1.42) / 2)  # max possible req. pad
        stamp_bbox_padded = Box2I(Point2I(0, 0), Extent2I(1, 1)).dilatedBy(stamp_radius_padded)

        stamps = []
        focal_plane_radii_mm = []
        for bright_star in bright_stars:
            pix_coord = Point2D(bright_star["pixel_x"], bright_star["pixel_y"])

            # Set NEIGHBOR mask plane for all sources except the current one
            neighbor_bit_mask = input_MI.mask.getPlaneBitMask(NEIGHBOR_MASK_PLANE)
            input_MI.mask.clearMaskPlane(input_MI.mask.getMaskPlaneDict()[NEIGHBOR_MASK_PLANE])
            bright_star_id = footprints[int(pix_coord.y), int(pix_coord.x)]
            neighbor_mask = (footprints != 0) & (footprints != bright_star_id)
            input_MI.mask.array[neighbor_mask] |= neighbor_bit_mask

            # Define linear shifting and rotation to recenter and align stamps
            boresight_pseudopixel_coord = pixels_to_boresight_pseudopixels.applyForward(pix_coord)
            shift = makeTransform(AffineTransform(Point2D(0, 0) - boresight_pseudopixel_coord))
            rotation = makeTransform(AffineTransform.makeRotation(-bright_star["angle_radians"] * radians))
            pixels_to_stamp_frame = pixels_to_boresight_pseudopixels.then(shift).then(rotation)

            # Warp the image and mask to the stamp frame
            stamp_MI = MaskedImageF(stamp_bbox_padded)
            warpImage(stamp_MI, input_MI, pixels_to_stamp_frame, warp_control)
            stamp_MI = stamp_MI[stamp_bbox]

            # Skip if masked area fraction is too high
            bad_bit_mask = stamp_MI.mask.getPlaneBitMask(self.config.bad_mask_planes)
            good = (stamp_MI.mask.array & bad_bit_mask) == 0
            good_frac = np.sum(good) / stamp_MI.mask.array.size
            if good_frac < self.config.min_area_fraction:
                continue

            # Define a WCS for the stamp consistent with the warping
            stamp_wcs = makeModifiedWcs(pixels_to_stamp_frame, input_exposure.wcs, False)
            projection = Projection.from_legacy(stamp_wcs, GeneralFrame(unit=u.pixel))

            # Compute the kernel image of the PSF at the stamp center
            psf_warped = WarpedPsf(input_exposure.getPsf(), pixels_to_stamp_frame, warp_control)
            psf_kernel_image = Image.from_legacy(psf_warped.computeKernelImage(Point2D(0, 0)))

            # Assemble the stamp info to be persisted alongside the image data
            stamp_info = BrightStarStampInfo(
                visit=input_exposure.visitInfo.getId(),
                detector=input_exposure.detector.getId(),
                ref_id=bright_star["id"],
                ref_mag=bright_star["mag"],
                position_x=pix_coord.x,
                position_y=pix_coord.y,
                focal_plane_radius=bright_star["radius_mm"] * u.mm,
                focal_plane_angle=bright_star["angle_radians"] * u.rad,
            )

            # Generate a bright star stamp and store outputs
            stamp = BrightStarStamp(
                image=Image.from_legacy(stamp_MI.image),
                mask=Mask.from_legacy(stamp_MI.mask),
                variance=Image.from_legacy(stamp_MI.variance),
                projection=projection,
                psf_kernel_image=psf_kernel_image,
                stamp_info=stamp_info,
            )
            stamps.append(stamp)
            focal_plane_radii_mm.append(bright_star["radius_mm"])

        num_stars = len(bright_stars)
        num_excluded = num_stars - len(stamps)
        percent_excluded = 100.0 * num_excluded / num_stars if num_stars > 0 else 0.0
        self.log.info(
            "Extracted %i bright star stamp%s. "
            "Excluded %i star%s (%.1f%%) with an unmasked area fraction below %s.",
            len(stamps),
            "" if len(stamps) == 1 else "s",
            num_excluded,
            "" if num_excluded == 1 else "s",
            percent_excluded,
            self.config.min_area_fraction,
        )

        if not stamps:
            self.log.warning(
                "No bright star stamps were retained from %i selected reference star%s.",
                num_stars,
                "" if num_stars == 1 else "s",
            )
            return None

        metadata = {
            "FOCAL_PLANE_RADIUS_MM_MIN": np.min(focal_plane_radii_mm),
            "FOCAL_PLANE_RADIUS_MM_MAX": np.max(focal_plane_radii_mm),
        }
        return BrightStarStamps(stamps, metadata=metadata)
