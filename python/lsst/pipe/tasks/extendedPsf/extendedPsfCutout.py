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

__all__ = ["ExtendedPsfCutoutConnections", "ExtendedPsfCutoutConfig", "ExtendedPsfCutoutTask"]

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

from .extendedPsfCandidates import ExtendedPsfCandidate, ExtendedPsfCandidateInfo, ExtendedPsfCandidates

NEIGHBOR_MASK_PLANE = "NEIGHBOR"


class ExtendedPsfCutoutConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),
):
    """Connections for ExtendedPsfCutoutTask."""

    ref_cat = PrerequisiteInput(
        name="the_monster_20250219",
        storageClass="SimpleCatalog",
        doc="Reference catalog that contains star positions.",
        dimensions=("skypix",),
        multiple=True,
        deferLoad=True,
    )
    input_exposure = Input(
        name="preliminary_visit_image",
        storageClass="ExposureF",
        doc="Background-subtracted input exposure from which to extract cutouts around a star.",
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
    extended_psf_candidates = Output(
        name="extended_psf_candidates",
        storageClass="ExtendedPsfCandidates",
        doc="Set of preprocessed cutouts, each centered on a single star.",
        dimensions=("visit", "detector"),
    )


class ExtendedPsfCutoutConfig(
    PipelineTaskConfig,
    pipelineConnections=ExtendedPsfCutoutConnections,
):
    """Configuration parameters for ExtendedPsfCutoutTask."""

    # Star selection
    mag_range = ListField[float](
        doc="Magnitude range in Gaia G. Cutouts will be made for all stars in this range.",
        default=[10, 18],
    )
    exclude_arcsec_radius = Field[float](
        doc="No cutouts will be generated for stars with a neighboring star in the range "
        "``exclude_mag_range`` mag within ``exclude_arcsec_radius`` arcseconds.",
        default=5,
    )
    exclude_mag_range = ListField[float](
        doc="No cutouts will be generated for stars with a neighboring star in the range "
        "``exclude_mag_range`` mag within ``exclude_arcsec_radius`` arcseconds.",
        default=[0, 20],
    )
    min_area_fraction = Field[float](
        doc="Minimum fraction of the cutout area, post-masking, that must remain for it to be retained.",
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

    # Cutout geometry
    cutout_size = ListField[int](
        doc="Size of the cutouts to be extracted, in pixels.",
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


class ExtendedPsfCutoutTask(PipelineTask):
    """Extract extended PSF cutouts, and warp to the same pixel grid.

    The ExtendedPsfCutoutTask is used to extract, process, and store small
    image cutouts around stars.
    This task essentially consists of two principal steps.
    First, it identifies stars within an exposure using a reference
    catalog and extracts a cutout around each.
    Second, it shifts and warps each cutout to remove optical distortions and
    sample all stars on the same pixel grid.
    """

    ConfigClass = ExtendedPsfCutoutConfig
    _DefaultName = "extendedPsfCutout"
    config: ExtendedPsfCutoutConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        ref_obj_loader = ReferenceObjectLoader(
            dataIds=[ref.datasetRef.dataId for ref in inputRefs.ref_cat],
            refCats=inputs.pop("ref_cat"),
            name=self.config.connections.ref_cat,
            config=self.config.load_reference_objects_config,
        )
        output = self.run(**inputs, ref_obj_loader=ref_obj_loader)
        # Only ingest if output exists; prevents ingesting an empty FITS file
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
        """Identify stars within an exposure using a reference catalog,
        extract cutouts around each and warp/shift cutouts onto a common frame.

        Parameters
        ----------
        input_exposure : `~lsst.afw.image.ExposureF`
            The background-subtracted image to extract cutouts around stars.
        input_background : `~lsst.afw.math.BackgroundList`
            The background model associated with the input exposure.
        input_source_catalog : `~lsst.afw.table.SourceCatalog`
            The source catalog containing footprints on the input exposure.
        ref_obj_loader : `~lsst.meas.algorithms.ReferenceObjectLoader`
            Loader to find objects within a reference catalog.

        Returns
        -------
        extended_psf_candidates :
                `~lsst.pipe.tasks.extendedPsf.ExtendedPsfCandidates`
            A set of cutouts, each centered on an extended PSF candidate.
        """
        extended_psf_candidate_table = self._get_extended_psf_candidate_table(ref_obj_loader, input_exposure)

        extended_psf_candidates = self._get_extended_psf_candidates(
            input_exposure,
            input_background,
            input_source_catalog,
            extended_psf_candidate_table,
        )

        return Struct(extended_psf_candidates=extended_psf_candidates)

    def _get_extended_psf_candidate_table(
        self,
        ref_obj_loader: ReferenceObjectLoader,
        input_exposure: ExposureF,
    ) -> Table:
        """Get a table of extended PSF candidates from the reference catalog.

        Trim the reference catalog to only those objects within the exposure
        bounding box.
        Then, select stars based on the specified magnitude range,
        isolation criteria, and optionally focal plane radius criteria.
        Finally, add columns with pixel coordinates and focal plane coordinates
        for each extended PSF candidate.

        Parameters
        ----------
        ref_obj_loader : `~lsst.meas.algorithms.ReferenceObjectLoader`
            Loader to find objects within a reference catalog.
        input_exposure : `~lsst.afw.image.ExposureF`
            The exposure for which extended PSF candidates are being selected.

        Returns
        -------
        extended_psf_candidate_table : `~astropy.table.Table`
            Table of extended PSF candidates within the exposure.
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
        maximal_subset = (ref_cat_full[flux_field] >= flux_min) & (ref_cat_full[flux_field] <= flux_max)
        ref_cat_subset_columns = ("id", "coord_ra", "coord_dec", flux_field)
        ref_cat_subset = Table(ref_cat_full.extract(*ref_cat_subset_columns, where=maximal_subset))
        flux_subset = ref_cat_subset[flux_field]

        # Identify candidate stars and their neighbors based on flux
        is_candidate = (flux_subset >= flux_range_candidate[0]) & (flux_subset <= flux_range_candidate[1])
        is_neighbor = (flux_subset >= flux_range_neighbor[0]) & (flux_subset <= flux_range_neighbor[1])

        # Trim star coordinates to candidate and neighbor subsets
        coords = SkyCoord(ref_cat_subset["coord_ra"], ref_cat_subset["coord_dec"], unit="rad")
        coords_candidate = coords[is_candidate]
        coords_neighbor = coords[is_neighbor]

        # Identify candidate stars that have no contaminant neighbors
        is_candidate_isolated = np.ones(len(coords_candidate), dtype=bool)
        if len(coords_neighbor) > 0:
            _, indices_candidate, angular_separation, _ = coords_candidate.search_around_sky(
                coords_neighbor, exclude_arcsec_radius
            )
            indices_candidate = indices_candidate[angular_separation > 0 * u.arcsec]  # Exclude self-matches
            is_candidate_isolated[indices_candidate] = False

        # Trim ref cat subset to isolated stars; add ancillary data
        extended_psf_candidate_table = ref_cat_subset[is_candidate][is_candidate_isolated]

        flux_nanojansky = extended_psf_candidate_table[flux_field][:] * u.nJy
        extended_psf_candidate_table["mag"] = flux_nanojansky.to(u.ABmag).to_value()  # AB magnitudes

        zip_ra_dec = zip(
            extended_psf_candidate_table["coord_ra"] * radians,
            extended_psf_candidate_table["coord_dec"] * radians,
        )
        sphere_points = [SpherePoint(ra, dec) for ra, dec in zip_ra_dec]
        pixel_coords = wcs.skyToPixel(sphere_points)
        extended_psf_candidate_table["pixel_x"] = [pixel_coord.x for pixel_coord in pixel_coords]
        extended_psf_candidate_table["pixel_y"] = [pixel_coord.y for pixel_coord in pixel_coords]

        mm_coords = detector.transform(pixel_coords, PIXELS, FOCAL_PLANE)
        mm_coords_x = np.array([mm_coord.x for mm_coord in mm_coords])
        mm_coords_y = np.array([mm_coord.y for mm_coord in mm_coords])
        radius_mm = np.sqrt(mm_coords_x**2 + mm_coords_y**2)
        angle_radians = np.arctan2(mm_coords_y, mm_coords_x)
        extended_psf_candidate_table["radius_mm"] = radius_mm
        extended_psf_candidate_table["angle_radians"] = angle_radians

        # Trim star catalog to those within the exposure bounding box,
        # and optionally within a range of focal plane radii
        within_bbox = extended_psf_candidate_table["pixel_x"] >= bbox.getMinX()
        within_bbox &= extended_psf_candidate_table["pixel_x"] <= bbox.getMaxX()
        within_bbox &= extended_psf_candidate_table["pixel_y"] >= bbox.getMinY()
        within_bbox &= extended_psf_candidate_table["pixel_y"] <= bbox.getMaxY()
        within_radii = extended_psf_candidate_table["radius_mm"] >= self.config.min_focal_plane_radius
        within_radii &= extended_psf_candidate_table["radius_mm"] <= self.config.max_focal_plane_radius
        extended_psf_candidate_table = extended_psf_candidate_table[within_bbox & within_radii]

        self.log.info(
            "Identified %i reference star%s in the field of view after applying magnitude and isolation "
            "cuts.",
            len(extended_psf_candidate_table),
            "s" if len(extended_psf_candidate_table) != 1 else "",
        )

        return extended_psf_candidate_table

    def _get_extended_psf_candidates(
        self,
        input_exposure: ExposureF,
        input_background: BackgroundList | None,
        footprints: SourceCatalog | np.ndarray,
        extended_psf_candidate_table: Table,
    ) -> ExtendedPsfCandidates | None:
        """Extract and warp extended PSF candidate cutouts.

        For each extended PSF candidate, extract a cutout from the input
        exposure centered on the candidate's pixel coordinates.
        Then, shift and warp the cutout to recenter on the candidate and align
        each to the same orientation.
        Finally, check the fraction of the cutout area that is masked
        (e.g. due to neighboring sources or bad pixels), and only retain those
        with sufficient unmasked area.

        Parameters
        ----------
        input_exposure : `~lsst.afw.image.ExposureF`
            The science image to extract extended PSF cutouts.
        input_background : `~lsst.afw.math.BackgroundList` | None
            The background model associated with the input exposure.
            If provided, this will be added back on to the input image.
        footprints : `~lsst.afw.table.SourceCatalog` | `numpy.ndarray`
            The source catalog containing footprints on the input exposure, or
            a 2D numpy array with the same dimensions as the input exposure
            where each pixel value corresponds to the source footprint ID.
        extended_psf_candidate_table : `~astropy.table.Table`
            Table of extended PSF candidates for which to extract cutouts.

        Returns
        -------
        extended_psf_candidates :
                `~lsst.pipe.tasks.extendedPsf.ExtendedPsfCandidates` | None
            A set of cutouts, each centered on an extended PSF candidate.
            If no cutouts are retained post-masking, returns `None`.
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

        # Cutout bounding boxes
        cutout_radius = floor(Extent2D(*self.config.cutout_size) / 2)
        cutout_bbox = Box2I(Point2I(0, 0), Extent2I(1, 1)).dilatedBy(
            cutout_radius
        )  # always odd, centered 0,0
        cutout_radius_padded = floor((Extent2D(*self.config.cutout_size) * 1.42) / 2)  # max possible req. pad
        cutout_bbox_padded = Box2I(Point2I(0, 0), Extent2I(1, 1)).dilatedBy(cutout_radius_padded)

        cutouts = []
        focal_plane_radii_mm = []
        for candidate in extended_psf_candidate_table:
            pix_coord = Point2D(candidate["pixel_x"], candidate["pixel_y"])

            # Set NEIGHBOR mask plane for all sources except the current one
            neighbor_bit_mask = input_MI.mask.getPlaneBitMask(NEIGHBOR_MASK_PLANE)
            input_MI.mask.clearMaskPlane(input_MI.mask.getMaskPlaneDict()[NEIGHBOR_MASK_PLANE])
            candidate_id = footprints[int(pix_coord.y), int(pix_coord.x)]
            neighbor_mask = (footprints != 0) & (footprints != candidate_id)
            input_MI.mask.array[neighbor_mask] |= neighbor_bit_mask

            # Define linear shifting and rotation to recenter and align cutouts
            boresight_pseudopixel_coord = pixels_to_boresight_pseudopixels.applyForward(pix_coord)
            shift = makeTransform(AffineTransform(Point2D(0, 0) - boresight_pseudopixel_coord))
            rotation = makeTransform(AffineTransform.makeRotation(-candidate["angle_radians"] * radians))
            pixels_to_cutout_frame = pixels_to_boresight_pseudopixels.then(shift).then(rotation)

            # Warp the image and mask to the cutout frame
            cutout_MI = MaskedImageF(cutout_bbox_padded)
            warpImage(cutout_MI, input_MI, pixels_to_cutout_frame, warp_control)
            cutout_MI = cutout_MI[cutout_bbox]

            # Skip if masked area fraction is too high
            bad_bit_mask = cutout_MI.mask.getPlaneBitMask(self.config.bad_mask_planes)
            good = (cutout_MI.mask.array & bad_bit_mask) == 0
            good_frac = np.sum(good) / cutout_MI.mask.array.size
            if good_frac < self.config.min_area_fraction:
                continue

            # Define a WCS for the cutout consistent with the warping
            cutout_wcs = makeModifiedWcs(pixels_to_cutout_frame, input_exposure.wcs, False)
            projection = Projection.from_legacy(cutout_wcs, GeneralFrame(unit=u.pixel))

            # Compute the kernel image of the PSF at the cutout center
            psf_warped = WarpedPsf(input_exposure.getPsf(), pixels_to_cutout_frame, warp_control)
            psf_kernel_image = Image.from_legacy(psf_warped.computeKernelImage(Point2D(0, 0)))

            # Assemble the star info to be persisted alongside the image data
            star_info = ExtendedPsfCandidateInfo(
                visit=input_exposure.visitInfo.getId(),
                detector=input_exposure.detector.getId(),
                ref_id=candidate["id"],
                ref_mag=candidate["mag"],
                position_x=pix_coord.x,
                position_y=pix_coord.y,
                focal_plane_radius=candidate["radius_mm"] * u.mm,
                focal_plane_angle=candidate["angle_radians"] * u.rad,
            )

            # Generate an extended PSF candidate and store outputs
            cutout = ExtendedPsfCandidate(
                image=Image.from_legacy(cutout_MI.image),
                mask=Mask.from_legacy(cutout_MI.mask),
                variance=Image.from_legacy(cutout_MI.variance),
                projection=projection,
                psf_kernel_image=psf_kernel_image,
                star_info=star_info,
            )
            cutouts.append(cutout)
            focal_plane_radii_mm.append(candidate["radius_mm"])

        num_stars = len(extended_psf_candidate_table)
        num_excluded = num_stars - len(cutouts)
        percent_excluded = 100.0 * num_excluded / num_stars if num_stars > 0 else 0.0
        self.log.info(
            "Extracted %i extended PSF candidate%s. "
            "Excluded %i star%s (%.1f%%) with an unmasked area fraction below %s.",
            len(cutouts),
            "" if len(cutouts) == 1 else "s",
            num_excluded,
            "" if num_excluded == 1 else "s",
            percent_excluded,
            self.config.min_area_fraction,
        )

        if not cutouts:
            self.log.warning(
                "No extended PSF candidates were retained from %i selected reference star%s.",
                num_stars,
                "" if num_stars == 1 else "s",
            )
            return None

        metadata = {
            "FOCAL_PLANE_RADIUS_MM_MIN": np.min(focal_plane_radii_mm),
            "FOCAL_PLANE_RADIUS_MM_MAX": np.max(focal_plane_radii_mm),
        }
        return ExtendedPsfCandidates(cutouts, metadata=metadata)
