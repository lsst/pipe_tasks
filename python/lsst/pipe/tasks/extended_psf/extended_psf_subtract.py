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
    "ExtendedPsfSubtractConnections",
    "ExtendedPsfSubtractConfig",
    "ExtendedPsfSubtractTask",
]

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE, PIXELS
from lsst.afw.geom.transformFactory import makeTransform
from lsst.afw.image import ExposureF, ImageF
from lsst.afw.math import BackgroundList, WarpingControl, warpImage
from lsst.geom import (
    AffineTransform,
    Box2I,
    Extent2I,
    Point2D,
    Point2I,
    SpherePoint,
    arcseconds,
    radians,
)
from lsst.meas.algorithms import (
    LoadReferenceObjectsConfig,
    ReferenceObjectLoader,
    SubtractBackgroundTask,
)
from lsst.pex.config import (
    ChoiceField,
    ConfigField,
    ConfigurableField,
    Field,
    ListField,
)
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input, Output, PrerequisiteInput
from lsst.utils.timer import timeMethod

from .extended_psf_image import ExtendedPsfImage


class ExtendedPsfSubtractConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),
):
    """Connections for ExtendedPsfSubtractTask."""

    ref_cat = PrerequisiteInput(
        name="the_monster_20250219",
        storageClass="SimpleCatalog",
        doc="Reference catalog that contains star positions.",
        dimensions=("skypix",),
        multiple=True,
        deferLoad=True,
    )
    preliminary_visit_image = Input(
        name="preliminary_visit_image",
        storageClass="ExposureF",
        doc="Input background-subtracted image.",
        dimensions=("visit", "detector"),
    )
    preliminary_visit_image_background = Input(
        name="preliminary_visit_image_background",
        storageClass="Background",
        doc=(
            "Supplied input background model associated with preliminary_visit_image. "
            "When do_restore_background is True, it is added back before fitting/subtracting the "
            "extended PSF."
        ),
        dimensions=("visit", "detector"),
    )
    extended_psf = Input(
        name="extended_psf",
        storageClass="ExtendedPsfImage",
        doc="Extended PSF model image.",
        dimensions=("visit", "detector"),
    )
    preliminary_visit_image_extended_psf_subtracted = Output(
        name="preliminary_visit_image_extended_psf_subtracted",
        storageClass="ExposureF",
        doc=(
            "Preliminary-visit image with fitted extended-PSF subtracted. "
            "Background may optionally be subtracted, depending on configuration."
        ),
        dimensions=("visit", "detector"),
    )
    preliminary_visit_image_extended_psf_subtracted_background = Output(
        name="preliminary_visit_image_extended_psf_subtracted_background",
        storageClass="Background",
        doc="Background model re-estimated after extended-PSF subtraction.",
        dimensions=("visit", "detector"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if config is not None and not config.do_restore_background:
            del self.preliminary_visit_image_background
        if config is not None and not config.do_rerun_background_subtraction:
            del self.preliminary_visit_image_extended_psf_subtracted_background


class ExtendedPsfSubtractConfig(
    PipelineTaskConfig,
    pipelineConnections=ExtendedPsfSubtractConnections,
):
    """Configuration parameters for ExtendedPsfSubtractTask."""

    # Star selection
    mag_range = ListField[float](
        doc="Magnitude range in Gaia G for subtraction stars.",
        default=[10, 18],
    )
    exclude_arcsec_radius = Field[float](
        doc=(
            "No subtraction fit will be done for stars that have a neighboring star in the "
            "exclude_mag_range within exclude_arcsec_radius arcseconds."
        ),
        default=5,
    )
    exclude_mag_range = ListField[float](
        doc="Magnitude range used when searching for neighboring contaminants.",
        default=[0, 20],
    )
    min_focal_plane_radius = Field[float](
        doc="Minimum focal-plane radius in mm for subtraction stars.",
        default=0.0,
    )
    max_focal_plane_radius = Field[float](
        doc="Maximum focal-plane radius in mm for subtraction stars.",
        default=np.inf,
    )
    max_stars_per_detector = Field[int](
        doc=(
            "Maximum number of stars to subtract per detector; 0 means no cap. "
            "If this limit is reached, the selected stars are truncated to the brightest "
            "objects first (lowest magnitude values)."
        ),
        default=0,
    )

    # Warping
    warping_kernel_name = ChoiceField[str](
        doc="Warping kernel for model image warping.",
        default="lanczos5",
        allowed={
            "bilinear": "bilinear interpolation",
            "lanczos3": "Lanczos kernel of order 3",
            "lanczos4": "Lanczos kernel of order 4",
            "lanczos5": "Lanczos kernel of order 5",
        },
    )

    # Fitting
    bad_mask_planes = ListField[str](
        doc="Mask planes excluded during amplitude fitting.",
        default=[
            "BAD",
            "COSMIC_RAY",
            "CROSSTALK",
            "DETECTION_EDGE",
            "NO_DATA",
            "SATURATED",
            "SUSPECT",
            "UNMASKED_NAN",
        ],
    )
    min_model_value = Field[float](
        doc=(
            "Minimum warped-model pixel value allowed in the per-star amplitude fit. "
            "Only pixels with model > min_model_value are used when solving for PSF scale, "
            "after mask and variance filtering. Increasing this emphasizes the bright core and "
            "suppresses noisy wings; decreasing it includes more wing pixels in the fit."
        ),
        default=0.0,
    )
    min_fit_pixels = Field[int](
        doc="Minimum number of usable pixels required to fit a star.",
        default=20,
    )
    bg_order = Field[int](
        doc=(
            "Order of polynomial to fit for local background around each star. "
            "Set to 0 for a constant pedestal, 1 for a planar background, and higher values "
            "for additional flexibility."
        ),
        default=1,
    )

    # Background
    do_restore_background = Field[bool](
        doc=(
            "Restore the supplied preliminary_visit_image_background onto the working image before "
            "star fitting/subtraction."
        ),
        default=True,
    )
    do_rerun_background_subtraction = Field[bool](
        doc="Re-estimate and subtract background after extended-PSF subtraction.",
        default=True,
    )
    subtract_background = ConfigurableField(
        target=SubtractBackgroundTask,
        doc="Subtask used to re-estimate background after extended-PSF subtraction.",
    )

    # Misc
    load_reference_objects_config = ConfigField[LoadReferenceObjectsConfig](
        doc="Reference object loader for astrometric lookup.",
    )
    ref_cat_filter_name = Field[str](
        doc="Name of the filter in the reference catalog to use for star selection.",
        default="phot_g_mean",
    )


class ExtendedPsfSubtractTask(PipelineTask):
    """Subtract a fitted extended PSF model from stars in a detector image."""

    ConfigClass = ExtendedPsfSubtractConfig
    _DefaultName = "extendedPsfSubtract"
    config: ExtendedPsfSubtractConfig

    def __init__(self, initInputs=None, *args, **kwargs):
        super().__init__(initInputs=initInputs, *args, **kwargs)
        self._bg_powers = [
            (i, j)
            for i in range(self.config.bg_order + 1)
            for j in range(self.config.bg_order + 1)
            if i + j <= self.config.bg_order
        ]
        self.makeSubtask("subtract_background")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        ref_obj_loader = ReferenceObjectLoader(
            dataIds=[ref.datasetRef.dataId for ref in inputRefs.ref_cat],
            refCats=inputs.pop("ref_cat"),
            name=self.config.connections.ref_cat,
            config=self.config.load_reference_objects_config,
        )
        output = self.run(ref_obj_loader=ref_obj_loader, **inputs)
        if output:
            butlerQC.put(output, outputRefs)

    @timeMethod
    def run(
        self,
        preliminary_visit_image: ExposureF,
        extended_psf: ExtendedPsfImage,
        ref_obj_loader: ReferenceObjectLoader,
        preliminary_visit_image_background: BackgroundList | None = None,
    ):
        """Subtract fitted extended-PSF models from selected stars.

        This method clones the input exposure, optionally restores the
        associated background model for fitting, selects subtraction stars from
        the reference catalog, and fits/subtracts the warped extended-PSF
        model for each star in sequence.
        Stars are always processed in magnitude order from brightest to
        faintest, and ``max_stars_per_detector`` (if non-zero) is applied after
        sorting so only the brightest stars are retained.
        After per-star subtraction, the method optionally re-estimates the
        background model. When re-estimation is enabled, both updated
        exposure and background outputs are returned.

        Parameters
        ----------
        preliminary_visit_image : `lsst.afw.image.ExposureF`
            Background-subtracted image.
        extended_psf : `ExtendedPsfImage`
            Extended PSF model to be warped and fit per star.
        ref_obj_loader : `lsst.meas.algorithms.ReferenceObjectLoader`
            Reference object loader used for star selection.
        preliminary_visit_image_background :
            `lsst.afw.math.BackgroundList`, optional
            Supplied input background model associated with the input image.
            This is restored onto the working exposure when
            ``do_restore_background`` is True.
        """
        output_exposure = preliminary_visit_image.clone()
        if self.config.do_restore_background:
            if preliminary_visit_image_background is None:
                raise RuntimeError("do_restore_background=True requires preliminary_visit_image_background.")
            output_exposure.maskedImage += preliminary_visit_image_background.getImage()

        star_table = self._get_subtraction_star_table(ref_obj_loader, output_exposure)
        order = np.argsort(star_table["mag"])
        star_table = star_table[order]
        if self.config.max_stars_per_detector > 0 and len(star_table) > self.config.max_stars_per_detector:
            star_table = star_table[: self.config.max_stars_per_detector]

        model_image_legacy = extended_psf.image.to_legacy(copy=False)
        warp_control = WarpingControl(self.config.warping_kernel_name)

        n_attempted = 0
        n_subtracted = 0
        amplitude_values = []

        for star in star_table:
            n_attempted += 1
            amplitude = self._subtract_one_star(
                output_exposure,
                model_image_legacy,
                warp_control,
                star,
            )
            if np.isfinite(amplitude):
                n_subtracted += 1
                amplitude_values.append(amplitude)

        if self.config.do_rerun_background_subtraction:
            # Subtract background in place and retrieve the background model.
            bg_result = self.subtract_background.run(
                exposure=output_exposure,
                background=None,
                backgroundToPhotometricRatio=None,
                stats=True,
            )
            output_background = bg_result.background

        metadata = output_exposure.getMetadata()
        metadata.set("EPSFSUB_ATTEMPTED", int(n_attempted))
        metadata.set("EPSFSUB_SUBTRACTED", int(n_subtracted))
        if amplitude_values:
            metadata.set("EPSFSUB_AMP_MIN", float(np.min(amplitude_values)))
            metadata.set("EPSFSUB_AMP_MAX", float(np.max(amplitude_values)))
            metadata.set("EPSFSUB_AMP_MED", float(np.median(amplitude_values)))

        self.log.info(
            "Extended-PSF subtraction attempted on %d stars; successfully subtracted %d.",
            n_attempted,
            n_subtracted,
        )

        if self.config.do_rerun_background_subtraction:
            return Struct(
                preliminary_visit_image_extended_psf_subtracted=output_exposure,
                preliminary_visit_image_extended_psf_subtracted_background=output_background,
            )
        return Struct(preliminary_visit_image_extended_psf_subtracted=output_exposure)

    def _get_subtraction_star_table(
        self,
        ref_obj_loader: ReferenceObjectLoader,
        exposure: ExposureF,
    ) -> Table:
        """Build a table of subtraction stars from the reference catalog.

        Parameters
        ----------
        ref_obj_loader : `~lsst.meas.algorithms.ReferenceObjectLoader`
            Loader used to query reference objects in the detector footprint.
        exposure : `~lsst.afw.image.ExposureF`
            Exposure used to define the detector bounding box, WCS, and focal
            plane geometry for star selection.

        Returns
        -------
        star_table : `~astropy.table.Table`
            Table of selected subtraction stars after magnitude, isolation,
            detector-footprint, and focal-plane-radius filtering.
            Includes per-star pixel and focal-plane coordinates.
        """
        bbox = exposure.getBBox()
        wcs = exposure.getWcs()
        detector = exposure.detector

        within_region = ref_obj_loader.loadPixelBox(bbox, wcs, self.config.ref_cat_filter_name)
        ref_cat_full = within_region.refCat
        flux_field: str = within_region.fluxField
        exclude_arcsec_radius = self.config.exclude_arcsec_radius * u.arcsec

        flux_range_candidate = sorted(((self.config.mag_range * u.ABmag).to(u.nJy)).to_value())
        flux_range_neighbor = sorted(((self.config.exclude_mag_range * u.ABmag).to(u.nJy)).to_value())

        flux_min = np.min((flux_range_candidate[0], flux_range_neighbor[0]))
        flux_max = np.max((flux_range_candidate[1], flux_range_neighbor[1]))
        maximal_subset = (ref_cat_full[flux_field] >= flux_min) & (ref_cat_full[flux_field] <= flux_max)
        ref_cat_subset = Table(
            ref_cat_full.extract(
                "id",
                "coord_ra",
                "coord_dec",
                flux_field,
                where=maximal_subset,
            )
        )
        flux_subset = ref_cat_subset[flux_field]

        is_candidate = (flux_subset >= flux_range_candidate[0]) & (flux_subset <= flux_range_candidate[1])
        is_neighbor = (flux_subset >= flux_range_neighbor[0]) & (flux_subset <= flux_range_neighbor[1])

        coords = SkyCoord(ref_cat_subset["coord_ra"], ref_cat_subset["coord_dec"], unit="rad")
        coords_candidate = coords[is_candidate]
        coords_neighbor = coords[is_neighbor]

        is_candidate_isolated = np.ones(len(coords_candidate), dtype=bool)
        if len(coords_neighbor) > 0:
            _, indices_candidate, angular_separation, _ = coords_candidate.search_around_sky(
                coords_neighbor, exclude_arcsec_radius
            )
            indices_candidate = indices_candidate[angular_separation > 0 * u.arcsec]
            is_candidate_isolated[indices_candidate] = False

        star_table = ref_cat_subset[is_candidate][is_candidate_isolated]
        flux_njy = star_table[flux_field][:] * u.nJy
        star_table["mag"] = flux_njy.to(u.ABmag).to_value()

        sphere_points = [
            SpherePoint(ra * radians, dec * radians)
            for ra, dec in zip(star_table["coord_ra"], star_table["coord_dec"])
        ]
        pixel_coords = wcs.skyToPixel(sphere_points)
        star_table["pixel_x"] = [pixel_coord.x for pixel_coord in pixel_coords]
        star_table["pixel_y"] = [pixel_coord.y for pixel_coord in pixel_coords]

        mm_coords = detector.transform(pixel_coords, PIXELS, FOCAL_PLANE)
        mm_x = np.array([mm_coord.x for mm_coord in mm_coords])
        mm_y = np.array([mm_coord.y for mm_coord in mm_coords])
        radius_mm = np.sqrt(mm_x**2 + mm_y**2)
        star_table["radius_mm"] = radius_mm
        star_table["angle_radians"] = np.arctan2(mm_y, mm_x)

        within_bbox = star_table["pixel_x"] >= bbox.getMinX()
        within_bbox &= star_table["pixel_x"] <= bbox.getMaxX()
        within_bbox &= star_table["pixel_y"] >= bbox.getMinY()
        within_bbox &= star_table["pixel_y"] <= bbox.getMaxY()
        within_radii = star_table["radius_mm"] >= self.config.min_focal_plane_radius
        within_radii &= star_table["radius_mm"] <= self.config.max_focal_plane_radius
        star_table = star_table[within_bbox & within_radii]

        return star_table

    def _subtract_one_star(
        self,
        exposure: ExposureF,
        model_image_legacy: ImageF,
        warp_control: WarpingControl,
        star: Table,
    ) -> float:
        """Warp, fit, and subtract one star model in detector coordinates.

        Parameters
        ----------
        exposure : `~lsst.afw.image.ExposureF`
            Exposure to update in place by subtracting the fitted PSF model.
        model_image_legacy : `~lsst.afw.image.ImageF`
            Empirical extended-PSF model image in the model frame.
        warp_control : `~lsst.afw.math.WarpingControl`
            Warping configuration used to map the model into detector
            coordinates.
        star : `~astropy.table.Row`
            Row containing per-star subtraction metadata, including pixel
            coordinates and focal-plane angle.

        Returns
        -------
        amplitude : `float`
            Fitted PSF amplitude if subtraction succeeds. Returns `numpy.nan`
            when no valid fit is obtained or subtraction is rejected.

        Notes
        -----
        This method modifies ``exposure`` in place for the selected local
        bounding box by subtracting the fitted PSF model from the image plane.
        """
        detector_bbox = exposure.getBBox()
        pix_coord = Point2D(star["pixel_x"], star["pixel_y"])

        pixel_scale = exposure.getWcs().getPixelScale(detector_bbox.getCenter()).asArcseconds() * arcseconds
        pixels_to_boresight_pseudopixels = exposure.detector.getTransform(PIXELS, FIELD_ANGLE).then(
            makeTransform(AffineTransform.makeScaling(1 / pixel_scale.asRadians()))
        )

        boresight_pseudopixel_coord = pixels_to_boresight_pseudopixels.applyForward(pix_coord)
        shift = makeTransform(AffineTransform(Point2D(0, 0) - boresight_pseudopixel_coord))
        rotation = makeTransform(AffineTransform.makeRotation(-star["angle_radians"] * radians))
        pixels_to_cutout_frame = pixels_to_boresight_pseudopixels.then(shift).then(rotation)
        cutout_to_pixels = pixels_to_cutout_frame.inverted()

        model_bbox = model_image_legacy.getBBox()
        radius_x = model_bbox.getWidth() // 2
        radius_y = model_bbox.getHeight() // 2
        x_center = int(np.round(pix_coord.x))
        y_center = int(np.round(pix_coord.y))
        local_bbox = Box2I(
            Point2I(x_center - radius_x, y_center - radius_y),
            Extent2I(model_bbox.getWidth(), model_bbox.getHeight()),
        )
        local_bbox.clip(detector_bbox)
        if local_bbox.getArea() == 0:
            return np.nan

        model_local = ImageF(local_bbox)
        warpImage(model_local, model_image_legacy, cutout_to_pixels, warp_control)
        # Replace NaN values with zero; prevents NaN corners after rotation.
        invalid_model = ~np.isfinite(model_local.array)
        if np.any(invalid_model):
            model_local.array[invalid_model] = 0.0

        local_mi = exposure.maskedImage[local_bbox]
        data = local_mi.image.array.astype(np.float64)
        variance = local_mi.variance.array.astype(np.float64)
        model = model_local.array.astype(np.float64)
        grid_y, grid_x = np.mgrid[
            local_bbox.getMinY() : local_bbox.getMaxY() + 1,
            local_bbox.getMinX() : local_bbox.getMaxX() + 1,
        ]
        background_terms = np.vstack(
            [(grid_x**i * grid_y**j).astype(np.float64).ravel() for i, j in self._bg_powers]
        ).T

        bitmask = local_mi.mask.getPlaneBitMask(self.config.bad_mask_planes)
        good = (local_mi.mask.array & bitmask) == 0
        good &= np.isfinite(data) & np.isfinite(variance) & np.isfinite(model)
        good &= variance > 0
        good &= model > self.config.min_model_value

        if np.count_nonzero(good) < self.config.min_fit_pixels:
            return np.nan

        good_flat = good.ravel()
        sigma = np.sqrt(variance[good])
        normalized_data = data[good] / sigma
        normalized_model = model[good] / sigma
        normalized_background_terms = background_terms[good_flat] / sigma[:, None]
        fit_matrix = np.hstack([normalized_model[:, None], normalized_background_terms])

        try:
            solution, _, _, _ = np.linalg.lstsq(fit_matrix, normalized_data, rcond=None)
        except np.linalg.LinAlgError:
            return np.nan

        amplitude = float(solution[0])
        if amplitude <= 0:
            return np.nan  # Signifies a failed fit; no subtraction is done.

        local_mi.image.array -= amplitude * model
        return amplitude
