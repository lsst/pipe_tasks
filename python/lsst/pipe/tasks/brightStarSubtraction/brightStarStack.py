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

"""Stack bright star postage stamp cutouts to produce an extended PSF model."""

__all__ = ["BrightStarStackConnections", "BrightStarStackConfig", "BrightStarStackTask"]

from collections import defaultdict
from copy import deepcopy

import numpy as np

from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS
from lsst.afw.geom import SkyWcs, SpanSet, makeModifiedWcs
from lsst.afw.image import ImageF, Mask, MaskedImageF
from lsst.afw.math import StatisticsControl, statisticsStack, stringToStatisticsProperty
from lsst.geom import Point2I
from lsst.meas.algorithms import BrightStarStamps
from lsst.pex.config import Field, ListField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input, Output, PrerequisiteInput
from lsst.utils.timer import timeMethod

from .brightStarCutout import NEIGHBOR_MASK_PLANE


class BrightStarStackConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "band"),
):
    """Connections for BrightStarStackTask."""

    camera = PrerequisiteInput(
        doc="Input camera.",
        name="camera",
        storageClass="Camera",
        dimensions=("instrument",),
        isCalibration=True,
    )
    bright_star_stamps = Input(
        name="bright_star_stamps",
        storageClass="BrightStarStamps",
        doc="Set of preprocessed postage stamp cutouts, each centered on a single bright star.",
        dimensions=("visit", "detector"),
        multiple=True,
        deferLoad=True,
    )
    extended_psf = Output(
        name="extended_psf",
        storageClass="MaskedImageF",
        doc="Extended PSF model, built from stacking bright star cutouts.",
        dimensions=("band",),
    )


class BrightStarStackConfig(
    PipelineTaskConfig,
    pipelineConnections=BrightStarStackConnections,
):
    """Configuration parameters for BrightStarStackTask."""

    global_reduced_chi_squared_threshold = Field[float](
        doc="Threshold for global reduced chi-squared for stamps.",
        default=5.0,
    )
    psf_reduced_chi_squared_threshold = Field[float](
        doc="Threshold for PSF reduced chi-squared for stamps.",
        default=50.0,
    )
    bright_star_threshold = Field[float](
        doc="Stars brighter than this magnitude, are considered as bright stars.",
        default=12.0,
    )
    bright_global_reduced_chi_squared_threshold = Field[float](
        doc="Threshold for global reduced chi-squared for bright star stamps.",
        default=250.0,
    )
    psf_bright_reduced_chi_squared_threshold = Field[float](
        doc="Threshold for PSF reduced chi-squared for bright star stamps.",
        default=400.0,
    )

    bad_mask_planes = ListField[str](
        doc="Mask planes that identify excluded (masked) pixels.",
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

    stack_type = Field[str](
        default="MEANCLIP",
        doc="Statistic name to use for stacking (from `~lsst.afw.math.Property`)",
    )
    stack_num_sigma_clip = Field[float](
        doc="Number of sigma to use for clipping when stacking.",
        default=3.0,
    )
    stack_num_iter = Field[int](
        doc="Number of iterations to use for clipping when stacking.",
        default=5,
    )
    magnitude_bins = ListField[int](
        doc="Bins of magnitudes for weighting purposes.",
        default=[20, 19, 18, 17, 16, 15, 13, 10],
    )
    subset_stamp_number = ListField[int](
        doc="Number of stamps per subset to generate stacked "
        "images for. The length of this parameter must be equal to the length of magnitude_bins minus one.",
        default=[300, 200, 150, 100, 100, 100, 1],
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
    use_extended_psf = Field[bool](
        doc="Use the extended PSF model to normalize bright star cutouts.",
        default=False,
    )
    psf_masked_flux_frac_threshold = Field[float](
        doc="Maximum allowed fraction of masked PSF flux for PSF fitting to occur.",
        default=0.97,
    )
    use_median_variance = Field[bool](
        doc="Use the median of the variance plane for PSF fitting.",
        default=False,
    )
    fit_iterations = Field[int](
        doc="Number of iterations over pedestal-gradient and scaling fit.",
        default=5,
    )
    bg_order = Field[int](
        doc="Order of polynomial to fit for background in bright star stamps.",
        default=2,
    )


class BrightStarStackTask(PipelineTask):
    """Stack bright star postage stamps to produce an extended PSF model."""

    ConfigClass = BrightStarStackConfig
    _DefaultName = "brightStarStack"
    config: BrightStarStackConfig

    def __init__(self, initInputs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        output = self.run(**inputs)
        butlerQC.put(output, outputRefs)

    @timeMethod
    def run(
        self,
        camera,
        bright_star_stamps: BrightStarStamps,
    ):
        """Identify bright stars within an exposure using a reference catalog,
        extract stamps around each, then preprocess them.

        Bright star preprocessing steps are: shifting, warping and potentially
        rotating them to the same pixel grid; computing their annular flux,
        and; normalizing them.

        Parameters
        ----------
        inputExposure : `~lsst.afw.image.ExposureF`
            The image from which bright star stamps should be extracted.
        inputBackground : `~lsst.afw.image.Background`
            The background model for the input exposure.
        refObjLoader : `~lsst.meas.algorithms.ReferenceObjectLoader`, optional
            Loader to find objects within a reference catalog.
        dataId : `dict` or `~lsst.daf.butler.DataCoordinate`
            The dataId of the exposure (including detector) that bright stars
            should be extracted from.

        Returns
        -------
        brightStarResults : `~lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``bright_star_stamps``
                (`~lsst.meas.algorithms.brightStarStamps.BrightStarStamps`)
        """
        stamps = self._get_stamps(camera, bright_star_stamps)
        psf_amplitudes = {stamp_id: None for stamp_id in range(len(stamps))}

        extended_psf = None
        for _ in range(self.config.fit_iterations):
            stamp_ims_normalized = []
            for stamp_id, stamp in enumerate(stamps):
                fit_results = self._fit_and_normalize(
                    stamp.stamp_im,
                    stamp.psf if extended_psf is None else extended_psf,
                    psf_amplitudes[stamp_id],
                    self.config.bg_order,
                )
                if fit_results is not None:
                    stamp_ims_normalized.append(fit_results.masked_image_normalized)
                    psf_amplitudes[stamp_id] = fit_results.psf_amplitude

            stack_type_property = stringToStatisticsProperty(self.config.stack_type)
            statistics_control = StatisticsControl(
                numSigmaClip=self.config.stack_num_sigma_clip, numIter=self.config.stack_num_iter
            )
            bad_mask_bit_mask = stamp_ims_normalized[0].mask.getPlaneBitMask(self.config.bad_mask_planes)
            statistics_control.setAndMask(bad_mask_bit_mask)
            extended_psf = MaskedImageF(stamp_ims_normalized[0].getBBox())
            statisticsStack(extended_psf, stamp_ims_normalized, stack_type_property, statistics_control)
            breakpoint()

        mag_bins_dict = {}
        subset_stampMIs = {}
        self.metadata["psf_star_count"] = {}
        self.metadata["psf_star_count"]["all"] = 0
        for i in range(len(self.config.subset_stamp_number)):
            self.metadata["psf_star_count"][str(self.config.magnitude_bins[i + 1])] = 0
        for stampsDDH in bright_star_stamps:
            stamps = stampsDDH.get()
            self.metadata["psf_star_count"]["all"] += len(stamps)
            for stamp in stamps:
                if stamp.ref_mag >= self.config.bright_star_threshold:
                    global_reduced_chi_squared_threshold = self.config.global_reduced_chi_squared_threshold
                    psf_reduced_chi_squared_threshold = self.config.psf_reduced_chi_squared_threshold
                else:
                    global_reduced_chi_squared_threshold = (
                        self.config.bright_global_reduced_chi_squared_threshold
                    )
                    psf_reduced_chi_squared_threshold = self.config.psf_bright_reduced_chi_squared_threshold
                for i in range(len(self.config.subset_stamp_number)):
                    if (
                        stamp.global_reduced_chi_squared > global_reduced_chi_squared_threshold
                        or stamp.psf_reduced_chi_squared > psf_reduced_chi_squared_threshold
                        or stamp.focal_plane_radius < self.config.min_focal_plane_radius
                        or stamp.focal_plane_radius > self.config.max_focal_plane_radius
                    ):
                        continue

                    if (
                        stamp.ref_mag < self.config.magnitude_bins[i]
                        and stamp.ref_mag > self.config.magnitude_bins[i + 1]
                    ):
                        self._applyStampFit(stamp)
                        if not self.config.magnitude_bins[i + 1] in mag_bins_dict.keys():
                            mag_bins_dict[self.config.magnitude_bins[i + 1]] = []
                        stampMI = stamp.stamp_im
                        mag_bins_dict[self.config.magnitude_bins[i + 1]].append(stampMI)
                        bad_mask_bit_mask = stampMI.mask.getPlaneBitMask(self.config.bad_mask_planes)
                        statistics_control.setAndMask(bad_mask_bit_mask)
                        if (
                            len(mag_bins_dict[self.config.magnitude_bins[i + 1]])
                            == self.config.subset_stamp_number[i]
                        ):
                            if self.config.magnitude_bins[i + 1] not in subset_stampMIs.keys():
                                subset_stampMIs[self.config.magnitude_bins[i + 1]] = []
                            subset_stampMIs[self.config.magnitude_bins[i + 1]].append(
                                statisticsStack(
                                    mag_bins_dict[self.config.magnitude_bins[i + 1]],
                                    stack_type_property,
                                    statistics_control,
                                )
                            )
                            self.metadata["psf_star_count"][str(self.config.magnitude_bins[i + 1])] += len(
                                mag_bins_dict[self.config.magnitude_bins[i + 1]]
                            )
                            mag_bins_dict[self.config.magnitude_bins[i + 1]] = []

        for key in mag_bins_dict.keys():
            if key not in subset_stampMIs.keys():
                subset_stampMIs[key] = []
                subset_stampMIs[key].append(
                    statisticsStack(mag_bins_dict[key], stack_type_property, statistics_control)
                )
                self.metadata["psf_star_count"][str(key)] += len(mag_bins_dict[key])

        final_subset_stampMIs = []
        for key in subset_stampMIs.keys():
            final_subset_stampMIs.extend(subset_stampMIs[key])
        bad_mask_bit_mask = final_subset_stampMIs[0].mask.getPlaneBitMask(self.config.bad_mask_planes)
        statistics_control.setAndMask(bad_mask_bit_mask)
        extendedPsfMI = statisticsStack(final_subset_stampMIs, stack_type_property, statistics_control)

        extendedPsfExtent = extendedPsfMI.getBBox().getDimensions()
        extendedPsfOrigin = Point2I(-1 * (extendedPsfExtent.x // 2), -1 * (extendedPsfExtent.y // 2))
        extendedPsfMI.setXY0(extendedPsfOrigin)

        return Struct(extendedPsf=extendedPsfMI)

    def _get_stamps(self, camera, bright_star_stamps):
        """Get bright star stamps that are within the specified fp radius."""

        stampsDDH_by_detector = defaultdict(list)
        for stamp in bright_star_stamps:
            stampsDDH_by_detector[stamp.dataId["detector"]].append(stamp)

        stampsDDH = []
        for detector in camera:
            det_id = detector.getId()
            if det_id not in stampsDDH_by_detector:
                continue
            corners_fp = detector.getCorners(detector.makeCameraSys(FOCAL_PLANE))
            corners_mm = np.hypot([p.getX() for p in corners_fp], [p.getY() for p in corners_fp])
            if (
                corners_mm.max() >= self.config.min_focal_plane_radius
                and corners_mm.min() <= self.config.max_focal_plane_radius
            ):
                stampsDDH.extend(stampsDDH_by_detector[det_id])

        self.log.info(
            "Isolated bright star stamp collections in %d detector%s.",
            len(stampsDDH),
            "s" if len(stampsDDH) != 1 else "",
        )

        stamps = [
            stamp
            for stampDDH in stampsDDH
            for stamp in stampDDH.get()
            if self.config.min_focal_plane_radius
            <= stamp.focal_plane_radius
            <= self.config.max_focal_plane_radius
        ]

        self.log.info(
            "Extracted %d stamp%s in focal plane radius range [%s, %s] mm.",
            len(stamps),
            "s" if len(stamps) != 1 else "",
            self.config.min_focal_plane_radius,
            self.config.max_focal_plane_radius,
        )

        return stamps

    def _fit_and_normalize(self, masked_image, psf, psf_amplitude, bg_order):
        """Fit the PSF model to a single bright star stamp."""
        if isinstance(psf, MaskedImageF):
            psf_image = psf.image  # Assumed to be warped, center at [0,0]
        else:
            psf_image = psf.computeKernelImage(psf.getAveragePosition())

        bit_mask_bad = masked_image.mask.getPlaneBitMask(self.config.bad_mask_planes)

        # Calculate the fraction of the PSF image flux masked by bad pixels
        psf_mask = ImageF(psf_image.getBBox())
        psf_mask.array[:, :] = (masked_image.mask[psf_image.getBBox()].array & bit_mask_bad).astype(bool)
        psf_masked_flux_frac = (
            np.dot(psf_image.array.flat, psf_mask.array.flat).astype(np.float64) / psf_image.array.sum()
        )
        if psf_masked_flux_frac > self.config.psf_masked_flux_frac_threshold:
            return None  # Handle cases where the PSF image is mostly masked

        # Fit the background
        bg = self._fit_bg(
            masked_image, list(self.config.bad_mask_planes) + ["DETECTED"], bg_order, psf_image, psf_amplitude
        )
        if bg is None:
            return None
        masked_image_fit = masked_image.clone()
        masked_image_fit.image -= bg

        # Fit the PSF amplitude
        psf_image_padded = ImageF(masked_image.getBBox())
        psf_image_padded[psf_image.getBBox()] = psf_image.convertF()
        psf_amplitude = self._fit_psf_amplitude(
            masked_image_fit, psf_image_padded, self.config.bad_mask_planes
        )
        if psf_amplitude is None or psf_amplitude <= 0:
            return None
        masked_image_fit /= psf_amplitude

        return Struct(masked_image_normalized=masked_image_fit, psf_amplitude=psf_amplitude)

    def _fit_bg(self, masked_image, mask_planes, order, psf_image, psf_amplitude):
        if psf_amplitude is not None:
            psf_image_scaled = psf_image.clone()
            psf_image_scaled.array *= psf_amplitude

        spans, image, sigma = self._get_span_data(
            masked_image, mask_planes, subtract_image=psf_image_scaled if psf_amplitude is not None else None
        )
        spans_yx = spans.indices()
        spans_x = spans_yx[1, :]
        spans_y = spans_yx[0, :]

        powers = [(i, j) for i in range(order + 1) for j in range(order + 1) if i + j <= order]
        design_matrix = np.ones((len(image), len(powers)), dtype=float)
        for k, (i, j) in enumerate(powers):
            if i == j == 0:
                design_matrix[:, k] /= sigma  # constant term
            else:
                design_matrix[:, k] = (spans_x**i * spans_y**j) / sigma

        solutions, covariance_matrix = self._solve_design_matrix(design_matrix, image)
        if solutions is None:
            return None

        bbox = masked_image.getBBox()
        grid_x, grid_y = np.meshgrid(bbox.getX().arange(), bbox.getY().arange())
        bg_array = sum(s * (grid_x**i * grid_y**j) for s, (i, j) in zip(solutions, powers))

        return ImageF(bg_array.astype(np.float32), xy0=masked_image.getXY0())

    def _fit_psf_amplitude(self, masked_image, psf_image, mask_planes, mask_zeros=True):
        psf_mask_array = np.isnan(psf_image.array) | (psf_image.array < 0)
        if mask_zeros:
            psf_mask_array |= psf_image.array == 0
        psf_mask = Mask(masked_image.getBBox())  # type: ignore
        psf_mask.array = psf_mask_array

        spans, image, sigma = self._get_span_data(masked_image, mask_planes, psf_mask)
        psf = spans.flatten(psf_image.array, psf_image.getXY0())
        psf /= sigma
        design_matrix = psf[:, None]

        solutions, covariance_matrix = self._solve_design_matrix(design_matrix, image)
        if solutions is None:
            return None

        return solutions[0]

    # def _applyStampFit(self, stamp):
    #     """Apply fitted stamp components to a single bright star stamp."""
    #     stampMI = stamp.stamp_im
    #     stamp_bbox = stampMI.getBBox()

    #     x_grid, y_grid = np.meshgrid(stamp_bbox.getX().arange(), stamp_bbox.getY().arange())

    #     x_plane = ImageF((x_grid * stamp.gradient_x).astype(np.float32), xy0=stampMI.getXY0())
    #     y_plane = ImageF((y_grid * stamp.gradient_y).astype(np.float32), xy0=stampMI.getXY0())

    #     x_curve = ImageF((x_grid**2 * stamp.curvature_x).astype(np.float32), xy0=stampMI.getXY0())
    #     y_curve = ImageF((y_grid**2 * stamp.curvature_y).astype(np.float32), xy0=stampMI.getXY0())
    #     xy_curve = ImageF((x_grid * y_grid * stamp.curvature_xy).astype(np.float32), xy0=stampMI.getXY0())

    #     stampMI -= stamp.pedestal
    #     stampMI -= x_plane
    #     stampMI -= y_plane
    #     stampMI -= x_curve
    #     stampMI -= y_curve
    #     stampMI -= xy_curve
    #     stampMI /= stamp.scale

    def _get_span_data(self, masked_image, mask_planes, additional_mask=None, subtract_image=None):
        bit_mask = masked_image.mask.getPlaneBitMask(mask_planes)
        bad_spans = SpanSet.fromMask(masked_image.mask, bit_mask)
        if additional_mask is not None:
            additional_bad_spans = SpanSet.fromMask(additional_mask, bit_mask)
            bad_spans = bad_spans.union(additional_bad_spans)
        spans = SpanSet(masked_image.getBBox()).intersectNot(bad_spans)
        image_data = masked_image.image.array - (subtract_image.array if subtract_image is not None else 0)
        image = spans.flatten(image_data, masked_image.getXY0())
        variance = spans.flatten(masked_image.variance.array, masked_image.getXY0())
        if self.config.use_median_variance:
            variance = np.median(variance)
        sigma = np.sqrt(variance)
        return spans, image / sigma, sigma

    def _solve_design_matrix(self, design_matrix, data):
        try:
            solutions, sum_squared_residuals, *_ = np.linalg.lstsq(design_matrix, data, rcond=None)
            covariance_matrix = np.linalg.inv(design_matrix.T @ design_matrix)
        except np.linalg.LinAlgError:
            return None, None  # Handle singular matrix errors
        if sum_squared_residuals.size == 0:
            return None, None  # Handle cases where sum of the squared residuals are empty

        return solutions, covariance_matrix
