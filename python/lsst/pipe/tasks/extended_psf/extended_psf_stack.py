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

__all__ = (
    "ExtendedPsfStackConnections",
    "ExtendedPsfStackConfig",
    "ExtendedPsfStackTask",
)

from collections.abc import Sequence

import numpy as np
from astropy.modeling.fitting import LMLSQFitter
from astropy.modeling.models import Moffat2D

from lsst.afw.image import MaskedImageF
from lsst.afw.math import StatisticsControl, statisticsStack, stringToStatisticsProperty
from lsst.images import Image
from lsst.pex.config import Field, ListField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.utils.timer import timeMethod

from .extended_psf_candidates import ExtendedPsfCandidate, ExtendedPsfCandidates
from .extended_psf_cutout import NEIGHBOR_MASK_PLANE
from .extended_psf_fit import ExtendedPsfMoffatFit
from .extended_psf_image import ExtendedPsfImage, ExtendedPsfImageInfo


class ExtendedPsfStackConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),
):
    """Connections for ExtendedPsfStackTask."""

    extended_psf_candidates = Input(
        name="extended_psf_candidates",
        storageClass="ExtendedPsfCandidates",
        doc="Set of preprocessed cutouts, each centered on a single star.",
        dimensions=("visit", "detector"),
    )
    extended_psf = Output(
        name="extended_psf",
        storageClass="ExtendedPsfImage",
        doc="Extended PSF model built from stacking candidate cutouts.",
        dimensions=("visit", "detector"),
    )


class ExtendedPsfStackConfig(
    PipelineTaskConfig,
    pipelineConnections=ExtendedPsfStackConnections,
):
    """Configuration parameters for ExtendedPsfStackTask."""

    # Candidate selection
    bad_mask_planes = ListField[str](
        doc="Mask planes that identify excluded (masked) pixels.",
        default=[
            "BAD",
            "COSMIC_RAY",
            "CROSSTALK",
            "DETECTION_EDGE",
            "NO_DATA",
            "SATURATED",
            "SUSPECT",
            "UNMASKED_NAN",
            NEIGHBOR_MASK_PLANE,
        ],
    )
    psf_masked_flux_frac_threshold = Field[float](
        doc="Maximum allowed fraction of masked PSF flux for fitting to occur.",
        default=0.97,
    )
    min_focal_plane_radius = Field[float](
        doc="Minimum distance to the center of the focal plane, in mm. "
        "Candidates with smaller focal plane radius will be omitted.",
        default=0.0,
    )
    max_focal_plane_radius = Field[float](
        doc="Maximum distance to the center of the focal plane, in mm. "
        "Candidates with larger focal plane radius will be omitted.",
        default=np.inf,
    )

    # Stacking control
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

    # Fitting
    use_median_variance = Field[bool](
        doc="Use the median of the variance plane for fitting.",
        default=False,
    )
    fit_iterations = Field[int](
        doc="Number of iterations over pedestal-gradient and scaling fit.",
        default=5,
    )
    bg_order = Field[int](
        doc="Order of polynomial to fit for background in candidate cutouts.",
        default=2,
    )


class ExtendedPsfStackTask(PipelineTask):
    """Stack candidate cutouts to produce an extended PSF model."""

    ConfigClass = ExtendedPsfStackConfig
    _DefaultName = "extendedPsfStack"
    config: ExtendedPsfStackConfig

    def __init__(self, initInputs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        order = self.config.bg_order
        self._bg_powers = [(i, j) for i in range(order + 1) for j in range(order + 1) if i + j <= order]

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        output = self.run(**inputs)
        # Guard against empty outputs, resulting from failed fitting runs.
        if output:
            butlerQC.put(output, outputRefs)

    @timeMethod
    def run(
        self,
        extended_psf_candidates: ExtendedPsfCandidates,
    ):
        """Run the stacking task to produce an extended PSF model.

        Parameters
        ----------
        extended_psf_candidates : `ExtendedPsfCandidates`
            Set of preprocessed cutouts, each centered on a single star.

        Returns
        -------
        extended_psf :
                `~lsst.pipe.tasks.extended_psf.ExtendedPsfImage` or `None`
            The extended PSF model if stacking succeeds; `None` if stacking
            was unsuccessful (e.g, no valid candidates).
        """
        candidates = self._get_candidates(extended_psf_candidates)
        if len(candidates) == 0:
            return None

        extended_psf = self._make_extended_psf(candidates)
        if extended_psf is None:
            return None

        fit = self._fit_moffat(extended_psf)
        extended_psf = ExtendedPsfImage(
            image=extended_psf.image,
            variance=extended_psf.variance,
            info=extended_psf.info,
            fit=fit,
            metadata=extended_psf.metadata,
        )

        return Struct(extended_psf=extended_psf)

    @staticmethod
    def _focal_plane_radius_mm(candidate: ExtendedPsfCandidate) -> float | None:
        radius = candidate.star_info.focal_plane_radius
        if radius is None:
            return None
        if hasattr(radius, "to_value"):
            return float(radius.to_value("mm"))
        return float(radius)

    def _get_candidates(self, extended_psf_candidates: ExtendedPsfCandidates) -> list[ExtendedPsfCandidate]:
        """Get candidates that are within the configured focal-plane radius.

        Parameters
        ----------
        extended_psf_candidates : `ExtendedPsfCandidates`
            Candidate cutout collection.

        Returns
        -------
        candidates : `list` [`ExtendedPsfCandidate`]
            Candidates within the configured focal-plane radius range.
        """
        candidates = []
        for candidate in extended_psf_candidates:
            radius_mm = self._focal_plane_radius_mm(candidate)
            if radius_mm is None:
                continue
            if self.config.min_focal_plane_radius <= radius_mm <= self.config.max_focal_plane_radius:
                candidates.append(candidate)

        if len(candidates) == 0:
            self.log.warning(
                "No candidate cutouts found within focal-plane radius range [%s, %s] mm.",
                self.config.min_focal_plane_radius,
                self.config.max_focal_plane_radius,
            )
        elif len(candidates) < len(extended_psf_candidates):
            self.log.info(
                "Only %d of %d candidate cutouts are within focal-plane radius range [%s, %s] mm.",
                len(candidates),
                len(extended_psf_candidates),
                self.config.min_focal_plane_radius,
                self.config.max_focal_plane_radius,
            )

        self.log.info(
            "Constructing an extended PSF from %d candidate cutout%s.",
            len(candidates),
            "s" if len(candidates) > 1 else "",
        )
        return candidates

    def _make_extended_psf(self, candidates: list[ExtendedPsfCandidate]) -> ExtendedPsfImage | None:
        """Stack candidate cutouts to produce an extended PSF model.

        Parameters
        ----------
        candidates : `list` [`ExtendedPsfCandidate`]
            Candidate cutouts to stack.

        Returns
        -------
        extended_psf : `ExtendedPsfImage` or `None`
            The stacked extended PSF model, or `None` if stacking fails.
        """
        stack_type_property = stringToStatisticsProperty(self.config.stack_type)
        statistics_control = StatisticsControl(
            numSigmaClip=self.config.stack_num_sigma_clip,
            numIter=self.config.stack_num_iter,
        )
        bad_mask_bit_mask = (
            candidates[0].to_legacy(copy=False).mask.getPlaneBitMask(self.config.bad_mask_planes)
        )
        statistics_control.setAndMask(bad_mask_bit_mask)

        psf_amplitudes: dict[int, float | None] = {
            candidate_id: None for candidate_id in range(len(candidates))
        }

        extended_psf = None
        # Iteratively fit and normalize candidate cutouts, then stack
        for iteration in range(self.config.fit_iterations):
            normalized_cutouts = []
            # Loop over all candidates
            for candidate_id, candidate in enumerate(candidates):
                # Use the PSF kernel image for the first iteration
                if extended_psf is None:
                    psf_image = candidate.psf_kernel_image
                else:
                    psf_image = extended_psf.image
                normalized_image, psf_amplitude = self._fit_and_normalize(
                    candidate,
                    psf_image,
                    psf_amplitudes[candidate_id],
                )
                if normalized_image is not None and psf_amplitude is not None:
                    normalized_cutouts.append(normalized_image)
                    psf_amplitudes[candidate_id] = psf_amplitude

            if len(normalized_cutouts) == 0:
                self.log.warning(
                    "Iteration %d: no candidates were successfully fit and normalized. "
                    "No further iterations will be attempted.",
                    iteration + 1,
                )
                break

            self.log.info(
                "Iteration %d: fit and normalized %d out of %d candidate cutout%s using the %s model.",
                iteration + 1,
                len(normalized_cutouts),
                len(candidates),
                "s" if len(candidates) > 1 else "",
                "kernel-image" if extended_psf is None else "extended PSF",
            )

            # statisticsStack requires legacy MaskedImageF.
            normalized_cutouts_legacy = [cutout.to_legacy(copy=False) for cutout in normalized_cutouts]
            extended_psf_legacy = MaskedImageF(normalized_cutouts_legacy[0].getBBox())
            statisticsStack(
                extended_psf_legacy,
                normalized_cutouts_legacy,
                stack_type_property,
                statistics_control,
            )
            extended_psf = ExtendedPsfImage(
                image=Image.from_legacy(extended_psf_legacy.image),
                variance=Image.from_legacy(extended_psf_legacy.variance),
                info=ExtendedPsfImageInfo(n_stars=len(candidates)),
            )

        return extended_psf

    def _fit_and_normalize(
        self,
        candidate: ExtendedPsfCandidate,
        psf_image: Image,
        psf_amplitude: float | None,
    ) -> tuple[ExtendedPsfCandidate | None, float | None]:
        """Fit and normalize a single candidate cutout.

        Parameters
        ----------
        candidate : `ExtendedPsfCandidate`
            Candidate cutout image to fit.
        psf_image : `~lsst.images.Image`
            PSF model image for fitting.
        psf_amplitude : `float` | `None`
            Prior PSF scaling to use during background fitting. If `None`, no
            PSF subtraction is performed during background fitting.

        Returns
        -------
        candidate_normalized : `ExtendedPsfCandidate` | `None`
            Normalized candidate cutout image, or `None` if fitting fails.
        psf_amplitude : `float` | `None`
            Fitted PSF amplitude, or `None` if fitting fails.
        """
        bit_mask_bad = candidate.mask.schema.bitmask(*self.config.bad_mask_planes)

        # Calculate the fraction of PSF flux masked by bad pixels.
        psf_mask = ((candidate.mask[psf_image.bbox].array & bit_mask_bad) != 0).any(axis=-1)
        psf_masked_flux = np.dot(psf_image.array.flat, psf_mask.astype(np.float64).flat).astype(np.float64)
        psf_masked_flux_frac = psf_masked_flux / psf_image.array.sum()
        if psf_masked_flux_frac > self.config.psf_masked_flux_frac_threshold:
            return None, None

        # Fit the background.
        bg_image = self._fit_bg(
            candidate,
            list(self.config.bad_mask_planes) + ["DETECTED"],
            psf_image,
            psf_amplitude,
        )
        if bg_image is None:
            return None, None
        candidate_fit: ExtendedPsfCandidate = candidate.copy()
        candidate_fit.image.array -= bg_image.array

        # Fit PSF amplitude.
        psf_image_padded = Image(0.0, dtype=psf_image.array.dtype, bbox=candidate.bbox, unit=psf_image.unit)
        psf_image_padded[psf_image.bbox] = psf_image
        psf_amplitude = self._fit_psf_amplitude(
            candidate_fit,
            psf_image_padded,
            self.config.bad_mask_planes,
        )
        if psf_amplitude is None or psf_amplitude <= 0:
            return None, None
        candidate_fit.image.array /= psf_amplitude
        candidate_fit.variance.array /= psf_amplitude**2

        return candidate_fit, psf_amplitude

    def _fit_bg(
        self,
        candidate: ExtendedPsfCandidate,
        mask_planes: Sequence[str],
        psf_image: Image,
        psf_amplitude: float | None,
    ) -> Image | None:
        """Fit a polynomial background to a candidate cutout.

        Parameters
        ----------
        candidate : `ExtendedPsfCandidate`
            Candidate cutout image to fit.
        mask_planes : `Sequence` [`str`]
            Mask planes for identifying bad pixels.
        psf_image : `~lsst.images.Image`
            PSF image to optionally subtract prior to fitting.
        psf_amplitude : `float` | `None`
            Amplitude to scale ``psf_image`` before subtraction. If `None`, no
            PSF subtraction is performed.

        Returns
        -------
        bg_image : `~lsst.images.Image` | `None`
            Fitted background image, or `None` if fitting fails.
        """
        if psf_amplitude is not None:
            psf_image_scaled = psf_image.copy()
            psf_image_scaled.array *= psf_amplitude

        good_pixels, image, sigma = self._get_span_data(
            candidate,
            mask_planes,
            subtract_image=psf_image_scaled if psf_amplitude is not None else None,
        )
        bbox = candidate.bbox
        grid = bbox.meshgrid()
        grid_x, grid_y = grid.x, grid.y
        spans_x = grid_x[good_pixels]
        spans_y = grid_y[good_pixels]

        design_matrix = np.ones((len(image), len(self._bg_powers)), dtype=float)
        for k, (i, j) in enumerate(self._bg_powers):
            if i == j == 0:
                design_matrix[:, k] /= sigma
            else:
                design_matrix[:, k] = (spans_x**i * spans_y**j) / sigma

        solutions, _ = self._solve_design_matrix(design_matrix, image)
        if solutions is None:
            return None

        bg_array = sum(s * (grid_x**i * grid_y**j) for s, (i, j) in zip(solutions, self._bg_powers))

        return Image(bg_array.astype(np.float32), bbox=candidate.bbox, unit=candidate.unit)

    def _fit_psf_amplitude(
        self,
        candidate: ExtendedPsfCandidate,
        psf_image: Image,
        mask_planes: Sequence[str],
        mask_zeros: bool = True,
    ) -> float | None:
        """Fit the amplitude of the PSF model to one candidate cutout.

        Parameters
        ----------
        candidate : `ExtendedPsfCandidate`
            Candidate cutout image to fit.
        psf_image : `~lsst.images.Image`
            PSF image to fit to the data.
        mask_planes : `Sequence` [`str`]
            Mask planes for identifying bad pixels.
        mask_zeros : `bool`, optional
            Whether to additionally mask zero-valued PSF pixels.

        Returns
        -------
        psf_amplitude : `float` | `None`
            Fitted PSF amplitude, or `None` if fitting fails.

        Notes
        -----
        NaN and negative values in the PSF image are always masked.
        """
        psf_mask_array = np.isnan(psf_image.array) | (psf_image.array < 0)
        if mask_zeros:
            psf_mask_array |= psf_image.array == 0

        good_pixels, image, sigma = self._get_span_data(candidate, mask_planes, psf_mask_array)
        psf = psf_image.array[good_pixels]
        psf /= sigma
        design_matrix = psf[:, None]

        solutions, _ = self._solve_design_matrix(design_matrix, image)
        if solutions is None:
            return None

        return float(solutions[0])

    def _get_span_data(
        self,
        candidate: ExtendedPsfCandidate,
        mask_planes: Sequence[str],
        additional_mask: np.ndarray | None = None,
        subtract_image: Image | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | float]:
        """Get fitting data and a boolean mask of good pixels.

        Parameters
        ----------
        candidate : `ExtendedPsfCandidate`
            Candidate cutout from which to extract fitting data.
        mask_planes : `Sequence` [`str`]
            Mask planes for identifying bad pixels.
        additional_mask : `numpy.ndarray` or `None`, optional
            Additional boolean mask to combine with the image mask.
        subtract_image : `~lsst.images.Image` or `None`, optional
            Image to subtract before fitting (e.g., PSF model).

        Returns
        -------
        good_pixels : `numpy.ndarray`
            Boolean array selecting good pixels.
        image : `numpy.ndarray`
            Pixel values at good pixels after optional subtraction.
        sigma : `numpy.ndarray` | `float`
            Pixel-wise uncertainties at good pixels.
        """
        bit_mask = candidate.mask.schema.bitmask(*mask_planes)
        bad_pixels = ((candidate.mask.array & bit_mask) != 0).any(axis=-1)
        if additional_mask is not None:
            bad_pixels |= additional_mask
        if subtract_image is None:
            image_data = candidate.image.array
        else:
            image_data_image = candidate.image.copy()
            image_data_image[subtract_image.bbox].array -= subtract_image.array
            image_data = image_data_image.array
        good_pixels = ~bad_pixels
        image = image_data[good_pixels]
        variance = candidate.variance.array[good_pixels]
        if self.config.use_median_variance:
            variance = float(np.median(variance))
        if np.isscalar(variance):
            sigma = float(np.sqrt(variance))
            if sigma == 0:
                sigma = np.inf
        else:
            sigma = np.sqrt(variance)
            sigma[sigma == 0] = np.inf
        return good_pixels, image / sigma, sigma

    def _solve_design_matrix(
        self,
        design_matrix: np.ndarray,
        data: np.ndarray,
        calculate_covariance: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Solve a linear system for the given design matrix and data.

        Parameters
        ----------
        design_matrix : `numpy.ndarray`
            Design matrix for the linear system.
        data : `numpy.ndarray`
            Data vector.
        calculate_covariance : `bool`, optional
            Whether to return the covariance matrix.

        Returns
        -------
        solutions : `numpy.ndarray` or `None`
            Solution vector, or `None` if solving fails.
        covariance_matrix : `numpy.ndarray` or `None`
            Covariance matrix, if requested and available.
        """
        covariance_matrix = None
        try:
            solutions, sum_squared_residuals, *_ = np.linalg.lstsq(design_matrix, data, rcond=None)
            if calculate_covariance:
                covariance_matrix = np.linalg.pinv(design_matrix.T @ design_matrix)
        except np.linalg.LinAlgError:
            return None, None
        if sum_squared_residuals.size == 0:
            return None, None
        return solutions, covariance_matrix

    def _fit_moffat(self, extended_psf: ExtendedPsfImage, fix_center: bool = False) -> ExtendedPsfMoffatFit:
        """Fit a Moffat2D model to the stacked extended PSF image.

        Parameters
        ----------
        extended_psf : `ExtendedPsfImage`
            Stacked extended PSF image to fit.
        fix_center : `bool`, optional
            Whether to fix the Moffat2D center at the image center.

        Returns
        -------
        fit : `ExtendedPsfMoffatFit`
            Moffat fit summary and fit statistics.
        """
        bbox = extended_psf.bbox
        grid = bbox.meshgrid()
        grid_x, grid_y = grid.x, grid.y

        extended_psf_image = extended_psf.image.array
        extended_psf_sigma = np.sqrt(extended_psf.variance.array)

        fitter = LMLSQFitter()
        moffat_init = Moffat2D(
            amplitude=extended_psf_image.max(),
            x_0=bbox.x.center,
            y_0=bbox.y.center,
            gamma=3.0,
            alpha=2.5,
            fixed={"x_0": fix_center, "y_0": fix_center},
        )
        weights = 1.0 / np.clip(extended_psf_sigma, 1e-12, None)
        moffat_fit = fitter(moffat_init, grid_x, grid_y, extended_psf_image, weights=weights)

        residuals = extended_psf_image - moffat_fit(grid_x, grid_y)
        chi2 = np.sum((residuals / extended_psf_sigma) ** 2)
        dof = extended_psf_image.size - len(moffat_fit.parameters)
        reduced_chi2 = np.nan if dof <= 0 else chi2 / dof

        self.log.info(
            "Extended PSF Moffat fit x_0=%.2f, y_0=%.2f, gamma=%.2f, alpha=%.2f, reduced_chi2=%.2f.",
            moffat_fit.x_0.value,
            moffat_fit.y_0.value,
            moffat_fit.gamma.value,
            moffat_fit.alpha.value,
            reduced_chi2,
        )

        return ExtendedPsfMoffatFit(
            chi2=float(chi2),
            dof=int(dof),
            reduced_chi2=float(reduced_chi2),
            amplitude=float(moffat_fit.amplitude.value),
            x_0=float(moffat_fit.x_0.value),
            y_0=float(moffat_fit.y_0.value),
            gamma=float(moffat_fit.gamma.value),
            alpha=float(moffat_fit.alpha.value),
        )
