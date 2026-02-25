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

from collections.abc import Sequence

import numpy as np
from astropy.modeling.fitting import LMLSQFitter
from astropy.modeling.models import Moffat2D

from lsst.afw.geom import SpanSet
from lsst.afw.image import ImageF, Mask, MaskedImageF
from lsst.afw.math import StatisticsControl, statisticsStack, stringToStatisticsProperty
from lsst.daf.base import PropertyList
from lsst.meas.algorithms import BrightStarStamp, BrightStarStamps, ImagePsf
from lsst.pex.config import Field, ListField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.utils.timer import timeMethod

from .brightStarCutout import NEIGHBOR_MASK_PLANE


class BrightStarStackConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),
):
    """Connections for BrightStarStackTask."""

    bright_star_stamps = Input(
        name="bright_star_stamps",
        storageClass="BrightStarStamps",
        doc="Set of preprocessed postage stamp cutouts, each centered on a single bright star.",
        dimensions=("visit", "detector"),
    )
    extended_psf = Output(
        name="extended_psf",
        storageClass="MaskedImageF",
        doc="Extended PSF model, built from stacking bright star cutouts.",
        dimensions=("visit", "detector"),
    )
    extended_psf_moffat_fit = Output(
        name="extended_psf_moffat_fit",
        storageClass="PropertyList",
        doc="Fitted Moffat2D parameters and fit statistics for the extended PSF model.",
        dimensions=("visit", "detector"),
    )


class BrightStarStackConfig(
    PipelineTaskConfig,
    pipelineConnections=BrightStarStackConnections,
):
    """Configuration parameters for BrightStarStackTask."""

    # Star selection
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
    psf_masked_flux_frac_threshold = Field[float](
        doc="Maximum allowed fraction of masked PSF flux for fitting to occur.",
        default=0.97,
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

        order = self.config.bg_order
        self._bg_powers = [(i, j) for i in range(order + 1) for j in range(order + 1) if i + j <= order]

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        output = self.run(**inputs)
        # Guard against empty outputs, resulting from failed fitting runs
        if output:
            butlerQC.put(output, outputRefs)

    @timeMethod
    def run(
        self,
        bright_star_stamps: BrightStarStamps,
    ):
        """Run the BrightStarStackTask to produce an extended PSF model from
        bright star stamps.

        Parameters
        ----------
        bright_star_stamps : `~lsst.meas.algorithms.BrightStarStamps
            Set of preprocessed postage stamp cutouts, each centered on a
            single bright star.

        Returns
        -------
        extended_psf : `~lsst.afw.image.MaskedImageF` or `None`
            The extended PSF model produced by stacking bright star stamps, or
            `None` if the stacking was unsuccessful (e.g., no valid stamps).
        """
        stamps = self._get_stamps(bright_star_stamps)
        if len(stamps) == 0:
            return None
        extended_psf = self._make_extended_psf(stamps)
        moffat_results = self._fit_moffat(extended_psf)

        return Struct(extended_psf=extended_psf, moffat_results=moffat_results)

    def _get_stamps(self, bright_star_stamps: BrightStarStamps) -> list[BrightStarStamp]:
        """Get bright star stamps that are within the specified fp radius.

        Parameters
        ----------
        bright_star_stamps : `~lsst.meas.algorithms.BrightStarStamps
            Set of preprocessed postage stamp cutouts, each centered on a
            single bright star.

        Returns
        -------
        stamps : `list`[`~lsst.meas.algorithms.BrightStarStamp]
            List of bright star stamps that are within the specified focal
            plane radii.
        """
        stamps = [
            stamp
            for stamp in bright_star_stamps
            if stamp.focal_plane_radius is not None
            and self.config.min_focal_plane_radius
            <= stamp.focal_plane_radius
            <= self.config.max_focal_plane_radius
        ]
        if len(stamps) == 0:
            self.log.warning(
                "No bright star stamps found within the focal plane radius range [%s, %s] mm.",
                self.config.min_focal_plane_radius,
                self.config.max_focal_plane_radius,
            )
        elif len(stamps) < len(bright_star_stamps):
            self.log.info(
                "Only %d of %d bright star stamps are within the focal plane radius range [%s, %s] mm.",
                len(stamps),
                len(bright_star_stamps),
                self.config.min_focal_plane_radius,
                self.config.max_focal_plane_radius,
            )
        self.log.info(
            "Constructing an extended PSF from %d bright star stamp%s.",
            len(stamps),
            "s" if len(stamps) > 1 else "",
        )
        return stamps

    def _make_extended_psf(self, stamps: list[BrightStarStamp]) -> MaskedImageF:
        """Stack bright star stamps to produce an extended PSF model.

        Parameters
        ----------
        stamps : `list`[`~lsst.meas.algorithms.BrightStarStamp]
            List of bright star stamps to stack.

        Returns
        -------
        extended_psf : `~lsst.afw.image.MaskedImageF`
            The extended PSF model produced by stacking bright star stamps.
        """
        stack_type_property = stringToStatisticsProperty(self.config.stack_type)
        statistics_control = StatisticsControl(
            numSigmaClip=self.config.stack_num_sigma_clip, numIter=self.config.stack_num_iter
        )
        bad_mask_bit_mask = stamps[0].stamp_im.mask.getPlaneBitMask(self.config.bad_mask_planes)
        statistics_control.setAndMask(bad_mask_bit_mask)

        psf_amplitudes: dict[int, float | None] = {stamp_id: None for stamp_id in range(len(stamps))}

        extended_psf = None
        for iteration in range(self.config.fit_iterations):
            stamp_ims_normalized = []
            for stamp_id, stamp in enumerate(stamps):
                masked_image_normalized, psf_amplitude = self._fit_and_normalize(
                    stamp.stamp_im,
                    stamp.psf if extended_psf is None else extended_psf,
                    psf_amplitudes[stamp_id],
                )
                if masked_image_normalized is not None and psf_amplitude is not None:
                    stamp_ims_normalized.append(masked_image_normalized)
                    psf_amplitudes[stamp_id] = psf_amplitude

            if len(stamp_ims_normalized) == 0:
                self.log.warning(
                    "Iteration %d: No stamps were successfully fit and normalized. "
                    "No further iterations will be attempted.",
                    iteration + 1,
                )
                break
            else:
                self.log.info(
                    "Iteration %d: Successfully fit and normalized %d out of %d stamp%s using the %s model.",
                    iteration + 1,
                    len(stamp_ims_normalized),
                    len(stamps),
                    "s" if len(stamps) > 1 else "",
                    "baseline PSF" if extended_psf is None else "extended PSF",
                )

            # Stack normalized stamps from an iteration to make an extended PSF
            extended_psf = MaskedImageF(stamp_ims_normalized[0].getBBox())
            statisticsStack(extended_psf, stamp_ims_normalized, stack_type_property, statistics_control)

        return extended_psf

    def _fit_and_normalize(
        self,
        masked_image: MaskedImageF,
        psf: ImagePsf | MaskedImageF,
        psf_amplitude: float | None,
    ) -> tuple[MaskedImageF | None, float | None]:
        """Fit the PSF model to a single bright star stamp.

        Parameters
        ----------
        masked_image : `~lsst.afw.image.MaskedImageF`
            The masked image of the bright star stamp to fit.
        psf : `~lsst.meas.algorithms.ImagePsf` | `~lsst.afw.image.MaskedImageF`
            The PSF model to fit to the data. This can be either an `ImagePsf`
            or a `MaskedImageF` (from, for example, a prior iteration).
        psf_amplitude : `float` | `None`
            The amplitude to scale the PSF image by before subtraction in the
            background fitting step. If `None`, no PSF subtraction is performed
            during background fitting.

        Returns
        -------
        masked_image_normalized : `~lsst.afw.image.MaskedImageF` | `None`
            The masked image of the bright star stamp after fitting and
            normalization, or `None` if the fit was unsuccessful.
        psf_amplitude : `float` | `None`
            The fitted amplitude of the PSF model, or `None` if the fit was
            unsuccessful.
        """
        if isinstance(psf, ImagePsf):
            psf_image = psf.computeKernelImage(psf.getAveragePosition())
        else:
            psf_image = psf.image  # Assumed to be warped, center at [0,0]

        bit_mask_bad = masked_image.mask.getPlaneBitMask(self.config.bad_mask_planes)

        # Calculate the fraction of the PSF image flux masked by bad pixels
        psf_mask = ImageF(psf_image.getBBox())
        psf_mask.array[:, :] = (masked_image.mask[psf_image.getBBox()].array & bit_mask_bad).astype(bool)
        psf_masked_flux_frac = (
            np.dot(psf_image.array.flat, psf_mask.array.flat).astype(np.float64) / psf_image.array.sum()
        )
        if psf_masked_flux_frac > self.config.psf_masked_flux_frac_threshold:
            return None, None  # Handle cases where the PSF image is mostly masked

        # Fit the background
        bg_image = self._fit_bg(
            masked_image, list(self.config.bad_mask_planes) + ["DETECTED"], psf_image, psf_amplitude
        )
        if bg_image is None:
            return None, None
        masked_image_fit = masked_image.clone()
        masked_image_fit.image -= bg_image

        # Fit the PSF amplitude
        psf_image_padded = ImageF(masked_image.getBBox())
        psf_image_padded[psf_image.getBBox()] = psf_image.convertF()
        psf_amplitude = self._fit_psf_amplitude(
            masked_image_fit, psf_image_padded, self.config.bad_mask_planes
        )
        if psf_amplitude is None or psf_amplitude <= 0:
            return None, None
        masked_image_fit /= psf_amplitude

        return masked_image_fit, psf_amplitude

    def _fit_bg(
        self,
        masked_image: MaskedImageF,
        mask_planes: Sequence[str],
        psf_image: ImageF,
        psf_amplitude: float | None,
    ) -> ImageF | None:
        """Fit a polynomial background to a bright star stamp, optionally
        subtracting a scaled PSF model.

        Parameters
        ----------
        masked_image : `~lsst.afw.image.MaskedImageF`
            The masked image of the bright star stamp to fit.
        mask_planes : `Sequence`[`str`]
            Sequence of mask planes to use for identifying bad pixels.
        psf_image : `~lsst.afw.image.ImageF`
            The PSF image to optionally subtract from the data before fitting.
        psf_amplitude : `float` | `None`
            The amplitude to scale the PSF image by before subtraction.
            If `None`, no PSF subtraction is performed.

        Returns
        -------
        bg_image : `~lsst.afw.image.ImageF` | `None`
            The fitted background image, or `None` if the fit was unsuccessful.
        """
        if psf_amplitude is not None:
            psf_image_scaled = psf_image.clone()
            psf_image_scaled.array *= psf_amplitude

        spans, image, sigma = self._get_span_data(
            masked_image, mask_planes, subtract_image=psf_image_scaled if psf_amplitude is not None else None
        )
        spans_yx = spans.indices()
        spans_x = spans_yx[1, :]
        spans_y = spans_yx[0, :]

        design_matrix = np.ones((len(image), len(self._bg_powers)), dtype=float)
        for k, (i, j) in enumerate(self._bg_powers):
            if i == j == 0:
                design_matrix[:, k] /= sigma  # constant term
            else:
                design_matrix[:, k] = (spans_x**i * spans_y**j) / sigma

        solutions, _ = self._solve_design_matrix(design_matrix, image)
        if solutions is None:
            return None

        bbox = masked_image.getBBox()
        grid_x, grid_y = np.meshgrid(bbox.getX().arange(), bbox.getY().arange())
        bg_array = sum(s * (grid_x**i * grid_y**j) for s, (i, j) in zip(solutions, self._bg_powers))

        return ImageF(bg_array.astype(np.float32), xy0=masked_image.getXY0())

    def _fit_psf_amplitude(
        self,
        masked_image: MaskedImageF,
        psf_image: ImageF,
        mask_planes: Sequence[str],
        mask_zeros: bool = True,
    ) -> float | None:
        """Fit the amplitude of the PSF model to a bright star stamp.

        Parameters
        ----------
        masked_image : `~lsst.afw.image.MaskedImageF`
            The masked image of the bright star stamp to fit.
        psf_image : `~lsst.afw.image.ImageF`
            The PSF image to fit to the data.
        mask_planes : `Sequence`[`str`]
            Sequence of mask planes to use for identifying bad pixels.
        mask_zeros : `bool`, optional
            Whether to mask pixels where the PSF image is zero, in addition to
            the bad pixels identified by the mask planes.

        Returns
        -------
        psf_amplitude : `float` | `None`
            The fitted amplitude of the PSF model, or `None` if the fit was
            unsuccessful.

        Notes
        -----
        NaN and negative values in the PSF image are always masked, in addition
        to the specified mask planes and optionally zero-valued pixels.
        """
        psf_mask_array = np.isnan(psf_image.array) | (psf_image.array < 0)
        if mask_zeros:
            psf_mask_array |= psf_image.array == 0
        psf_mask = Mask(masked_image.getBBox())  # type: ignore
        psf_mask.array = psf_mask_array

        spans, image, sigma = self._get_span_data(masked_image, mask_planes, psf_mask)
        psf = spans.flatten(psf_image.array, psf_image.getXY0())
        psf /= sigma
        design_matrix = psf[:, None]  # A single column for amplitude fit alone

        solutions, _ = self._solve_design_matrix(design_matrix, image)
        if solutions is None:
            return None

        return float(solutions[0])

    def _get_span_data(
        self,
        masked_image: MaskedImageF,
        mask_planes: Sequence[str],
        additional_mask: Mask | None = None,
        subtract_image: ImageF | None = None,
    ) -> tuple[SpanSet, np.ndarray, np.ndarray | float]:
        """Get the data and corresponding spans for fitting.

        Parameters
        ----------
        masked_image : `~lsst.afw.image.MaskedImageF`
            The masked image from which to extract data for fitting.
        mask_planes : `Sequence`[`str`]
            Sequence of mask planes to use for identifying bad pixels.
        additional_mask : `~lsst.afw.image.Mask` or `None`, optional
            An additional mask to combine with the masked_image's mask.
        subtract_image : `~lsst.afw.image.ImageF` or `None`, optional
            An image to subtract from the masked_image data before fitting
            (e.g., a PSF model). If `None`, no subtraction is performed.

        Returns
        -------
        spans : `~lsst.afw.geom.SpanSet`
            The spans corresponding to the good pixels to be used for fitting.
        image : `numpy.ndarray`
            The pixel values from the masked_image data (after optional
            subtraction) corresponding to the derived span sets.
        sigma : `numpy.ndarray` | `float`
            The pixel-wise uncertainties from the masked_image's variance
            corresponding to the derived span sets.
            This can be a single value if `use_median_variance` is `True`.
        """
        bit_mask = masked_image.mask.getPlaneBitMask(mask_planes)
        bad_spans = SpanSet.fromMask(masked_image.mask, bit_mask)
        if additional_mask is not None:
            additional_bit_mask = additional_mask.getPlaneBitMask(mask_planes)
            additional_bad_spans = SpanSet.fromMask(additional_mask, additional_bit_mask)
            bad_spans = bad_spans.union(additional_bad_spans)
        spans = SpanSet(masked_image.getBBox()).intersectNot(bad_spans)
        image_data = masked_image.image.array - (subtract_image.array if subtract_image is not None else 0)
        image = spans.flatten(image_data, masked_image.getXY0())
        variance = spans.flatten(masked_image.variance.array, masked_image.getXY0())
        if self.config.use_median_variance:
            variance = np.median(variance)
        sigma = np.sqrt(variance)
        sigma[sigma == 0] = np.inf  # Guard against zero variance pixels
        return spans, image / sigma, sigma

    def _solve_design_matrix(
        self,
        design_matrix: np.ndarray,
        data: np.ndarray,
        calculate_covariance: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Solve the linear system for the given design matrix and data.

        Parameters
        ----------
        design_matrix : `numpy.ndarray`
            The design matrix for the linear system. Each column corresponds to
            a basis function evaluated at the data points. The number of rows
            should match the length of the data vector.
        data : `numpy.ndarray`
            The data vector for the linear system. The length should match the
            number of rows in the design matrix.
        calculate_covariance : `bool`, optional
            Whether to calculate and return the covariance matrix of the
            solutions.

        Returns
        -------
        solutions : `numpy.ndarray` or `None`
            The solutions to the linear system, or `None` if the system could
            not be solved.
        covariance_matrix : `numpy.ndarray` or `None`
            The covariance matrix of the solutions, or `None` if the system
            could not be solved. Only returned if `calculate_covariance` is
            `True`, otherwise `None`.
        """
        covariance_matrix = None
        try:
            solutions, sum_squared_residuals, *_ = np.linalg.lstsq(design_matrix, data, rcond=None)
            if calculate_covariance:
                covariance_matrix = np.linalg.pinv(design_matrix.T @ design_matrix)
        except np.linalg.LinAlgError:
            return None, None  # Handle singular matrix errors
        if sum_squared_residuals.size == 0:
            return None, None  # Handle cases where sum of the squared residuals are empty
        return solutions, covariance_matrix

    def _fit_moffat(self, extended_psf: MaskedImageF, fix_center: bool = False) -> PropertyList:
        """Fit a Moffat2D model to the extended PSF image.

        Parameters
        ----------
        extended_psf : `~lsst.afw.image.MaskedImageF`
            The extended PSF image to fit.
        fix_center : `bool`, optional
            Whether to fix the center of the Moffat2D model to the center of
            the extended PSF image. If `False`, the center will be allowed to
            vary in the fit.

        Returns
        -------
        moffat_results : `~lsst.daf.base.PropertyList`
            A PropertyList containing the fitted Moffat2D parameters and fit
            statistics (chi-squared and reduced chi-squared).
        """
        bbox = extended_psf.getBBox()
        grid_x, grid_y = np.meshgrid(bbox.getX().arange(), bbox.getY().arange())

        extended_psf_image = extended_psf.image.array
        extended_psf_sigma = np.sqrt(extended_psf.variance.array)

        fitter = LMLSQFitter()
        moffat_init = Moffat2D(
            amplitude=extended_psf_image.max(),
            x_0=bbox.getCenter().x,
            y_0=bbox.getCenter().y,
            gamma=3.0,
            alpha=2.5,
            fixed={"x_0": fix_center, "y_0": fix_center},
        )
        weights = 1.0 / np.clip(extended_psf_sigma, 1e-12, None)
        moffat_fit = fitter(moffat_init, grid_x, grid_y, extended_psf_image, weights=weights)

        residuals = extended_psf_image - moffat_fit(grid_x, grid_y)
        chi2 = np.sum((residuals / extended_psf_sigma) ** 2)
        dof = extended_psf_image.size - len(moffat_fit.parameters)
        reduced_chi2 = chi2 / dof

        moffat_results = PropertyList()
        for name, value in zip(moffat_fit.param_names, moffat_fit.parameters):
            moffat_results.setDouble(f"MOFFAT_{name.upper()}", float(value))
        moffat_results.setDouble("MOFFAT_CHI2", float(chi2))
        moffat_results.setDouble("MOFFAT_REDUCED_CHI2", float(reduced_chi2))

        self.log.info(
            "Extended PSF Moffat fit results: x_0=%.2f, y_0=%.2f, gamma=%.2f, alpha=%.2f, reduced_chi2=%.2f",
            moffat_fit.x_0.value,
            moffat_fit.y_0.value,
            moffat_fit.gamma.value,
            moffat_fit.alpha.value,
            reduced_chi2,
        )

        return moffat_results
