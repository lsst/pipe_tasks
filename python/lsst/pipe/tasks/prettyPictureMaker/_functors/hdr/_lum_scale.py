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

from __future__ import annotations

__all__ = ("HDRLumCompressor",)

import skimage
import numpy as np
import logging
from functools import partial
from scipy.optimize import brentq


from lsst.pipe.tasks.prettyPictureMaker.types import FloatImagePlane, WhitePoint
from lsst.pex.config.configurableActions import ConfigurableAction
from lsst.pex.config import Field, ListField
from lsst.rubinoxide import rgb

from ..._equalizers import contrast_equalizer, tone_equalizer

logger = logging.getLogger(__name__)


def hdr_lum_scale(
    intensities: FloatImagePlane | float,
    turn_over: float,
    exp_growth_factor: float,
    arcsinh_scale: float,
    width: float = 1,
) -> FloatImagePlane | float:
    """
    Spliced function: arcsinh(x) for x <= cutoff,
    and a smooth exponential function A*e^(b*x) + C for x > cutoff.

    Parameters
    ----------
    intensities : `FloatImagePlane` or `float`
        The intensity values to scale
    turn_over : `float`
        The value at which the function turns over from arcsinh to exp
    exp_growth_factor : `float`
        Growth parameter (b > 0). Higher values cause faster growth.
    width : `float`
        Controls the smoothness of the transition. Smaller values make
        the transition sharper.

    Returns
    -------
    scaled_intensities : `FloatImagePlane` or `float`
        The scaled intensity(s)

    """

    # 1. Define the Arcsinh component
    # f_arc(x) = A * arcsinh(k * x)
    f_arc = np.arcsinh(arcsinh_scale * intensities)

    # Calculate value and derivative of arcsinh at the transition point
    # Derivative of arcsinh(u) is 1 / sqrt(1 + u^2) * du/dx
    # Here u = k*x, so du/dx = k
    val_at_x0 = np.arcsinh(arcsinh_scale * turn_over)
    deriv_at_x0 = arcsinh_scale / np.sqrt(1 + (arcsinh_scale * turn_over) ** 2)

    # 2. Define the Exponential component
    # We construct an exponential that matches the value and slope at x0
    f_exp = val_at_x0 + (deriv_at_x0 / exp_growth_factor) * (
        np.exp(exp_growth_factor * (intensities - turn_over)) - 1
    )

    # 3. Define the Smooth Step Function (Sigmoid)
    # S(x) goes from 0 to 1 around x0
    S = 1.0 / (1.0 + np.exp(-(intensities - turn_over) / width))

    # 4. Blend the two functions
    # When x << x0, S -> 0, result -> f_arc
    # When x >> x0, S -> 1, result -> f_exp
    scaled_intensities = (1 - S) * f_arc + S * f_exp

    return scaled_intensities


def smooth_arcsinh_quadratic(x, x0, floor, max_nits, arc_scale=1.0, x_end=1.0):
    """
    Creates a function that transitions from a scaled arcsinh to a quadratic.

    The transition is exact: the quadratic matches both the value and the slope
    of the arcsinh at x0. The switch is sharp, using np.where, which eliminates
    any dipping artifacts caused by sigmoid blending.

    Parameters:
    -----------
    x : array_like
        Input values.
    x0 : float
        The turnover point where the transition occurs.
    max_nits : float
        Scaling parameter. The target value at x_end is (max_nits/100) * value_at_x0.
    arc_scale : float
        Scaling factor for the input of the arcsinh function (arcsinh(x * arc_scale)).
    x_end : float
        The x-coordinate where the quadratic target value is enforced.

    Returns:
    --------
    y : ndarray
        The calculated function values.
    """
    x = np.asarray(x)

    # 1. Scaling Factor for the target height
    nit_scale = max_nits / 100.0
    scale_factor = (np.arcsinh(x0 * arc_scale) - np.arcsinh(floor * arc_scale)) * nit_scale

    # 2. Define the Scaled Arcsinh Component
    arg = x * arc_scale
    f_arc = np.arcsinh(arg)

    # Calculate value and derivative at the turnover point x0
    y_at_x0 = np.arcsinh(x0 * arc_scale)

    # Derivative: d/dx [arcsinh(u)] = 1/sqrt(1+u^2) * du/dx
    # Here u = x * arc_scale, so du/dx = arc_scale
    slope_at_x0 = arc_scale / np.sqrt(1 + (x0 * arc_scale) ** 2)

    # Target endpoint at x_end
    y_at_end = scale_factor

    # 3. Solve for the Quadratic: y = a*x^2 + b*x + c
    # We have 3 constraints for 3 unknowns (a, b, c):
    # Eq 1 (Value at x0): a*x0^2 + b*x0 + c = y_at_x0
    # Eq 2 (Slope at x0): 2*a*x0 + b = slope_at_x0
    # Eq 3 (Value at x_end): a*x_end^2 + b*x_end + c = y_at_end

    # From Eq 2, we can express b in terms of a:
    # b = slope_at_x0 - 2*a*x0

    # Substitute b into Eq 1 to express c in terms of a:
    # c = y_at_x0 - a*x0^2 - b*x0
    # c = y_at_x0 - a*x0^2 - (slope_at_x0 - 2*a*x0)*x0
    # c = y_at_x0 - a*x0^2 - slope_at_x0*x0 + 2*a*x0^2
    # c = y_at_x0 - slope_at_x0*x0 + a*x0^2

    # Now substitute b and c into Eq 3 to solve for a:
    # a*x_end^2 + (slope_at_x0 - 2*a*x0)*x_end + (y_at_x0 - slope_at_x0*x0 + a*x0^2) = y_at_end

    # Group terms by 'a':
    # a * (x_end^2 - 2*x0*x_end + x0^2) = y_at_end - y_at_x0 - slope_at_x0*x_end + slope_at_x0*x0
    # a * (x_end - x0)^2 = (y_at_end - y_at_x0) - slope_at_x0 * (x_end - x0)

    dx = x_end - x0
    if np.isclose(dx, 0):
        raise ValueError("x_end cannot be equal to x0")

    a = ((y_at_end - y_at_x0) - slope_at_x0 * dx) / (dx**2)

    # Now solve for b and c using the calculated 'a'
    b = slope_at_x0 - 2 * a * x0
    c = y_at_x0 - a * x0**2 - b * x0

    # Define the Quadratic function
    def f_quad(val):
        return a * (val**2) + b * val + c

    # 4. Create the final function using np.where for a sharp transition
    # This eliminates the dipping artifact from sigmoid blending.
    y = np.where(x < x0, f_arc, f_quad(x))

    return y


def smooth_arcsinh_exponential(x, x0, floor, max_nits, arc_scale=1.0, x_end=1.0, k_guess=1.0):
    """
    Creates a function that transitions from a scaled arcsinh to an exponential.

    The transition is exact: the exponential matches both the value and the slope
    of the arcsinh at x0. The switch is sharp, using np.where.

    The growth rate 'k' is solved numerically to ensure the exponential hits the
    target value at x_end.

    Parameters:
    -----------
    x : array_like
        Input values.
    x0 : float
        The turnover point where the transition occurs.
    floor : float
        The floor value for the scaling calculation (same as in your quadratic version).
    max_nits : float
        Scaling parameter. The target value at x_end is (max_nits/100) * (arcsinh(x0) - arcsinh(floor)).
    arc_scale : float
        Scaling factor for the input of the arcsinh function (arcsinh(x * arc_scale)).
    x_end : float
        The x-coordinate where the exponential target value is enforced.
    k_guess : float
        Initial guess for the growth rate solver.

    Returns:
    --------
    y : ndarray
        The calculated function values.
    """
    x = np.asarray(x)

    # 1. Scaling Factor for the target height (Same as your quadratic version)
    nit_scale = max_nits / 100.0
    scale_factor = (np.arcsinh(x0 * arc_scale) - np.arcsinh(floor * arc_scale)) * nit_scale

    # 2. Define the Scaled Arcsinh Component
    arg = x * arc_scale
    f_arc = np.arcsinh(arg)

    # Calculate value and derivative at the turnover point x0
    y_at_x0 = np.arcsinh(x0 * arc_scale)

    # Derivative: d/dx [arcsinh(u)] = 1/sqrt(1+u^2) * du/dx
    # Here u = x * arc_scale, so du/dx = arc_scale
    slope_at_x0 = arc_scale / np.sqrt(1 + (x0 * arc_scale) ** 2)

    # Target endpoint at x_end
    y_at_end = scale_factor

    # 3. Solve for the Exponential: Q(x) = A * exp(k * (x - x0)) + B
    # We have 3 unknowns (A, B, k) and 3 constraints:
    # Eq 1 (Value at x0): A * exp(0) + B = y_at_x0  =>  A + B = y_at_x0
    # Eq 2 (Slope at x0): A * k * exp(0) = slope_at_x0  =>  A * k = slope_at_x0
    # Eq 3 (Value at x_end): A * exp(k * (x_end - x0)) + B = y_at_end

    # From Eq 1 and Eq 2:
    # B = y_at_x0 - A
    # A = slope_at_x0 / k
    # So, B = y_at_x0 - (slope_at_x0 / k)

    # Substitute A and B into Eq 3:
    # (slope_at_x0 / k) * exp(k * dx) + (y_at_x0 - slope_at_x0 / k) = y_at_end
    # (slope_at_x0 / k) * (exp(k * dx) - 1) = y_at_end - y_at_x0

    dx = x_end - x0
    if np.isclose(dx, 0):
        raise ValueError("x_end cannot be equal to x0")

    # We need to solve for k in:
    # (slope_at_x0 / k) * (exp(k * dx) - 1) - (y_at_end - y_at_x0) = 0

    def equation_to_solve(k):
        if k == 0:
            return 0  # Limit as k->0 is slope_at_x0 * dx, which is the linear case
        return (slope_at_x0 / k) * (np.exp(k * dx) - 1) - (y_at_end - y_at_x0)

    # Find the root for k
    # We need to provide a reasonable range for k.
    # If the target is higher than the start, k should be positive.
    # If the target is lower, k should be negative.
    if y_at_end > y_at_x0:
        # Target is higher, so we expect positive growth
        k_min = 1e-6
        k_max = 100.0  # Upper bound, adjust if needed
        # Ensure the function changes sign in the interval
        if equation_to_solve(k_max) < 0:
            # If even at k_max the function is negative, we need a larger max
            k_max = 1000.0
        try:
            k = brentq(equation_to_solve, k_min, k_max)
        except ValueError:
            # Fallback if root finding fails
            k = k_guess
    else:
        # Target is lower, so we expect negative growth (decay)
        k_min = -100.0
        k_max = -1e-6
        try:
            k = brentq(equation_to_solve, k_min, k_max)
        except ValueError:
            k = k_guess

    # Now solve for A and B using the found k
    A = slope_at_x0 / k
    B = y_at_x0 - A

    # Define the Exponential function
    def f_exp(val):
        return A * np.exp(k * (val - x0)) + B

    # 4. Create the final function using np.where for a sharp transition
    y = np.where(x < x0, f_arc, f_exp(x))

    return y


def smooth_arcsinh_exponential_controlled(
    x, x0, floor, max_nits, growth_rate, arc_scale=1.0, x_end=1.0, width=0.05
):
    """
    Creates a function transitioning from arcsinh to an exponential with controlled growth.

    Constraints:
    1. Value at x0 matches arcsinh.
    2. Value at x_end matches target.
    3. Growth rate 'k' is fixed by the user.

    The slope at x0 is NOT constrained (it will be whatever is needed to fit).
    This eliminates the need for numerical solvers and gives full control.
    """
    x = np.asarray(x)

    # 1. Scaling Factor (Same as your quadratic version)
    nit_scale = max_nits / 100.0
    scale_factor = (np.arcsinh(x0 * arc_scale) - np.arcsinh(floor * arc_scale)) * nit_scale

    # 2. Define the Scaled Arcsinh Component
    arg = x * arc_scale
    f_arc = np.arcsinh(arg)

    # Values at key points
    y_at_x0 = np.arcsinh(x0 * arc_scale)
    y_at_end = scale_factor

    # 3. Solve for Exponential: Q(x) = A * exp(k * (x - x0)) + B
    # We have 3 unknowns (A, B) and 2 equations (plus fixed k).
    # Eq 1: Q(x0) = A * exp(0) + B = y_at_x0  =>  A + B = y_at_x0
    # Eq 2: Q(x_end) = A * exp(k * dx) + B = y_at_end

    k = growth_rate
    dx = x_end - x0

    if np.isclose(dx, 0):
        raise ValueError("x_end cannot be equal to x0")
    if np.isclose(k, 0):
        raise ValueError("Growth rate cannot be zero for this formulation (use linear)")

    # From Eq 1: B = y_at_x0 - A
    # Substitute into Eq 2:
    # A * exp(k * dx) + (y_at_x0 - A) = y_at_end
    # A * (exp(k * dx) - 1) = y_at_end - y_at_x0

    # Solve for A:
    denom = np.exp(k * dx) - 1
    A = (y_at_end - y_at_x0) / denom

    # Solve for B:
    B = y_at_x0 - A

    # Define the Exponential function
    def f_exp(val):
        return A * np.exp(k * (val - x0)) + B

    S = 1.0 / (1.0 + np.exp(-(x - x0) / width))

    # 4. Blend the two functions
    # When x << x0, S -> 0, result -> f_arc
    # When x >> x0, S -> 1, result -> f_exp
    scaled_intensities = (1 - S) * f_arc + S * f_exp(x)
    return scaled_intensities

    # 4. Create the final function using np.where for a sharp transition
    y = np.where(x < x0, f_arc, f_exp(x))

    return y


class HDRLumCompressor(ConfigurableAction):
    """Compress and enhance luminance using multi-stage processing.

    This class implements luminance compression for RGB image generation using
    a multi-stage algorithm that includes:

    - Asinh stretching for non-linear brightness mapping
    - Linear contrast manipulation
    - Midtone adjustment
    - Optional Gaussian denoising
    - Optional contrast equalization
    - Optional tone adjustment

    The configuration fields control the parameters for each stage of the
    processing pipeline.
    """

    arcsinh_stretch = Field[float](doc="The stretch of the luminance in asinh", default=400)
    exp_stretch = Field[float](doc="Growth factor for exponential part of curve", default=20)
    blend_width = Field[float](doc="Blending width between arcsinh and exp", default=0.2)
    max = Field[float](doc="The maximum allowed luminance on a 0 to inf scale", default=10)
    floor = Field[float](
        doc="A value scaled flux units (i.e. rgb vlaue) that is used to map luminances to a very dark gray",
        default=0.0,
    )
    reference_white = Field[float](
        doc=(
            "The value scaled flux units (i.e. rgb value) that should correspond to a value of 1, "
            "roughly 100 nits in an hdr image"
        ),
        default=1,
    )
    Q = Field[float](
        doc="softening parameter",
        default=0.7,
        deprecated="This field is no longer used and will be removed after v31.",
    )
    highlight = Field[float](
        doc="The value of highlights in scaling factor applied to post asinh streaching", default=1.0
    )
    shadow = Field[float](
        doc="The value of shadows in scaling factor applied to post asinh streaching", default=0.0
    )
    midtone = Field[float](
        doc="The value of midtone in scaling factor applied to post asinh streaching", default=0.5
    )
    equalizerLevels = ListField[float](
        doc=(
            "A list of factors to modify the constrast in a scale-dependent way. "
            "One coefficient for each spatial scale, starting from the largest. "
            "Values large than 1 increase contrast, while less than 1 decreases. "
            "This adjustment is multiplicative. "
            "Only scales upto and including the largest to be modified need specified, "
            "i.e. [1.1,0.9] modifies the first two [1.1,1,0.9] modifies the first three."
        ),
        optional=True,
    )
    toneAdjustment = ListField[float](
        doc=(
            "A list of length 10 that adjusts the brightness of the image ranging "
            "from dark regions to light. These 10 values represent control points along "
            "the lumanance interval 0-1, but the actual adjustments made are continuous "
            "and are calculated from these control points."
        ),
        length=10,
        optional=True,
    )
    toneWidth = Field[float](
        doc=(
            "This parameters controls how each tone control point affect the adjustment "
            "of the values in between. Increase the value to have a more continuous "
            "change between control points, decrease to make the control sharper. Value "
            "must be greater than zero."
        ),
        default=0.07,
    )
    doDenoise = Field[bool](doc="Denoise the luminance image", default=False)

    def __call__(self, intensities: FloatImagePlane, white_point: WhitePoint) -> FloatImagePlane:
        """Compress and enhance luminance using multi-stage processing.

        This method applies the configured luminance compression algorithm to
        the input image. The processing pipeline includes optional denoising,
        asinh stretching, linear contrast manipulation, midtone adjustment,
        contrast equalization, and tone adjustment.

        Parameters
        ----------
        intensities : `FloatImagePlane`
            Input image with pixel intensities. This FloatImagePlane should
            contain the luminance data to be compressed.
        white_point : `WhitePoint`
            The cie white point from which the intensities are derived.

        Returns
        -------
        result : `FloatImagePlane`
            The processed image with luminance compression applied. Values
            are clipped to the range [0, 1].

        Notes
        -----
        The processing pipeline consists of the following stages:

        1. Optional wavelet denoising if doDenoise is True
        2. Asinh stretching with configurable stretch parameter
        3. Linear contrast adjustment using highlight, shadow parameters
        4. Midtone adjustment using midtone parameter
        5. Optional contrast equalization if equalizerLevels is configured
        6. Optional tone adjustment if toneAdjustment is configured
        7. Final clipping to [0, 1] range

        The configuration fields (stretch, highlight, shadow, midtone,
        equalizerLevels, toneAdjustment, toneWidth) control the behavior
        of each processing stage.
        """
        if self.doDenoise:
            intensities = skimage.restoration.denoise_wavelet(intensities)

        # Scale the values with linear manipulation for contrast
        intensities = abs(intensities)
        dark_gray_to_lum = rgb.RGB_to_Oklab(
            np.array([[[self.floor, self.floor, self.floor]]], dtype=float), white_point
        )[0, 0, 0]

        white_to_lum = rgb.RGB_to_Oklab(
            np.array([[[self.reference_white, self.reference_white, self.reference_white]]], dtype=float),
            white_point,
        )[0, 0, 0]

        # lum_scaler = partial(
        #     hdr_lum_scale,
        #     turn_over=white_to_lum,
        #     exp_growth_factor=self.exp_stretch,
        #     arcsinh_scale=self.arcsinh_stretch,
        #     width=self.blend_width,
        # )
        lum_scaler = partial(
            # smooth_arcsinh_quadratic,
            smooth_arcsinh_exponential,
            x0=white_to_lum,
            floor=dark_gray_to_lum,
            max_nits=1000,
            # growth_rate=self.exp_stretch,
            # growth_factor=self.exp_stretch,
            arc_scale=self.arcsinh_stretch,
            # width=self.blend_width,
            x_end=1,
        )

        top = lum_scaler(white_to_lum)
        bottom = (lum_scaler(dark_gray_to_lum) - 0.2 * top) / 0.8
        print(dark_gray_to_lum, white_to_lum)
        intensities = (lum_scaler(intensities) - bottom) / (top - bottom)
        logger.debug("arcsinh max %.4f", intensities.max())
        # intensities = (intensities - self.shadow) / ((self.highlight) - self.shadow)
        logger.debug("post lin stretch max %.4f", intensities.max())
        intensities = ((self.midtone - 1) * intensities) / (
            ((2 * self.midtone - 1) * intensities) - self.midtone
        )
        logger.debug("midtone adjustment max %.4f", intensities.max())

        if self.equalizerLevels is not None:
            intensities = contrast_equalizer(intensities, self.equalizerLevels)
            logger.debug("equalizer max %.4f", intensities.max())

        if self.toneAdjustment is not None:
            intensities = np.clip(intensities, 0, self.max)
            intensities = tone_equalizer(intensities, self.toneAdjustment, self.toneWidth, 10, 5)

        intensities = np.clip(intensities, 0, self.max)

        return intensities
