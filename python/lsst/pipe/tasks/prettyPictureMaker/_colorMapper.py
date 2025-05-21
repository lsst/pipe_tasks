from __future__ import annotations
__all__ = ("mapUpperBounds", "latLum", "colorConstantSat", "lsstRGB")

from lsst.meas.algorithms import SubtractBackgroundTask
import logging
import numpy as np
import skimage
import colour
import cv2
from skimage.restoration import inpaint_biharmonic
import pywt

from ._localContrast import localContrast, makeGaussianPyramid, makeLapPyramid, levelPadder
from lsst.cpputils import fixGamutOK

from numpy.typing import NDArray
from typing import Callable, Mapping, Literal

import numpy as np
from scipy.ndimage import gaussian_filter, zoom, label, binary_dilation
from scipy.interpolate import interp2d, RBFInterpolator, LinearNDInterpolator
from scipy.stats import norm
from . import oklab_rgb
from lsst.afw.image import MaskedImageF, Mask, ExposureF


from itertools import chain, combinations_with_replacement
from typing import Iterable, Tuple

import numpy as np


class PolynomialFeatures:
    """Generate polynomial features

    For each feature vector, it generates polynomial features of degree d consisting
    of all polynomial combinations of the features with degree less than or equal to
    the specified degree.

    For example, if an input sample is two dimensional and of the form
    [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

    With degree 1, it simply transform to homogenous coordinates (adding a constant 1 to the features)

    See PolynomialFeatures from scikit-learn for a complete documentation and implementation.
    """

    def __init__(self, degree=1):
        self.degree = degree
        self._fitted = False
        self.n_input_features = 0
        self.n_output_features = 0

    @staticmethod
    def _combinations(n_features: int, degree: int) -> Iterable[Tuple[int, ...]]:
        return chain.from_iterable(combinations_with_replacement(range(n_features), i) for i in range(degree + 1))

    @staticmethod
    def _combinations_length(n_features: int, degree: int) -> int:
        # TODO: Can be computed mathematically
        return sum(1 for _ in PolynomialFeatures._combinations(n_features, degree))

    def fit(self, X: np.ndarray):
        """Compute number of output features."""
        _, n_features = X.shape
        self.n_input_features = n_features
        self.n_output_features = self._combinations_length(n_features, self.degree)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray):
        """Transform features to polynomial features

        Args:
            X (np.ndarray): Features for multiple samples
                Shape: (n_samples, n_features)

        Returns
            np.ndarrray: Polynomial features
                Shape: (n_samples, n_output_features)
        """
        assert self._fitted, "Please first fit the PolynomialFeatures"

        n_samples, n_features = X.shape

        if n_features != self.n_input_features:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        polynomial_features = np.empty((n_samples, self.n_output_features), dtype=X.dtype)

        combinations = self._combinations(n_features, self.degree)
        for i, c in enumerate(combinations):
            polynomial_features[:, i] = X[:, c].prod(1)

        return polynomial_features

from scipy.spatial.distance import cdist  # type: ignore

class ThinPlateSpline:
    """Solve the Thin Plate Spline interpolation.

    In fact, this class supports the more general multi-dimensional polyharmonic spline interpolation.

    It learns a non-linear elastic mapping function f: R^d -> R given n control points (x_i, y_i)_i
    where x_i \\in R^d and y_i \\in R.

    The problem is formally stated as:
    f = min_f E_{fit}(f) + \\alpha E_{reg}(f)      (1)

    where E_{fit}(f) = \\sum_{i=1}^n (y_i - f(x_i))^2  (Least square objectif)
    and E_{reg}(f) = \\int \\|\\nabla^{order} f \\|_2^2 dx (Regularization)
    which penalizes the norm of the order-th derivatives of the learnt function.

    When the regularization alpha is 0, then this becomes equivalent to minimizing E_{reg}
    under the constraints f(x_i) = y_i.

    NOTE: In the classical 2 dimensionnal TPS, the spline order is 2.

    Using Euler-Lagrange, one can show that the function solutions are of the form:
    f(x) = P(x) + \\sum_i w_i G(\\|x - x_i\\|_2), with P a polynom of degree = order - 1
    and G are radial basis functions (RBF).

    It can be shown that the RBF follows:

    G(r) = r**(2 * order - d) log(r) for even dimension d
    or
    G(r) = r**(2 * order - d) for odd dimension d

    For TPS, this sets G(r) = r**2 log(r)
    As RBF, are not always defined in r = 0 (if 2 * order \\le d), we decided to use the common
    (but incorrect) TPS kernel instead of the mathematical one.

    Fitting the spline, amounts at finding the polynoms coefficients and the weights (w_i).

    Vectorized solution:
    --------------------

    Let X = (x_i) \\in R^{n x d} and Y = (y_i) \\in R^{n x v} be the control points and associated values.
    And X' = (x'_i) \\in R^{n' x d} be any set of points where we want to interpolate f.

    Note that we now have several values to interpolate for each position, which simply amounts at finding
    a mapping f_j for each value and concatenating them.

    X and X' can expended as its polynomial features X_p \\in R^{n x d_p} and X'_p \\in R^{n' x d_p}.
    For instance, TPS, the polynom is of degree one and the polynomial extension is simply the homogenous
    coordinates: X_p = (1, X). See PolynomialFeatures for more information.

    Let's call K(X') \\in R^{n' x n} the matrix of RBF values for any X'. K(X') = G(cdist(X', X)): it is the
    RBF applied to the radial distances to the control points.

    The polynoms coefficients can be vectorized into a matrix C \\in R^{d_p x v} (one polynom for each value)
    and finally we can also vectorize the RBF weights into W \\in R^{n x v} (One weight for each control point
    and each value).

    The function f can now be vectorized as:
    f(X') = X'_p C + K(X') W

    Learning the parameters C and W is done by solving the following system with the control points X:
            A      .   P   =   B
                          <=>
    |  K(X) , X_p |  | W |   |Y |
    |             |  |   | = |  |       (2)
    | X_p^T ,  0  |  | C |   |0 |

    The first row enforces f(X) = Y and the second adds orthogonality constraints.
    The system can be relaxed when \\alpha != 0, then one can show that same system can be solved
    by simply replacing K(X) by K(X) + alpha I.

    To transform X', one can simply apply the learned parameters P = (W, C) with
    Y' = f(X') = X'_p C + K(X') W

    In our notations, we have:
    A \\in R^{(n + d_p) x (n + d_p)}
    P \\in R^{(n + d_p) x v}
    Y \\in R^{(n + d_p) x v}

    Attrs:
        alpha (float): Regularization parameter
            Default: 0.0 (The mapping will enforce f(X) = Y)
        order (int): Order of the spline (minimizes the squared norm of the order-th derivatives)
            Default: 2 (Suited for TPS interpolation)
        enforce_tps_kernel (bool): Always use the RBF G(r) = r**2 log(r).
            This should be sub-optimal, but it yields good results in practice.
            Default: False
        parameters (ndarray): All the parameters P = (W, C). Shape: (n + d_p, v)
        control_points (ndarray): Control points fitted (the last X given to fit). Shape: (n, d)
    """

    eps = 0  # Relaxed orthogonality constraints by setting it to 1e-6 ?

    def __init__(self, alpha=0.0, order=2, enforce_tps_kernel=False) -> None:
        self._fitted = False
        self._polynomial_features = PolynomialFeatures(order - 1)

        self.alpha = alpha
        self.order = order
        self.enforce_tps_kernel = enforce_tps_kernel

        self.parameters = np.array([], dtype=np.float32)
        self.control_points = np.array([], dtype=np.float32)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> ThinPlateSpline:
        """Learn the non-linear elastic mapping f that matches Y given X

        It solves equation (2) to find the spline parameters.

        Args:
            X (ndarray): Control points
                Shape: (n, d)
            Y (ndarray): Values for these controls points
                Shape: (n, v)

        Returns:
            ThinPlateSpline: self
        """
        X = _ensure_2d(X)
        Y = _ensure_2d(Y)
        assert X.shape[0] == Y.shape[0]

        n, _ = X.shape  # (n, d)
        self.control_points = X

        # Compute radial distances
        phi = self._radial_distance(X)  # (n, n) (phi = K(X))

        # Polynomial expansion
        X_p = self._polynomial_features.fit(X).transform(X)  # (n, d_p)

        # Assemble system A P = B
        # A = |K + alpha I, X_p|
        #     |     X_p.T ,  0 |
        A = np.vstack(  # ((n + d_p), (n + d_p))
            [
                np.hstack([phi + self.alpha * np.eye(n, dtype=X.dtype), X_p]),  # (n, (n + d_p))
                np.hstack([X_p.T, self.eps * np.eye(X_p.shape[1], dtype=X.dtype)]),  # (d_p, (n + d_p))
            ]
        )

        # B = | Y |
        #     | 0 |
        B = np.vstack([Y, np.zeros((X_p.shape[1], Y.shape[1]), dtype=X.dtype)])  # ((n + d_p), v)

        self.parameters = np.linalg.solve(A, B)  # ((n + d_p), v)
        self._fitted = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the mapping to X (interpolate)

        It returns f(X)

        Args:
            X (ndarray): Set of points to interpolate
                Shape: (n', d)

        Returns:
            ndarray: Interpolated values of the set of points
                Shape: (n', v)

        """
        assert self._fitted, "Needs to be fitted first."

        X = _ensure_2d(X)
        assert X.shape[1] == self.control_points.shape[1], "The number of features do not match training data"

        # Compute radial distances
        phi = self._radial_distance(X)  # (n', n)

        # Polynomial expansion
        X_p = self._polynomial_features.transform(X)  # (n', d_p)

        # Compute f(X)
        X_aug = np.hstack([phi, X_p])  # (n', (n + d_p))
        return X_aug @ self.parameters  # (n', v)

    def _radial_distance(self, X: np.ndarray) -> np.ndarray:
        """Compute the pairwise RBF values for the given points to the control points

        Args:
            X (ndarray): Points to be interpolated
                Shape: (n', d)

        Returns:
            ndarray: The RBF evaluated for each point from a control point (K(X))
                Shape: (n', n)
        """
        dist = cdist(X, self.control_points).astype(X.dtype)

        power = self.order * 2 - self.control_points.shape[-1]

        # As negatif power leads to ill-defined RBF at r = 0
        # We use the TPS RBF r^2 log r, though it is not optimal
        if power <= 0 or self.enforce_tps_kernel:
            power = 2  # Use r^2 log r (TPS kernel)

        if power % 2:  # Odd
            return dist**power

        # Even
        dist[dist == 0] = 1  # phi(r) = r^power log(r) ->  (phi(0) = 0)
        return dist**power * np.log(dist)


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    """Ensure that array is a 2d array

    In case of 1d array, let's expand the last dim
    """
    assert array.ndim in (1, 2)

    # Expand last dim in order to interpret this as (n, 1) points
    if array.ndim == 1:
        array = array[:, None]

    return array



def eigf_variance_analysis_no_mask(guide, sigma):
    """
    Computes average and variance of guide using Gaussian filtering.

    Parameters:
        guide (numpy.ndarray): 2D array representing the guide image.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        numpy.ndarray: Array where each pixel has [average, variance].
    """
    # Compute average of guide
    mu_guide = gaussian_filter(guide, sigma=sigma)

    # Compute average of squared guide values
    guide_squared = guide**2
    mu_guide_squared = gaussian_filter(guide_squared, sigma=sigma)

    # Calculate variance as E[guide^2] - (E[guide])^2
    var_guide = mu_guide_squared - mu_guide**2

    # Combine into an output array with shape (height, width, 2)
    output = np.stack((mu_guide, var_guide), axis=2)

    return output


def eigf_blending_no_mask(image, av, feathering, filter_type):
    """
    Applies blending without a mask using averages and variances.

    Parameters:
        image (numpy.ndarray): 2D input image array.
        av (numpy.ndarray): Array with shape (height*width, 2) containing averages and variances.
        feathering (float): Feathering parameter for blending.
        filter_type (int): Blending type: 0 for linear, 1 for geometric mean.
    """
    # Reshape 'av' to match image dimensions
    av_reshaped = av.reshape(image.shape[0], image.shape[1], -1)

    avg_g = av_reshaped[..., 0]
    var_g = av_reshaped[..., 1]

    norm_g = np.maximum(avg_g * image, 1e-6)
    normalized_var_guide = var_g / norm_g

    a = normalized_var_guide / (normalized_var_guide + feathering)
    b = avg_g - a * avg_g

    # Apply blending
    if filter_type == 0:  # Linear blending
        image[:] = np.maximum(image * a + b, np.finfo(float).min)
    else:  # Geometric mean blending
        image[:] *= np.maximum(image * a + b, np.finfo(float).min)
        image[:] = np.sqrt(image[:])


def fast_eigf_surface_blur(image, sigma, feathering, iterations=1, filter_type=1):
    """
    Applies exposure-independent guided blur with down-scaling and up-sampling.

    Parameters:
        image (numpy.ndarray): Input image array of shape (height, width).
        sigma (float): Standard deviation for Gaussian kernel.
        feathering (float): Feathering parameter.
        iterations (int): Number of iterations to model diffusion.
        filter_type (int): Blending type: 0 for linear, 1 for geometric mean.
    """
    scaling = np.maximum(np.minimum(sigma, 4.0), 1.0)
    ds_sigma = np.maximum(sigma / scaling, 1.0)

    # Down-sampling dimensions

    for _ in range(iterations):
        av = eigf_variance_analysis_no_mask(image, ds_sigma)
        eigf_blending_no_mask(image, av.reshape(-1, 2), feathering, filter_type)


def tone_equalizer(image, tone_factors, weight, sigma, feathering, iterations=1, filter_type=1):
    # need to blur the input luminance image
    luminance = np.copy(image)
    fast_eigf_surface_blur(luminance, sigma, feathering, iterations, filter_type)
    exposure = luminance
    corrections = np.zeros_like(luminance)
    EXPOSURE_CENTERS = np.linspace(0, 1, 10)
    for eq_val, factor in zip(EXPOSURE_CENTERS, tone_factors):
        corrections += np.exp(-1 * (exposure - eq_val) ** 2 / (2 * weight**2)) * factor
    return image + corrections


def contrast_equalizer(image, contrast_factors):
    maxLevel = int(np.min(np.log2(image.shape)))
    support = 1 << (maxLevel - 1)
    padY_amounts = levelPadder(image.shape[0] + support, maxLevel)
    padX_amounts = levelPadder(image.shape[1] + support, maxLevel)
    imagePadded = cv2.copyMakeBorder(
        image, *(0, support), *(0, support), cv2.BORDER_REPLICATE, None, None
    ).astype(image.dtype)
    lap = makeLapPyramid(imagePadded, padY_amounts, padX_amounts, None, None)
    for i, factor in enumerate(contrast_factors):
        # negative indexing so +1, +1 to skip the highest pyramid
        i = i + 2
        if i > len(lap):
            break
        lap[-1 * i] *= factor
    output = lap[-1]
    for i in range(-2, -1 * len(lap) - 1, -1):
        upsampled = cv2.pyrUp(output)
        upsampled = upsampled[
            : upsampled.shape[0] - 2 * padY_amounts[i + 1], : upsampled.shape[1] - 2 * padX_amounts[i + 1]
        ]
        output = lap[i] + upsampled
    return output[: image.shape[0], : image.shape[1]]


from scipy.stats import skewnorm


def skew(lum, amp, k, loc, scale):
    # good params 0.27,7,0.1,0.39
    return lum + amp * skewnorm.pdf(lum, k, loc, scale)


def lumScale(
    values,
    highlight,
    shadow,
    midtone,
    equalizer_levels,
    tone_adjustment,
    tone_width=0.07,
    max=1,
    floor=0,
    stretch=400,
    doDenoise=False,
    brackets=None,
    localContrastArgs=None,
):
    if doDenoise:
        values = skimage.restoration.denoise_wavelet(values)
        values = abs(values)
    stack = []
    if brackets is None:
        brackets = [1]
    for bracket in brackets:
        intensities = values / bracket
        # Scale the values with linear manipulation for contrast
        intensities = abs(intensities)
        nj_to_lum = oklab_rgb.RGB_to_Oklab(np.array([[[floor, floor, floor]]], dtype=float), (0.28,0.28))[0,0,0]
        top = np.arcsinh(stretch)
        bottom = (np.arcsinh(nj_to_lum*stretch) - 0.2*top)/0.8
        intensities = (np.arcsinh((intensities) * stretch) - bottom) / (top - bottom)
        intensities = np.clip(intensities, 0, 1)
        # intensities = abs(intensities)
        intensities = (intensities - shadow) / ((highlight) - shadow)
        intensities = ((midtone - 1) * intensities) / (((2 * midtone - 1) * intensities) - midtone)
        intensities = np.clip(intensities, 0, max)

        if equalizer_levels is not None:
            intensities = contrast_equalizer(intensities, equalizer_levels)
        if localContrastArgs:
            intensities = localContrast(intensities, **localContrastArgs)
        stack.append(intensities)
    if len(stack) == 1:
        intensities = stack[0]
    else:
        intensities = _fuseExposureLum(stack)

    if tone_adjustment is not None:
        if len(tone_adjustment) != 10:
            raise ValueError("Tone adjustment must be given by a len 10 sequence")
        intensities = np.clip(intensities, 0, max)
        intensities = tone_equalizer(intensities, tone_adjustment, tone_width, 10, 5)

    return np.clip(intensities, 0, max)


def latLum(
    values,
    stretch: float = 400,
    max: float = 1,
    floor: float = 0.00,
    Q: float = 0.7,
    doDenoise: bool = False,
    highlight: float = 1.0,
    shadow: float = 0.0,
    midtone: float = 0.5,
    equalizer_levels: list[float] | None = None,
) -> NDArray:
    """
    Scale the input luminosity values to maximize the dynamic range visible.

    Parameters
    ----------
    values : `NDArray`
        The input image luminosity data of of.
    stretch : `float`, optional
        A parameter for the arcsinh function.
    max : `float`, optional
        Maximum value for intensity scaling on a scale of 0-1.
    floor : `float`, optional
        A value added to each pixel in arcsinh transform, this ensures values in
        the arcsinh transform is no smaller than the supplied value.
    Q : `float`, optional
        Another parameter for the arcsinh function and scaling factor for
        softening.
    doDenoise : `bool`, optional
        Denoise the image if desired.
    highlight : `float`
        This is the value (between 0 and 1) that maps to be "white" in the
        output. Decreasing this makes fainter things more luminous but
        clips the brightest end. This is a linear transform applied to the
        values after arcsinh.
    shadow : `float`
        This is the value (between 0 and 1) that maps to be "black" in the
        output. Increasing this makes fainter things darker but
        clips the lowest values. This is a linear transform applied to the
        values after arcsinh.
    midtone : `float`
        This is the value (between 0 and 1) that adjusts the balance between
        white and black. Decreasing this makes fainter things more luminous,
        increasing does the opposite. This is a linear transform applied to the
        values after arcsinh.

    Returns:
        luminance : `NDArray`
            The stretched luminosity data.
    """

    # De-noise the input image using wavelet de-noising.
    if doDenoise:
        values = skimage.restoration.denoise_wavelet(values)
        values = abs(values)

    # Scale values from 0-1 to 0-100 as various algorithm expect that range.
    # values *= 100

    # Calculate the slope for arcsinh transformation based on Q and stretch
    # parameters.
    slope = 0.1 * 1 / np.arcsinh(0.1 * Q)

    # Apply the modified luminosity transformation using arcsinh function.
    soften = Q / stretch
    intensities = np.arcsinh((abs(values) * soften + floor) * slope)

    # Always normalize by what the original max value (100) scales to.
    maximum_intensity = np.arcsinh((1 * soften + floor) * slope)

    intensities /= maximum_intensity

    # contrast equalizer
    if equalizer_levels is not None:
        intensities = contrast_equalizer(intensities, equalizer_levels)

    # Scale the intensities with linear manipulation for contrast
    intensities = (intensities - shadow) / (highlight - shadow)
    intensities = np.clip(intensities, 0, max)
    intensities = ((midtone - 1) * intensities) / (((2 * midtone - 1) * intensities) - midtone)

    intensities = np.clip(intensities, 0, max)

    # Reset the input array.
    # values /= 100

    return intensities


def mapUpperBounds(img: NDArray, quant: float = 0.9, absMax: float | None = None) -> NDArray:
    """Bound images to a range between zero and one.

    Some images supplied aren't properly bounded with a maximum value of 1.
    Either the images exceed the bounds of 1, or that no value seems to close,
    implying indeterminate maximum value. This function determines an
    appropriate maximum either by taking the value supplied in the absMax
    argument or by scaling the maximum across all channels with the
    supplied quant variable.

    Parameters
    ----------
    img : `NDArray` like
        Must have dimensions of y,x,3 where the channels are in RGB order
    quant : `float`
        Value to scale the maximum pixel value, in any channel, by to
        determine the maximum flux allowable in all channels. Ignored
        if absMax isn't None.
    absMax : `float` or `None`
        If this value is not None, use it as the maximum pixel value
        for all channels, unless scaleBoundFactor is set in which case
        it is only the maximum if the value determined from the image
        and quant is larger than scaleBoundFactor times absMax. This is
        to prevent individual frames in a mosaic from being scaled too
        faint if absMax is too large for one region.

    Returns
    -------
    image : `NDArray`
        The result of the remapping process
    """
    if np.max(img) == 1:
        return img

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    if absMax is not None:
        scale = absMax
    else:
        r_quant = np.quantile(r, 0.95)
        g_quant = np.quantile(g, 0.95)
        b_quant = np.quantile(b, 0.95)
        turnover = np.max((r_quant, g_quant, b_quant))
        scale = turnover * quant

    image = np.copy(img)
    image += 1.5
    image /= scale
    # image[:, :, 0] = r / scale
    # image[:, :, 1] = g / scale
    # image[:, :, 2] = b / scale

    # Clip values that exceed the bound to ensure all values are within [0, absMax]
    image = np.clip(image, 0, 1)

    return image


def colorConstantSat(
    oldLum: NDArray,
    luminance: NDArray,
    a: NDArray,
    b: NDArray,
    saturation: float = 0.6,
    maxChroma: float = 80,
    equalizer_levels: list[float] | None = None,
) -> tuple[NDArray, NDArray]:
    """
    Adjusts the color saturation while keeping the hue constant.

    This function adjusts the chromaticity (a, b) of colors to maintain a
    consistent saturation level, based on their original luminance. It uses
    the CIELAB color space representation and the `luminance` is the new target
    luminance for all colors.

    Parameters
    ----------
    oldLum : `NDArray`
        Luminance values of the original colors.
    luminance : `NDArray`
        Target luminance values for the transformed colors.
    a : `NDArray`
        Chromaticity parameter 'a' corresponding to green-red axis in CIELAB.
    b : `NDArray`
        Chromaticity parameter 'b' corresponding to blue-yellow axis in CIELAB.
    saturation : `float`, optional
        Desired saturation level for the output colors. Defaults to 1.
    maxChroma : `float`, optional
        Maximum chroma value allowed for any color. Defaults to 50.

    Returns
    -------
    new_a : NDArray
        New a values representing the adjusted chromaticity.
    new_b : NDArray
        New b values representing the adjusted chromaticity.
    """
    # Calculate the square of the chroma, which is the distance from origin in
    # the a-b plane.
    chroma1_2 = a**2 + b**2
    chroma1 = np.sqrt(chroma1_2)

    # Calculate the hue angle, taking the absolute value to ensure non-negative
    # angle representation.
    chromaMask = chroma1 == 0
    chroma1[chromaMask] = 1
    sinHue = b / chroma1
    cosHue = a / chroma1
    sinHue[chromaMask] = 0
    cosHue[chromaMask] = 0

    # Compute a divisor for saturation calculation, adding 1 to avoid division
    # by zero.
    div = chroma1_2 + oldLum**2
    div[div <= 0] = 1

    # Calculate the square of the new chroma based on desired saturation
    sat_original_2 = chroma1_2 / div
    sat_mult = np.exp(-saturation * luminance)
    chroma2_2 = sat_mult*sat_original_2 * luminance**2 / (1 - sat_original_2)
    # chroma2_2 = saturation * sat_original_2 * luminance**2 / (1 - sat_original_2)

    # Compute new 'a' values using the square root of adjusted chroma and
    # considering hue direction.
    chroma2 = np.sqrt(chroma2_2)
    # chroma2 *= sat_mult
    # Try equalizing
    if equalizer_levels is not None:
        chroma2 = contrast_equalizer(chroma2, equalizer_levels)
    # Cap the chroma to avoid excessive values that are visually unrealistic
    #
    chroma2[chroma2 > maxChroma] = maxChroma

    new_a = chroma2 * cosHue

    # Compute new 'b' values using the root of the adjusted chroma and hue
    # direction.
    new_b = chroma2 * sinHue

    return new_a, new_b


def fixOutOfGamutColors(
    Lab: NDArray,
    xyz_whitepoint,
    colourspace: str = "Display P3",
    gamutMethod: Literal["mapping", "inpaint"] = "inpaint",
) -> NDArray:
    """Remap colors that fall outside an RGB color gamut back into it.

    This function modifies the input Lab array in-place for memory reasons.

    Parameters
    ----------
    Lab : `NDArray`
        A NxMX3 array that contains data in the Lab colorspace.
    colourspace : `str`, optional
        The target colourspace to map outlying pixels into. This must
        correspond to an RGB colourspace understood by the colour-science
        python package.
    gamut_method : `str`, optional
        This determines what algorithm will be used to map out of gamut
        colors. Must be one of ``mapping`` or ``inpaint``.
    """
    # Convert back into the CIE XYZ colourspace.
    # xyz_prime = colour.Oklab_to_XYZ(Lab)

    # And then back to the specified RGB colourspace.
    # rgb_prime = colour.XYZ_to_RGB(xyz_prime, colourspace=colourspace)
    rgb_prime = oklab_rgb.Oklab_to_RGB(np.ascontiguousarray(Lab), xyz_whitepoint)

    # Determine if there are any out of bounds pixels
    outOfBounds = np.bitwise_or(
        np.bitwise_or(rgb_prime[:, :, 0] > 1, rgb_prime[:, :, 1] > 1), rgb_prime[:, :, 2] > 1
    )
    # skip really large regions
    from scipy.ndimage import label

    labels, num_features = label(outOfBounds)
    for i in range(1, num_features + 1):
        if np.sum(labels == i) > 2000:
            outOfBounds[labels == i] = 0

    # If all pixels are in bounds, return immediately.
    if not np.any(outOfBounds):
        logging.info("There are no out of gamut pixels.")
        return rgb_prime

    logging.info("There are out of gamut pixels, remapping colors")
    match gamutMethod:
        case "inpaint":
            results = inpaint_biharmonic(rgb_prime, outOfBounds, channel_axis=-1)
        case "mapping":
            results = fixGamutOK(Lab[outOfBounds])
            Lab[outOfBounds] = results
            # results = colour.XYZ_to_RGB(colour.Oklab_to_XYZ(Lab), colourspace=colourspace)
            results = oklab_rgb.Oklab_to_RGB(np.ascontiguousarray(Lab), xyz_whitepoint)
        case _:
            raise ValueError(f"gamut correction {gamutMethod} is not supported")

    logging.debug(f"The total number of remapped pixels is: {np.sum(outOfBounds)}")
    return results


def _fuseExposure(images, sigma=0.2, maxLevel=3):
    weights = np.zeros((len(images), *images[0].shape[:2]))
    for i, image in enumerate(images):
        exposure = np.exp(-((image[:, :, 0] - 0.5) ** 2) / (2 * sigma))

        weights[i, :, :] = exposure
    norm = np.sum(weights, axis=0)
    np.divide(weights, norm, out=weights)

    # loop over each image again to build pyramids
    g_pyr = []
    l_pyr = []
    maxImageLevel = int(np.min(np.log2(images[0].shape[:2])))
    if maxLevel is None:
        maxLevel = maxImageLevel
    if maxImageLevel < maxLevel:
        raise ValueError(
            f"The supplied max level {maxLevel} is is greater than the max of the image: {maxImageLevel}"
        )
    support = 1 << (maxLevel - 1)
    padY_amounts = levelPadder(image.shape[0] + support, maxLevel)
    padX_amounts = levelPadder(image.shape[1] + support, maxLevel)
    for image, weight in zip(images, weights):
        imagePadded = cv2.copyMakeBorder(
            image, *(0, support), *(0, support), cv2.BORDER_REFLECT, None, None
        ).astype(image.dtype)
        weightPadded = cv2.copyMakeBorder(
            weight, *(0, support), *(0, support), cv2.BORDER_REFLECT, None, None
        ).astype(image.dtype)

        g_pyr.append(list(makeGaussianPyramid(weightPadded, padY_amounts, padX_amounts, None)))
        l_pyr.append(list(makeLapPyramid(imagePadded, padY_amounts, padX_amounts, None, None)))

    # time to blend
    blended = []
    for level in range(len(padY_amounts)):
        accumulate = np.zeros_like(l_pyr[0][level])
        for img in range(len(g_pyr)):
            for i in range(3):
                accumulate[:, :, i] += l_pyr[img][level][:, :, i] * g_pyr[img][level]
        blended.append(accumulate)

    # time to reconstruct
    output = blended[-1]
    for i in range(-2, -1 * len(blended) - 1, -1):
        upsampled = cv2.pyrUp(output)
        upsampled = upsampled[
            : upsampled.shape[0] - 2 * padY_amounts[i + 1], : upsampled.shape[1] - 2 * padX_amounts[i + 1]
        ]
        output = blended[i] + upsampled
    return output[:-support, :-support]


def _fuseExposureLum(images, sigma=0.1, maxLevel=3):
    weights = np.zeros((len(images), *images[0].shape[:2]))
    for i, image in enumerate(images):
        exposure = np.exp(-((image[:, :] - 0.7) ** 2) / (2 * sigma))
        # dont weight at all values greater than 1
        exposure[image > 1] *= 0.5

        weights[i, :, :] = exposure
    norm = np.sum(weights, axis=0)
    np.divide(weights, norm, out=weights)

    # loop over each image again to build pyramids
    g_pyr = []
    l_pyr = []
    maxImageLevel = int(np.min(np.log2(images[0].shape[:2])))
    if maxLevel is None:
        maxLevel = maxImageLevel
    if maxImageLevel < maxLevel:
        raise ValueError(
            f"The supplied max level {maxLevel} is is greater than the max of the image: {maxImageLevel}"
        )
    support = 1 << (maxLevel - 1)
    padY_amounts = levelPadder(image.shape[0] + support, maxLevel)
    padX_amounts = levelPadder(image.shape[1] + support, maxLevel)
    for image, weight in zip(images, weights):
        imagePadded = cv2.copyMakeBorder(
            image, *(0, support), *(0, support), cv2.BORDER_REFLECT, None, None
        ).astype(image.dtype)
        weightPadded = cv2.copyMakeBorder(
            weight, *(0, support), *(0, support), cv2.BORDER_REFLECT, None, None
        ).astype(image.dtype)

        g_pyr.append(list(makeGaussianPyramid(weightPadded, padY_amounts, padX_amounts, None)))
        l_pyr.append(list(makeLapPyramid(imagePadded, padY_amounts, padX_amounts, None, None)))

    # time to blend
    blended = []
    for level in range(len(padY_amounts)):
        accumulate = np.zeros_like(l_pyr[0][level])
        for img in range(len(g_pyr)):
            accumulate[:, :] += l_pyr[img][level][:, :] * g_pyr[img][level]
        blended.append(accumulate)

    # time to reconstruct
    output = blended[-1]
    for i in range(-2, -1 * len(blended) - 1, -1):
        upsampled = cv2.pyrUp(output)
        upsampled = upsampled[
            : upsampled.shape[0] - 2 * padY_amounts[i + 1], : upsampled.shape[1] - 2 * padX_amounts[i + 1]
        ]
        output = blended[i] + upsampled
    return output[:-support, :-support]


def _handelLuminance(
    img: NDArray,
    scaleLum: Callable[..., NDArray] | None = latLum,
    scaleLumKWargs: Mapping | None = None,
    remapBounds: Callable | None = mapUpperBounds,
    remapBoundsKwargs: Mapping | None = None,
    doLocalContrast: bool = True,
    sigma: float = 30,
    highlights: float = -0.9,
    shadows: float = 0.5,
    clarity: float = 0.1,
    maxLevel: int | None = None,
    cieWhitePoint: tuple[float, float] = (0.28, 0.28),
    bracket: float = 1,
    psf: NDArray | None = None,
):
    img = np.copy(img)
    # remap the bounds of the image if there is a function to do so.
    if remapBounds is not None:
        img = remapBounds(img, **(remapBoundsKwargs or {}))

    # scale to the supplied bracket
    # img /= bracket

    # Convert the starting image into the OK L*a*b* color space.
    # https://en.wikipedia.org/wiki/Oklab_color_space

    # Lab = colour.XYZ_to_Oklab(
    #     colour.RGB_to_XYZ(
    #         img,
    #         colourspace="CIE RGB",
    #         illuminant=np.array(cieWhitePoint), 
    #         chromatic_adaptation_transform="bradford",
    #     )
    # )
    Lab = oklab_rgb.RGB_to_Oklab(img, cieWhitePoint)
    lum = Lab[:, :, 0]

    if False:
        bgConfig = SubtractBackgroundTask.ConfigClass()
        bgConfig.algorithm = "AKIMA_SPLINE"
        bgConfig.useApprox = False
        bgConfig.binSizeX = 32
        bgConfig.binSizeY = 32
        bgTask = SubtractBackgroundTask(config=bgConfig)
        
        hue = np.arctan2(Lab[...,2], Lab[...,1])
        hue[hue < 0] += 2*np.pi
        hue = np.rad2deg(hue)

        mesh = np.stack([_x.flatten() for _x in np.indices(img[:,:,0].shape)]).T

        faint_nJy = 7 / 16500
        faint_lum = oklab_rgb.RGB_to_Oklab(np.array([[[faint_nJy, faint_nJy, faint_nJy]]]), cieWhitePoint)[0,0,0]
        faint_mask = Lab[:,:,0] < faint_lum
        blue_mask = (hue > 238) * (hue < 242)

        blue_mask = binary_dilation(blue_mask, iterations=2)

        masked_blue = np.ma.array(img[:,:,2], mask=~(blue_mask*faint_mask)).reshape((32,16,32,16))
        masked_blue_avg = np.mean(masked_blue, axis=(1,3))
        places = np.mgrid[16:512:32, 16:512:32]
        breakpoint()
        # blue_interp_maker = LinearNDInterpolator(list(zip(places)), masked_blue_avg.filled(0).flat)
        # blue_interp = blue_interp_maker([u.flatten() for u in np.mgrid[0:512,0:512]])
        tps = ThinPlateSpline(alpha=0.5)
        tps.fit(np.array([u.flatten() for u in places]).T, masked_blue_avg.filled(0).flatten())
        blue_interp = tps.transform(np.array([u.flatten() for u in np.mgrid[0:512, 0:512]]).T)
        img[:,:,2] -= blue_interp.reshape(512,512)

        # blue_afw_mask = Mask(*(blue_mask.shape[::-1]))
        # blue_afw_mask.array[:,:] = (~faint_mask) * blue_afw_mask.getMaskPlaneDict()['DETECTED']
        # blue_MI = MaskedImageF(*(blue_mask.shape[::-1]))
        # blue_MI.image.array[:,:] = np.copy(img[:,:,2])
        # blue_MI.image.array[~blue_mask] = 0
        # blue_MI.mask = blue_afw_mask
        # blue_bg = bgTask.run(ExposureF(blue_MI)).background
        # img[:,:,2] -= blue_bg.getImage().array.astype(float)
        
        # if np.sum(blue_mask) > 0.2*(512*512):
        #     blue_points = np.stack(blue_mask.nonzero()).T
        #     blue_surface = RBFInterpolator(blue_points, img[blue_mask,2], smoothing=0.05, neighbors=10)(mesh)
        #     img[:,:,2] -= blue_surface.reshape(img[:,:,0].shape)

    
        green_mask = (hue > 118) * (hue < 122)

        green_mask = binary_dilation(green_mask, iterations=2)

        masked_green = np.ma.array(img[:,:,1], mask=~(green_mask*faint_mask)).reshape((32,16,32,16))
        masked_green_avg = np.mean(masked_green, axis=(1,3))
        places = np.mgrid[16:512:32, 16:512:32]
        green_interp_maker = LinearNDInterpolator(list(zip(places)), masked_green_avg.filled(0).flat)
        green_interp = green_interp_maker([u.flatten() for u in np.mgrid[0:512,0:512]])
        img[:,:,1] -= green_interp.reshape(512,512)

        # green_afw_mask = Mask(*(green_mask.shape[::-1]))
        # green_afw_mask.array[:,:] = (~faint_mask) * green_afw_mask.getMaskPlaneDict()['DETECTED']
        # green_MI = MaskedImageF(*(green_mask.shape[::-1]))
        # green_MI.image.array[:,:] = np.copy(img[:,:,1])
        # green_MI.image.array[~green_mask] = 0
        # green_MI.mask = green_afw_mask
        # green_bg = bgTask.run(ExposureF(green_MI)).background
        # img[:,:,1] -= green_bg.getImage().array.astype(float)

        # if np.sum(green_mask) > 0.2*(512*512):
        #     green_points = np.stack(green_mask.nonzero()).T
        #     green_surface = RBFInterpolator(green_points, img[green_mask,2], smoothing=0.05, neighbors=10)(mesh)
        #     img[:,:,1] -= green_surface.reshape(img[:,:,0].shape)

        red_mask = (hue > 358) * (hue < 2)

        red_mask = binary_dilation(red_mask, iterations=2)

        masked_red = np.ma.array(img[:,:,0], mask=~(red_mask*faint_mask)).reshape((32,16,32,16))
        masked_red_avg = np.mean(masked_red, axis=(1,3))
        places = np.mgrid[16:512:32, 16:512:32]
        red_interp_maker = LinearNDInterpolator(list(zip(places)), masked_red_avg.filled(0).flat)
        red_interp = red_interp_maker([u.flatten() for u in np.mgrid[0:512,0:512]])
        img[:,:,0] -= red_interp.reshape(512,512)

        
        # red_afw_mask = Mask(*(red_mask.shape[::-1]))
        # red_afw_mask.array[:,:] = (~faint_mask) * red_afw_mask.getMaskPlaneDict()['DETECTED']
        # red_MI = MaskedImageF(*(red_mask.shape[::-1]))
        # red_MI.image.array[:,:] = np.copy(img[:,:,0])
        # red_MI.image.array[~red_mask] = 0
        # red_MI.mask = red_afw_mask
        # red_bg = bgTask.run(ExposureF(red_MI)).background
        # img[:,:,0] -= red_bg.getImage().array.astype(float)

        
        # if np.sum(red_mask) > 0.2*(512*512):
        #     red_points = np.stack(red_mask.nonzero()).T
        #     red_surface = RBFInterpolator(red_points, img[red_mask,2], smoothing=0.05, neighbors=10)(mesh)
        #     img[:,:,0] -= red_surface.reshape(img[:,:,0].shape)

        Lab = oklab_rgb.RGB_to_Oklab(img, cieWhitePoint)
        lum = Lab[:, :, 0]


    # Scale the luminance channel if possible.
    newLum = lum
    if scaleLum is not None:
        if doLocalContrast:
            lcArgs = dict(
                sigma=sigma, highlights=highlights, shadows=shadows, clarity=clarity, maxLevel=maxLevel
            )
        else:
            lcArgs = None
        lRemapped = scaleLum(newLum, **(scaleLumKWargs or {}), brackets=bracket, localContrastArgs=lcArgs)
    else:
        lRemapped = newLum
    # if doLocalContrast:
    #     lRemapped = localContrast(lRemapped, sigma, highlights, shadows, clarity=clarity, maxLevel=maxLevel)
    #     lRemapped = np.clip(lRemapped, 0, 1)
    # else:
    #     lRemapped = lRemapped

    if psf is not None:
        lRemapped = skimage.restoration.richardson_lucy(lRemapped, psf=psf, clip=False, num_iter=5)
    return lRemapped, Lab


def lsstRGB(
    rArray: NDArray,
    gArray: NDArray,
    bArray: NDArray,
    doLocalContrast: bool = True,
    scaleLum: Callable[..., NDArray] | None = latLum,
    scaleLumKWargs: Mapping | None = None,
    scaleColor: Callable[..., tuple[NDArray, NDArray]] | None = colorConstantSat,
    scaleColorKWargs: Mapping | None = None,
    remapBounds: Callable | None = mapUpperBounds,
    remapBoundsKwargs: Mapping | None = None,
    cieWhitePoint: tuple[float, float] = (0.28, 0.28),
    sigma: float = 30,
    highlights: float = -0.9,
    shadows: float = 0.5,
    clarity: float = 0.1,
    maxLevel: int | None = None,
    psf: NDArray | None = None,
    brackets: list[float] | None = None,
    doRemapGamut: bool = True,
    gamutMethod: Literal["mapping", "inpaint"] = "inpaint",
) -> NDArray:
    """Enhance the lightness and color preserving hue using perceptual methods.

    Parameters
    ----------
    rArray : `NDArray`
        The array used as the red channel
    gArray : `NDArray`
        The array used as the green channel
    bArray : `NDArray`
        The array used as the blue channel
    doLocalContrast: `bool`
        Apply local contrast enhancement algorithms to the luminance channel.
    scaleLum : `Callable` or `None`
        This is a callable that's passed the luminance values as well as
        any defined scaleLumKWargs, and should return a scaled luminance array
        the same shape as the input. Set to None for no scaling.
    scaleLumKWargs : `Mapping` or `None`
        Key word arguments that passed to the scaleLum function.
    scaleColor : `Callable` or `None`
        This is a callable that's passed the original luminance, the remapped
        luminance values, the a values for each pixel, and the b values for
        each pixel. The function is also passed any parameters defined in
        scaleColorKWargs. This function is responsible for scaling chroma
        values. This should return two arrays corresponding to the scaled a and
        b values. Set to None for no modification.
    scaleColorKWargs : `Mapping` or `None`
        Key word arguments passed to the scaleColor function.
    remapBounds : `Callable` or `None`
        This is a callable that should remaps the input arrays such that each of
        them fall within a zero to one range. This callable is given the
        initial image as well as any parameters defined in the remapBoundsKwargs
        parameter. Set to None for no remapping.
    remapBoundsKwargs : `Mapping` or None
    cieWhitePoint : `tuple` of `float`, `float`
        This is the white point of the input of the input arrays in CIE XY
        coordinates. Altering this affects the relative balance of colors
        in the input image, and therefore also the output image.
    sigma : `float`
        The scale over which local contrast considers edges real and not noise.
    highlights : `float`
        A parameter that controls how local contrast enhances or reduces
        highlights. Contrary to intuition, negative values increase highlights.
    shadows : `float`
        A parameter that controls how local contrast will deepen or reduce
        shadows.
    clarity : `float`
        A parameter that relates to the local contrast between highlights and
        shadow.
    maxLevel : `int` or `None`
        The maximum number of image pyramid levels to enhance the local contrast
        over. Each level has a spatial scale of roughly 2^(level) pixels.
    psf : `NDArray` or `None`
        If this parameter is an image of a PSF kernel the luminance channel is
        deconvolved with it. Set to None to skip deconvolution.
    brackets : `list` of `float` or `None`
        If a list brackets is supplied, an image will  be generated at each of
        the brackets and the results will be used in exposure fusioning to
        increase the apparent dynamic range of the image. The image post bounds
        remapping will be divided by each of the values specified in this list,
        which can be used to create for instance an under, over, and ballanced
        expoisure. Theese will then be fusioned into a final single exposure
        selecting the proper elements from each of the images.
    doRemapGamut : `bool`, optional
        If this is `True` then any pixels which lay outside the representable
        color gamut after manipulation will be remapped to a "best" value
        which will be some compromise in hue, chroma, and lum. If this is
        `False` then the values will clip. This may be useful for
        seeing where problems in processing occur.

    Returns
    -------
    result : `NDArray`
        The brightness and color calibrated image.

    Raises
    ------
    ValueError
        Raised if the shapes of the input array don't match
    """
    if rArray.shape != gArray.shape or rArray.shape != bArray.shape:
        raise ValueError("The shapes of all the input arrays must be the same")

    # Construct a new image array in the proper byte ordering.
    img = np.empty((*rArray.shape, 3))
    img[:, :, 0] = rArray
    img[:, :, 1] = gArray
    img[:, :, 2] = bArray
    # If there are nan's in the image there is no real option other than to
    # set them to zero or throw.
    img[np.isnan(img)] = 0

    if not brackets:
        brackets = [1]

    exposures = []
    tmp_lum, Lab = _handelLuminance(
        img,
        scaleLum,
        scaleLumKWargs=scaleLumKWargs,
        remapBounds=remapBounds,
        remapBoundsKwargs=remapBoundsKwargs,
        doLocalContrast=doLocalContrast,
        sigma=sigma,
        highlights=highlights,
        clarity=clarity,
        shadows=shadows,
        maxLevel=maxLevel,
        cieWhitePoint=cieWhitePoint,
        bracket=brackets,
        psf=psf,
    )

    # # This is a hack to fix chromatic backgrounds
    # hue = np.arctan2(Lab[...,2], Lab[...,1])
    # hue[hue < 0] += 2*np.pi
    # hue = np.rad2deg(hue)

    # faint_nJy = 40 / 16500
    # faint_lum = oklab_rgb.RGB_to_Oklab(np.array([[[faint_nJy, faint_nJy, faint_nJy]]]), cieWhitePoint)[0,0,0]
    # faint_mask = Lab[:,:,0] < faint_lum
    # blue_mask = (hue > 235) * (hue < 245)

    # # drop any large areas
    # # labels, _ = label(faint_mask * blue_mask)
    # # unique, lcounts = np.unique(labels, return_counts=True)
    # # lcounts_mask = lcounts > 3
    # # labels = np.where(np.isin(labels, lcounts[lcounts_mask]), 0, labels)
    # # blue_mask = labels > 0
    
    # green_mask = (hue > 115) * (hue < 125)
    # # labels, _ = label(faint_mask * green_mask)
    # # unique, lcounts = np.unique(labels, return_counts=True)
    # # lcounts_mask = lcounts > 3
    # # labels = np.where(np.isin(labels, lcounts[lcounts_mask]), 0, labels)
    # # green_mask = labels > 0

    # red_mask = (hue > 355) * (hue < 5)
    # # labels, _ = label(faint_mask * red_mask)
    # # unique, lcounts = np.unique(labels, return_counts=True)
    # # lcounts_mask = lcounts > 3
    # # labels = np.where(np.isin(labels, lcounts[lcounts_mask]), 0, labels)
    # # red_mask = labels > 0

    # joint_mask = faint_mask * (blue_mask + green_mask + red_mask)

    # tmp_lum[joint_mask] *= 0.05

    if scaleColor is not None:
        new_a, new_b = scaleColor(
            Lab[:, :, 0],
            tmp_lum,
            Lab[:, :, 1],
            Lab[:, :, 2],
            **(scaleColorKWargs or {}),
        )
        # Replace the color information with the new scaled color information.
        Lab[..., 1] = new_a
        Lab[..., 2] = new_b
        # Lab[joint_mask, 1] *= 0
        # Lab[joint_mask, 2] *= 0
    # Replace the luminance information with the new scaled luminance information
    Lab[:, :, 0] = tmp_lum
    exposures.append(Lab)

    cieWhitePoint = (0.31272, 0.32903)
    # Fix any colors that fall outside of the RGB colour gamut.
    if doRemapGamut:
        result = fixOutOfGamutColors(Lab, cieWhitePoint, gamutMethod=gamutMethod)
    else:
        # result = colour.XYZ_to_RGB(colour.Oklab_to_XYZ(Lab), "Display P3")
        result = oklab_rgb.Oklab_to_RGB(np.ascontiguousarray(Lab), cieWhitePoint)

    # explicitly cut at 1 even though the mapping was to map colors
    # appropriately because the Z matrix transform can produce values greater
    # than 1 and is a known feature of the transform.
    result = np.clip(result, 0, 1)
    return result
