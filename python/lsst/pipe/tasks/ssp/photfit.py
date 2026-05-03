import numpy as np
from collections import namedtuple
from scipy.interpolate import CubicSpline
from scipy.optimize import leastsq, least_squares
import warnings

HG12FitResult = namedtuple(
    "HG12FitResult",
    ["H", "G12", "H_err", "G12_err", "HG_cov", "chi2dof", "nobs"],
)

# Constants

A = [3.332, 1.862]
B = [0.631, 1.218]
C = [0.986, 0.238]

# values taken from sbpy for convenience

alpha_12 = np.deg2rad([7.5, 30.0, 60, 90, 120, 150])

phi_1_sp = [7.5e-1, 3.3486016e-1, 1.3410560e-1, 5.1104756e-2, 2.1465687e-2, 3.6396989e-3]
phi_1_derivs = [-1.9098593, -9.1328612e-2]

phi_2_sp = [9.25e-1, 6.2884169e-1, 3.1755495e-1, 1.2716367e-1, 2.2373903e-2, 1.6505689e-4]
phi_2_derivs = [-5.7295780e-1, -8.6573138e-8]

alpha_3 = np.deg2rad([0.0, 0.3, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 30.0])

phi_3_sp = [
    1.0,
    8.3381185e-1,
    5.7735424e-1,
    4.2144772e-1,
    2.3174230e-1,
    1.0348178e-1,
    6.1733473e-2,
    1.6107006e-2,
    0.0,
]

phi_3_derivs = [-1.0630097, 0]


phi_1 = CubicSpline(alpha_12, phi_1_sp, bc_type=((1, phi_1_derivs[0]), (1, phi_1_derivs[1])))
phi_2 = CubicSpline(alpha_12, phi_2_sp, bc_type=((1, phi_2_derivs[0]), (1, phi_2_derivs[1])))
phi_3 = CubicSpline(alpha_3, phi_3_sp, bc_type=((1, phi_3_derivs[0]), (1, phi_3_derivs[1])))


def HG_model(phase, params):
    sin_a = np.sin(phase)
    tan_ah = np.tan(phase / 2)

    W = np.exp(-90.56 * tan_ah * tan_ah)
    scale_sina = sin_a / (0.119 + 1.341 * sin_a - 0.754 * sin_a * sin_a)

    phi_1_S = 1 - C[0] * scale_sina
    phi_2_S = 1 - C[1] * scale_sina

    phi_1_L = np.exp(-A[0] * np.power(tan_ah, B[0]))
    phi_2_L = np.exp(-A[1] * np.power(tan_ah, B[1]))

    phi_1 = W * phi_1_S + (1 - W) * phi_1_L
    phi_2 = W * phi_2_S + (1 - W) * phi_2_L
    return params[0] - 2.5 * np.log10((1 - params[1]) * phi_1 + (params[1]) * phi_2)


def HG1G2_model(phase, params):
    phi_1_ev = phi_1(phase)
    phi_2_ev = phi_2(phase)
    phi_3_ev = phi_3(phase)

    msk = phase < 7.5 * np.pi / 180

    phi_1_ev[msk] = 1 - 6 * phase[msk] / np.pi
    phi_2_ev[msk] = 1 - 9 * phase[msk] / (5 * np.pi)

    phi_3_ev[phase > np.pi / 6] = 0

    return params[0] - 2.5 * np.log10(
        params[1] * phi_1_ev + params[2] * phi_2_ev + (1 - params[1] - params[2]) * phi_3_ev
    )


def HG12_model(phase, params):
    if params[1] >= 0.2:
        G1 = +0.9529 * params[1] + 0.02162
        G2 = -0.6125 * params[1] + 0.5572
    else:
        G1 = +0.7527 * params[1] + 0.06164
        G2 = -0.9612 * params[1] + 0.6270

    return HG1G2_model(phase, [params[0], G1, G2])


def HG12star_model(phase, params):
    G1 = 0 + params[1] * 0.84293649
    G2 = 0.53513350 - params[1] * 0.53513350

    return HG1G2_model(phase, [params[0], G1, G2])


def chi2(params, mag, phase, mag_err, model):
    pred = model(phase, params)
    return (mag - pred) / mag_err


def fit(mag, phase, sigma, model=HG12_model, params=[0.1]):
    phase = np.deg2rad(phase)

    sol = leastsq(chi2, [mag[0]] + params, (mag, phase, sigma, model), full_output=True)

    return sol


def fitHG12(
    mag, magSigma, phaseAngle, tdist, rdist,
    fixedG12=None, magSigmaFloor=0.0, nSigmaClip=None,
):
    """Fit the HG12 phase curve model (Muinonen et al. 2010).

    Fits absolute magnitude H (and optionally the slope parameter
    G12) to apparent magnitude observations at known phase angles
    and distances.

    Parameters
    ----------
    mag : array_like
        Apparent magnitudes.
    magSigma : array_like
        Magnitude uncertainties (1-sigma).
    phaseAngle : array_like
        Phase angles in degrees.
    tdist : array_like
        Topocentric (observer-target) distances in AU.
    rdist : array_like
        Heliocentric (sun-target) distances in AU.
    fixedG12 : float or None, optional
        If set, fix G12 to this value and only fit H.
        If None (default), both H and G12 are fit.
    magSigmaFloor : float, optional
        Systematic error floor (mag) added in quadrature to
        ``magSigma`` before fitting. Default is 0.0.
    nSigmaClip : float or None, optional
        If set, perform outlier rejection: an initial robust fit
        (soft_l1 loss) followed by sigma clipping at this
        threshold, then a final linear least-squares refit on the
        clipped data. If None (default), no clipping is performed.

    Returns
    -------
    result : `HG12FitResult`
        Named tuple with fields:

        ``H``
            Best-fit absolute magnitude.
        ``G12``
            Best-fit (or fixed) slope parameter.
        ``H_err``
            Uncertainty on H from the covariance matrix.
        ``G12_err``
            Uncertainty on G12 (NaN if ``fixedG12`` is set).
        ``HG_cov``
            H-G12 covariance (NaN if ``fixedG12`` is set).
        ``chi2dof``
            Reduced chi-squared of the fit.
        ``nobs``
            Number of observations used (after clipping).

        On failure, all float fields are NaN and ``nobs`` is 0.
    """
    nobsv = len(mag)

    if nobsv == 0:
        return HG12FitResult(*(np.nan,) * 6, nobs=0)

    # ensure these are plain ndarrays
    (mag, magSigma, phaseAngle, tdist, rdist) = map(
        np.asarray, (mag, magSigma, phaseAngle, tdist, rdist)
    )

    # add systematic error floor in quadrature
    if magSigmaFloor > 0:
        magSigma = np.sqrt(magSigma**2 + magSigmaFloor**2)

    # filter to finite magnitudes and positive errors
    good = (
        np.isfinite(mag) & np.isfinite(magSigma)
        & (magSigma > 0)
    )
    mag = mag[good]
    magSigma = magSigma[good]
    phaseAngle = phaseAngle[good]
    tdist = tdist[good]
    rdist = rdist[good]
    nobsv = len(mag)

    if nobsv == 0:
        return HG12FitResult(*(np.nan,) * 6, nobs=0)

    # correct the mag to 1AU distance
    dmag = -5.0 * np.log10(tdist * rdist)
    mag = mag + dmag

    if fixedG12 is not None:
        def model(phase, params):
            return HG12_model(phase, [params[0], fixedG12])

        nparams = 1
    else:
        model = HG12_model
        nparams = 2

    phase_rad = np.deg2rad(phaseAngle)
    x0 = np.array(
        [mag[0]] + ([] if fixedG12 is not None else [0.1])
    )

    def residuals(params):
        return (mag - model(phase_rad, params)) / magSigma

    # fit, suppressing warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if nSigmaClip is not None and nobsv > nparams + 1:
            # Stage 1: robust fit with soft_l1 loss
            sol_robust = least_squares(
                residuals, x0, loss='soft_l1', f_scale=1.0,
            )
            if not sol_robust.success:
                return HG12FitResult(*(np.nan,) * 6, nobs=0)

            # Sigma clipping on residuals from robust fit
            resid = residuals(sol_robust.x)
            keep = np.abs(resid) < nSigmaClip
            mag = mag[keep]
            magSigma = magSigma[keep]
            phase_rad = phase_rad[keep]
            nobsv = len(mag)

            if nobsv <= nparams:
                return HG12FitResult(*(np.nan,) * 6, nobs=0)

            # Redefine residuals for clipped data
            def residuals(params):
                return (
                    (mag - model(phase_rad, params)) / magSigma
                )

            x0 = sol_robust.x

        # Final fit (linear loss for proper chi2/covariance)
        sol = least_squares(residuals, x0, loss='linear')

        if not sol.success:
            return HG12FitResult(*(np.nan,) * 6, nobs=0)

        # Extract results
        chi2_total = np.sum(sol.fun ** 2)

        # Covariance from Jacobian: cov = inv(J^T J)
        J = sol.jac
        try:
            cov = np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            return HG12FitResult(*(np.nan,) * 6, nobs=0)

        H = sol.x[0]
        H_err = np.sqrt(cov[0, 0])

        if fixedG12 is not None:
            G = fixedG12
            G_err = np.nan
            HG_cov = np.nan
        else:
            G = sol.x[1]
            G_err = np.sqrt(cov[1, 1])
            HG_cov = cov[0, 1]

        return HG12FitResult(
            H=H, G12=G, H_err=H_err, G12_err=G_err,
            HG_cov=HG_cov,
            chi2dof=chi2_total / (nobsv - nparams),
            nobs=nobsv,
        )


####################


def phase_angle_deg(r_obj_sun, r_obs_sun):
    """
    Compute phase angle (Sun–Object–Observer) in degrees.

    Parameters
    ----------
    r_obj_sun : array, shape (3,) or (3, N)
        Object position vector wrt Sun (Sun → object).
    r_obs_sun : array, shape (3,) or (3, N)
        Observer position vector wrt Sun (Sun → observer).

    Returns
    -------
    float or ndarray
        Phase angle(s) in degrees, in [0, 180].
    """
    r_obj_sun = np.asarray(r_obj_sun)
    r_obs_sun = np.asarray(r_obs_sun)

    # Vectors at the object
    v_sun = -r_obj_sun  # object → Sun
    v_obs = r_obs_sun - r_obj_sun  # object → observer

    # Dot products and norms along axis 0
    dot = np.sum(v_sun * v_obs, axis=0)
    norm_sun = np.linalg.norm(v_sun, axis=0)
    norm_obs = np.linalg.norm(v_obs, axis=0)

    cosang = dot / (norm_sun * norm_obs)
    cosang = np.clip(cosang, -1.0, 1.0)

    return np.degrees(np.arccos(cosang))


def hg_V_mag(H, G, r, delta, phase_deg):
    """
    Compute apparent V magnitude from the IAU H–G system.

    Parameters
    ----------
    H : float or ndarray
        Absolute magnitude (V-band).
    G : float or ndarray
        Slope parameter.
    r : float or ndarray
        Heliocentric distance in AU.
    delta : float or ndarray
        Observer distance (Δ) in AU.
    phase_deg : float or ndarray
        Phase angle in degrees.
    """
    a = np.radians(phase_deg) / 2.0

    # Phase functions
    phi1 = np.exp(-3.33 * np.tan(a) ** 0.63)
    phi2 = np.exp(-1.87 * np.tan(a) ** 1.22)

    phi = (1 - G) * phi1 + G * phi2

    return H + 5 * np.log10(r * delta) - 2.5 * np.log10(phi)
