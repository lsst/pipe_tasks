import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import leastsq
import warnings

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


def fitHG12(mag, magSigma, phaseAngle, tdist, rdist):
    # This uses the HG12 function from Muinonen et al (2010)
    nobsv = len(mag)

    # ensure these are plain ndarrays
    (mag, magSigma, phaseAngle, tdist, rdist) = map(np.asarray, (mag, magSigma, phaseAngle, tdist, rdist))

    # correct the mag to 1AU distance
    dmag = -5.0 * np.log10(tdist * rdist)
    mag = mag + dmag
    # phaseAngle = np.deg2rad(phaseAngle)

    # fit, suppressing warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        sol = fit(mag, phaseAngle, magSigma)
        if sol[-1] in [1, 2, 3] and sol[1] is not None:  # least squares solver flags
            chi2 = np.sum(sol[2]["fvec"] ** 2)
            H = sol[0][0]
            G = sol[0][1]
            H_err = np.sqrt(sol[1][0, 0])
            G_err = np.sqrt(sol[1][1, 1])
            HG_cov = sol[1][0, 1]

            return H, G, H_err, G_err, HG_cov, chi2 / (nobsv - 2), nobsv
        else:
            # fit failed
            return (np.nan,) * 6 + (0,)


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
