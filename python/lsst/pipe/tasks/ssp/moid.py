import numpy as np
from astropy import units as u
from astropy.constants import au as AU_CONST, GM_sun
from collections import namedtuple

# Convert AU to km from astropy
AU_KM = AU_CONST.to_value(u.km)


EarthElements = namedtuple(
    "EarthElements",
    ["a_AU", "e", "inc_deg", "Omega_deg", "omega_deg"],
)

# Named result type
MOIDResult = namedtuple(
    "MOIDResult",
    [
        "MOID_AU",
        "DeltaV_kms",
        "EclipticLongitude_deg",
        "TrueAnomaly1_deg",
        "TrueAnomaly2_deg",
    ],
)


def earth_orbit(epoch_mjd=51544.5):
    """Return Earth’s heliocentric Keplerian elements at a given epoch.

    Evaluates a linear secular model for Earth’s orbital elements
    in the J2000 ecliptic and equinox frame. The model coefficients
    are from JPL’s "Keplerian Elements for Approximate Positions of
    the Major Planets" (Standish 1992, Table 1), valid for
    3000 BC -- 3000 AD.

    Parameters
    ----------
    epoch_mjd : float, optional
        Epoch as Modified Julian Date. Default is 51544.5
        (J2000.0 = 2000 Jan 1.5 TDB).

    Returns
    -------
    EarthElements
        A namedtuple with fields:

        ``a_AU``
            Semi-major axis in AU.
        ``e``
            Eccentricity.
        ``inc_deg``
            Inclination to the ecliptic in degrees. Set to a small
            nonzero value (0.00005°) to avoid singularities in
            rotation matrices; the true value is ~0 by definition
            since the ecliptic *is* Earth’s mean orbital plane.
        ``Omega_deg``
            Longitude of the ascending node in degrees. Zero by
            definition in the ecliptic frame.
        ``omega_deg``
            Argument of perihelion in degrees. Since Omega = 0 in
            the ecliptic frame, this equals the longitude of
            perihelion (ϖ).

    Notes
    -----
    The linear model for each element is::

        element(T) = element_0 + element_dot * T

    where T is Julian centuries from J2000.0. The coefficients are:

    ======  ===============  =================
    Param   Value at J2000   Rate (per century)
    ======  ===============  =================
    a       1.00000261 AU    +0.00000562 AU
    e       0.01671123       -0.00004392
    ϖ       102.93768193°    +0.32327364°
    ======  ===============  =================

    Inclination and Omega are fixed at ~0 (ecliptic frame).

    MOID is a purely geometric quantity (no mean anomaly / phase
    dependence), but the *shape and orientation* of Earth’s orbit
    do evolve slowly. Evaluating at the asteroid’s osculating
    element epoch ensures the two orbits are compared at a
    consistent time.

    References
    ----------
    Standish, E.M. (1992). "Keplerian Elements for Approximate
    Positions of the Major Planets." Solar System Dynamics Group,
    JPL. https://ssd.jpl.nasa.gov/planets/approx_pos.html
    """
    T = (epoch_mjd - 51544.5) / 36525.0  # Julian centuries from J2000
    return EarthElements(
        a_AU=1.00000261 + 0.00000562 * T,
        e=0.01671123 - 0.00004392 * T,
        inc_deg=0.00005,
        Omega_deg=0.0,
        omega_deg=102.93768193 + 0.32327364 * T,
    )


def earth_orbit_J2000():
    """Return Earth’s elements at J2000.0. Deprecated; use
    `earth_orbit(epoch_mjd)` instead.
    """
    return earth_orbit(51544.5)


class MOIDSolver:
    """
    MOID solver using an adaptive 2D grid in (f1, f2).

    Units:
      - Semi-major axes a: AU
      - Distances / MOID: AU
      - Velocities: km/s
      - μ: astropy quantity (km^3/s^2) or float
      - Public angles: degrees
      - Internal angles: radians
    """

    def __init__(
        self,
        mu=GM_sun,
        n_samples=128,
        refine_factor=5.0,
        tol_MOID_abs=1e-7 * u.AU,
        tol_MOID_rel=1e-6,
        max_refine=10,
    ):
        # Convert mu to float in km^3/s^2
        self.mu = mu.to_value(u.km**3 / u.s**2) if hasattr(mu, "unit") else float(mu)

        self.n_samples = int(n_samples)
        self.refine_factor = float(refine_factor)

        # Absolute tolerance expressed in AU
        self.tol_MOID_abs = tol_MOID_abs.to_value(u.AU)
        self.tol_MOID_rel = float(tol_MOID_rel)

        self.max_refine = int(max_refine)

    # ------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------

    @staticmethod
    def _make_rotation_matrix(inc, Om, om):
        """Rz(Omega) * Rx(inc) * Rz(omega)."""
        cO = np.cos(Om)
        sO = np.sin(Om)
        ci = np.cos(inc)
        si = np.sin(inc)
        cw = np.cos(om)
        sw = np.sin(om)

        A00 = cw
        A01 = -sw
        A02 = 0.0

        A10 = ci * sw
        A11 = ci * cw
        A12 = -si

        A20 = si * sw
        A21 = si * cw
        A22 = ci

        Q = np.empty((3, 3), float)

        Q[0, 0] = cO * A00 - sO * A10
        Q[0, 1] = cO * A01 - sO * A11
        Q[0, 2] = cO * A02 - sO * A12

        Q[1, 0] = sO * A00 + cO * A10
        Q[1, 1] = sO * A01 + cO * A11
        Q[1, 2] = sO * A02 + cO * A12

        Q[2, 0] = A20
        Q[2, 1] = A21
        Q[2, 2] = A22

        return Q

    @classmethod
    def _make_orbit_params(cls, a_AU, e, inc, Om, om):
        """Return p in AU, rotation matrix Q."""
        p = a_AU * (1 - e * e)  # AU
        Q = cls._make_rotation_matrix(inc, Om, om)
        return p, e, Q

    @staticmethod
    def _orbit_positions(p_AU, e, Q, f):
        """Vectorized positions in AU."""
        f = np.asarray(f)
        cf = np.cos(f)
        sf = np.sin(f)

        r = p_AU / (1 + e * cf)  # AU

        x_pf = r * cf
        y_pf = r * sf

        X = Q[0, 0] * x_pf + Q[0, 1] * y_pf
        Y = Q[1, 0] * x_pf + Q[1, 1] * y_pf
        Z = Q[2, 0] * x_pf + Q[2, 1] * y_pf
        return np.column_stack((X, Y, Z))  # AU

    @staticmethod
    def _rv_from_params(p_AU, e, Q, f, mu_km3_s2):
        """Return (r in AU, v in km/s)."""
        cf = np.cos(f)
        sf = np.sin(f)

        # Position (AU)
        r_AU = p_AU / (1 + e * cf)
        x_pf = r_AU * cf
        y_pf = r_AU * sf

        X = Q[0, 0] * x_pf + Q[0, 1] * y_pf
        Y = Q[1, 0] * x_pf + Q[1, 1] * y_pf
        Z = Q[2, 0] * x_pf + Q[2, 1] * y_pf
        r_vec_AU = np.array([X, Y, Z], float)

        # Velocity (km/s): convert p (AU) to km
        p_km = p_AU * AU_KM
        fac = np.sqrt(mu_km3_s2 / p_km)

        vx_pf = -fac * sf
        vy_pf = fac * (e + cf)

        VX = Q[0, 0] * vx_pf + Q[0, 1] * vy_pf
        VY = Q[1, 0] * vx_pf + Q[1, 1] * vy_pf
        VZ = Q[2, 0] * vx_pf + Q[2, 1] * vy_pf
        v_vec_kms = np.array([VX, VY, VZ], float)

        return r_vec_AU, v_vec_kms

    # ------------------------------------------------------------
    # Adaptive 2D grid MOID
    # ------------------------------------------------------------

    def _moid_grid_search(self, p1, e1, Q1, p2, e2, Q2):
        """
        Adaptive search in AU.
        Returns: (MOID_AU, f1_best, f2_best)
        """
        n = self.n_samples
        rf = self.refine_factor
        tol_abs = self.tol_MOID_abs
        tol_rel = self.tol_MOID_rel
        max_ref = self.max_refine

        two_pi = 2 * np.pi

        center1 = np.pi
        center2 = np.pi
        width1 = two_pi
        width2 = two_pi

        best_d2 = np.inf
        best_f1 = 0.0
        best_f2 = 0.0
        prev_moid = np.inf

        for level in range(max_ref + 1):
            f1_lo = center1 - width1 / 2
            f1_hi = center1 + width1 / 2
            f2_lo = center2 - width2 / 2
            f2_hi = center2 + width2 / 2

            f1_grid = (f1_lo + (np.arange(n) + 0.5) * (f1_hi - f1_lo) / n) % two_pi
            f2_grid = (f2_lo + (np.arange(n) + 0.5) * (f2_hi - f2_lo) / n) % two_pi

            r1 = self._orbit_positions(p1, e1, Q1, f1_grid)
            r2 = self._orbit_positions(p2, e2, Q2, f2_grid)

            diff = r1[:, None, :] - r2[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", diff, diff)

            i0, j0 = np.unravel_index(np.argmin(d2), d2.shape)
            d2_local = d2[i0, j0]
            f1_local, f2_local = f1_grid[i0], f2_grid[j0]

            if d2_local < best_d2:
                best_d2 = d2_local
                best_f1 = f1_local
                best_f2 = f2_local

            moid = float(np.sqrt(best_d2))  # AU

            if level > 0:
                delta = abs(moid - prev_moid)
                scale = max(abs(moid), abs(prev_moid), 1.0)
                if delta <= tol_abs + tol_rel * scale:
                    break

            prev_moid = moid
            center1 = f1_local
            center2 = f2_local
            width1 /= rf
            width2 /= rf

        return moid, best_f1, best_f2

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def compute(self, el1, el2):
        """
        Compute MOID and related quantities.

        el = (a_AU, e, inc_deg, Ω_deg, ω_deg)
        Returns MOIDResult namedtuple.
        """
        a1, e1, i1d, O1d, w1d = el1
        a2, e2, i2d, O2d, w2d = el2

        i1 = np.deg2rad(i1d)
        O1 = np.deg2rad(O1d)
        w1 = np.deg2rad(w1d)

        i2 = np.deg2rad(i2d)
        O2 = np.deg2rad(O2d)
        w2 = np.deg2rad(w2d)

        p1, e1, Q1 = self._make_orbit_params(a1, e1, i1, O1, w1)
        p2, e2, Q2 = self._make_orbit_params(a2, e2, i2, O2, w2)

        MOID_AU, f1_best, f2_best = self._moid_grid_search(p1, e1, Q1, p2, e2, Q2)

        r1, v1 = self._rv_from_params(p1, e1, Q1, f1_best, self.mu)
        r2, v2 = self._rv_from_params(p2, e2, Q2, f2_best, self.mu)

        dv_kms = float(np.linalg.norm(v1 - v2))

        lon = np.degrees(np.arctan2(r1[1], r1[0])) % 360
        f1_deg = np.degrees(f1_best) % 360
        f2_deg = np.degrees(f2_best) % 360

        return MOIDResult(
            MOID_AU=MOID_AU,
            DeltaV_kms=dv_kms,
            EclipticLongitude_deg=lon,
            TrueAnomaly1_deg=f1_deg,
            TrueAnomaly2_deg=f2_deg,
        )


# -------------------------------------------------------------------
# Example
# -------------------------------------------------------------------
if __name__ == "__main__":
    solver = MOIDSolver(
        mu=GM_sun,
        n_samples=128,
        refine_factor=5.0,
        tol_MOID_abs=1e-8 * u.AU,
        tol_MOID_rel=1e-8,
        max_refine=12,
    )

    el1 = (1.0, 0.1, 5.0, 30.0, 45.0)
    el2 = (1.5, 0.2, 15.0, 60.0, 10.0)

    result = solver.compute(el1, el2)
    print(result)
