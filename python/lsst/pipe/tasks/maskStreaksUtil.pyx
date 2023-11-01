#cython: language_level=3
import cython

from libc.math cimport cos, sin, M_PI
import numpy as np
cimport numpy as np
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.nonecheck(False)
def _makeMaskedProfile(double sigma,
                       double theta,
                       double rho,
                       float [:] _mxmesh,
                       float [:] _mymesh,
                       float [:] _maskWeights,
                       float [:] _maskData):
    """Construct the line model in the masked region and calculate its
    derivatives.

    Parameters
    ----------
    line : `Line`
        Parameters of line profile for which to make profile in the masked
        region.
    fitFlux : `bool`
        Fit the amplitude of the line profile to the data.

    Returns
    -------
    model : `np.ndarray`
        Model in the masked region.
    dModel : `np.ndarray`
        Derivative of the model in the masked region.
    """
    cdef double drad = M_PI / 180
    cdef double invSigma = 1/sigma

    # Calculate distance between pixels and line
    cdef double radtheta = theta * drad
    cdef double costheta = cos(radtheta)
    cdef double sintheta = sin(radtheta)
    cdef size_t size = _maskData.shape[0]

    cdef double distance
    cdef np.ndarray[np.float32_t, ndim=1] distanceSquared = np.empty(size, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] dDistanceSqdRho = np.empty(size, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] dDistanceSqdTheta = np.empty(size, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] profile = np.empty(size, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] dProfile = np.empty(size, dtype=np.float32)
    cdef double fluxUpper = 0
    cdef double fluxLower = 0
    cdef int i
    for i in range(size):
        distance = (costheta * _mxmesh[i] + sintheta * _mymesh[i] - rho)
        distanceSquared[i] = distance * distance

        # Calculate partial derivatives of distance
        dDistanceSqdRho[i] = -2 * distance
        dDistanceSqdTheta[i] = (2 * distance * (-sintheta * _mxmesh[i] + costheta * _mymesh[i]) * drad)

        # Use pixel-line distances to make Moffat profile
        profile[i] = (1 + distanceSquared[i] * invSigma**2)**-2.5
        dProfile[i] = -2.5 * (1 + distanceSquared[i] * invSigma**2)**-3.5

        # Calculate line flux from profile and data
        fluxUpper += (_maskWeights[i] * _maskData[i] * profile[i])
        fluxLower += (_maskWeights[i] * profile[i]**2)

    cdef double flux
    if fluxLower == 0:
        flux = 0
    else:
        flux = fluxUpper / fluxLower

    cdef np.ndarray[np.float32_t, ndim=1] model = np.empty(size, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] dModeldRho = np.empty(size, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] dModeldTheta = np.empty(size, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] dModeldInvSigma = np.empty(size, dtype=np.float32)
    cdef float fluxdProfile
    cdef float fluxdProfileInvSigma
    for i in range(size):
        model[i] = flux * profile[i]
        # Calculate model derivatives
        fluxdProfile = flux * dProfile[i]
        fluxdProfileInvSigma = fluxdProfile * invSigma**2
        dModeldRho[i] = fluxdProfileInvSigma * dDistanceSqdRho[i]
        dModeldTheta[i] = fluxdProfileInvSigma * dDistanceSqdTheta[i]
        dModeldInvSigma[i] = fluxdProfile * distanceSquared[i] * 2 * invSigma

    return model, dModeldRho, dModeldTheta, dModeldInvSigma
