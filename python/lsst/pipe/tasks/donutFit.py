from __future__ import print_function
#
# LSST Data Management System
# Copyright 2008-2017 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
import numpy as np
import galsim
import lmfit

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.afw.math as afwMath


__all__ = ["DonutFitConfig", "DonutFitTask"]

HSC_diam = 8.2  # meters
HSC_obs_frac = 0.231  # dimensionless linear diameter obstruction fraction
HSC_obs_rad = HSC_obs_frac * HSC_diam/2  # radius of central obstruction in meters
HSC_strut_angle = 51.75  # degrees
HSC_strut_thick = 0.22  # meters

# d(lens_center)/d(theta) in meters per degree
lens_obs_rate = 0.0276 * 3600 / 128.9 * HSC_diam/2
# d(camera_center)/d(theta) in meters per degree
cam_obs_rate = 0.00558 * 3600 / 128.9 * HSC_diam/2
# Radius of lens obstruction in meters
lens_obs_rad = HSC_diam/2 * 138./128.98


def distance(p1, p2, p0):
    """Compute the distance between the line through p1-p2 and p0.

    @param p1  2-tuple for one point on line
    @param p2  2-tuple for another point on line
    @param p0  2-tuple for which to calculate distance to line
    @returns distance
    """
    x1, y1 = p1
    x2, y2 = p2
    x0, y0 = p0
    dy21 = y2 - y1
    dx21 = x2 - x1
    return np.abs(dy21*x0 - dx21*y0+x2*y1-y2*x1)/np.hypot(dy21, dx21)


def HSC_obscured_Airy(lam, pad_factor):
    """Get a simple circular obscured HSC Aperture from galsim.  Useful for getting
    a reasonable starting size/resolution when building a more complex Aperture.

    @param lam         Wavelength in nm
    @param pad_factor  Additional focal-plane padding factor
                       (=resolution in pupil_plane)
    @returns           galsim.Aperture object.
    """
    return galsim.Aperture(lam=lam, diam=HSC_diam,
                           obscuration=HSC_obs_frac, pad_factor=pad_factor)


def clear_aper(aper):
    """Convert potentially complicated Aperture into a simple circular aperture.
    @param aper  galsim.Aperture
    @returns     galsim.Aperture
    """
    return galsim.Aperture(diam=aper.diam,
                           pupil_plane_scale=aper.pupil_plane_scale,
                           pupil_plane_size=aper.pupil_plane_size)


def cut_circle_interior(aper, u0, v0, r):
    """Cut out the interior of a circular region from an Aperture.

    @param aper    galsim.Aperture
    @param u0, v0  Region center in pupil coordinates.
    @param r       Region radius in pupil coordinates.
    @returns       galsim.Aperture
    """
    aper_img = aper.illuminated
    r2 = (aper.u-u0)**2 + (aper.v-v0)**2
    aper_img[r2 < r**2] = False
    return galsim.Aperture(diam=aper.diam,
                           pupil_plane_im=aper_img.astype(np.int32),
                           pupil_plane_scale=aper.pupil_plane_scale,
                           pupil_plane_size=aper.pupil_plane_size)


def cut_circle_exterior(aper, u0, v0, r):
    """Cut out the exterior of a circular region from an Aperture.

    @param aper    galsim.Aperture
    @param u0, v0  Region center in pupil coordinates.
    @param r       Region radius in pupil coordinates.
    @returns       galsim.Aperture
    """
    aper_img = aper.illuminated
    r2 = (aper.u-u0)**2 + (aper.v-v0)**2
    aper_img[r2 > r**2] = False
    return galsim.Aperture(diam=aper.diam,
                           pupil_plane_im=aper_img.astype(np.int32),
                           pupil_plane_scale=aper.pupil_plane_scale,
                           pupil_plane_size=aper.pupil_plane_size)


def cut_ray(aper, u0, v0, angle, thickness):
    """Cut out a ray from an Aperture.

    @param aper       galsim.Aperture
    @param u0, v0     Ray origin in pupil coordinates.
    @param angle      Ray angle measured CCW from +x in radians.
    @param thickness  Thickness of cutout in pupil plane units.
    @returns          galsim.Aperture
    """
    aper_img = aper.illuminated
     # the 1 is arbitrary, just need something to define the line
    u1 = u0 + 1
    v1 = v0 + np.tan(angle)
    d = distance((u0, v0), (u1, v1), (aper.u, aper.v))
    aper_img[(d<0.5*thickness)
             & ((aper.u-u0)*np.cos(angle)
                + (aper.v-v0)*np.sin(angle) >= 0)] = 0.0
    return galsim.Aperture(diam=aper.diam,
                           pupil_plane_im=aper_img.astype(np.int32),
                           pupil_plane_scale=aper.pupil_plane_scale,
                           pupil_plane_size=aper.pupil_plane_size)


def HSC_aper(theta_x, theta_y, lam, pad_factor):
    """Get HSC Aperture function given field angle.

    @param theta_x, theta_y   Field angle in degrees.
    @returns                  galsim.Aperture
    """
    aper = HSC_obscured_Airy(lam, pad_factor)
    aper = clear_aper(aper)
    cam_x = theta_x * cam_obs_rate
    cam_y = theta_y * cam_obs_rate
    aper = cut_circle_interior(aper, cam_x, cam_y, HSC_obs_rad)
    lens_x = theta_x*lens_obs_rate
    lens_y = theta_y*lens_obs_rate
    aper = cut_circle_exterior(aper, lens_x, lens_y, lens_obs_rad)
    aper = cut_ray(aper, 0.61+cam_x, cam_y,
                   HSC_strut_angle*np.pi/180, HSC_strut_thick)
    aper = cut_ray(aper, 0.61+cam_x, cam_y,
                   -HSC_strut_angle*np.pi/180, HSC_strut_thick)
    aper = cut_ray(aper, -0.61+cam_x, cam_y,
                   (180-HSC_strut_angle)*np.pi/180, HSC_strut_thick)
    aper = cut_ray(aper, -0.61+cam_x, cam_y,
                   (180+HSC_strut_angle)*np.pi/180, HSC_strut_thick)
    return aper


class ZFit:
    """Class to fit Zernike aberrations of donut images"""
    def __init__(self, exposure, jmax, bitmask, lam, aper, **kwargs):
        """
        @param exposure  Exposure object.
        @param jmax      Maximum Zernike order to fit.
        @param bitmask   Bitmask defining bad pixels.
        @param lam       Wavelength to use for model.
        @param aper      galsim.Aperture to use for model.
        @param **kwargs  Additional kwargs to pass to lmfit.minimize.
        """
        self.exposure = exposure
        self.jmax = jmax
        self.kwargs = kwargs
        self.init_params()
        self.bitmask = bitmask
        self.lam = lam
        self.aper = aper
        self.mask = (np.bitwise_and(self.exposure.getMask().getArray().astype(np.uint16),
                                    self.bitmask) == 0)
        self.image = self.exposure.getImage().getArray()
        self.sigma = np.sqrt(self.exposure.getVariance().getArray())


    def fit(self):
        """Do the fit
        @returns  lmfit result.
        """
        import time
        t0 = time.time()
        self.result = lmfit.minimize(self.resid, self.params, **self.kwargs)
        t1 = time.time()
        print("Fitting took {:.1f} seconds".format(t1-t0))
        return self.result

    def init_params(self):
        """Initialize lmfit Parameters object.
        """
        params = lmfit.Parameters()
        params.add('z4', 13.0, min=9.0, max=18.0)
        for i in range(5, self.jmax+1):
            params.add('z{}'.format(i), 0.0, min=-2.0, max=2.0)
        params.add('r0', 0.2, min=0.1, max=0.4)
        params.add('dx', 0.0, min=-2, max=2)
        params.add('dy', 0.0, min=-2, max=2)
        flux = float(np.sum(self.exposure.getImage().getArray()))
        params.add('flux', flux, min=0.8*flux, max=1.2*flux)
        self.params = params

    def model(self, params):
        """Construct model image from parameters

        @param params  lmfit.Parameters object
        @returns       numpy array image
        """
        v = params.valuesdict()
        aberrations = [0,0,0,0]
        for i in range(4, self.jmax+1):
            aberrations.append(v['z{}'.format(i)])
        opt_psf = galsim.OpticalPSF(lam=self.lam,
                                    diam=self.aper.diam,
                                    aper=self.aper,
                                    aberrations=aberrations)
        atm_psf = galsim.Kolmogorov(lam=self.lam, r0=v['r0'])
        psf = (galsim.Convolve(opt_psf, atm_psf)
               .shift(v['dx'], v['dy'])
               * v['flux'])
        model_img = psf.drawImage(nx=73, ny=73, scale=0.168)
        return model_img.array

    def resid(self, params):
        """Compute 'chi' image.

        @param params  lmfit.Parameters object.
        @returns       Unraveled chi vector.
        """
        model_img = self.model(params)
        chi = (self.image - model_img) / self.sigma * self.mask
        return chi.ravel()

    def report(self):
        """Report fit results.
        """
        if not hasattr(self, 'result'):
            self.fit()
        lmfit.report_fit(self.result)


class DonutFitConfig(pexConfig.Config):
    jmax = pexConfig.Field(
        dtype=int, default=15,
        doc="Number of Zernike coefficients to fit",
    )

    lam = pexConfig.Field(
        dtype=float,
        doc="If specified, use this wavelength (in nanometers) to model donuts.  "
            "If not specified, then use filter effective wavelength.",
        optional=True
    )

    padFactor = pexConfig.Field(
        dtype=float, default=3.0,
        doc="Padding factor to use for pupil image.",
    )

    r1cut = pexConfig.Field(
        dtype=float, default=50.0,
        doc="Rejection cut flux25/flux3 [default: 50.0]",
    )

    r2cut = pexConfig.Field(
        dtype=float, default=1.05,
        doc="Rejection cut flux35/flux25 [default: 1.05]",
    )

    snthresh = pexConfig.Field(
        dtype=float, default=250.0,
        doc="Donut signal-to-noise threshold [default: 250.0]",
    )

    stamp_size = pexConfig.Field(
        dtype=int, default=72,
        doc="Size of donut postage stamps [default: 72]",
    )

    bitmask = pexConfig.Field(
        dtype=int, default=130,
        doc="Bitmask indicating pixels to exclude from fit [default: 130]"
    )


class DonutFitTask(pipeBase.Task):

    ConfigClass = DonutFitConfig
    _DefaultName = "donutFit"

    def __init__(self, schema=None, **kwargs):
        """!Construct a DonutFitTask
        """
        pipeBase.Task.__init__(self, **kwargs)
        if schema is None:
            schema = afwTable.SourceTable.makeMinimalSchema()
        self.schema = schema
        self.r0 = schema.addField("r0", type=float)
        self.z = []
        for i in range(4, self.config.jmax+1):
            self.z.append(schema.addField("z{}".format(i), type=float))

    @pipeBase.timeMethod
    def run(self, icExp, icSrc):
        """!Fit donuts
        """
        lam = self.config.lam
        if lam is None:
            lam = icExp.getFilter().getFilterProperty().getLambdaEff()
            self.log.info("Using filter effective wavelength of {} nm".format(lam))
        nquarter = icExp.getDetector().getOrientation().getNQuarter()
        self.log.info("Nquarter = {}".format(nquarter))
        select = self.selectDonuts(icSrc)
        donutCat = icSrc.subset(select)
        for record in donutCat:
            x, y = record.getX(), record.getY()
            theta_x = x * 0.168 / 3600
            theta_y = y * 0.168 / 3600
            subexp = afwMath.rotateImageBy90(self.cutoutDonut(x, y, icExp), nquarter)
            aper = HSC_aper(theta_x, theta_y, lam, self.config.padFactor)
            zfit = ZFit(subexp, self.config.jmax, self.config.bitmask, lam, aper, xtol=1e-2)
            self.log.info("Fitting")
            zfit.fit()
            zfit.report()
            result = zfit.result.params.valuesdict()
            record.set(self.r0, result['r0'])
            for i, z in zip(range(4, self.config.jmax+1), self.z):
                record.set(z, result['z{}'.format(i)])
        return pipeBase.Struct(
            icExp=icExp,
            donutCat=donutCat
        )

    @pipeBase.timeMethod
    def selectDonuts(self, icSrc):
        s2n = (icSrc['base_CircularApertureFlux_25_0_flux'] /
               icSrc['base_CircularApertureFlux_25_0_fluxSigma'])
        rej1 = (icSrc['base_CircularApertureFlux_25_0_flux'] /
                icSrc['base_CircularApertureFlux_3_0_flux'])
        rej2 = (icSrc['base_CircularApertureFlux_35_0_flux'] /
                icSrc['base_CircularApertureFlux_25_0_flux'])

        select = (np.isfinite(s2n) &
                  np.isfinite(rej1) &
                  np.isfinite(rej2))
        for i, s in enumerate(select):
            if not s: continue
            if ((s2n[i] < self.config.snthresh) |
                (rej1[i] < self.config.r1cut) |
                (rej2[i] > self.config.r2cut)):
                select[i] = False
        self.log.info("Selected {} of {} detected donuts.".format(sum(select), len(select)))
        return select

    @pipeBase.timeMethod
    def cutoutDonut(self, x, y, icExp):
        point = afwGeom.Point2I(int(x), int(y))
        box = afwGeom.Box2I(point, point)
        box.grow(afwGeom.Extent2I(self.config.stamp_size//2, self.config.stamp_size//2))

        subMaskedImage = icExp.getMaskedImage().Factory(
              icExp.getMaskedImage(),
              box,
              afwImage.PARENT
        )
        return subMaskedImage
