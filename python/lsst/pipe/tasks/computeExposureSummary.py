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

import warnings
import numpy as np
from dataclasses import dataclass, asdict
import yaml
import astropy.units as units
from astropy.time import Time
from astropy.coordinates import AltAz, SkyCoord, EarthLocation
from lsst.daf.base import DateTime

from lsst.afw.typehandling import Storable, StorableHelperFactory
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.geom


__all__ = ("ComputeExposureSummaryTask", "ComputeExposureSummaryConfig", "ExposureSummary")


@dataclass
class ExposureSummary(Storable):
    _factory = StorableHelperFactory(__name__, "ExposureSummary")

    psfSigma: float
    psfArea: float
    psfIxx: float
    psfIyy: float
    psfIxy: float
    ra: float
    decl: float
    zenithDistance: float
    zeroPoint: float
    skyBg: float
    skyNoise: float
    meanVar: float
    raCorners: list[float]
    decCorners: list[float]

    def __post_init__(self):
        Storable.__init__(self)  # required for trampoline

    def isPersistable(self):
        return True

    def _getPersistenceName(self):
        return "ExposureSummary"

    def _getPythonModule(self):
        return __name__

    def _write(self):
        return yaml.dump(asdict(self), encoding='utf-8')

    @staticmethod
    def _read(bytes):
        return ExposureSummary(**yaml.load(bytes, Loader=yaml.SafeLoader))


class ComputeExposureSummaryConfig(pexConfig.Config):
    """Config for ComputeExposureSummaryTask"""
    sigmaClip = pexConfig.Field(
        dtype=float,
        doc="Sigma for outlier rejection for sky noise.",
        default=3.0,
    )
    clipIter = pexConfig.Field(
        dtype=int,
        doc="Number of iterations of outlier rejection for sky noise.",
        default=2,
    )
    badMaskPlanes = pexConfig.ListField(
        dtype=str,
        doc="Mask planes that, if set, the associated pixel should not be included sky noise calculation.",
        default=("NO_DATA", "SUSPECT"),
    )


class ComputeExposureSummaryTask(pipeBase.Task):
    """Task to compute exposure summary statistics.

    This task computes various quantities suitable for DPDD and other
    downstream processing, including:
    - psfSigma
    - psfArea
    - psfIxx
    - psfIyy
    - psfIxy
    - ra
    - decl
    - zenithDistance
    - zeroPoint
    - skyBg
    - skyNoise
    - meanVar
    - raCorners
    - decCorners
    """
    ConfigClass = ComputeExposureSummaryConfig
    _DefaultName = "computeExposureSummary"

    @pipeBase.timeMethod
    def run(self, exposure, sources, background):
        """Measure exposure statistics from the exposure, sources, and background.

        Parameters
        ----------
        exposure : `lsst.afw.image.ExposureF`
        sources : `lsst.afw.table.SourceCatalog`
        background : `lsst.afw.math.BackgroundList`

        Returns
        -------
        summary : `lsst.pipe.base.ExposureSummary`
        """
        self.log.info("Measuring exposure statistics")

        bbox = exposure.getBBox()

        psf = exposure.getPsf()
        if psf is not None:
            shape = psf.computeShape(bbox.getCenter())
            psfSigma = shape.getDeterminantRadius()
            psfIxx = shape.getIxx()
            psfIyy = shape.getIyy()
            psfIxy = shape.getIxy()
            im = psf.computeKernelImage(bbox.getCenter())
            # The calculation of effective psf area is taken from
            # meas_base/src/PsfFlux.cc#L112. See
            # https://github.com/lsst/meas_base/blob/
            # 750bffe6620e565bda731add1509507f5c40c8bb/src/PsfFlux.cc#L112
            psfArea = np.sum(im.array)/np.sum(im.array**2.)
        else:
            psfSigma = np.nan
            psfIxx = np.nan
            psfIyy = np.nan
            psfIxy = np.nan
            psfArea = np.nan

        wcs = exposure.getWcs()
        if wcs is not None:
            sph_pts = wcs.pixelToSky(lsst.geom.Box2D(bbox).getCorners())
            raCorners = [float(sph.getRa().asDegrees()) for sph in sph_pts]
            decCorners = [float(sph.getDec().asDegrees()) for sph in sph_pts]

            sph_pt = wcs.pixelToSky(bbox.getCenter())
            ra = sph_pt.getRa().asDegrees()
            decl = sph_pt.getDec().asDegrees()
        else:
            raCorners = [float(np.nan)]*4
            decCorners = [float(np.nan)]*4
            ra = np.nan
            decl = np.nan

        photoCalib = exposure.getPhotoCalib()
        if photoCalib is not None:
            zeroPoint = 2.5*np.log10(photoCalib.getInstFluxAtZeroMagnitude())
        else:
            zeroPoint = np.nan

        visitInfo = exposure.getInfo().getVisitInfo()
        observatory = visitInfo.getObservatory()
        loc = EarthLocation(lat=observatory.getLatitude().asDegrees()*units.deg,
                            lon=observatory.getLongitude().asDegrees()*units.deg,
                            height=observatory.getElevation()*units.m)
        obstime = Time(visitInfo.getDate().get(system=DateTime.MJD),
                       location=loc, format='mjd')
        coord = SkyCoord(ra*units.degree, decl*units.degree, obstime=obstime, location=loc)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            altaz = coord.transform_to(AltAz)

        zenithDistance = altaz.alt.degree

        if background is not None:
            bgStats = (bg[0].getStatsImage().getImage().array
                       for bg in background)
            skyBg = sum(np.median(bg[np.isfinite(bg)]) for bg in bgStats)
        else:
            skyBg = np.nan

        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(self.config.sigmaClip)
        statsCtrl.setNumIter(self.config.clipIter)
        statsCtrl.setAndMask(afwImage.Mask.getPlaneBitMask(self.config.badMaskPlanes))
        statsCtrl.setNanSafe(True)

        statObj = afwMath.makeStatistics(exposure.getMaskedImage(), afwMath.STDEVCLIP,
                                         statsCtrl)
        skyNoise, _ = statObj.getResult(afwMath.STDEVCLIP)

        statObj = afwMath.makeStatistics(exposure.getMaskedImage().getVariance(),
                                         exposure.getMaskedImage().getMask(),
                                         afwMath.MEANCLIP, statsCtrl)
        meanVar, _ = statObj.getResult(afwMath.MEANCLIP)

        summary = ExposureSummary(psfSigma=float(psfSigma),
                                  psfArea=float(psfArea),
                                  psfIxx=float(psfIxx),
                                  psfIyy=float(psfIyy),
                                  psfIxy=float(psfIxy),
                                  ra=float(ra),
                                  decl=float(decl),
                                  zenithDistance=float(zenithDistance),
                                  zeroPoint=float(zeroPoint),
                                  skyBg=float(skyBg),
                                  skyNoise=float(skyNoise),
                                  meanVar=float(meanVar),
                                  raCorners=raCorners,
                                  decCorners=decCorners)

        return summary
