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

__all__ = ["ComputeExposureSummaryStatsTask", "ComputeExposureSummaryStatsConfig"]

import warnings
import numpy as np
from scipy.stats import median_abs_deviation as sigmaMad
import astropy.units as units
from astropy.time import Time
from astropy.coordinates import AltAz, SkyCoord, EarthLocation
from lsst.daf.base import DateTime

import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.geom
from lsst.utils.timer import timeMethod


class ComputeExposureSummaryStatsConfig(pexConfig.Config):
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
    starSelection = pexConfig.Field(
        doc="Field to select sources to be used in the PSF statistics computation.",
        dtype=str,
        default="calib_psf_used"
    )
    starShape = pexConfig.Field(
        doc="Base name of columns to use for the source shape in the PSF statistics computation.",
        dtype=str,
        default="base_SdssShape"
    )
    psfShape = pexConfig.Field(
        doc="Base name of columns to use for the PSF shape in the PSF statistics computation.",
        dtype=str,
        default="base_SdssShape_psf"
    )


class ComputeExposureSummaryStatsTask(pipeBase.Task):
    """Task to compute exposure summary statistics.

    This task computes various quantities suitable for DPDD and other
    downstream processing at the detector centers, including:
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
    - astromOffsetMean
    - astromOffsetStd

    These additional quantities are computed from the stars in the detector:
    - psfStarDeltaE1Median
    - psfStarDeltaE2Median
    - psfStarDeltaE1Scatter
    - psfStarDeltaE2Scatter
    - psfStarDeltaSizeMedian
    - psfStarDeltaSizeScatter
    - psfStarScaledDeltaSizeScatter
    """
    ConfigClass = ComputeExposureSummaryStatsConfig
    _DefaultName = "computeExposureSummaryStats"

    @timeMethod
    def run(self, exposure, sources, background):
        """Measure exposure statistics from the exposure, sources, and background.

        Parameters
        ----------
        exposure : `lsst.afw.image.ExposureF`
        sources : `lsst.afw.table.SourceCatalog`
        background : `lsst.afw.math.BackgroundList`

        Returns
        -------
        summary : `lsst.afw.image.ExposureSummary`
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
        date = visitInfo.getDate()

        if date.isValid():
            # We compute the zenithDistance at the center of the detector rather
            # than use the boresight value available via the visitInfo, because
            # the zenithDistance may vary significantly over a large field of view.
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

            zenithDistance = 90.0 - altaz.alt.degree
        else:
            zenithDistance = np.nan

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

        md = exposure.getMetadata()
        if 'SFM_ASTROM_OFFSET_MEAN' in md:
            astromOffsetMean = md['SFM_ASTROM_OFFSET_MEAN']
            astromOffsetStd = md['SFM_ASTROM_OFFSET_STD']
        else:
            astromOffsetMean = np.nan
            astromOffsetStd = np.nan

        psfStats = self._computePsfStats(sources)

        # Note that all numpy values have to be explicitly converted to
        # python floats for yaml serialization.
        summary = afwImage.ExposureSummaryStats(
            psfSigma=float(psfSigma),
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
            decCorners=decCorners,
            astromOffsetMean=astromOffsetMean,
            astromOffsetStd=astromOffsetStd,
            nPsfStar=psfStats.nPsfStar,
            psfStarDeltaE1Median=psfStats.psfStarDeltaE1Median,
            psfStarDeltaE2Median=psfStats.psfStarDeltaE2Median,
            psfStarDeltaE1Scatter=psfStats.psfStarDeltaE1Scatter,
            psfStarDeltaE2Scatter=psfStats.psfStarDeltaE2Scatter,
            psfStarDeltaSizeMedian=psfStats.psfStarDeltaSizeMedian,
            psfStarDeltaSizeScatter=psfStats.psfStarDeltaSizeScatter,
            psfStarScaledDeltaSizeScatter=psfStats.psfStarScaledDeltaSizeScatter
        )

        return summary

    def _computePsfStats(self, sources):
        """Compute psf residual statistics.

        All residuals are computed using median statistics on the difference
        between the sources and the models.

        Parameters
        ----------
        sources : `lsst.afw.table.SourceCatalog`
            Source catalog on which to measure the PSF statistics.

        Returns
        -------
        psfStats : `lsst.pipe.base.Struct`
            Struct with various psf stats.
        """
        psfStats = pipeBase.Struct(nPsfStar=0,
                                   psfStarDeltaE1Median=float(np.nan),
                                   psfStarDeltaE2Median=float(np.nan),
                                   psfStarDeltaE1Scatter=float(np.nan),
                                   psfStarDeltaE2Scatter=float(np.nan),
                                   psfStarDeltaSizeMedian=float(np.nan),
                                   psfStarDeltaSizeScatter=float(np.nan),
                                   psfStarScaledDeltaSizeScatter=float(np.nan))

        if sources is None:
            # No sources are available (as in some tests)
            return psfStats

        names = sources.schema.getNames()
        if self.config.starSelection not in names or self.config.starShape + '_flag' not in names:
            # The source catalog does not have the necessary fields (as in some tests)
            return psfStats

        mask = sources[self.config.starSelection] & (~sources[self.config.starShape + '_flag'])
        nPsfStar = mask.sum()

        if nPsfStar == 0:
            # No stars to measure statistics, so we must return the defaults
            # of 0 stars and NaN values.
            return psfStats

        starXX = sources[self.config.starShape + '_xx'][mask]
        starYY = sources[self.config.starShape + '_yy'][mask]
        starXY = sources[self.config.starShape + '_xy'][mask]
        psfXX = sources[self.config.psfShape + '_xx'][mask]
        psfYY = sources[self.config.psfShape + '_yy'][mask]
        psfXY = sources[self.config.psfShape + '_xy'][mask]

        starSize = (starXX*starYY - starXY**2.)**0.25
        starE1 = (starXX - starYY)/(starXX + starYY)
        starE2 = 2*starXY/(starXX + starYY)
        starSizeMedian = np.median(starSize)

        psfSize = (psfXX*psfYY - psfXY**2)**0.25
        psfE1 = (psfXX - psfYY)/(psfXX + psfYY)
        psfE2 = 2*psfXY/(psfXX + psfYY)

        psfStarDeltaE1Median = np.median(starE1 - psfE1)
        psfStarDeltaE1Scatter = sigmaMad(starE1 - psfE1, scale='normal')
        psfStarDeltaE2Median = np.median(starE2 - psfE2)
        psfStarDeltaE2Scatter = sigmaMad(starE2 - psfE2, scale='normal')

        psfStarDeltaSizeMedian = np.median(starSize - psfSize)
        psfStarDeltaSizeScatter = sigmaMad(starSize - psfSize, scale='normal')
        psfStarScaledDeltaSizeScatter = psfStarDeltaSizeScatter/starSizeMedian**2.

        psfStats.nPsfStar = int(nPsfStar)
        psfStats.psfStarDeltaE1Median = float(psfStarDeltaE1Median)
        psfStats.psfStarDeltaE2Median = float(psfStarDeltaE2Median)
        psfStats.psfStarDeltaE1Scatter = float(psfStarDeltaE1Scatter)
        psfStats.psfStarDeltaE2Scatter = float(psfStarDeltaE2Scatter)
        psfStats.psfStarDeltaSizeMedian = float(psfStarDeltaSizeMedian)
        psfStats.psfStarDeltaSizeScatter = float(psfStarDeltaSizeScatter)
        psfStats.psfStarScaledDeltaSizeScatter = float(psfStarScaledDeltaSizeScatter)

        return psfStats
