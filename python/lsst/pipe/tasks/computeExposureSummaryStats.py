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


__all__ = ("ComputeExposureSummaryStatsTask", "ComputeExposureSummaryStatsConfig")


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

        summary = afwImage.ExposureSummaryStats()

        bbox = exposure.getBBox()

        psf = exposure.getPsf()
        self.update_psf_stats(summary, psf, bbox, sources)

        wcs = exposure.getWcs()
        visitInfo = exposure.getInfo().getVisitInfo()
        self.update_wcs_stats(summary, wcs, bbox, visitInfo)

        photoCalib = exposure.getPhotoCalib()
        self.update_photo_calib_stats(summary, photoCalib)

        self.update_background_stats(summary, background)

        self.update_masked_image_stats(summary, exposure.getMaskedImage())

        md = exposure.getMetadata()
        if 'SFM_ASTROM_OFFSET_MEAN' in md:
            summary.astromOffsetMean = md['SFM_ASTROM_OFFSET_MEAN']
            summary.astromOffsetStd = md['SFM_ASTROM_OFFSET_STD']

        return summary

    def update_psf_stats(self, summary, psf, bbox, sources=None):
        """Compute all summary-statistic fields that depend on the PSF model.

        Parameters
        ----------
        summary : `lsst.afw.image.ExposureSummaryStats`
            Summary object to update in-place.
        psf : `lsst.afw.detection.Psf` or `None`
            Point spread function model.  If `None`, all fields that depend on
            the PSF will be reset (generally to NaN).
        bbox : `lsst.geom.Box2I`
            Bounding box of the image for which summary stats are being
            computed.
        sources : `lsst.afw.table.SourceCatalog`, optional
            Catalog for quantities that are computed from source table columns.
            If `None`, these quantities will be reset (generally to NaN).
        """
        nan = float("nan")
        summary.psfSigma = nan
        summary.psfIxx = nan
        summary.psfIyy = nan
        summary.psfIxy = nan
        summary.psfArea = nan
        summary.nPsfStar = 0
        summary.psfStarDeltaE1Median = nan
        summary.psfStarDeltaE2Median = nan
        summary.psfStarDeltaE1Scatter = nan
        summary.psfStarDeltaE2Scatter = nan
        summary.psfStarDeltaSizeMedian = nan
        summary.psfStarDeltaSizeScatter = nan
        summary.psfStarScaledDeltaSizeScatter = nan

        if psf is None:
            return
        shape = psf.computeShape(bbox.getCenter())
        summary.psfSigma = shape.getDeterminantRadius()
        summary.psfIxx = shape.getIxx()
        summary.psfIyy = shape.getIyy()
        summary.psfIxy = shape.getIxy()
        im = psf.computeKernelImage(bbox.getCenter())
        # The calculation of effective psf area is taken from
        # meas_base/src/PsfFlux.cc#L112. See
        # https://github.com/lsst/meas_base/blob/
        # 750bffe6620e565bda731add1509507f5c40c8bb/src/PsfFlux.cc#L112
        summary.psfArea = float(np.sum(im.array)/np.sum(im.array**2.))

        if sources is None:
            # No sources are available (as in some tests)
            return

        names = sources.schema.getNames()
        if self.config.starSelection not in names or self.config.starShape + '_flag' not in names:
            # The source catalog does not have the necessary fields (as in some tests)
            return

        mask = sources[self.config.starSelection] & (~sources[self.config.starShape + '_flag'])
        nPsfStar = mask.sum()

        if nPsfStar == 0:
            # No stars to measure statistics, so we must return the defaults
            # of 0 stars and NaN values.
            return

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

        summary.nPsfStar = int(nPsfStar)
        summary.psfStarDeltaE1Median = float(psfStarDeltaE1Median)
        summary.psfStarDeltaE2Median = float(psfStarDeltaE2Median)
        summary.psfStarDeltaE1Scatter = float(psfStarDeltaE1Scatter)
        summary.psfStarDeltaE2Scatter = float(psfStarDeltaE2Scatter)
        summary.psfStarDeltaSizeMedian = float(psfStarDeltaSizeMedian)
        summary.psfStarDeltaSizeScatter = float(psfStarDeltaSizeScatter)
        summary.psfStarScaledDeltaSizeScatter = float(psfStarScaledDeltaSizeScatter)

    def update_wcs_stats(self, summary, wcs, bbox, visitInfo):
        """Compute all summary-statistic fields that depend on the WCS model.

        Parameters
        ----------
        summary : `lsst.afw.image.ExposureSummaryStats`
            Summary object to update in-place.
        wcs : `lsst.afw.geom.SkyWcs` or `None`
            Astrometric calibration model.  If `None`, all fields that depend
            on the WCS will be reset (generally to NaN).
        bbox : `lsst.geom.Box2I`
            Bounding box of the image for which summary stats are being
            computed.
        visitInfo : `lsst.afw.image.VisitInfo`
            Observation information used in together with ``wcs`` to compute
            the zenith distance.
        """
        nan = float("nan")
        summary.raCorners = [nan]*4
        summary.decCorners = [nan]*4
        summary.ra = nan
        summary.decl = nan
        summary.zenithDistance = nan

        if wcs is None:
            return

        sph_pts = wcs.pixelToSky(lsst.geom.Box2D(bbox).getCorners())
        summary.raCorners = [float(sph.getRa().asDegrees()) for sph in sph_pts]
        summary.decCorners = [float(sph.getDec().asDegrees()) for sph in sph_pts]

        sph_pt = wcs.pixelToSky(bbox.getCenter())
        summary.ra = sph_pt.getRa().asDegrees()
        summary.decl = sph_pt.getDec().asDegrees()

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
            coord = SkyCoord(
                summary.ra*units.degree,
                summary.decl*units.degree,
                obstime=obstime,
                location=loc,
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                altaz = coord.transform_to(AltAz)

            summary.zenithDistance = float(90.0 - altaz.alt.degree)

    def update_photo_calib_stats(self, summary, photo_calib):
        """Compute all summary-statistic fields that depend on the photometric
        calibration model.

        Parameters
        ----------
        summary : `lsst.afw.image.ExposureSummaryStats`
            Summary object to update in-place.
        photo_calib : `lsst.afw.image.PhotoCalib` or `None`
            Photometric calibration model.  If `None`, all fields that depend
            on the photometric calibration will be reset (generally to NaN).
        """
        if photo_calib is not None:
            summary.zeroPoint = float(2.5*np.log10(photo_calib.getInstFluxAtZeroMagnitude()))
        else:
            summary.zeroPoint = float("nan")

    def update_background_stats(self, summary, background):
        """Compute summary-statistic fields that depend only on the
        background model.

        Parameters
        ----------
        summary : `lsst.afw.image.ExposureSummaryStats`
            Summary object to update in-place.
        background : `lsst.afw.math.BackgroundList` or `None`
            Background model.  If `None`, all fields that depend on the
            background will be reset (generally to NaN).

        Notes
        -----
        This does not include fields that depend on the background-subtracted
        masked image; when the background changes, it should generally be
        applied to the image and `update_masked_image_stats` should be called
        as well.
        """
        if background is not None:
            bgStats = (bg[0].getStatsImage().getImage().array
                       for bg in background)
            summary.skyBg = float(sum(np.median(bg[np.isfinite(bg)]) for bg in bgStats))
        else:
            summary.skyBg = float("nan")

    def update_masked_image_stats(self, summary, masked_image):
        """Compute summary-statistic fields that depend on the masked image
        itself.

        Parameters
        ----------
        summary : `lsst.afw.image.ExposureSummaryStats`
            Summary object to update in-place.
        masked_image : `lsst.afw.image.MaskedImage` or `None`
            Masked image.  If `None`, all fields that depend
            on the masked image will be reset (generally to NaN).
        """
        nan = float("nan")
        if masked_image is None:
            summary.skyNoise = nan
            summary.meanVar = nan
            return
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(self.config.sigmaClip)
        statsCtrl.setNumIter(self.config.clipIter)
        statsCtrl.setAndMask(afwImage.Mask.getPlaneBitMask(self.config.badMaskPlanes))
        statsCtrl.setNanSafe(True)

        statObj = afwMath.makeStatistics(masked_image, afwMath.STDEVCLIP, statsCtrl)
        skyNoise, _ = statObj.getResult(afwMath.STDEVCLIP)
        summary.skyNoise = skyNoise

        statObj = afwMath.makeStatistics(masked_image.variance, masked_image.mask, afwMath.MEANCLIP,
                                         statsCtrl)
        meanVar, _ = statObj.getResult(afwMath.MEANCLIP)
        summary.meanVar = meanVar
