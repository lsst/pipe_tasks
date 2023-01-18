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
import lsst.geom as geom
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
    psfSampling = pexConfig.Field(
        dtype=int,
        doc="Sampling rate in pixels in each dimension for the maxDistToNearestPsf metric "
        "caclulation grid (the tradeoff is between adequate sampling versus speed).",
        default=8,
    )
    psfGridSampling = pexConfig.Field(
        dtype=int,
        doc="Sampling rate in pixels in each dimension for PSF model robustness metric "
        "caclulations grid (the tradeoff is between adequate sampling versus speed).",
        default=96,
    )
    psfBadMaskPlanes = pexConfig.ListField(
        dtype=str,
        doc="Mask planes that, if set, the associated pixel should not be included in the PSF model "
        "robutsness metric calculations (namely, maxDistToNearestPsf and psfTraceRadiusDelta).",
        default=("BAD", "CR", "EDGE", "INTRP", "NO_DATA", "SAT", "SUSPECT"),
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

    These quantities are computed based on the PSF model and image mask
    to assess the robustness of the PSF model across a given detector
    (against, e.g., extrapolation instability):
    - maxDistToNearestPsf
    - psfTraceRadiusDelta
    """
    ConfigClass = ComputeExposureSummaryStatsConfig
    _DefaultName = "computeExposureSummaryStats"

    @timeMethod
    def run(self, exposure, sources, background):
        """Measure exposure statistics from the exposure, sources, and
        background.

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
        self.update_psf_stats(summary, psf, bbox, sources, image_mask=exposure.mask)

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

    def update_psf_stats(self, summary, psf, bbox, sources=None, image_mask=None, sources_columns=None):
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
        image_mask : `lsst.afw.image.Mask`, optional
            Mask image that may be used to compute distance-to-nearest-star
            metrics.
        sources_columns : `collections.abc.Set` [ `str` ], optional
            Set of all column names in ``sources``.  If provided, ``sources``
            may be any table type for which string indexes yield column arrays.
            If not provided, ``sources`` is assumed to be an
            `lsst.afw.table.SourceCatalog`.
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
        summary.maxDistToNearestPsf = nan
        summary.psfTraceRadiusDelta = nan

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

        if image_mask is not None:
            psfTraceRadiusDelta = psf_trace_radius_delta(
                image_mask,
                psf,
                sampling=self.config.psfGridSampling,
                bad_mask_bits=self.config.psfBadMaskPlanes
            )
            summary.psfTraceRadiusDelta = float(psfTraceRadiusDelta)

        if sources is None:
            # No sources are available (as in some tests)
            return

        if sources_columns is None:
            sources_columns = sources.schema.getNames()
        if (
            self.config.starSelection not in sources_columns
            or self.config.starShape + '_flag' not in sources_columns
        ):
            # The source catalog does not have the necessary fields (as in some
            # tests).
            return

        psf_mask = sources[self.config.starSelection] & (~sources[self.config.starShape + '_flag'])
        nPsfStar = psf_mask.sum()

        if nPsfStar == 0:
            # No stars to measure statistics, so we must return the defaults
            # of 0 stars and NaN values.
            return
        psf_cat = sources[psf_mask].copy(deep=True)

        starXX = psf_cat[self.config.starShape + '_xx']
        starYY = psf_cat[self.config.starShape + '_yy']
        starXY = psf_cat[self.config.starShape + '_xy']
        psfXX = psf_cat[self.config.psfShape + '_xx']
        psfYY = psf_cat[self.config.psfShape + '_yy']
        psfXY = psf_cat[self.config.psfShape + '_xy']

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

        if image_mask is not None:
            maxDistToNearestPsf = maximum_nearest_psf_distance(
                image_mask,
                psf_cat,
                sampling=self.config.psfSampling,
                bad_mask_bits=self.config.psfBadMaskPlanes
            )
            summary.maxDistToNearestPsf = float(maxDistToNearestPsf)

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

        sph_pts = wcs.pixelToSky(geom.Box2D(bbox).getCorners())
        summary.raCorners = [float(sph.getRa().asDegrees()) for sph in sph_pts]
        summary.decCorners = [float(sph.getDec().asDegrees()) for sph in sph_pts]

        sph_pt = wcs.pixelToSky(bbox.getCenter())
        summary.ra = sph_pt.getRa().asDegrees()
        summary.decl = sph_pt.getDec().asDegrees()

        date = visitInfo.getDate()

        if date.isValid():
            # We compute the zenithDistance at the center of the detector
            # rather than use the boresight value available via the visitInfo,
            # because the zenithDistance may vary significantly over a large
            # field of view.
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


def maximum_nearest_psf_distance(
    image_mask,
    psf_cat,
    sampling=8,
    bad_mask_bits=["BAD", "CR", "INTRP", "SAT", "SUSPECT", "NO_DATA", "EDGE"],
):
    """Compute the maximum distance of an unmasked pixel to its nearest PSF.

    Parameters
    ----------
    image_mask : `lsst.afw.image.Mask`
        The mask plane assosiated with the exposure.
    psf_cat : `lsst.afw.table.SourceCatalog`
        Catalog containing only the stars used in the PSF modeling.
    sampling : `int`
        Sampling rate in each dimension to create the grid of points on which
        to evaluate the distance to the nearest PSF star. The tradeoff is
        between adequate sampling versus speed.
    bad_mask_bits : `list` [`str`]
        Mask bits required to be absent for a pixel to be considered
        "unmasked".

    Returns
    -------
    max_dist_to_nearest_psf : `float`
        The maximum distance (in pixels) of an unmasked pixel to its nearest
        PSF model star.
    """
    mask_arr = image_mask.array[::sampling, ::sampling]
    bitmask = image_mask.getPlaneBitMask(bad_mask_bits)
    good = ((mask_arr & bitmask) == 0)

    x = np.arange(good.shape[1]) * sampling
    y = np.arange(good.shape[0]) * sampling
    xx, yy = np.meshgrid(x, y)

    dist_to_nearest_psf = np.full(good.shape, np.inf)
    for psf in psf_cat:
        x_psf = psf.getX()
        y_psf = psf.getY()
        dist_to_nearest_psf = np.minimum(dist_to_nearest_psf, np.hypot(xx - x_psf, yy - y_psf))
        unmasked_dists = dist_to_nearest_psf * good
        max_dist_to_nearest_psf = np.max(unmasked_dists)

    return max_dist_to_nearest_psf


def psf_trace_radius_delta(
    image_mask,
    image_psf,
    sampling=96,
    bad_mask_bits=["BAD", "CR", "INTRP", "SAT", "SUSPECT", "NO_DATA", "EDGE"],
):
    """Compute the delta between the maximum and minimum model PSF trace radius
    values evaluated on a grid of points lying in the unmasked region of the
    image.

    Parameters
    ----------
    image_mask : `lsst.afw.image.Mask`
        The mask plane assosiated with the exposure.
    image_psf : `lsst.afw.detection.Psf`
        The PSF model assosiated with the exposure.
    sampling : `int`
        Sampling rate in each dimension to create the grid of points at which
        to evaluate ``image_psf``s trace radius value. The tradeoff is between
        adequate sampling versus speed.
    bad_mask_bits : `list` [`str`]
        Mask bits required to be absent for a pixel to be considered
        "unmasked".

    Returns
    -------
    psf_trace_radius_delta : `float`
        The delta (in pixels) between the maximum and minimum model PSF trace
        radius values evaluated on the x,y-grid subsampled on the unmasked
        detector pixels by a factor of ``sampling``.  If any model PSF trace
        radius value on the grid evaluates to NaN, then NaN is returned
        immediately.
    """
    psf_trace_radius_list = []
    mask_arr = image_mask.array[::sampling, ::sampling]
    bitmask = image_mask.getPlaneBitMask(bad_mask_bits)
    good = ((mask_arr & bitmask) == 0)

    x = np.arange(good.shape[1]) * sampling
    y = np.arange(good.shape[0]) * sampling
    xx, yy = np.meshgrid(x, y)

    for x_mesh, y_mesh, good_mesh in zip(xx, yy, good):
        for x_point, y_point, is_good in zip(x_mesh, y_mesh, good_mesh):
            if is_good:
                psf_trace_radius = image_psf.computeShape(geom.Point2D(x_point, y_point)).getTraceRadius()
                if ~np.isfinite(psf_trace_radius):
                    return float("nan")
                psf_trace_radius_list.append(psf_trace_radius)

    psf_trace_radius_delta = np.max(psf_trace_radius_list) - np.min(psf_trace_radius_list)

    return psf_trace_radius_delta
