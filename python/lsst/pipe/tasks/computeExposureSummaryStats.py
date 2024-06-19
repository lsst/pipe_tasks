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
import lsst.geom as geom
from lsst.meas.algorithms import ScienceSourceSelectorTask
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
        doc="Field to select full list of sources used for PSF modeling.",
        dtype=str,
        default="calib_psf_used",
    )
    starSelector = pexConfig.ConfigurableField(
        target=ScienceSourceSelectorTask,
        doc="Selection of sources to compute PSF star statistics.",
    )
    starShape = pexConfig.Field(
        doc="Base name of columns to use for the source shape in the PSF statistics computation.",
        dtype=str,
        default="slot_Shape"
    )
    psfShape = pexConfig.Field(
        doc="Base name of columns to use for the PSF shape in the PSF statistics computation.",
        dtype=str,
        default="slot_PsfShape"
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
    minPsfApRadiusPix = pexConfig.Field(
        dtype=float,
        doc="Minimum radius in pixels of the aperture within which to measure the flux of "
        "the PSF model for the psfApFluxDelta metric calculation (the radius is computed as "
        "max(``minPsfApRadius``, 3*psfSigma)).",
        default=2.0,
    )
    psfApCorrFieldName = pexConfig.Field(
        doc="Name of the flux column associated with the aperture correction of the PSF model "
        "to use for the psfApCorrSigmaScaledDelta metric calculation.",
        dtype=str,
        default="base_PsfFlux_instFlux"
    )
    psfBadMaskPlanes = pexConfig.ListField(
        dtype=str,
        doc="Mask planes that, if set, the associated pixel should not be included in the PSF model "
        "robutsness metric calculations (namely, maxDistToNearestPsf and psfTraceRadiusDelta).",
        default=("BAD", "CR", "EDGE", "INTRP", "NO_DATA", "SAT", "SUSPECT"),
    )
    fiducialSkyBackground = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        doc="Fiducial sky background level (ADU/s) assumed when calculating effective exposure time. "
        "Keyed by band.",
        default={'u': 1.0, 'g': 1.0, 'r': 1.0, 'i': 1.0, 'z': 1.0, 'y': 1.0},
    )
    fiducialPsfSigma = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        doc="Fiducial PSF sigma (pixels) assumed when calculating effective exposure time. "
        "Keyed by band.",
        default={'u': 1.0, 'g': 1.0, 'r': 1.0, 'i': 1.0, 'z': 1.0, 'y': 1.0},
    )
    fiducialZeroPoint = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        doc="Fiducial zero point assumed when calculating effective exposure time. "
        "Keyed by band.",
        default={'u': 25.0, 'g': 25.0, 'r': 25.0, 'i': 25.0, 'z': 25.0, 'y': 25.0},
    )
    maxEffectiveTransparency = pexConfig.Field(
        dtype=float,
        doc="Maximum value allowed for effective transparency scale factor (often inf or 1.0).",
        default=float('inf')
    )

    def setDefaults(self):
        super().setDefaults()

        self.starSelector.setDefaults()
        self.starSelector.doFlags = True
        self.starSelector.doSignalToNoise = True
        self.starSelector.doUnresolved = False
        self.starSelector.doIsolated = False
        self.starSelector.doRequireFiniteRaDec = False
        self.starSelector.doRequirePrimary = False

        self.starSelector.signalToNoise.minimum = 50.0
        self.starSelector.signalToNoise.maximum = 1000.0

        self.starSelector.flags.bad = ["slot_Shape_flag", "slot_PsfFlux_flag"]
        # Select stars used for PSF modeling.
        self.starSelector.flags.good = ["calib_psf_used"]

        self.starSelector.signalToNoise.fluxField = "slot_PsfFlux_instFlux"
        self.starSelector.signalToNoise.errField = "slot_PsfFlux_instFluxErr"


class ComputeExposureSummaryStatsTask(pipeBase.Task):
    """Task to compute exposure summary statistics.

    This task computes various quantities suitable for DPDD and other
    downstream processing at the detector centers, including:
    - expTime
    - psfSigma
    - psfArea
    - psfIxx
    - psfIyy
    - psfIxy
    - ra
    - dec
    - pixelScale (arcsec/pixel)
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
    - psfApFluxDelta

    This quantity is computed based on the aperture correction map, the
    psfSigma, and the image mask to assess the robustness of the aperture
    corrections across a given detector:
    - psfApCorrSigmaScaledDelta

    These quantities are computed to assess depth:
    - effTime
    - effTimePsfSigmaScale
    - effTimeSkyBgScale
    - effTimeZeroPointScale
    """
    ConfigClass = ComputeExposureSummaryStatsConfig
    _DefaultName = "computeExposureSummaryStats"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.makeSubtask("starSelector")

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

        # Set exposure time.
        exposureTime = exposure.getInfo().getVisitInfo().getExposureTime()
        summary.expTime = exposureTime

        bbox = exposure.getBBox()

        psf = exposure.getPsf()
        self.update_psf_stats(
            summary, psf, bbox, sources, image_mask=exposure.mask, image_ap_corr_map=exposure.apCorrMap
        )

        wcs = exposure.getWcs()
        visitInfo = exposure.getInfo().getVisitInfo()
        self.update_wcs_stats(summary, wcs, bbox, visitInfo)

        photoCalib = exposure.getPhotoCalib()
        self.update_photo_calib_stats(summary, photoCalib)

        self.update_background_stats(summary, background)

        self.update_masked_image_stats(summary, exposure.getMaskedImage())

        self.update_effective_time_stats(summary, exposure)

        md = exposure.getMetadata()
        if 'SFM_ASTROM_OFFSET_MEAN' in md:
            summary.astromOffsetMean = md['SFM_ASTROM_OFFSET_MEAN']
            summary.astromOffsetStd = md['SFM_ASTROM_OFFSET_STD']

        return summary

    def update_psf_stats(
            self,
            summary,
            psf,
            bbox,
            sources=None,
            image_mask=None,
            image_ap_corr_map=None,
            sources_is_astropy=False,
    ):
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
        sources : `lsst.afw.table.SourceCatalog` or `astropy.table.Table`
            Catalog for quantities that are computed from source table columns.
            If `None`, these quantities will be reset (generally to NaN).
            The type of this table must correspond to the
            ``sources_is_astropy`` argument.
        image_mask : `lsst.afw.image.Mask`, optional
            Mask image that may be used to compute distance-to-nearest-star
            metrics.
        sources_is_astropy : `bool`, optional
            Whether ``sources`` is an `astropy.table.Table` instance instead
            of an `lsst.afw.table.Catalog` instance.  Default is `False` (the
            latter).
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
        summary.psfApFluxDelta = nan
        summary.psfApCorrSigmaScaledDelta = nan

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
            psfApRadius = max(self.config.minPsfApRadiusPix, 3.0*summary.psfSigma)
            self.log.debug("Using radius of %.3f (pixels) for psfApFluxDelta metric", psfApRadius)
            psfTraceRadiusDelta, psfApFluxDelta = compute_psf_image_deltas(
                image_mask,
                psf,
                sampling=self.config.psfGridSampling,
                ap_radius_pix=psfApRadius,
                bad_mask_bits=self.config.psfBadMaskPlanes
            )
            summary.psfTraceRadiusDelta = float(psfTraceRadiusDelta)
            summary.psfApFluxDelta = float(psfApFluxDelta)
            if image_ap_corr_map is not None:
                if self.config.psfApCorrFieldName not in image_ap_corr_map.keys():
                    self.log.warn(f"{self.config.psfApCorrFieldName} not found in "
                                  "image_ap_corr_map.  Setting psfApCorrSigmaScaledDelta to NaN.")
                    psfApCorrSigmaScaledDelta = nan
                else:
                    image_ap_corr_field = image_ap_corr_map[self.config.psfApCorrFieldName]
                    psfApCorrSigmaScaledDelta = compute_ap_corr_sigma_scaled_delta(
                        image_mask,
                        image_ap_corr_field,
                        summary.psfSigma,
                        sampling=self.config.psfGridSampling,
                        bad_mask_bits=self.config.psfBadMaskPlanes,
                    )
                summary.psfApCorrSigmaScaledDelta = float(psfApCorrSigmaScaledDelta)

        if sources is None:
            # No sources are available (as in some tests and rare cases where
            # the selection criteria in finalizeCharacterization lead to no
            # good sources).
            return

        # Count the total number of psf stars used (prior to stats selection).
        nPsfStar = sources[self.config.starSelection].sum()
        summary.nPsfStar = int(nPsfStar)

        psf_mask = self.starSelector.run(sources).selected
        nPsfStarsUsedInStats = psf_mask.sum()

        if nPsfStarsUsedInStats == 0:
            # No stars to measure statistics, so we must return the defaults
            # of 0 stars and NaN values.
            return

        if sources_is_astropy:
            psf_cat = sources[psf_mask]
        else:
            psf_cat = sources[psf_mask].copy(deep=True)

        starXX = psf_cat[self.config.starShape + '_xx']
        starYY = psf_cat[self.config.starShape + '_yy']
        starXY = psf_cat[self.config.starShape + '_xy']
        psfXX = psf_cat[self.config.psfShape + '_xx']
        psfYY = psf_cat[self.config.psfShape + '_yy']
        psfXY = psf_cat[self.config.psfShape + '_xy']

        # Use the trace radius for the star size.
        starSize = np.sqrt(starXX/2. + starYY/2.)

        starE1 = (starXX - starYY)/(starXX + starYY)
        starE2 = 2*starXY/(starXX + starYY)
        starSizeMedian = np.median(starSize)

        # Use the trace radius for the psf size.
        psfSize = np.sqrt(psfXX/2. + psfYY/2.)
        psfE1 = (psfXX - psfYY)/(psfXX + psfYY)
        psfE2 = 2*psfXY/(psfXX + psfYY)

        psfStarDeltaE1Median = np.median(starE1 - psfE1)
        psfStarDeltaE1Scatter = sigmaMad(starE1 - psfE1, scale='normal')
        psfStarDeltaE2Median = np.median(starE2 - psfE2)
        psfStarDeltaE2Scatter = sigmaMad(starE2 - psfE2, scale='normal')

        psfStarDeltaSizeMedian = np.median(starSize - psfSize)
        psfStarDeltaSizeScatter = sigmaMad(starSize - psfSize, scale='normal')
        psfStarScaledDeltaSizeScatter = psfStarDeltaSizeScatter/starSizeMedian

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
        summary.dec = nan
        summary.pixelScale = nan
        summary.zenithDistance = nan

        if wcs is None:
            return

        sph_pts = wcs.pixelToSky(geom.Box2D(bbox).getCorners())
        summary.raCorners = [float(sph.getRa().asDegrees()) for sph in sph_pts]
        summary.decCorners = [float(sph.getDec().asDegrees()) for sph in sph_pts]

        sph_pt = wcs.pixelToSky(bbox.getCenter())
        summary.ra = sph_pt.getRa().asDegrees()
        summary.dec = sph_pt.getDec().asDegrees()
        summary.pixelScale = wcs.getPixelScale().asArcseconds()

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
                summary.dec*units.degree,
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

    def update_effective_time_stats(self, summary, exposure):
        """Compute effective exposure time statistics to estimate depth.

        The effective exposure time is the equivalent shutter open
        time that would be needed under nominal conditions to give the
        same signal-to-noise for a point source as what is achieved by
        the observation of interest. This metric combines measurements
        of the point-spread function, the sky brightness, and the
        transparency.

        .. _teff_definitions:

        The effective exposure time and its subcomponents are defined in [1]_

        References
        ----------

        .. [1] Neilsen, E.H., Bernstein, G., Gruendl, R., and Kent, S. (2016).
               Limiting Magnitude, \tau, teff, and Image Quality in DES Year 1
               https://www.osti.gov/biblio/1250877/


        Parameters
        ----------
        summary : `lsst.afw.image.ExposureSummaryStats`
            Summary object to update in-place.
        exposure : `lsst.afw.image.ExposureF`
            Exposure to grab band and exposure time metadata

        """
        self.log.info("Updating effective exposure time")

        nan = float("nan")
        summary.effTime = nan
        summary.effTimePsfSigmaScale = nan
        summary.effTimeSkyBgScale = nan
        summary.effTimeZeroPointScale = nan

        exposureTime = exposure.getInfo().getVisitInfo().getExposureTime()
        filterLabel = exposure.getFilter()
        if (filterLabel is None) or (not filterLabel.hasBandLabel):
            band = None
        else:
            band = filterLabel.bandLabel

        if band is None:
            self.log.warn("No band associated with exposure; effTime not calculated.")
            return

        # PSF component
        if np.isnan(summary.psfSigma):
            self.log.debug("PSF sigma is NaN")
            f_eff = nan
        elif band not in self.config.fiducialPsfSigma:
            self.log.debug(f"Fiducial PSF value not found for {band}")
            f_eff = nan
        else:
            fiducialPsfSigma = self.config.fiducialPsfSigma[band]
            f_eff = (summary.psfSigma / fiducialPsfSigma)**-2

        # Transparency component (note that exposure time may be removed from zeropoint)
        if np.isnan(summary.zeroPoint):
            self.log.debug("Zero point is NaN")
            c_eff = nan
        elif band not in self.config.fiducialZeroPoint:
            self.log.debug(f"Fiducial zero point value not found for {band}")
            c_eff = nan
        else:
            fiducialZeroPoint = self.config.fiducialZeroPoint[band]
            zeroPointDiff = fiducialZeroPoint - (summary.zeroPoint - 2.5*np.log10(exposureTime))
            c_eff = min(10**(-2.0*(zeroPointDiff)/2.5), self.config.maxEffectiveTransparency)

        # Sky brightness component (convert to cts/s)
        if np.isnan(summary.skyBg):
            self.log.debug("Sky background is NaN")
            b_eff = nan
        elif band not in self.config.fiducialSkyBackground:
            self.log.debug(f"Fiducial sky background value not found for {band}")
            b_eff = nan
        else:
            fiducialSkyBackground = self.config.fiducialSkyBackground[band]
            b_eff = fiducialSkyBackground/(summary.skyBg/exposureTime)

        # Effective exposure time scale factor
        t_eff = f_eff * c_eff * b_eff

        # Effective exposure time (seconds)
        effectiveTime = t_eff * exposureTime

        # Output quantities
        summary.effTime = float(effectiveTime)
        summary.effTimePsfSigmaScale = float(f_eff)
        summary.effTimeSkyBgScale = float(b_eff)
        summary.effTimeZeroPointScale = float(c_eff)


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
        The mask plane associated with the exposure.
    psf_cat : `lsst.afw.table.SourceCatalog` or `astropy.table.Table`
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
        x_psf = psf["slot_Centroid_x"]
        y_psf = psf["slot_Centroid_y"]
        dist_to_nearest_psf = np.minimum(dist_to_nearest_psf, np.hypot(xx - x_psf, yy - y_psf))
        unmasked_dists = dist_to_nearest_psf * good
        max_dist_to_nearest_psf = np.max(unmasked_dists)

    return max_dist_to_nearest_psf


def compute_psf_image_deltas(
    image_mask,
    image_psf,
    sampling=96,
    ap_radius_pix=3.0,
    bad_mask_bits=["BAD", "CR", "INTRP", "SAT", "SUSPECT", "NO_DATA", "EDGE"],
):
    """Compute the delta between the maximum and minimum model PSF trace radius
    values evaluated on a grid of points lying in the unmasked region of the
    image.

    Parameters
    ----------
    image_mask : `lsst.afw.image.Mask`
        The mask plane associated with the exposure.
    image_psf : `lsst.afw.detection.Psf`
        The PSF model associated with the exposure.
    sampling : `int`, optional
        Sampling rate in each dimension to create the grid of points at which
        to evaluate ``image_psf``s trace radius value. The tradeoff is between
        adequate sampling versus speed.
    ap_radius_pix : `float`, optional
        Radius in pixels of the aperture on which to measure the flux of the
        PSF model.
    bad_mask_bits : `list` [`str`], optional
        Mask bits required to be absent for a pixel to be considered
        "unmasked".

    Returns
    -------
    psf_trace_radius_delta, psf_ap_flux_delta : `float`
        The delta (in pixels) between the maximum and minimum model PSF trace
        radius values and the PSF aperture fluxes (with aperture radius of
        max(2, 3*psfSigma)) evaluated on the x,y-grid subsampled on the
        unmasked detector pixels by a factor of ``sampling``.  If both the
        model PSF trace radius value and aperture flux value on the grid
        evaluate to NaN, then NaNs are returned immediately.
    """
    psf_trace_radius_list = []
    psf_ap_flux_list = []
    mask_arr = image_mask.array[::sampling, ::sampling]
    bitmask = image_mask.getPlaneBitMask(bad_mask_bits)
    good = ((mask_arr & bitmask) == 0)

    x = np.arange(good.shape[1]) * sampling
    y = np.arange(good.shape[0]) * sampling
    xx, yy = np.meshgrid(x, y)

    for x_mesh, y_mesh, good_mesh in zip(xx, yy, good):
        for x_point, y_point, is_good in zip(x_mesh, y_mesh, good_mesh):
            if is_good:
                position = geom.Point2D(x_point, y_point)
                psf_trace_radius = image_psf.computeShape(position).getTraceRadius()
                psf_ap_flux = image_psf.computeApertureFlux(ap_radius_pix, position)
                if ~np.isfinite(psf_trace_radius) and ~np.isfinite(psf_ap_flux):
                    return float("nan"), float("nan")
                psf_trace_radius_list.append(psf_trace_radius)
                psf_ap_flux_list.append(psf_ap_flux)

    psf_trace_radius_delta = np.max(psf_trace_radius_list) - np.min(psf_trace_radius_list)
    if np.any(np.asarray(psf_ap_flux_list) < 0.0):  # Consider any -ve flux entry as "bad".
        psf_ap_flux_delta = float("nan")
    else:
        psf_ap_flux_delta = np.max(psf_ap_flux_list) - np.min(psf_ap_flux_list)

    return psf_trace_radius_delta, psf_ap_flux_delta


def compute_ap_corr_sigma_scaled_delta(
    image_mask,
    image_ap_corr_field,
    psfSigma,
    sampling=96,
    bad_mask_bits=["BAD", "CR", "INTRP", "SAT", "SUSPECT", "NO_DATA", "EDGE"],
):
    """Compute the delta between the maximum and minimum aperture correction
    values scaled (divided) by ``psfSigma`` for the given field representation,
    ``image_ap_corr_field`` evaluated on a grid of points lying in the
    unmasked region of the image.

    Parameters
    ----------
    image_mask : `lsst.afw.image.Mask`
        The mask plane associated with the exposure.
    image_ap_corr_field : `lsst.afw.math.ChebyshevBoundedField`
        The ChebyshevBoundedField representation of the aperture correction
        of interest for the exposure.
    psfSigma : `float`
        The PSF model second-moments determinant radius (center of chip)
        in pixels.
    sampling : `int`, optional
        Sampling rate in each dimension to create the grid of points at which
        to evaluate ``image_psf``s trace radius value. The tradeoff is between
        adequate sampling versus speed.
    bad_mask_bits : `list` [`str`], optional
        Mask bits required to be absent for a pixel to be considered
        "unmasked".

    Returns
    -------
    ap_corr_sigma_scaled_delta : `float`
        The delta between the maximum and minimum of the (multiplicative)
        aperture correction values scaled (divided) by ``psfSigma`` evaluated
        on the x,y-grid subsampled on the unmasked detector pixels by a factor
        of ``sampling``.  If the aperture correction evaluates to NaN on any
        of the grid points, this is set to NaN.
    """
    mask_arr = image_mask.array[::sampling, ::sampling]
    bitmask = image_mask.getPlaneBitMask(bad_mask_bits)
    good = ((mask_arr & bitmask) == 0)

    x = np.arange(good.shape[1], dtype=np.float64) * sampling
    y = np.arange(good.shape[0], dtype=np.float64) * sampling
    xx, yy = np.meshgrid(x, y)

    ap_corr = image_ap_corr_field.evaluate(xx.ravel(), yy.ravel()).reshape(xx.shape)
    ap_corr_good = ap_corr[good]
    if ~np.isfinite(ap_corr_good).all():
        ap_corr_sigma_scaled_delta = float("nan")
    else:
        ap_corr_sigma_scaled_delta = (np.max(ap_corr_good) - np.min(ap_corr_good))/psfSigma

    return ap_corr_sigma_scaled_delta
