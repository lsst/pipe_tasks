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

__all__ = ["PhotoCalTask", "PhotoCalConfig"]

import math
import sys

import numpy as np
import astropy.units as u

import lsst.pex.config as pexConf
import lsst.pipe.base as pipeBase
from lsst.afw.image import abMagErrFromFluxErr, makePhotoCalibFromCalibZeroPoint
import lsst.afw.table as afwTable
from lsst.meas.astrom import DirectMatchTask, DirectMatchConfigWithoutLoader
import lsst.afw.display as afwDisplay
from lsst.meas.algorithms import getRefFluxField, ReserveSourcesTask
from lsst.utils.timer import timeMethod
from .colorterms import ColortermLibrary, Colorterm


class PhotoCalConfig(pexConf.Config):
    """Config for PhotoCal."""

    match = pexConf.ConfigField("Match to reference catalog",
                                DirectMatchConfigWithoutLoader)
    reserve = pexConf.ConfigurableField(target=ReserveSourcesTask, doc="Reserve sources from fitting")
    fluxField = pexConf.Field(
        dtype=str,
        default="slot_CalibFlux_instFlux",
        doc=("Name of the source instFlux field to use.\nThe associated flag field "
             "('<name>_flags') will be implicitly included in badFlags."),
    )
    applyColorTerms = pexConf.Field(
        dtype=bool,
        default=False,
        doc=("Apply photometric color terms to reference stars?\n"
             "`True`: attempt to apply color terms; fail if color term data is "
             "not available for the specified reference catalog and filter.\n"
             "`False`: do not apply color terms."),
        optional=True,
    )
    sigmaMax = pexConf.Field(
        dtype=float,
        default=0.25,
        doc="maximum sigma to use when clipping",
        optional=True,
    )
    nSigma = pexConf.Field(
        dtype=float,
        default=3.0,
        doc="clip at nSigma",
    )
    useMedian = pexConf.Field(
        dtype=bool,
        default=True,
        doc="use median instead of mean to compute zeropoint",
    )
    nIter = pexConf.Field(
        dtype=int,
        default=20,
        doc="number of iterations",
    )
    colorterms = pexConf.ConfigField(
        dtype=ColortermLibrary,
        doc="Library of photometric reference catalog name: color term dict (see also applyColorTerms).",
    )
    photoCatName = pexConf.Field(
        dtype=str,
        optional=True,
        doc=("Name of photometric reference catalog; used to select a color term dict in colorterms.\n"
             "See also applyColorTerms."),
    )
    magErrFloor = pexConf.RangeField(
        dtype=float,
        default=0.0,
        doc="Additional magnitude uncertainty to be added in quadrature with measurement errors.",
        min=0.0,
    )

    def validate(self):
        pexConf.Config.validate(self)
        if self.applyColorTerms and self.photoCatName is None:
            raise RuntimeError("applyColorTerms=True requires photoCatName is non-None")
        if self.applyColorTerms and len(self.colorterms.data) == 0:
            raise RuntimeError("applyColorTerms=True requires colorterms be provided")
        if self.fluxField != self.match.sourceSelection.signalToNoise.fluxField:
            raise RuntimeError("Configured flux field %s does not match source selector field %s",
                               self.fluxField, self.match.sourceSelection.signalToNoise.fluxField)
        if self.fluxField + "Err" != self.match.sourceSelection.signalToNoise.errField:
            raise RuntimeError("Configured flux field %sErr does not match source selector error field %s",
                               self.fluxField, self.match.sourceSelection.signalToNoise.errField)

    def setDefaults(self):
        pexConf.Config.setDefaults(self)
        self.match.sourceSelection.doRequirePrimary = True
        self.match.sourceSelection.doFlags = True
        self.match.sourceSelection.flags.bad = [
            "base_PixelFlags_flag_edge",
            "base_PixelFlags_flag_interpolated",
            "base_PixelFlags_flag_saturated",
        ]
        self.match.sourceSelection.doUnresolved = True
        self.match.sourceSelection.doSignalToNoise = True
        self.match.sourceSelection.signalToNoise.minimum = 10.0
        self.match.sourceSelection.signalToNoise.fluxField = self.fluxField
        self.match.sourceSelection.signalToNoise.errField = self.fluxField + "Err"


class PhotoCalTask(pipeBase.Task):
    """Calculate an Exposure's zero-point given a set of flux measurements
    of stars matched to an input catalogue.

    Parameters
    ----------
    refObjLoader : `lsst.meas.algorithms.ReferenceObjectLoader`
        A reference object loader object; gen3 pipeline tasks will pass `None`
        and call `match.setRefObjLoader` in `runQuantum`.
    schema : `lsst.afw.table.Schema`, optional
        The schema of the detection catalogs used as input to this task.
    **kwds
        Additional keyword arguments.

    Notes
    -----
    The type of flux to use is specified by PhotoCalConfig.fluxField.

    The algorithm clips outliers iteratively, with parameters set in the configuration.

    This task can adds fields to the schema, so any code calling this task must ensure that
    these columns are indeed present in the input match list; see `pipe_tasks_photocal_Example`.

    Debugging:

    The available `~lsst.base.lsstDebug` variables in PhotoCalTask are:

    display :
        If True enable other debug outputs.
    displaySources :
        If True, display the exposure on ds9's frame 1 and overlay the source catalogue.

    red o :
        Reserved objects.
    green o :
        Objects used in the photometric calibration.

    scatterPlot :
        Make a scatter plot of flux v. reference magnitude as a function of reference magnitude:

        - good objects in blue
        - rejected objects in red

    (if scatterPlot is 2 or more, prompt to continue after each iteration)
    """

    ConfigClass = PhotoCalConfig
    _DefaultName = "photoCal"

    def __init__(self, refObjLoader=None, schema=None, **kwds):
        pipeBase.Task.__init__(self, **kwds)
        self.scatterPlot = None
        self.fig = None
        if schema is not None:
            self.usedKey = schema.addField("calib_photometry_used", type="Flag",
                                           doc="set if source was used in photometric calibration")
        else:
            self.usedKey = None
        self.match = DirectMatchTask(config=self.config.match, refObjLoader=refObjLoader,
                                     name="match", parentTask=self)
        self.makeSubtask("reserve", columnName="calib_photometry", schema=schema,
                         doc="set if source was reserved from photometric calibration")

    def getSourceKeys(self, schema):
        """Return a struct containing the source catalog keys for fields used
        by PhotoCalTask.

        Parameters
        ----------
        schema : `lsst.afw.table.schema`
            Schema of the catalog to get keys from.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``instFlux``
                Instrument flux key.
            ``instFluxErr``
                Instrument flux error key.
        """
        instFlux = schema.find(self.config.fluxField).key
        instFluxErr = schema.find(self.config.fluxField + "Err").key
        return pipeBase.Struct(instFlux=instFlux, instFluxErr=instFluxErr)

    @timeMethod
    def extractMagArrays(self, matches, filterLabel, sourceKeys):
        """Extract magnitude and magnitude error arrays from the given matches.

        Parameters
        ----------
        matches : `lsst.afw.table.ReferenceMatchVector`
            Reference/source matches.
        filterLabel : `str`
            Label of filter being calibrated.
        sourceKeys : `lsst.pipe.base.Struct`
            Struct of source catalog keys, as returned by getSourceKeys().

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``srcMag``
                Source magnitude (`np.array`).
            ``refMag``
                Reference magnitude (`np.array`).
            ``srcMagErr``
                Source magnitude error (`np.array`).
            ``refMagErr``
                Reference magnitude error (`np.array`).
            ``magErr``
                An error in the magnitude; the error in ``srcMag`` - ``refMag``.
                If nonzero, ``config.magErrFloor`` will be added to ``magErr`` only
                (not ``srcMagErr`` or ``refMagErr``), as
                ``magErr`` is what is later used to determine the zero point (`np.array`).
            ``refFluxFieldList``
                A list of field names of the reference catalog used for fluxes (1 or 2 strings) (`list`).
        """
        srcInstFluxArr = np.array([m.second.get(sourceKeys.instFlux) for m in matches])
        srcInstFluxErrArr = np.array([m.second.get(sourceKeys.instFluxErr) for m in matches])
        if not np.all(np.isfinite(srcInstFluxErrArr)):
            # this is an unpleasant hack; see DM-2308 requesting a better solution
            self.log.warning("Source catalog does not have flux uncertainties; using sqrt(flux).")
            srcInstFluxErrArr = np.sqrt(srcInstFluxArr)

        # convert source instFlux from DN to an estimate of nJy
        referenceFlux = (0*u.ABmag).to_value(u.nJy)
        srcInstFluxArr = srcInstFluxArr * referenceFlux
        srcInstFluxErrArr = srcInstFluxErrArr * referenceFlux

        if not matches:
            raise RuntimeError("No reference stars are available")
        refSchema = matches[0].first.schema

        if self.config.applyColorTerms:
            self.log.info("Applying color terms for filter=%r, config.photoCatName=%s",
                          filterLabel.physicalLabel, self.config.photoCatName)
            if colorterm_model := self.match.refObjLoader.getColorterm(filterLabel.physicalLabel):
                colorterm = Colorterm._from_model(colorterm_model)
            else:
                colorterm = self.config.colorterms.getColorterm(filterLabel.physicalLabel,
                                                                self.config.photoCatName,
                                                                doRaise=True)
            refCat = afwTable.SimpleCatalog(matches[0].first.schema)

            # extract the matched refCat as a Catalog for the colorterm code
            refCat.reserve(len(matches))
            for x in matches:
                record = refCat.addNew()
                record.assign(x.first)

            refMagArr, refMagErrArr = colorterm.getCorrectedMagnitudes(refCat)
            fluxFieldList = [getRefFluxField(refSchema, filt) for filt in (colorterm.primary,
                                                                           colorterm.secondary)]
        else:
            self.log.info("Not applying color terms.")
            colorterm = None

            fluxFieldList = [getRefFluxField(refSchema, filterLabel.bandLabel)]
            fluxField = getRefFluxField(refSchema, filterLabel.bandLabel)
            fluxKey = refSchema.find(fluxField).key
            refFluxArr = np.array([m.first.get(fluxKey) for m in matches])

            try:
                fluxErrKey = refSchema.find(fluxField + "Err").key
                refFluxErrArr = np.array([m.first.get(fluxErrKey) for m in matches])
            except KeyError:
                # Reference catalogue may not have flux uncertainties; HACK DM-2308
                self.log.warning("Reference catalog does not have flux uncertainties for %s;"
                                 " using sqrt(flux).", fluxField)
                refFluxErrArr = np.sqrt(refFluxArr)

            refMagArr = u.Quantity(refFluxArr, u.nJy).to_value(u.ABmag)
            # HACK convert to Jy until we have a replacement for this (DM-16903)
            refMagErrArr = abMagErrFromFluxErr(refFluxErrArr*1e-9, refFluxArr*1e-9)

        # compute the source catalog magnitudes and errors
        srcMagArr = u.Quantity(srcInstFluxArr, u.nJy).to_value(u.ABmag)
        # Fitting with error bars in both axes is hard
        # for now ignore reference flux error, but ticket DM-2308 is a request for a better solution
        # HACK convert to Jy until we have a replacement for this (DM-16903)
        magErrArr = abMagErrFromFluxErr(srcInstFluxErrArr*1e-9, srcInstFluxArr*1e-9)
        if self.config.magErrFloor != 0.0:
            magErrArr = (magErrArr**2 + self.config.magErrFloor**2)**0.5

        srcMagErrArr = abMagErrFromFluxErr(srcInstFluxErrArr*1e-9, srcInstFluxArr*1e-9)

        good = np.isfinite(srcMagArr) & np.isfinite(refMagArr)

        return pipeBase.Struct(
            srcMag=srcMagArr[good],
            refMag=refMagArr[good],
            magErr=magErrArr[good],
            srcMagErr=srcMagErrArr[good],
            refMagErr=refMagErrArr[good],
            refFluxFieldList=fluxFieldList,
        )

    @timeMethod
    def run(self, exposure, sourceCat, expId=0):
        """Do photometric calibration - select matches to use and (possibly iteratively) compute
        the zero point.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure upon which the sources in the matches were detected.
        sourceCat : `lsst.afw.table.SourceCatalog`
            Good stars selected for use in calibration.
        expId : `int`, optional
            Exposure ID.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``photoCalib``
                Object containing the zero point (`lsst.afw.image.Calib`).
            ``arrays``
                Magnitude arrays returned be `PhotoCalTask.extractMagArrays`.
            ``matches``
                ReferenceMatchVector, as returned by the matcher
            ``matchMeta`` :  metadata needed to unpersist matches, as returned
                by the matcher (`lsst.daf.base.PropertyList`)
            ``zp``
                Photometric zero point (mag, `float`).
            ``sigma``
                Standard deviation of fit of photometric zero point (mag, `float`).
            ``ngood``
                Number of sources used to fit photometric zero point (`int`).

        Raises
        ------
        RuntimeError
            Raised if any of the following occur:
            - No matches to use for photocal.
            - No matches are available (perhaps no sources/references were selected by the matcher).
            - No reference stars are available.
            - No matches are available from which to extract magnitudes.

        Notes
        -----
        The exposure is only used to provide the name of the filter being calibrated (it may also be
        used to generate debugging plots).

        The reference objects:
        - Must include a field ``photometric``; True for objects which should be considered as
        photometric standards.
        - Must include a field ``flux``; the flux used to impose a magnitude limit and also to calibrate
        the data to (unless a color term is specified, in which case ColorTerm.primary is used;
        See https://jira.lsstcorp.org/browse/DM-933).
        - May include a field ``stargal``; if present, True means that the object is a star.
        - May include a field ``var``; if present, True means that the object is variable.

        The measured sources:
        - Must include PhotoCalConfig.fluxField; the flux measurement to be used for calibration.
        """
        import lsstDebug

        display = lsstDebug.Info(__name__).display
        displaySources = display and lsstDebug.Info(__name__).displaySources
        self.scatterPlot = display and lsstDebug.Info(__name__).scatterPlot

        if self.scatterPlot:
            from matplotlib import pyplot
            try:
                self.fig.clf()
            except Exception:
                self.fig = pyplot.figure()

        filterLabel = exposure.getFilter()

        # Match sources
        matchResults = self.match.run(sourceCat, filterLabel.bandLabel)
        matches = matchResults.matches

        reserveResults = self.reserve.run([mm.second for mm in matches], expId=expId)
        if displaySources:
            self.displaySources(exposure, matches, reserveResults.reserved)
        if reserveResults.reserved.sum() > 0:
            matches = [mm for mm, use in zip(matches, reserveResults.use) if use]
        if len(matches) == 0:
            raise RuntimeError("No matches to use for photocal")
        if self.usedKey is not None:
            for mm in matches:
                mm.second.set(self.usedKey, True)

        # Prepare for fitting
        sourceKeys = self.getSourceKeys(matches[0].second.schema)
        arrays = self.extractMagArrays(matches, filterLabel, sourceKeys)

        # Fit for zeropoint
        r = self.getZeroPoint(arrays.srcMag, arrays.refMag, arrays.magErr)
        self.log.info("Magnitude zero point: %f +/- %f from %d stars", r.zp, r.sigma, r.ngood)

        # Prepare the results
        flux0 = 10**(0.4*r.zp)  # Flux of mag=0 star
        flux0err = 0.4*math.log(10)*flux0*r.sigma  # Error in flux0
        photoCalib = makePhotoCalibFromCalibZeroPoint(flux0, flux0err)
        self.log.info("Photometric calibration factor (nJy/ADU): %f +/- %f",
                      photoCalib.getCalibrationMean(),
                      photoCalib.getCalibrationErr())

        return pipeBase.Struct(
            photoCalib=photoCalib,
            arrays=arrays,
            matches=matches,
            matchMeta=matchResults.matchMeta,
            zp=r.zp,
            sigma=r.sigma,
            ngood=r.ngood,
        )

    def displaySources(self, exposure, matches, reserved, frame=1):
        """Display sources we'll use for photocal.

        Sources that will be actually used will be green.
        Sources reserved from the fit will be red.

        Parameters
        ----------
        exposure : `lsst.afw.image.ExposureF`
            Exposure to display.
        matches : `list` of `lsst.afw.table.RefMatch`
            Matches used for photocal.
        reserved : `numpy.ndarray` of type `bool`
            Boolean array indicating sources that are reserved.
        frame : `int`, optional
            Frame number for display.
        """
        disp = afwDisplay.getDisplay(frame=frame)
        disp.mtv(exposure, title="photocal")
        with disp.Buffering():
            for mm, rr in zip(matches, reserved):
                x, y = mm.second.getCentroid()
                ctype = afwDisplay.RED if rr else afwDisplay.GREEN
                disp.dot("o", x, y, size=4, ctype=ctype)

    def getZeroPoint(self, src, ref, srcErr=None, zp0=None):
        """Flux calibration code, returning (ZeroPoint, Distribution Width, Number of stars).

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``zp``
                Photometric zero point (mag, `float`).
            ``sigma``
                Standard deviation of fit of photometric zero point (mag, `float`).
            ``ngood``
                Number of sources used to fit photometric zero point (`int`).

        Notes
        -----
        We perform nIter iterations of a simple sigma-clipping algorithm with a couple of twists:
        - We use the median/interquartile range to estimate the position to clip around, and the
        "sigma" to use.
        - We never allow sigma to go _above_ a critical value sigmaMax --- if we do, a sufficiently
        large estimate will prevent the clipping from ever taking effect.
        - Rather than start with the median we start with a crude mode.  This means that a set of magnitude
        residuals with a tight core and asymmetrical outliers will start in the core.  We use the width of
        this core to set our maximum sigma (see second bullet).
        """
        sigmaMax = self.config.sigmaMax

        dmag = ref - src

        indArr = np.argsort(dmag)
        dmag = dmag[indArr]

        if srcErr is not None:
            dmagErr = srcErr[indArr]
        else:
            dmagErr = np.ones(len(dmag))

        # need to remove nan elements to avoid errors in stats calculation with numpy
        ind_noNan = np.array([i for i in range(len(dmag))
                              if (not np.isnan(dmag[i]) and not np.isnan(dmagErr[i]))])
        dmag = dmag[ind_noNan]
        dmagErr = dmagErr[ind_noNan]

        IQ_TO_STDEV = 0.741301109252802    # 1 sigma in units of interquartile (assume Gaussian)

        npt = len(dmag)
        ngood = npt
        good = None  # set at end of first iteration
        for i in range(self.config.nIter):
            if i > 0:
                npt = sum(good)

            center = None
            if i == 0:
                #
                # Start by finding the mode
                #
                nhist = 20
                try:
                    hist, edges = np.histogram(dmag, nhist, new=True)
                except TypeError:
                    hist, edges = np.histogram(dmag, nhist)  # they removed new=True around numpy 1.5
                imode = np.arange(nhist)[np.where(hist == hist.max())]

                if imode[-1] - imode[0] + 1 == len(imode):  # Multiple modes, but all contiguous
                    if zp0:
                        center = zp0
                    else:
                        center = 0.5*(edges[imode[0]] + edges[imode[-1] + 1])

                    peak = sum(hist[imode])/len(imode)  # peak height

                    # Estimate FWHM of mode
                    j = imode[0]
                    while j >= 0 and hist[j] > 0.5*peak:
                        j -= 1
                    j = max(j, 0)
                    q1 = dmag[sum(hist[range(j)])]

                    j = imode[-1]
                    while j < nhist and hist[j] > 0.5*peak:
                        j += 1
                    j = min(j, nhist - 1)
                    j = min(sum(hist[range(j)]), npt - 1)
                    q3 = dmag[j]

                    if q1 == q3:
                        q1 = dmag[int(0.25*npt)]
                        q3 = dmag[int(0.75*npt)]

                    sig = (q3 - q1)/2.3  # estimate of standard deviation (based on FWHM; 2.358 for Gaussian)

                    if sigmaMax is None:
                        sigmaMax = 2*sig   # upper bound on st. dev. for clipping. multiplier is a heuristic

                    self.log.debug("Photo calibration histogram: center = %.2f, sig = %.2f", center, sig)

                else:
                    if sigmaMax is None:
                        sigmaMax = dmag[-1] - dmag[0]

                    center = np.median(dmag)
                    q1 = dmag[int(0.25*npt)]
                    q3 = dmag[int(0.75*npt)]
                    sig = (q3 - q1)/2.3  # estimate of standard deviation (based on FWHM; 2.358 for Gaussian)

            if center is None:              # usually equivalent to (i > 0)
                gdmag = dmag[good]
                if self.config.useMedian:
                    center = np.median(gdmag)
                else:
                    gdmagErr = dmagErr[good]
                    center = np.average(gdmag, weights=gdmagErr)

                q3 = gdmag[min(int(0.75*npt + 0.5), npt - 1)]
                q1 = gdmag[min(int(0.25*npt + 0.5), npt - 1)]

                sig = IQ_TO_STDEV*(q3 - q1)     # estimate of standard deviation

            good = abs(dmag - center) < self.config.nSigma*min(sig, sigmaMax)  # don't clip too softly

            # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            if self.scatterPlot:
                try:
                    self.fig.clf()

                    axes = self.fig.add_axes((0.1, 0.1, 0.85, 0.80))

                    axes.plot(ref[good], dmag[good] - center, "b+")
                    axes.errorbar(ref[good], dmag[good] - center, yerr=dmagErr[good],
                                  linestyle='', color='b')

                    bad = np.logical_not(good)
                    if len(ref[bad]) > 0:
                        axes.plot(ref[bad], dmag[bad] - center, "r+")
                        axes.errorbar(ref[bad], dmag[bad] - center, yerr=dmagErr[bad],
                                      linestyle='', color='r')

                    axes.plot((-100, 100), (0, 0), "g-")
                    for x in (-1, 1):
                        axes.plot((-100, 100), x*0.05*np.ones(2), "g--")

                    axes.set_ylim(-1.1, 1.1)
                    axes.set_xlim(24, 13)
                    axes.set_xlabel("Reference")
                    axes.set_ylabel("Reference - Instrumental")

                    self.fig.show()

                    if self.scatterPlot > 1:
                        reply = None
                        while i == 0 or reply != "c":
                            try:
                                reply = input("Next iteration? [ynhpc] ")
                            except EOFError:
                                reply = "n"

                            if reply == "h":
                                print("Options: c[ontinue] h[elp] n[o] p[db] y[es]", file=sys.stderr)
                                continue

                            if reply in ("", "c", "n", "p", "y"):
                                break
                            else:
                                print("Unrecognised response: %s" % reply, file=sys.stderr)

                        if reply == "n":
                            break
                        elif reply == "p":
                            import pdb
                            pdb.set_trace()
                except Exception as e:
                    print("Error plotting in PhotoCal.getZeroPoint: %s" % e, file=sys.stderr)

            # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

            old_ngood = ngood
            ngood = sum(good)
            if ngood == 0:
                msg = "PhotoCal.getZeroPoint: no good stars remain"

                if i == 0:                  # failed the first time round -- probably all fell in one bin
                    center = np.average(dmag, weights=dmagErr)
                    msg += " on first iteration; using average of all calibration stars"

                self.log.warning(msg)

                return pipeBase.Struct(
                    zp=center,
                    sigma=sig,
                    ngood=len(dmag))
            elif ngood == old_ngood:
                break

            if False:
                ref = ref[good]
                dmag = dmag[good]
                dmagErr = dmagErr[good]

        dmag = dmag[good]
        dmagErr = dmagErr[good]
        zp, weightSum = np.average(dmag, weights=1/dmagErr**2, returned=True)
        sigma = np.sqrt(1.0/weightSum)
        return pipeBase.Struct(
            zp=zp,
            sigma=sigma,
            ngood=len(dmag),
        )
