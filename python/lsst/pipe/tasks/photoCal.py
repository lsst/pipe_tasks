#
# LSST Data Management System
# Copyright 2008-2016 AURA/LSST.
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
# \package lsst.pipe.tasks.

from __future__ import absolute_import, division, print_function
from builtins import zip
from builtins import input
from builtins import range

import math
import sys

import numpy as np

import lsst.pex.config as pexConf
import lsst.pipe.base as pipeBase
from lsst.afw.image import abMagFromFlux, abMagErrFromFluxErr, fluxFromABMag, Calib
import lsst.afw.math as afwMath
from lsst.meas.astrom import RefMatchTask, RefMatchConfig
import lsst.afw.display.ds9 as ds9
from lsst.meas.algorithms import getRefFluxField
from .colorterms import ColortermLibrary

__all__ = ["PhotoCalTask", "PhotoCalConfig"]


def checkSourceFlags(source, sourceKeys):
    """!Return True if the given source has all good flags set and none of the bad flags set.

    \param[in] source      SourceRecord object to process.
    \param[in] sourceKeys  Struct of source catalog keys, as returned by PhotCalTask.getSourceKeys()
    """
    for k in sourceKeys.goodFlags:
        if not source.get(k):
            return False
    if source.getPsfFluxFlag():
        return False
    for k in sourceKeys.badFlags:
        if source.get(k):
            return False
    return True


class PhotoCalConfig(RefMatchConfig):
    """Config for PhotoCal"""
    magLimit = pexConf.Field(
        dtype=float,
        default=22.0,
        doc="Don't use objects fainter than this magnitude",
    )
    reserveFraction = pexConf.Field(
        dtype=float,
        doc="Fraction of candidates to reserve from fitting; none if <= 0",
        default=-1.0,
    )
    reserveSeed = pexConf.Field(
        dtype=int,
        doc="This number will be multiplied by the exposure ID "
        "to set the random seed for reserving candidates",
        default=1,
    )
    fluxField = pexConf.Field(
        dtype=str,
        default="slot_CalibFlux_flux",
        doc=("Name of the source flux field to use.  The associated flag field\n"
             "('<name>_flags') will be implicitly included in badFlags."),
    )
    applyColorTerms = pexConf.Field(
        dtype=bool,
        default=None,
        doc=("Apply photometric color terms to reference stars? One of:\n"
             "None: apply if colorterms and photoCatName are not None;\n"
             "      fail if color term data is not available for the specified ref catalog and filter.\n"
             "True: always apply colorterms; fail if color term data is not available for the\n"
             "      specified reference catalog and filter.\n"
             "False: do not apply."),
        optional=True,
    )
    goodFlags = pexConf.ListField(
        dtype=str,
        default=[],
        doc="List of source flag fields that must be set for a source to be used.",
    )
    badFlags = pexConf.ListField(
        dtype=str,
        default=["base_PixelFlags_flag_edge", "base_PixelFlags_flag_interpolated",
                 "base_PixelFlags_flag_saturated"],
        doc="List of source flag fields that will cause a source to be rejected when they are set.",
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
        doc="Library of photometric reference catalog name: color term dict",
    )
    photoCatName = pexConf.Field(
        dtype=str,
        optional=True,
        doc=("Name of photometric reference catalog; used to select a color term dict in colorterms."
             " see also applyColorTerms"),
    )
    magErrFloor = pexConf.RangeField(
        dtype=float,
        default=0.0,
        doc="Additional magnitude uncertainty to be added in quadrature with measurement errors.",
        min=0.0,
    )
    doSelectUnresolved = pexConf.Field(
        dtype=bool,
        default=True,
        doc=("Use the extendedness parameter to select objects to use in photometric calibration?\n"
             "This applies only to the sources detected on the exposure, not the reference catalog"),
    )

    def validate(self):
        pexConf.Config.validate(self)
        if self.applyColorTerms and self.photoCatName is None:
            raise RuntimeError("applyColorTerms=True requires photoCatName is non-None")
        if self.applyColorTerms and len(self.colorterms.data) == 0:
            raise RuntimeError("applyColorTerms=True requires colorterms be provided")


## \addtogroup LSST_task_documentation
## \{
## \page photoCalTask
## \ref PhotoCalTask_ "PhotoCalTask"
##      Detect positive and negative sources on an exposure and return a new SourceCatalog.
## \}

class PhotoCalTask(RefMatchTask):
    r"""!
\anchor PhotoCalTask_

\brief Calculate the zero point of an exposure given a lsst.afw.table.ReferenceMatchVector.

\section pipe_tasks_photocal_Contents Contents

 - \ref pipe_tasks_photocal_Purpose
 - \ref pipe_tasks_photocal_Initialize
 - \ref pipe_tasks_photocal_IO
 - \ref pipe_tasks_photocal_Config
 - \ref pipe_tasks_photocal_Debug
 - \ref pipe_tasks_photocal_Example

\section pipe_tasks_photocal_Purpose	Description

\copybrief PhotoCalTask

Calculate an Exposure's zero-point given a set of flux measurements of stars matched to an input catalogue.
The type of flux to use is specified by PhotoCalConfig.fluxField.

The algorithm clips outliers iteratively, with parameters set in the configuration.

\note This task can adds fields to the schema, so any code calling this task must ensure that
these columns are indeed present in the input match list; see \ref pipe_tasks_photocal_Example

\section pipe_tasks_photocal_Initialize	Task initialisation

\copydoc \_\_init\_\_

\section pipe_tasks_photocal_IO		Inputs/Outputs to the run method

\copydoc run

\section pipe_tasks_photocal_Config       Configuration parameters

See \ref PhotoCalConfig

\section pipe_tasks_photocal_Debug		Debug variables

The \link lsst.pipe.base.cmdLineTask.CmdLineTask command line task\endlink interface supports a
flag \c -d to import \b debug.py from your \c PYTHONPATH; see \ref baseDebug for more about \b debug.py files.

The available variables in PhotoCalTask are:
<DL>
  <DT> \c display
  <DD> If True enable other debug outputs
  <DT> \c displaySources
  <DD> If True, display the exposure on ds9's frame 1 and overlay the source catalogue:
    <DL>
      <DT> red x
      <DD> Bad objects
      <DT> blue +
      <DD> Matched objects deemed unsuitable for photometric calibration.
            Additional information is:
        - a cyan o for galaxies
        - a magenta o for variables
      <DT> magenta *
      <DD> Objects that failed the flux cut
      <DT> green o
      <DD> Objects used in the photometric calibration
    </DL>
  <DT> \c scatterPlot
  <DD> Make a scatter plot of flux v. reference magnitude as a function of reference magnitude.
    - good objects in blue
    - rejected objects in red
  (if \c scatterPlot is 2 or more, prompt to continue after each iteration)
</DL>

\section pipe_tasks_photocal_Example	A complete example of using PhotoCalTask

This code is in \link examples/photoCalTask.py\endlink, and can be run as \em e.g.
\code
examples/photoCalTask.py
\endcode
\dontinclude photoCalTask.py

Import the tasks (there are some other standard imports; read the file for details)
\skipline from lsst.pipe.tasks.astrometry
\skipline measPhotocal

We need to create both our tasks before processing any data as the task constructors
can add extra columns to the schema which we get from the input catalogue, \c scrCat:
\skipline getSchema

Astrometry first:
\skip AstrometryTask.ConfigClass
\until aTask
(that \c filterMap line is because our test code doesn't use a filter that the reference catalogue recognises,
so we tell it to use the \c r band)

Then photometry:
\skip measPhotocal
\until pTask

If the schema has indeed changed we need to add the new columns to the source table
(yes; this should be easier!)
\skip srcCat
\until srcCat = cat

We're now ready to process the data (we could loop over multiple exposures/catalogues using the same
task objects):
\skip matches
\until result

We can then unpack and use the results:
\skip calib
\until np.log

<HR>
To investigate the \ref pipe_tasks_photocal_Debug, put something like
\code{.py}
    import lsstDebug
    def DebugInfo(name):
        di = lsstDebug.getInfo(name)        # N.b. lsstDebug.Info(name) would call us recursively
        if name.endswith(".PhotoCal"):
            di.display = 1

        return di

    lsstDebug.Info = DebugInfo
\endcode
into your debug.py file and run photoCalTask.py with the \c --debug flag.
    """
    ConfigClass = PhotoCalConfig
    _DefaultName = "photoCal"

    def __init__(self, refObjLoader, schema=None, **kwds):
        """!Create the photometric calibration task.  See PhotoCalTask.init for documentation
        """
        RefMatchTask.__init__(self, refObjLoader, schema=None, **kwds)
        self.scatterPlot = None
        self.fig = None
        if schema is not None:
            self.usedKey = schema.addField("calib_photometryUsed", type="Flag",
                                           doc="set if source was used in photometric calibration")
            self.candidateKey = schema.addField("calib_photometryCandidate", type="Flag",
                                                doc="set if source was a candidate for use in calibration")
            self.reservedKey = schema.addField("calib_photometryReserved", type="Flag",
                                               doc="set if source was reserved, so not used in calibration")
        else:
            self.usedKey = None
            self.candidateKey = None
            self.reservedKey = None

    def getSourceKeys(self, schema):
        """!Return a struct containing the source catalog keys for fields used by PhotoCalTask.

        Returned fields include:
        - flux
        - fluxErr
        - goodFlags: a list of keys for field names in self.config.goodFlags
        - badFlags: a list of keys for field names in self.config.badFlags
        - starGal: key for star/galaxy classification
        """
        goodFlags = [schema.find(name).key for name in self.config.goodFlags]
        flux = schema.find(self.config.fluxField).key
        fluxErr = schema.find(self.config.fluxField + "Sigma").key
        badFlags = [schema.find(name).key for name in self.config.badFlags]
        try:
            starGal = schema.find("base_ClassificationExtendedness_value").key
        except KeyError:
            starGal = None
        return pipeBase.Struct(flux=flux, fluxErr=fluxErr, goodFlags=goodFlags, badFlags=badFlags,
                               starGal=starGal)

    def isUnresolved(self, source, starGalKey=None):
        """!Return whether the provided source is unresolved or not

        This particular implementation is designed to work with the
        base_ClassificationExtendedness_value=0.0 or 1.0 scheme.  Because
        of the diversity of star/galaxy classification outputs (binary
        decision vs probabilities; signs), it's difficult to make this
        configurable without using code.  This method should therefore
        be overridden to use the appropriate classification output.

        \param[in] source      Source to test
        \param[in] starGalKey  Struct of schema keys for source
        \return    boolean value for starGalKey (True indicates Unresolved)
        """
        return source.get(starGalKey) < 0.5 if starGalKey is not None else True

    @pipeBase.timeMethod
    def selectMatches(self, matches, sourceKeys, filterName, frame=None):
        """!Select reference/source matches according the criteria specified in the config.

        \param[in] matches ReferenceMatchVector (not modified)
        \param[in] sourceKeys  Struct of source catalog keys, as returned by getSourceKeys()
        \param[in] filterName  name of camera filter; used to obtain the reference flux field
        \param[in] frame   ds9 frame number to use for debugging display
        if frame is non-None, display information about trimmed objects on that ds9 frame:
         - Bad:               red x
         - Unsuitable objects: blue +  (and a cyan o if a galaxy)
         - Failed flux cut:   magenta *

        \return a \link lsst.afw.table.ReferenceMatchVector\endlink that contains only the selected matches.
        If a schema was passed during task construction, a flag field will be set on sources
        in the selected matches.

        \throws ValueError There are no valid matches.
        """
        self.log.debug("Number of input matches: %d", len(matches))

        if self.config.doSelectUnresolved:
            # Select only resolved sources if asked to do so.
            matches = [m for m in matches if self.isUnresolved(m.second, sourceKeys.starGal)]
            self.log.debug("Number of matches after culling resolved sources: %d", len(matches))

        if len(matches) == 0:
            raise ValueError("No input matches")

        for m in matches:
            if self.candidateKey is not None:
                m.second.set(self.candidateKey, True)

        # Only use stars for which the flags indicate the photometry is good.
        afterFlagCutInd = [i for i, m in enumerate(matches) if checkSourceFlags(m.second, sourceKeys)]
        afterFlagCut = [matches[i] for i in afterFlagCutInd]
        self.log.debug("Number of matches after source flag cuts: %d", len(afterFlagCut))

        if len(afterFlagCut) != len(matches):
            if frame is not None:
                with ds9.Buffering():
                    for i, m in enumerate(matches):
                        if i not in afterFlagCutInd:
                            x, y = m.second.getCentroid()
                            ds9.dot("x", x, y, size=4, frame=frame, ctype=ds9.RED)

            matches = afterFlagCut

        if len(matches) == 0:
            raise ValueError("All matches eliminated by source flags")

        refSchema = matches[0].first.schema
        try:
            photometricKey = refSchema.find("photometric").key
            try:
                resolvedKey = refSchema.find("resolved").key
            except:
                resolvedKey = None

            try:
                variableKey = refSchema.find("variable").key
            except:
                variableKey = None
        except:
            self.log.warn("No 'photometric' flag key found in reference schema.")
            photometricKey = None

        if photometricKey is not None:
            afterRefCutInd = [i for i, m in enumerate(matches) if m.first.get(photometricKey)]
            afterRefCut = [matches[i] for i in afterRefCutInd]

            if len(afterRefCut) != len(matches):
                if frame is not None:
                    with ds9.Buffering():
                        for i, m in enumerate(matches):
                            if i not in afterRefCutInd:
                                x, y = m.second.getCentroid()
                                ds9.dot("+", x, y, size=4, frame=frame, ctype=ds9.BLUE)

                                if resolvedKey and m.first.get(resolvedKey):
                                    ds9.dot("o", x, y, size=6, frame=frame, ctype=ds9.CYAN)
                                if variableKey and m.first.get(variableKey):
                                    ds9.dot("o", x, y, size=6, frame=frame, ctype=ds9.MAGENTA)

                matches = afterRefCut

        self.log.debug("Number of matches after reference catalog cuts: %d", len(matches))
        if len(matches) == 0:
            raise RuntimeError("No sources remain in match list after reference catalog cuts.")
        fluxName = getRefFluxField(refSchema, filterName)
        fluxKey = refSchema.find(fluxName).key
        if self.config.magLimit is not None:
            fluxLimit = fluxFromABMag(self.config.magLimit)

            afterMagCutInd = [i for i, m in enumerate(matches) if (m.first.get(fluxKey) > fluxLimit and
                                                                   m.second.getPsfFlux() > 0.0)]
        else:
            afterMagCutInd = [i for i, m in enumerate(matches) if m.second.getPsfFlux() > 0.0]

        afterMagCut = [matches[i] for i in afterMagCutInd]

        if len(afterMagCut) != len(matches):
            if frame is not None:
                with ds9.Buffering():
                    for i, m in enumerate(matches):
                        if i not in afterMagCutInd:
                            x, y = m.second.getCentroid()
                            ds9.dot("*", x, y, size=4, frame=frame, ctype=ds9.MAGENTA)

            matches = afterMagCut

        self.log.debug("Number of matches after magnitude limit cuts: %d", len(matches))

        if len(matches) == 0:
            raise RuntimeError("No sources remaining in match list after magnitude limit cuts.")

        if frame is not None:
            with ds9.Buffering():
                for m in matches:
                    x, y = m.second.getCentroid()
                    ds9.dot("o", x, y, size=4, frame=frame, ctype=ds9.GREEN)

        result = []
        for m in matches:
            if self.usedKey is not None:
                m.second.set(self.usedKey, True)
            result.append(m)
        return result

    @pipeBase.timeMethod
    def extractMagArrays(self, matches, filterName, sourceKeys):
        """!Extract magnitude and magnitude error arrays from the given matches.

        \param[in] matches Reference/source matches, a \link lsst::afw::table::ReferenceMatchVector\endlink
        \param[in] filterName  Name of filter being calibrated
        \param[in] sourceKeys  Struct of source catalog keys, as returned by getSourceKeys()

        \return Struct containing srcMag, refMag, srcMagErr, refMagErr, and magErr numpy arrays
        where magErr is an error in the magnitude; the error in srcMag - refMag
        If nonzero, config.magErrFloor will be added to magErr *only* (not srcMagErr or refMagErr), as
        magErr is what is later used to determine the zero point.
        Struct also contains refFluxFieldList: a list of field names of the reference catalog used for fluxes
        (1 or 2 strings)
        \note These magnitude arrays are the \em inputs to the photometric calibration, some may have been
        discarded by clipping while estimating the calibration (https://jira.lsstcorp.org/browse/DM-813)
        """
        srcFluxArr = np.array([m.second.get(sourceKeys.flux) for m in matches])
        srcFluxErrArr = np.array([m.second.get(sourceKeys.fluxErr) for m in matches])
        if not np.all(np.isfinite(srcFluxErrArr)):
            # this is an unpleasant hack; see DM-2308 requesting a better solution
            self.log.warn("Source catalog does not have flux uncertainties; using sqrt(flux).")
            srcFluxErrArr = np.sqrt(srcFluxArr)

        # convert source flux from DN to an estimate of Jy
        JanskysPerABFlux = 3631.0
        srcFluxArr = srcFluxArr * JanskysPerABFlux
        srcFluxErrArr = srcFluxErrArr * JanskysPerABFlux

        if not matches:
            raise RuntimeError("No reference stars are available")
        refSchema = matches[0].first.schema

        applyColorTerms = self.config.applyColorTerms
        applyCTReason = "config.applyColorTerms is %s" % (self.config.applyColorTerms,)
        if self.config.applyColorTerms is None:
            # apply color terms if color term data is available and photoCatName specified
            ctDataAvail = len(self.config.colorterms.data) > 0
            photoCatSpecified = self.config.photoCatName is not None
            applyCTReason += " and data %s available" % ("is" if ctDataAvail else "is not")
            applyCTReason += " and photoRefCat %s None" % ("is not" if photoCatSpecified else "is")
            applyColorTerms = ctDataAvail and photoCatSpecified

        if applyColorTerms:
            self.log.info("Applying color terms for filterName=%r, config.photoCatName=%s because %s",
                          filterName, self.config.photoCatName, applyCTReason)
            ct = self.config.colorterms.getColorterm(
                filterName=filterName, photoCatName=self.config.photoCatName, doRaise=True)
        else:
            self.log.info("Not applying color terms because %s", applyCTReason)
            ct = None

        if ct:                          # we have a color term to worry about
            fluxFieldList = [getRefFluxField(refSchema, filt) for filt in (ct.primary, ct.secondary)]
            missingFluxFieldList = []
            for fluxField in fluxFieldList:
                try:
                    refSchema.find(fluxField).key
                except KeyError:
                    missingFluxFieldList.append(fluxField)

            if missingFluxFieldList:
                self.log.warn("Source catalog does not have fluxes for %s; ignoring color terms",
                              " ".join(missingFluxFieldList))
                ct = None

        if not ct:
            fluxFieldList = [getRefFluxField(refSchema, filterName)]

        refFluxArrList = []  # list of ref arrays, one per flux field
        refFluxErrArrList = []  # list of ref flux arrays, one per flux field
        for fluxField in fluxFieldList:
            fluxKey = refSchema.find(fluxField).key
            refFluxArr = np.array([m.first.get(fluxKey) for m in matches])
            try:
                fluxErrKey = refSchema.find(fluxField + "Sigma").key
                refFluxErrArr = np.array([m.first.get(fluxErrKey) for m in matches])
            except KeyError:
                # Reference catalogue may not have flux uncertainties; HACK
                self.log.warn("Reference catalog does not have flux uncertainties for %s; using sqrt(flux).",
                              fluxField)
                refFluxErrArr = np.sqrt(refFluxArr)

            refFluxArrList.append(refFluxArr)
            refFluxErrArrList.append(refFluxErrArr)

        if ct:                          # we have a color term to worry about
            refMagArr1 = np.array([abMagFromFlux(rf1) for rf1 in refFluxArrList[0]])  # primary
            refMagArr2 = np.array([abMagFromFlux(rf2) for rf2 in refFluxArrList[1]])  # secondary

            refMagArr = ct.transformMags(refMagArr1, refMagArr2)
            refFluxErrArr = ct.propagateFluxErrors(refFluxErrArrList[0], refFluxErrArrList[1])
        else:
            refMagArr = np.array([abMagFromFlux(rf) for rf in refFluxArrList[0]])

        srcMagArr = np.array([abMagFromFlux(sf) for sf in srcFluxArr])

        # Fitting with error bars in both axes is hard
        # for now ignore reference flux error, but ticket DM-2308 is a request for a better solution
        magErrArr = np.array([abMagErrFromFluxErr(fe, sf) for fe, sf in zip(srcFluxErrArr, srcFluxArr)])
        if self.config.magErrFloor != 0.0:
            magErrArr = (magErrArr**2 + self.config.magErrFloor**2)**0.5

        srcMagErrArr = np.array([abMagErrFromFluxErr(sfe, sf) for sfe, sf in zip(srcFluxErrArr, srcFluxArr)])
        refMagErrArr = np.array([abMagErrFromFluxErr(rfe, rf) for rfe, rf in zip(refFluxErrArr, refFluxArr)])

        return pipeBase.Struct(
            srcMag=srcMagArr,
            refMag=refMagArr,
            magErr=magErrArr,
            srcMagErr=srcMagErrArr,
            refMagErr=refMagErrArr,
            refFluxFieldList=fluxFieldList,
        )

    @pipeBase.timeMethod
    def run(self, exposure, sourceCat, expId=0):
        """!Do photometric calibration - select matches to use and (possibly iteratively) compute
        the zero point.

        \param[in]  exposure  Exposure upon which the sources in the matches were detected.
        \param[in]  sourceCat  A catalog of sources to use in the calibration
        (\em i.e. a list of lsst.afw.table.Match with
        \c first being of type lsst.afw.table.SimpleRecord and \c second type lsst.afw.table.SourceRecord ---
        the reference object and matched object respectively).
        (will not be modified  except to set the outputField if requested.).

        \return Struct of:
         - calib -------  \link lsst::afw::image::Calib\endlink object containing the zero point
         - arrays ------ Magnitude arrays returned be PhotoCalTask.extractMagArrays
         - matches ----- Final ReferenceMatchVector, as returned by PhotoCalTask.selectMatches.
         - zp ---------- Photometric zero point (mag)
         - sigma ------- Standard deviation of fit of photometric zero point (mag)
         - ngood ------- Number of sources used to fit photometric zero point

        The exposure is only used to provide the name of the filter being calibrated (it may also be
        used to generate debugging plots).

        The reference objects:
         - Must include a field \c photometric; True for objects which should be considered as
            photometric standards
         - Must include a field \c flux; the flux used to impose a magnitude limit and also to calibrate
            the data to (unless a color term is specified, in which case ColorTerm.primary is used;
            See https://jira.lsstcorp.org/browse/DM-933)
         - May include a field \c stargal; if present, True means that the object is a star
         - May include a field \c var; if present, True means that the object is variable

        The measured sources:
        - Must include PhotoCalConfig.fluxField; the flux measurement to be used for calibration

        \throws RuntimeError with the following strings:

        <DL>
        <DT> `sources' schema does not contain the calibration object flag "XXX"`
        <DD> The constructor added fields to the schema that aren't in the Sources
        <DT> No input matches
        <DD> The input match vector is empty
        <DT> All matches eliminated by source flags
        <DD> The flags specified by \c badFlags in the config eliminated all candidate objects
        <DT> No sources remain in match list after reference catalog cuts
        <DD> The reference catalogue has a column "photometric", but no matched objects have it set
        <DT> No sources remaining in match list after magnitude limit cuts
        <DD> All surviving matches are either too faint in the catalogue or have negative or \c NaN flux
        <DT> No reference stars are available
        <DD> No matches survive all the checks
        </DL>
        """
        import lsstDebug

        display = lsstDebug.Info(__name__).display
        displaySources = display and lsstDebug.Info(__name__).displaySources
        self.scatterPlot = display and lsstDebug.Info(__name__).scatterPlot

        if self.scatterPlot:
            from matplotlib import pyplot
            try:
                self.fig.clf()
            except:
                self.fig = pyplot.figure()

        if displaySources:
            frame = 1
            ds9.mtv(exposure, frame=frame, title="photocal")
        else:
            frame = None

        res = self.loadAndMatch(exposure, sourceCat)

        # from res.matches, reserve a fraction of the sources from res.matches and mark the sources reserved

        if self.config.reserveFraction > 0:
            # Note that the seed can't be set to 0, so guard against an improper expId.
            random = afwMath.Random(seed=self.config.reserveSeed*(expId if expId else 1))
            reserveList = []
            n = len(res.matches)
            for i in range(int(n*self.config.reserveFraction)):
                index = random.uniformInt(n)
                n -= 1
                candidate = res.matches[index]
                res.matches.remove(candidate)
                reserveList.append(candidate)

            if reserveList and self.reservedKey is not None:
                for candidate in reserveList:
                    candidate.second.set(self.reservedKey, True)

        matches = res.matches
        for m in matches:
            if self.candidateKey is not None:
                m.second.set(self.candidateKey, True)

        filterName = exposure.getFilter().getName()
        sourceKeys = self.getSourceKeys(matches[0].second.schema)

        matches = self.selectMatches(matches=matches, sourceKeys=sourceKeys, filterName=filterName,
                                     frame=frame)
        arrays = self.extractMagArrays(matches=matches, filterName=filterName, sourceKeys=sourceKeys)

        if matches and self.usedKey:
            try:
                # matches[].second is a measured source, wherein we wish to set outputField.
                # Check that the field is present in the Sources schema.
                matches[0].second.getSchema().find(self.usedKey)
            except:
                raise RuntimeError("sources' schema does not contain the calib_photometryUsed flag \"%s\"" %
                                   self.usedKey)

        # Fit for zeropoint.  We can run the code more than once, so as to
        # give good stars that got clipped by a bad first guess a second
        # chance.

        calib = Calib()
        zp = None                           # initial guess
        r = self.getZeroPoint(arrays.srcMag, arrays.refMag, arrays.magErr, zp0=zp)
        zp = r.zp
        self.log.info("Magnitude zero point: %f +/- %f from %d stars", r.zp, r.sigma, r.ngood)

        flux0 = 10**(0.4*r.zp)  # Flux of mag=0 star
        flux0err = 0.4*math.log(10)*flux0*r.sigma  # Error in flux0

        calib.setFluxMag0(flux0, flux0err)

        return pipeBase.Struct(
            calib=calib,
            arrays=arrays,
            matches=matches,
            zp=r.zp,
            sigma=r.sigma,
            ngood=r.ngood,
        )

    def getZeroPoint(self, src, ref, srcErr=None, zp0=None):
        """!Flux calibration code, returning (ZeroPoint, Distribution Width, Number of stars)

        We perform nIter iterations of a simple sigma-clipping algorithm with a couple of twists:
        1.  We use the median/interquartile range to estimate the position to clip around, and the
        "sigma" to use.
        2.  We never allow sigma to go _above_ a critical value sigmaMax --- if we do, a sufficiently
        large estimate will prevent the clipping from ever taking effect.
        3.  Rather than start with the median we start with a crude mode.  This means that a set of magnitude
        residuals with a tight core and asymmetrical outliers will start in the core.  We use the width of
        this core to set our maximum sigma (see 2.)

        \return Struct of:
         - zp ---------- Photometric zero point (mag)
         - sigma ------- Standard deviation of fit of zero point (mag)
         - ngood ------- Number of sources used to fit zero point
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

                self.log.warn(msg)

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
