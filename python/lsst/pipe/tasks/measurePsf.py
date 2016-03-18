#
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011 LSST Corporation.
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
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import random

import lsst.afw.math as afwMath
import lsst.afw.display.ds9 as ds9
import lsst.meas.algorithms as measAlg
import lsst.meas.algorithms.utils as maUtils
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

class MeasurePsfConfig(pexConfig.Config):
    starSelector = pexConfig.ConfigurableField(
        target = measAlg.ObjectSizeStarSelectorTask,
        doc = "Star selection algorithm",
    )
    psfDeterminer = measAlg.psfDeterminerRegistry.makeField("PSF Determination algorithm", default="pca")
    reserveFraction = pexConfig.Field(
        dtype = float,
        doc = "Fraction of PSF candidates to reserve from fitting; none if <= 0",
        default = -1.0,
    )
    reserveSeed = pexConfig.Field(
        dtype = int,
        doc = "This number will be multplied by the exposure ID to set the random seed for reserving candidates",
        default = 1,
    )

## \addtogroup LSST_task_documentation
## \{
## \page MeasurePsfTask
## \ref MeasurePsfTask_ "MeasurePsfTask"
## \copybrief MeasurePsfTask
## \}

class MeasurePsfTask(pipeBase.Task):
    """!
\anchor MeasurePsfTask_

\brief Measure the PSF

\section pipe_tasks_measurePsf_Contents Contents

 - \ref pipe_tasks_measurePsf_Purpose
 - \ref pipe_tasks_measurePsf_Initialize
 - \ref pipe_tasks_measurePsf_IO
 - \ref pipe_tasks_measurePsf_Config
 - \ref pipe_tasks_measurePsf_Debug
 - \ref pipe_tasks_measurePsf_Example

\section pipe_tasks_measurePsf_Purpose	Description

A task that wraps two algorithms set via a pair of registries specified in the task's
\ref pipe_tasks_measurePsf_Config.
Both algorithms are classes with a constructor taking a pex.config.Config object (\em e.g.
lsst.meas.algorithms.objectSizeStarSelector.ObjectSizeStarSelector.__init__).

The algorithms are:
 - a star selector, a subclass of lsst.meas.algorithms.StarSelector.selectStars

 - a psf estimator with API:
\code
determinePsf(exposure, psfCandidateList, metadata=None, flagKey=None)
\endcode
which returns an lsst.afw.detection.Psf and lsst.afw.math.SpatialCellSet (\em e.g.
lsst.meas.algorithms.pcaPsfDeterminer.PcaPsfDeterminer.determinePsf).
MeasurePsfTask calls determinePsf with \c flagKey set to
"calib.psf.used" if a schema is passed to its constructor (see \ref pipe_tasks_measurePsf_Initialize).

See also lsst.meas.algorithms.psfDeterminerRegistry.psfDeterminerRegistry.

\warning
There is no establised set of configuration parameters for these algorithms, so once you start modifying
parameters (as we do in \ref pipe_tasks_measurePsf_Example) your code is no longer portable.

\section pipe_tasks_measurePsf_Initialize	Task initialisation

\copydoc \_\_init\_\_

\section pipe_tasks_measurePsf_IO		Invoking the Task

\copydoc run

\section pipe_tasks_measurePsf_Config       Configuration parameters

See \ref MeasurePsfConfig.

\section pipe_tasks_measurePsf_Debug		Debug variables

The \link lsst.pipe.base.cmdLineTask.CmdLineTask command line task\endlink interface supports a
flag \c -d to import \b debug.py from your \c PYTHONPATH; see \ref baseDebug for more about \b debug.py files.

<DL>
  <DT> \c display
  <DD> If True, display debugging plots
  <DT> displayExposure
  <DD> display the Exposure + spatialCells
  <DT> displayPsfCandidates
  <DD> show mosaic of candidates
  <DT> showBadCandidates
  <DD> Include bad candidates
  <DT> displayPsfMosaic
  <DD> show mosaic of reconstructed PSF(xy)
  <DT> displayResiduals
  <DD> show residuals
  <DT> normalizeResiduals
  <DD> Normalise residuals by object amplitude
</DL>

Additionally you can enable any debug outputs that your chosen star selector and psf determiner support.

\section pipe_tasks_measurePsf_Example	A complete example of using MeasurePsfTask

This code is in \link measurePsfTask.py\endlink in the examples directory, and can be run as \em e.g.
\code
examples/measurePsfTask.py --ds9
\endcode
\dontinclude measurePsfTask.py

The example also runs SourceDetectionTask and SourceMeasurementTask; see \ref meas_algorithms_measurement_Example for more explanation.

Import the tasks (there are some other standard imports; read the file to see them all):

\skip SourceDetectionTask
\until MeasurePsfTask

We need to create the tasks before processing any data as the task constructor
can add an extra column to the schema, but first we need an almost-empty
Schema:

\skipline makeMinimalSchema

We can now call the constructors for the tasks we need to find and characterize candidate
PSF stars:

\skip SourceDetectionTask.ConfigClass
\until measureTask

Note that we've chosen a minimal set of measurement plugins: we need the
outputs of \c base_SdssCentroid, \c base_SdssShape and \c base_CircularApertureFlux as
inputs to the PSF measurement algorithm, while \c base_PixelFlags identifies
and flags bad sources (e.g. with pixels too close to the edge) so they can be
removed later.

Now we can create and configure the task that we're interested in:

\skip MeasurePsfTask
\until measurePsfTask

We're now ready to process the data (we could loop over multiple exposures/catalogues using the same
task objects).  First create the output table:

\skipline afwTable

And process the image:

\skip sources =
\until result

We can then unpack and use the results:

\skip psf
\until cellSet

If you specified \c --ds9 you can see the PSF candidates:

\skip display
\until RED

<HR>

To investigate the \ref pipe_tasks_measurePsf_Debug, put something like
\code{.py}
    import lsstDebug
    def DebugInfo(name):
        di = lsstDebug.getInfo(name)        # N.b. lsstDebug.Info(name) would call us recursively

        if name == "lsst.pipe.tasks.measurePsf" :
            di.display = True
            di.displayExposure = False          # display the Exposure + spatialCells
            di.displayPsfCandidates = True      # show mosaic of candidates
            di.displayPsfMosaic = True          # show mosaic of reconstructed PSF(xy)
            di.displayResiduals = True          # show residuals
            di.showBadCandidates = True         # Include bad candidates
            di.normalizeResiduals = False       # Normalise residuals by object amplitude

        return di

    lsstDebug.Info = DebugInfo
\endcode
into your debug.py file and run measurePsfTask.py with the \c --debug flag.
    """
    ConfigClass = MeasurePsfConfig
    _DefaultName = "measurePsf"

    def __init__(self, schema=None, **kwargs):
        """!Create the detection task.  Most arguments are simply passed onto pipe.base.Task.

        \param schema An lsst::afw::table::Schema used to create the output lsst.afw.table.SourceCatalog
        \param **kwargs Keyword arguments passed to lsst.pipe.base.task.Task.__init__.

        If schema is not None, 'calib.psf.candidate' and 'calib.psf.used' fields will be added to
        identify which stars were employed in the PSF estimation.

        \note This task can add fields to the schema, so any code calling this task must ensure that
        these fields are indeed present in the input table.
        """

        pipeBase.Task.__init__(self, **kwargs)
        if schema is not None:
            self.candidateKey = schema.addField(
                "calib_psfCandidate", type="Flag",
                doc=("Flag set if the source was a candidate for PSF determination, "
                     "as determined by the star selector.")
            )
            self.usedKey = schema.addField(
                "calib_psfUsed", type="Flag",
                doc=("Flag set if the source was actually used for PSF determination, "
                     "as determined by the '%s' PSF determiner.") % self.config.psfDeterminer.name
            )
            self.reservedKey = schema.addField(
                "calib_psfReserved", type="Flag",
                doc=("Flag set if the source was selected as a PSF candidate, but was "
                     "reserved from the PSF fitting."))
        else:
            self.candidateKey = None
            self.usedKey = None
        self.makeSubtask("starSelector")
        self.psfDeterminer = self.config.psfDeterminer.apply()

    @pipeBase.timeMethod
    def run(self, exposure, sources, expId=0, matches=None):
        """!Measure the PSF

        \param[in,out]   exposure      Exposure to process; measured PSF will be added.
        \param[in,out]   sources       Measured sources on exposure; flag fields will be set marking
                                       stars chosen by the star selector and the PSF determiner if a schema
                                       was passed to the task constructor.
        \param[in]       expId         Exposure id used for generating random seed.
        \param[in] matches a list of lsst.afw.table.ReferenceMatch objects (\em i.e. of lsst.afw.table.Match
        			       with \c first being of type lsst.afw.table.SimpleRecord and \c second
        			       type lsst.afw.table.SourceRecord --- the reference object and detected
        			       object respectively) as returned by \em e.g. the AstrometryTask.
                                       Used by star selectors that choose to refer to an external catalog.

        \return a pipe.base.Struct with fields:
         - psf: The measured PSF (also set in the input exposure)
         - cellSet: an lsst.afw.math.SpatialCellSet containing the PSF candidates as returned by the psf determiner.
        """
        self.log.info("Measuring PSF")

        import lsstDebug
        display = lsstDebug.Info(__name__).display
        displayExposure = lsstDebug.Info(__name__).displayExposure     # display the Exposure + spatialCells
        displayPsfMosaic = lsstDebug.Info(__name__).displayPsfMosaic # show mosaic of reconstructed PSF(x,y)
        displayPsfCandidates = lsstDebug.Info(__name__).displayPsfCandidates # show mosaic of candidates
        displayResiduals = lsstDebug.Info(__name__).displayResiduals   # show residuals
        showBadCandidates = lsstDebug.Info(__name__).showBadCandidates # include bad candidates
        normalizeResiduals = lsstDebug.Info(__name__).normalizeResiduals # normalise residuals by object peak

        #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        # Run star selector
        #
        starCat = self.starSelector.selectStars(exposure=exposure, sourceCat=sources, matches=matches).starCat
        psfCandidateList = self.starSelector.makePsfCandidates(exposure=exposure, starCat=starCat)
        reserveList = []
        
        if self.config.reserveFraction > 0 :
            random.seed(self.config.reserveSeed*expId)
            reserveList = random.sample(psfCandidateList, 
                                        int((self.config.reserveFraction)*len(psfCandidateList)))

            for cand in reserveList:
                psfCandidateList.remove(cand)

            if reserveList and self.reservedKey is not None:
                for cand in reserveList:
                    source = cand.getSource()
                    source.set(self.reservedKey,True)
            
        if psfCandidateList and self.candidateKey is not None:
            for cand in psfCandidateList:
                source = cand.getSource()
                source.set(self.candidateKey, True)

        self.log.info("PSF star selector found %d candidates" % len(psfCandidateList))
        if self.config.reserveFraction > 0 :
            self.log.info("Reserved %d candidates from the fitting" % len(reserveList))

        if display:
            frame = display
            if displayExposure:
                ds9.mtv(exposure, frame=frame, title="psf determination")

        #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #
        # Determine PSF
        #
        psf, cellSet = self.psfDeterminer.determinePsf(exposure, psfCandidateList, self.metadata,
                                                       flagKey=self.usedKey)
        self.log.info("PSF determination using %d/%d stars." %
                     (self.metadata.get("numGoodStars"), self.metadata.get("numAvailStars")))

        exposure.setPsf(psf)

        if display:
            frame = display
            if displayExposure:
                showPsfSpatialCells(exposure, cellSet, showBadCandidates, frame=frame)
                frame += 1

            if displayPsfCandidates:    # Show a mosaic of  PSF candidates
                plotPsfCandidates(cellSet, showBadCandidates, frame)
                frame += 1

            if displayResiduals:
                frame = plotResiduals(exposure, cellSet,
                                      showBadCandidates=showBadCandidates,
                                      normalizeResiduals=normalizeResiduals,
                                      frame=frame)
            if displayPsfMosaic:
                maUtils.showPsfMosaic(exposure, psf, frame=frame, showFwhm=True)
                ds9.scale(0, 1, "linear", frame=frame)
                frame += 1

        return pipeBase.Struct(
            psf = psf,
            cellSet = cellSet,
        )

    @property
    def usesMatches(self):
        """Return True if this task makes use of the "matches" argument to the run method"""
        return self.starSelector.usesMatches

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# Debug code
#
def showPsfSpatialCells(exposure, cellSet, showBadCandidates, frame=1):
    maUtils.showPsfSpatialCells(exposure, cellSet,
                                symb="o", ctype=ds9.CYAN, ctypeUnused=ds9.YELLOW,
                                size=4, frame=frame)
    for cell in cellSet.getCellList():
        for cand in cell.begin(not showBadCandidates): # maybe include bad candidates
            cand = measAlg.cast_PsfCandidateF(cand)
            status = cand.getStatus()
            ds9.dot('+', *cand.getSource().getCentroid(), frame=frame,
                    ctype=ds9.GREEN if status == afwMath.SpatialCellCandidate.GOOD else
                    ds9.YELLOW if status == afwMath.SpatialCellCandidate.UNKNOWN else ds9.RED)

def plotPsfCandidates(cellSet, showBadCandidates=False, frame=1):
    import lsst.afw.display.utils as displayUtils

    stamps = []
    for cell in cellSet.getCellList():
        for cand in cell.begin(not showBadCandidates): # maybe include bad candidates
            cand = measAlg.cast_PsfCandidateF(cand)

            try:
                im = cand.getMaskedImage()

                chi2 = cand.getChi2()
                if chi2 < 1e100:
                    chi2 = "%.1f" % chi2
                else:
                    chi2 = float("nan")

                stamps.append((im, "%d%s" %
                               (maUtils.splitId(cand.getSource().getId(), True)["objId"], chi2),
                               cand.getStatus()))
            except Exception:
                continue

    mos = displayUtils.Mosaic()
    for im, label, status in stamps:
        im = type(im)(im, True)
        try:
            im /= afwMath.makeStatistics(im, afwMath.MAX).getValue()
        except NotImplementedError:
            pass

        mos.append(im, label,
                   ds9.GREEN if status == afwMath.SpatialCellCandidate.GOOD else
                   ds9.YELLOW if status == afwMath.SpatialCellCandidate.UNKNOWN else ds9.RED)

    if mos.images:
        mos.makeMosaic(frame=frame, title="Psf Candidates")

def plotResiduals(exposure, cellSet, showBadCandidates=False, normalizeResiduals=True, frame=2):
    psf = exposure.getPsf()
    while True:
        try:
            maUtils.showPsfCandidates(exposure, cellSet, psf=psf, frame=frame,
                                      normalize=normalizeResiduals,
                                      showBadCandidates=showBadCandidates)
            frame += 1
            maUtils.showPsfCandidates(exposure, cellSet, psf=psf, frame=frame,
                                      normalize=normalizeResiduals,
                                      showBadCandidates=showBadCandidates,
                                      variance=True)
            frame += 1
        except Exception:
            if not showBadCandidates:
                showBadCandidates = True
                continue
        break

    return frame
