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

import lsst.afw.display as afwDisplay
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
import lsst.meas.algorithms.utils as maUtils
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.meas.extensions.psfex.psfexPsfDeterminer  # noqa: F401
from lsst.utils.timer import timeMethod


class MeasurePsfConfig(pexConfig.Config):
    starSelector = measAlg.sourceSelectorRegistry.makeField(
        "Star selection algorithm",
        default="objectSize"
    )
    makePsfCandidates = pexConfig.ConfigurableField(
        target=measAlg.MakePsfCandidatesTask,
        doc="Task to make psf candidates from selected stars.",
    )
    psfDeterminer = measAlg.psfDeterminerRegistry.makeField(
        "PSF Determination algorithm",
        default="psfex"
    )
    reserve = pexConfig.ConfigurableField(
        target=measAlg.ReserveSourcesTask,
        doc="Reserve sources from fitting"
    )

    def validate(self):
        super().validate()
        if (self.psfDeterminer.name == "piff"
                and self.psfDeterminer["piff"].kernelSize > self.makePsfCandidates.kernelSize):
            msg = (f"PIFF kernelSize={self.psfDeterminer['piff'].kernelSize}"
                   f" must be >= psf candidate kernelSize={self.makePsfCandidates.kernelSize}.")
            raise pexConfig.FieldValidationError(MeasurePsfConfig.makePsfCandidates, self, msg)

## @addtogroup LSST_task_documentation
## @{
## @page page_MeasurePsfTask MeasurePsfTask
## @ref MeasurePsfTask_ "MeasurePsfTask"
## @copybrief MeasurePsfTask
## @}


class MeasurePsfTask(pipeBase.Task):
    r"""!
@anchor MeasurePsfTask_

@brief Measure the PSF

@section pipe_tasks_measurePsf_Contents Contents

 - @ref pipe_tasks_measurePsf_Purpose
 - @ref pipe_tasks_measurePsf_Initialize
 - @ref pipe_tasks_measurePsf_IO
 - @ref pipe_tasks_measurePsf_Config
 - @ref pipe_tasks_measurePsf_Debug
 - @ref pipe_tasks_measurePsf_Example

@section pipe_tasks_measurePsf_Purpose	Description

A task that selects stars from a catalog of sources and uses those to measure the PSF.

The star selector is a subclass of
@ref lsst.meas.algorithms.starSelector.BaseStarSelectorTask "lsst.meas.algorithms.BaseStarSelectorTask"
and the PSF determiner is a sublcass of
@ref lsst.meas.algorithms.psfDeterminer.BasePsfDeterminerTask "lsst.meas.algorithms.BasePsfDeterminerTask"

@warning
There is no establised set of configuration parameters for these algorithms, so once you start modifying
parameters (as we do in @ref pipe_tasks_measurePsf_Example) your code is no longer portable.

@section pipe_tasks_measurePsf_Initialize	Task initialisation

@copydoc \_\_init\_\_

@section pipe_tasks_measurePsf_IO		Invoking the Task

@copydoc run

@section pipe_tasks_measurePsf_Config       Configuration parameters

See @ref MeasurePsfConfig.

@section pipe_tasks_measurePsf_Debug		Debug variables

The command line task interface supports a
flag @c -d to import @b debug.py from your @c PYTHONPATH; see
<a href="https://pipelines.lsst.io/modules/lsstDebug/">the lsstDebug documentation</a>
for more about @b debug.py files.

<DL>
  <DT> @c display
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

@section pipe_tasks_measurePsf_Example	A complete example of using MeasurePsfTask

This code is in `measurePsfTask.py` in the examples directory, and can be run as @em e.g.
@code
examples/measurePsfTask.py --doDisplay
@endcode
@dontinclude measurePsfTask.py

The example also runs SourceDetectionTask and SingleFrameMeasurementTask.

Import the tasks (there are some other standard imports; read the file to see them all):

@skip SourceDetectionTask
@until MeasurePsfTask

We need to create the tasks before processing any data as the task constructor
can add an extra column to the schema, but first we need an almost-empty
Schema:

@skipline makeMinimalSchema

We can now call the constructors for the tasks we need to find and characterize candidate
PSF stars:

@skip SourceDetectionTask.ConfigClass
@until measureTask

Note that we've chosen a minimal set of measurement plugins: we need the
outputs of @c base_SdssCentroid, @c base_SdssShape and @c base_CircularApertureFlux as
inputs to the PSF measurement algorithm, while @c base_PixelFlags identifies
and flags bad sources (e.g. with pixels too close to the edge) so they can be
removed later.

Now we can create and configure the task that we're interested in:

@skip MeasurePsfTask
@until measurePsfTask

We're now ready to process the data (we could loop over multiple exposures/catalogues using the same
task objects).  First create the output table:

@skipline afwTable

And process the image:

@skip sources =
@until result

We can then unpack and use the results:

@skip psf
@until cellSet

If you specified @c  --doDisplay you can see the PSF candidates:

@skip display
@until RED

<HR>

To investigate the @ref pipe_tasks_measurePsf_Debug, put something like
@code{.py}
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
@endcode
into your debug.py file and run measurePsfTask.py with the @c --debug flag.
    """
    ConfigClass = MeasurePsfConfig
    _DefaultName = "measurePsf"

    def __init__(self, schema=None, **kwargs):
        """!Create the detection task.  Most arguments are simply passed onto pipe.base.Task.

        @param schema An lsst::afw::table::Schema used to create the output lsst.afw.table.SourceCatalog
        @param **kwargs Keyword arguments passed to lsst.pipe.base.task.Task.__init__.

        If schema is not None, 'calib_psf_candidate' and 'calib_psf_used' fields will be added to
        identify which stars were employed in the PSF estimation.

        @note This task can add fields to the schema, so any code calling this task must ensure that
        these fields are indeed present in the input table.
        """

        pipeBase.Task.__init__(self, **kwargs)
        if schema is not None:
            self.candidateKey = schema.addField(
                "calib_psf_candidate", type="Flag",
                doc=("Flag set if the source was a candidate for PSF determination, "
                     "as determined by the star selector.")
            )
            self.usedKey = schema.addField(
                "calib_psf_used", type="Flag",
                doc=("Flag set if the source was actually used for PSF determination, "
                     "as determined by the '%s' PSF determiner.") % self.config.psfDeterminer.name
            )
        else:
            self.candidateKey = None
            self.usedKey = None
        self.makeSubtask("starSelector")
        self.makeSubtask("makePsfCandidates")
        self.makeSubtask("psfDeterminer", schema=schema)
        self.makeSubtask("reserve", columnName="calib_psf", schema=schema,
                         doc="set if source was reserved from PSF determination")

    @timeMethod
    def run(self, exposure, sources, expId=0, matches=None):
        """!Measure the PSF

        Parameters
        ----------
        exposure : `Unknown`
            Exposure to process; measured PSF will be added.
        Parameters
        ----------
        sources : `Unknown`
            Measured sources on exposure; flag fields will be set marking
                                    stars chosen by the star selector and the PSF determiner if a schema
                                    was passed to the task constructor.
        Parameters
        ----------
        expId : `Unknown`
            Exposure id used for generating random seed.
        Parameters
        ----------
        matches : `Unknown`
            A list of lsst.afw.table.ReferenceMatch objects
                                    (@em i.e. of lsst.afw.table.Match
                                    with @c first being of type lsst.afw.table.SimpleRecord and @c second
                                    type lsst.afw.table.SourceRecord --- the reference object and detected
                                    object respectively) as returned by @em e.g. the AstrometryTask.
                                    Used by star selectors that choose to refer to an external catalog.

        Returns
        -------
        Unknown: `Unknown`
            a pipe.base.Struct with fields:
         - psf: The measured PSF (also set in the input exposure)
         - cellSet: an lsst.afw.math.SpatialCellSet containing the PSF candidates
            as returned by the psf determiner.
        """
        self.log.info("Measuring PSF")

        import lsstDebug
        display = lsstDebug.Info(__name__).display
        displayExposure = lsstDebug.Info(__name__).displayExposure     # display the Exposure + spatialCells
        displayPsfMosaic = lsstDebug.Info(__name__).displayPsfMosaic  # show mosaic of reconstructed PSF(x,y)
        displayPsfCandidates = lsstDebug.Info(__name__).displayPsfCandidates  # show mosaic of candidates
        displayResiduals = lsstDebug.Info(__name__).displayResiduals   # show residuals
        showBadCandidates = lsstDebug.Info(__name__).showBadCandidates  # include bad candidates
        normalizeResiduals = lsstDebug.Info(__name__).normalizeResiduals  # normalise residuals by object peak

        #
        # Run star selector
        #
        stars = self.starSelector.run(sourceCat=sources, matches=matches, exposure=exposure)
        selectionResult = self.makePsfCandidates.run(stars.sourceCat, exposure=exposure)
        self.log.info("PSF star selector found %d candidates", len(selectionResult.psfCandidates))
        reserveResult = self.reserve.run(selectionResult.goodStarCat, expId=expId)
        # Make list of psf candidates to send to the determiner (omitting those marked as reserved)
        psfDeterminerList = [cand for cand, use
                             in zip(selectionResult.psfCandidates, reserveResult.use) if use]

        if selectionResult.psfCandidates and self.candidateKey is not None:
            for cand in selectionResult.psfCandidates:
                source = cand.getSource()
                source.set(self.candidateKey, True)

        self.log.info("Sending %d candidates to PSF determiner", len(psfDeterminerList))

        if display:
            frame = 1
            if displayExposure:
                disp = afwDisplay.Display(frame=frame)
                disp.mtv(exposure, title="psf determination")
                frame += 1
        #
        # Determine PSF
        #
        psf, cellSet = self.psfDeterminer.determinePsf(exposure, psfDeterminerList, self.metadata,
                                                       flagKey=self.usedKey)
        self.log.info("PSF determination using %d/%d stars.",
                      self.metadata.getScalar("numGoodStars"), self.metadata.getScalar("numAvailStars"))

        exposure.setPsf(psf)

        if display:
            frame = display
            if displayExposure:
                disp = afwDisplay.Display(frame=frame)
                showPsfSpatialCells(exposure, cellSet, showBadCandidates, frame=frame)
                frame += 1

            if displayPsfCandidates:    # Show a mosaic of  PSF candidates
                plotPsfCandidates(cellSet, showBadCandidates=showBadCandidates, frame=frame)
                frame += 1

            if displayResiduals:
                frame = plotResiduals(exposure, cellSet,
                                      showBadCandidates=showBadCandidates,
                                      normalizeResiduals=normalizeResiduals,
                                      frame=frame)
            if displayPsfMosaic:
                disp = afwDisplay.Display(frame=frame)
                maUtils.showPsfMosaic(exposure, psf, display=disp, showFwhm=True)
                disp.scale("linear", 0, 1)
                frame += 1

        return pipeBase.Struct(
            psf=psf,
            cellSet=cellSet,
        )

    @property
    def usesMatches(self):
        """Return True if this task makes use of the "matches" argument to the run method"""
        return self.starSelector.usesMatches

#
# Debug code
#


def showPsfSpatialCells(exposure, cellSet, showBadCandidates, frame=1):
    disp = afwDisplay.Display(frame=frame)
    maUtils.showPsfSpatialCells(exposure, cellSet,
                                symb="o", ctype=afwDisplay.CYAN, ctypeUnused=afwDisplay.YELLOW,
                                size=4, display=disp)
    for cell in cellSet.getCellList():
        for cand in cell.begin(not showBadCandidates):  # maybe include bad candidates
            status = cand.getStatus()
            disp.dot('+', *cand.getSource().getCentroid(),
                     ctype=afwDisplay.GREEN if status == afwMath.SpatialCellCandidate.GOOD else
                     afwDisplay.YELLOW if status == afwMath.SpatialCellCandidate.UNKNOWN else afwDisplay.RED)


def plotPsfCandidates(cellSet, showBadCandidates=False, frame=1):
    stamps = []
    for cell in cellSet.getCellList():
        for cand in cell.begin(not showBadCandidates):  # maybe include bad candidates
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

    mos = afwDisplay.utils.Mosaic()
    disp = afwDisplay.Display(frame=frame)
    for im, label, status in stamps:
        im = type(im)(im, True)
        try:
            im /= afwMath.makeStatistics(im, afwMath.MAX).getValue()
        except NotImplementedError:
            pass

        mos.append(im, label,
                   afwDisplay.GREEN if status == afwMath.SpatialCellCandidate.GOOD else
                   afwDisplay.YELLOW if status == afwMath.SpatialCellCandidate.UNKNOWN else afwDisplay.RED)

    if mos.images:
        disp.mtv(mos.makeMosaic(), title="Psf Candidates")


def plotResiduals(exposure, cellSet, showBadCandidates=False, normalizeResiduals=True, frame=2):
    psf = exposure.getPsf()
    disp = afwDisplay.Display(frame=frame)
    while True:
        try:
            maUtils.showPsfCandidates(exposure, cellSet, psf=psf, display=disp,
                                      normalize=normalizeResiduals,
                                      showBadCandidates=showBadCandidates)
            frame += 1
            maUtils.showPsfCandidates(exposure, cellSet, psf=psf, display=disp,
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
