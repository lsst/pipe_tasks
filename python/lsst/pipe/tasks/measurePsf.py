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
import lsst.afw.math as afwMath
import lsst.afw.display.ds9 as ds9
import lsst.meas.algorithms as measAlg
import lsst.meas.algorithms.utils as maUtils
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

__all__ = ['MeasurePsfConfig', 'MeasurePsfTask']


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
        default="pca"
    )
    reserve = pexConfig.ConfigurableField(
        target=measAlg.ReserveSourcesTask,
        doc="Reserve sources from fitting"
    )


class MeasurePsfTask(pipeBase.Task):
    """Select stars from a catalog of sources and use them to measure the PSF.

    Parameters
    ----------
    schema : `lsst.afw.table.Schema`, optional
        Schema used to create the output source catalog.
    **kwargs
        Other keyword arguments for `lsst.pipe.base.Task`.

    Notes
    -----
    If schema is not None, 'calib_psf_candidate' and 'calib_psf_used' fields
    will be added to identify which stars were employed in the PSF estimation.

    This task can add fields to the schema, so any code calling this task must
    ensure that these fields are indeed present in the input table.

    The star selector is a subclass of
    ``lsst.meas.algorithms.starSelector.BaseStarSelectorTask``
    and the PSF determiner is a sublcass of
    ``lsst.meas.algorithms.psfDeterminer.BasePsfDeterminerTask``.

    There is no establised set of configuration parameters for these
    algorithms, so once you start modifying parameters, your code is no
    longer portable.
    """    
    ConfigClass = MeasurePsfConfig
    _DefaultName = "measurePsf"

    def __init__(self, schema=None, **kwargs):
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

    @pipeBase.timeMethod
    def run(self, exposure, sources, expId=0, matches=None):
        """Measure the PSF.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to process; measured PSF will be added.
        sources : `lsst.afw.table.SourceCatalog`
            Measured sources on exposure; flag fields will be set marking
            stars chosen by the star selector and the PSF determiner if a
            schema was passed to the task constructor.
        expId : `int`
            Exposure id used for generating random seed.
        matches : `list` of `lsst.afw.table.ReferenceMatch`
            A list of objects as returned by, e.g., ``AstrometryTask``.
            (The list may be of type ``lsst.afw.table.Match``
            with the first member being of type ``lsst.afw.table.SimpleRecord``
            and the second being of type ``lsst.afw.table.SourceRecord`` for
            the reference object and detected object, respectively.)
            Used by star selectors that choose to refer to an external catalog.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result struct with components:
            - ``psf`` : an ``lsst.afw.detection.Psf``, containing the measured
                PSF (also set in the input exposure).
            - ``cellSet`` : an ``lsst.afw.math.SpatialCellSet`` containing the
                PSF candidates as returned by the psf determiner.
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
        self.log.info("PSF star selector found %d candidates" % len(selectionResult.psfCandidates))
        reserveResult = self.reserve.run(selectionResult.goodStarCat, expId=expId)
        # Make list of psf candidates to send to the determiner (omitting those marked as reserved)
        psfDeterminerList = [cand for cand, use
                             in zip(selectionResult.psfCandidates, reserveResult.use) if use]

        if selectionResult.psfCandidates and self.candidateKey is not None:
            for cand in selectionResult.psfCandidates:
                source = cand.getSource()
                source.set(self.candidateKey, True)

        self.log.info("Sending %d candidates to PSF determiner" % len(psfDeterminerList))

        if display:
            frame = display
            if displayExposure:
                ds9.mtv(exposure, frame=frame, title="psf determination")

        #
        # Determine PSF
        #
        psf, cellSet = self.psfDeterminer.determinePsf(exposure, psfDeterminerList, self.metadata,
                                                       flagKey=self.usedKey)
        self.log.info("PSF determination using %d/%d stars." %
                      (self.metadata.getScalar("numGoodStars"), self.metadata.getScalar("numAvailStars")))

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
            psf=psf,
            cellSet=cellSet,
        )

    @property
    def usesMatches(self):
        """Does this task makes use of the "matches" argument?

        Returns
        -------
        result : `bool`
            True if this task uses the "matches" argument to the run method.
        """
        return self.starSelector.usesMatches

#
# Debug code
#


def showPsfSpatialCells(exposure, cellSet, showBadCandidates, frame=1):
    maUtils.showPsfSpatialCells(exposure, cellSet,
                                symb="o", ctype=ds9.CYAN, ctypeUnused=ds9.YELLOW,
                                size=4, frame=frame)
    for cell in cellSet.getCellList():
        for cand in cell.begin(not showBadCandidates):  # maybe include bad candidates
            status = cand.getStatus()
            ds9.dot('+', *cand.getSource().getCentroid(), frame=frame,
                    ctype=ds9.GREEN if status == afwMath.SpatialCellCandidate.GOOD else
                    ds9.YELLOW if status == afwMath.SpatialCellCandidate.UNKNOWN else ds9.RED)


def plotPsfCandidates(cellSet, showBadCandidates=False, frame=1):
    import lsst.afw.display.utils as displayUtils

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
