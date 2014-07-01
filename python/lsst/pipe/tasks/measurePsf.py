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
import lsst.afw.table as afwTable

class MeasurePsfConfig(pexConfig.Config):
    starSelector = measAlg.starSelectorRegistry.makeField("Star selection algorithm", default="secondMoment")
    psfDeterminer = measAlg.psfDeterminerRegistry.makeField("PSF Determination algorithm", default="pca")

## \addtogroup LSST_task_documentation
## \{
## \page MeasurePsfTask
## \ref MeasurePsfTask_ "MeasurePsfTask"
##      Measure the PSF
## \}

class MeasurePsfTask(pipeBase.Task):
    """!
\anchor MeasurePsfTask_
    
Conversion notes:
    Split out of Calibrate since it seemed a good self-contained task
    
    @warning
    - I'm not sure I'm using metadata correctly (to replace old sdqa code)
    - The star selector and psf determiner registries will have to be modified to return a class,
      which has a ConfigClass attribute and can be instantiated with a config. Until then, there's no
      obvious way to get a registry algorithm's Config from another Config.
    """
    ConfigClass = MeasurePsfConfig

    def __init__(self, schema=None, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)
        if schema is not None:
            self.candidateKey = schema.addField(
                "calib.psf.candidate", type="Flag",
                doc=("Flag set if the source was a candidate for PSF determination, "
                     "as determined by the '%s' star selector.") % self.config.starSelector.name
                )
            self.usedKey = schema.addField(
                "calib.psf.used", type="Flag",
                doc=("Flag set if the source was actually used for PSF determination, "
                     "as determined by the '%s' PSF determiner.") % self.config.psfDeterminer.name
                )
        else:
            self.candidateKey = None
            self.usedKey = None
        self.starSelector = self.config.starSelector.apply()
        self.psfDeterminer = self.config.psfDeterminer.apply()
        
    @pipeBase.timeMethod
    def run(self, exposure, sources, matches=None):
        """Measure the PSF

        @param[in,out]   exposure      Exposure to process; measured PSF will be installed here as well.
        @param[in,out]   sources       Measured sources on exposure; flag fields will be set marking
                                       stars chosen by the star selector and PSF determiner.
        @param[in]       matches       ReferenceMatchVector, as returned by the AstrometryTask, used
                                       by star selectors that refer to an external catalog.
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
        psfCandidateList = self.starSelector.selectStars(exposure, sources, matches=matches)
        if psfCandidateList and self.candidateKey is not None:
            for cand in psfCandidateList:
                source = cand.getSource()
                source.set(self.candidateKey, True)

        self.log.info("PSF star selector found %d candidates" % len(psfCandidateList))

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
                    chi2 = numpy.nan

                stamps.append((im, "%d%s" %
                               (maUtils.splitId(cand.getSource().getId(), True)["objId"], chi2),
                               cand.getStatus()))
            except Exception, e:
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
        except Exception as e:
            if not showBadCandidates:
                showBadCandidates = True
                continue
        break

    return frame
