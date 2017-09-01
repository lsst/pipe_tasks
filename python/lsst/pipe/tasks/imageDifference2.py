#
# LSST Data Management System
# Copyright 2012 LSST Corporation.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
from __future__ import absolute_import, division, print_function

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
import lsst.afw.table as afwTable
from lsst.ip.diffim import MakeDiffimTask, ProcessDiffimTask, \
    GetCoaddAsTemplateTask, KernelCandidateF, \
    SourceFlagChecker, DipoleAnalysis
import lsst.ip.diffim.diffimTools as diffimTools
import lsst.ip.diffim.utils as diUtils


class ImageDifferenceConfig(pexConfig.Config):
    """Config for ImageDifferenceTask
    """
    doAddCalexpBackground = pexConfig.Field(dtype=bool, default=True,
                                            doc="Add background to calexp before processing it.  "
                                            "Useful as ipDiffim does background matching.")

    doDetection = pexConfig.Field(dtype=bool, default=True, doc="Detect sources?")

    coaddName = pexConfig.Field(
        doc="coadd name: typically one of deep or goodSeeing",
        dtype=str,
        default="deep",
    )

    makeDiffim = pexConfig.ConfigurableField(
        target=MakeDiffimTask,
        doc="Perform image subtraction"
    )

    processDiffim = pexConfig.ConfigurableField(
        target=ProcessDiffimTask,
        doc="Process subtracted image: detect and measure DIASources"
    )

    getTemplate = pexConfig.ConfigurableField(
        target=GetCoaddAsTemplateTask,
        doc="Subtask to retrieve template exposure and sources",
    )

    def setDefaults(self):
        # Add filtered flux measurement, the correct measurement for pre-convolved images.
        # Enable all measurements, regardless of doPreConvolved, as it makes data harvesting easier.
        # To change that you must modify algorithms.names in the task's applyOverrides method,
        # after the user has set doPreConvolved.
        self.processDiffim.measurement.algorithms.names.add('base_PeakLikelihoodFlux')

    def validate(self):
        pexConfig.Config.validate(self)
        if self.processDiffim.doMeasurement and not self.doDetection:
            raise ValueError("Cannot run source measurement without source detection.")
        if self.processDiffim.doMerge and not self.doDetection:
            raise ValueError("Cannot run source merging without source detection.")


class ImageDifferenceTaskRunner(pipeBase.ButlerInitializedTaskRunner):

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return pipeBase.TaskRunner.getTargetList(parsedCmd, templateIdList=parsedCmd.templateId.idList,
                                                 **kwargs)


class ImageDifferenceTask(pipeBase.CmdLineTask):
    """Subtract an image from a template and measure the result
    """
    ConfigClass = ImageDifferenceConfig
    RunnerClass = ImageDifferenceTaskRunner
    _DefaultName = "imageDifference"

    def __init__(self, butler=None, **kwargs):
        """!Construct an ImageDifference Task

        @param[in] butler  Butler object to use in constructing reference object loaders
        """
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.makeSubtask("getTemplate")

        self.schema = afwTable.SourceTable.makeMinimalSchema()

        self.makeSubtask("makeDiffim", butler=butler)
        if self.config.doDetection:
            self.makeSubtask("processDiffim", schema=self.schema)

        self.algMetadata = dafBase.PropertyList()
        self.makeDiffim.algMetadata = self.processDiffim.algMetadata = self.algMetadata

        self.makeDiffim.doAddCalexpBackground = self.config.doAddCalexpBackground
        if self.config.doDetection:
            self.processDiffim.doAddCalexpBackground = self.config.doAddCalexpBackground

    @pipeBase.timeMethod
    def run(self, sensorRef, templateIdList=None):
        """Subtract an image from a template coadd and measure the result

        Steps include:
        - warp template coadd to match WCS of image
        - PSF match image to warped template
        - subtract image from PSF-matched, warped template
        - persist difference image
        - detect sources
        - measure sources

        @param sensorRef: sensor-level butler data reference, used for the following data products:
        Input only:
        - calexp
        - psf
        - ccdExposureId
        - ccdExposureId_bits
        - self.config.coaddName + "Coadd_skyMap"
        - self.config.coaddName + "Coadd"
        Input or output, depending on config:
        - self.config.coaddName + "Diff_subtractedExp"
        Output, depending on config:
        - self.config.coaddName + "Diff_matchedExp"
        - self.config.coaddName + "Diff_src"

        @return pipe_base Struct containing these fields:
        - subtractedExposure: exposure after subtracting template;
            the unpersisted version if subtraction not run but detection run
            None if neither subtraction nor detection run (i.e. nothing useful done)
        - subtractRes: results of subtraction task; None if subtraction not run
        - sources: detected and possibly measured sources; None if detection not run
        """
        self.log.info("Processing %s" % (sensorRef.dataId))

        # We make one IdFactory that will be used by both icSrc and src datasets;
        # I don't know if this is the way we ultimately want to do things, but at least
        # this ensures the source IDs are fully unique.
        expBits = sensorRef.get("ccdExposureId_bits")
        expId = int(sensorRef.get("ccdExposureId"))
        idFactory = afwTable.IdFactory.makeSource(expId, 64 - expBits)

        # Retrieve the science image we wish to analyze
        exposure = sensorRef.get("calexp", immediate=True)
        if self.config.doAddCalexpBackground:
            mi = exposure.getMaskedImage()
            mi += sensorRef.get("calexpBackground").getImage()
        if not exposure.hasPsf():
            raise pipeBase.TaskError("Exposure has no psf")

        template = self.getTemplate.run(exposure, sensorRef, templateIdList=templateIdList)

        mdResult = self.makeDiffim.doMakeDiffim(template, exposure,
                                                idFactory=idFactory, sensorRef=sensorRef)
        subtractedExposure = mdResult.subtractedExposure
        # Zogy does not yet return a matched exposure
        matchedExposure = mdResult.matchedExposure if 'matchedExposure' in mdResult.getDict() else None

        selectSources = None
        if 'selectSourceResult' in mdResult.getDict():
            selectSourceResult = mdResult.selectSourceResult
            selectSources = selectSourceResult.selectSources
        pdResult = self.processDiffim.doProcessDiffim(subtractedExposure, exposure, matchedExposure,
                                                      self.makeDiffim.config.doPreConvolve,
                                                      selectSources=selectSources,
                                                      idFactory=idFactory, sensorRef=sensorRef)
        diaSources = pdResult.sources

        if (self.makeDiffim.config.doAddMetrics and self.makeDiffim.config.doSelectSources and
                'selectSourceResult' in mdResult.getDict()):
            selectSourceResult = mdResult.selectSourceResult
            selectSources = selectSourceResult.selectSources
            self.doEvaluateMetrics(mdResult, selectSourceResult.controlSources, selectSources,
                                   diaSources, selectSourceResult.kcQa, selectSourceResult.nparam,
                                   mdResult.allresids, exposure)
            sensorRef.put(selectSources, self.config.coaddName + "Diff_kernelSrc")

            kernelSources = selectSourceResult.kernelSources
            self.runDebug(exposure, mdResult, selectSources, kernelSources, diaSources)

        return pipeBase.Struct(
            subtractedExposure=subtractedExposure,
            subtractRes=mdResult,
            sources=diaSources,
        )

    def doEvaluateMetrics(self, subtractRes, controlSources, selectSources, diaSources, kcQa,
                          nparam, allresids, exposure):
        self.log.info("Evaluating metrics and control sample")

        kernelCandList = []
        for cell in subtractRes.kernelCellSet.getCellList():
            for cand in cell.begin(False):  # include bad candidates
                kernelCandList.append(cand)

        # Get basis list to build control sample kernels
        basisList = kernelCandList[0].getKernel(KernelCandidateF.ORIG).getKernelList()

        controlCandList = \
            diffimTools.sourceTableToCandidateList(controlSources,
                                                   subtractRes.warpedExposure, exposure,
                                                   self.config.subtract.kernel.active,
                                                   self.config.subtract.kernel.active.detectionConfig,
                                                   self.log, doBuild=True, basisList=basisList)

        kcQa.apply(kernelCandList, subtractRes.psfMatchingKernel, subtractRes.backgroundModel,
                   dof=nparam)
        kcQa.apply(controlCandList, subtractRes.psfMatchingKernel, subtractRes.backgroundModel)

        if self.config.doDetection:
            self.kcQa.aggregate(selectSources, self.metadata, allresids, diaSources)
        else:
            self.kcQa.aggregate(selectSources, self.metadata, allresids)

    def runDebug(self, exposure, subtractRes, selectSources, kernelSources, diaSources):
        """@todo Test and update for current debug display and slot names
        """
        import lsstDebug
        display = lsstDebug.Info(__name__).display
        showSubtracted = lsstDebug.Info(__name__).showSubtracted
        showPixelResiduals = lsstDebug.Info(__name__).showPixelResiduals
        showDiaSources = lsstDebug.Info(__name__).showDiaSources
        showDipoles = lsstDebug.Info(__name__).showDipoles
        maskTransparency = lsstDebug.Info(__name__).maskTransparency
        if display:
            import lsst.afw.display.ds9 as ds9
            if not maskTransparency:
                maskTransparency = 0
            ds9.setMaskTransparency(maskTransparency)

        if display and showSubtracted:
            ds9.mtv(subtractRes.subtractedExposure, frame=lsstDebug.frame, title="Subtracted image")
            mi = subtractRes.subtractedExposure.getMaskedImage()
            x0, y0 = mi.getX0(), mi.getY0()
            with ds9.Buffering():
                for s in diaSources:
                    x, y = s.getX() - x0, s.getY() - y0
                    ctype = "red" if s.get("flags.negative") else "yellow"
                    if (s.get("flags.pixel.interpolated.center") or s.get("flags.pixel.saturated.center") or
                            s.get("flags.pixel.cr.center")):
                        ptype = "x"
                    elif (s.get("flags.pixel.interpolated.any") or s.get("flags.pixel.saturated.any") or
                          s.get("flags.pixel.cr.any")):
                        ptype = "+"
                    else:
                        ptype = "o"
                    ds9.dot(ptype, x, y, size=4, frame=lsstDebug.frame, ctype=ctype)
            lsstDebug.frame += 1

        if display and showPixelResiduals and selectSources:
            nonKernelSources = []
            for source in selectSources:
                if source not in kernelSources:
                    nonKernelSources.append(source)

            diUtils.plotPixelResiduals(exposure,
                                       subtractRes.warpedExposure,
                                       subtractRes.subtractedExposure,
                                       subtractRes.kernelCellSet,
                                       subtractRes.psfMatchingKernel,
                                       subtractRes.backgroundModel,
                                       nonKernelSources,
                                       self.subtract.config.kernel.active.detectionConfig,
                                       origVariance=False)
            diUtils.plotPixelResiduals(exposure,
                                       subtractRes.warpedExposure,
                                       subtractRes.subtractedExposure,
                                       subtractRes.kernelCellSet,
                                       subtractRes.psfMatchingKernel,
                                       subtractRes.backgroundModel,
                                       nonKernelSources,
                                       self.subtract.config.kernel.active.detectionConfig,
                                       origVariance=True)
        if display and showDiaSources:
            flagChecker = SourceFlagChecker(diaSources)
            isFlagged = [flagChecker(x) for x in diaSources]
            isDipole = [x.get("classification.dipole") for x in diaSources]
            diUtils.showDiaSources(diaSources, subtractRes.subtractedExposure, isFlagged, isDipole,
                                   frame=lsstDebug.frame)
            lsstDebug.frame += 1

        if display and showDipoles:
            DipoleAnalysis().displayDipoles(subtractRes.subtractedExposure, diaSources,
                                            frame=lsstDebug.frame)
            lsstDebug.frame += 1

    def _getConfigName(self):
        """Return the name of the config dataset
        """
        return "%sDiff_config" % (self.config.coaddName,)

    def _getMetadataName(self):
        """Return the name of the metadata dataset
        """
        return "%sDiff_metadata" % (self.config.coaddName,)

    def getSchemaCatalogs(self):
        """Return a dict of empty catalogs for each catalog dataset produced by this task."""
        diaSrc = afwTable.SourceCatalog(self.schema)
        diaSrc.getTable().setMetadata(self.algMetadata)
        return {self.config.coaddName + "Diff_diaSrc": diaSrc}

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "calexp", help="data ID, e.g. --id visit=12345 ccd=1,2")
        parser.add_id_argument("--templateId", "calexp", doMakeDataRefList=True,
                               help="Optional template data ID (visit only), e.g. --templateId visit=6789")
        return parser

