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
import math
import random
import numpy

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.meas.astrom as measAstrom
from lsst.pipe.tasks.registerImage import RegisterTask
from lsst.meas.algorithms import SourceDetectionTask, PsfAttributes, SingleGaussianPsf
from lsst.ip.diffim import ImagePsfMatchTask, DipoleMeasurementTask, DipoleAnalysis, \
    SourceFlagChecker, KernelCandidateF, cast_KernelCandidateF, makeKernelBasisList, \
    KernelCandidateQa, DiaCatalogSourceSelectorTask, DiaCatalogSourceSelectorConfig, \
    GetCoaddAsTemplateTask, GetCalexpAsTemplateTask
import lsst.ip.diffim.diffimTools as diffimTools
import lsst.ip.diffim.utils as diUtils

FwhmPerSigma = 2 * math.sqrt(2 * math.log(2))
IqrToSigma = 0.741

class ImageDifferenceConfig(pexConfig.Config):
    """Config for ImageDifferenceTask
    """
    doAddCalexpBackground = pexConfig.Field(dtype=bool, default=True,
        doc="Add background to calexp before processing it.  Useful as ipDiffim does background matching.")
    doUseRegister = pexConfig.Field(dtype=bool, default=True,
        doc="Use image-to-image registration to align template with science image")
    doDebugRegister = pexConfig.Field(dtype=bool, default=False,
        doc="Writing debugging data for doUseRegister")
    doSelectSources = pexConfig.Field(dtype=bool, default=True,
        doc="Select stars to use for kernel fitting")
    doSelectDcrCatalog = pexConfig.Field(dtype=bool, default=False,
        doc="Select stars of extreme color as part of the control sample") 
    doSelectVariableCatalog = pexConfig.Field(dtype=bool, default=False,
        doc="Select stars that are variable to be part of the control sample") 
    doSubtract = pexConfig.Field(dtype=bool, default=True, doc="Compute subtracted exposure?")
    doPreConvolve = pexConfig.Field(dtype=bool, default=True,
        doc="Convolve science image by its PSF before PSF-matching?")
    useGaussianForPreConvolution = pexConfig.Field(dtype=bool, default=True,
        doc="Use a simple gaussian PSF model for pre-convolution (else use fit PSF)? "
            "Ignored if doPreConvolve false.")
    doDetection = pexConfig.Field(dtype=bool, default=True, doc="Detect sources?")
    doMerge = pexConfig.Field(dtype=bool, default=True,
        doc="Merge positive and negative diaSources with grow radius set by growFootprint")
    doMatchSources = pexConfig.Field(dtype=bool, default=True,
        doc="Match diaSources with input calexp sources and ref catalog sources")
    doMeasurement = pexConfig.Field(dtype=bool, default=True, doc="Measure diaSources?")
    doWriteSubtractedExp = pexConfig.Field(dtype=bool, default=True, doc="Write difference exposure?")
    doWriteMatchedExp = pexConfig.Field(dtype=bool, default=False,
        doc="Write warped and PSF-matched template coadd exposure?")
    doWriteSources = pexConfig.Field(dtype=bool, default=True, doc="Write sources?")
    doAddMetrics = pexConfig.Field(dtype=bool, default=True,
        doc="Add columns to the source table to hold analysis metrics?")

    coaddName = pexConfig.Field(
        doc="coadd name: typically one of deep or goodSeeing",
        dtype=str,
        default="deep",
    )
    convolveTemplate = pexConfig.Field(
        doc="Which image gets convolved (default = template)",
        dtype=bool,
        default=True
    )
    astrometer = pexConfig.ConfigurableField(
        target = measAstrom.AstrometryTask,
        doc = "astrometry task; used to match sources to reference objects, but not to fit a WCS",
    )
    sourceSelector = pexConfig.ConfigurableField(
        target=DiaCatalogSourceSelectorTask,
        doc="Source selection algorithm",
    )
    subtract = pexConfig.ConfigurableField(
        target=ImagePsfMatchTask,
        doc="Warp and PSF match template to exposure, then subtract",
    )
    detection = pexConfig.ConfigurableField(
        target=SourceDetectionTask,
        doc="Low-threshold detection for final measurement",
    )
    measurement = pexConfig.ConfigurableField(
        target=DipoleMeasurementTask,
        doc="Final source measurement on low-threshold detections; dipole fitting enabled.",
    )
    getTemplate = pexConfig.ConfigurableField(
        target = GetCoaddAsTemplateTask,
        doc = "Subtask to retrieve template exposure and sources",
    )
    controlStepSize = pexConfig.Field(
        doc="What step size (every Nth one) to select a control sample from the kernelSources",
        dtype=int,
        default=5
    )
    controlRandomSeed = pexConfig.Field(
        doc="Random seed for shuffing the control sample",
        dtype=int,
        default=10
    )
    register = pexConfig.ConfigurableField(
        target=RegisterTask,
        doc="Task to enable image-to-image image registration (warping)",
    )
    kernelSourcesFromRef = pexConfig.Field(
        doc="Select sources to measure kernel from reference catalog if True, template if false",
        dtype=bool,
        default=False
    )
    templateSipOrder = pexConfig.Field(dtype=int, default=2,
        doc="Sip Order for fitting the Template Wcs (default is too high, overfitting)")

    growFootprint = pexConfig.Field(dtype=int, default=2,
        doc="Grow positive and negative footprints by this amount before merging")

    diaSourceMatchRadius = pexConfig.Field(dtype=float, default=0.5,
        doc="Match radius (in arcseconds) for DiaSource to Source association")

    def setDefaults(self):
        # Set default source selector and configure defaults for that one and some common alternatives
        self.sourceSelector.name = "secondMoment"
        self.sourceSelector["secondMoment"].clumpNSigma = 2.0
        # defaults are OK for catalog and diacatalog

        self.subtract.kernel.name = "AL"
        self.subtract.kernel.active.fitForBackground = True
        self.subtract.kernel.active.spatialKernelOrder = 1
        self.subtract.kernel.active.spatialBgOrder = 0
        self.doPreConvolve = False
        self.doMatchSources = False
        self.doAddMetrics = False
        self.doUseRegister = False

        # DiaSource Detection
        self.detection.thresholdPolarity = "both"
        self.detection.thresholdValue = 5.5
        self.detection.reEstimateBackground = False
        self.detection.thresholdType = "pixel_stdev"

        # Add filtered flux measurement, the correct measurement for pre-convolved images.
        # Enable all measurements, regardless of doPreConvolved, as it makes data harvesting easier.
        # To change that you must modify algorithms.names in the task's applyOverrides method,
        # after the user has set doPreConvolved.
        self.measurement.algorithms.names.add('base_PeakLikelihoodFlux')

        # For shuffling the control sample
        random.seed(self.controlRandomSeed)

    def validate(self):
        pexConfig.Config.validate(self)
        if self.doMeasurement and not self.doDetection:
            raise ValueError("Cannot run source measurement without source detection.")
        if self.doMerge and not self.doDetection:
            raise ValueError("Cannot run source merging without source detection.")
        if self.doUseRegister and not self.doSelectSources:
            raise ValueError("doUseRegister=True and doSelectSources=False. " +
                             "Cannot run RegisterTask without selecting sources.")


class ImageDifferenceTaskRunner(pipeBase.TaskRunner):
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

    def __init__(self, **kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.makeSubtask("subtract")
        self.makeSubtask("getTemplate")

        if self.config.doUseRegister:
            self.makeSubtask("register")

        if self.config.doSelectSources:
            self.sourceSelector = self.config.sourceSelector.apply()
            self.makeSubtask("astrometer")
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        if self.config.doDetection:
            self.makeSubtask("detection", schema=self.schema)
        if self.config.doMeasurement:
            self.makeSubtask("measurement", schema=self.schema,
                             algMetadata=self.algMetadata)
        if self.config.doMatchSources:
            self.schema.addField("refMatchId", "L", "unique id of reference catalog match")
            self.schema.addField("srcMatchId", "L", "unique id of source match")

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

        # initialize outputs and some intermediate products
        subtractedExposure = None
        subtractRes = None
        selectSources = None
        kernelSources = None
        controlSources = None
        diaSources = None

        # We make one IdFactory that will be used by both icSrc and src datasets;
        # I don't know if this is the way we ultimately want to do things, but at least
        # this ensures the source IDs are fully unique.
        expBits = sensorRef.get("ccdExposureId_bits")
        expId = long(sensorRef.get("ccdExposureId"))
        idFactory = afwTable.IdFactory.makeSource(expId, 64 - expBits)

        # Retrieve the science image we wish to analyze
        exposure = sensorRef.get("calexp", immediate=True)
        if self.config.doAddCalexpBackground:
            mi = exposure.getMaskedImage()
            mi += sensorRef.get("calexpBackground").getImage()
        if not exposure.hasPsf():
            raise pipeBase.TaskError("Exposure has no psf")
        sciencePsf = exposure.getPsf()

        subtractedExposureName = self.config.coaddName + "Diff_differenceExp"
        templateExposure = None  # Stitched coadd exposure
        templateSources = None   # Sources on the template image
        if self.config.doSubtract:
            template = self.getTemplate.run(exposure, sensorRef, templateIdList=templateIdList)
            templateExposure = template.exposure
            templateSources = template.sources

            # compute scienceSigmaOrig: sigma of PSF of science image before pre-convolution
            ctr = afwGeom.Box2D(exposure.getBBox()).getCenter()
            psfAttr = PsfAttributes(sciencePsf, afwGeom.Point2I(ctr))
            scienceSigmaOrig = psfAttr.computeGaussianWidth(psfAttr.ADAPTIVE_MOMENT)

            # sigma of PSF of template image before warping
            ctr = afwGeom.Box2D(templateExposure.getBBox()).getCenter()
            psfAttr = PsfAttributes(templateExposure.getPsf(), afwGeom.Point2I(ctr))
            templateSigma = psfAttr.computeGaussianWidth(psfAttr.ADAPTIVE_MOMENT)

            # if requested, convolve the science exposure with its PSF
            # (properly, this should be a cross-correlation, but our code does not yet support that)
            # compute scienceSigmaPost: sigma of science exposure with pre-convolution, if done,
            # else sigma of original science exposure
            if self.config.doPreConvolve:
                convControl = afwMath.ConvolutionControl()
                # cannot convolve in place, so make a new MI to receive convolved image
                srcMI = exposure.getMaskedImage()
                destMI = srcMI.Factory(srcMI.getDimensions())
                srcPsf = sciencePsf
                if self.config.useGaussianForPreConvolution:
                    # convolve with a simplified PSF model: a double Gaussian
                    kWidth, kHeight = sciencePsf.getLocalKernel().getDimensions()
                    preConvPsf = SingleGaussianPsf(kWidth, kHeight, scienceSigmaOrig)
                else:
                    # convolve with science exposure's PSF model
                    preConvPsf = srcPsf
                afwMath.convolve(destMI, srcMI, preConvPsf.getLocalKernel(), convControl)
                exposure.setMaskedImage(destMI)
                scienceSigmaPost = scienceSigmaOrig * math.sqrt(2)
            else:
                scienceSigmaPost = scienceSigmaOrig

            # If requested, find sources in the image
            if self.config.doSelectSources:
                if not sensorRef.datasetExists("src"):
                    self.log.warn("Src product does not exist; running detection, measurement, selection")
                    # Run own detection and measurement; necessary in nightly processing
                    selectSources = self.subtract.getSelectSources(
                        exposure,
                        sigma = scienceSigmaPost,
                        doSmooth = not self.doPreConvolve,
                        idFactory = idFactory,
                    )
                else:
                    self.log.info("Source selection via src product")
                    # Sources already exist; for data release processing
                    selectSources = sensorRef.get("src")

                # Number of basis functions
                nparam = len(makeKernelBasisList(self.subtract.config.kernel.active,
                                                 referenceFwhmPix=scienceSigmaPost * FwhmPerSigma,
                                                 targetFwhmPix=templateSigma * FwhmPerSigma))

                if self.config.doAddMetrics:
                    # Modify the schema of all Sources
                    kcQa = KernelCandidateQa(nparam)
                    selectSources = kcQa.addToSchema(selectSources)

                if self.config.kernelSourcesFromRef:
                    # match exposure sources to reference catalog
                    astromRet = self.astrometer.loadAndMatch(exposure=exposure, sourceCat=selectSources)
                    matches = astromRet.matches
                elif templateSources:
                    # match exposure sources to template sources
                    matches = afwTable.matchRaDec(templateSources, selectSources, 1.0*afwGeom.arcseconds,
                                                  False)
                else:
                    raise RuntimeError("doSelectSources=True and kernelSourcesFromRef=False," +
                                       "but template sources not available. Cannot match science " +
                                       "sources with template sources. Run process* on data from " +
                                       "which templates are built.")

                kernelSources = self.sourceSelector.selectStars(exposure, selectSources,
                    matches=matches).starCat

                random.shuffle(kernelSources, random.random)
                controlSources = kernelSources[::self.config.controlStepSize]
                kernelSources = [k for i,k in enumerate(kernelSources) if i % self.config.controlStepSize]

                if self.config.doSelectDcrCatalog:
                    redSelector  = DiaCatalogSourceSelectorTask(
                        DiaCatalogSourceSelectorConfig(grMin=self.sourceSelector.config.grMax, grMax=99.999))
                    redSources   = redSelector.selectStars(exposure, selectSources, matches=matches).starCat
                    controlSources.extend(redSources)

                    blueSelector = DiaCatalogSourceSelectorTask(
                        DiaCatalogSourceSelectorConfig(grMin=-99.999, grMax=self.sourceSelector.config.grMin))
                    blueSources  = blueSelector.selectStars(exposure, selectSources, matches=matches).starCat
                    controlSources.extend(blueSources)

                if self.config.doSelectVariableCatalog:
                    varSelector = DiaCatalogSourceSelectorTask(
                        DiaCatalogSourceSelectorConfig(includeVariable=True))
                    varSources  = varSelector.selectStars(exposure, selectSources, matches=matches).starCat
                    controlSources.extend(varSources)

                self.log.info("Selected %d / %d sources for Psf matching (%d for control sample)" 
                              % (len(kernelSources), len(selectSources), len(controlSources)))
            allresids = {}
            if self.config.doUseRegister:
                self.log.info("Registering images")

                if templateSources is None:
                    # Run detection on the template, which is
                    # temporarily background-subtracted
                    templateSources = self.subtract.getSelectSources(
                        templateExposure,
                        sigma=templateSigma,
                        doSmooth=True,
                        idFactory=idFactory
                    )

                # Third step: we need to fit the relative astrometry.
                #
                wcsResults = self.fitAstrometry(templateSources, templateExposure, selectSources)
                warpedExp = self.register.warpExposure(templateExposure, wcsResults.wcs,
                                            exposure.getWcs(), exposure.getBBox())
                templateExposure = warpedExp

                # Create debugging outputs on the astrometric
                # residuals as a function of position.  Persistence
                # not yet implemented; expected on (I believe) #2636.
                if self.config.doDebugRegister:
                    # Grab matches to reference catalog
                    srcToMatch = {x.second.getId() : x.first for x in matches}

                    refCoordKey = wcsResults.matches[0].first.getTable().getCoordKey()
                    inCentroidKey = wcsResults.matches[0].second.getTable().getCentroidKey()
                    sids      = [m.first.getId() for m in wcsResults.matches]
                    positions = [m.first.get(refCoordKey) for m in wcsResults.matches]
                    residuals = [m.first.get(refCoordKey).getOffsetFrom(wcsResults.wcs.pixelToSky(
                                m.second.get(inCentroidKey))) for m in wcsResults.matches]
                    allresids = dict(zip(sids, zip(positions, residuals)))

                    cresiduals = [m.first.get(refCoordKey).getTangentPlaneOffset(
                            wcsResults.wcs.pixelToSky(
                                m.second.get(inCentroidKey))) for m in wcsResults.matches]
                    colors    = numpy.array([-2.5*numpy.log10(srcToMatch[x].get("g"))
                                              + 2.5*numpy.log10(srcToMatch[x].get("r")) 
                                              for x in sids if x in srcToMatch.keys()])
                    dlong     = numpy.array([r[0].asArcseconds() for s,r in zip(sids, cresiduals) 
                                             if s in srcToMatch.keys()])
                    dlat      = numpy.array([r[1].asArcseconds() for s,r in zip(sids, cresiduals) 
                                             if s in srcToMatch.keys()])
                    idx1      = numpy.where(colors<self.sourceSelector.config.grMin)
                    idx2      = numpy.where((colors>=self.sourceSelector.config.grMin)&
                                            (colors<=self.sourceSelector.config.grMax))
                    idx3      = numpy.where(colors>self.sourceSelector.config.grMax)
                    rms1Long  = IqrToSigma*(numpy.percentile(dlong[idx1],75)-numpy.percentile(dlong[idx1],25))
                    rms1Lat   = IqrToSigma*(numpy.percentile(dlat[idx1],75)-numpy.percentile(dlat[idx1],25))
                    rms2Long  = IqrToSigma*(numpy.percentile(dlong[idx2],75)-numpy.percentile(dlong[idx2],25))
                    rms2Lat   = IqrToSigma*(numpy.percentile(dlat[idx2],75)-numpy.percentile(dlat[idx2],25))
                    rms3Long  = IqrToSigma*(numpy.percentile(dlong[idx3],75)-numpy.percentile(dlong[idx3],25))
                    rms3Lat   = IqrToSigma*(numpy.percentile(dlat[idx3],75)-numpy.percentile(dlat[idx3],25))
                    self.log.info("Blue star offsets'': %.3f %.3f, %.3f %.3f"  % (numpy.median(dlong[idx1]), 
                                                                                  rms1Long,
                                                                                  numpy.median(dlat[idx1]), 
                                                                                  rms1Lat))
                    self.log.info("Green star offsets'': %.3f %.3f, %.3f %.3f"  % (numpy.median(dlong[idx2]), 
                                                                                   rms2Long,
                                                                                   numpy.median(dlat[idx2]), 
                                                                                   rms2Lat))
                    self.log.info("Red star offsets'': %.3f %.3f, %.3f %.3f"  % (numpy.median(dlong[idx3]), 
                                                                                 rms3Long,
                                                                                 numpy.median(dlat[idx3]), 
                                                                                 rms3Lat))

                    self.metadata.add("RegisterBlueLongOffsetMedian", numpy.median(dlong[idx1]))
                    self.metadata.add("RegisterGreenLongOffsetMedian", numpy.median(dlong[idx2]))
                    self.metadata.add("RegisterRedLongOffsetMedian", numpy.median(dlong[idx3]))
                    self.metadata.add("RegisterBlueLongOffsetStd", rms1Long)
                    self.metadata.add("RegisterGreenLongOffsetStd", rms2Long)
                    self.metadata.add("RegisterRedLongOffsetStd", rms3Long)

                    self.metadata.add("RegisterBlueLatOffsetMedian", numpy.median(dlat[idx1]))
                    self.metadata.add("RegisterGreenLatOffsetMedian", numpy.median(dlat[idx2]))
                    self.metadata.add("RegisterRedLatOffsetMedian", numpy.median(dlat[idx3]))
                    self.metadata.add("RegisterBlueLatOffsetStd", rms1Lat)
                    self.metadata.add("RegisterGreenLatOffsetStd", rms2Lat)
                    self.metadata.add("RegisterRedLatOffsetStd", rms3Lat)

            # warp template exposure to match exposure,
            # PSF match template exposure to exposure,
            # then return the difference

            #Return warped template...  Construct sourceKernelCand list after subtract
            self.log.info("Subtracting images")
            subtractRes = self.subtract.subtractExposures(
                templateExposure=templateExposure,
                scienceExposure=exposure,
                candidateList=kernelSources,
                convolveTemplate=self.config.convolveTemplate,
                doWarping=not self.config.doUseRegister
            )
            subtractedExposure = subtractRes.subtractedExposure

            if self.config.doWriteMatchedExp:
                sensorRef.put(subtractRes.matchedExposure, self.config.coaddName + "Diff_matchedExp")

        if self.config.doDetection:
            self.log.info("Running diaSource detection")
            if subtractedExposure is None:
                subtractedExposure = sensorRef.get(subtractedExposureName)

            # Get Psf from the appropriate input image if it doesn't exist
            if not subtractedExposure.hasPsf():
                if self.config.convolveTemplate:
                    subtractedExposure.setPsf(exposure.getPsf())
                else:
                    if templateExposure is None:
                        template = self.getTemplate.run(exposure, sensorRef, templateIdList=templateIdList)
                    subtractedExposure.setPsf(template.exposure.getPsf())

            # Erase existing detection mask planes
            mask  = subtractedExposure.getMaskedImage().getMask()
            mask &= ~(mask.getPlaneBitMask("DETECTED") | mask.getPlaneBitMask("DETECTED_NEGATIVE"))

            table = afwTable.SourceTable.make(self.schema, idFactory)
            table.setMetadata(self.algMetadata)
            results = self.detection.makeSourceCatalog(
                table=table,
                exposure=subtractedExposure,
                doSmooth=not self.config.doPreConvolve
                )

            if self.config.doMerge:
                fpSet = results.fpSets.positive
                fpSet.merge(results.fpSets.negative, self.config.growFootprint,
                            self.config.growFootprint, False)
                diaSources = afwTable.SourceCatalog(table)
                fpSet.makeSources(diaSources)
                self.log.info("Merging detections into %d sources" % (len(diaSources)))
            else:
                diaSources = results.sources

            if self.config.doMeasurement:
                self.log.info("Running diaSource measurement")
                self.measurement.run(diaSources, subtractedExposure)

            # Match with the calexp sources if possible
            if self.config.doMatchSources:
                if sensorRef.datasetExists("src"):
                    # Create key,val pair where key=diaSourceId and val=sourceId
                    matchRadAsec = self.config.diaSourceMatchRadius
                    matchRadPixel = matchRadAsec / exposure.getWcs().pixelScale().asArcseconds()

                    srcMatches = afwTable.matchXy(sensorRef.get("src"), diaSources, matchRadPixel, True)
                    srcMatchDict = dict([(srcMatch.second.getId(), srcMatch.first.getId()) for 
                                         srcMatch in srcMatches])
                    self.log.info("Matched %d / %d diaSources to sources" % (len(srcMatchDict),
                                                                             len(diaSources)))
                else:
                    self.log.warn("Src product does not exist; cannot match with diaSources")
                    srcMatchDict = {}

                # Create key,val pair where key=diaSourceId and val=refId
                refAstromConfig = measAstrom.AstrometryConfig()
                refAstromConfig.matcher.maxMatchDistArcSec = matchRadAsec
                refAstrometer = measAstrom.AstrometryTask(refAstromConfig)
                astromRet = refAstrometer.run(exposure=exposure, sourceCat=diaSources)
                refMatches = astromRet.matches
                if refMatches is None:
                    self.log.warn("No diaSource matches with reference catalog")
                    refMatchDict = {}
                else:
                    self.log.info("Matched %d / %d diaSources to reference catalog" % (len(refMatches),
                                                                                       len(diaSources)))
                    refMatchDict = dict([(refMatch.second.getId(), refMatch.first.getId()) for \
                                             refMatch in refMatches])

                # Assign source Ids
                for diaSource in diaSources:
                    sid = diaSource.getId()
                    if srcMatchDict.has_key(sid):
                        diaSource.set("srcMatchId", srcMatchDict[sid])
                    if refMatchDict.has_key(sid):
                        diaSource.set("refMatchId", refMatchDict[sid])

            if diaSources is not None and self.config.doWriteSources:
                sensorRef.put(diaSources, self.config.coaddName + "Diff_diaSrc")

            if self.config.doAddMetrics and self.config.doSelectSources:
                self.log.info("Evaluating metrics and control sample")

                kernelCandList = []
                for cell in subtractRes.kernelCellSet.getCellList():
                    for cand in cell.begin(False): # include bad candidates
                        kernelCandList.append(cast_KernelCandidateF(cand))

                # Get basis list to build control sample kernels
                basisList = afwMath.cast_LinearCombinationKernel(
                    kernelCandList[0].getKernel(KernelCandidateF.ORIG)).getKernelList()

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
                    kcQa.aggregate(selectSources, self.metadata, allresids, diaSources)
                else:
                    kcQa.aggregate(selectSources, self.metadata, allresids)

                sensorRef.put(selectSources, self.config.coaddName + "Diff_kernelSrc")

        if self.config.doWriteSubtractedExp:
            sensorRef.put(subtractedExposure, subtractedExposureName)

        self.runDebug(exposure, subtractRes, selectSources, kernelSources, diaSources)
        return pipeBase.Struct(
            subtractedExposure=subtractedExposure,
            subtractRes=subtractRes,
            sources=diaSources,
        )

    def fitAstrometry(self, templateSources, templateExposure, selectSources):
        """Fit the relative astrometry between templateSources and selectSources

        @todo remove this method. It originally fit a new WCS to the template before calling register.run
        because our TAN-SIP fitter behaved badly for points far from CRPIX, but that's been fixed.
        It remains because a subtask overrides it.
        """
        results = self.register.run(templateSources, templateExposure.getWcs(),
                                    templateExposure.getBBox(), selectSources)
        return results

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
                if not source in kernelSources:
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
            flagChecker   = SourceFlagChecker(diaSources)
            isFlagged     = [flagChecker(x) for x in diaSources]
            isDipole      = [x.get("classification.dipole") for x in diaSources]
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
        parser.add_id_argument("--templateId", "calexp", doMakeDataRefList=False,
                               help="Optional template data ID (visit only), e.g. --templateId visit=6789")
        return parser

class Winter2013ImageDifferenceConfig(ImageDifferenceConfig):
    winter2013WcsShift = pexConfig.Field(dtype=float, default=0.0,
        doc="Shift stars going into RegisterTask by this amount")
    winter2013WcsRms = pexConfig.Field(dtype=float, default=0.0,
        doc="Perturb stars going into RegisterTask by this amount")

    def setDefaults(self):
        ImageDifferenceConfig.setDefaults(self)
        self.getTemplate.retarget(GetCalexpAsTemplateTask)


class Winter2013ImageDifferenceTask(ImageDifferenceTask):
    """!Image difference Task used in the Winter 2013 data challege.
    Enables testing the effects of registration shifts and scatter.

    For use with winter 2013 simulated images:
    Use --templateId visit=88868666 for sparse data
        --templateId visit=22222200 for dense data (g)
        --templateId visit=11111100 for dense data (i)
    """
    ConfigClass = Winter2013ImageDifferenceConfig
    _DefaultName = "winter2013ImageDifference"

    def __init__(self, **kwargs):
        ImageDifferenceTask.__init__(self, **kwargs)

    def fitAstrometry(self, templateSources, templateExposure, selectSources):
        """Fit the relative astrometry between templateSources and selectSources"""
        if self.config.winter2013WcsShift > 0.0:
            offset = afwGeom.Extent2D(self.config.winter2013WcsShift,
                                      self.config.winter2013WcsShift)
            cKey = templateSources[0].getTable().getCentroidKey()
            for source in templateSources:
                centroid = source.get(cKey)
                source.set(cKey, centroid+offset)
        elif self.config.winter2013WcsRms > 0.0:
            cKey = templateSources[0].getTable().getCentroidKey()
            for source in templateSources:
                offset = afwGeom.Extent2D(self.config.winter2013WcsRms*numpy.random.normal(),
                                          self.config.winter2013WcsRms*numpy.random.normal())
                centroid = source.get(cKey)
                source.set(cKey, centroid+offset)

        results = self.register.run(templateSources, templateExposure.getWcs(),
                                    templateExposure.getBBox(), selectSources)
        return results
