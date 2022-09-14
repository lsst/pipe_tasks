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

__all__ = ["AssembleCoaddTask", "AssembleCoaddConnections", "AssembleCoaddConfig",
           "CompareWarpAssembleCoaddTask", "CompareWarpAssembleCoaddConfig"]

import copy
import numpy
import warnings
import logging
import lsst.pex.config as pexConfig
import lsst.pex.exceptions as pexExceptions
import lsst.geom as geom
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
import lsst.meas.algorithms as measAlg
import lsstDebug
import lsst.utils as utils
from lsst.skymap import BaseSkyMap
from .coaddBase import CoaddBaseTask, makeSkyInfo, reorderAndPadList
from .interpImage import InterpImageTask
from .scaleZeroPoint import ScaleZeroPointTask
from .maskStreaks import MaskStreaksTask
from .healSparseMapping import HealSparseInputMapTask
from lsst.meas.algorithms import SourceDetectionTask, AccumulatorMeanStack, ScaleVarianceTask
from lsst.utils.timer import timeMethod
from deprecated.sphinx import deprecated

log = logging.getLogger(__name__)


class AssembleCoaddConnections(pipeBase.PipelineTaskConnections,
                               dimensions=("tract", "patch", "band", "skymap"),
                               defaultTemplates={"inputCoaddName": "deep",
                                                 "outputCoaddName": "deep",
                                                 "warpType": "direct",
                                                 "warpTypeSuffix": ""}):

    inputWarps = pipeBase.connectionTypes.Input(
        doc=("Input list of warps to be assemebled i.e. stacked."
             "WarpType (e.g. direct, psfMatched) is controlled by the warpType config parameter"),
        name="{inputCoaddName}Coadd_{warpType}Warp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
        deferLoad=True,
        multiple=True
    )
    skyMap = pipeBase.connectionTypes.Input(
        doc="Input definition of geometry/bbox and projection/wcs for coadded exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap", ),
    )
    selectedVisits = pipeBase.connectionTypes.Input(
        doc="Selected visits to be coadded.",
        name="{outputCoaddName}Visits",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "tract", "patch", "skymap", "band")
    )
    brightObjectMask = pipeBase.connectionTypes.PrerequisiteInput(
        doc=("Input Bright Object Mask mask produced with external catalogs to be applied to the mask plane"
             " BRIGHT_OBJECT."),
        name="brightObjectMask",
        storageClass="ObjectMaskCatalog",
        dimensions=("tract", "patch", "skymap", "band"),
    )
    coaddExposure = pipeBase.connectionTypes.Output(
        doc="Output coadded exposure, produced by stacking input warps",
        name="{outputCoaddName}Coadd{warpTypeSuffix}",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band"),
    )
    nImage = pipeBase.connectionTypes.Output(
        doc="Output image of number of input images per pixel",
        name="{outputCoaddName}Coadd_nImage",
        storageClass="ImageU",
        dimensions=("tract", "patch", "skymap", "band"),
    )
    inputMap = pipeBase.connectionTypes.Output(
        doc="Output healsparse map of input images",
        name="{outputCoaddName}Coadd_inputMap",
        storageClass="HealSparseMap",
        dimensions=("tract", "patch", "skymap", "band"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config.doMaskBrightObjects:
            self.prerequisiteInputs.remove("brightObjectMask")

        if not config.doSelectVisits:
            self.inputs.remove("selectedVisits")

        if not config.doNImage:
            self.outputs.remove("nImage")

        if not self.config.doInputMap:
            self.outputs.remove("inputMap")


class AssembleCoaddConfig(CoaddBaseTask.ConfigClass, pipeBase.PipelineTaskConfig,
                          pipelineConnections=AssembleCoaddConnections):
    warpType = pexConfig.Field(
        doc="Warp name: one of 'direct' or 'psfMatched'",
        dtype=str,
        default="direct",
    )
    subregionSize = pexConfig.ListField(
        dtype=int,
        doc="Width, height of stack subregion size; "
        "make small enough that a full stack of images will fit into memory at once.",
        length=2,
        default=(2000, 2000),
    )
    statistic = pexConfig.Field(
        dtype=str,
        doc="Main stacking statistic for aggregating over the epochs.",
        default="MEANCLIP",
    )
    doOnlineForMean = pexConfig.Field(
        dtype=bool,
        doc="Perform online coaddition when statistic=\"MEAN\" to save memory?",
        default=False,
    )
    doSigmaClip = pexConfig.Field(
        dtype=bool,
        doc="Perform sigma clipped outlier rejection with MEANCLIP statistic? (DEPRECATED)",
        default=False,
    )
    sigmaClip = pexConfig.Field(
        dtype=float,
        doc="Sigma for outlier rejection; ignored if non-clipping statistic selected.",
        default=3.0,
    )
    clipIter = pexConfig.Field(
        dtype=int,
        doc="Number of iterations of outlier rejection; ignored if non-clipping statistic selected.",
        default=2,
    )
    calcErrorFromInputVariance = pexConfig.Field(
        dtype=bool,
        doc="Calculate coadd variance from input variance by stacking statistic."
            "Passed to StatisticsControl.setCalcErrorFromInputVariance()",
        default=True,
    )
    scaleZeroPoint = pexConfig.ConfigurableField(
        target=ScaleZeroPointTask,
        doc="Task to adjust the photometric zero point of the coadd temp exposures",
    )
    doInterp = pexConfig.Field(
        doc="Interpolate over NaN pixels? Also extrapolate, if necessary, but the results are ugly.",
        dtype=bool,
        default=True,
    )
    interpImage = pexConfig.ConfigurableField(
        target=InterpImageTask,
        doc="Task to interpolate (and extrapolate) over NaN pixels",
    )
    doWrite = pexConfig.Field(
        doc="Persist coadd?",
        dtype=bool,
        default=True,
    )
    doNImage = pexConfig.Field(
        doc="Create image of number of contributing exposures for each pixel",
        dtype=bool,
        default=False,
    )
    doUsePsfMatchedPolygons = pexConfig.Field(
        doc="Use ValidPolygons from shrunk Psf-Matched Calexps? Should be set to True by CompareWarp only.",
        dtype=bool,
        default=False,
    )
    maskPropagationThresholds = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        doc=("Threshold (in fractional weight) of rejection at which we propagate a mask plane to "
             "the coadd; that is, we set the mask bit on the coadd if the fraction the rejected frames "
             "would have contributed exceeds this value."),
        default={"SAT": 0.1},
    )
    removeMaskPlanes = pexConfig.ListField(dtype=str, default=["NOT_DEBLENDED"],
                                           doc="Mask planes to remove before coadding")
    doMaskBrightObjects = pexConfig.Field(dtype=bool, default=False,
                                          doc="Set mask and flag bits for bright objects?")
    brightObjectMaskName = pexConfig.Field(dtype=str, default="BRIGHT_OBJECT",
                                           doc="Name of mask bit used for bright objects")
    coaddPsf = pexConfig.ConfigField(
        doc="Configuration for CoaddPsf",
        dtype=measAlg.CoaddPsfConfig,
    )
    doAttachTransmissionCurve = pexConfig.Field(
        dtype=bool, default=False, optional=False,
        doc=("Attach a piecewise TransmissionCurve for the coadd? "
             "(requires all input Exposures to have TransmissionCurves).")
    )
    hasFakes = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Should be set to True if fake sources have been inserted into the input data."
    )
    doSelectVisits = pexConfig.Field(
        doc="Coadd only visits selected by a SelectVisitsTask",
        dtype=bool,
        default=False,
    )
    doInputMap = pexConfig.Field(
        doc="Create a bitwise map of coadd inputs",
        dtype=bool,
        default=False,
    )
    inputMapper = pexConfig.ConfigurableField(
        doc="Input map creation subtask.",
        target=HealSparseInputMapTask,
    )

    def setDefaults(self):
        super().setDefaults()
        self.badMaskPlanes = ["NO_DATA", "BAD", "SAT", "EDGE"]

    def validate(self):
        super().validate()
        if self.doPsfMatch:
            # Backwards compatibility.
            # Configs do not have loggers
            log.warning("Config doPsfMatch deprecated. Setting warpType='psfMatched'")
            self.warpType = 'psfMatched'
        if self.doSigmaClip and self.statistic != "MEANCLIP":
            log.warning('doSigmaClip deprecated. To replicate behavior, setting statistic to "MEANCLIP"')
            self.statistic = "MEANCLIP"
        if self.doInterp and self.statistic not in ['MEAN', 'MEDIAN', 'MEANCLIP', 'VARIANCE', 'VARIANCECLIP']:
            raise ValueError("Must set doInterp=False for statistic=%s, which does not "
                             "compute and set a non-zero coadd variance estimate." % (self.statistic))

        unstackableStats = ['NOTHING', 'ERROR', 'ORMASK']
        if not hasattr(afwMath.Property, self.statistic) or self.statistic in unstackableStats:
            stackableStats = [str(k) for k in afwMath.Property.__members__.keys()
                              if str(k) not in unstackableStats]
            raise ValueError("statistic %s is not allowed. Please choose one of %s."
                             % (self.statistic, stackableStats))


class AssembleCoaddTask(CoaddBaseTask, pipeBase.PipelineTask):
    """Assemble a coadded image from a set of warps.

    Each Warp that goes into a coadd will typically have an independent
    photometric zero-point. Therefore, we must scale each Warp to set it to
    a common photometric zeropoint. WarpType may be one of 'direct' or
    'psfMatched', and the boolean configs `config.makeDirect` and
    `config.makePsfMatched` set which of the warp types will be coadded.
    The coadd is computed as a mean with optional outlier rejection.
    Criteria for outlier rejection are set in `AssembleCoaddConfig`.
    Finally, Warps can have bad 'NaN' pixels which received no input from the
    source calExps. We interpolate over these bad (NaN) pixels.

    `AssembleCoaddTask` uses several sub-tasks. These are

    - `~lsst.pipe.tasks.ScaleZeroPointTask`
    - create and use an ``imageScaler`` object to scale the photometric zeropoint for each Warp
    - `~lsst.pipe.tasks.InterpImageTask`
    - interpolate across bad pixels (NaN) in the final coadd

    You can retarget these subtasks if you wish.

    Parameters
    ----------
    *args
        Additional positional arguments.
    **kwargs
        Additional keyword arguments.

    Raises
    ------
    RuntimeError
        Raised if unable to define mask plane for bright objects.

    Notes
    -----
    Debugging:
    `AssembleCoaddTask` has no debug variables of its own. Some of the
    subtasks may support `~lsst.base.lsstDebug` variables. See the
    documentation for the subtasks for further information.

    Examples
    --------
    `AssembleCoaddTask` assembles a set of warped images into a coadded image.
    The `AssembleCoaddTask` can be invoked by running ``assembleCoadd.py``
    with the flag '--legacyCoadd'. Usage of assembleCoadd.py expects two
    inputs: a data reference to the tract patch and filter to be coadded, and
    a list of Warps to attempt to coadd. These are specified using ``--id`` and
    ``--selectId``, respectively:

    .. code-block:: none

       --id = [KEY=VALUE1[^VALUE2[^VALUE3...] [KEY=VALUE1[^VALUE2[^VALUE3...] ...]]
       --selectId [KEY=VALUE1[^VALUE2[^VALUE3...] [KEY=VALUE1[^VALUE2[^VALUE3...] ...]]

    Only the Warps that cover the specified tract and patch will be coadded.
    A list of the available optional arguments can be obtained by calling
    ``assembleCoadd.py`` with the ``--help`` command line argument:

    .. code-block:: none

       assembleCoadd.py --help

    To demonstrate usage of the `AssembleCoaddTask` in the larger context of
    multi-band processing, we will generate the HSC-I & -R band coadds from
    HSC engineering test data provided in the ``ci_hsc`` package. To begin,
    assuming that the lsst stack has been already set up, we must set up the
    obs_subaru and ``ci_hsc`` packages. This defines the environment variable
    ``$CI_HSC_DIR`` and points at the location of the package. The raw HSC
    data live in the ``$CI_HSC_DIR/raw directory``. To begin assembling the
    coadds, we must first run:

    - processCcd
    - process the individual ccds in $CI_HSC_RAW to produce calibrated exposures
    - makeSkyMap
    - create a skymap that covers the area of the sky present in the raw exposures
    - makeCoaddTempExp
    - warp the individual calibrated exposures to the tangent plane of the coadd

    We can perform all of these steps by running

    .. code-block:: none

       $CI_HSC_DIR scons warp-903986 warp-904014 warp-903990 warp-904010 warp-903988

    This will produce warped exposures for each visit. To coadd the warped
    data, we call assembleCoadd.py as follows:

    .. code-block:: none

       assembleCoadd.py --legacyCoadd $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-I \
       --selectId visit=903986 ccd=16 --selectId visit=903986 ccd=22 --selectId visit=903986 ccd=23 \
       --selectId visit=903986 ccd=100 --selectId visit=904014 ccd=1 --selectId visit=904014 ccd=6 \
       --selectId visit=904014 ccd=12 --selectId visit=903990 ccd=18 --selectId visit=903990 ccd=25 \
       --selectId visit=904010 ccd=4 --selectId visit=904010 ccd=10 --selectId visit=904010 ccd=100 \
       --selectId visit=903988 ccd=16 --selectId visit=903988 ccd=17 --selectId visit=903988 ccd=23 \
       --selectId visit=903988 ccd=24

    that will process the HSC-I band data. The results are written in
    ``$CI_HSC_DIR/DATA/deepCoadd-results/HSC-I``.

    You may also choose to run:

    .. code-block:: none

       scons warp-903334 warp-903336 warp-903338 warp-903342 warp-903344 warp-903346
       assembleCoadd.py --legacyCoadd $CI_HSC_DIR/DATA --id patch=5,4 tract=0 filter=HSC-R \
       --selectId visit=903334 ccd=16 --selectId visit=903334 ccd=22 --selectId visit=903334 ccd=23 \
       --selectId visit=903334 ccd=100 --selectId visit=903336 ccd=17 --selectId visit=903336 ccd=24 \
       --selectId visit=903338 ccd=18 --selectId visit=903338 ccd=25 --selectId visit=903342 ccd=4 \
       --selectId visit=903342 ccd=10 --selectId visit=903342 ccd=100 --selectId visit=903344 ccd=0 \
       --selectId visit=903344 ccd=5 --selectId visit=903344 ccd=11 --selectId visit=903346 ccd=1 \
       --selectId visit=903346 ccd=6 --selectId visit=903346 ccd=12

    to generate the coadd for the HSC-R band if you are interested in
    following multiBand Coadd processing as discussed in `pipeTasks_multiBand`
    (but note that normally, one would use the `SafeClipAssembleCoaddTask`
    rather than `AssembleCoaddTask` to make the coadd.
    """

    ConfigClass = AssembleCoaddConfig
    _DefaultName = "assembleCoadd"

    def __init__(self, *args, **kwargs):
        # TODO: DM-17415 better way to handle previously allowed passed args e.g.`AssembleCoaddTask(config)`
        if args:
            argNames = ["config", "name", "parentTask", "log"]
            kwargs.update({k: v for k, v in zip(argNames, args)})
            warnings.warn("AssembleCoadd received positional args, and casting them as kwargs: %s. "
                          "PipelineTask will not take positional args" % argNames, FutureWarning)

        super().__init__(**kwargs)
        self.makeSubtask("interpImage")
        self.makeSubtask("scaleZeroPoint")

        if self.config.doMaskBrightObjects:
            mask = afwImage.Mask()
            try:
                self.brightObjectBitmask = 1 << mask.addMaskPlane(self.config.brightObjectMaskName)
            except pexExceptions.LsstCppException:
                raise RuntimeError("Unable to define mask plane for bright objects; planes used are %s" %
                                   mask.getMaskPlaneDict().keys())
            del mask

        if self.config.doInputMap:
            self.makeSubtask("inputMapper")

        self.warpType = self.config.warpType

    @utils.inheritDoc(pipeBase.PipelineTask)
    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputData = butlerQC.get(inputRefs)

        # Construct skyInfo expected by run
        # Do not remove skyMap from inputData in case _makeSupplementaryData needs it
        skyMap = inputData["skyMap"]
        outputDataId = butlerQC.quantum.dataId

        inputData['skyInfo'] = makeSkyInfo(skyMap,
                                           tractId=outputDataId['tract'],
                                           patchId=outputDataId['patch'])

        if self.config.doSelectVisits:
            warpRefList = self.filterWarps(inputData['inputWarps'], inputData['selectedVisits'])
        else:
            warpRefList = inputData['inputWarps']

        inputs = self.prepareInputs(warpRefList)
        self.log.info("Found %d %s", len(inputs.tempExpRefList),
                      self.getTempExpDatasetName(self.warpType))
        if len(inputs.tempExpRefList) == 0:
            raise pipeBase.NoWorkFound("No coadd temporary exposures found")

        supplementaryData = self._makeSupplementaryData(butlerQC, inputRefs, outputRefs)
        retStruct = self.run(inputData['skyInfo'], inputs.tempExpRefList, inputs.imageScalerList,
                             inputs.weightList, supplementaryData=supplementaryData)

        inputData.setdefault('brightObjectMask', None)
        self.processResults(retStruct.coaddExposure, inputData['brightObjectMask'], outputDataId)

        if self.config.doWrite:
            butlerQC.put(retStruct, outputRefs)
        return retStruct

    def processResults(self, coaddExposure, brightObjectMasks=None, dataId=None):
        """Interpolate over missing data and mask bright stars.

        Parameters
        ----------
        coaddExposure : `lsst.afw.image.Exposure`
            The coadded exposure to process.
        brightObjectMasks : `lsst.afw.table` or `None`, optional
            Table of bright objects to mask.
        dataId : `lsst.daf.butler.DataId` or `None`, optional
            Data identification.
        """
        if self.config.doInterp:
            self.interpImage.run(coaddExposure.getMaskedImage(), planeName="NO_DATA")
            # The variance must be positive; work around for DM-3201.
            varArray = coaddExposure.variance.array
            with numpy.errstate(invalid="ignore"):
                varArray[:] = numpy.where(varArray > 0, varArray, numpy.inf)

        if self.config.doMaskBrightObjects:
            self.setBrightObjectMasks(coaddExposure, brightObjectMasks, dataId)

    def _makeSupplementaryData(self, butlerQC, inputRefs, outputRefs):
        """Make additional inputs to run() specific to subclasses (Gen3).

        Duplicates interface of `runQuantum` method.
        Available to be implemented by subclasses only if they need the
        coadd dataRef for performing preliminary processing before
        assembling the coadd.

        Parameters
        ----------
        butlerQC : `~lsst.pipe.base.ButlerQuantumContext`
            Gen3 Butler object for fetching additional data products before
            running the Task specialized for quantum being processed.
        inputRefs : `~lsst.pipe.base.InputQuantizedConnection`
            Attributes are the names of the connections describing input dataset types.
            Values are DatasetRefs that task consumes for corresponding dataset type.
            DataIds are guaranteed to match data objects in ``inputData``.
        outputRefs : `~lsst.pipe.base.OutputQuantizedConnection`
            Attributes are the names of the connections describing output dataset types.
            Values are DatasetRefs that task is to produce
            for corresponding dataset type.
        """
        return pipeBase.Struct()

    @deprecated(
        reason="makeSupplementaryDataGen3 is deprecated in favor of _makeSupplementaryData",
        version="v25.0",
        category=FutureWarning
    )
    def makeSupplementaryDataGen3(self, butlerQC, inputRefs, outputRefs):
        return self._makeSupplementaryData(butlerQC, inputRefs, outputRefs)

    def prepareInputs(self, refList):
        """Prepare the input warps for coaddition by measuring the weight for
        each warp and the scaling for the photometric zero point.

        Each Warp has its own photometric zeropoint and background variance.
        Before coadding these Warps together, compute a scale factor to
        normalize the photometric zeropoint and compute the weight for each Warp.

        Parameters
        ----------
        refList : `list`
            List of data references to tempExp.

        Returns
        -------
        result : `~lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``tempExprefList``
                `list` of data references to tempExp.
            ``weightList``
                `list` of weightings.
            ``imageScalerList``
                `list` of image scalers.
        """
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(self.config.sigmaClip)
        statsCtrl.setNumIter(self.config.clipIter)
        statsCtrl.setAndMask(self.getBadPixelMask())
        statsCtrl.setNanSafe(True)
        # compute tempExpRefList: a list of tempExpRef that actually exist
        # and weightList: a list of the weight of the associated coadd tempExp
        # and imageScalerList: a list of scale factors for the associated coadd tempExp
        tempExpRefList = []
        weightList = []
        imageScalerList = []
        tempExpName = self.getTempExpDatasetName(self.warpType)
        for tempExpRef in refList:
            tempExp = tempExpRef.get()
            # Ignore any input warp that is empty of data
            if numpy.isnan(tempExp.image.array).all():
                continue
            maskedImage = tempExp.getMaskedImage()
            imageScaler = self.scaleZeroPoint.computeImageScaler(
                exposure=tempExp,
                dataRef=tempExpRef,  # FIXME
            )
            try:
                imageScaler.scaleMaskedImage(maskedImage)
            except Exception as e:
                self.log.warning("Scaling failed for %s (skipping it): %s", tempExpRef.dataId, e)
                continue
            statObj = afwMath.makeStatistics(maskedImage.getVariance(), maskedImage.getMask(),
                                             afwMath.MEANCLIP, statsCtrl)
            meanVar, meanVarErr = statObj.getResult(afwMath.MEANCLIP)
            weight = 1.0 / float(meanVar)
            if not numpy.isfinite(weight):
                self.log.warning("Non-finite weight for %s: skipping", tempExpRef.dataId)
                continue
            self.log.info("Weight of %s %s = %0.3f", tempExpName, tempExpRef.dataId, weight)

            del maskedImage
            del tempExp

            tempExpRefList.append(tempExpRef)
            weightList.append(weight)
            imageScalerList.append(imageScaler)

        return pipeBase.Struct(tempExpRefList=tempExpRefList, weightList=weightList,
                               imageScalerList=imageScalerList)

    def prepareStats(self, mask=None):
        """Prepare the statistics for coadding images.

        Parameters
        ----------
        mask : `int`, optional
            Bit mask value to exclude from coaddition.

        Returns
        -------
        stats : `~lsst.pipe.base.Struct`
            Statistics as a struct with attributes:

            ``statsCtrl``
                Statistics control object for coadd (`~lsst.afw.math.StatisticsControl`).
            ``statsFlags``
                Statistic for coadd (`~lsst.afw.math.Property`).
        """
        if mask is None:
            mask = self.getBadPixelMask()
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(self.config.sigmaClip)
        statsCtrl.setNumIter(self.config.clipIter)
        statsCtrl.setAndMask(mask)
        statsCtrl.setNanSafe(True)
        statsCtrl.setWeighted(True)
        statsCtrl.setCalcErrorFromInputVariance(self.config.calcErrorFromInputVariance)
        for plane, threshold in self.config.maskPropagationThresholds.items():
            bit = afwImage.Mask.getMaskPlane(plane)
            statsCtrl.setMaskPropagationThreshold(bit, threshold)
        statsFlags = afwMath.stringToStatisticsProperty(self.config.statistic)
        return pipeBase.Struct(ctrl=statsCtrl, flags=statsFlags)

    @timeMethod
    def run(self, skyInfo, tempExpRefList, imageScalerList, weightList,
            altMaskList=None, mask=None, supplementaryData=None):
        """Assemble a coadd from input warps.

        Assemble the coadd using the provided list of coaddTempExps. Since
        the full coadd covers a patch (a large area), the assembly is
        performed over small areas on the image at a time in order to
        conserve memory usage. Iterate over subregions within the outer
        bbox of the patch using `assembleSubregion` to stack the corresponding
        subregions from the coaddTempExps with the statistic specified.
        Set the edge bits the coadd mask based on the weight map.

        Parameters
        ----------
        skyInfo : `~lsst.pipe.base.Struct`
            Struct with geometric information about the patch.
        tempExpRefList : `list`
            List of data references to Warps (previously called CoaddTempExps).
        imageScalerList : `list`
            List of image scalers.
        weightList : `list`
            List of weights.
        altMaskList : `list`, optional
            List of alternate masks to use rather than those stored with
            tempExp.
        mask : `int`, optional
            Bit mask value to exclude from coaddition.
        supplementaryData : `~lsst.pipe.base.Struct`, optional
            Struct with additional data products needed to assemble coadd.
            Only used by subclasses that implement ``_makeSupplementaryData``
            and override `run`.

        Returns
        -------
        result : `~lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``coaddExposure``
                Coadded exposure (``lsst.afw.image.Exposure``).
            ``nImage``
                Exposure count image (``lsst.afw.image.Image``), if requested.
            ``inputMap``
                Bit-wise map of inputs, if requested.
            ``warpRefList``
                Input list of refs to the warps (``lsst.daf.butler.DeferredDatasetHandle``)
                (unmodified).
            ``imageScalerList``
                Input list of image scalers (`list`) (unmodified).
            ``weightList``
                Input list of weights (`list`) (unmodified).
        """
        tempExpName = self.getTempExpDatasetName(self.warpType)
        self.log.info("Assembling %s %s", len(tempExpRefList), tempExpName)
        stats = self.prepareStats(mask=mask)

        if altMaskList is None:
            altMaskList = [None]*len(tempExpRefList)

        coaddExposure = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
        coaddExposure.setPhotoCalib(self.scaleZeroPoint.getPhotoCalib())
        coaddExposure.getInfo().setCoaddInputs(self.inputRecorder.makeCoaddInputs())
        self.assembleMetadata(coaddExposure, tempExpRefList, weightList)
        coaddMaskedImage = coaddExposure.getMaskedImage()
        subregionSizeArr = self.config.subregionSize
        subregionSize = geom.Extent2I(subregionSizeArr[0], subregionSizeArr[1])
        # if nImage is requested, create a zero one which can be passed to assembleSubregion
        if self.config.doNImage:
            nImage = afwImage.ImageU(skyInfo.bbox)
        else:
            nImage = None
        # If inputMap is requested, create the initial version that can be masked in
        # assembleSubregion.
        if self.config.doInputMap:
            self.inputMapper.build_ccd_input_map(skyInfo.bbox,
                                                 skyInfo.wcs,
                                                 coaddExposure.getInfo().getCoaddInputs().ccds)

        if self.config.doOnlineForMean and self.config.statistic == "MEAN":
            try:
                self.assembleOnlineMeanCoadd(coaddExposure, tempExpRefList, imageScalerList,
                                             weightList, altMaskList, stats.ctrl,
                                             nImage=nImage)
            except Exception as e:
                self.log.exception("Cannot compute online coadd %s", e)
                raise
        else:
            for subBBox in self._subBBoxIter(skyInfo.bbox, subregionSize):
                try:
                    self.assembleSubregion(coaddExposure, subBBox, tempExpRefList, imageScalerList,
                                           weightList, altMaskList, stats.flags, stats.ctrl,
                                           nImage=nImage)
                except Exception as e:
                    self.log.exception("Cannot compute coadd %s: %s", subBBox, e)
                    raise

        # If inputMap is requested, we must finalize the map after the accumulation.
        if self.config.doInputMap:
            self.inputMapper.finalize_ccd_input_map_mask()
            inputMap = self.inputMapper.ccd_input_map
        else:
            inputMap = None

        self.setInexactPsf(coaddMaskedImage.getMask())
        # Despite the name, the following doesn't really deal with "EDGE" pixels: it identifies
        # pixels that didn't receive any unmasked inputs (as occurs around the edge of the field).
        coaddUtils.setCoaddEdgeBits(coaddMaskedImage.getMask(), coaddMaskedImage.getVariance())
        return pipeBase.Struct(coaddExposure=coaddExposure, nImage=nImage,
                               warpRefList=tempExpRefList, imageScalerList=imageScalerList,
                               weightList=weightList, inputMap=inputMap)

    def assembleMetadata(self, coaddExposure, tempExpRefList, weightList):
        """Set the metadata for the coadd.

        This basic implementation sets the filter from the first input.

        Parameters
        ----------
        coaddExposure : `lsst.afw.image.Exposure`
            The target exposure for the coadd.
        tempExpRefList : `list`
            List of data references to tempExp.
        weightList : `list`
            List of weights.

        Raises
        ------
        AssertionError
            Raised if there is a length mismatch.
        """
        assert len(tempExpRefList) == len(weightList), "Length mismatch"

        # We load a single pixel of each coaddTempExp, because we just want to get at the metadata
        # (and we need more than just the PropertySet that contains the header), which is not possible
        # with the current butler (see #2777).
        bbox = geom.Box2I(coaddExposure.getBBox().getMin(), geom.Extent2I(1, 1))

        tempExpList = [tempExpRef.get(parameters={'bbox': bbox}) for tempExpRef in tempExpRefList]

        numCcds = sum(len(tempExp.getInfo().getCoaddInputs().ccds) for tempExp in tempExpList)

        # Set the coadd FilterLabel to the band of the first input exposure:
        # Coadds are calibrated, so the physical label is now meaningless.
        coaddExposure.setFilter(afwImage.FilterLabel(tempExpList[0].getFilter().bandLabel))
        coaddInputs = coaddExposure.getInfo().getCoaddInputs()
        coaddInputs.ccds.reserve(numCcds)
        coaddInputs.visits.reserve(len(tempExpList))

        for tempExp, weight in zip(tempExpList, weightList):
            self.inputRecorder.addVisitToCoadd(coaddInputs, tempExp, weight)

        if self.config.doUsePsfMatchedPolygons:
            self.shrinkValidPolygons(coaddInputs)

        coaddInputs.visits.sort()
        coaddInputs.ccds.sort()
        if self.warpType == "psfMatched":
            # The modelPsf BBox for a psfMatchedWarp/coaddTempExp was dynamically defined by
            # ModelPsfMatchTask as the square box bounding its spatially-variable, pre-matched WarpedPsf.
            # Likewise, set the PSF of a PSF-Matched Coadd to the modelPsf
            # having the maximum width (sufficient because square)
            modelPsfList = [tempExp.getPsf() for tempExp in tempExpList]
            modelPsfWidthList = [modelPsf.computeBBox(modelPsf.getAveragePosition()).getWidth()
                                 for modelPsf in modelPsfList]
            psf = modelPsfList[modelPsfWidthList.index(max(modelPsfWidthList))]
        else:
            psf = measAlg.CoaddPsf(coaddInputs.ccds, coaddExposure.getWcs(),
                                   self.config.coaddPsf.makeControl())
        coaddExposure.setPsf(psf)
        apCorrMap = measAlg.makeCoaddApCorrMap(coaddInputs.ccds, coaddExposure.getBBox(afwImage.PARENT),
                                               coaddExposure.getWcs())
        coaddExposure.getInfo().setApCorrMap(apCorrMap)
        if self.config.doAttachTransmissionCurve:
            transmissionCurve = measAlg.makeCoaddTransmissionCurve(coaddExposure.getWcs(), coaddInputs.ccds)
            coaddExposure.getInfo().setTransmissionCurve(transmissionCurve)

    def assembleSubregion(self, coaddExposure, bbox, tempExpRefList, imageScalerList, weightList,
                          altMaskList, statsFlags, statsCtrl, nImage=None):
        """Assemble the coadd for a sub-region.

        For each coaddTempExp, check for (and swap in) an alternative mask
        if one is passed. Remove mask planes listed in
        `config.removeMaskPlanes`. Finally, stack the actual exposures using
        `lsst.afw.math.statisticsStack` with the statistic specified by
        statsFlags. Typically, the statsFlag will be one of lsst.afw.math.MEAN for
        a mean-stack or `lsst.afw.math.MEANCLIP` for outlier rejection using
        an N-sigma clipped mean where N and iterations are specified by
        statsCtrl.  Assign the stacked subregion back to the coadd.

        Parameters
        ----------
        coaddExposure : `lsst.afw.image.Exposure`
            The target exposure for the coadd.
        bbox : `lsst.geom.Box`
            Sub-region to coadd.
        tempExpRefList : `list`
            List of data reference to tempExp.
        imageScalerList : `list`
            List of image scalers.
        weightList : `list`
            List of weights.
        altMaskList : `list`
            List of alternate masks to use rather than those stored with
            tempExp, or None.  Each element is dict with keys = mask plane
            name to which to add the spans.
        statsFlags : `lsst.afw.math.Property`
            Property object for statistic for coadd.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd.
        nImage : `lsst.afw.image.ImageU`, optional
            Keeps track of exposure count for each pixel.
        """
        self.log.debug("Computing coadd over %s", bbox)

        coaddExposure.mask.addMaskPlane("REJECTED")
        coaddExposure.mask.addMaskPlane("CLIPPED")
        coaddExposure.mask.addMaskPlane("SENSOR_EDGE")
        maskMap = self.setRejectedMaskMapping(statsCtrl)
        clipped = afwImage.Mask.getPlaneBitMask("CLIPPED")
        maskedImageList = []
        if nImage is not None:
            subNImage = afwImage.ImageU(bbox.getWidth(), bbox.getHeight())
        for tempExpRef, imageScaler, altMask in zip(tempExpRefList, imageScalerList, altMaskList):

            exposure = tempExpRef.get(parameters={'bbox': bbox})

            maskedImage = exposure.getMaskedImage()
            mask = maskedImage.getMask()
            if altMask is not None:
                self.applyAltMaskPlanes(mask, altMask)
            imageScaler.scaleMaskedImage(maskedImage)

            # Add 1 for each pixel which is not excluded by the exclude mask.
            # In legacyCoadd, pixels may also be excluded by afwMath.statisticsStack.
            if nImage is not None:
                subNImage.getArray()[maskedImage.getMask().getArray() & statsCtrl.getAndMask() == 0] += 1
            if self.config.removeMaskPlanes:
                self.removeMaskPlanes(maskedImage)
            maskedImageList.append(maskedImage)

            if self.config.doInputMap:
                visit = exposure.getInfo().getCoaddInputs().visits[0].getId()
                self.inputMapper.mask_warp_bbox(bbox, visit, mask, statsCtrl.getAndMask())

        with self.timer("stack"):
            coaddSubregion = afwMath.statisticsStack(maskedImageList, statsFlags, statsCtrl, weightList,
                                                     clipped,  # also set output to CLIPPED if sigma-clipped
                                                     maskMap)
        coaddExposure.maskedImage.assign(coaddSubregion, bbox)
        if nImage is not None:
            nImage.assign(subNImage, bbox)

    def assembleOnlineMeanCoadd(self, coaddExposure, tempExpRefList, imageScalerList, weightList,
                                altMaskList, statsCtrl, nImage=None):
        """Assemble the coadd using the "online" method.

        This method takes a running sum of images and weights to save memory.
        It only works for MEAN statistics.

        Parameters
        ----------
        coaddExposure : `lsst.afw.image.Exposure`
            The target exposure for the coadd.
        tempExpRefList : `list`
            List of data reference to tempExp.
        imageScalerList : `list`
            List of image scalers.
        weightList : `list`
            List of weights.
        altMaskList : `list`
            List of alternate masks to use rather than those stored with
            tempExp, or None.  Each element is dict with keys = mask plane
            name to which to add the spans.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd.
        nImage : `lsst.afw.image.ImageU`, optional
            Keeps track of exposure count for each pixel.
        """
        self.log.debug("Computing online coadd.")

        coaddExposure.mask.addMaskPlane("REJECTED")
        coaddExposure.mask.addMaskPlane("CLIPPED")
        coaddExposure.mask.addMaskPlane("SENSOR_EDGE")
        maskMap = self.setRejectedMaskMapping(statsCtrl)
        thresholdDict = AccumulatorMeanStack.stats_ctrl_to_threshold_dict(statsCtrl)

        bbox = coaddExposure.maskedImage.getBBox()

        stacker = AccumulatorMeanStack(
            coaddExposure.image.array.shape,
            statsCtrl.getAndMask(),
            mask_threshold_dict=thresholdDict,
            mask_map=maskMap,
            no_good_pixels_mask=statsCtrl.getNoGoodPixelsMask(),
            calc_error_from_input_variance=self.config.calcErrorFromInputVariance,
            compute_n_image=(nImage is not None)
        )

        for tempExpRef, imageScaler, altMask, weight in zip(tempExpRefList,
                                                            imageScalerList,
                                                            altMaskList,
                                                            weightList):
            exposure = tempExpRef.get()
            maskedImage = exposure.getMaskedImage()
            mask = maskedImage.getMask()
            if altMask is not None:
                self.applyAltMaskPlanes(mask, altMask)
            imageScaler.scaleMaskedImage(maskedImage)
            if self.config.removeMaskPlanes:
                self.removeMaskPlanes(maskedImage)

            stacker.add_masked_image(maskedImage, weight=weight)

            if self.config.doInputMap:
                visit = exposure.getInfo().getCoaddInputs().visits[0].getId()
                self.inputMapper.mask_warp_bbox(bbox, visit, mask, statsCtrl.getAndMask())

        stacker.fill_stacked_masked_image(coaddExposure.maskedImage)

        if nImage is not None:
            nImage.array[:, :] = stacker.n_image

    def removeMaskPlanes(self, maskedImage):
        """Unset the mask of an image for mask planes specified in the config.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            The masked image to be modified.

        Raises
        ------
        InvalidParameterError
            Raised if no mask plane with that name was found.
        """
        mask = maskedImage.getMask()
        for maskPlane in self.config.removeMaskPlanes:
            try:
                mask &= ~mask.getPlaneBitMask(maskPlane)
            except pexExceptions.InvalidParameterError:
                self.log.debug("Unable to remove mask plane %s: no mask plane with that name was found.",
                               maskPlane)

    @staticmethod
    def setRejectedMaskMapping(statsCtrl):
        """Map certain mask planes of the warps to new planes for the coadd.

        If a pixel is rejected due to a mask value other than EDGE, NO_DATA,
        or CLIPPED, set it to REJECTED on the coadd.
        If a pixel is rejected due to EDGE, set the coadd pixel to SENSOR_EDGE.
        If a pixel is rejected due to CLIPPED, set the coadd pixel to CLIPPED.

        Parameters
        ----------
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd.

        Returns
        -------
        maskMap : `list` of `tuple` of `int`
            A list of mappings of mask planes of the warped exposures to
            mask planes of the coadd.
        """
        edge = afwImage.Mask.getPlaneBitMask("EDGE")
        noData = afwImage.Mask.getPlaneBitMask("NO_DATA")
        clipped = afwImage.Mask.getPlaneBitMask("CLIPPED")
        toReject = statsCtrl.getAndMask() & (~noData) & (~edge) & (~clipped)
        maskMap = [(toReject, afwImage.Mask.getPlaneBitMask("REJECTED")),
                   (edge, afwImage.Mask.getPlaneBitMask("SENSOR_EDGE")),
                   (clipped, clipped)]
        return maskMap

    def applyAltMaskPlanes(self, mask, altMaskSpans):
        """Apply in place alt mask formatted as SpanSets to a mask.

        Parameters
        ----------
        mask : `lsst.afw.image.Mask`
            Original mask.
        altMaskSpans : `dict`
            SpanSet lists to apply. Each element contains the new mask
            plane name (e.g. "CLIPPED and/or "NO_DATA") as the key,
            and list of SpanSets to apply to the mask.

        Returns
        -------
        mask : `lsst.afw.image.Mask`
            Updated mask.
        """
        if self.config.doUsePsfMatchedPolygons:
            if ("NO_DATA" in altMaskSpans) and ("NO_DATA" in self.config.badMaskPlanes):
                # Clear away any other masks outside the validPolygons. These pixels are no longer
                # contributing to inexact PSFs, and will still be rejected because of NO_DATA
                # self.config.doUsePsfMatchedPolygons should be True only in CompareWarpAssemble
                # This mask-clearing step must only occur *before* applying the new masks below
                for spanSet in altMaskSpans['NO_DATA']:
                    spanSet.clippedTo(mask.getBBox()).clearMask(mask, self.getBadPixelMask())

        for plane, spanSetList in altMaskSpans.items():
            maskClipValue = mask.addMaskPlane(plane)
            for spanSet in spanSetList:
                spanSet.clippedTo(mask.getBBox()).setMask(mask, 2**maskClipValue)
        return mask

    def shrinkValidPolygons(self, coaddInputs):
        """Shrink coaddInputs' ccds' ValidPolygons in place.

        Either modify each ccd's validPolygon in place, or if CoaddInputs
        does not have a validPolygon, create one from its bbox.

        Parameters
        ----------
        coaddInputs : `lsst.afw.image.coaddInputs`
            Original mask.
        """
        for ccd in coaddInputs.ccds:
            polyOrig = ccd.getValidPolygon()
            validPolyBBox = polyOrig.getBBox() if polyOrig else ccd.getBBox()
            validPolyBBox.grow(-self.config.matchingKernelSize//2)
            if polyOrig:
                validPolygon = polyOrig.intersectionSingle(validPolyBBox)
            else:
                validPolygon = afwGeom.polygon.Polygon(geom.Box2D(validPolyBBox))
            ccd.setValidPolygon(validPolygon)

    def setBrightObjectMasks(self, exposure, brightObjectMasks, dataId=None):
        """Set the bright object masks.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure under consideration.
        brightObjectMasks : `lsst.afw.table`
            Table of bright objects to mask.
        dataId : `lsst.daf.butler.DataId`, optional
            Data identifier dict for patch.
        """
        if brightObjectMasks is None:
            self.log.warning("Unable to apply bright object mask: none supplied")
            return
        self.log.info("Applying %d bright object masks to %s", len(brightObjectMasks), dataId)
        mask = exposure.getMaskedImage().getMask()
        wcs = exposure.getWcs()
        plateScale = wcs.getPixelScale().asArcseconds()

        for rec in brightObjectMasks:
            center = geom.PointI(wcs.skyToPixel(rec.getCoord()))
            if rec["type"] == "box":
                assert rec["angle"] == 0.0, ("Angle != 0 for mask object %s" % rec["id"])
                width = rec["width"].asArcseconds()/plateScale    # convert to pixels
                height = rec["height"].asArcseconds()/plateScale  # convert to pixels

                halfSize = geom.ExtentI(0.5*width, 0.5*height)
                bbox = geom.Box2I(center - halfSize, center + halfSize)

                bbox = geom.BoxI(geom.PointI(int(center[0] - 0.5*width), int(center[1] - 0.5*height)),
                                 geom.PointI(int(center[0] + 0.5*width), int(center[1] + 0.5*height)))
                spans = afwGeom.SpanSet(bbox)
            elif rec["type"] == "circle":
                radius = int(rec["radius"].asArcseconds()/plateScale)   # convert to pixels
                spans = afwGeom.SpanSet.fromShape(radius, offset=center)
            else:
                self.log.warning("Unexpected region type %s at %s", rec["type"], center)
                continue
            spans.clippedTo(mask.getBBox()).setMask(mask, self.brightObjectBitmask)

    def setInexactPsf(self, mask):
        """Set INEXACT_PSF mask plane.

        If any of the input images isn't represented in the coadd (due to
        clipped pixels or chip gaps), the `CoaddPsf` will be inexact. Flag
        these pixels.

        Parameters
        ----------
        mask : `lsst.afw.image.Mask`
            Coadded exposure's mask, modified in-place.
        """
        mask.addMaskPlane("INEXACT_PSF")
        inexactPsf = mask.getPlaneBitMask("INEXACT_PSF")
        sensorEdge = mask.getPlaneBitMask("SENSOR_EDGE")  # chip edges (so PSF is discontinuous)
        clipped = mask.getPlaneBitMask("CLIPPED")  # pixels clipped from coadd
        rejected = mask.getPlaneBitMask("REJECTED")  # pixels rejected from coadd due to masks
        array = mask.getArray()
        selected = array & (sensorEdge | clipped | rejected) > 0
        array[selected] |= inexactPsf

    @staticmethod
    def _subBBoxIter(bbox, subregionSize):
        """Iterate over subregions of a bbox.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box over which to iterate.
        subregionSize : `lsst.geom.Extent2I`
            Size of sub-bboxes.

        Yields
        ------
        subBBox : `lsst.geom.Box2I`
            Next sub-bounding box of size ``subregionSize`` or smaller; each ``subBBox``
            is contained within ``bbox``, so it may be smaller than ``subregionSize`` at
            the edges of ``bbox``, but it will never be empty.

        Raises
        ------
        RuntimeError
            Raised if any of the following occur:
            - The given bbox is empty.
            - The subregionSize is 0.
        """
        if bbox.isEmpty():
            raise RuntimeError("bbox %s is empty" % (bbox,))
        if subregionSize[0] < 1 or subregionSize[1] < 1:
            raise RuntimeError("subregionSize %s must be nonzero" % (subregionSize,))

        for rowShift in range(0, bbox.getHeight(), subregionSize[1]):
            for colShift in range(0, bbox.getWidth(), subregionSize[0]):
                subBBox = geom.Box2I(bbox.getMin() + geom.Extent2I(colShift, rowShift), subregionSize)
                subBBox.clip(bbox)
                if subBBox.isEmpty():
                    raise RuntimeError("Bug: empty bbox! bbox=%s, subregionSize=%s, "
                                       "colShift=%s, rowShift=%s" %
                                       (bbox, subregionSize, colShift, rowShift))
                yield subBBox

    def filterWarps(self, inputs, goodVisits):
        """Return list of only inputRefs with visitId in goodVisits ordered by goodVisit.

        Parameters
        ----------
        inputs : `list` of `~lsst.pipe.base.connections.DeferredDatasetRef`
            List of `lsst.pipe.base.connections.DeferredDatasetRef` with dataId containing visit.
        goodVisit : `dict`
            Dictionary with good visitIds as the keys. Value ignored.

        Returns
        -------
        filteredInputs : `list` of `~lsst.pipe.base.connections.DeferredDatasetRef`
            Filtered and sorted list of inputRefs with visitId in goodVisits ordered by goodVisit.
        """
        inputWarpDict = {inputRef.ref.dataId['visit']: inputRef for inputRef in inputs}
        filteredInputs = []
        for visit in goodVisits.keys():
            if visit in inputWarpDict:
                filteredInputs.append(inputWarpDict[visit])
        return filteredInputs


def countMaskFromFootprint(mask, footprint, bitmask, ignoreMask):
    """Function to count the number of pixels with a specific mask in a
    footprint.

    Find the intersection of mask & footprint. Count all pixels in the mask
    that are in the intersection that have bitmask set but do not have
    ignoreMask set. Return the count.

    Parameters
    ----------
    mask : `lsst.afw.image.Mask`
        Mask to define intersection region by.
    footprint : `lsst.afw.detection.Footprint`
        Footprint to define the intersection region by.
    bitmask : `Unknown`
        Specific mask that we wish to count the number of occurances of.
    ignoreMask : `Unknown`
        Pixels to not consider.

    Returns
    -------
    result : `int`
        Number of pixels in footprint with specified mask.
    """
    bbox = footprint.getBBox()
    bbox.clip(mask.getBBox(afwImage.PARENT))
    fp = afwImage.Mask(bbox)
    subMask = mask.Factory(mask, bbox, afwImage.PARENT)
    footprint.spans.setMask(fp, bitmask)
    return numpy.logical_and((subMask.getArray() & fp.getArray()) > 0,
                             (subMask.getArray() & ignoreMask) == 0).sum()


class CompareWarpAssembleCoaddConnections(AssembleCoaddConnections):
    psfMatchedWarps = pipeBase.connectionTypes.Input(
        doc=("PSF-Matched Warps are required by CompareWarp regardless of the coadd type requested. "
             "Only PSF-Matched Warps make sense for image subtraction. "
             "Therefore, they must be an additional declared input."),
        name="{inputCoaddName}Coadd_psfMatchedWarp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit"),
        deferLoad=True,
        multiple=True
    )
    templateCoadd = pipeBase.connectionTypes.Output(
        doc=("Model of the static sky, used to find temporal artifacts. Typically a PSF-Matched, "
             "sigma-clipped coadd. Written if and only if assembleStaticSkyModel.doWrite=True"),
        name="{outputCoaddName}CoaddPsfMatched",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.assembleStaticSkyModel.doWrite:
            self.outputs.remove("templateCoadd")
        config.validate()


class CompareWarpAssembleCoaddConfig(AssembleCoaddConfig,
                                     pipelineConnections=CompareWarpAssembleCoaddConnections):
    assembleStaticSkyModel = pexConfig.ConfigurableField(
        target=AssembleCoaddTask,
        doc="Task to assemble an artifact-free, PSF-matched Coadd to serve as a"
            " naive/first-iteration model of the static sky.",
    )
    detect = pexConfig.ConfigurableField(
        target=SourceDetectionTask,
        doc="Detect outlier sources on difference between each psfMatched warp and static sky model"
    )
    detectTemplate = pexConfig.ConfigurableField(
        target=SourceDetectionTask,
        doc="Detect sources on static sky model. Only used if doPreserveContainedBySource is True"
    )
    maskStreaks = pexConfig.ConfigurableField(
        target=MaskStreaksTask,
        doc="Detect streaks on difference between each psfMatched warp and static sky model. Only used if "
            "doFilterMorphological is True. Adds a mask plane to an exposure, with the mask plane name set by"
            "streakMaskName"
    )
    streakMaskName = pexConfig.Field(
        dtype=str,
        default="STREAK",
        doc="Name of mask bit used for streaks"
    )
    maxNumEpochs = pexConfig.Field(
        doc="Charactistic maximum local number of epochs/visits in which an artifact candidate can appear  "
            "and still be masked.  The effective maxNumEpochs is a broken linear function of local "
            "number of epochs (N): min(maxFractionEpochsLow*N, maxNumEpochs + maxFractionEpochsHigh*N). "
            "For each footprint detected on the image difference between the psfMatched warp and static sky "
            "model, if a significant fraction of pixels (defined by spatialThreshold) are residuals in more "
            "than the computed effective maxNumEpochs, the artifact candidate is deemed persistant rather "
            "than transient and not masked.",
        dtype=int,
        default=2
    )
    maxFractionEpochsLow = pexConfig.RangeField(
        doc="Fraction of local number of epochs (N) to use as effective maxNumEpochs for low N. "
            "Effective maxNumEpochs = "
            "min(maxFractionEpochsLow * N, maxNumEpochs + maxFractionEpochsHigh * N)",
        dtype=float,
        default=0.4,
        min=0., max=1.,
    )
    maxFractionEpochsHigh = pexConfig.RangeField(
        doc="Fraction of local number of epochs (N) to use as effective maxNumEpochs for high N. "
            "Effective maxNumEpochs = "
            "min(maxFractionEpochsLow * N, maxNumEpochs + maxFractionEpochsHigh * N)",
        dtype=float,
        default=0.03,
        min=0., max=1.,
    )
    spatialThreshold = pexConfig.RangeField(
        doc="Unitless fraction of pixels defining how much of the outlier region has to meet the "
            "temporal criteria. If 0, clip all. If 1, clip none.",
        dtype=float,
        default=0.5,
        min=0., max=1.,
        inclusiveMin=True, inclusiveMax=True
    )
    doScaleWarpVariance = pexConfig.Field(
        doc="Rescale Warp variance plane using empirical noise?",
        dtype=bool,
        default=True,
    )
    scaleWarpVariance = pexConfig.ConfigurableField(
        target=ScaleVarianceTask,
        doc="Rescale variance on warps",
    )
    doPreserveContainedBySource = pexConfig.Field(
        doc="Rescue artifacts from clipping that completely lie within a footprint detected"
            "on the PsfMatched Template Coadd. Replicates a behavior of SafeClip.",
        dtype=bool,
        default=True,
    )
    doPrefilterArtifacts = pexConfig.Field(
        doc="Ignore artifact candidates that are mostly covered by the bad pixel mask, "
            "because they will be excluded anyway. This prevents them from contributing "
            "to the outlier epoch count image and potentially being labeled as persistant."
            "'Mostly' is defined by the config 'prefilterArtifactsRatio'.",
        dtype=bool,
        default=True
    )
    prefilterArtifactsMaskPlanes = pexConfig.ListField(
        doc="Prefilter artifact candidates that are mostly covered by these bad mask planes.",
        dtype=str,
        default=('NO_DATA', 'BAD', 'SAT', 'SUSPECT'),
    )
    prefilterArtifactsRatio = pexConfig.Field(
        doc="Prefilter artifact candidates with less than this fraction overlapping good pixels",
        dtype=float,
        default=0.05
    )
    doFilterMorphological = pexConfig.Field(
        doc="Filter artifact candidates based on morphological criteria, i.g. those that appear to "
            "be streaks.",
        dtype=bool,
        default=False
    )
    growStreakFp = pexConfig.Field(
        doc="Grow streak footprints by this number multiplied by the PSF width",
        dtype=float,
        default=5
    )

    def setDefaults(self):
        AssembleCoaddConfig.setDefaults(self)
        self.statistic = 'MEAN'
        self.doUsePsfMatchedPolygons = True

        # Real EDGE removed by psfMatched NO_DATA border half the width of the matching kernel
        # CompareWarp applies psfMatched EDGE pixels to directWarps before assembling
        if "EDGE" in self.badMaskPlanes:
            self.badMaskPlanes.remove('EDGE')
        self.removeMaskPlanes.append('EDGE')
        self.assembleStaticSkyModel.badMaskPlanes = ["NO_DATA", ]
        self.assembleStaticSkyModel.warpType = 'psfMatched'
        self.assembleStaticSkyModel.connections.warpType = 'psfMatched'
        self.assembleStaticSkyModel.statistic = 'MEANCLIP'
        self.assembleStaticSkyModel.sigmaClip = 2.5
        self.assembleStaticSkyModel.clipIter = 3
        self.assembleStaticSkyModel.calcErrorFromInputVariance = False
        self.assembleStaticSkyModel.doWrite = False
        self.detect.doTempLocalBackground = False
        self.detect.reEstimateBackground = False
        self.detect.returnOriginalFootprints = False
        self.detect.thresholdPolarity = "both"
        self.detect.thresholdValue = 5
        self.detect.minPixels = 4
        self.detect.isotropicGrow = True
        self.detect.thresholdType = "pixel_stdev"
        self.detect.nSigmaToGrow = 0.4
        # The default nSigmaToGrow for SourceDetectionTask is already 2.4,
        # Explicitly restating because ratio with detect.nSigmaToGrow matters
        self.detectTemplate.nSigmaToGrow = 2.4
        self.detectTemplate.doTempLocalBackground = False
        self.detectTemplate.reEstimateBackground = False
        self.detectTemplate.returnOriginalFootprints = False

    def validate(self):
        super().validate()
        if self.assembleStaticSkyModel.doNImage:
            raise ValueError("No dataset type exists for a PSF-Matched Template N Image."
                             "Please set assembleStaticSkyModel.doNImage=False")

        if self.assembleStaticSkyModel.doWrite and (self.warpType == self.assembleStaticSkyModel.warpType):
            raise ValueError("warpType (%s) == assembleStaticSkyModel.warpType (%s) and will compete for "
                             "the same dataset name. Please set assembleStaticSkyModel.doWrite to False "
                             "or warpType to 'direct'. assembleStaticSkyModel.warpType should ways be "
                             "'PsfMatched'" % (self.warpType, self.assembleStaticSkyModel.warpType))


class CompareWarpAssembleCoaddTask(AssembleCoaddTask):
    """Assemble a compareWarp coadded image from a set of warps
    by masking artifacts detected by comparing PSF-matched warps.

    In ``AssembleCoaddTask``, we compute the coadd as an clipped mean (i.e.,
    we clip outliers). The problem with doing this is that when computing the
    coadd PSF at a given location, individual visit PSFs from visits with
    outlier pixels contribute to the coadd PSF and cannot be treated correctly.
    In this task, we correct for this behavior by creating a new badMaskPlane
    'CLIPPED' which marks pixels in the individual warps suspected to contain
    an artifact. We populate this plane on the input warps by comparing
    PSF-matched warps with a PSF-matched median coadd which serves as a
    model of the static sky. Any group of pixels that deviates from the
    PSF-matched template coadd by more than config.detect.threshold sigma,
    is an artifact candidate. The candidates are then filtered to remove
    variable sources and sources that are difficult to subtract such as
    bright stars. This filter is configured using the config parameters
    ``temporalThreshold`` and ``spatialThreshold``. The temporalThreshold is
    the maximum fraction of epochs that the deviation can appear in and still
    be considered an artifact. The spatialThreshold is the maximum fraction of
    pixels in the footprint of the deviation that appear in other epochs
    (where other epochs is defined by the temporalThreshold). If the deviant
    region meets this criteria of having a significant percentage of pixels
    that deviate in only a few epochs, these pixels have the 'CLIPPED' bit
    set in the mask. These regions will not contribute to the final coadd.
    Furthermore, any routine to determine the coadd PSF can now be cognizant
    of clipped regions. Note that the algorithm implemented by this task is
    preliminary and works correctly for HSC data. Parameter modifications and
    or considerable redesigning of the algorithm is likley required for other
    surveys.

    ``CompareWarpAssembleCoaddTask`` sub-classes
    ``AssembleCoaddTask`` and instantiates ``AssembleCoaddTask``
    as a subtask to generate the TemplateCoadd (the model of the static sky).

    Notes
    -----
    Debugging:
    This task supports the following debug variables:
    - ``saveCountIm``
        If True then save the Epoch Count Image as a fits file in the `figPath`
    - ``figPath``
        Path to save the debug fits images and figures
    """

    ConfigClass = CompareWarpAssembleCoaddConfig
    _DefaultName = "compareWarpAssembleCoadd"

    def __init__(self, *args, **kwargs):
        AssembleCoaddTask.__init__(self, *args, **kwargs)
        self.makeSubtask("assembleStaticSkyModel")
        detectionSchema = afwTable.SourceTable.makeMinimalSchema()
        self.makeSubtask("detect", schema=detectionSchema)
        if self.config.doPreserveContainedBySource:
            self.makeSubtask("detectTemplate", schema=afwTable.SourceTable.makeMinimalSchema())
        if self.config.doScaleWarpVariance:
            self.makeSubtask("scaleWarpVariance")
        if self.config.doFilterMorphological:
            self.makeSubtask("maskStreaks")

    @utils.inheritDoc(AssembleCoaddTask)
    def _makeSupplementaryData(self, butlerQC, inputRefs, outputRefs):
        """Generate a templateCoadd to use as a naive model of static sky to
        subtract from PSF-Matched warps.

        Returns
        -------
        result : `~lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``templateCoadd``
                Coadded exposure (`lsst.afw.image.Exposure`).
            ``nImage``
                Keeps track of exposure count for each pixel (`lsst.afw.image.ImageU`).

        Raises
        ------
        RuntimeError
            Raised if ``templateCoadd`` is `None`.
        """
        # Ensure that psfMatchedWarps are used as input warps for template generation
        staticSkyModelInputRefs = copy.deepcopy(inputRefs)
        staticSkyModelInputRefs.inputWarps = inputRefs.psfMatchedWarps

        # Because subtasks don't have connections we have to make one.
        # The main task's `templateCoadd` is the subtask's `coaddExposure`
        staticSkyModelOutputRefs = copy.deepcopy(outputRefs)
        if self.config.assembleStaticSkyModel.doWrite:
            staticSkyModelOutputRefs.coaddExposure = staticSkyModelOutputRefs.templateCoadd
            # Remove template coadd from both subtask's and main tasks outputs,
            # because it is handled by the subtask as `coaddExposure`
            del outputRefs.templateCoadd
            del staticSkyModelOutputRefs.templateCoadd

        # A PSF-Matched nImage does not exist as a dataset type
        if 'nImage' in staticSkyModelOutputRefs.keys():
            del staticSkyModelOutputRefs.nImage

        templateCoadd = self.assembleStaticSkyModel.runQuantum(butlerQC, staticSkyModelInputRefs,
                                                               staticSkyModelOutputRefs)
        if templateCoadd is None:
            raise RuntimeError(self._noTemplateMessage(self.assembleStaticSkyModel.warpType))

        return pipeBase.Struct(templateCoadd=templateCoadd.coaddExposure,
                               nImage=templateCoadd.nImage,
                               warpRefList=templateCoadd.warpRefList,
                               imageScalerList=templateCoadd.imageScalerList,
                               weightList=templateCoadd.weightList)

    def _noTemplateMessage(self, warpType):
        warpName = (warpType[0].upper() + warpType[1:])
        message = """No %(warpName)s warps were found to build the template coadd which is
            required to run CompareWarpAssembleCoaddTask. To continue assembling this type of coadd,
            first either rerun makeCoaddTempExp with config.make%(warpName)s=True or
            coaddDriver with config.makeCoadTempExp.make%(warpName)s=True, before assembleCoadd.

            Alternatively, to use another algorithm with existing warps, retarget the CoaddDriverConfig to
            another algorithm like:

                from lsst.pipe.tasks.assembleCoadd import SafeClipAssembleCoaddTask
                config.assemble.retarget(SafeClipAssembleCoaddTask)
        """ % {"warpName": warpName}
        return message

    @utils.inheritDoc(AssembleCoaddTask)
    @timeMethod
    def run(self, skyInfo, tempExpRefList, imageScalerList, weightList,
            supplementaryData):
        """Assemble the coadd.

        Find artifacts and apply them to the warps' masks creating a list of
        alternative masks with a new "CLIPPED" plane and updated "NO_DATA"
        plane. Then pass these alternative masks to the base class's ``run``
        method.
        """
        # Check and match the order of the supplementaryData
        # (PSF-matched) inputs to the order of the direct inputs,
        # so that the artifact mask is applied to the right warp
        dataIds = [ref.dataId for ref in tempExpRefList]
        psfMatchedDataIds = [ref.dataId for ref in supplementaryData.warpRefList]

        if dataIds != psfMatchedDataIds:
            self.log.info("Reordering and or/padding PSF-matched visit input list")
            supplementaryData.warpRefList = reorderAndPadList(supplementaryData.warpRefList,
                                                              psfMatchedDataIds, dataIds)
            supplementaryData.imageScalerList = reorderAndPadList(supplementaryData.imageScalerList,
                                                                  psfMatchedDataIds, dataIds)

        # Use PSF-Matched Warps (and corresponding scalers) and coadd to find artifacts
        spanSetMaskList = self.findArtifacts(supplementaryData.templateCoadd,
                                             supplementaryData.warpRefList,
                                             supplementaryData.imageScalerList)

        badMaskPlanes = self.config.badMaskPlanes[:]
        badMaskPlanes.append("CLIPPED")
        badPixelMask = afwImage.Mask.getPlaneBitMask(badMaskPlanes)

        result = AssembleCoaddTask.run(self, skyInfo, tempExpRefList, imageScalerList, weightList,
                                       spanSetMaskList, mask=badPixelMask)

        # Propagate PSF-matched EDGE pixels to coadd SENSOR_EDGE and INEXACT_PSF
        # Psf-Matching moves the real edge inwards
        self.applyAltEdgeMask(result.coaddExposure.maskedImage.mask, spanSetMaskList)
        return result

    def applyAltEdgeMask(self, mask, altMaskList):
        """Propagate alt EDGE mask to SENSOR_EDGE AND INEXACT_PSF planes.

        Parameters
        ----------
        mask : `lsst.afw.image.Mask`
            Original mask.
        altMaskList : `list` of `dict`
            List of Dicts containing ``spanSet`` lists.
            Each element contains the new mask plane name (e.g. "CLIPPED
            and/or "NO_DATA") as the key, and list of ``SpanSets`` to apply to
            the mask.
        """
        maskValue = mask.getPlaneBitMask(["SENSOR_EDGE", "INEXACT_PSF"])
        for visitMask in altMaskList:
            if "EDGE" in visitMask:
                for spanSet in visitMask['EDGE']:
                    spanSet.clippedTo(mask.getBBox()).setMask(mask, maskValue)

    def findArtifacts(self, templateCoadd, tempExpRefList, imageScalerList):
        """Find artifacts.

        Loop through warps twice. The first loop builds a map with the count
        of how many epochs each pixel deviates from the templateCoadd by more
        than ``config.chiThreshold`` sigma. The second loop takes each
        difference image and filters the artifacts detected in each using
        count map to filter out variable sources and sources that are
        difficult to subtract cleanly.

        Parameters
        ----------
        templateCoadd : `lsst.afw.image.Exposure`
            Exposure to serve as model of static sky.
        tempExpRefList : `list`
            List of data references to warps.
        imageScalerList : `list`
            List of image scalers.

        Returns
        -------
        altMasks : `list` of `dict`
            List of dicts containing information about CLIPPED
            (i.e., artifacts), NO_DATA, and EDGE pixels.
        """
        self.log.debug("Generating Count Image, and mask lists.")
        coaddBBox = templateCoadd.getBBox()
        slateIm = afwImage.ImageU(coaddBBox)
        epochCountImage = afwImage.ImageU(coaddBBox)
        nImage = afwImage.ImageU(coaddBBox)
        spanSetArtifactList = []
        spanSetNoDataMaskList = []
        spanSetEdgeList = []
        spanSetBadMorphoList = []
        badPixelMask = self.getBadPixelMask()

        # mask of the warp diffs should = that of only the warp
        templateCoadd.mask.clearAllMaskPlanes()

        if self.config.doPreserveContainedBySource:
            templateFootprints = self.detectTemplate.detectFootprints(templateCoadd)
        else:
            templateFootprints = None

        for warpRef, imageScaler in zip(tempExpRefList, imageScalerList):
            warpDiffExp = self._readAndComputeWarpDiff(warpRef, imageScaler, templateCoadd)
            if warpDiffExp is not None:
                # This nImage only approximates the final nImage because it uses the PSF-matched mask
                nImage.array += (numpy.isfinite(warpDiffExp.image.array)
                                 * ((warpDiffExp.mask.array & badPixelMask) == 0)).astype(numpy.uint16)
                fpSet = self.detect.detectFootprints(warpDiffExp, doSmooth=False, clearMask=True)
                fpSet.positive.merge(fpSet.negative)
                footprints = fpSet.positive
                slateIm.set(0)
                spanSetList = [footprint.spans for footprint in footprints.getFootprints()]

                # Remove artifacts due to defects before they contribute to the epochCountImage
                if self.config.doPrefilterArtifacts:
                    spanSetList = self.prefilterArtifacts(spanSetList, warpDiffExp)

                # Clear mask before adding prefiltered spanSets
                self.detect.clearMask(warpDiffExp.mask)
                for spans in spanSetList:
                    spans.setImage(slateIm, 1, doClip=True)
                    spans.setMask(warpDiffExp.mask, warpDiffExp.mask.getPlaneBitMask("DETECTED"))
                epochCountImage += slateIm

                if self.config.doFilterMorphological:
                    maskName = self.config.streakMaskName
                    _ = self.maskStreaks.run(warpDiffExp)
                    streakMask = warpDiffExp.mask
                    spanSetStreak = afwGeom.SpanSet.fromMask(streakMask,
                                                             streakMask.getPlaneBitMask(maskName)).split()
                    # Pad the streaks to account for low-surface brightness wings
                    psf = warpDiffExp.getPsf()
                    for s, sset in enumerate(spanSetStreak):
                        psfShape = psf.computeShape(sset.computeCentroid())
                        dilation = self.config.growStreakFp * psfShape.getDeterminantRadius()
                        sset_dilated = sset.dilated(int(dilation))
                        spanSetStreak[s] = sset_dilated

                # PSF-Matched warps have less available area (~the matching kernel) because the calexps
                # undergo a second convolution. Pixels with data in the direct warp
                # but not in the PSF-matched warp will not have their artifacts detected.
                # NaNs from the PSF-matched warp therefore must be masked in the direct warp
                nans = numpy.where(numpy.isnan(warpDiffExp.maskedImage.image.array), 1, 0)
                nansMask = afwImage.makeMaskFromArray(nans.astype(afwImage.MaskPixel))
                nansMask.setXY0(warpDiffExp.getXY0())
                edgeMask = warpDiffExp.mask
                spanSetEdgeMask = afwGeom.SpanSet.fromMask(edgeMask,
                                                           edgeMask.getPlaneBitMask("EDGE")).split()
            else:
                # If the directWarp has <1% coverage, the psfMatchedWarp can have 0% and not exist
                # In this case, mask the whole epoch
                nansMask = afwImage.MaskX(coaddBBox, 1)
                spanSetList = []
                spanSetEdgeMask = []
                spanSetStreak = []

            spanSetNoDataMask = afwGeom.SpanSet.fromMask(nansMask).split()

            spanSetNoDataMaskList.append(spanSetNoDataMask)
            spanSetArtifactList.append(spanSetList)
            spanSetEdgeList.append(spanSetEdgeMask)
            if self.config.doFilterMorphological:
                spanSetBadMorphoList.append(spanSetStreak)

        if lsstDebug.Info(__name__).saveCountIm:
            path = self._dataRef2DebugPath("epochCountIm", tempExpRefList[0], coaddLevel=True)
            epochCountImage.writeFits(path)

        for i, spanSetList in enumerate(spanSetArtifactList):
            if spanSetList:
                filteredSpanSetList = self.filterArtifacts(spanSetList, epochCountImage, nImage,
                                                           templateFootprints)
                spanSetArtifactList[i] = filteredSpanSetList
            if self.config.doFilterMorphological:
                spanSetArtifactList[i] += spanSetBadMorphoList[i]

        altMasks = []
        for artifacts, noData, edge in zip(spanSetArtifactList, spanSetNoDataMaskList, spanSetEdgeList):
            altMasks.append({'CLIPPED': artifacts,
                             'NO_DATA': noData,
                             'EDGE': edge})
        return altMasks

    def prefilterArtifacts(self, spanSetList, exp):
        """Remove artifact candidates covered by bad mask plane.

        Any future editing of the candidate list that does not depend on
        temporal information should go in this method.

        Parameters
        ----------
        spanSetList : `list` of `lsst.afw.geom.SpanSet`
            List of SpanSets representing artifact candidates.
        exp : `lsst.afw.image.Exposure`
            Exposure containing mask planes used to prefilter.

        Returns
        -------
        returnSpanSetList : `list` of `lsst.afw.geom.SpanSet`
            List of SpanSets with artifacts.
        """
        badPixelMask = exp.mask.getPlaneBitMask(self.config.prefilterArtifactsMaskPlanes)
        goodArr = (exp.mask.array & badPixelMask) == 0
        returnSpanSetList = []
        bbox = exp.getBBox()
        x0, y0 = exp.getXY0()
        for i, span in enumerate(spanSetList):
            y, x = span.clippedTo(bbox).indices()
            yIndexLocal = numpy.array(y) - y0
            xIndexLocal = numpy.array(x) - x0
            goodRatio = numpy.count_nonzero(goodArr[yIndexLocal, xIndexLocal])/span.getArea()
            if goodRatio > self.config.prefilterArtifactsRatio:
                returnSpanSetList.append(span)
        return returnSpanSetList

    def filterArtifacts(self, spanSetList, epochCountImage, nImage, footprintsToExclude=None):
        """Filter artifact candidates.

        Parameters
        ----------
        spanSetList : `list` of `lsst.afw.geom.SpanSet`
            List of SpanSets representing artifact candidates.
        epochCountImage : `lsst.afw.image.Image`
            Image of accumulated number of warpDiff detections.
        nImage : `lsst.afw.image.ImageU`
            Image of the accumulated number of total epochs contributing.

        Returns
        -------
        maskSpanSetList : `list`
            List of SpanSets with artifacts.
        """
        maskSpanSetList = []
        x0, y0 = epochCountImage.getXY0()
        for i, span in enumerate(spanSetList):
            y, x = span.indices()
            yIdxLocal = [y1 - y0 for y1 in y]
            xIdxLocal = [x1 - x0 for x1 in x]
            outlierN = epochCountImage.array[yIdxLocal, xIdxLocal]
            totalN = nImage.array[yIdxLocal, xIdxLocal]

            # effectiveMaxNumEpochs is broken line (fraction of N) with characteristic config.maxNumEpochs
            effMaxNumEpochsHighN = (self.config.maxNumEpochs
                                    + self.config.maxFractionEpochsHigh*numpy.mean(totalN))
            effMaxNumEpochsLowN = self.config.maxFractionEpochsLow * numpy.mean(totalN)
            effectiveMaxNumEpochs = int(min(effMaxNumEpochsLowN, effMaxNumEpochsHighN))
            nPixelsBelowThreshold = numpy.count_nonzero((outlierN > 0)
                                                        & (outlierN <= effectiveMaxNumEpochs))
            percentBelowThreshold = nPixelsBelowThreshold / len(outlierN)
            if percentBelowThreshold > self.config.spatialThreshold:
                maskSpanSetList.append(span)

        if self.config.doPreserveContainedBySource and footprintsToExclude is not None:
            # If a candidate is contained by a footprint on the template coadd, do not clip
            filteredMaskSpanSetList = []
            for span in maskSpanSetList:
                doKeep = True
                for footprint in footprintsToExclude.positive.getFootprints():
                    if footprint.spans.contains(span):
                        doKeep = False
                        break
                if doKeep:
                    filteredMaskSpanSetList.append(span)
            maskSpanSetList = filteredMaskSpanSetList

        return maskSpanSetList

    def _readAndComputeWarpDiff(self, warpRef, imageScaler, templateCoadd):
        """Fetch a warp from the butler and return a warpDiff.

        Parameters
        ----------
        warpRef : `lsst.daf.butler.DeferredDatasetHandle`
            Handle for the warp.
        imageScaler : `lsst.pipe.tasks.scaleZeroPoint.ImageScaler`
            An image scaler object.
        templateCoadd : `lsst.afw.image.Exposure`
            Exposure to be substracted from the scaled warp.

        Returns
        -------
        warp : `lsst.afw.image.Exposure`
            Exposure of the image difference between the warp and template.
        """
        # If the PSF-Matched warp did not exist for this direct warp
        # None is holding its place to maintain order in Gen 3
        if warpRef is None:
            return None

        warp = warpRef.get()
        # direct image scaler OK for PSF-matched Warp
        imageScaler.scaleMaskedImage(warp.getMaskedImage())
        mi = warp.getMaskedImage()
        if self.config.doScaleWarpVariance:
            try:
                self.scaleWarpVariance.run(mi)
            except Exception as exc:
                self.log.warning("Unable to rescale variance of warp (%s); leaving it as-is", exc)
        mi -= templateCoadd.getMaskedImage()
        return warp
