from __future__ import division, absolute_import
#
# LSST Data Management System
# Copyright 2008-2015 AURA/LSST.
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
import numpy

import lsst.pex.config as pexConfig
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase
import lsst.meas.algorithms as measAlg

from lsst.afw.fits import FitsError
from lsst.coadd.utils import CoaddDataIdContainer
from .selectImages import WcsSelectImagesTask, SelectStruct
from .coaddInputRecorder import CoaddInputRecorderTask

try:
    from lsst.meas.mosaic import applyMosaicResults
except ImportError:
    applyMosaicResults = None

__all__ = ["CoaddBaseTask", "getSkyInfo"]

class CoaddBaseConfig(pexConfig.Config):
    """Config for CoaddBaseTask
    """
    coaddName = pexConfig.Field(
        doc = "Coadd name: typically one of deep or goodSeeing.",
        dtype = str,
        default = "deep",
    )
    select = pexConfig.ConfigurableField(
        doc = "Image selection subtask.",
        target = WcsSelectImagesTask,
    )
    badMaskPlanes = pexConfig.ListField(
        dtype = str,
        doc = "Mask planes that, if set, the associated pixel should not be included in the coaddTempExp.",
        default = ("NO_DATA",),
    )
    inputRecorder = pexConfig.ConfigurableField(
        doc = "Subtask that helps fill CoaddInputs catalogs added to the final Exposure",
        target = CoaddInputRecorderTask
    )
    doPsfMatch = pexConfig.Field(dtype=bool, doc="Match to modelPsf?", default=False)
    modelPsf = measAlg.GaussianPsfFactory.makeField(doc = "Model Psf factory")
    doApplyUberCal = pexConfig.Field(
        dtype = bool,
        doc = "Apply meas_mosaic ubercal results to input calexps?",
        default = False
    )

class CoaddTaskRunner(pipeBase.TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return pipeBase.TaskRunner.getTargetList(parsedCmd, selectDataList=parsedCmd.selectId.dataList,
                                                 **kwargs)

class CoaddBaseTask(pipeBase.CmdLineTask):
    """Base class for coaddition.

    Subclasses must specify _DefaultName
    """
    ConfigClass = CoaddBaseConfig
    RunnerClass = CoaddTaskRunner

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("select")
        self.makeSubtask("inputRecorder")

    def selectExposures(self, patchRef, skyInfo=None, selectDataList=[]):
        """!
        \brief Select exposures to coadd

        Get the corners of the bbox supplied in skyInfo using \ref afwGeom.Box2D and convert the pixel 
        positions of the bbox corners to sky coordinates using \ref skyInfo.wcs.pixelToSky. Use the 
        \ref WcsSelectImagesTask_ "WcsSelectImagesTask" to select exposures that lie inside the patch 
        indicated by the dataRef.

        \param[in] patchRef  data reference for sky map patch. Must include keys "tract", "patch",
                             plus the camera-specific filter key (e.g. "filter" or "band")
        \param[in] skyInfo   geometry for the patch; output from getSkyInfo
        \return    a list of science exposures to coadd, as butler data references
        """
        if skyInfo is None:
            skyInfo = self.getSkyInfo(patchRef)
        cornerPosList = afwGeom.Box2D(skyInfo.bbox).getCorners()
        coordList = [skyInfo.wcs.pixelToSky(pos) for pos in cornerPosList]
        return self.select.runDataRef(patchRef, coordList, selectDataList=selectDataList).dataRefList

    def getSkyInfo(self, patchRef):
        """!
        \brief Use \ref getSkyinfo to return the skyMap, tract and patch information, wcs and the outer bbox 
        of the patch.

        \param[in] patchRef  data reference for sky map. Must include keys "tract" and "patch"

        \return pipe_base Struct containing:
        - skyMap: sky map
        - tractInfo: information for chosen tract of sky map
        - patchInfo: information about chosen patch of tract
        - wcs: WCS of tract
        - bbox: outer bbox of patch, as an afwGeom Box2I
        """
        return getSkyInfo(coaddName=self.config.coaddName, patchRef=patchRef)

    def getCalExp(self, dataRef, bgSubtracted):
        """!Return one "calexp" calibrated exposure

        @param[in] dataRef        a sensor-level data reference
        @param[in] bgSubtracted   return calexp with background subtracted? If False get the
                                  calexp's background background model and add it to the calexp.
        @return calibrated exposure

        If config.doApplyUberCal, meas_mosaic calibrations will be applied to
        the returned exposure using applyMosaicResults.
        """
        exposure = dataRef.get("calexp", immediate=True)
        if not bgSubtracted:
            background = dataRef.get("calexpBackground", immediate=True)
            mi = exposure.getMaskedImage()
            mi += background.getImage()
            del mi
        if not self.config.doApplyUberCal:
            return exposure
        if applyMosaicResults is None:
            raise RuntimeError(
                "Cannot use improved calibrations for %s because meas_mosaic could not be imported."
                % dataRef.dataId
                )
        else:
            applyMosaicResults(dataRef, calexp=exposure)
        return exposure

    def getCoaddDatasetName(self):
        return self.config.coaddName + "Coadd"

    def getTempExpDatasetName(self):
        return self.config.coaddName + "Coadd_tempExp"

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd", help="data ID, e.g. --id tract=12345 patch=1,2",
                               ContainerClass=CoaddDataIdContainer)
        parser.add_id_argument("--selectId", "calexp", help="data ID, e.g. --selectId visit=6789 ccd=0..9",
                               ContainerClass=SelectDataIdContainer)
        return parser

    def _getConfigName(self):
        """Return the name of the config dataset
        """
        return "%s_%s_config" % (self.config.coaddName, self._DefaultName)

    def _getMetadataName(self):
        """Return the name of the metadata dataset
        """
        return "%s_%s_metadata" % (self.config.coaddName, self._DefaultName)

    def getBadPixelMask(self):
        """!
        \brief Convenience method to provide the bitmask from the mask plane names
        """
        return afwImage.MaskU.getPlaneBitMask(self.config.badMaskPlanes)

    def writeCoaddOutput(self, dataRef, obj, suffix=None):
        """!
        \brief Write a coadd product through the butler

        \param[in]      dataRef  data reference for coadd
        \param[in,out]  obj      coadd product to write
        \param[in]      suffix   suffix to apply to coadd dataset name
        """
        objName = self.getCoaddDatasetName()
        if suffix is not None:
            objName += "_" + suffix
        self.log.info("Persisting %s" % objName)
        dataRef.put(obj, objName)

class SelectDataIdContainer(pipeBase.DataIdContainer):
    """!
    \brief A dataId container for inputs to be selected.

    Read the header (including the size and Wcs) for all specified
    inputs and pass those along, ultimately for the SelectImagesTask.
    This is most useful when used with multiprocessing, as input headers are
    only read once.
    """
    def makeDataRefList(self, namespace):
        """Add a dataList containing useful information for selecting images"""
        super(SelectDataIdContainer, self).makeDataRefList(namespace)
        self.dataList = []
        for ref in self.refList:
            try:
                md = ref.get("calexp_md", immediate=True)
                wcs = afwImage.makeWcs(md)
                data = SelectStruct(dataRef=ref, wcs=wcs, dims=(md.get("NAXIS1"), md.get("NAXIS2")))
            except FitsError as e:
                namespace.log.warn("Unable to construct Wcs from %s" % (ref.dataId))
                continue
            self.dataList.append(data)

def getSkyInfo(coaddName, patchRef):
    """!
    \brief Return the SkyMap, tract and patch information, wcs, and outer bbox of the patch to be coadded.

    \param[in]  coaddName  coadd name; typically one of deep or goodSeeing
    \param[in]  patchRef   data reference for sky map. Must include keys "tract" and "patch"

    \return pipe_base Struct containing:
    - skyMap: sky map
    - tractInfo: information for chosen tract of sky map
    - patchInfo: information about chosen patch of tract
    - wcs: WCS of tract
    - bbox: outer bbox of patch, as an afwGeom Box2I
    """
    skyMap = patchRef.get(coaddName + "Coadd_skyMap")
    tractId = patchRef.dataId["tract"]
    tractInfo = skyMap[tractId]

    # patch format is "xIndex,yIndex"
    patchIndex = tuple(int(i) for i in patchRef.dataId["patch"].split(","))
    patchInfo = tractInfo.getPatchInfo(patchIndex)

    return pipeBase.Struct(
        skyMap = skyMap,
        tractInfo = tractInfo,
        patchInfo = patchInfo,
        wcs = tractInfo.getWcs(),
        bbox = patchInfo.getOuterBBox(),
    )

def scaleVariance(maskedImage, maskPlanes, log=None):
    """!
    \brief Scale the variance in a maskedImage

    The variance plane in a convolved or warped image (or a coadd derived
    from warped images) does not accurately reflect the noise properties of
    the image because variance has been lost to covariance. This function
    attempts to correct for this by scaling the variance plane to match
    the observed variance in the image. This is not perfect (because we're
    not tracking the covariance) but it's simple and is often good enough.

    @param maskedImage  MaskedImage to operate on; variance will be scaled
    @param maskPlanes  List of mask planes for pixels to reject
    @param log  Log for reporting the renormalization factor; or None
    @return renormalisation factor
    """
    variance = maskedImage.getVariance()
    sigNoise = maskedImage.getImage().getArray()/numpy.sqrt(variance.getArray())
    maskVal = maskedImage.getMask().getPlaneBitMask(maskPlanes)
    good = (maskedImage.getMask().getArray() & maskVal) == 0
    # Robust measurement of stdev
    q1, q3 = numpy.percentile(sigNoise[good], (25, 75))
    stdev = 0.74*(q3 - q1)
    ratio = stdev**2
    if log:
        log.info("Renormalizing variance by %f" % (ratio,))
    variance *= ratio
    return ratio
