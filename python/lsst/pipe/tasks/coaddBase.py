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
import lsst.pex.config as pexConfig
import lsst.geom as geom
import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase
import lsst.meas.algorithms as measAlg

from lsst.meas.algorithms import ScaleVarianceTask
from .selectImages import WcsSelectImagesTask
from .coaddInputRecorder import CoaddInputRecorderTask

__all__ = ["CoaddBaseTask", "getSkyInfo", "makeSkyInfo"]


class CoaddBaseConfig(pexConfig.Config):
    """!Configuration parameters for CoaddBaseTask

    @anchor CoaddBaseConfig_

    @brief Configuration parameters shared between MakeCoaddTempExp and AssembleCoadd
    """
    coaddName = pexConfig.Field(
        doc="Coadd name: typically one of deep or goodSeeing.",
        dtype=str,
        default="deep",
    )
    select = pexConfig.ConfigurableField(
        doc="Image selection subtask.",
        target=WcsSelectImagesTask,
    )
    badMaskPlanes = pexConfig.ListField(
        dtype=str,
        doc="Mask planes that, if set, the associated pixel should not be included in the coaddTempExp.",
        default=("NO_DATA",),
    )
    inputRecorder = pexConfig.ConfigurableField(
        doc="Subtask that helps fill CoaddInputs catalogs added to the final Exposure",
        target=CoaddInputRecorderTask
    )
    doPsfMatch = pexConfig.Field(
        dtype=bool,
        doc="Match to modelPsf? Deprecated. Sets makePsfMatched=True, makeDirect=False",
        default=False
    )
    modelPsf = measAlg.GaussianPsfFactory.makeField(doc="Model Psf factory")
    doApplyExternalPhotoCalib = pexConfig.Field(
        dtype=bool,
        default=False,
        doc=("Whether to apply external photometric calibration via an "
             "`lsst.afw.image.PhotoCalib` object.  Uses the "
             "`externalPhotoCalibName` field to determine which calibration "
             "to load.")
    )
    useGlobalExternalPhotoCalib = pexConfig.Field(
        dtype=bool,
        default=True,
        doc=("When using doApplyExternalPhotoCalib, use 'global' calibrations "
             "that are not run per-tract.  When False, use per-tract photometric "
             "calibration files.")
    )
    externalPhotoCalibName = pexConfig.ChoiceField(
        # TODO: Remove this config with the removal of Gen2 in DM-20572.
        dtype=str,
        doc=("Type of external PhotoCalib if `doApplyExternalPhotoCalib` is True. "
             "This field is only used for Gen2 middleware."),
        default="jointcal",
        allowed={
            "jointcal": "Use jointcal_photoCalib",
            "fgcm": "Use fgcm_photoCalib",
            "fgcm_tract": "Use fgcm_tract_photoCalib"
        }
    )
    doApplyExternalSkyWcs = pexConfig.Field(
        dtype=bool,
        default=False,
        doc=("Whether to apply external astrometric calibration via an "
             "`lsst.afw.geom.SkyWcs` object.  Uses `externalSkyWcsName` "
             "field to determine which calibration to load.")
    )
    useGlobalExternalSkyWcs = pexConfig.Field(
        dtype=bool,
        default=False,
        doc=("When using doApplyExternalSkyWcs, use 'global' calibrations "
             "that are not run per-tract.  When False, use per-tract wcs "
             "files.")
    )
    externalSkyWcsName = pexConfig.ChoiceField(
        # TODO: Remove this config with the removal of Gen2 in DM-20572.
        dtype=str,
        doc=("Type of external SkyWcs if `doApplyExternalSkyWcs` is True. "
             "This field is only used for Gen2 middleware."),
        default="jointcal",
        allowed={
            "jointcal": "Use jointcal_wcs"
        }
    )
    includeCalibVar = pexConfig.Field(
        dtype=bool,
        doc="Add photometric calibration variance to warp variance plane.",
        default=False
    )
    matchingKernelSize = pexConfig.Field(
        dtype=int,
        doc="Size in pixels of matching kernel. Must be odd.",
        default=21,
        check=lambda x: x % 2 == 1
    )


class CoaddBaseTask(pipeBase.PipelineTask):
    """!Base class for coaddition.

    Subclasses must specify _DefaultName
    """
    ConfigClass = CoaddBaseConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("select")
        self.makeSubtask("inputRecorder")

    def selectExposures(self, patchRef, skyInfo=None, selectDataList=[]):
        """!
        @brief Select exposures to coadd

        Get the corners of the bbox supplied in skyInfo using @ref geom.Box2D and convert the pixel
        positions of the bbox corners to sky coordinates using
        @ref afw::geom::SkyWcs::pixelToSky "skyInfo.wcs.pixelToSky". Use the
        @ref selectImages::WcsSelectImagesTask "WcsSelectImagesTask" to select exposures that lie
        inside the patch indicated by the dataRef.

        @param[in] patchRef  data reference for sky map patch. Must include keys "tract", "patch",
                             plus the camera-specific filter key (e.g. "filter" or "band")
        @param[in] skyInfo   geometry for the patch; output from getSkyInfo
        @param[in] selectDataList list of @ref selectImages::SelectStruct "SelectStruct"
                             to consider for selection
        @return    a list of science exposures to coadd, as butler data references
        """
        if skyInfo is None:
            skyInfo = self.getSkyInfo(patchRef)
        cornerPosList = geom.Box2D(skyInfo.bbox).getCorners()
        coordList = [skyInfo.wcs.pixelToSky(pos) for pos in cornerPosList]
        return self.select.runDataRef(patchRef, coordList, selectDataList=selectDataList).dataRefList

    def getSkyInfo(self, patchRef):
        """!
        @brief Use @ref coaddBase::getSkyInfo "getSkyInfo" to return the skyMap,
        tract and patch information, wcs and the outer bbox
        of the patch.

        @param[in] patchRef  data reference for sky map. Must include keys "tract" and "patch"

        @return pipe_base Struct containing:
        - skyMap: sky map
        - tractInfo: information for chosen tract of sky map
        - patchInfo: information about chosen patch of tract
        - wcs: WCS of tract
        - bbox: outer bbox of patch, as an geom Box2I
        """
        return getSkyInfo(coaddName=self.config.coaddName, patchRef=patchRef)

    def getTempExpDatasetName(self, warpType="direct"):
        """Return warp name for given warpType and task config

        Parameters
        ----------
        warpType : string
            Either 'direct' or 'psfMatched'

        Returns
        -------
        WarpDatasetName : `string`
        """
        return self.config.coaddName + "Coadd_" + warpType + "Warp"

    def getBadPixelMask(self):
        """!
        @brief Convenience method to provide the bitmask from the mask plane names
        """
        return afwImage.Mask.getPlaneBitMask(self.config.badMaskPlanes)


def getSkyInfo(coaddName, patchRef):
    """!
    @brief Return the SkyMap, tract and patch information, wcs, and outer bbox of the patch to be coadded.

    @param[in]  coaddName  coadd name; typically one of deep or goodSeeing
    @param[in]  patchRef   data reference for sky map. Must include keys "tract" and "patch"

    @return pipe_base Struct containing:
    - skyMap: sky map
    - tractInfo: information for chosen tract of sky map
    - patchInfo: information about chosen patch of tract
    - wcs: WCS of tract
    - bbox: outer bbox of patch, as an geom Box2I
    """
    skyMap = patchRef.get(coaddName + "Coadd_skyMap")
    return makeSkyInfo(skyMap, patchRef.dataId["tract"], patchRef.dataId["patch"])


def makeSkyInfo(skyMap, tractId, patchId):
    """Return SkyInfo Struct

    Constructs SkyInfo used by coaddition tasks for multiple
    patchId formats.

    Parameters
    ----------
    skyMap : `lsst.skyMap.SkyMap`
    tractId : int
    patchId : str or int or tuple of int
        Either Gen2-style comma delimited string (e.g. '4,5'),
        tuple of integers (e.g (4, 5), Gen3-style integer.
    """
    tractInfo = skyMap[tractId]

    if isinstance(patchId, str) and ',' in patchId:
        #  patch format is "xIndex,yIndex"
        patchIndex = tuple(int(i) for i in patchId.split(","))
    else:
        patchIndex = patchId

    patchInfo = tractInfo.getPatchInfo(patchIndex)

    return pipeBase.Struct(
        skyMap=skyMap,
        tractInfo=tractInfo,
        patchInfo=patchInfo,
        wcs=tractInfo.getWcs(),
        bbox=patchInfo.getOuterBBox(),
    )


def scaleVariance(maskedImage, maskPlanes, log=None):
    """!
    @brief Scale the variance in a maskedImage

    The variance plane in a convolved or warped image (or a coadd derived
    from warped images) does not accurately reflect the noise properties of
    the image because variance has been lost to covariance. This function
    attempts to correct for this by scaling the variance plane to match
    the observed variance in the image. This is not perfect (because we're
    not tracking the covariance) but it's simple and is often good enough.

    @deprecated Use the ScaleVarianceTask instead.

    @param maskedImage  MaskedImage to operate on; variance will be scaled
    @param maskPlanes  List of mask planes for pixels to reject
    @param log  Log for reporting the renormalization factor; or None
    @return renormalisation factor
    """
    config = ScaleVarianceTask.ConfigClass()
    config.maskPlanes = maskPlanes
    task = ScaleVarianceTask(config=config, name="scaleVariance", log=log)
    return task.run(maskedImage)


def reorderAndPadList(inputList, inputKeys, outputKeys, padWith=None):
    """Match the order of one list to another, padding if necessary

    Parameters
    ----------
    inputList : list
        List to be reordered and padded. Elements can be any type.
    inputKeys :  iterable
        Iterable of values to be compared with outputKeys.
        Length must match `inputList`
    outputKeys : iterable
        Iterable of values to be compared with inputKeys.
    padWith :
        Any value to be inserted where inputKey not in outputKeys

    Returns
    -------
    list
        Copy of inputList reordered per outputKeys and padded with `padWith`
        so that the length matches length of outputKeys.
    """
    outputList = []
    for d in outputKeys:
        if d in inputKeys:
            outputList.append(inputList[inputKeys.index(d)])
        else:
            outputList.append(padWith)
    return outputList
