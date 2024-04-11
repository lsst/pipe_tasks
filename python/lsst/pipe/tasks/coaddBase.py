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

__all__ = ["CoaddBaseTask", "makeSkyInfo"]

import lsst.pex.config as pexConfig
import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase
import lsst.geom as geom

from lsst.meas.algorithms import ScaleVarianceTask
from .selectImages import PsfWcsSelectImagesTask
from .coaddInputRecorder import CoaddInputRecorderTask


class CoaddBaseConfig(pexConfig.Config):
    """Configuration parameters for CoaddBaseTask

    Configuration parameters shared between MakeCoaddTempExp and AssembleCoadd
    """

    coaddName = pexConfig.Field(
        doc="Coadd name: typically one of deep or goodSeeing.",
        dtype=str,
        default="deep",
    )
    select = pexConfig.ConfigurableField(
        doc="Image selection subtask.",
        target=PsfWcsSelectImagesTask,
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
        doc="Match to modelPsf? Sets makePsfMatched=True, makeDirect=False",
        deprecated="This field is no longer used. Will be removed after v26.",  # TODO: DM-39841
        default=False
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
    """Base class for coaddition.

    Subclasses must specify _DefaultName
    """

    ConfigClass = CoaddBaseConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("select")
        self.makeSubtask("inputRecorder")

    def getTempExpDatasetName(self, warpType="direct"):
        """Return warp name for given warpType and task config

        Parameters
        ----------
        warpType : `str`
            Either 'direct' or 'psfMatched'.

        Returns
        -------
        WarpDatasetName : `str`
        """
        return self.config.coaddName + "Coadd_" + warpType + "Warp"

    def getBadPixelMask(self):
        """Convenience method to provide the bitmask from the mask plane names
        """
        return afwImage.Mask.getPlaneBitMask(self.config.badMaskPlanes)


def makeSkyInfo(skyMap, tractId, patchId):
    """Constructs SkyInfo used by coaddition tasks for multiple
    patchId formats.

    Parameters
    ----------
    skyMap : `lsst.skyMap.SkyMap`
        Sky map.
    tractId : `int`
        The ID of the tract.
    patchId : `str` or `int` or `tuple` of `int`
        Either Gen2-style comma delimited string (e.g. '4,5'),
        tuple of integers (e.g (4, 5), Gen3-style integer.

    Returns
    -------
    makeSkyInfo : `lsst.pipe.base.Struct`
        pipe_base Struct with attributes:

        ``skyMap``
            Sky map (`lsst.skyMap.SkyMap`).
        ``tractInfo``
            Information for chosen tract of sky map (`lsst.skyMap.TractInfo`).
        ``patchInfo``
            Information about chosen patch of tract (`lsst.skyMap.PatchInfo`).
        ``wcs``
            WCS of tract (`lsst.afw.image.SkyWcs`).
        ``bbox``
            Outer bbox of patch, as an geom Box2I (`lsst.afw.geom.Box2I`).
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
    """Scale the variance in a maskedImage

    This is deprecated. Use the ScaleVarianceTask instead.

    Parameters
    ----------
    maskedImage : `lsst.afw.image.MaskedImage`
        MaskedImage to operate on; variance will be scaled.
    maskPlanes : `list`
        List of mask planes for pixels to reject.
    log : `Unknown`
        Log for reporting the renormalization factor; or None.

    Returns
    -------
    task.run : `Unknown`
        Renormalization factor.

    Notes
    -----
    The variance plane in a convolved or warped image (or a coadd derived
    from warped images) does not accurately reflect the noise properties of
    the image because variance has been lost to covariance. This function
    attempts to correct for this by scaling the variance plane to match
    the observed variance in the image. This is not perfect (because we're
    not tracking the covariance) but it's simple and is often good enough.
    """
    config = ScaleVarianceTask.ConfigClass()
    config.maskPlanes = maskPlanes
    task = ScaleVarianceTask(config=config, name="scaleVariance", log=log)
    return task.run(maskedImage)


def reorderAndPadList(inputList, inputKeys, outputKeys, padWith=None):
    """Match the order of one list to another, padding if necessary

    Parameters
    ----------
    inputList : `list`
        List to be reordered and padded. Elements can be any type.
    inputKeys :  `iterable`
        Iterable of values to be compared with outputKeys. Length must match `inputList`.
    outputKeys : `iterable`
        Iterable of values to be compared with inputKeys.
    padWith : `Unknown`
        Any value to be inserted where inputKey not in outputKeys.

    Returns
    -------
    outputList : `list`
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


def subBBoxIter(bbox, subregionSize):
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
