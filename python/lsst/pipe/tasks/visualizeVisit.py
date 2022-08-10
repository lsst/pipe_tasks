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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.afw.cameraGeom.utils as afwUtils


__all__ = ['VisualizeBinExpConfig', 'VisualizeBinExpTask',
           'VisualizeMosaicExpConfig', 'VisualizeMosaicExpTask']


class VisualizeBinExpConnections(pipeBase.PipelineTaskConnections,
                                 dimensions=("instrument", "detector")):
    inputExp = cT.Input(
        name="calexp",
        doc="Input exposure data to mosaic.",
        storageClass="ExposureF",
        dimensions=("instrument", "detector"),
    )
    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Input camera to use for mosaic geometry.",
        storageClass="Camera",
        dimensions=("instrument",),
        isCalibration=True,
    )

    outputExp = cT.Output(
        name="calexpBin",
        doc="Output binned image.",
        storageClass="ExposureF",
        dimensions=("instrument", "detector"),
    )


class VisualizeBinExpConfig(pipeBase.PipelineTaskConfig,
                            pipelineConnections=VisualizeBinExpConnections):
    """Configuration for focal plane visualization.
    """
    binning = pexConfig.Field(
        dtype=int,
        default=8,
        doc="Binning factor to apply to each input exposure's image data.",
    )
    detectorKeyword = pexConfig.Field(
        dtype=str,
        default='DET-ID',
        doc="Metadata keyword to use to find detector if not available from input.",
    )


class VisualizeBinExpTask(pipeBase.PipelineTask):
    """Bin the detectors of an exposure.

    The outputs of this task should be passed to
    VisualizeMosaicExpTask to be mosaicked into a full focal plane
    visualization image.
    """
    ConfigClass = VisualizeBinExpConfig
    _DefaultName = 'VisualizeBinExp'

    def run(self, inputExp, camera):
        """Bin input image, attach associated detector.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            Input exposure data to bin.
        camera : `lsst.afw.cameraGeom.Camera`
            Input camera to use for mosaic geometry.

        Returns
        -------
        output : `lsst.pipe.base.Struct`
            Results struct with attribute:

            ``outputExp``
                Binned version of input image (`lsst.afw.image.Exposure`).
        """
        if inputExp.getDetector() is None:
            detectorId = inputExp.getMetadata().get(self.config.detectorKeyword)
            if detectorId is not None:
                inputExp.setDetector(camera[detectorId])

        binned = inputExp.getMaskedImage()
        binned = afwMath.binImage(binned, self.config.binning)
        outputExp = afwImage.makeExposure(binned)

        outputExp.setInfo(inputExp.getInfo())

        return pipeBase.Struct(
            outputExp=outputExp,
        )


class VisualizeMosaicExpConnections(pipeBase.PipelineTaskConnections,
                                    dimensions=("instrument", )):
    inputExps = cT.Input(
        name="calexpBin",
        doc="Input binned images mosaic.",
        storageClass="ExposureF",
        dimensions=("instrument", "detector"),
        multiple=True,
    )
    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Input camera to use for mosaic geometry.",
        storageClass="Camera",
        dimensions=("instrument",),
        isCalibration=True,
    )

    outputData = cT.Output(
        name="calexpFocalPlane",
        doc="Output binned mosaicked frame.",
        storageClass="ImageF",
        dimensions=("instrument", ),
    )


class VisualizeMosaicExpConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=VisualizeMosaicExpConnections):
    """Configuration for focal plane visualization.
    """
    binning = pexConfig.Field(
        dtype=int,
        default=8,
        doc="Binning factor previously applied to input exposures.",
    )


class VisualizeMosaicExpTask(pipeBase.PipelineTask):
    """Task to mosaic binned products.

    The config.binning parameter must match that used in the
    VisualizeBinExpTask.  Otherwise there will be a mismatch between
    the input image size and the expected size of that image in the
    full focal plane frame.
    """
    ConfigClass = VisualizeMosaicExpConfig
    _DefaultName = 'VisualizeMosaicExp'

    def makeCameraImage(self, inputExps, camera, binning):
        """Make an image of an entire focal plane.

        Parameters
        ----------
        exposures: `dict` [`int`, `lsst.afw.image.Exposure`]
            CCD exposures, binned by `binning`.  The keys are the
            detectorIDs, with the values the binned image exposure.

        Returns
        -------
        image : `lsst.afw.image.Image`
            Image mosaicked from the individual binned images for each
            detector.
        """
        image = afwUtils.makeImageFromCamera(
            camera,
            imageSource=ImageSource(inputExps),
            imageFactory=afwImage.ImageF,
            binSize=binning
        )
        return image

    def run(self, inputExps, camera):
        """Mosaic inputs together to create focal plane image.

        Parameters
        ----------
        inputExp : `list` [`lsst.afw.image.Exposure`]
            Input exposure data to bin.
        camera : `lsst.afw.cameraGeom.Camera`
            Input camera to use for mosaic geometry.

        Returns
        -------
        output : `lsst.pipe.base.Struct`
            Results struct with attribute:

            ``outputExp``
                Binned version of input image (`lsst.afw.image.Exposure`).
        """
        expDict = {exp.getDetector().getId(): exp for exp in inputExps}
        image = self.makeCameraImage(expDict, camera, self.config.binning)

        return pipeBase.Struct(
            outputData=image,
        )


class ImageSource:
    """Source of images for makeImageFromCamera"""
    def __init__(self, exposures):
        self.exposures = exposures
        self.isTrimmed = True
        self.background = np.nan

    def getCcdImage(self, detector, imageFactory, binSize):
        """Provide image of CCD to makeImageFromCamera

        Parameters
        ----------
        detector : `int`
            Detector ID to get image data for.
        imageFactory : `lsst.afw.image.Image`
            Type of image to construct.
        binSize : `int`
            Binsize to use to recompute dimensions.

        Returns
        -------
        image : `lsst.afw.image.Image`
            Appropriately rotated, binned, and transformed
            image to be mosaicked.
        detector : `lsst.afw.cameraGeom.Detector`
            Camera detector that the returned image data
            belongs to.
        """
        detId = detector.getId()

        if detId not in self.exposures:
            dims = detector.getBBox().getDimensions()/binSize
            image = imageFactory(*[int(xx) for xx in dims])
            image.set(self.background)
        else:
            image = self.exposures[detector.getId()]
        if hasattr(image, "getMaskedImage"):
            image = image.getMaskedImage()
        if hasattr(image, "getMask"):
            mask = image.getMask()
            isBad = mask.getArray() & mask.getPlaneBitMask("NO_DATA") > 0
            image = image.clone()
            image.getImage().getArray()[isBad] = self.background
        if hasattr(image, "getImage"):
            image = image.getImage()

        # afwMath.rotateImageBy90 checks NQuarter values,
        # so we don't need to here.
        image = afwMath.rotateImageBy90(image, detector.getOrientation().getNQuarter())
        return image, detector
