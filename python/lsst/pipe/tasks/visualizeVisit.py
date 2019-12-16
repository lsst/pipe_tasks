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


__all__ = ['VisBinConfig', 'VisBinTask', 'VisMosaicConfig', 'VisMosaicTask']


class VisBinConnections(pipeBase.PipelineTaskConnections,
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
        dimensions=("instrument", "calibration_label"),
    )

    outputExp = cT.Output(
        name="calexpBin",
        doc="Output binned image.",
        storageClass="ExposureF",
        dimensions=("instrument", "detector"),
    )


class VisBinConfig(pipeBase.PipelineTaskConfig,
                   pipelineConnections=VisBinConnections):
    """Configuration for focal plane visualization.
    """
    binning = pexConfig.Field(
        dtype=int,
        default=8,
        doc="Binning factor to apply.",
    )
    detectorKeyword = pexConfig.Field(
        dtype=str,
        default='DET-ID',
        doc="Metadata keyword to use to find detector if not available from input.",
    )


class VisBinTask(pipeBase.PipelineTask,
                 pipeBase.CmdLineTask):
    """Task for focal plane visualization.
    """
    ConfigClass = VisBinConfig
    _DefaultName = 'VisBin'

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
                Binned version of input image. (`lsst.afw.image.Exposure`)
        """
        if inputExp.getDetector() is None:
            detectorId = inputExp.getMetadata().get(self.config.detectorKeyword)
            if detectorId is not None:
                inputExp.setDetector(camera[detectorId])

        binned = inputExp.getMaskedImage()
        binned = afwMath.binImage(binned, self.config.binning)
        outputExp = afwImage.makeExposure(binned)

        if outputExp.getDetector() is None:
            outputExp.setDetector(inputExp.getDetector())
        return pipeBase.Struct(
            outputExp=outputExp,
        )


class VisMosaicConnections(pipeBase.PipelineTaskConnections,
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
        dimensions=("instrument", "calibration_label"),
    )

    outputData = cT.Output(
        name="calexpFP",
        doc="Output binned mosaicked frame.",
        storageClass="ImageF",
        dimensions=("instrument", ),
    )


class VisMosaicConfig(pipeBase.PipelineTaskConfig,
                      pipelineConnections=VisMosaicConnections):
    """Configuration for focal plane visualization.
    """
    binning = pexConfig.Field(
        dtype=int,
        default=8,
        doc="Binning factor to apply.",
    )


class VisMosaicTask(pipeBase.PipelineTask,
                    pipeBase.CmdLineTask):
    """Task to mosaic binned products.
    """
    ConfigClass = VisMosaicConfig
    _DefaultName = 'VisMosaic'

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
        class ImageSource:
            """Source of images for makeImageFromCamera"""
            def __init__(self, exposures):
                self.isTrimmed = True
                self.exposures = exposures
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

                image = afwMath.rotateImageBy90(image, detector.getOrientation().getNQuarter())
                return image, detector

        image = afwUtils.makeImageFromCamera(
            camera,
            imageSource=ImageSource(inputExps),
            imageFactory=afwImage.ImageF,
            binSize=binning
        )
        return image

    def run(self, inputExps, camera):
        """Mosaic inputs.

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
                Binned version of input image. (`lsst.afw.image.Exposure`)
        """
        expDict = dict((exp.getDetector().getId(), exp) for exp in inputExps)
        image = self.makeCameraImage(expDict, camera, self.config.binning)

        return pipeBase.Struct(
            outputData=image,
        )
