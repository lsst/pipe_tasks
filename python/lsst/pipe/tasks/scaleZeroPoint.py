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

__all__ = ["ImageScaler", "SpatialImageScaler", "ScaleZeroPointTask"]

import numpy
import lsst.geom as geom
import lsst.afw.image as afwImage
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.selectImages import BaseSelectImagesTask


class ImageScaler:
    """A class that scales an image.

    This version uses a single scalar. Fancier versions may use a spatially varying scale.

    Parameters
    ----------
    scale : `float`, optional
        Scale correction to apply (see ``scaleMaskedImage``).
    """

    def __init__(self, scale=1.0):
        self._scale = scale

    def scaleMaskedImage(self, maskedImage):
        """Scale the specified image or masked image in place.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Masked image to scale.
        """
        maskedImage *= self._scale


class SpatialImageScaler(ImageScaler):
    """Multiplicative image scaler using interpolation over a grid of points.

    Contains the x, y positions in tract coordinates and the scale factors.
    Interpolates only when scaleMaskedImage() or getInterpImage() is called.

    Currently the only type of 'interpolation' implemented is CONSTANT which calculates the mean.

    Parameters
    ----------
    interpStyle : `Unknown`
        Interpolation style (`CONSTANT` is only option).
    xList : `list` of `int`
        List of X pixel positions.
    yList : `list` of `int`
        List of Y pixel positions.
    scaleList : `Unknown`
        List of multiplicative scale factors at (x,y).

    Raises
    ------
    RuntimeError
        Raised if the lists have different lengths.
    """

    def __init__(self, interpStyle, xList, yList, scaleList):
        if len(xList) != len(yList) or len(xList) != len(scaleList):
            raise RuntimeError(
                "len(xList)=%s len(yList)=%s, len(scaleList)=%s but all lists must have the same length" %
                (len(xList), len(yList), len(scaleList)))

        # Eventually want this do be: self.interpStyle = getattr(afwMath.Interpolate2D, interpStyle)
        self._xList = xList
        self._yList = yList
        self._scaleList = scaleList

    def scaleMaskedImage(self, maskedImage):
        """Apply scale correction to the specified masked image.

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage`
            To scale; scale is applied in place.
        """
        scale = self.getInterpImage(maskedImage.getBBox())
        maskedImage *= scale

    def getInterpImage(self, bbox):
        """Return an image containing the scale correction with same bounding box as supplied.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Integer bounding box for image.

        Raises
        ------
        RuntimeError
            Raised if there are no fluxMag0s to interpolate.
        """
        npoints = len(self._xList)

        if npoints < 1:
            raise RuntimeError("Cannot create scaling image. Found no fluxMag0s to interpolate")

        image = afwImage.ImageF(bbox, numpy.mean(self._scaleList))

        return image


class ScaleZeroPointConfig(pexConfig.Config):
    """Config for ScaleZeroPointTask.
    """

    zeroPoint = pexConfig.Field(
        dtype=float,
        doc="desired photometric zero point",
        default=27.0,
    )


class SpatialScaleZeroPointConfig(ScaleZeroPointConfig):
    selectFluxMag0 = pexConfig.ConfigurableField(
        doc="Task to select data to compute spatially varying photometric zeropoint",
        target=BaseSelectImagesTask,
    )

    interpStyle = pexConfig.ChoiceField(
        dtype=str,
        doc="Algorithm to interpolate the flux scalings;"
        "Currently only one choice implemented",
        default="CONSTANT",
        allowed={
            "CONSTANT": "Use a single constant value",
        }
    )


class ScaleZeroPointTask(pipeBase.Task):
    """Compute scale factor to scale exposures to a desired photometric zero point.

    This simple version assumes that the zero point is spatially invariant.
    """

    ConfigClass = ScaleZeroPointConfig
    _DefaultName = "scaleZeroPoint"

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)

        # flux at mag=0 is 10^(zeroPoint/2.5)   because m = -2.5*log10(F/F0)
        fluxMag0 = 10**(0.4 * self.config.zeroPoint)
        self._photoCalib = afwImage.makePhotoCalibFromCalibZeroPoint(fluxMag0, 0.0)

    def run(self, exposure, dataRef=None):
        """Scale the specified exposure to the desired photometric zeropoint.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to scale; masked image is scaled in place.
        dataRef : `Unknown`
            Data reference for exposure.
            Not used, but in API so that users can switch between spatially variant
            and invariant tasks.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``imageScaler``
                The image scaling object used to scale exposure.
        """
        imageScaler = self.computeImageScaler(exposure=exposure, dataRef=dataRef)
        mi = exposure.getMaskedImage()
        imageScaler.scaleMaskedImage(mi)
        return pipeBase.Struct(
            imageScaler=imageScaler,
        )

    def computeImageScaler(self, exposure, dataRef=None):
        """Compute image scaling object for a given exposure.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure for which scaling is desired.
        dataRef : `Unknown`, optional
            Data reference for exposure.
            Not used, but in API so that users can switch between spatially variant
            and invariant tasks.
        """
        scale = self.scaleFromPhotoCalib(exposure.getPhotoCalib()).scale
        return ImageScaler(scale)

    def getPhotoCalib(self):
        """Get desired PhotoCalib.

        Returns
        -------
        calibration : `lsst.afw.image.PhotoCalib`
            Calibration with ``fluxMag0`` set appropriately for config.zeroPoint.
        """
        return self._photoCalib

    def scaleFromPhotoCalib(self, calib):
        """Compute the scale for the specified PhotoCalib.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            `scale`
                Scale, such that if pixelCalib describes the photometric zeropoint of a pixel
                then the following scales that pixel to the photometric zeropoint specified by config.zeroPoint:
                    scale = computeScale(pixelCalib)
                    pixel *= scale

        Notes
        -----
        Returns a struct to leave room for scaleErr in a future implementation.
        """
        fluxAtZeroPoint = calib.magnitudeToInstFlux(self.config.zeroPoint)
        return pipeBase.Struct(
            scale=1.0 / fluxAtZeroPoint,
        )

    def scaleFromFluxMag0(self, fluxMag0):
        """Compute the scale for the specified fluxMag0.

        This is a wrapper around scaleFromPhotoCalib, which see for more information.

        Parameters
        ----------
        fluxMag0 : `float`
            Flux at magnitude zero.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            `scale`
                Scale, such that if pixelCalib describes the photometric zeropoint of a pixel
                then the following scales that pixel to the photometric zeropoint specified by config.zeroPoint:
                    scale = computeScale(pixelCalib)
                    pixel *= scale
        """
        calib = afwImage.makePhotoCalibFromCalibZeroPoint(fluxMag0, 0.0)
        return self.scaleFromPhotoCalib(calib)


class SpatialScaleZeroPointTask(ScaleZeroPointTask):
    """Compute spatially varying scale factor to scale exposures to a desired photometric zero point.
    """

    ConfigClass = SpatialScaleZeroPointConfig
    _DefaultName = "scaleZeroPoint"

    def __init__(self, *args, **kwargs):
        ScaleZeroPointTask.__init__(self, *args, **kwargs)
        self.makeSubtask("selectFluxMag0")

    def run(self, exposure, dataRef):
        """Scale the specified exposure to the desired photometric zeropoint.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to scale; masked image is scaled in place.
        dataRef : `Unknown`
            Data reference for exposure.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``imageScaler``
                The image scaling object used to scale exposure.
        """
        imageScaler = self.computeImageScaler(exposure=exposure, dataRef=dataRef)
        mi = exposure.getMaskedImage()
        imageScaler.scaleMaskedImage(mi)
        return pipeBase.Struct(
            imageScaler=imageScaler,
        )

    def computeImageScaler(self, exposure, dataRef):
        """Compute image scaling object for a given exposure.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure for which scaling is desired. Only wcs and bbox are used.
        dataRef : `Unknown`
            Data reference of exposure.
            dataRef.dataId used to retrieve all applicable fluxMag0's from a database.

        Returns
        -------
        result : `SpatialImageScaler`
        """
        wcs = exposure.getWcs()

        fluxMagInfoList = self.selectFluxMag0.run(dataRef.dataId).fluxMagInfoList

        xList = []
        yList = []
        scaleList = []

        for fluxMagInfo in fluxMagInfoList:
            # find center of field in tract coordinates
            if not fluxMagInfo.coordList:
                raise RuntimeError("no x,y data for fluxMagInfo")
            ctr = geom.Extent2D()
            for coord in fluxMagInfo.coordList:
                # accumulate x, y
                ctr += geom.Extent2D(wcs.skyToPixel(coord))
            # and find average x, y as the center of the chip
            ctr = geom.Point2D(ctr / len(fluxMagInfo.coordList))
            xList.append(ctr.getX())
            yList.append(ctr.getY())
            scaleList.append(self.scaleFromFluxMag0(fluxMagInfo.fluxMag0).scale)

        self.log.info("Found %d flux scales for interpolation: %s",
                      len(scaleList), [f"{s:%0.4f}" for s in scaleList])
        return SpatialImageScaler(
            interpStyle=self.config.interpStyle,
            xList=xList,
            yList=yList,
            scaleList=scaleList,
        )
