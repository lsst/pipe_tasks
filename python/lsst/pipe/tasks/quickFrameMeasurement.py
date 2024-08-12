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

__all__ = ["QuickFrameMeasurementTaskConfig", "QuickFrameMeasurementTask"]

import numpy as np
import scipy.ndimage as ndImage

import lsst.afw.detection as afwDetect
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.meas.base as measBase
import lsst.daf.base as dafBase
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from lsst.meas.base import MeasurementError
from lsst.meas.algorithms.installGaussianPsf import InstallGaussianPsfTask


class QuickFrameMeasurementTaskConfig(pexConfig.Config):
    """Config class for the QuickFrameMeasurementTask.
    """
    installPsf = pexConfig.ConfigurableField(
        target=InstallGaussianPsfTask,
        doc="Task for installing an initial PSF",
    )
    maxNonRoundness = pexConfig.Field(
        dtype=float,
        doc="Ratio of xx to yy (or vice versa) above which to cut, in order to exclude spectra",
        default=5.,
    )
    maxExtendedness = pexConfig.Field(
        dtype=float,
        doc="Max absolute value of xx and yy above which to cut, in order to exclude large/things",
        default=100,
    )
    doExtendednessCut = pexConfig.Field(
        dtype=bool,
        doc="Apply the extendeness cut, as definted by maxExtendedness",
        default=False,
    )
    centroidPixelPercentile = pexConfig.Field(
        dtype=float,
        doc="The image's percentile value which the centroid must be greater than to pass the final peak"
            " check. Ignored if doCheckCentroidPixelValue is False",
        default=90,
    )
    doCheckCentroidPixelValue = pexConfig.Field(
        dtype=bool,
        doc="Check that the centroid found is actually in the centroidPixelPercentile percentile of the"
            " image? Set to False for donut images.",
        default=True,
    )
    initialPsfWidth = pexConfig.Field(
        dtype=float,
        doc="Guess at the initial PSF FWHM in pixels.",
        default=10,
    )
    nSigmaDetection = pexConfig.Field(
        dtype=float,
        doc="Number of sigma for the detection limit.",
        default=20,
    )
    nPixMinDetection = pexConfig.Field(
        dtype=int,
        doc="Minimum number of pixels in a detected source.",
        default=10,
    )
    donutDiameter = pexConfig.Field(
        dtype=int,
        doc="The expected diameter of donuts in a donut image, in pixels.",
        default=400,
    )

    def setDefaults(self):
        super().setDefaults()
        self.installPsf.fwhm = self.initialPsfWidth


class QuickFrameMeasurementTask(pipeBase.Task):
    """WARNING: An experimental new task with changable API! Do not rely on yet!

    This task finds the centroid of the brightest source in a given CCD-image
    and returns its centroid and a rough estimate of the seeing/PSF.

    It is designed for speed, such that it can be used in observing scripts
    to provide pointing offsets, allowing subsequent pointings to place
    a source at an exact pixel position.

    The approach taken here is deliberately sub-optimal in the detection and
    measurement sense, with all optimisation being done for speed and robustness
    of the result.

    A small set of unit tests exist for this task, which run automatically
    if afwdata is setup. These, however, are stricky unit tests, and will not
    catch algorithmic regressions. TODO: DM-29038 exists to merge a regression
    real test which runs against 1,000 LATISS images, but is therefore slow
    and requires access to the data.

    Parameters
    ----------
    config : `lsst.pipe.tasks.quickFrameMeasurement.QuickFrameMeasurementTaskConfig`
        Configuration class for the QuickFrameMeasurementTask.
    display : `lsst.afw.display.Display`, optional
        The display to use for showing the images, detections and centroids.

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        Return strucure containing whether the task was successful, the main
        source's centroid, its the aperture fluxes, the ixx and iyy of the
        source, and the median ixx, iyy of the detections in the exposure.
        See run() method for further details.

    Raises
    ------
    This task should *never* raise, as the run() method is enclosed in an
    except Exception block, so that it will never fail during observing.
    Failure modes should be limited to returning a return Struct() with the same
    structure as the success case, with all value set to np.nan but with
    result.success=False.
    """
    ConfigClass = QuickFrameMeasurementTaskConfig
    _DefaultName = 'quickFrameMeasurementTask'

    def __init__(self, config, *, display=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.makeSubtask("installPsf")

        self.display = None
        if display:
            self.display = display

        self.centroidName = "base_SdssCentroid"
        self.shapeName = "base_SdssShape"
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.schema.getAliasMap().set("slot_Centroid", self.centroidName)
        self.schema.getAliasMap().set("slot_Shape", self.shapeName)
        self.control = measBase.SdssCentroidControl()
        self.control.maxDistToPeak = -1
        self.centroider = measBase.SdssCentroidAlgorithm(self.control, self.centroidName, self.schema)
        self.sdssShape = measBase.SdssShapeControl()
        self.shaper = measBase.SdssShapeAlgorithm(self.sdssShape, self.shapeName, self.schema)
        self.apFluxControl = measBase.ApertureFluxControl()
        md = dafBase.PropertySet()
        self.apFluxer = measBase.CircularApertureFluxAlgorithm(self.apFluxControl, "aperFlux",
                                                               self.schema, md)

        self.table = afwTable.SourceTable.make(self.schema)  # make sure to call this last!

    @staticmethod
    def detectObjectsInExp(exp, nSigma, nPixMin, grow=0):
        """Run a very basic but fast threshold-based object detection on an exposure
        Return the footPrintSet for the objects in a postISR exposure.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            Image in which to detect objects.
        nSigma : `float`
            nSigma above image's stddev at which to set the detection threshold.
        nPixMin : `int`
            Minimum number of pixels for detection.
        grow : `int`
            Grow the detected footprint by this many pixels.

        Returns
        -------
        footPrintSet : `lsst.afw.detection.FootprintSet`
            FootprintSet containing the detections.
        """
        threshold = afwDetect.Threshold(nSigma, afwDetect.Threshold.STDEV)
        footPrintSet = afwDetect.FootprintSet(exp.getMaskedImage(), threshold, "DETECTED", nPixMin)
        if grow > 0:
            isotropic = True
            footPrintSet = afwDetect.FootprintSet(footPrintSet, grow, isotropic)
        return footPrintSet

    @staticmethod
    def checkResult(exp, centroid, srcNum, percentile):
        """Perform a final check that centroid location is actually bright.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            The exposure on which to operate
        centroid : `tuple` of `float`
            Location of the centroid in pixel coordinates
        scrNum : `int`
            Number of the source in the source catalog. Only used if the check
            is failed, for debug purposes.
        percentile : `float`
            Image's percentile above which the pixel containing the centroid
            must be in order to pass the check.

        Raises
        ------
        ValueError
            Raised if the centroid's pixel is not above the percentile threshold
        """
        threshold = np.percentile(exp.image.array, percentile)
        pixelValue = exp.image[centroid]
        if pixelValue < threshold:
            msg = (f"Final centroid pixel value check failed: srcNum {srcNum} at {centroid}"
                   f" has central pixel = {pixelValue:3f} <"
                   f" {percentile} percentile of image = {threshold:3f}")
            raise ValueError(msg)
        return

    @staticmethod
    def _calcMedianXxYy(objData):
        """Return the median ixx and iyy for object in the image.
        """
        medianXx = np.nanmedian([element['xx'] for element in objData.values()])
        medianYy = np.nanmedian([element['yy'] for element in objData.values()])
        return medianXx, medianYy

    @staticmethod
    def _getCenterOfMass(exp, nominalCentroid, boxSize):
        """Get the centre of mass around a point in the image.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            The exposure in question.
        nominalCentroid : `tuple` of `float`
            Nominal location of the centroid in pixel coordinates.
        boxSize : `int`
            The size of the box around the nominalCentroid in which to measure
            the centre of mass.

        Returns
        -------
        com : `tuple` of `float`
            The locaiton of the centre of mass of the brightest source in pixel
            coordinates.
        """
        centroidPoint = geom.Point2I(nominalCentroid)
        extent = geom.Extent2I(1, 1)
        bbox = geom.Box2I(centroidPoint, extent)
        bbox = bbox.dilatedBy(int(boxSize//2))
        bbox = bbox.clippedTo(exp.getBBox())
        data = exp[bbox].image.array
        xy0 = exp[bbox].getXY0()

        peak = ndImage.center_of_mass(data)
        peak = (peak[1], peak[0])  # numpy coords returned
        com = geom.Point2D(xy0)
        com.shift(geom.Extent2D(*peak))
        return (com[0], com[1])

    def _calcBrightestObjSrcNum(self, objData):
        """Find the brightest source which passes the cuts among the sources.

        Parameters
        ----------
        objData : `dict` of `dict`
            Dictionary, keyed by source number, containing the measurements.

        Returns
        -------
        srcNum : `int`
            The source number of the brightest source which passes the cuts.
        """
        max70, max70srcNum = -1, -1
        max25, max25srcNum = -1, -1

        for srcNum in sorted(objData.keys()):  # srcNum not contiguous so don't use a list comp
            # skip flag used rather than continue statements so we have all the
            # metrics computed for debug purposes as this task is whack-a-mole
            skip = False
            xx = objData[srcNum]['xx']
            yy = objData[srcNum]['yy']

            xx = max(xx, 1e-9)  # need to protect against division by zero
            yy = max(yy, 1e-9)  # because we don't `continue` on zero moments

            if self.config.doExtendednessCut:
                if xx > self.config.maxExtendedness or yy > self.config.maxExtendedness:
                    skip = True

            nonRoundness = xx/yy
            nonRoundness = max(nonRoundness, 1/nonRoundness)
            if nonRoundness > self.config.maxNonRoundness:
                skip = True

            if self.log.isEnabledFor(self.log.DEBUG):
                text = f"src {srcNum}: {objData[srcNum]['xCentroid']:.0f}, {objData[srcNum]['yCentroid']:.0f}"
                text += f" - xx={xx:.1f}, yy={yy:.1f}, nonRound={nonRoundness:.1f}"
                text += f" - ap70={objData[srcNum]['apFlux70']:,.0f}"
                text += f" - ap25={objData[srcNum]['apFlux25']:,.0f}"
                text += f" - skip={skip}"
                self.log.debug(text)

            if skip:
                continue

            ap70 = objData[srcNum]['apFlux70']
            ap25 = objData[srcNum]['apFlux25']
            if ap70 > max70:
                max70 = ap70
                max70srcNum = srcNum
            if ap25 > max25:
                max25 = ap25
                max25srcNum = srcNum
        if max70srcNum != max25srcNum:
            self.log.warning("WARNING! Max apFlux70 for different object than with max apFlux25")

        if max70srcNum >= 0:  # starts as -1, return None if nothing is acceptable
            return max70srcNum
        return None

    def _measureFp(self, fp, exp):
        """Run the measurements on a footprint.

        Parameters
        ----------
        fp : `lsst.afw.detection.Footprint`
            The footprint to measure.
        exp : `lsst.afw.image.Exposure`
            The footprint's parent exposure.

        Returns
        -------
        src : `lsst.afw.table.SourceRecord`
            The source record containing the measurements.
        """
        src = self.table.makeRecord()
        src.setFootprint(fp)
        self.centroider.measure(src, exp)
        self.shaper.measure(src, exp)
        self.apFluxer.measure(src, exp)
        return src

    def _getDataFromSrcRecord(self, src):
        """Extract the shapes and centroids from a source record.

        Parameters
        ----------
        src : `lsst.afw.table.SourceRecord`
            The source record from which to extract the measurements.

        Returns
        -------
        srcData : `lsst.pipe.base.Struct`
            The struct containing the extracted measurements.
        """
        pScale = self.plateScale
        xx = np.sqrt(src['base_SdssShape_xx'])*2.355*pScale  # 2.355 for FWHM, pScale for platescale from exp
        yy = np.sqrt(src['base_SdssShape_yy'])*2.355*pScale
        xCentroid = src['base_SdssCentroid_x']
        yCentroid = src['base_SdssCentroid_y']
        # apFluxes available: 70, 50, 35, 25, 17, 12 9, 6, 4.5, 3
        apFlux70 = src['aperFlux_70_0_instFlux']
        apFlux25 = src['aperFlux_25_0_instFlux']
        return pipeBase.Struct(xx=xx,
                               yy=yy,
                               xCentroid=xCentroid,
                               yCentroid=yCentroid,
                               apFlux70=apFlux70,
                               apFlux25=apFlux25)

    @staticmethod
    def _getDataFromFootprintOnly(fp, exp):
        """Get the shape, centroid and flux from a footprint.

        Parameters
        ----------
        fp : `lsst.afw.detection.Footprint`
            The footprint to measure.
        exp : `lsst.afw.image.Exposure`
            The footprint's parent exposure.

        Returns
        -------
        srcData : `lsst.pipe.base.Struct`
            The struct containing the extracted measurements.
        """
        xx = fp.getShape().getIxx()
        yy = fp.getShape().getIyy()
        xCentroid, yCentroid = fp.getCentroid()
        apFlux70 = np.sum(exp[fp.getBBox()].image.array)
        apFlux25 = np.sum(exp[fp.getBBox()].image.array)
        return pipeBase.Struct(xx=xx,
                               yy=yy,
                               xCentroid=xCentroid,
                               yCentroid=yCentroid,
                               apFlux70=apFlux70,
                               apFlux25=apFlux25)

    @staticmethod
    def _measurementResultToDict(measurementResult):
        """Convenience function to repackage measurement results to a dict.

        Parameters
        ----------
        measurementResult : `lsst.afw.table.SourceRecord`
            The source record to convert to a dict.

        Returns
        -------
        objData : `dict`
            The dict containing the extracted data.
        """
        objData = {}
        objData['xx'] = measurementResult.xx
        objData['yy'] = measurementResult.yy
        objData['xCentroid'] = measurementResult.xCentroid
        objData['yCentroid'] = measurementResult.yCentroid
        objData['apFlux70'] = measurementResult.apFlux70
        objData['apFlux25'] = measurementResult.apFlux25
        return objData

    @staticmethod
    def _makeEmptyReturnStruct():
        """Make the default/template return struct, with defaults to False/nan.

        Returns
        -------
        objData : `lsst.pipe.base.Struct`
            The default template return structure.
        """
        result = pipeBase.Struct()
        result.success = False
        result.brightestObjCentroid = (np.nan, np.nan)
        result.brightestObjCentroidCofM = None
        result.brightestObj_xXyY = (np.nan, np.nan)
        result.brightestObjApFlux70 = np.nan
        result.brightestObjApFlux25 = np.nan
        result.medianXxYy = (np.nan, np.nan)
        return result

    def run(self, exp, *, donutDiameter=None, doDisplay=False):
        """Calculate position, flux and shape of the brightest star in an image.

        Given an an assembled (and at least minimally ISRed exposure),
        quickly and robustly calculate the centroid of the
        brightest star in the image.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            The exposure in which to find and measure the brightest star.
        donutDiameter : `int` or `float`, optional
            The expected diameter of donuts in pixels for use in the centre of
            mass centroid measurement. If None is provided, the config option
            is used.
        doDisplay : `bool`
            Display the image and found sources. A diplay object must have
            been passed to the task constructor.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Struct containing:
                Whether the task ran successfully and found the object (bool)
                The object's centroid (float, float)
                The object's ixx, iyy (float, float)
                The object's 70 pixel aperture flux (float)
                The object's 25 pixel aperture flux (float)
                The images's median ixx, iyy (float, float)
            If unsuccessful, the success field is False and all other results
            are np.nan of the expected shape.

        Notes
        -----
        Because of this task's involvement in observing scripts, the run method
        should *never* raise. Failure modes are noted by returning a Struct with
        the same structure as the success case, with all value set to np.nan and
        result.success=False.
        """
        try:
            result = self._run(exp=exp, donutDiameter=donutDiameter, doDisplay=doDisplay)
            return result
        except Exception as e:
            self.log.warning("Failed to find main source centroid %s", e)
            result = self._makeEmptyReturnStruct()
            return result

    def _run(self, exp, *, donutDiameter=None, doDisplay=False):
        """The actual run method, called by run()

        Behaviour is documented in detail in the main run().
        """
        if donutDiameter is None:
            donutDiameter = self.config.donutDiameter

        self.plateScale = exp.getWcs().getPixelScale(exp.getBBox().getCenter()).asArcseconds()
        median = np.nanmedian(exp.image.array)
        exp.image -= median  # is put back later
        self.installPsf.run(exp)
        sources = self.detectObjectsInExp(exp, nSigma=self.config.nSigmaDetection,
                                          nPixMin=self.config.nPixMinDetection)

        if doDisplay:
            if self.display is None:
                raise RuntimeError("Display failed as no display provided during init()")
            self.display.mtv(exp)

        fpSet = sources.getFootprints()
        self.log.info("Found %d sources in exposure", len(fpSet))

        objData = {}
        nMeasured = 0

        for srcNum, fp in enumerate(fpSet):
            try:
                src = self._measureFp(fp, exp)
                result = self._getDataFromSrcRecord(src)
            except MeasurementError:
                try:
                    # gets shape and centroid from footprint
                    result = self._getDataFromFootprintOnly(fp, exp)
                except MeasurementError as e:
                    self.log.info("Skipped measuring source %s: %s", srcNum, e)
                    continue
            objData[srcNum] = self._measurementResultToDict(result)
            nMeasured += 1

        self.log.info("Measured %d of %d sources in exposure", nMeasured, len(fpSet))

        medianXxYy = self._calcMedianXxYy(objData)

        brightestObjSrcNum = self._calcBrightestObjSrcNum(objData)
        if brightestObjSrcNum is None:
            raise RuntimeError("No sources in image passed cuts")

        x = objData[brightestObjSrcNum]['xCentroid']
        y = objData[brightestObjSrcNum]['yCentroid']
        brightestObjCentroid = (x, y)
        xx = objData[brightestObjSrcNum]['xx']
        yy = objData[brightestObjSrcNum]['yy']
        brightestObjApFlux70 = objData[brightestObjSrcNum]['apFlux70']
        brightestObjApFlux25 = objData[brightestObjSrcNum]['apFlux25']

        exp.image += median  # put background back in
        if self.config.doCheckCentroidPixelValue:
            self.checkResult(exp, brightestObjCentroid, brightestObjSrcNum,
                             self.config.centroidPixelPercentile)

        boxSize = donutDiameter * 1.3  # allow some slack, as cutting off side of donut is very bad
        centreOfMass = self._getCenterOfMass(exp, brightestObjCentroid, boxSize)

        result = self._makeEmptyReturnStruct()
        result.success = True
        result.brightestObjCentroid = brightestObjCentroid
        result.brightestObj_xXyY = (xx, yy)
        result.brightestObjApFlux70 = brightestObjApFlux70
        result.brightestObjApFlux25 = brightestObjApFlux25
        result.medianXxYy = medianXxYy
        result.brightestObjCentroidCofM = centreOfMass

        return result
