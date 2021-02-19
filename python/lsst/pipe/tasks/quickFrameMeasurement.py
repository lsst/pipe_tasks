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

import numpy as np
import lsst.afw.detection as afwDetect
import lsst.afw.table as afwTable
import lsst.meas.base as measBase
import lsst.daf.base as dafBase
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from lsst.meas.base import MeasurementError
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask
from lsst.meas.algorithms.installGaussianPsf import InstallGaussianPsfTask


def detectObjectsInExp(exp, nSigma, nPixMin, grow=0):
    """Return the footPrintSet for the objects in a postISR exposure."""
    threshold = afwDetect.Threshold(nSigma, afwDetect.Threshold.STDEV)
    footPrintSet = afwDetect.FootprintSet(exp.getMaskedImage(), threshold, "DETECTED", nPixMin)
    if grow > 0:
        isotropic = True
        footPrintSet = afwDetect.FootprintSet(footPrintSet, grow, isotropic)
    return footPrintSet


def checkResult(exp, centroid, percentile=90):
    """Sanity check that the centroid of the source is actually bright."""
    threshold = np.percentile(exp.image.array, percentile)
    pixelValue = exp.image[centroid]
    if pixelValue < threshold:
        raise ValueError(f"Value of brightest star central pixel = {pixelValue:3f} < "
                         f"{percentile} percentile of image = {threshold:3f}")
    return


class QuickFrameMeasurementTaskConfig(pexConfig.Config):
    imageIsDispersed = pexConfig.Field(  # XXX Doesn't seem like this should be necessary
        dtype=bool,
        doc="Is this a dispersed (spectroscopic) image?",
        default=False,
    )
    maxNonRoundness = pexConfig.Field(
        dtype=float,
        doc="Ratio of xx to yy (or vice versa) above which to cut, in order to exclude spectra",
        default=15.,
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
    initialPsfWidth = pexConfig.Field(
        dtype=float,
        doc="Guess at the initial PSF in XXX ???? pixels FWHM or sigma or something?",
        default=10,
    )
    nSigmaDetection = pexConfig.Field(
        dtype=float,
        doc="Number of sigma for the detection limit",
        default=20,
    )
    nPixMinDetection = pexConfig.Field(
        dtype=int,
        doc="Minimum number of pixels in a detected source",
        default=10,
    )
    doPrintSourceData = pexConfig.Field(  # XXX move this to debug log messages instead
        dtype=bool,
        doc="Print source data for debug?",
        default=False,
    )


class QuickFrameMeasurementTask(pipeBase.Task):
    """WARNING: Experimental new task with changable API! Do not rely on yet!

    This task finds the centroid of the brightest source in a given CCD-image
    and returns its centroid and a rough estimate of the seeing/PSF.

    It is designed for speed, such that it can be used in observing scripts
    to provide pointing offsets, allowing subsequent pointings to place
    a source at an exact pixel position.
    """
    ConfigClass = QuickFrameMeasurementTaskConfig
    _DefaultName = 'quickFrameMeasurementTask'

    def __init__(self, config, *, display=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.display = None
        if display:
            self.display = display

        psfInstallConfig = InstallGaussianPsfTask.ConfigClass()
        psfInstallConfig.fwhm = self.config.initialPsfWidth
        self.installPsfTask = InstallGaussianPsfTask(config=psfInstallConfig)

        self.centroidName = "base_SdssCentroid"
        self.shapeName = "base_SdssShape"
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.schema.getAliasMap().set("slot_Centroid", self.centroidName)
        self.schema.getAliasMap().set("slot_Shape", self.shapeName)
        self.control = measBase.SdssCentroidControl()
        self.centroider = measBase.SdssCentroidAlgorithm(self.control, self.centroidName, self.schema)
        self.sdssShape = measBase.SdssShapeControl()
        self.shaper = measBase.SdssShapeAlgorithm(self.sdssShape, self.shapeName, self.schema)
        self.apFluxControl = measBase.ApertureFluxControl()
        md = dafBase.PropertySet()
        self.apFluxer = measBase.CircularApertureFluxAlgorithm(self.apFluxControl, "aperFlux",
                                                               self.schema, md)

        # make sure to call this last!
        self.table = afwTable.SourceTable.make(self.schema)

    @staticmethod
    def _calcMedianPsf(objData):
        medianXx = np.nanmedian([objData[i]['xx'] for i in objData.keys()])
        medianYy = np.nanmedian([objData[i]['yy'] for i in objData.keys()])
        return medianXx, medianYy

    def _calcBrightestObjSrcNum(self, objData):
        max70, max70srcNum = 0, 0
        max25, max25srcNum = 0, 0

        for srcNum in sorted(objData.keys()):  # srcNum not contiguous so don't use a list comp
            skip = False
            xx = objData[srcNum]['xx']
            yy = objData[srcNum]['yy']

            # don't just skip because we want to always run for debug purposes
            # but need to protect against division by zero
            xx = max(xx, 1e-9)
            yy = max(yy, 1e-9)

            if self.config.doExtendednessCut:
                if xx > self.config.maxExtendedness or yy > self.config.maxExtendedness:
                    skip = skip or True

            nonRoundness = xx/yy
            nonRoundness = max(nonRoundness, 1/nonRoundness)
            if nonRoundness > self.config.maxNonRoundness:
                skip = skip or True

            if self.config.doPrintSourceData:
                text = f"src {srcNum}: {objData[srcNum]['xCentroid']:.0f}, {objData[srcNum]['yCentroid']:.0f}"
                text += f" - xx={xx:.1f}, yy={yy:.1f}, nonRound={nonRoundness:.1f}"
                text += f" - ap70={objData[srcNum]['apFlux70']:,.0f}"
                text += f" - ap25={objData[srcNum]['apFlux25']:,.0f}"
                text += f" - skip={skip}"
                print(text)

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
            print("WARNING! Max apFlux70 for different object than with max apFlux25")
        return max70srcNum

    def _measureFp(self, fp, exp):
        src = self.table.makeRecord()
        src.setFootprint(fp)
        self.centroider.measure(src, exp)
        self.shaper.measure(src, exp)
        self.apFluxer.measure(src, exp)
        return src

    @staticmethod
    def _getDataFromSrcRecord(src):
        xx = np.sqrt(src['base_SdssShape_xx'])*2.355*.1  # 2.355 for FWHM, .1 for platescale
        yy = np.sqrt(src['base_SdssShape_yy'])*2.355*.1
        xCentroid = src['base_SdssCentroid_x']
        yCentroid = src['base_SdssCentroid_y']
        # apFlux available: 70, 50, 35, 25, 17, 12 9, 6, 4.5, 3
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
        # TODO: the ixx iyy MUST go! (replace with just a roundness measure?)
        xx = fp.getShape().getIxx()
        yy = fp.getShape().getIyy()
        xCentroid, yCentroid = fp.getCentroid()
        # pretty gross, but we want them both to exist
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
        objData = {}
        objData['xx'] = measurementResult.xx
        objData['yy'] = measurementResult.yy
        objData['xCentroid'] = measurementResult.xCentroid
        objData['yCentroid'] = measurementResult.yCentroid
        objData['apFlux70'] = measurementResult.apFlux70
        objData['apFlux25'] = measurementResult.apFlux25
        return objData

    def run(self, exp, doDisplay=False):
        try:
            result = self._run(exp=exp, doDisplay=doDisplay)
            return result
        except Exception as e:
            raise RuntimeError("Failed to find main source centroid") from e

    def _run(self, exp, doDisplay=False):
        median = np.nanmedian(exp.image.array)
        exp.image -= median
        self.installPsfTask.run(exp)
        sources = detectObjectsInExp(exp, nSigma=self.config.nSigmaDetection,
                                     nPixMin=self.config.nPixMinDetection)

        if doDisplay:  # TODO: check if display still works
            if self.display is None:
                raise RuntimeError("Display failed as no display provided during init()")
            self.display.mtv(exp)

        fpSet = sources.getFootprints()
        self.log.info(f"Found {len(fpSet)} sources in exposure")

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
                    self.log.info(f"Skipped measuring source {srcNum}: {e}")
                    continue
            objData[srcNum] = self._measurementResultToDict(result)
            nMeasured += 1

        self.log.info(f"Measured {nMeasured} of {len(fpSet)} sources in exposure")

        medianPsf = self._calcMedianPsf(objData)

        brightestObjSrcNum = self._calcBrightestObjSrcNum(objData)
        x = objData[brightestObjSrcNum]['xCentroid']
        y = objData[brightestObjSrcNum]['yCentroid']
        brightestObjCentroid = (x, y)
        xx = objData[brightestObjSrcNum]['xx']
        yy = objData[brightestObjSrcNum]['yy']
        brightestObjApFlux70 = objData[brightestObjSrcNum]['apFlux70']
        brightestObjApFlux25 = objData[brightestObjSrcNum]['apFlux25']

        exp.image += median  # put background back in
        checkResult(exp, brightestObjCentroid)
        return pipeBase.Struct(brightestObjCentroid=brightestObjCentroid,
                               brightestObj_xXyY=(xx, yy),
                               brightestObjApFlux70=brightestObjApFlux70,
                               brightestObjApFlux25=brightestObjApFlux25,
                               medianPsf=medianPsf,)

    def runSlow(self, exp):

        imCharConfig = CharacterizeImageTask.ConfigClass()
        imCharConfig.doMeasurePsf = True
        imCharConfig.doApCorr = False
        imCharConfig.doDeblend = False

        imCharConfig.doWrite = False
        imCharConfig.doWriteExposure = False
        imCharConfig.psfIterations = 1
        imCharConfig.detection.reEstimateBackground = False
        imCharConfig.repair.doInterpolate = False
        imCharConfig.repair.doCosmicRay = False

        # imCharConfig.repair.cosmicray.nCrPixelMax = 200000
        imCharTask = CharacterizeImageTask(config=imCharConfig)
        _ = imCharTask.run(exp)

        psf = exp.getPsf()
        psfShape = psf.computeShape()

        ixx = psf.computeShape().getIxx()
        iyy = psf.computeShape().getIyy()
        ixx = np.sqrt(ixx)*2.355*.1
        iyy = np.sqrt(iyy)*2.355*.1
        print(f"Psf shape from imChar task (x,y) = ({ixx:.3f}, {iyy:.3f}) FWHM arcsec")

        return psfShape


if __name__ == '__main__':
    # import lsst.afw.image as afwImage
    # exp = afwImage.ExposureF('/home/mfl/big_repos/afwdata/LATISS/postISRCCD/postISRCCD_2020021800073-'
    #                          'KPNO_406_828nm~EMPTY-det000.fits.fz')

    dataId = {'dayObs': '2021-02-18', 'seqNum': 277}

    try:
        import lsst.daf.persistence as dafPersist
        butler = dafPersist.Butler('/project/shared/auxTel/rerun/quickLook')
        exp = butler.get('quickLookExp', **dataId)
    except Exception:
        from lsst.rapid.analysis.bestEffort import BestEffortIsr
        REPODIR = '/project/shared/auxTel/'
        bestEffort = BestEffortIsr(REPODIR)
        exp = bestEffort.getExposure(dataId)

    config = QuickFrameMeasurementTaskConfig()
    config.imageIsDispersed = False
    config.doPrintSourceData = True
    config.doExtendednessCut = False
    qfm = QuickFrameMeasurementTask(config=config)

    result = qfm.run(exp)
    print(result)
