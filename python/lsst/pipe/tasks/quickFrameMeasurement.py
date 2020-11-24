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
import lsst.afw.table as afwTable
import lsst.meas.base as measBase
import lsst.daf.base as dafBase
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
from lsst.meas.base import MeasurementError
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask
from lsst.meas.algorithms.installGaussianPsf import InstallGaussianPsfTask
import lsst.afw.display as afwDisplay

from lsst.rapid.analysis.utils import detectObjectsInExp

MAX_NON_ROUNDNESS = 3.5  # spectra tend to be >25 and stars are often around 1.3


class QuickFrameMeasurementConfig(pexConfig.Config):
    imageIsDispersed = pexConfig.Field(
        dtype=bool,
        doc="Is this a dispersed (spectroscopic) image?",
        default=False,
    )


class QuickFrameMeasurement(pipeBase.Task):
    """WARNING: Experimental new task with changable API! Do not rely on yet!

    This task finds the centroid of the brightest source in a given CCD-image
    and returns its centroid and a rough estimate of the seeing/PSF.

    It is designed for speed, such that it can be used in observing scripts
    to provide pointing offsets, allowing subsequent pointings to place
    a source at an exact pixel position.
    """

    def __init__(self, initialPsfWidth=20, display=None, **kwargs):
        self.display = None
        if display:
            self.display = display

        psfInstallConfig = InstallGaussianPsfTask.ConfigClass()
        psfInstallConfig.fwhm = initialPsfWidth
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

    def _getDayObsSeqNumFromExpId(self, expId):
        return self.butler.queryMetadata('raw', ['dayObs', 'seqNum'], expId=expId)[0]

    @staticmethod
    def _calcMedianPsf(objData):
        medianXx = np.nanmedian([objData[i]['xx'] for i in objData.keys()])
        medianYy = np.nanmedian([objData[i]['yy'] for i in objData.keys()])
        return medianXx, medianYy

    @staticmethod
    def _calcBrightestObjSrcNum(objData):
        max70, max70srcNum = 0, 0
        max25, max25srcNum = 0, 0
        # import ipdb as pdb; pdb.set_trace()
        for srcNum in sorted(objData.keys()):  # srcNum not contiguous so don't use a list comp
            nonRoundness = objData[srcNum]['xx']/objData[srcNum]['yy']
            nonRoundness = max(nonRoundness, 1/nonRoundness)
            if nonRoundness > MAX_NON_ROUNDNESS:
                continue  # skip very unround things

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

    def run(self, exp, nSigma=20, doDisplay=False):
        median = np.nanmedian(exp.image.array)
        exp.image -= median
        self.installPsfTask.run(exp)
        sources = detectObjectsInExp(exp, nSigma=nSigma)
        if doDisplay:
            if self.display is None:
                raise RuntimeError("Display failed as no display provided during init()")
            self.display.mtv(exp)

        fpSet = sources.getFootprints()
        print(f"Found {len(fpSet)} sources in exposure")

        objData = {}
        nMeasured = 0
        for srcNum, fp in enumerate(fpSet):
            try:
                src = self._measureFp(fp, exp)
                result = self._getDataFromSrcRecord(src)
                objData[srcNum] = self._measurementResultToDict(result)
                nMeasured += 1
                if doDisplay:  # TODO: Add buffering? Messier due to optional display
                    self.display.dot(src.getShape(), *src.getCentroid(), ctype=afwDisplay.BLUE)
            except MeasurementError:
                try:
                    # gets shape and centroid from footprint
                    result = self._getDataFromFootprintOnly(fp, exp)
                    objData[srcNum] = self._measurementResultToDict(result)
                    nMeasured += 1
                except MeasurementError as e:
                    print(f"Skipped measuring source {srcNum}: {e}")
                pass

        print(f"Measured {nMeasured} of {len(fpSet)} sources in exposure")

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
        imCharConfig.repair.cosmicray.nCrPixelMax = 200000
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
    import lsst.afw.image as afwImage
    exp = afwImage.ExposureF('/home/mfl/big_repos/afwdata/LATISS/postISRCCD/temp/postISRCCD_2020021800073-KPNO_406_828nm~EMPTY-det000.fits.fz')
    
    config = QuickFrameMeasurementConfig()
    config.imageIsDispersed = False
    qfm = QuickFrameMeasurement(config=config)

    qfm.run(exp)
    
    # from lsst.rapid.analysis.bestEffort import BestEffortIsr
    # REPODIR = '/project/shared/auxTel/'
    # bestEffort = BestEffortIsr(REPODIR)
    # dataId = {'dayObs': '2020-02-18', 'seqNum': 82}  # undispersed
    # exp = bestEffort.getExposure(dataId)
    # qm = QuickFrameMeasurement()
    # result = qm.run(exp)
    # print(result)
    # expectedCentroid = (1534.98, 1497.54)  # for sigma=20
    # assert abs(result.brightestObjCentroid[0] - expectedCentroid[0]) < 2
    # assert abs(result.brightestObjCentroid[1] - expectedCentroid[1]) < 2

    # dataId = {'dayObs': '2020-03-12', 'seqNum': 319}  # dispersed
    # exp = bestEffort.getExposure(dataId)
    # qm = QuickFrameMeasurement()
    # result = qm.run(exp)
    # print(result)
    # expectedCentroid = (1788.08, 1670.57)  # for sigma=20
    # assert abs(result.brightestObjCentroid[0] - expectedCentroid[0]) < 2
    # assert abs(result.brightestObjCentroid[1] - expectedCentroid[1]) < 2
