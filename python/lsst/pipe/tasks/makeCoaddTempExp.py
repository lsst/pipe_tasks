from __future__ import division, absolute_import
#
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011, 2012 LSST Corporation.
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

import numpy; numpy.set_printoptions(linewidth=300); import pdb

import lsst.pex.config as pexConfig
import lsst.afw.image as afwImage
import lsst.afw.display as afwDisplay
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
from lsst.meas.algorithms import CoaddPsf
from .coaddBase import CoaddBaseTask
from .warpAndPsfMatch import WarpAndPsfMatchTask
from .coaddHelpers import groupPatchExposures, getGroupDataRef

__all__ = ["MakeCoaddTempExpTask"]

class MakeCoaddTempExpConfig(CoaddBaseTask.ConfigClass):
    """Config for MakeCoaddTempExpTask
    """
    warpAndPsfMatch = pexConfig.ConfigurableField(
        target = WarpAndPsfMatchTask,
        doc = "Task to warp and PSF-match calexp",
    )
    doWrite = pexConfig.Field(
        doc = "persist <coaddName>Coadd_tempExp",
        dtype = bool,
        default = True,
    )
    doOverwrite = pexConfig.Field(
        doc = "overwrite <coaddName>Coadd_tempExp; If False, continue if the file exists on disk",
        dtype = bool,
        default = True,
    )
    bgSubtracted = pexConfig.Field(
        doc = "Work with a background subtracted calexp?",
        dtype = bool,
        default = True,
    )


class CovView(object):
    """Class to aid in visualizing covariance matrix
    """

    def __init__(self, destImage, covImage, covName=''):
        self.destImage = destImage
        self.covArr = covImage.getArray()
        self.destKernelWidth = covImage.getWidth()/destImage.getWidth()
        self.destKernelHeight = covImage.getHeight()/destImage.getHeight()
        self.covName = covName

    def __getitem__(self, (loc1X, loc1Y, loc2X, loc2Y)):
        if loc1X < loc2X or loc1Y < loc2Y:
            loc2X, loc1X = loc1X, loc2X
            loc2Y, loc1Y = loc1Y, loc2Y
        if loc2X - loc1X > self.destKernelWidth or loc2Y - loc1Y > self.destKernelHeight:
            val = 0.0
        else:
            val = self.covArr[int(loc1X*self.destKernelWidth) + (loc2X - loc1X),
                              int(loc1Y*self.destKernelHeight) + (loc2Y - loc1Y)]
        return val

    def setName(covName):
        self.covName = covName


class MakeCoaddTempExpTask(CoaddBaseTask):
    """Task to produce <coaddName>Coadd_tempExp images
    """
    ConfigClass = MakeCoaddTempExpConfig
    _DefaultName = "makeCoaddTempExp"

    def __init__(self, *args, **kwargs):
        CoaddBaseTask.__init__(self, *args, **kwargs)
        self.makeSubtask("warpAndPsfMatch")

    @pipeBase.timeMethod
    def run(self, patchRef, selectDataList=[]):
        """Produce <coaddName>Coadd_tempExp images

        <coaddName>Coadd_tempExp are produced by PSF-matching (optional) and warping.

        @param[in] patchRef: data reference for sky map patch. Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter" or "band")
        @return: dataRefList: a list of data references for the new <coaddName>Coadd_tempExp

        @warning: this task assumes that all exposures in a coaddTempExp have the same filter.

        @warning: this task sets the Calib of the coaddTempExp to the Calib of the first calexp
        with any good pixels in the patch. For a mosaic camera the resulting Calib should be ignored
        (assembleCoadd should determine zeropoint scaling without referring to it).
        """
        skyInfo = self.getSkyInfo(patchRef)

        calExpRefList = self.selectExposures(patchRef, skyInfo, selectDataList=selectDataList)
        if len(calExpRefList) == 0:
            self.log.warn("No exposures to coadd for patch %s", patchRef.dataId)
            return None
        self.log.info("Selected %d calexps for patch %s", len(calExpRefList), patchRef.dataId)
        calExpRefList = [calExpRef for calExpRef in calExpRefList if calExpRef.datasetExists("calexp")]
        self.log.info("Processing %d existing calexps for patch %s", len(calExpRefList), patchRef.dataId)

        groupData = groupPatchExposures(patchRef, calExpRefList, self.getCoaddDatasetName(),
                                        self.getTempExpDatasetName())
        self.log.info("Processing %d tempExps for patch %s", len(groupData.groups), patchRef.dataId)

        dataRefList = []
        for i, (tempExpTuple, calexpRefList) in enumerate(groupData.groups.iteritems()):
            tempExpRef = getGroupDataRef(patchRef.getButler(), self.getTempExpDatasetName(),
                                         tempExpTuple, groupData.keys)
            if not self.config.doOverwrite and tempExpRef.datasetExists(datasetType=self.getTempExpDatasetName()):
                self.log.info("tempCoaddExp %s exists; skipping", tempExpRef.dataId)
                dataRefList.append(tempExpRef)
                continue
            self.log.info("Processing tempExp %d/%d: id=%s", i, len(groupData.groups), tempExpRef.dataId)

            # TODO: mappers should define a way to go from the "grouping keys" to a numeric ID (#2776).
            # For now, we try to get a long integer "visit" key, and if we can't, we just use the index
            # of the visit in the list.
            try:
                visitId = long(tempExpRef.dataId["visit"])
            except (KeyError, ValueError):
                visitId = i

            res = self.createTempExp(calexpRefList, skyInfo, visitId)
            if res is not None:
                exp, cov = res
                dataRefList.append(tempExpRef)
                if self.config.doWrite:
                    self.writeCoaddOutput(tempExpRef, exp, "tempExp", cov)
            else:
                self.log.warn("tempExp %s could not be created", tempExpRef.dataId)
        return dataRefList

    def _getCovMultiplier(self, calexpRefList, modelPsf, skyInfo):
        multX = 0
        multY = 0
        for calExpInd, calExpRef in enumerate(calexpRefList):
            try:
                ccdId = calExpRef.get("ccdExposureId", immediate=True)
            except Exception:
                ccdId = calExpInd
            calExpRef = calExpRef.butlerSubset.butler.dataRef("calexp", dataId=calExpRef.dataId,
                                                              tract=skyInfo.tractInfo.getId())
            calExp = self.getCalExp(calExpRef, bgSubtracted=self.config.bgSubtracted)
            warpRes = self.warpAndPsfMatch.run(calExp, modelPsf=modelPsf, wcs=skyInfo.wcs,
                                               maxBBox=skyInfo.bbox)
            exposure = warpRes.exposure
            covImage = warpRes.covImage
            try:
                multX = int(covImage.getWidth()/exposure.getWidth())
                multY = int(covImage.getHeight()/exposure.getHeight())
            except ZeroDivisionError:
                continue
            else:
                break
        return (multX, multY)

    def createTempExp(self, calexpRefList, skyInfo, visitId=0):
        """Create a tempExp from inputs

        We iterate over the multiple calexps in a single exposure to construct
        the warp ("tempExp") of that exposure to the supplied tract/patch.

        Pixels that receive no pixels are set to NAN; this is not correct
        (violates LSST algorithms group policy), but will be fixed up by
        interpolating after the coaddition.

        @param calexpRefList: List of data references for calexps that (may)
            overlap the patch of interest
        @param skyInfo: Struct from CoaddBaseTask.getSkyInfo() with geometric
            information about the patch
        @param visitId: integer identifier for visit, for the table that will
            produce the CoaddPsf
        @return warped exposure, or None if no pixels overlap
        """
        inputRecorder = self.inputRecorder.makeCoaddTempExpRecorder(visitId, len(calexpRefList))
        coaddTempExp = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
        coaddTempExp.getMaskedImage().set(numpy.nan, afwImage.MaskU.getPlaneBitMask("NO_DATA"), numpy.inf)
        # FOR DEBUG
        '''coaddTempExp_orig = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
        coaddTempExp_orig.getMaskedImage().set(numpy.nan, afwImage.MaskU.getPlaneBitMask("NO_DATA"),
                                               numpy.inf)
        DEBUG_OLDALG = TRUE'''
        # END DEBUG
        totGoodPix = 0
        didSetMetadata = False
        modelPsf = self.config.modelPsf.apply() if self.config.doPsfMatch else None
        multX, multY = self._getCovMultiplier(calexpRefList, modelPsf, skyInfo)
        coaddTempCov = afwImage.ImageD(coaddTempExp.getWidth()*multX, coaddTempExp.getHeight()*multY,
                                       numpy.inf)
        for calExpInd, calExpRef in enumerate(calexpRefList):
            self.log.info("Processing calexp %d of %d for this tempExp: id=%s",
                          calExpInd+1, len(calexpRefList), calExpRef.dataId)
            try:
                ccdId = calExpRef.get("ccdExposureId", immediate=True)
            except Exception:
                ccdId = calExpInd
            numGoodPix = 0
            try:
                # We augment the dataRef here with the tract, which is harmless for loading things
                # like calexps that don't need the tract, and necessary for meas_mosaic outputs,
                # which do.
                calExpRef = calExpRef.butlerSubset.butler.dataRef("calexp", dataId=calExpRef.dataId,
                                                                  tract=skyInfo.tractInfo.getId())
                calExp = self.getCalExp(calExpRef, bgSubtracted=self.config.bgSubtracted)
                covName = 'covar'
                for key, val in calExpRef.dataId.items():
                    covName += '_%s_%s'%(key, val)
                warpRes = self.warpAndPsfMatch.run(calExp, modelPsf=modelPsf, wcs=skyInfo.wcs,
                                                   maxBBox=skyInfo.bbox, multX=multX, multY=multY)
                exposure = warpRes.exposure
                covImage = warpRes.covImage
                # FOR DEBUG
                '''if exposure.getHeight() != 0 and exposure.getWidth() != 0:
                    afwDisplay.getDisplay(0).mtv(exposure.getMaskedImage().getImage())
                    afwDisplay.getDisplay(1).mtv(calExp.getMaskedImage().getImage())
                    afwDisplay.getDisplay(2).mtv(calExp.getMaskedImage().getVariance())
                    afwDisplay.getDisplay(3).mtv(exposure.getMaskedImage().getVariance())
                    afwDisplay.getDisplay(4).mtv(covImage)
                    view = covView(exposure, covImage, covName=covName)
                    destArr = exposure.getMaskedImage().getImage().getArray()
                    import pdb
                    pdb.set_trace()'''
                # END DEBUG
                if didSetMetadata:
                    mimg = exposure.getMaskedImage()
                    mimg *= (coaddTempExp.getCalib().getFluxMag0()[0] / exposure.getCalib().getFluxMag0()[0])
                    del mimg
                numGoodPix = coaddUtils.copyGoodPixels(
                    coaddTempExp.getMaskedImage(), exposure.getMaskedImage(), self.getBadPixelMask(),
                    coaddTempCov, covImage)
                # FOR DEBUG
                '''if DEBUG_OLDALG:
                    numGoodPix_orig = coaddUtils.copyGoodPixels(
                        coaddTempExp_orig.getMaskedImage(), exposure.getMaskedImage(), self.getBadPixelMask())
                    print "numGoodPix_orig: %d"%(numGoodPix_orig)
                    afwDisplay.getDisplay(0).mtv(coaddTempExp_orig.getMaskedImage().getImage())
                    afwDisplay.getDisplay(1).mtv(coaddTempExp_orig.getMaskedImage().getImage())
                print "numGoodPix: %d"%(numGoodPix)
                afwDisplay.getDisplay(2).mtv(coaddTempExp.getMaskedImage().getVariance())
                afwDisplay.getDisplay(3).mtv(coaddTempExp.getMaskedImage().getVariance())
                afwDisplay.getDisplay(4).mtv(covImage)
                afwDisplay.getDisplay(5).mtv(coaddTempCov)
                import pdb
                pdb.set_trace()'''
                # END DEBUG
                totGoodPix += numGoodPix
                self.log.debug("Calexp %s has %d good pixels in this patch (%.1f%%)",
                               calExpRef.dataId, numGoodPix, 100.0*numGoodPix/skyInfo.bbox.getArea())
                if numGoodPix > 0 and not didSetMetadata:
                    coaddTempExp.setCalib(exposure.getCalib())
                    coaddTempExp.setFilter(exposure.getFilter())
                    didSetMetadata = True
            except Exception, e:
                self.log.warn("Error processing calexp %s; skipping it: %s", calExpRef.dataId, e)
                continue
            inputRecorder.addCalExp(calExp, ccdId, numGoodPix)

        inputRecorder.finish(coaddTempExp, totGoodPix)
        if totGoodPix > 0 and didSetMetadata:
            coaddTempExp.setPsf(modelPsf if self.config.doPsfMatch else
                                CoaddPsf(inputRecorder.coaddInputs.ccds, skyInfo.wcs))

        self.log.info("coaddTempExp has %d good pixels (%.1f%%)",
                      totGoodPix, 100.0*totGoodPix/skyInfo.bbox.getArea())
        return (coaddTempExp, coaddTempCov) if totGoodPix > 0 and didSetMetadata else None
