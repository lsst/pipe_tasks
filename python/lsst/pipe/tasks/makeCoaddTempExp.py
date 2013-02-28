#!/usr/bin/env python
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
import math

import numpy

import lsst.pex.config as pexConfig
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
from .coaddBase import CoaddBaseTask
from .warpAndPsfMatch import WarpAndPsfMatchTask
from .coaddHelpers import groupPatchExposures, getGroupDataRef

__all__ = ["MakeCoaddTempExpTask"]

class MakeCoaddTempExpConfig(CoaddBaseTask.ConfigClass):
    """Config for MakeCoaddTempExpTask
    """
    desiredFwhm = pexConfig.Field(
        doc = "desired FWHM of coadd (arc seconds); None for no FWHM matching",
        dtype = float,
        optional = True,
        check = lambda x: x is None or x > 0.0,
    )
    coaddKernelSizeFactor = pexConfig.Field(
        dtype = float,
        doc = "coadd kernel size = coadd FWHM converted to pixels * coaddKernelSizeFactor",
        default = 3.0,
    )
    warpAndPsfMatch = pexConfig.ConfigurableField(
        target = WarpAndPsfMatchTask,
        doc = "Task to warp, PSF-match and zero-point-match calexp",
    )
    doWrite = pexConfig.Field(
        doc = "persist <coaddName>Coadd_tempExp and (if desiredFwhm not None) <coaddName>Coadd_initPsf?",
        dtype = bool,
        default = True,
    )
    doOverwrite = pexConfig.Field(
        doc = "overwrite <coaddName>Coadd_tempExp and (if desiredFwhm not None) <coaddName>Coadd_initPsf?  If False, continue if the file exists on disk",
        dtype = bool,
        default = True,
    )
    bgSubtracted = pexConfig.Field(
        doc = "Work with a background subtracted calexp?",
        dtype = bool,
        default = False,
    )


class MakeCoaddTempExpTask(CoaddBaseTask):
    """Task to produce <coaddName>Coadd_tempExp images and (optional) <coaddName>Coadd_initPsf
    """
    ConfigClass = MakeCoaddTempExpConfig
    _DefaultName = "makeCoaddTempExp"
    
    def __init__(self, *args, **kwargs):
        CoaddBaseTask.__init__(self, *args, **kwargs)
        self.makeSubtask("warpAndPsfMatch")

    @pipeBase.timeMethod
    def run(self, patchRef, selectDataList=[]):
        """Produce <coaddName>Coadd_tempExp images and (optional) <coaddName>Coadd_initPsf
        
        <coaddName>Coadd_tempExp are produced by PSF-matching (optional) and warping.
        If PSF-matching is used then <coaddName>Coadd_initPsf is also computed.
        
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
            self.log.warn("No exposures to coadd for patch %s" % patchRef.dataId)
            return None
        self.log.info("Processing %d calexps for patch %s" % (len(calExpRefList), patchRef.dataId))

        groupData = groupPatchExposures(patchRef, calExpRefList, self.getCoaddDataset(),
                                        self.getTempExpDataset())
        self.log.info("Processing %d tempExps for patch %s" % (len(groupData.groups), patchRef.dataId))

        dataRefList = []
        for i, (tempExpTuple, calexpRefList) in enumerate(groupData.groups.iteritems()):
            tempExpRef = getGroupDataRef(patchRef.getButler(), self.getTempExpDataset(),
                                         tempExpTuple, groupData.keys)
            if not self.config.doOverwrite and tempExpRef.datasetExists(datasetType=tempExpName):
                self.log.info("tempCoaddExp %s exists; skipping" % (tempExpRef.dataId,))
                dataRefList.append(tempExpRef)
                continue
            self.log.info("Processing tempExp %d/%d: id=%s" % (i, len(groupData.groups), tempExpRef.dataId))
            exp = self.createTempExp(calexpRefList, skyInfo)
            if exp is not None:
                dataRefList.append(tempExpRef)
                if self.config.doWrite:
                    self.writeCoaddOutput(tempExpRef, exp, "tempExp")
                    if self.config.desiredFwhm is not None:
                        psf = self.makeModelPsf(fwhm=self.config.desiredFwhm, wcs=wcs,
                                                sizeFactor=self.config.coaddKernelSizeFactor)
                        self.writeCoaddOutput(patchRef, psf, "initPsf")

            else:
                self.log.warn("tempExp %s could not be created" % (tempExpRef.dataId,))
        return dataRefList

    def createTempExp(self, calexpRefList, skyInfo):
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
        @return warped exposure, or None if no pixels overlap
        """
        coaddTempExp = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
        edgeMask = afwImage.MaskU.getPlaneBitMask("EDGE")
        coaddTempExp.getMaskedImage().set(numpy.nan, edgeMask, numpy.inf) # XXX these are the wrong values!
        totGoodPix = 0
        didSetMetadata = False
        modelPsf = self.makeModelPsf(fwhm=self.config.desiredFwhm, wcs=skyInfo.wcs,
                                     sizeFactor=self.config.coaddKernelSizeFactor)
        for calExpInd, calExpRef in enumerate(calexpRefList):
            self.log.info("Processing calexp %d of %d for this tempExp: id=%s" %
                          (calExpInd+1, len(calexpRefList), calExpRef.dataId))
            try:
                exposure = self.getCalExp(calExpRef, getPsf=self.config.desiredFwhm is not None,
                                          bgSubtracted=self.config.bgSubtracted)
                exposure = self.warpAndPsfMatch.run(exposure, modelPsf=modelPsf, wcs=skyInfo.wcs,
                                                    maxBBox=skyInfo.bbox).exposure
                numGoodPix = coaddUtils.copyGoodPixels(
                    coaddTempExp.getMaskedImage(), exposure.getMaskedImage(), self.getBadPixelMask())
                totGoodPix += numGoodPix
                self.log.logdebug("Calexp %s has %d good pixels in this patch" %
                                  (calExpRef.dataId, numGoodPix))
                if numGoodPix > 0 and not didSetMetadata:
                    coaddTempExp.setCalib(exposure.getCalib())
                    coaddTempExp.setFilter(exposure.getFilter())
                    didSetMetadata = True
            except Exception, e:
                self.log.warn("Error processing calexp %s; skipping it: %s" % (calExpRef.dataId, e))
                continue

        self.log.info("coaddTempExp has %d good pixels" % (totGoodPix))
        return coaddTempExp if totGoodPix > 0 and didSetMetadata else None
