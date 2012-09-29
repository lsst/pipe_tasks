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

import lsst.pex.config as pexConfig
import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
import lsst.ip.isr as ipIsr
from lsst.ip.diffim import ModelPsfMatchTask
from lsst.pipe.tasks.selectImages import BadSelectImagesTask

__all__ = ["MakeCoaddTempExpTask", "CoaddArgumentParser"]

FWHMPerSigma = 2 * math.sqrt(2 * math.log(2))

class MakeCoaddTempExpConfig(pexConfig.Config):
    """Config for MakeCoaddTempExpTask
    """
    coaddName = pexConfig.Field(
        doc = "coadd name: typically one of deep or goodSeeing",
        dtype = str,
        default = "deep",
    )
    select = pexConfig.ConfigurableField(
        doc = "image selection subtask",
        target = BadSelectImagesTask,
    )
    desiredFwhm = pexConfig.Field(
        doc = "desired FWHM of coadd (arc seconds); None for no FWHM matching",
        dtype = float,
        optional = True,
    )
    coaddZeroPoint = pexConfig.Field(
        dtype = float,
        doc = "photometric zero point of coadd (mag)",
        default = 27.0,
    )
    psfMatch = pexConfig.ConfigurableField(
        target = ModelPsfMatchTask,
        doc = "PSF matching model to model task",
    )
    warp = pexConfig.ConfigField(
        dtype = afwMath.Warper.ConfigClass,
        doc = "warper configuration",
    )
    badMaskPlanes = pexConfig.ListField(
        dtype = str,
        doc = "mask planes that, if set, the associated pixel should not be included in the coaddTempExp",
        default = ("EDGE", "SAT"),
    )
    consolidateKeys = pexConfig.ListField(
        dtype = str,
        doc = "data ID keys to consolidate on a single temporary exposure. " \
            "This is intended for mosaic cameras where there is sure to be no overlap; " \
            "for example LSST consolidates raft and sensor data" \
            "Warning: if you specify the wrong value the coadd temp exposure cannot be persisted",
        optional = True,
    )
    doWrite = pexConfig.Field(
        doc = "persist <coaddName>Coadd_tempExp and (if desiredFwhm not None) <coaddName>Coadd_initPsf?",
        dtype = bool,
        default = True,
    )


class MakeCoaddTempExpTask(pipeBase.CmdLineTask):
    """Coadd temporary images by PSF-matching (optional), warping and computing a weighted sum
    """
    ConfigClass = MakeCoaddTempExpConfig
    _DefaultName = "makeCoaddTempExp"
    
    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("select")
        self.makeSubtask("psfMatch")
        self.warper = afwMath.Warper.fromConfig(self.config.warp)
        self.zeroPointScaler = coaddUtils.ZeroPointScaler(self.config.coaddZeroPoint)
        self._badPixelMask = afwImage.MaskU.getPlaneBitMask(self.config.badMaskPlanes)

    @pipeBase.timeMethod
    def run(self, patchRef):
        """Produce <coaddName>Coadd_tempExp images and (optional) <coaddName>Coadd_initPsf
        
        <coaddName>Coadd_tempExp are produced by PSF-matching (optional) and warping.
        If PSF-matching is used then <coaddName>Coadd_initPsf is also computed.
        
        PSF matching is to a double gaussian model with core FWHM = self.config.desiredFwhm
        and wings of amplitude 1/10 of core and FWHM = 2.5 * core.
        The size of the PSF matching kernel is the same as the size of the kernel
        found in the first calibrated science exposure, since there is no benefit
        to making it any other size.
        
        PSF-matching is performed before warping so the code can use the PSF models
        associated with the calibrated science exposures (without having to warp those models).
    
        @param[in] patchRef: data reference for sky map patch. Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter" or "band")
        @return: a pipeBase.Struct with fields:
        - dataRefList: a list of data references for the new <coaddName>Coadd_tempExp
        """
        skyInfo = self.getSkyInfo(patchRef)
        
        tractWcs = skyInfo.wcs
        patchBBox = skyInfo.bbox
        print "patchBBox=", patchBBox
        
        calExpRefList = self.selectExposures(patchRef=patchRef, wcs=tractWcs, bbox=patchBBox)
        
        # initialize outputs
        dataRefList = []
        
        numExp = len(calExpRefList)
        if numExp < 1:
            raise pipeBase.TaskError("No exposures to coadd")
        self.log.log(self.log.INFO, "Coadd %s calexp" % (numExp,))
    
        doPsfMatch = self.config.desiredFwhm is not None
        if not doPsfMatch:
            self.log.log(self.log.INFO, "No PSF matching will be done (desiredFwhm is None)")

        tempExpName = self.config.coaddName + "Coadd_tempExp"
        # a dict of tempExp ID: full calexp ID
        # where partial tempExp ID excludes tract and patch
        # (it is just the components that can be gleaned from calexp)
        tempExpIdDict = dict()
        if self.config.consolidateKeys:
            consolidateKeySet = set(self.config.consolidateKeys)
        else:
            consolidateKeySet = set()

        numCalExp = 0
        firstCalExpId = calExpRefList[0].dataId
        consolidateKeySet = set(self.config.consolidateKeys)
        calExpKeySet = set(firstCalExpId.keys())
        if consolidateKeySet - calExpKeySet:
            raise RuntimeError("The following key(s) in self.config.consolidateKeys were not found: %s" % \
                (sorted(tuple(consolidateKeySet - calExpKeySet)),))
        
        # tempKeyList is a tuple of ID key names in a calExpId that identify a coaddTempExp;
        # this is all calExpId key names except those in consolidatedKeys;
        # note that you must also specify tract and patch to make a compete coaddTempExp ID
        tempExpKeyList = tuple(sorted(calExpKeySet - consolidateKeySet))

        # compute tempExpIdDict, a dict whose:
        # - keys are tuples of coaddTempExp ID values in tempKeyList order
        # - values are a list of calExp data references for calExp that belong in this coaddTempExp
        for calExpRef in calExpRefList:
            calExpId = calExpRef.dataId
            if not calExpRef.datasetExists("calexp"):
                self.log.warn("Could not find calexp %s; skipping it" % (calExpId,))
                continue
            
            numCalExp += 1
            tempExpIdTuple = tuple(calExpId[key] for key in tempExpKeyList)
            calExpSubsetRefList = tempExpIdDict.get(tempExpIdTuple)
            if calExpSubsetRefList:
                calExpSubsetRefList.append(calExpRef)
            else:
                tempExpIdDict[tempExpIdTuple] = [calExpRef]

        numTempExp = len(tempExpIdDict)
        for tempExpInd, calExpSubsetRefList in enumerate(tempExpIdDict.itervalues()):
            # derive tempExpId from the first calExpId
            tempExpId = dict((key, calExpSubsetRefList[0].dataId[key]) for key in tempExpKeyList)
            tempExpId.update(patchRef.dataId)
            tempExpRef = calExpRef.butlerSubset.butler.dataRef(
                datasetType = tempExpName,
                dataId = tempExpId,
            )
            self.log.log(self.log.INFO, "Computing coaddTempExp %d of %d: id=%s" % \
                (tempExpInd+1, numTempExp, tempExpId))

            for calExpInd, calExpRef in enumerate(calExpSubsetRefList):
                self.log.log(self.log.INFO, "Processing calexp %d of %d for this tempExp: id=%s" % \
                    (calExpInd+1, len(calExpSubsetRefList), calExpRef.dataId))
                calexp = self.getCalExp(calExpRef, getPsf=doPsfMatch)
                try:
                    if calExpInd == 0:
                        # make a full-sized exposure and use it as the coaddTempExp
                        coaddTempExp = self.processCalexp(calexp, wcs=tractWcs, destBBox=patchBBox)
                    else:
                        # make as small an exposure within coaddTempExp as possible
                        exposure = self.processCalexp(calexp, wcs=tractWcs, maxBBox=patchBBox)
                except Exception, e:
                    self.log.warn("Error processing calexp %s; skipping it: %s" % \
                        (calExpRef.dataId, e))
                    continue
                if calExpInd > 0:
                    numGoodPix = coaddUtils.copyGoodPixels(
                        coaddTempExp.getMaskedImage(), exposure.getMaskedImage(), self._badPixelMask)
                    if numGoodPix == 0:
                        self.log.warn("Calexp %s has no good pixels in this patch" % \
                            (calExpRef.dataId))
                    else:
                        self.log.info("Calexp %s has %s good pixels in this patch" % (numGoodPix,))
                
            if self.config.doWrite:
                tempExpRef.put(coaddTempExp, tempExpName)
                if self.config.desiredFwhm is not None:
                    psfName = self.config.coaddName + "Coadd_initPsf"
                    self.log.info("Persisting %s" % (psfName,))
                    wcs = coaddExposure.getWcs()
                    fwhmPixels = self.config.desiredFwhm / wcs.pixelScale().asArcseconds()
                    kernelSize = int(round(fwhmPixels * self.config.coaddKernelSizeFactor))
                    kernelDim = afwGeom.Point2I(kernelSize, kernelSize)
                    coaddPsf = self.makeModelPsf(fwhmPixels=fwhmPixels, kernelDim=kernelDim)
                    patchRef.put(coaddPsf, psfName)

            dataRefList.append(tempExpRef)
        
        return pipeBase.Struct(
            dataRefList = dataRefList,
        )
    
    def selectExposures(self, patchRef, wcs, bbox):
        """Select exposures to coadd
        
        @param patchRef: data reference for sky map patch. Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter" or "band")
        @param[in] wcs: WCS of coadd patch
        @param[in] bbox: bbox of coadd patch
        @return a list of science exposures to coadd, as butler data references
        """
        cornerPosList = afwGeom.Box2D(bbox).getCorners()
        coordList = [wcs.pixelToSky(pos) for pos in cornerPosList]
        return self.select.runDataRef(patchRef, coordList).dataRefList
    
    def getCalExp(self, dataRef, getPsf=True):
        """Return one "calexp" calibrated exposure, perhaps with psf
        
        @param dataRef: a sensor-level data reference
        @param getPsf: include the PSF?
        @return calibrated exposure with psf
        """
        exposure = dataRef.get("calexp")
        if getPsf:
            psf = dataRef.get("psf")
            exposure.setPsf(psf)
        return exposure

    def getSkyInfo(self, patchRef):
        """Return SkyMap, tract and patch

        @param patchRef: data reference for sky map. Must include keys "tract" and "patch"
        
        @return pipe_base Struct containing:
        - skyMap: sky map
        - tractInfo: information for chosen tract of sky map
        - patchInfo: information about chosen patch of tract
        - wcs: WCS of tract
        - bbox: outer bbox of patch, as an afwGeom Box2I
        """
        skyMap = patchRef.get(self.config.coaddName + "Coadd_skyMap")
        tractId = patchRef.dataId["tract"]
        tractInfo = skyMap[tractId]

        # patch format is "xIndex,yIndex"
        patchIndex = tuple(int(i) for i in patchRef.dataId["patch"].split(","))
        patchInfo = tractInfo.getPatchInfo(patchIndex)
        
        return pipeBase.Struct(
            skyMap = skyMap,
            tractInfo = tractInfo,
            patchInfo = patchInfo,
            wcs = tractInfo.getWcs(),
            bbox = patchInfo.getOuterBBox(),
        )

    def makeModelPsf(self, fwhmPixels, kernelDim):
        """Construct a model PSF, or reuse the prior model, if possible
        
        The model PSF is a double Gaussian with core FWHM = fwhmPixels
        and wings of amplitude 1/10 of core and FWHM = 2.5 * core.
        
        @param fwhmPixels: desired FWHM of core Gaussian, in pixels
        @param kernelDim: desired dimensions of PSF kernel, in pixels
        @return model PSF
        """
        self.log.log(self.log.DEBUG,
            "Create double Gaussian PSF model with core fwhm %0.1f pixels and size %dx%d" % \
            (fwhmPixels, kernelDim[0], kernelDim[1]))
        coreSigma = fwhmPixels / FWHMPerSigma
        return afwDetection.createPsf("DoubleGaussian", kernelDim[0], kernelDim[1],
            coreSigma, coreSigma * 2.5, 0.1)
    
    def processCalexp(self, exposure, wcs, maxBBox=None, destBBox=None):
        """PSF-match exposure (if self.config.desiredFwhm is not None), warp and scale
        
        @param[in,out] exposure: exposure to preprocess; FWHM fitting is done in place
        @param[in] wcs: desired WCS of temporary images
        @param maxBBox: maximum allowed parent bbox of warped exposure (an afwGeom.Box2I or None);
            if None then the warped exposure will be just big enough to contain all warped pixels;
            if provided then the warped exposure may be smaller, and so missing some warped pixels;
            ignored if destBBox is not None
        @param destBBox: exact parent bbox of warped exposure (an afwGeom.Box2I or None);
            if None then maxBBox is used to determine the bbox, otherwise maxBBox is ignored
        
        @return preprocessed exposure
        """
        if self.config.desiredFwhm is not None:
            self.log.log(self.log.INFO, "PSF-match exposure")
            fwhmPixels = self.config.desiredFwhm / wcs.pixelScale().asArcseconds()
            kernelDim = exposure.getPsf().getKernel().getDimensions()
            modelPsf = self.makeModelPsf(fwhmPixels=fwhmPixels, kernelDim=kernelDim)
            exposure = self.psfMatch.run(exposure, modelPsf).psfMatchedExposure
        self.log.log(self.log.INFO, "Warp exposure")
        with self.timer("warp"):
            exposure = self.warper.warpExposure(wcs, exposure, maxBBox=maxBBox, destBBox=destBBox)
        
        self.zeroPointScaler.scaleExposure(exposure)

        return exposure
    
    def postprocessCoadd(self, coaddExposure):
        """Postprocess the exposure coadd, e.g. interpolate over edge pixels
        """
        if self.config.doInterp:
            self.interpolateEdgePixels(exposure=coaddExposure)
        else:
            self.log.log(self.log.INFO, "config.doInterp is None; do not interpolate over EDGE pixels")

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        return CoaddArgumentParser(name=cls._DefaultName, datasetType="deepCoadd")

    def _getConfigName(self):
        """Return the name of the config dataset
        """
        return "%s_makeCoaddTempExp_config" % (self.config.coaddName,)
    
    def _getMetadataName(self):
        """Return the name of the metadata dataset
        """
        return "%s_makeCoaddTempExp_metadata" % (self.config.coaddName,)


class CoaddArgumentParser(pipeBase.ArgumentParser):
    """A version of lsst.pipe.base.ArgumentParser specialized for coaddition.
    
    Required because butler.subset does not support patch and tract
    """
    def _makeDataRefList(self, namespace):
        """Make namespace.dataRefList from namespace.dataIdList
        """
        datasetType = namespace.config.coaddName + "Coadd"
        validKeys = namespace.butler.getKeys(datasetType=datasetType, level=self._dataRefLevel)

        namespace.dataRefList = []
        for dataId in namespace.dataIdList:
            # tract and patch are required
            for key in validKeys:
                if key not in dataId:
                    self.error("--id must include " + key)
            dataRef = namespace.butler.dataRef(
                datasetType = datasetType,
                dataId = dataId,
            )
            namespace.dataRefList.append(dataRef)
