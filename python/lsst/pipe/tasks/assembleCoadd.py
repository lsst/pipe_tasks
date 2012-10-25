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
import lsst.pex.config as pexConfig
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
from .coaddBase import CoaddBaseTask
from .interpImage import InterpImageTask
from .matchBackgrounds import MatchBackgroundsTask


__all__ = ["AssembleCoaddTask"]

class AssembleCoaddConfig(CoaddBaseTask.ConfigClass):
    subregionSize = pexConfig.ListField(
        dtype = int,
        doc = "Width, height of stack subregion size; " \
              "make small enough that a full stack of images will fit into memory at once.",
        length = 2,
        default = (2000, 2000),
    )
    doSigmaClip = pexConfig.Field(
        dtype = bool,
        doc = "Perform sigma clipped outlier rejection? If False then compute a simple mean.",
        default = True,
    )
    sigmaClip = pexConfig.Field(
        dtype = float,
        doc = "Sigma for outlier rejection; ignored if doSigmaClip false.",
        default = 3.0,
    )
    clipIter = pexConfig.Field(
        dtype = int,
        doc = "Number of iterations of outlier rejection; ignored if doSigmaClip false.",
        default = 2,
    )
    doInterp = pexConfig.Field(
        doc = "Interpolate over NaN pixels? Also extrapolate, if necessary, but the results are ugly.",
        dtype = bool,
        default = True,
    )
    interpFwhm = pexConfig.Field(
        doc = "FWHM of PSF used for interplation (arcsec)",
        dtype = float,
        default = 1.5,
    )
    interpImage = pexConfig.ConfigurableField(
        target = InterpImageTask,
        doc = "Task to interpolate (and extrapolate) over NaN pixels",
    )
    matchBackgrounds = pexConfig.ConfigurableField(
        target = MatchBackgroundsTask,
        doc = "Task to match backgrounds",
    )
    
    doWrite = pexConfig.Field(
        doc = "Persist coadd?",
        dtype = bool,
        default = True,
    )
    
    doMatchBackgrounds = pexConfig.Field(
        doc = "MatchBackgrounds?",
        dtype = bool,
        default = True,
    )
    autoReference = pexConfig.Field(
        doc = "If doMatchBackgrounds ==True, tell match Backgrounds to calculate best reference image?",
        dtype = bool,
        default = True,
    )

    
class AssembleCoaddTask(CoaddBaseTask):
    """Assemble a coadd from a set of coaddTempExp
    """
    ConfigClass = AssembleCoaddConfig
    _DefaultName = "assembleCoadd"

    def __init__(self, *args, **kwargs):
        CoaddBaseTask.__init__(self, *args, **kwargs)
        self.makeSubtask("interpImage")
        self.makeSubtask("matchBackgrounds")
        print "Done initializing"
    
    @pipeBase.timeMethod
    def run(self, patchRef):
        """Assemble a coadd from a set of coaddTempExp
        
        The coadd is computed as a mean with optional outlier rejection.
        
        @param patchRef: data reference for sky map. Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter")
        Used to access the following data products (depending on the config):
        - [in] self.config.coaddName + "Coadd_skyMap"
        - [in] self.config.coaddName + "Coadd_tempExp"
        - [out] self.config.coaddName + "Coadd"

        @return: a pipeBase.Struct with fields:
        - coaddExposure: coadd exposure
        """
        skyInfo = self.getSkyInfo(patchRef)
        datasetType = self.config.coaddName + "Coadd"
        
        wcs = skyInfo.wcs
        bbox = skyInfo.bbox
        
        calExpRefList = self.selectExposures(patchRef=patchRef, wcs=wcs, bbox=bbox)
        
        numExp = len(calExpRefList)
        if numExp < 1:
            raise pipeBase.TaskError("No exposures to coadd")
        self.log.info("Selected %s calexp" % (numExp,))

        tempExpName = self.config.coaddName + "Coadd_tempExp"
        tempExpSubName = tempExpName + "_sub"

        # compute tempKeyList: a tuple of ID key names in a calExpId that identify a coaddTempExp.
        # You must also specify tract and patch to make a complete coaddTempExp ID.
        butler = patchRef.butlerSubset.butler
        tempExpKeySet = set(butler.getKeys(datasetType=tempExpName, level="Ccd")) - set(("patch", "tract"))
        coaddKeySet = set(butler.getKeys(datasetType=datasetType, level="Ccd")) - set(("patch", "tract"))
        
        tempExpKeyList = tuple(sorted(tempExpKeySet))


        if self.config.autoReference:
            refVisitRef = None
        else:    
            # define refVisitRef and take out visit or run from the patchRef to make it a true patchRef
            refVistRef = butler.dataRef(datasetType = tempExpName, dataId=patchRef.dataId)
            for key in tempExpKeySet:
                if key not in coaddKeySet:
                    del patchRef.dataId[key]
                
        # compute tempExpIdDict, a dict whose:
        # - keys are tuples of coaddTempExp ID values in tempKeyList order
        # - values are tempExpRef
        # Do not check for existence yet (to avoid checking one coaddTempExp multiple times);
        # wait until all desired coaddTempExp have been identified
        tempExpIdDict = dict()
        for calExpRef in calExpRefList:
            calExpId = calExpRef.dataId
            tempExpIdTuple = tuple(calExpId[key] for key in tempExpKeyList)    
            if tempExpIdTuple not in tempExpIdDict:
                tempExpId = dict((key, calExpId[key]) for key in tempExpKeyList)
                tempExpId.update(patchRef.dataId)
                tempExpRef = calExpRef.butlerSubset.butler.dataRef(
                    datasetType = tempExpName,
                    dataId = tempExpId,
                )
                tempExpIdDict[tempExpIdTuple] = tempExpRef

        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(3.0)
        statsCtrl.setNumIter(2)
        statsCtrl.setAndMask(self._badPixelMask)
        statsCtrl.setNanSafe(True)
        
        # compute tempExpRefList: a list of tempExpRef that actually exist
        # and weightList: a list of the weight of the associated tempExp
        tempExpRefList = []
        weightList = []
        for tempExpRef in tempExpIdDict.itervalues():
            if not tempExpRef.datasetExists(tempExpName):
                self.log.warn("Could not find %s %s; skipping it" % (tempExpName, tempExpRef.dataId))
                continue

            tempExp = tempExpRef.get(tempExpName)
            maskedImage = tempExp.getMaskedImage()
            statObj = afwMath.makeStatistics(maskedImage.getVariance(), maskedImage.getMask(),
                afwMath.MEANCLIP, statsCtrl)
            meanVar, meanVarErr = statObj.getResult(afwMath.MEANCLIP);
            weight = 1.0 / float(meanVar)
            self.log.info("Weight of %s %s = %0.3f" % (tempExpName, tempExpRef.dataId, weight))
            del maskedImage
            del tempExp
            
            tempExpRefList.append(tempExpRef)
            weightList.append(weight)
               
        del tempExpIdDict

        if not tempExpRefList:
            raise pipeBase.TaskError("No coadd temporary exposures found")
        self.log.info("Found %s %s" % (len(tempExpRefList), tempExpName))

        if self.config.doMatchBackgrounds:
            #This is messier than neccessary. 
            matchBackgroundsStruct = self.matchBackgrounds.run(tempExpRefList, refVisitRef = refVistRef,tempExpName=tempExpName)
            backgroundModelsList = matchBackgroundsStruct.backgroundModelList
            fitRMSList = matchBackgroundsStruct.fitRMSList
            weightMatchBackgroundsList = []
            shorterTempExpRefList = []
            backgroundList =[]
            # the number of good backgrounds may be < than len(tempExpList)
            # sync these up and correct weights
            for tempExpRef in tempExpRefList:
                if backgroundModelsList[tempExpRefList.index(tempExpRef)] is None:
                    self.log.info("No background offset model available for %s: skipping"%(tempExpRef.dataId))
                    continue
                weightMatchBackgroundsList.append(1/(1/weightList[tempExpRefList.index(tempExpRef)] + fitRMSList[tempExpRefList.index(tempExpRef)]**2))
                shorterTempExpRefList.append(tempExpRef)
                backgroundList.append(backgroundModelsList[tempExpRefList.index(tempExpRef)])
            weightList = weightMatchBackgroundsList
            tempExpRefList = shorterTempExpRefList
            backgroundModelsList = backgroundList
            
        print len(weightList), len(tempExpRefList)
        self.log.info("Assembling %s %s" % (len(tempExpRefList), tempExpName))                                          
        edgeMask = afwImage.MaskU.getPlaneBitMask("EDGE")
        
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(self.config.sigmaClip)
        statsCtrl.setNumIter(self.config.clipIter)
        statsCtrl.setAndMask(self.getBadPixelMask())
        statsCtrl.setNanSafe(True)
        statsCtrl.setCalcErrorFromInputVariance(True)

        if self.config.doSigmaClip:
            statsFlags = afwMath.MEANCLIP
        else:
            statsFlags = afwMath.MEAN
    
        coaddExposure = afwImage.ExposureF(bbox, wcs)
        coaddMaskedImage = coaddExposure.getMaskedImage()
        subregionSizeArr = self.config.subregionSize
        subregionSize = afwGeom.Extent2I(subregionSizeArr[0], subregionSizeArr[1])
        didSetMetadata = False
        for subBBox in _subBBoxIter(bbox, subregionSize):
            try:
                self.log.info("Computing coadd %s" % (subBBox,))
                coaddView = afwImage.MaskedImageF(coaddMaskedImage, subBBox, afwImage.PARENT, False)
                maskedImageList = afwImage.vectorMaskedImageF() # [] is rejected by afwMath.statisticsStack
                for tempExpRef in tempExpRefList:
                    exposure = tempExpRef.get(tempExpSubName, bbox=subBBox, imageOrigin="PARENT")
                    maskedImage = exposure.getMaskedImage()
                    if not didSetMetadata:
                        coaddExposure.setFilter(exposure.getFilter())
                        coaddExposure.setCalib(exposure.getCalib())
                        didSetMetadata = True   
                    if self.config.doMatchBackgrounds:
                        idx = tempExpRefList.index(tempExpRef)
                        if self.matchBackgrounds.config.useDetectionBackground:
                            bboxInImageCoords = afwGeom.Box2I(subBBox)
                            bboxInImageCoords.shift(-afwGeom.Extent2I(bbox.getBegin()))
                            maskedImage += afwImage.ImageF(backgroundModelsList[idx].getImageF(),bboxInImageCoords, afwImage.LOCAL, False)
                        else:
                            maskedImage += backgroundModelsList[idx]                       
                        #var = maskedImage.getVariance()
                        #var += fitRMSList[tempExpRefList.index(tempExpRef)]**2
                    maskedImageList.append(maskedImage)

                with self.timer("stack"):
                    coaddSubregion = afwMath.statisticsStack(
                        maskedImageList, statsFlags, statsCtrl, weightList)
    
                coaddView <<= coaddSubregion
            except Exception, e:
                self.log.fatal("Cannot compute coadd %s: %s" % (subBBox, e,))
    
        coaddUtils.setCoaddEdgeBits(coaddMaskedImage.getMask(), coaddMaskedImage.getVariance())

        if self.config.doInterp:
            fwhmPixels = self.config.interpFwhm / wcs.pixelScale().asArcseconds()
            self.interpImage.interpolateOnePlane(
                maskedImage = coaddExposure.getMaskedImage(),
                planeName = "EDGE",
                fwhmPixels = fwhmPixels,
            )

        if self.config.doWrite:
            coaddName = self.config.coaddName + "Coadd"
            self.log.info("Persisting %s" % (coaddName,))
            patchRef.put(coaddExposure, coaddName)

        return pipeBase.Struct(
            coaddExposure = coaddExposure,
        )


    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        #not sure about parameters but it doesn't seem to matter:
        return AssembleCoaddArgumentParser(
            name=cls._DefaultName,
            datasetType=cls.ConfigClass().coaddName + "Coadd_tempExp"
            )
                                      
    def _getConfigName(self):
        """Return the name of the config dataset
        """
        return "%s_%s_config" % (self.config.coaddName, self._DefaultName)
    
    def _getMetadataName(self):
        """Return the name of the metadata dataset
        """
        return "%s_%s_metadata" % (self.config.coaddName, self._DefaultName)
    
def _subBBoxIter(bbox, subregionSize):
    """Iterate over subregions of a bbox
    
    @param[in] bbox: bounding box over which to iterate: afwGeom.Box2I
    @param[in] subregionSize: size of sub-bboxes

    @return subBBox: next sub-bounding box of size subregionSize or smaller;
        each subBBox is contained within bbox, so it may be smaller than subregionSize at the edges of bbox,
        but it will never be empty
    """
    if bbox.isEmpty():
        raise RuntimeError("bbox %s is empty" % (bbox,))
    if subregionSize[0] < 1 or subregionSize[1] < 1:
        raise RuntimeError("subregionSize %s must be nonzero" % (subregionSize,))

    for rowShift in range(0, bbox.getHeight(), subregionSize[1]):
        for colShift in range(0, bbox.getWidth(), subregionSize[0]):
            subBBox = afwGeom.Box2I(bbox.getMin() + afwGeom.Extent2I(colShift, rowShift), subregionSize)
            subBBox.clip(bbox)
            if subBBox.isEmpty():
                raise RuntimeError("Bug: empty bbox! bbox=%s, subregionSize=%s, colShift=%s, rowShift=%s" % \
                    (bbox, subregionSize, colShift, rowShift))
            yield subBBox




class AssembleCoaddArgumentParser(pipeBase.ArgumentParser):
    """A version of lsst.pipe.base.ArgumentParser specialized for assembleCoadd.
    """
    def _makeDataRefList(self, namespace):
        """Make namespace.dataRefList from namespace.dataIdList
        """    
        if namespace.config.doMatchBackgrounds:  
            if namespace.config.autoReference: #matcher will pick it's own reference image
                namespace.datasetType = namespace.config.coaddName + "Coadd"
            else:
                namespace.datasetType = namespace.config.coaddName + "Coadd_tempExp"
        else: #bkg subtracted coadd         
            namespace.datasetType = namespace.config.coaddName + "Coadd"           
        validKeys = namespace.butler.getKeys(datasetType=namespace.datasetType, level=self._dataRefLevel)

        namespace.dataRefList = []
        for dataId in namespace.dataIdList:
            # tract and patch are required
            for key in validKeys:
                if key not in dataId:
                    self.error("--id must include " + key)
            dataRef = namespace.butler.dataRef(
                datasetType = namespace.datasetType,
                dataId = dataId,
            )
            namespace.dataRefList.append(dataRef)


# Another Idea is to do away with config.autoReference
# Look for id's in coadd_tempExp and if you don't find visit/run it's ok! Automatically switch to autoReference.
# Still in Dev
class LessFlexibleButStillIntuitiveAssembleCoaddArgumentParser(pipeBase.ArgumentParser):
    def _makeDataRefList(self, namespace):
        if namespace.config.doMatchBackgrounds:
            namespace.datasetType = namespace.config.coaddName + "Coadd_tempExp"
        else:
            namespace.datasetType = namespace.config.coaddName + "Coadd"
            
        import pdb; pdb.set_trace()
        validKeysCoadd = namespace.butler.getKeys(datasetType=namespace.config.coaddName + "Coadd", level=self._dataRefLevel)
        validKeys = namespace.butler.getKeys(datasetType=namespace.config.coaddName + "Coadd_tempExp", level=self._dataRefLevel)

         #need to make a list of keys in coaddTempExp that are not in Coadd
         #e.g. "visit" for imsim or "run" for sdss
         
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
