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
import argparse

import lsst.pex.exceptions as pexExceptions
import lsst.pex.config as pexConfig
import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase
import lsst.meas.algorithms as measAlg

from lsst.afw.fits.fitsLib import FitsError
from .selectImages import WcsSelectImagesTask, SelectStruct
from .coaddInputRecorder import CoaddInputRecorderTask

try:
    from lsst.meas.mosaic import applyMosaicResults
except ImportError:
    applyMosaicResults = None

__all__ = ["CoaddBaseTask", "getSkyInfo"]

FwhmPerSigma = 2 * math.sqrt(2 * math.log(2))

class DoubleGaussianPsfConfig(pexConfig.Config):
    """Configuration for DoubleGaussian model Psf"""
    fwhm = pexConfig.Field(dtype=float, doc="FWHM of core (arcseconds)",
                           default=1.0, check=lambda x: x is None or x > 0.0)
    sizeFactor = pexConfig.Field(dtype=float, doc="Multiplier of fwhm for kernel size", default=3.0,
                                 check=lambda x: x > 0.0)
    wingFwhmFactor = pexConfig.Field(dtype=float, doc="Multiplier of fwhm for wing fwhm", default=2.5,
                                     check=lambda x: x > 0)
    wingAmplitude = pexConfig.Field(dtype=float, doc="Relative amplitude of wing", default=0.1,
                                    check=lambda x: x >= 0)

    def apply(self, wcs):
        """Construct a model PSF

        The model PSF is a double Gaussian with core self.fwhm
        and wings of self.wingAmplitude relative to the core
        and width self.wingFwhmFactor relative to self.fwhm.

        @param wcs: Wcs of the image (for pixel scale)
        @return model PSF or None
        """
        fwhmPixels = self.fwhm / wcs.pixelScale().asArcseconds()
        kernelDim = int(self.sizeFactor * fwhmPixels + 0.5)
        self.log.logdebug("Create double Gaussian PSF model with core fwhm %0.1f pixels and size %dx%d" %
                          (fwhmPixels, kernelDim, kernelDim))
        coreSigma = fwhmPixels / FwhmPerSigma
        return measAlg.DoubleGaussianPsf(kernelDim, kernelDim, coreSigma,
                                         coreSigma * self.wingFwhmFactor, self.wingAmplitude)

class CoaddBaseConfig(pexConfig.Config):
    """Config for CoaddBaseTask
    """
    coaddName = pexConfig.Field(
        doc = "Coadd name: typically one of deep or goodSeeing.",
        dtype = str,
        default = "deep",
    )
    select = pexConfig.ConfigurableField(
        doc = "Image selection subtask.",
        target = WcsSelectImagesTask,
    )
    badMaskPlanes = pexConfig.ListField(
        dtype = str,
        doc = "Mask planes that, if set, the associated pixel should not be included in the coaddTempExp.",
        default = ("NO_DATA",),
    )
    inputRecorder = pexConfig.ConfigurableField(
        doc = "Subtask that helps fill CoaddInputs catalogs added to the final Exposure",
        target = CoaddInputRecorderTask
    )
    doPsfMatch = pexConfig.Field(dtype=bool, doc="Match to modelPsf?", default=False)
    modelPsf = pexConfig.ConfigField(dtype=DoubleGaussianPsfConfig, doc="Model Psf specification")
    doApplyUberCal = pexConfig.Field(
        dtype = bool,
        doc = "Apply meas_mosaic ubercal results to input calexps?",
        default = True
    )

class CoaddTaskRunner(pipeBase.TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return pipeBase.TaskRunner.getTargetList(parsedCmd, selectDataList=parsedCmd.selectId.dataList,
                                                 **kwargs)

class CoaddBaseTask(pipeBase.CmdLineTask):
    """Base class for coaddition.
    
    Subclasses must specify _DefaultName
    """
    ConfigClass = CoaddBaseConfig
    RunnerClass = CoaddTaskRunner
    
    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("select")
        self.makeSubtask("inputRecorder")

    def selectExposures(self, patchRef, skyInfo=None, selectDataList=[]):
        """Select exposures to coadd
        
        @param patchRef: data reference for sky map patch. Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter" or "band")
        @param[in] skyInfo: geometry for the patch; output from getSkyInfo
        @return a list of science exposures to coadd, as butler data references
        """
        if skyInfo is None:
            skyInfo = self.getSkyInfo(patchRef)
        cornerPosList = afwGeom.Box2D(skyInfo.bbox).getCorners()
        coordList = [skyInfo.wcs.pixelToSky(pos) for pos in cornerPosList]
        return self.select.runDataRef(patchRef, coordList, selectDataList=selectDataList).dataRefList
    
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
        return getSkyInfo(coaddName=self.config.coaddName, patchRef=patchRef)

    def getCalExp(self, dataRef, bgSubtracted):
        """Return one "calexp" calibrated exposure

        @param dataRef: a sensor-level data reference
        @param bgSubtracted: return calexp with background subtracted? If False
            get the calexp's background background model and add it to the calexp.
        @return calibrated exposure

        If config.doApplyUberCal, meas_mosaic calibrations will be applied to
        the returned exposure using applyMosaicResults.
        """
        exposure = dataRef.get("calexp", immediate=True)            
        if not bgSubtracted:
            background = dataRef.get("calexpBackground", immediate=True)
            mi = exposure.getMaskedImage()
            mi += background.getImage()
            del mi
        if not self.config.doApplyUberCal:
            return exposure
        if applyMosaicResults is None:
            raise RuntimeError(
                ("Cannot use improved calibrations for %s because meas_mosaic could not be imported; "
                 "either run meas_mosaic or set doApplyUberCal = False")
                % dataRef.dataId
                )
        else:
            try:
                applyMosaicResults(dataRef, calexp=exposure)
            except Exception as err:
                raise RuntimeError(
                    ("Cannot use improved calibrations for %s (%s); "
                     "either run meas_mosaic or set doApplyUberCal = False")
                    % (dataRef.dataId, err)
                    )
        return exposure

    def getCoaddDatasetName(self):
        return self.config.coaddName + "Coadd"

    def getTempExpDatasetName(self):
        return self.config.coaddName + "Coadd_tempExp"

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd", help="data ID, e.g. --id tract=12345 patch=1,2",
                               ContainerClass=CoaddDataIdContainer)
        parser.add_id_argument("--selectId", "calexp", help="data ID, e.g. --selectId visit=6789 ccd=0..9",
                               ContainerClass=SelectDataIdContainer)
        return parser

    def _getConfigName(self):
        """Return the name of the config dataset
        """
        return "%s_%s_config" % (self.config.coaddName, self._DefaultName)
    
    def _getMetadataName(self):
        """Return the name of the metadata dataset
        """
        return "%s_%s_metadata" % (self.config.coaddName, self._DefaultName)

    def getBadPixelMask(self):
        """Convenience method to provide the bitmask from the mask plane names"""
        return afwImage.MaskU.getPlaneBitMask(self.config.badMaskPlanes)

    def writeCoaddOutput(self, dataRef, obj, suffix=None):
        """Write a coadd product through the butler

        @param dataRef: data reference for coadd
        @param obj: coadd product to write
        @param suffix: suffix to apply to coadd dataset name
        """
        objName = self.getCoaddDatasetName()
        if suffix is not None:
            objName += "_" + suffix
        self.log.info("Persisting %s" % objName)
        dataRef.put(obj, objName)

class CoaddDataIdContainer(pipeBase.DataIdContainer):
    """A version of lsst.pipe.base.DataIdContainer specialized for coaddition.
    
    Required because butler.subset does not support patch and tract
    """
    def getSkymap(self, namespace, datasetType):
        """Only retrieve skymap if required"""
        if not hasattr(self, "_skymap"):
            self._skymap = namespace.butler.get(datasetType + "_skyMap")
        return self._skymap

    def makeDataRefList(self, namespace):
        """Make self.refList from self.idList
        """
        datasetType = namespace.config.coaddName + "Coadd"
        validKeys = namespace.butler.getKeys(datasetType=datasetType, level=self.level)

        for dataId in self.idList:
            for key in validKeys:
                if key in ("tract", "patch"):
                    # Will deal with these explicitly
                    continue
                if key not in dataId:
                    raise argparse.ArgumentError(None, "--id must include " + key)

            # tract and patch are required; iterate over them if not provided
            if not "tract" in dataId:
                if "patch" in dataId:
                    raise RuntimeError("'patch' cannot be specified without 'tract'")
                addList = [dict(tract=tract.getId(), patch="%d,%d" % patch.getIndex(), **dataId)
                           for tract in self.getSkymap(namespace, datasetType) for patch in tract]
            elif not "patch" in dataId:
                tract = self.getSkymap(namespace, datasetType)[dataId["tract"]]
                addList = [dict(patch="%d,%d" % patch.getIndex(), **dataId) for patch in tract]
            else:
                addList = [dataId]

            self.refList += [namespace.butler.dataRef(datasetType=datasetType, dataId=addId)
                             for addId in addList]


class ExistingCoaddDataIdContainer(CoaddDataIdContainer):
    """A version of CoaddDataIdContainer that only produces references that exist"""
    def makeDataRefList(self, namespace):
        super(ExistingCoaddDataIdContainer, self).makeDataRefList(namespace)
        self.refList = [ref for ref in self.refList if ref.datasetExists()]


class SelectDataIdContainer(pipeBase.DataIdContainer):
    """A dataId container for inputs to be selected.

    We will read the header (including the size and Wcs) for all specified
    inputs and pass those along, ultimately for the SelectImagesTask.
    This is most useful when used with multiprocessing, as input headers are
    only read once.
    """
    def makeDataRefList(self, namespace):
        """Add a dataList containing useful information for selecting images"""
        super(SelectDataIdContainer, self).makeDataRefList(namespace)
        self.dataList = []
        for ref in self.refList:
            try:
                md = ref.get("calexp_md", immediate=True)
                wcs = afwImage.makeWcs(md)
                data = SelectStruct(dataRef=ref, wcs=wcs, dims=(md.get("NAXIS1"), md.get("NAXIS2")))
            except pexExceptions.LsstCppException, e:
                if not isinstance(e.message, FitsError): # Unable to open file
                    raise
                namespace.log.warn("Unable to construct Wcs from %s" % (ref.dataId))
                continue
            self.dataList.append(data)

def getSkyInfo(coaddName, patchRef):
    """Return SkyMap, tract and patch

    @param coaddName: coadd name; typically one of deep or goodSeeing
    @param patchRef: data reference for sky map. Must include keys "tract" and "patch"
    
    @return pipe_base Struct containing:
    - skyMap: sky map
    - tractInfo: information for chosen tract of sky map
    - patchInfo: information about chosen patch of tract
    - wcs: WCS of tract
    - bbox: outer bbox of patch, as an afwGeom Box2I
    """
    skyMap = patchRef.get(coaddName + "Coadd_skyMap")
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
