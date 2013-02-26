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
import lsst.pipe.base as pipeBase
from .selectImages import WcsSelectImagesTask, SelectStruct

__all__ = ["CoaddBaseTask"]

FwhmPerSigma = 2 * math.sqrt(2 * math.log(2))

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
        default = ("EDGE",),
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
        self._badPixelMask = afwImage.MaskU.getPlaneBitMask(self.config.badMaskPlanes)

    def getBadPixelMask(self):
        """Get the bad pixel mask
        """
        return self._badPixelMask

    def selectExposures(self, patchRef, wcs, bbox, selectDataList=[]):
        """Select exposures to coadd
        
        @param patchRef: data reference for sky map patch. Must include keys "tract", "patch",
            plus the camera-specific filter key (e.g. "filter" or "band")
        @param[in] wcs: WCS of coadd patch
        @param[in] bbox: bbox of coadd patch
        @return a list of science exposures to coadd, as butler data references
        """
        cornerPosList = afwGeom.Box2D(bbox).getCorners()
        coordList = [wcs.pixelToSky(pos) for pos in cornerPosList]
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
        self.log.logdebug("Create double Gaussian PSF model with core fwhm %0.1f pixels and size %dx%d" % \
            (fwhmPixels, kernelDim[0], kernelDim[1]))
        coreSigma = fwhmPixels / FwhmPerSigma
        return afwDetection.createPsf("DoubleGaussian", kernelDim[0], kernelDim[1],
            coreSigma, coreSigma * 2.5, 0.1)

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

class CoaddDataIdContainer(pipeBase.DataIdContainer):
    """A version of lsst.pipe.base.DataIdContainer specialized for coaddition.
    
    Required because butler.subset does not support patch and tract
    """
    def makeDataRefList(self, namespace):
        """Make self.refList from self.idList
        """
        datasetType = namespace.config.coaddName + "Coadd"
        validKeys = namespace.butler.getKeys(datasetType=datasetType, level=self.level)

        for dataId in self.idList:
            # tract and patch are required
            for key in validKeys:
                if key not in dataId:
                    self.error("--id must include " + key)
            dataRef = namespace.butler.dataRef(
                datasetType = datasetType,
                dataId = dataId,
            )
            self.refList.append(dataRef)

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
                md = ref.get("calexp_md")
                wcs = afwImage.makeWcs(md)
                data = SelectStruct(dataRef=ref, header=md, wcs=wcs,
                                    dims=(md.get("NAXIS1"), md.get("NAXIS2")))
            except:
                data = None
            self.dataList.append(data)

