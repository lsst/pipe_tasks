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
from .selectImages import BadSelectImagesTask

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
        target = BadSelectImagesTask,
    )
    badMaskPlanes = pexConfig.ListField(
        dtype = str,
        doc = "Mask planes that, if set, the associated pixel should not be included in the coaddTempExp.",
        default = ("EDGE",),
    )


class CoaddBaseTask(pipeBase.CmdLineTask):
    """Base class for coaddition.
    
    Subclasses must specify _DefaultName
    """
    ConfigClass = CoaddBaseConfig
    
    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("select")

    def selectExposures(self, patchRef, skyInfo=None):
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
        return self.select.runDataRef(patchRef, coordList).dataRefList
    
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

    def getCalExp(self, dataRef, getPsf=True, bgSubtracted=False):
        """Return one "calexp" calibrated exposure, optionally with psf

        @param dataRef: a sensor-level data reference
        @param getPsf: include the PSF?
        @param bgSubtracted: return calexp with background subtracted? If False
            get the calexp's background background model and add it to the cale
        @return calibrated exposure with psf
        """
        exposure = dataRef.get("calexp", immediate=True)
        if not bgSubtracted:
            background = dataRef.get("calexpBackground", immediate=True)
            mi = exposure.getMaskedImage()
            mi += background
            del mi
        if getPsf:
            psf = dataRef.get("psf", immediate=True)
            exposure.setPsf(psf)
        return exposure

    def makeModelPsf(self, fwhm, wcs, sizeFactor=3.0):
        """Construct a model PSF, or reuse the prior model, if possible
        
        The model PSF is a double Gaussian with core FWHM = fwhmPixels
        and wings of amplitude 1/10 of core and FWHM = 2.5 * core.
        
        @param fwhm: desired FWHM of core Gaussian, in arcseconds
        @param wcs: Wcs of the image (for pixel scale)
        @param sizeFactor: multiplier of fwhm for kernel size
        @return model PSF
        """
        if fwhm is None or fwhm <= 0:
            return None
        fwhmPixels = fwhm / wcs.pixelScale().asArcseconds()
        kernelDim = int(sizeFactor * fwhmPixels + 0.5)
        self.log.logdebug("Create double Gaussian PSF model with core fwhm %0.1f pixels and size %dx%d" %
                          (fwhmPixels, kernelDim, kernelDim))
        coreSigma = fwhmPixels / FwhmPerSigma
        return afwDetection.createPsf("DoubleGaussian", kernelDim, kernelDim, coreSigma, coreSigma * 2.5, 0.1)

    def getCoaddDataset(self):
        """Return the name of the coadd dataset"""
        return self.config.coaddName + "Coadd"

    def getTempExpDataset(self):
        """Return the name of the coadd tempExp (i.e., warp) dataset"""
        return self.config.coaddName + "Coadd_tempExp"

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd", help="data ID, e.g. --id tract=12345 patch=1,2",
                               ContainerClass=CoaddDataIdContainer)
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
