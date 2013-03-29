#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
import lsst.pex.exceptions as pexExceptions
import lsst.afw.geom as afwGeom
import lsst.pipe.base as pipeBase

__all__ = ["BaseSelectImagesTask", "BaseExposureInfo", "WcsSelectImagesTask", "DatabaseSelectImagesConfig"]

class DatabaseSelectImagesConfig(pexConfig.Config):
    """Configuration for BaseSelectImagesTask, using a database"""
    host = pexConfig.Field(
        doc = "Database server host name",
        dtype = str,
    )
    port = pexConfig.Field(
        doc = "Database server port",
        dtype = int,
    )
    database = pexConfig.Field(
        doc = "Name of database",
        dtype = str,
    )
    maxExposures = pexConfig.Field(
        doc = "maximum exposures to select; intended for debugging; ignored if None",
        dtype = int,
        optional = True,
    )

class BaseExposureInfo(pipeBase.Struct):
    """Data about a selected exposure
    """
    def __init__(self, dataId, coordList):
        """Create exposure information that can be used to generate data references

        The object has the following fields:
        - dataId: data ID of exposure (a dict)
        - coordList: a list of corner coordinates of the exposure (list of afwCoord.IcrsCoord)
        plus any others items that are desired
        """
        super(BaseExposureInfo, self).__init__(dataId=dataId, coordList=coordList)


class BaseSelectImagesTask(pipeBase.Task):
    """Base task for selecting images suitable for coaddition
    """
    ConfigClass = pexConfig.Config
    _DefaultName = "selectImages"
    
    @pipeBase.timeMethod
    def run(self, coordList):
        """Select images suitable for coaddition in a particular region
        
        @param[in] coordList: list of coordinates defining region of interest; if None then select all images
        subclasses may add additional keyword arguments, as required
        
        @return a pipeBase Struct containing:
        - exposureInfoList: a list of exposure information objects (subclasses of BaseExposureInfo),
            which have at least the following fields:
            - dataId: data ID dictionary
            - coordList: coordinates of the corner of the exposure (list of afwCoord.IcrsCoord)
        """
        raise NotImplementedError()
    
    def _runArgDictFromDataId(self, dataId):
        """Extract keyword arguments for run (other than coordList) from a data ID
        
        @return keyword arguments for run (other than coordList), as a dict
        """
        raise NotImplementedError()
    
    def runDataRef(self, dataRef, coordList, makeDataRefList=True, selectDataList=[]):
        """Run based on a data reference

        This delegates to run() and _runArgDictFromDataId() to do the actual
        selection. In the event that the selectDataList is non-empty, this will
        be used to further restrict the selection, providing the user with
        additional control over the selection.

        @param[in] dataRef: data reference; must contain any extra keys needed by the subclass
        @param[in] coordList: list of coordinates defining region of interest; if None, search the whole sky
        @param[in] makeDataRefList: if True, return dataRefList
        @param[in] selectDataList: List of SelectStruct with dataRefs to consider for selection
        @return a pipeBase Struct containing:
        - exposureInfoList: a list of objects derived from ExposureInfo
        - dataRefList: a list of data references (None if makeDataRefList False)
        """
        runArgDict = self._runArgDictFromDataId(dataRef.dataId)
        exposureInfoList = self.run(coordList, **runArgDict).exposureInfoList

        if len(selectDataList) > 0 and len(exposureInfoList) > 0:
            # Restrict the exposure selection further
            ccdKeys, ccdValues = _extractKeyValue(exposureInfoList)
            inKeys, inValues = _extractKeyValue([s.dataRef for s in selectDataList], keys=ccdKeys)
            inValues = set(inValues)
            newExposureInfoList = []
            for info, ccdVal in zip(exposureInfoList, ccdValues):
                if ccdVal in inValues:
                    newExposureInfoList.append(info)
                else:
                    self.log.info("De-selecting exposure %s: not in selectDataList" % info.dataId)
            exposureInfoList = newExposureInfoList

        if makeDataRefList:
            butler = dataRef.butlerSubset.butler
            dataRefList = [butler.dataRef(datasetType = "calexp",
                                          dataId = expInfo.dataId,
                                          ) for expInfo in exposureInfoList]
        else:
            dataRefList = None

        return pipeBase.Struct(
            dataRefList = dataRefList,
            exposureInfoList = exposureInfoList,
        )


def _extractKeyValue(dataList, keys=None):
    """Extract the keys and values from a list of dataIds

    The input dataList is a list of objects that have 'dataId' members.
    This allows it to be used for both a list of data references and a
    list of ExposureInfo
    """
    assert len(dataList) > 0
    if keys is None:
        keys = sorted(dataList[0].dataId.keys())
    keySet = set(keys)
    values = list()
    for data in dataList:
        thisKeys = set(data.dataId.keys())
        if thisKeys != keySet:
            raise RuntimeError("DataId keys inconsistent: %s vs %s" % (keySet, thisKeys))
        values.append(tuple(data.dataId[k] for k in keys))
    return keys, values


class SelectStruct(pipeBase.Struct):
    """A container for data to be passed to the WcsSelectImagesTask"""
    def __init__(self, dataRef, wcs, dims):
        super(SelectStruct, self).__init__(dataRef=dataRef, wcs=wcs, dims=dims)


class WcsSelectImagesTask(BaseSelectImagesTask):
    """Select images using their Wcs"""
    def runDataRef(self, dataRef, coordList, makeDataRefList=True, selectDataList=[]):
        """Images in the selectDataList that overlap the region specified by the
        coordList are selected.

        This comparison is conservative and non-exact: images that do not
        actually overlap the coordList (but are close to it) may also be
        selected.

        @param dataRef: Data reference for coadd/tempExp (with tract, patch)
        @param coordList: List of Coord specifying boundary of patch
        @param makeDataRefList: Construct a list of data references?
        @param selectDataList: List of SelectStruct, to consider for selection
        """
        dataRefList = []
        exposureInfoList = []
        for data in selectDataList:
            dataRef = data.dataRef
            wcs = data.wcs
            nx,ny = data.dims
            try:
                bbox = afwGeom.Box2D(afwGeom.Point2D(0,0), afwGeom.Extent2D(nx, ny))
                bounds = afwGeom.Box2D()
                for coord in coordList:
                    pix = wcs.skyToPixel(coord) # May throw() if wcslib barfs
                    bounds.include(pix)
                if not bbox.overlaps(bounds):
                    self.log.logdebug("De-selecting calexp %s" % dataRef.dataId)
                    continue
                self.log.info("Selecting calexp %s" % dataRef.dataId)
                dataRefList.append(dataRef)
                corners = [wcs.pixelToSky(x,y) for x in (0, nx) for y in (0, ny)]
                exposureInfoList.append(BaseExposureInfo(dataRef.dataId, corners))
            except pexExceptions.LsstCppException, e:
                if not isinstance(e.message, pexExceptions.DomainErrorException):
                    raise
                # Particularly interested in catching problems from wcslib, which may throw() if this exposure
                # is far from the coordinates under consideration.
                self.log.logdebug("WCS error in testing calexp %s (%s): deselecting" % (dataRef.dataId, e))

        return pipeBase.Struct(
            dataRefList = dataRefList if makeDataRefList else None,
            exposureInfoList = exposureInfoList,
        )
