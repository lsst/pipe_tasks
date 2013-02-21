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
import lsst.pipe.base as pipeBase

__all__ = ["BaseSelectImagesTask", "BaseExposureInfo", "BadSelectImagesTask", "DatabaseSelectImagesConfig"]

class DatabaseSelectImagesConfig(SelectImagesConfig):
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

class BaseExposureInfo(object):
    """Data about a selected exposure
    """
    def __init__(self):
        """Create exposure information from a query result from a db connection
        
        The object has the following fields:
        - dataId: data ID of exposure (a dict)
        - coordList: a list of corner coordinates of the exposure (list of afwCoord.IcrsCoord)
        plus any others items that are desired
        
        Subclasses must provide __init__ (which calls this one) and override getColumnNames.
        """
        self._ind = -1

    @property
    def _nextInd(self):
        self._ind += 1
        return self._ind

    @staticmethod
    def getColumnNames():
        """Set database query columns to be consistent with constructor
        
        For example:
        return "raftName, visit, ccdName, filterName, ra1, dec1, ra2, dec2, ra3, dec3, ra4, dec4"
        """
        raise NotImplementedError()


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
    
    def runDataRef(self, dataRef, coordList, makeDataRefList=True):
        """Run based on a data reference
        
        @param[in] dataRef: data reference; must contain any extra keys needed by the subclass
        @param[in] coordList: list of coordinates defining region of interest; if None, search the whole sky
        @param[in] makeDataRefList: if True, return dataRefList
        @return a pipeBase Struct containing:
        - exposureInfoList: a list of ccdInfo objects
        - dataRefList: a list of data references (None if makeDataRefList False)
        """
        runArgDict = self._runArgDictFromDataId(dataRef.dataId)
        exposureInfoList = self.run(coordList, **runArgDict).exposureInfoList

        if makeDataRefList:        
            butler = dataRef.butlerSubset.butler
            dataRefList = [butler.dataRef(
                datasetType = "calexp",
                dataId = ccdInfo.dataId,
            ) for ccdInfo in exposureInfoList]
        else:
            dataRefList = None

        return pipeBase.Struct(
            dataRefList = dataRefList,
            exposureInfoList = exposureInfoList,
        )

class BadSelectImagesTask(BaseSelectImagesTask):
    """Non-functional selection task intended as a placeholder subtask
    """
    def run(self, coordList):
        raise RuntimeError("No select task specified")

    def _runArgDictFromDataId(self, dataId):        
        raise RuntimeError("No select task specified")

