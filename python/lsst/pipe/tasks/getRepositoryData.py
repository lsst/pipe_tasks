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
"""Retrieve collections of metadata or data based on a set of data references

Use this as a base task for creating graphs and reports for a set of data.
"""
import collections
import itertools
import re

import numpy

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

__all__ = ["DataRefListRunner", "GetRepositoryDataTask"]

class DataRefListRunner(pipeBase.TaskRunner):
    """A task runner that calls run with a list of data references
    
    Differs from the default TaskRunner by providing all data references at once,
    instead of iterating over them one at a time.
    """
    @staticmethod
    def getTargetList(parsedCmd):
        """Return a list of targets (arguments for __call__); one entry per invocation
        """
        return [parsedCmd.dataRefList] # one argument consisting of a list of dataRefs

    def __call__(self, dataRefList):
        """Run GetRepositoryDataTask.run on a single target
        
        @param dataRefList: argument dict for run; contains one key: dataRefList

        @return:
        - None if doReturnResults false
        - A pipe_base Struct containing these fields if doReturnResults true:
            - dataRefList: the argument dict sent to runDataRef
            - metadata: task metadata after execution of runDataRef
            - result: result returned by task runDataRef
        """
        task = self.TaskClass(config=self.config, log=self.log)
        result = task.run(dataRefList)
        
        if self.doReturnResults:
            return Struct(
                dataRefList = dataRefList,
                metadata = task.metadata,
                result = result,
            )


class GetRepositoryDataTask(pipeBase.CmdLineTask):
    """Retrieve data from a repository, e.g. for plotting or analysis purposes
    """
    ConfigClass = pexConfig.Config # nothing to configure
    RunnerClass = DataRefListRunner
    _DefaultName = "getTaskData"
    
    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

    @pipeBase.timeMethod
    def run(self, dataRefList):
        """Get data from a repository for a collection of data references
    
        @param dataRefList: a list of data references
        """
        raise NotImplementedError("subclass must specify a run method")
    
    def getIdList(self, dataRefList, datasetType):
        """Get a list of data IDs in a form that can be used as dictionary keys
        
        @param dataRefList: a list of data references
        @return a pipe_base Struct with fields:
        - idKeyTuple: a tuple of dataRef data ID keys
        - idValList: a list of data ID value tuples, each tuple contains values in the order in idKeyTuple
        """
        if not dataRefList:
            raise RuntimeError("No data refs")
        idKeyTuple = tuple(sorted(dataRefList[0].dataId.keys()))
        
        idValList = []
        for dataRef in dataRefList:
            idValTuple = tuple(dataRef.dataId[key] for key in idKeyTuple)
            idValList.append(idValTuple)

        return pipeBase.Struct(
            idKeyTuple = idKeyTuple,
            idValList = idValList,
        )
    
    def getDataList(self, dataRefList, datasetType):
        """Retrieve a list of data

        @param dataRefList: a list of data references
        @param datasetType: datasetType of data to be retrieved
        @return a list of data, one entry per dataRef in dataRefList (in order)
        """
        return [dataRef.get(datasetType=datasetType) for dataRef in dataRefList]
    
    def getMetadataItems(self, dataRefList, datasetType, nameList):
        """Retrieve a list of tuples of items of metadata
        
        @param dataRefList: a list of data references
        @param datasetType: datasetType of metadata (or any object that supports get(name))
        @return a list of tuples of metadata:
        - each entry in the list corresponds to a dataRef in dataRefList
        - each entry in the tuple contains one entry per name in nameList
        """
        valList = []
        for dataRef in dataRefList:
            metadata = dataRef.get(datasetType=datasetType)
            valList.append(dict((name, metadata.get(name)) for name in nameList))
        return valList
