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
"""Tools to help you iterate over a set of repositories.

Helpful while creating them or harvesting data from them.
"""
from __future__ import absolute_import, division, print_function
from builtins import zip
from builtins import object
import itertools

import numpy

STR_PADDING = 5  # used by _getDTypeList; the number of characters to add to the first string value seen
# when estimating the number of characters needed to store values for a key


def _getDTypeList(keyTuple, valTuple):
    """Construct a numpy dtype for a data ID or repository ID

    @param[in] keyTuple: ID key names, in order
    @param[in] valTuple: a value tuple
    @return numpy dtype as a list

    @warning: this guesses at string length (STR_PADDING + length of string in valTuple);
    longer strings will be truncated when inserted into numpy structured arrays
    """
    typeList = []
    for name, val in zip(keyTuple, valTuple):
        if isinstance(val, str):
            predLen = len(val) + STR_PADDING
            typeList.append((name, str, predLen))
        else:
            typeList.append((name, numpy.array([val]).dtype))
    return typeList


class SourceData(object):
    """Accumulate a set of measurements from a set of source tables

    To use:
    - specify the desired source measurements when constructing this object
    - call addSourceMetrics for each repository you harvest data from
    - call finalize to produce the final data

    Data available after calling finalize:
    - self.sourceArr: a numpy structured array of shape (num repositories, num sources)
        containing named columns for:
        - source ID
        - each data ID key
        - each item of data extracted from the source table
    - self.sourceIdDict: a dict of (source ID: index of axis 1 of self.sourceArr)
    - self.repoArr: a numpy structured array of shape (num repositories,)
        containing a named column for each repository key (see RepositoryIterator)

    @note: sources that had non-finite data (e.g. NaN) for every value extracted are silently omitted
    """

    def __init__(self, datasetType, sourceKeyTuple):
        """
        @param[in] datasetType: dataset type for source
        @param[in] sourceKeyTuple: list of keys of data items to extract from the source tables

        @raise RuntimeError if sourceKeyTuple is empty
        """
        if len(sourceKeyTuple) < 1:
            raise RuntimeError("Must specify at least one key in sourceKeyTuple")
        self.datasetType = datasetType
        self._sourceKeyTuple = tuple(sourceKeyTuple)

        self._idKeyTuple = None  # tuple of data ID keys, in order; set by first call to _getSourceMetrics
        self._idKeyDTypeList = None  # numpy dtype for data ID tuple, as a list of (key, type);
        # set by first call to _getSourceMetrics
        self._sourceDTypeList = None  # numpy dtype for source data, as a list of (key, type);
        # set by first call to _getSourceMetrics
        self._repoKeyTuple = None  # tuple of repo ID keys, in order; set by first call to addSourceMetrics
        self._repoDTypeList = None  # numpy dtype for repoArr, as a list of (key, type);
        # set by first call to addSourceMetrics

        self._tempDataList = []  # list (one entry per repository)
        # of dict of source ID: tuple of data ID data concatenated with source metric data, where:
        # data ID data is in order self._idKeyTuple
        # source metric data is in order self._sourceKeyTuple
        self.repoInfoList = []  # list of repoInfo

    def _getSourceMetrics(self, idKeyTuple, idValList, sourceTableList):
        """Obtain the desired source measurements from a list of source tables

        Extracts a set of source measurements (specified by sourceKeyTuple) from a list of source tables
        (one per data ID) and saves them as a dict of source ID: list of data

        @param[in] idKeyTuple: a tuple of data ID keys; must be the same for each call
        @param[in] idValList: a list of data ID value tuples;
            each tuple contains values in the order in idKeyTuple
        @param[in] sourceTableList: a list of source tables, one per entry in idValList

        @return a dict of source id: data id tuple + source data tuple
            where source data tuple order matches sourceKeyTuple
            and data id tuple matches self._idKeyTuple (which is set from the first idKeyTuple)

        @raise RuntimeError if idKeyTuple is different than it was for the first call.

        GetRepositoryDataTask.run returns idKeyTuple and idValList; you can easily make
        a subclass of GetRepositoryDataTask that also returns sourceTableList.

        Updates instance variables:
        - self._idKeyTuple if not already set.
        """
        if self._idKeyTuple is None:
            self._idKeyTuple = tuple(idKeyTuple)
            self._idKeyDTypeList = _getDTypeList(keyTuple=self._idKeyTuple,
                                                 valTuple=idValList[0])
        else:
            if self._idKeyTuple != tuple(idKeyTuple):
                raise RuntimeError("idKeyTuple = %s != %s = first idKeyTuple; must be the same each time" %
                                   (idKeyTuple, self._idKeyTuple))

        dataDict = {}
        for idTuple, sourceTable in zip(idValList, sourceTableList):
            if len(sourceTable) == 0:
                continue

            idList = sourceTable.get("id")
            dataList = [sourceTable.get(key) for key in self._sourceKeyTuple]

            if self._sourceDTypeList is None:
                self._sourceDTypeList = [(key, arr.dtype)
                                         for key, arr in zip(self._sourceKeyTuple, dataList)]

            transposedDataList = list(zip(*dataList))
            del dataList

            dataDict.update((srcId, idTuple + tuple(data))
                            for srcId, data in zip(idList, transposedDataList))
        return dataDict

    def addSourceMetrics(self, repoInfo, idKeyTuple, idValList, sourceTableList):
        """Accumulate source measurements from a list of source tables.

        Once you have accumulated all source measurements, call finalize to process the data.

        @param[in] repoInfo: a RepositoryInfo instance
        @param[in] idKeyTuple: a tuple of data ID keys; must be the same for each call
        @param[in] idValList: a list of data ID value tuples;
            each tuple contains values in the order in idKeyTuple
        @param[in] sourceTableList: a list of source tables, one per entry in idValList

        @raise RuntimeError if idKeyTuple is different than it was for the first call.

        Accumulates the data in temporary cache self._tempDataList.

        @return number of sources
        """
        if self._repoKeyTuple is None:
            self._repoKeyTuple = repoInfo.keyTuple
            self._repoDTypeList = repoInfo.dtype

        dataDict = self._getSourceMetrics(idKeyTuple, idValList, sourceTableList)

        self._tempDataList.append(dataDict)
        self.repoInfoList.append(repoInfo)
        return len(dataDict)

    def finalize(self):
        """Process the accumulated source measurements to create the final data products.

        Only call this after you have added all source metrics using addSourceMetrics.

        Reads temporary cache self._tempDataList and then deletes it.
        """
        if len(self._tempDataList) == 0:
            raise RuntimeError("No data found")

        fullSrcIdSet = set()
        for dataIdDict in self._tempDataList:
            fullSrcIdSet.update(iter(dataIdDict.keys()))

        # source data
        sourceArrDType = [("sourceId", int)] + self._idKeyDTypeList + self._sourceDTypeList
        # data for missing sources (only for the data in the source data dict, so excludes srcId)
        nullSourceTuple = tuple(numpy.zeros(1, dtype=self._idKeyDTypeList + self._sourceDTypeList)[0])

        sourceData = [[(srcId,) + srcDataDict.get(srcId, nullSourceTuple) for srcId in fullSrcIdSet]
                      for srcDataDict in self._tempDataList]

        self.sourceArr = numpy.array(sourceData, dtype=sourceArrDType)
        del sourceData

        self.sourceIdDict = dict((srcId, i) for i, srcId in enumerate(fullSrcIdSet))

        # repository data
        repoData = [repoInfo.valTuple for repoInfo in self.repoInfoList]
        self.repoArr = numpy.array(repoData, dtype=self._repoDTypeList)

        self._tempDataList = None


class RepositoryInfo(object):
    """Information about one data repository

    Constructed by RepositoryIterator and used by SourceData.
    """

    def __init__(self, keyTuple, valTuple, dtype, name):
        if len(keyTuple) != len(valTuple):
            raise RuntimeError("lengths of keyTuple=%s and valTuple=%s do not match" % (keyTuple, valTuple))
        self.keyTuple = tuple(keyTuple)
        self.valTuple = tuple(valTuple)
        self.dtype = dtype
        self.name = name


class RepositoryIterator(object):
    """Iterate over a set of data repositories that use a naming convention based on parameter values
    """

    def __init__(self, formatStr, **dataDict):
        """Construct a repository iterator from a dict of name: valueList

        @param[in] formatStr: format string using dictionary notation, e.g.: "%(foo)s_%(bar)d"
        @param[in] **dataDict: name=valueList pairs
        """
        self._formatStr = formatStr
        self._keyTuple = tuple(sorted(dataDict.keys()))
        self._valListOfLists = [numpy.array(dataDict[key]) for key in self._keyTuple]
        self._dtype = [(key, self._valListOfLists[i].dtype)
                       for i, key in enumerate(self._keyTuple)]

    def __iter__(self):
        """Retrieve next RepositoryInfo object
        """
        for valTuple in itertools.product(*self._valListOfLists):
            valDict = dict(zip(self._keyTuple, valTuple))
            name = self.format(valDict)
            yield RepositoryInfo(keyTuple=self._keyTuple, valTuple=valTuple, dtype=self._dtype, name=name)

    def __len__(self):
        """Return the number of items in the iterator"""
        n = 1
        for valTuple in self._valListOfLists:
            n *= len(valTuple)
        return n

    def format(self, valDict):
        """Return formatted string for a specified value dictionary

        @param[in] valDict: a dict of key: value pairs that identify a repository
        """
        return self._formatStr % valDict

    def getKeyTuple(self):
        """Return the a tuple of keys in the same order as items in value tuples
        """
        return self._keyTuple

    def _getDTypeList(self):
        """Get a dtype for a structured array of repository keys
        """
        return self._dtype
