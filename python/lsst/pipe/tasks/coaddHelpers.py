#
# LSST Data Management System
# Copyright 2008-2013 LSST Corporation.
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

from lsst.pipe.base import Struct

"""Helper functions for coaddition.

We often want to use a data reference as a key in a dict (e.g., inputs as a
function of data reference for a warp/tempExp), but neither data references
(lsst.daf.persistence.ButlerDataRef) nor data identifiers (dict) are hashable.
One solution is to use tuples (which are hashable) of the data identifier
values, and carry the data identifier keys separately.  Doing the key/value
gymnastics can be annoying, so we provide these helper functions to do this.
"""

def groupExposures(keys, dataRefList, checkDataset=None):
    """Group data references into groups with the same values specified by keys

    @param keys: List of keys to consider when grouping
    @param dataRefList: List of data references to group
    @param checkDataset: If not None, include only if dataset exists
    @return Dict of <group tuple>: <list of data references for group>
    """
    groupDict = dict()
    for dataRef in dataRefList:
        dataId = dataRef.dataId
        if checkDataset is not None and not dataRef.datasetExists(checkDataset):
            self.log.warn("Could not find %s; skipping it" % (checkDataset, dataId,))
            continue

        values = tuple(dataId[key] for key in keys) # NOT dataId.values() as we must preserve order
        group = groupDict.get(values)
        if group:
            group.append(dataRef)
        else:
            groupDict[values] = [dataRef]

    return groupDict

def groupPatchExposures(patchRef, calexpRefList, coaddDataset="deepCoadd", tempExpDataset="deepCoadd_tempExp",
                        calexpDataset="calexp", checkExist=True):
    """Group calexp references into groups of exposures

    @param patchRef: Data reference for patch
    @param calexpRefList: List of data references for calexps
    @param coaddDataset: Dataset name for tempExps
    @param calexpDataset: Dataset name for calexp
    @return Struct with:
    - groups: Dict of <group tuple>: <list of data references for group>
    - keys: List of keys for group tuple
    """
    butler = patchRef.getButler()
    tempExpKeys = sorted(butler.getKeys(datasetType=tempExpDataset))
    coaddKeys = sorted(butler.getKeys(datasetType=coaddDataset))
    keys = sorted(set(tempExpKeys) - set(coaddKeys)) # Keys that will specify an exposure
    patchId = patchRef.dataId
    groups = groupExposures(keys, calexpRefList, checkDataset=calexpDataset if checkExist else None)

    # Supplement the groups with the coadd-specific information (e.g., tract, patch; these are constant)
    coaddValues = tuple(patchId[k] for k in coaddKeys)
    groups = dict((k + coaddValues, v) for k,v in groups.iteritems())
    keys += tuple(coaddKeys)

    return Struct(groups=groups, keys=keys)

def getGroupDataId(groupTuple, keys):
    """Reconstitute a data identifier from a tuple and corresponding keys

    @param groupTuple: Tuple with values specifying a group
    @param keys: List of keys for group tuple
    @return Data identifier dict
    """
    if len(groupTuple) != len(keys):
        raise RuntimeError("Number of values (%d) and keys (%d) do not match" % (len(groupTuple), len(keys)))
    return dict(zip(keys, groupTuple))

def getGroupDataRef(butler, datasetType, groupTuple, keys):
    """Construct a data reference from a tuple and corresponding keys

    @param butler: Data butler
    @param datasetType: Name of dataset
    @param groupTuple: Tuple with values specifying a group
    @param keys: List of keys for group tuple
    @return Data reference
    """
    dataId = getGroupDataId(groupTuple, keys)
    return butler.dataRef(datasetType=datasetType, dataId=dataId)
