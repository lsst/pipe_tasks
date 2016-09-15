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


def groupDataRefs(keys, dataRefIterable):
    """Group data references by data identifier value-tuple.

    Value-tuples are built from the values of the given keys.
    The effect is that the data references in each group have the same
    values for the provided keys.

    @param keys: List of keys to consider when grouping (order is important)
    @param dataRefIterable: Iterable of data references to group
    @return Dict of <value-tuple>: <list of data references for group>
    """
    groupDict = dict()
    for dataRef in dataRefIterable:
        dataId = dataRef.dataId
        values = tuple(dataId[key] for key in keys)  # NOT dataId.values() as we must preserve order
        group = groupDict.get(values)
        if group:
            group.append(dataRef)
        else:
            groupDict[values] = [dataRef]

    return groupDict


def groupPatchExposures(patchDataRef, calexpDataRefList, coaddDatasetType="deepCoadd",
                        tempExpDatasetType="deepCoadd_tempExp"):
    """Group calibrated exposures overlapping a patch by the warped
    (temporary) exposure they contribute to.

    For example, if the instrument has a mosaic camera, each group would
    consist of the subset of CCD exposures from a single camera exposure
    that potentially overlap the patch.

    @return Struct with:
    - groups: Dict of <group tuple>: <list of data references for group>
    - keys: List of keys for group tuple
    """
    butler = patchDataRef.getButler()
    tempExpKeys = butler.getKeys(datasetType=tempExpDatasetType)
    coaddKeys = sorted(butler.getKeys(datasetType=coaddDatasetType))
    keys = sorted(set(tempExpKeys) - set(coaddKeys))  # Keys that will specify an exposure
    patchId = patchDataRef.dataId
    groups = groupDataRefs(keys, calexpDataRefList)

    # Supplement the groups with the coadd-specific information (e.g., tract, patch; these are constant)
    coaddValues = tuple(patchId[k] for k in coaddKeys)
    groups = dict((k + coaddValues, v) for k, v in groups.iteritems())
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
