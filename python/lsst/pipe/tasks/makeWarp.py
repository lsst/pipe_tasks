# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from .coaddBase import reorderAndPadList
from collections.abc import Iterable


def reorderRefs(inputRefs, outputSortKeyOrder, dataIdKey):
    """Reorder inputRefs per outputSortKeyOrder.

    Any inputRefs which are lists will be resorted per specified key e.g.,
    'detector.' Only iterables will be reordered, and values can be of type
    `lsst.pipe.base.connections.DeferredDatasetRef` or
    `lsst.daf.butler.core.datasets.ref.DatasetRef`.

    Returned lists of refs have the same length as the outputSortKeyOrder.
    If an outputSortKey not in the inputRef, then it will be padded with None.
    If an inputRef contains an inputSortKey that is not in the
    outputSortKeyOrder it will be removed.

    Parameters
    ----------
    inputRefs : `lsst.pipe.base.connections.QuantizedConnection`
        Input references to be reordered and padded.
    outputSortKeyOrder : `iterable`
        Iterable of values to be compared with inputRef's dataId[dataIdKey].
    dataIdKey : `str`
        The data ID key in the dataRefs to compare with the outputSortKeyOrder.

    Returns
    -------
    inputRefs : `lsst.pipe.base.connections.QuantizedConnection`
        Quantized Connection with sorted DatasetRef values sorted if iterable.
    """
    for connectionName, refs in inputRefs:
        if isinstance(refs, Iterable):
            if hasattr(refs[0], "dataId"):
                inputSortKeyOrder = [ref.dataId[dataIdKey] for ref in refs]
            else:
                inputSortKeyOrder = [handle.datasetRef.dataId[dataIdKey] for handle in refs]
            if inputSortKeyOrder != outputSortKeyOrder:
                setattr(inputRefs, connectionName,
                        reorderAndPadList(refs, inputSortKeyOrder, outputSortKeyOrder))
    return inputRefs
