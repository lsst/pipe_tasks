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

__all__ = ["TableVStack"]

import astropy.table
from astro_metadata_translator.headers import merge_headers
import numpy as np

from lsst.obs.base.utils import strip_provenance_from_fits_header


class TableVStack:
    """A helper class for stacking astropy tables without having them all in
    memory at once.

    Parameters
    ----------
    capacity : `int`
        Full size of the final table.

    Notes
    -----
    Unlike `astropy.table.vstack`, this class requires all tables to have the
    exact same columns (it's slightly more strict than even the
    ``join_type="exact"`` argument to `astropy.table.vstack`).
    """

    def __init__(self, capacity):
        self.index = 0
        self.capacity = capacity
        self.result = None

    @classmethod
    def from_handles(cls, handles):
        """Construct from an iterable of
        `lsst.daf.butler.DeferredDatasetHandle`.

        Parameters
        ----------
        handles : `~collections.abc.Iterable` [ \
                `lsst.daf.butler.DeferredDatasetHandle` ]
            Iterable of handles.   Must have a storage class that supports the
            "rowcount" component, which is all that will be fetched.

        Returns
        -------
        vstack : `TableVStack`
            An instance of this class, initialized with capacity equal to the
            sum of the rowcounts of all the given table handles.
        """
        capacity = sum(handle.get(component="rowcount") for handle in handles)
        return cls(capacity=capacity)

    def extend(self, table):
        """Add a single table to the stack.

        Parameters
        ----------
        table : `astropy.table.Table`
            An astropy table instance.
        """
        if self.result is None:
            self.result = astropy.table.Table()
            for name in table.colnames:
                column = table[name]
                column_cls = type(column)
                self.result[name] = column_cls.info.new_like([column], self.capacity, name=name)
                self.result[name][:len(table)] = column
            self.index = len(table)
            self.result.meta = table.meta.copy()
        else:
            next_index = self.index + len(table)
            if set(self.result.colnames) != set(table.colnames):
                raise TypeError(
                    "Inconsistent columns in concatentation: "
                    f"{set(self.result.colnames).symmetric_difference(table.colnames)}"
                )
            for name in table.colnames:
                out_col = self.result[name]
                in_col = table[name]
                if out_col.dtype != in_col.dtype:
                    raise TypeError(f"Type mismatch on column {name!r}: {out_col.dtype} != {in_col.dtype}.")
                self.result[name][self.index:next_index] = table[name]
            self.index = next_index
            # Butler provenance should be stripped on merge. It will be
            # added by butler on write. No attempt is made here to combine
            # provenance from multiple input tables.
            self.result.meta = merge_headers([self.result.meta, table.meta], mode="drop")
            strip_provenance_from_fits_header(self.result.meta)

    @classmethod
    def vstack_handles(cls, handles):
        """Vertically stack tables represented by deferred dataset handles.

        Parameters
        ----------
        handles : `~collections.abc.Iterable` [ \
                `lsst.daf.butler.DeferredDatasetHandle` ]
            Iterable of handles.   Must have the "ArrowAstropy" storage class
            and identical columns.

        Returns
        -------
        table : `astropy.table.Table`
            Concatenated table with the same columns as each input table and
            the rows of all of them.
        """
        handles = tuple(handles)  # guard against single-pass iterators
        # Ensure that zero length catalogs are not included
        rowcount = tuple(handle.get(component="rowcount") for handle in handles)
        handles = tuple(handle for handle, count in zip(handles, rowcount) if count > 0)

        vstack = cls.from_handles(handles)
        for handle in handles:
            vstack.extend(handle.get())
        return vstack.result
