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

"""
Implementation of thin wrappers to pyarrow.ParquetFile.
"""

import re
import json
from itertools import product
import pyarrow
import pyarrow.parquet
import numpy as np
import pandas as pd


class ParquetTable(object):
    """Thin wrapper to pyarrow's ParquetFile object

    Call `toDataFrame` method to get a `pandas.DataFrame` object,
    optionally passing specific columns.

    The main purpose of having this wrapper rather than directly
    using `pyarrow.ParquetFile` is to make it nicer to load
    selected subsets of columns, especially from dataframes with multi-level
    column indices.

    Instantiated with either a path to a parquet file or a dataFrame

    Parameters
    ----------
    filename : str, optional
        Path to Parquet file.
    dataFrame : dataFrame, optional
    """

    def __init__(self, filename=None, dataFrame=None):
        self.filename = filename
        if filename is not None:
            self._pf = pyarrow.parquet.ParquetFile(filename)
            self._df = None
            self._pandasMd = None
        elif dataFrame is not None:
            self._df = dataFrame
            self._pf = None
        else:
            raise ValueError("Either filename or dataFrame must be passed.")

        self._columns = None
        self._columnIndex = None

    def write(self, filename):
        """Write pandas dataframe to parquet

        Parameters
        ----------
        filename : str
            Path to which to write.
        """
        if self._df is None:
            raise ValueError("df property must be defined to write.")
        table = pyarrow.Table.from_pandas(self._df)
        pyarrow.parquet.write_table(table, filename)

    @property
    def pandasMd(self):
        if self._pf is None:
            raise AttributeError("This property is only accessible if ._pf is set.")
        if self._pandasMd is None:
            self._pandasMd = json.loads(self._pf.metadata.metadata[b"pandas"])
        return self._pandasMd

    @property
    def columnIndex(self):
        """Columns as a pandas Index
        """
        if self._columnIndex is None:
            self._columnIndex = self._getColumnIndex()
        return self._columnIndex

    def _getColumnIndex(self):
        if self._df is not None:
            return self._df.columns
        else:
            return pd.Index(self.columns)

    @property
    def columns(self):
        """List of column names (or column index if df is set)

        This may either be a list of column names, or a
        pandas.Index object describing the column index, depending
        on whether the ParquetTable object is wrapping a ParquetFile
        or a DataFrame.
        """
        if self._columns is None:
            self._columns = self._getColumns()
        return self._columns

    def _getColumns(self):
        if self._df is not None:
            return self._sanitizeColumns(self._df.columns)
        else:
            return self._pf.metadata.schema.names

    def _sanitizeColumns(self, columns):
        return [c for c in columns if c in self.columnIndex]

    def toDataFrame(self, columns=None):
        """Get table (or specified columns) as a pandas DataFrame

        Parameters
        ----------
        columns : list, optional
            Desired columns.  If `None`, then all columns will be
            returned.
        """
        if self._pf is None:
            if columns is None:
                return self._df
            else:
                return self._df[columns]

        if columns is None:
            return self._pf.read().to_pandas()

        df = self._pf.read(columns=columns, use_pandas_metadata=True).to_pandas()
        return df


class MultilevelParquetTable(ParquetTable):
    """Wrapper to access dataframe with multi-level column index from Parquet

    This subclass of `ParquetTable` to handle the multi-level is necessary
    because there is not a convenient way to request specific table subsets
    by level via Parquet through pyarrow, as there is with a `pandas.DataFrame`.

    Additionally, pyarrow stores multilevel index information in a very strange
    way. Pandas stores it as a tuple, so that one can access a single column
    from a pandas dataframe as `df[('ref', 'HSC-G', 'coord_ra')]`.  However, for
    some reason pyarrow saves these indices as "stringified" tuples, such that
    in order to read thissame column from a table written to Parquet, you would
    have to do the following:

        pf = pyarrow.ParquetFile(filename)
        df = pf.read(columns=["('ref', 'HSC-G', 'coord_ra')"])

    See also https://github.com/apache/arrow/issues/1771, where we've raised
    this issue.

    As multilevel-indexed dataframes can be very useful to store data like
    multiple filters' worth of data in the same table, this case deserves a
    wrapper to enable easier access;
    that's what this object is for.  For example,

        parq = MultilevelParquetTable(filename)
        columnDict = {'dataset':'meas',
                      'filter':'HSC-G',
                      'column':['coord_ra', 'coord_dec']}
        df = parq.toDataFrame(columns=columnDict)

    will return just the coordinate columns; the equivalent of calling
    `df['meas']['HSC-G'][['coord_ra', 'coord_dec']]` on the total dataframe,
    but without having to load the whole frame into memory---this reads just
    those columns from disk.  You can also request a sub-table; e.g.,

        parq = MultilevelParquetTable(filename)
        columnDict = {'dataset':'meas',
                      'filter':'HSC-G'}
        df = parq.toDataFrame(columns=columnDict)

    and this will be the equivalent of `df['meas']['HSC-G']` on the total dataframe.

    Parameters
    ----------
    filename : str, optional
        Path to Parquet file.
    dataFrame : dataFrame, optional
    """

    def __init__(self, *args, **kwargs):
        super(MultilevelParquetTable, self).__init__(*args, **kwargs)

        self._columnLevelNames = None

    @property
    def columnLevelNames(self):
        if self._columnLevelNames is None:
            self._columnLevelNames = {
                level: list(np.unique(np.array(self.columns)[:, i]))
                for i, level in enumerate(self.columnLevels)
            }
        return self._columnLevelNames

    @property
    def columnLevels(self):
        """Names of levels in column index
        """
        return self.columnIndex.names

    def _getColumnIndex(self):
        if self._df is not None:
            return super()._getColumnIndex()
        else:
            levelNames = [f["name"] for f in self.pandasMd["column_indexes"]]
            return pd.MultiIndex.from_tuples(self.columns, names=levelNames)

    def _getColumns(self):
        if self._df is not None:
            return super()._getColumns()
        else:
            columns = self._pf.metadata.schema.names
            n = len(self.pandasMd["column_indexes"])
            pattern = re.compile(", ".join(["'(.*)'"] * n))
            matches = [re.search(pattern, c) for c in columns]
            return [m.groups() for m in matches if m is not None]

    def toDataFrame(self, columns=None, droplevels=True):
        """Get table (or specified columns) as a pandas DataFrame

        To get specific columns in specified sub-levels:

            parq = MultilevelParquetTable(filename)
            columnDict = {'dataset':'meas',
                      'filter':'HSC-G',
                      'column':['coord_ra', 'coord_dec']}
            df = parq.toDataFrame(columns=columnDict)

        Or, to get an entire subtable, leave out one level name:

            parq = MultilevelParquetTable(filename)
            columnDict = {'dataset':'meas',
                          'filter':'HSC-G'}
            df = parq.toDataFrame(columns=columnDict)

        Parameters
        ----------
        columns : list or dict, optional
            Desired columns.  If `None`, then all columns will be
            returned.  If a list, then the names of the columns must
            be *exactly* as stored by pyarrow; that is, stringified tuples.
            If a dictionary, then the entries of the dictionary must
            correspond to the level names of the column multi-index
            (that is, the `columnLevels` attribute).  Not every level
            must be passed; if any level is left out, then all entries
            in that level will be implicitly included.
        droplevels : bool
            If True drop levels of column index that have just one entry

        """
        if columns is None:
            if self._pf is None:
                return self._df
            else:
                return self._pf.read().to_pandas()

        if isinstance(columns, dict):
            columns = self._colsFromDict(columns)

        if self._pf is None:
            try:
                df = self._df[columns]
            except (AttributeError, KeyError):
                newColumns = [c for c in columns if c in self.columnIndex]
                if not newColumns:
                    raise ValueError("None of the requested columns ({}) are available!".format(columns))
                df = self._df[newColumns]
        else:
            pfColumns = self._stringify(columns)
            try:
                df = self._pf.read(columns=pfColumns, use_pandas_metadata=True).to_pandas()
            except (AttributeError, KeyError):
                newColumns = [c for c in columns if c in self.columnIndex]
                if not newColumns:
                    raise ValueError("None of the requested columns ({}) are available!".format(columns))
                pfColumns = self._stringify(newColumns)
                df = self._pf.read(columns=pfColumns, use_pandas_metadata=True).to_pandas()

        if droplevels:
            # Drop levels of column index that have just one entry
            levelsToDrop = [n for lev, n in zip(df.columns.levels, df.columns.names) if len(lev) == 1]

            # Prevent error when trying to drop *all* columns
            if len(levelsToDrop) == len(df.columns.names):
                levelsToDrop.remove(df.columns.names[-1])

            df.columns = df.columns.droplevel(levelsToDrop)

        return df

    def _colsFromDict(self, colDict):
        new_colDict = {}
        for i, lev in enumerate(self.columnLevels):
            if lev in colDict:
                if isinstance(colDict[lev], str):
                    new_colDict[lev] = [colDict[lev]]
                else:
                    new_colDict[lev] = colDict[lev]
            else:
                new_colDict[lev] = self.columnIndex.levels[i]

        levelCols = [new_colDict[lev] for lev in self.columnLevels]
        cols = product(*levelCols)
        return list(cols)

    def _stringify(self, cols):
        return [str(c) for c in cols]
