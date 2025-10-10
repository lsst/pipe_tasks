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

__all__ = ["init_fromDict", "Functor", "CompositeFunctor", "mag_aware_eval",
           "CustomFunctor", "Column", "Index", "CoordColumn", "RAColumn",
           "DecColumn", "SinglePrecisionFloatColumn", "HtmIndex20", "fluxName", "fluxErrName", "Mag",
           "MagErr", "MagDiff", "Color", "DeconvolvedMoments", "SdssTraceSize",
           "PsfSdssTraceSizeDiff", "HsmTraceSize", "PsfHsmTraceSizeDiff",
           "HsmFwhm", "E1", "E2", "RadiusFromQuadrupole", "LocalWcs",
           "ComputePixelScale", "ConvertPixelToArcseconds",
           "ConvertPixelSqToArcsecondsSq",
           "ConvertDetectorAngleToPositionAngle",
           "ReferenceBand", "Photometry",
           "NanoJansky", "NanoJanskyErr", "LocalPhotometry", "LocalNanojansky",
           "LocalNanojanskyErr", "LocalDipoleMeanFlux",
           "LocalDipoleMeanFluxErr", "LocalDipoleDiffFlux",
           "LocalDipoleDiffFluxErr", "Ebv",
           "MomentsIuuSky", "MomentsIvvSky", "MomentsIuvSky",
           "CorrelationIuuSky", "CorrelationIvvSky", "CorrelationIuvSky",
           "PositionAngleFromMoments", "PositionAngleFromCorrelation",
           "SemimajorAxisFromMoments", "SemimajorAxisFromCorrelation",
           "SemiminorAxisFromMoments", "SemiminorAxisFromCorrelation",
           ]

import logging
import os
import os.path
import re
import warnings
from contextlib import redirect_stdout
from itertools import product

import astropy.units as u
import lsst.geom as geom
import lsst.sphgeom as sphgeom
import numpy as np
import pandas as pd
import yaml
from astropy.coordinates import SkyCoord
from lsst.daf.butler import DeferredDatasetHandle
from lsst.pipe.base import InMemoryDatasetHandle
from lsst.utils import doImport
from lsst.utils.introspection import get_full_type_name


def init_fromDict(initDict, basePath='lsst.pipe.tasks.functors',
                  typeKey='functor', name=None):
    """Initialize an object defined in a dictionary.

    The object needs to be importable as f'{basePath}.{initDict[typeKey]}'.
    The positional and keyword arguments (if any) are contained in "args" and
    "kwargs" entries in the dictionary, respectively.
    This is used in `~lsst.pipe.tasks.functors.CompositeFunctor.from_yaml` to
    initialize a composite functor from a specification in a YAML file.

    Parameters
    ----------
    initDict : dictionary
        Dictionary describing object's initialization.
        Must contain an entry keyed by ``typeKey`` that is the name of the
        object, relative to ``basePath``.
    basePath : str
        Path relative to module in which ``initDict[typeKey]`` is defined.
    typeKey : str
        Key of ``initDict`` that is the name of the object (relative to
        ``basePath``).
    """
    initDict = initDict.copy()
    # TO DO: DM-21956 We should be able to define functors outside this module
    pythonType = doImport(f'{basePath}.{initDict.pop(typeKey)}')
    args = []
    if 'args' in initDict:
        args = initDict.pop('args')
        if isinstance(args, str):
            args = [args]
    try:
        element = pythonType(*args, **initDict)
    except Exception as e:
        message = f'Error in constructing functor "{name}" of type {pythonType.__name__} with args: {args}'
        raise type(e)(message, e.args)
    return element


class Functor(object):
    """Define and execute a calculation on a DataFrame or Handle holding a
    DataFrame.

    The `__call__` method accepts either a `~pandas.DataFrame` object or a
    `~lsst.daf.butler.DeferredDatasetHandle` or
    `~lsst.pipe.base.InMemoryDatasetHandle`, and returns the
    result of the calculation as a single column.
    Each functor defines what columns are needed for the calculation, and only
    these columns are read from the dataset handle.

    The action of `__call__` consists of two steps: first, loading the
    necessary columns from disk into memory as a `~pandas.DataFrame` object;
    and second, performing the computation on this DataFrame and returning the
    result.

    To define a new `Functor`, a subclass must define a `_func` method,
    that takes a `~pandas.DataFrame` and returns result in a `~pandas.Series`.
    In addition, it must define the following attributes:

    * `_columns`: The columns necessary to perform the calculation
    * `name`: A name appropriate for a figure axis label
    * `shortname`: A name appropriate for use as a dictionary key

    On initialization, a `Functor` should declare what band (``filt`` kwarg)
    and dataset (e.g. ``'ref'``, ``'meas'``, ``'forced_src'``) it is intended
    to be applied to.
    This enables the `_get_data` method to extract the proper columns from the
    underlying data.
    If not specified, the dataset will fall back on the `_defaultDataset`
    attribute.
    If band is not specified and ``dataset`` is anything other than ``'ref'``,
    then an error will be raised when trying to perform the calculation.

    Originally, `Functor` was set up to expect datasets formatted like the
    ``deepCoadd_obj`` dataset; that is, a DataFrame with a multi-level column
    index, with the levels of the column index being ``band``, ``dataset``, and
    ``column``.
    It has since been generalized to apply to DataFrames without multi-level
    indices and multi-level indices with just ``dataset`` and ``column``
    levels.
    In addition, the `_get_data` method that reads the columns from the
    underlying data will return a DataFrame with column index levels defined by
    the `_dfLevels` attribute; by default, this is ``column``.

    The `_dfLevels` attributes should generally not need to be changed, unless
    `_func` needs columns from multiple filters or datasets to do the
    calculation.
    An example of this is the `~lsst.pipe.tasks.functors.Color` functor, for
    which `_dfLevels = ('band', 'column')`, and `_func` expects the DataFrame
    it gets to have those levels in the column index.

    Parameters
    ----------
    filt : str
        Band upon which to do the calculation.

    dataset : str
        Dataset upon which to do the calculation (e.g., 'ref', 'meas',
        'forced_src').
    """

    _defaultDataset = 'ref'
    _dfLevels = ('column',)
    _defaultNoDup = False

    def __init__(self, filt=None, dataset=None, noDup=None):
        self.filt = filt
        self.dataset = dataset if dataset is not None else self._defaultDataset
        self._noDup = noDup
        self.log = logging.getLogger(type(self).__name__)

    @property
    def noDup(self):
        """Do not explode by band if used on object table."""
        if self._noDup is not None:
            return self._noDup
        else:
            return self._defaultNoDup

    @property
    def columns(self):
        """Columns required to perform calculation."""
        if not hasattr(self, '_columns'):
            raise NotImplementedError('Must define columns property or _columns attribute')
        return self._columns

    def _get_data_columnLevels(self, data, columnIndex=None):
        """Gets the names of the column index levels.

        This should only be called in the context of a multilevel table.

        Parameters
        ----------
        data : various
            The data to be read, can be a
            `~lsst.daf.butler.DeferredDatasetHandle` or
            `~lsst.pipe.base.InMemoryDatasetHandle`.
        columnIndex (optional): pandas `~pandas.Index` object
            If not passed, then it is read from the
            `~lsst.daf.butler.DeferredDatasetHandle`
            for `~lsst.pipe.base.InMemoryDatasetHandle`.
        """
        if columnIndex is None:
            columnIndex = data.get(component="columns")
        return columnIndex.names

    def _get_data_columnLevelNames(self, data, columnIndex=None):
        """Gets the content of each of the column levels for a multilevel
        table.
        """
        if columnIndex is None:
            columnIndex = data.get(component="columns")

        columnLevels = columnIndex.names
        columnLevelNames = {
            level: list(np.unique(np.array([c for c in columnIndex])[:, i]))
            for i, level in enumerate(columnLevels)
        }
        return columnLevelNames

    def _colsFromDict(self, colDict, columnIndex=None):
        """Converts dictionary column specficiation to a list of columns."""
        new_colDict = {}
        columnLevels = self._get_data_columnLevels(None, columnIndex=columnIndex)

        for i, lev in enumerate(columnLevels):
            if lev in colDict:
                if isinstance(colDict[lev], str):
                    new_colDict[lev] = [colDict[lev]]
                else:
                    new_colDict[lev] = colDict[lev]
            else:
                new_colDict[lev] = columnIndex.levels[i]

        levelCols = [new_colDict[lev] for lev in columnLevels]
        cols = list(product(*levelCols))
        colsAvailable = [col for col in cols if col in columnIndex]
        return colsAvailable

    def multilevelColumns(self, data, columnIndex=None, returnTuple=False):
        """Returns columns needed by functor from multilevel dataset.

        To access tables with multilevel column structure, the
        `~lsst.daf.butler.DeferredDatasetHandle` or
        `~lsst.pipe.base.InMemoryDatasetHandle` needs to be passed
        either a list of tuples or a dictionary.

        Parameters
        ----------
        data : various
            The data as either `~lsst.daf.butler.DeferredDatasetHandle`, or
            `~lsst.pipe.base.InMemoryDatasetHandle`.
        columnIndex (optional): pandas `~pandas.Index` object
            Either passed or read in from
            `~lsst.daf.butler.DeferredDatasetHandle`.
        `returnTuple` : `bool`
            If true, then return a list of tuples rather than the column
            dictionary specification.
            This is set to `True` by `CompositeFunctor` in order to be able to
            combine columns from the various component functors.

        """
        if not isinstance(data, (DeferredDatasetHandle, InMemoryDatasetHandle)):
            raise RuntimeError(f"Unexpected data type. Got {get_full_type_name(data)}.")

        if columnIndex is None:
            columnIndex = data.get(component="columns")

        # Confirm that the dataset has the column levels the functor is
        # expecting it to have.
        columnLevels = self._get_data_columnLevels(data, columnIndex)

        columnDict = {'column': self.columns,
                      'dataset': self.dataset}
        if self.filt is None:
            columnLevelNames = self._get_data_columnLevelNames(data, columnIndex)
            if "band" in columnLevels:
                if self.dataset == "ref":
                    columnDict["band"] = columnLevelNames["band"][0]
                else:
                    raise ValueError(f"'filt' not set for functor {self.name}"
                                     f"(dataset {self.dataset}) "
                                     "and DataFrame "
                                     "contains multiple filters in column index. "
                                     "Set 'filt' or set 'dataset' to 'ref'.")
        else:
            columnDict['band'] = self.filt

        if returnTuple:
            return self._colsFromDict(columnDict, columnIndex=columnIndex)
        else:
            return columnDict

    def _func(self, df, dropna=True):
        raise NotImplementedError('Must define calculation on DataFrame')

    def _get_columnIndex(self, data):
        """Return columnIndex."""

        if isinstance(data, (DeferredDatasetHandle, InMemoryDatasetHandle)):
            return data.get(component="columns")
        else:
            return None

    def _get_data(self, data):
        """Retrieve DataFrame necessary for calculation.

        The data argument can be a `~pandas.DataFrame`, a
        `~lsst.daf.butler.DeferredDatasetHandle`, or
        an `~lsst.pipe.base.InMemoryDatasetHandle`.

        Returns a DataFrame upon which `self._func` can act.
        """
        # We wrap a DataFrame in a handle here to take advantage of the
        # DataFrame delegate DataFrame column wrangling abilities.
        if isinstance(data, pd.DataFrame):
            _data = InMemoryDatasetHandle(data, storageClass="DataFrame")
        elif isinstance(data, (DeferredDatasetHandle, InMemoryDatasetHandle)):
            _data = data
        else:
            raise RuntimeError(f"Unexpected type provided for data. Got {get_full_type_name(data)}.")

        # First thing to do: check to see if the data source has a multilevel
        # column index or not.
        columnIndex = self._get_columnIndex(_data)
        is_multiLevel = isinstance(columnIndex, pd.MultiIndex)

        # Get proper columns specification for this functor.
        if is_multiLevel:
            columns = self.multilevelColumns(_data, columnIndex=columnIndex)
        else:
            columns = self.columns

        # Load in-memory DataFrame with appropriate columns the gen3 way.
        df = _data.get(parameters={"columns": columns})

        # Drop unnecessary column levels.
        if is_multiLevel:
            df = self._setLevels(df)

        return df

    def _setLevels(self, df):
        levelsToDrop = [n for n in df.columns.names if n not in self._dfLevels]
        df.columns = df.columns.droplevel(levelsToDrop)
        return df

    def _dropna(self, vals):
        return vals.dropna()

    def __call__(self, data, dropna=False):
        df = self._get_data(data)
        try:
            vals = self._func(df)
        except Exception as e:
            self.log.error("Exception in %s call: %s: %s", self.name, type(e).__name__, e)
            vals = self.fail(df)
        if dropna:
            vals = self._dropna(vals)

        return vals

    def difference(self, data1, data2, **kwargs):
        """Computes difference between functor called on two different
        DataFrame/Handle objects.
        """
        return self(data1, **kwargs) - self(data2, **kwargs)

    def fail(self, df):
        return pd.Series(np.full(len(df), np.nan), index=df.index)

    @property
    def name(self):
        """Full name of functor (suitable for figure labels)."""
        return NotImplementedError

    @property
    def shortname(self):
        """Short name of functor (suitable for column name/dict key)."""
        return self.name


class CompositeFunctor(Functor):
    """Perform multiple calculations at once on a catalog.

    The role of a `CompositeFunctor` is to group together computations from
    multiple functors.
    Instead of returning `~pandas.Series` a `CompositeFunctor` returns a
    `~pandas.DataFrame`, with the column names being the keys of ``funcDict``.

    The `columns` attribute of a `CompositeFunctor` is the union of all columns
    in all the component functors.

    A `CompositeFunctor` does not use a `_func` method itself; rather, when a
    `CompositeFunctor` is called, all its columns are loaded at once, and the
    resulting DataFrame is passed to the `_func` method of each component
    functor.
    This has the advantage of only doing I/O (reading from parquet file) once,
    and works because each individual `_func` method of each component functor
    does not care if there are *extra* columns in the DataFrame being passed;
    only that it must contain *at least* the `columns` it expects.

    An important and useful class method is `from_yaml`, which takes as an
    argument the path to a YAML file specifying a collection of functors.

    Parameters
    ----------
    funcs : `dict` or `list`
        Dictionary or list of functors.
        If a list, then it will be converted into a dictonary according to the
        `.shortname` attribute of each functor.
    """
    dataset = None
    name = "CompositeFunctor"

    def __init__(self, funcs, **kwargs):

        if type(funcs) is dict:
            self.funcDict = funcs
        else:
            self.funcDict = {f.shortname: f for f in funcs}

        self._filt = None

        super().__init__(**kwargs)

    @property
    def filt(self):
        return self._filt

    @filt.setter
    def filt(self, filt):
        if filt is not None:
            for _, f in self.funcDict.items():
                f.filt = filt
        self._filt = filt

    def update(self, new):
        """Update the functor with new functors."""
        if isinstance(new, dict):
            self.funcDict.update(new)
        elif isinstance(new, CompositeFunctor):
            self.funcDict.update(new.funcDict)
        else:
            raise TypeError('Can only update with dictionary or CompositeFunctor.')

        # Make sure new functors have the same 'filt' set.
        if self.filt is not None:
            self.filt = self.filt

    @property
    def columns(self):
        return list(set([x for y in [f.columns for f in self.funcDict.values()] for x in y]))

    def multilevelColumns(self, data, **kwargs):
        # Get the union of columns for all component functors.
        # Note the need to have `returnTuple=True` here.
        return list(
            set(
                [
                    x
                    for y in [
                        f.multilevelColumns(data, returnTuple=True, **kwargs) for f in self.funcDict.values()
                    ]
                    for x in y
                ]
            )
        )

    def __call__(self, data, **kwargs):
        """Apply the functor to the data table.

        Parameters
        ----------
        data : various
            The data represented as `~lsst.daf.butler.DeferredDatasetHandle`,
            `~lsst.pipe.base.InMemoryDatasetHandle`, or `~pandas.DataFrame`.
            The table or a pointer to a table on disk from which columns can
            be accessed.
        """
        if isinstance(data, pd.DataFrame):
            _data = InMemoryDatasetHandle(data, storageClass="DataFrame")
        elif isinstance(data, (DeferredDatasetHandle, InMemoryDatasetHandle)):
            _data = data
        else:
            raise RuntimeError(f"Unexpected type provided for data. Got {get_full_type_name(data)}.")

        columnIndex = self._get_columnIndex(_data)

        if isinstance(columnIndex, pd.MultiIndex):
            columns = self.multilevelColumns(_data, columnIndex=columnIndex)
            df = _data.get(parameters={"columns": columns})

            valDict = {}
            for k, f in self.funcDict.items():
                try:
                    subdf = f._setLevels(
                        df[f.multilevelColumns(_data, returnTuple=True, columnIndex=columnIndex)]
                    )
                    valDict[k] = f._func(subdf)
                except Exception as e:
                    self.log.exception(
                        "Exception in %s (funcs: %s) call: %s",
                        self.name,
                        str(list(self.funcDict.keys())),
                        type(e).__name__,
                    )
                    try:
                        valDict[k] = f.fail(subdf)
                    except NameError:
                        raise e

        else:
            df = _data.get(parameters={"columns": self.columns})

            valDict = {k: f._func(df) for k, f in self.funcDict.items()}

            # Check that output columns are actually columns.
            for name, colVal in valDict.items():
                if len(colVal.shape) != 1:
                    raise RuntimeError("Transformed column '%s' is not the shape of a column. "
                                       "It is shaped %s and type %s." % (name, colVal.shape, type(colVal)))

        try:
            valDf = pd.concat(valDict, axis=1)
        except TypeError:
            print([(k, type(v)) for k, v in valDict.items()])
            raise

        if kwargs.get('dropna', False):
            valDf = valDf.dropna(how='any')

        return valDf

    @classmethod
    def renameCol(cls, col, renameRules):
        if renameRules is None:
            return col
        for old, new in renameRules:
            if col.startswith(old):
                col = col.replace(old, new)
        return col

    @classmethod
    def from_file(cls, filename, **kwargs):
        # Allow environment variables in the filename.
        filename = os.path.expandvars(filename)
        with open(filename) as f:
            translationDefinition = yaml.safe_load(f)

        return cls.from_yaml(translationDefinition, **kwargs)

    @classmethod
    def from_yaml(cls, translationDefinition, **kwargs):
        funcs = {}
        for func, val in translationDefinition['funcs'].items():
            funcs[func] = init_fromDict(val, name=func)

        if 'flag_rename_rules' in translationDefinition:
            renameRules = translationDefinition['flag_rename_rules']
        else:
            renameRules = None

        if 'calexpFlags' in translationDefinition:
            for flag in translationDefinition['calexpFlags']:
                funcs[cls.renameCol(flag, renameRules)] = Column(flag, dataset='calexp')

        if 'refFlags' in translationDefinition:
            for flag in translationDefinition['refFlags']:
                funcs[cls.renameCol(flag, renameRules)] = Column(flag, dataset='ref')

        if 'forcedFlags' in translationDefinition:
            for flag in translationDefinition['forcedFlags']:
                funcs[cls.renameCol(flag, renameRules)] = Column(flag, dataset='forced_src')

        if 'flags' in translationDefinition:
            for flag in translationDefinition['flags']:
                funcs[cls.renameCol(flag, renameRules)] = Column(flag, dataset='meas')

        return cls(funcs, **kwargs)


def mag_aware_eval(df, expr, log):
    """Evaluate an expression on a DataFrame, knowing what the 'mag' function
    means.

    Builds on `pandas.DataFrame.eval`, which parses and executes math on
    DataFrames.

    Parameters
    ----------
    df : ~pandas.DataFrame
        DataFrame on which to evaluate expression.

    expr : str
        Expression.
    """
    try:
        expr_new = re.sub(r'mag\((\w+)\)', r'-2.5*log(\g<1>)/log(10)', expr)
        val = df.eval(expr_new)
    except Exception as e:  # Should check what actually gets raised
        log.error("Exception in mag_aware_eval: %s: %s", type(e).__name__, e)
        expr_new = re.sub(r'mag\((\w+)\)', r'-2.5*log(\g<1>_instFlux)/log(10)', expr)
        val = df.eval(expr_new)
    return val


class CustomFunctor(Functor):
    """Arbitrary computation on a catalog.

    Column names (and thus the columns to be loaded from catalog) are found by
    finding all words and trying to ignore all "math-y" words.

    Parameters
    ----------
    expr : str
        Expression to evaluate, to be parsed and executed by
        `~lsst.pipe.tasks.functors.mag_aware_eval`.
    """
    _ignore_words = ('mag', 'sin', 'cos', 'exp', 'log', 'sqrt')

    def __init__(self, expr, **kwargs):
        self.expr = expr
        super().__init__(**kwargs)

    @property
    def name(self):
        return self.expr

    @property
    def columns(self):
        flux_cols = re.findall(r'mag\(\s*(\w+)\s*\)', self.expr)

        cols = [c for c in re.findall(r'[a-zA-Z_]+', self.expr) if c not in self._ignore_words]
        not_a_col = []
        for c in flux_cols:
            if not re.search('_instFlux$', c):
                cols.append(f'{c}_instFlux')
                not_a_col.append(c)
            else:
                cols.append(c)

        return list(set([c for c in cols if c not in not_a_col]))

    def _func(self, df):
        return mag_aware_eval(df, self.expr, self.log)


class Column(Functor):
    """Get column with a specified name."""

    def __init__(self, col, **kwargs):
        self.col = col
        super().__init__(**kwargs)

    @property
    def name(self):
        return self.col

    @property
    def columns(self):
        return [self.col]

    def _func(self, df):
        return df[self.col]


class Index(Functor):
    """Return the value of the index for each object."""

    columns = ['coord_ra']  # Just a dummy; something has to be here.
    _defaultDataset = 'ref'
    _defaultNoDup = True

    def _func(self, df):
        return pd.Series(df.index, index=df.index)


class CoordColumn(Column):
    """Base class for coordinate column, in degrees."""
    _radians = True

    def __init__(self, col, **kwargs):
        super().__init__(col, **kwargs)

    def _func(self, df):
        # Must not modify original column in case that column is used by
        # another functor.
        output = df[self.col] * 180 / np.pi if self._radians else df[self.col]
        return output


class RAColumn(CoordColumn):
    """Right Ascension, in degrees."""
    name = 'RA'
    _defaultNoDup = True

    def __init__(self, **kwargs):
        super().__init__('coord_ra', **kwargs)

    def __call__(self, catalog, **kwargs):
        return super().__call__(catalog, **kwargs)


class DecColumn(CoordColumn):
    """Declination, in degrees."""
    name = 'Dec'
    _defaultNoDup = True

    def __init__(self, **kwargs):
        super().__init__('coord_dec', **kwargs)

    def __call__(self, catalog, **kwargs):
        return super().__call__(catalog, **kwargs)


class RAErrColumn(CoordColumn):
    """Uncertainty in Right Ascension, in degrees."""
    name = 'RAErr'
    _defaultNoDup = True

    def __init__(self, **kwargs):
        super().__init__('coord_raErr', **kwargs)


class DecErrColumn(CoordColumn):
    """Uncertainty in declination, in degrees."""
    name = 'DecErr'
    _defaultNoDup = True

    def __init__(self, **kwargs):
        super().__init__('coord_decErr', **kwargs)


class RADecCovColumn(Column):
    """Coordinate covariance column, in degrees."""
    _radians = True
    name = 'RADecCov'
    _defaultNoDup = True

    def __init__(self, **kwargs):
        super().__init__('coord_ra_dec_Cov', **kwargs)

    def _func(self, df):
        # Must not modify original column in case that column is used by
        # another functor.
        output = df[self.col]*(180/np.pi)**2 if self._radians else df[self.col]
        return output


class MultibandColumn(Column):
    """A column with a band in a multiband table."""
    def __init__(self, col, band_to_check, **kwargs):
        self._band_to_check = band_to_check
        super().__init__(col=col, **kwargs)

    @property
    def band_to_check(self):
        return self._band_to_check


class MultibandSinglePrecisionFloatColumn(MultibandColumn):
    """A float32 MultibandColumn"""
    def _func(self, df):
        return super()._func(df).astype(np.float32)


class SinglePrecisionFloatColumn(Column):
    """Return a column cast to a single-precision float."""

    def _func(self, df):
        return df[self.col].astype(np.float32)


class HtmIndex20(Functor):
    """Compute the level 20 HtmIndex for the catalog.

    Notes
    -----
    This functor was implemented to satisfy requirements of old APDB interface
    which required the ``pixelId`` column in DiaObject with HTM20 index.
    The APDB interface had migrated to not need that information, but we keep
    this class in case it may be useful for something else.
    """
    name = "Htm20"
    htmLevel = 20
    _radians = True

    def __init__(self, ra, dec, **kwargs):
        self.pixelator = sphgeom.HtmPixelization(self.htmLevel)
        self.ra = ra
        self.dec = dec
        self._columns = [self.ra, self.dec]
        super().__init__(**kwargs)

    def _func(self, df):

        def computePixel(row):
            if self._radians:
                sphPoint = geom.SpherePoint(row[self.ra],
                                            row[self.dec],
                                            geom.radians)
            else:
                sphPoint = geom.SpherePoint(row[self.ra],
                                            row[self.dec],
                                            geom.degrees)
            return self.pixelator.index(sphPoint.getVector())

        return df.apply(computePixel, axis=1, result_type='reduce').astype('int64')


def fluxName(col):
    """Append _instFlux to the column name if it doesn't have it already."""
    if not col.endswith('_instFlux'):
        col += '_instFlux'
    return col


def fluxErrName(col):
    """Append _instFluxErr to the column name if it doesn't have it already."""
    if not col.endswith('_instFluxErr'):
        col += '_instFluxErr'
    return col


class Mag(Functor):
    """Compute calibrated magnitude.

    Returns the flux at mag=0.
    The default ``fluxMag0`` is 63095734448.0194, which is default for HSC.
    TO DO: This default should be made configurable in DM-21955.

    This calculation hides warnings about invalid values and dividing by zero.

    As with all functors, a ``dataset`` and ``filt`` kwarg should be provided
    upon initialization.
    Unlike the default `Functor`, however, the default dataset for a `Mag` is
    ``'meas'``, rather than ``'ref'``.

    Parameters
    ----------
    col : `str`
        Name of flux column from which to compute magnitude.
        Can be parseable by the `~lsst.pipe.tasks.functors.fluxName` function;
        that is, you can pass ``'modelfit_CModel'`` instead of
        ``'modelfit_CModel_instFlux'``, and it will understand.
    """
    _defaultDataset = 'meas'

    def __init__(self, col, **kwargs):
        self.col = fluxName(col)
        # TO DO: DM-21955 Replace hard coded photometic calibration values.
        self.fluxMag0 = 63095734448.0194

        super().__init__(**kwargs)

    @property
    def columns(self):
        return [self.col]

    def _func(self, df):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered')
            warnings.filterwarnings('ignore', r'divide by zero')
            return -2.5*np.log10(df[self.col] / self.fluxMag0)

    @property
    def name(self):
        return f'mag_{self.col}'


class MagErr(Mag):
    """Compute calibrated magnitude uncertainty.

    Parameters
    ----------
    col : `str`
        Name of the flux column.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TO DO: DM-21955 Replace hard coded photometic calibration values.
        self.fluxMag0Err = 0.

    @property
    def columns(self):
        return [self.col, self.col + 'Err']

    def _func(self, df):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered')
            warnings.filterwarnings('ignore', r'divide by zero')
            fluxCol, fluxErrCol = self.columns
            x = df[fluxErrCol] / df[fluxCol]
            y = self.fluxMag0Err / self.fluxMag0
            magErr = (2.5 / np.log(10.)) * np.sqrt(x*x + y*y)
            return magErr

    @property
    def name(self):
        return super().name + '_err'


class MagDiff(Functor):
    """Functor to calculate magnitude difference."""
    _defaultDataset = 'meas'

    def __init__(self, col1, col2, **kwargs):
        self.col1 = fluxName(col1)
        self.col2 = fluxName(col2)
        super().__init__(**kwargs)

    @property
    def columns(self):
        return [self.col1, self.col2]

    def _func(self, df):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered')
            warnings.filterwarnings('ignore', r'divide by zero')
            return -2.5*np.log10(df[self.col1]/df[self.col2])

    @property
    def name(self):
        return f'(mag_{self.col1} - mag_{self.col2})'

    @property
    def shortname(self):
        return f'magDiff_{self.col1}_{self.col2}'


class Color(Functor):
    """Compute the color between two filters.

    Computes color by initializing two different `Mag` functors based on the
    ``col`` and filters provided, and then returning the difference.

    This is enabled by the `_func` method expecting a DataFrame with a
    multilevel column index, with both ``'band'`` and ``'column'``, instead of
    just ``'column'``, which is the `Functor` default.
    This is controlled by the `_dfLevels` attribute.

    Also of note, the default dataset for `Color` is ``forced_src'``, whereas
    for `Mag` it is ``'meas'``.

    Parameters
    ----------
    col : str
        Name of the flux column from which to compute; same as would be passed
        to `~lsst.pipe.tasks.functors.Mag`.

    filt2, filt1 : str
        Filters from which to compute magnitude difference.
        Color computed is ``Mag(filt2) - Mag(filt1)``.
    """
    _defaultDataset = 'forced_src'
    _dfLevels = ('band', 'column')
    _defaultNoDup = True

    def __init__(self, col, filt2, filt1, **kwargs):
        self.col = fluxName(col)
        if filt2 == filt1:
            raise RuntimeError("Cannot compute Color for %s: %s - %s " % (col, filt2, filt1))
        self.filt2 = filt2
        self.filt1 = filt1

        self.mag2 = Mag(col, filt=filt2, **kwargs)
        self.mag1 = Mag(col, filt=filt1, **kwargs)

        super().__init__(**kwargs)

    @property
    def filt(self):
        return None

    @filt.setter
    def filt(self, filt):
        pass

    def _func(self, df):
        mag2 = self.mag2._func(df[self.filt2])
        mag1 = self.mag1._func(df[self.filt1])
        return mag2 - mag1

    @property
    def columns(self):
        return [self.mag1.col, self.mag2.col]

    def multilevelColumns(self, parq, **kwargs):
        return [(self.dataset, self.filt1, self.col), (self.dataset, self.filt2, self.col)]

    @property
    def name(self):
        return f'{self.filt2} - {self.filt1} ({self.col})'

    @property
    def shortname(self):
        return f"{self.col}_{self.filt2.replace('-', '')}m{self.filt1.replace('-', '')}"


class DeconvolvedMoments(Functor):
    """This functor subtracts the trace of the PSF second moments from the
    trace of the second moments of the source.

    If the HsmShapeAlgorithm measurement is valid, then these will be used for
    the sources.
    Otherwise, the SdssShapeAlgorithm measurements will be used.
    """
    name = 'Deconvolved Moments'
    shortname = 'deconvolvedMoments'
    _columns = ("ext_shapeHSM_HsmSourceMoments_xx",
                "ext_shapeHSM_HsmSourceMoments_yy",
                "base_SdssShape_xx", "base_SdssShape_yy",
                "ext_shapeHSM_HsmPsfMoments_xx",
                "ext_shapeHSM_HsmPsfMoments_yy")

    def _func(self, df):
        """Calculate deconvolved moments."""
        if "ext_shapeHSM_HsmSourceMoments_xx" in df.columns:  # _xx added by tdm
            hsm = df["ext_shapeHSM_HsmSourceMoments_xx"] + df["ext_shapeHSM_HsmSourceMoments_yy"]
        else:
            hsm = np.ones(len(df))*np.nan
        sdss = df["base_SdssShape_xx"] + df["base_SdssShape_yy"]
        if "ext_shapeHSM_HsmPsfMoments_xx" in df.columns:
            psf = df["ext_shapeHSM_HsmPsfMoments_xx"] + df["ext_shapeHSM_HsmPsfMoments_yy"]
        else:
            # LSST does not have shape.sdss.psf.
            # We could instead add base_PsfShape to the catalog using
            # exposure.getPsf().computeShape(s.getCentroid()).getIxx().
            raise RuntimeError('No psf shape parameter found in catalog')

        return hsm.where(np.isfinite(hsm), sdss) - psf


class SdssTraceSize(Functor):
    """Functor to calculate the SDSS trace radius size for sources.

    The SDSS trace radius size is a measure of size equal to the square root of
    half of the trace of the second moments tensor measured with the
    SdssShapeAlgorithm plugin.
    This has units of pixels.
    """
    name = "SDSS Trace Size"
    shortname = 'sdssTrace'
    _columns = ("base_SdssShape_xx", "base_SdssShape_yy")

    def _func(self, df):
        srcSize = np.sqrt(0.5*(df["base_SdssShape_xx"] + df["base_SdssShape_yy"]))
        return srcSize


class PsfSdssTraceSizeDiff(Functor):
    """Functor to calculate the SDSS trace radius size difference (%) between
    the object and the PSF model.

    See Also
    --------
    SdssTraceSize
    """
    name = "PSF - SDSS Trace Size"
    shortname = 'psf_sdssTrace'
    _columns = ("base_SdssShape_xx", "base_SdssShape_yy",
                "base_SdssShape_psf_xx", "base_SdssShape_psf_yy")

    def _func(self, df):
        srcSize = np.sqrt(0.5*(df["base_SdssShape_xx"] + df["base_SdssShape_yy"]))
        psfSize = np.sqrt(0.5*(df["base_SdssShape_psf_xx"] + df["base_SdssShape_psf_yy"]))
        sizeDiff = 100*(srcSize - psfSize)/(0.5*(srcSize + psfSize))
        return sizeDiff


class HsmTraceSize(Functor):
    """Functor to calculate the HSM trace radius size for sources.

    The HSM trace radius size is a measure of size equal to the square root of
    half of the trace of the second moments tensor measured with the
    HsmShapeAlgorithm plugin.
    This has units of pixels.
    """
    name = 'HSM Trace Size'
    shortname = 'hsmTrace'
    _columns = ("ext_shapeHSM_HsmSourceMoments_xx",
                "ext_shapeHSM_HsmSourceMoments_yy")

    def _func(self, df):
        srcSize = np.sqrt(0.5*(df["ext_shapeHSM_HsmSourceMoments_xx"]
                               + df["ext_shapeHSM_HsmSourceMoments_yy"]))
        return srcSize


class PsfHsmTraceSizeDiff(Functor):
    """Functor to calculate the HSM trace radius size difference (%) between
    the object and the PSF model.

    See Also
    --------
    HsmTraceSize
    """
    name = 'PSF - HSM Trace Size'
    shortname = 'psf_HsmTrace'
    _columns = ("ext_shapeHSM_HsmSourceMoments_xx",
                "ext_shapeHSM_HsmSourceMoments_yy",
                "ext_shapeHSM_HsmPsfMoments_xx",
                "ext_shapeHSM_HsmPsfMoments_yy")

    def _func(self, df):
        srcSize = np.sqrt(0.5*(df["ext_shapeHSM_HsmSourceMoments_xx"]
                               + df["ext_shapeHSM_HsmSourceMoments_yy"]))
        psfSize = np.sqrt(0.5*(df["ext_shapeHSM_HsmPsfMoments_xx"]
                               + df["ext_shapeHSM_HsmPsfMoments_yy"]))
        sizeDiff = 100*(srcSize - psfSize)/(0.5*(srcSize + psfSize))
        return sizeDiff


class HsmFwhm(Functor):
    """Functor to calculate the PSF FWHM with second moments measured from the
    HsmShapeAlgorithm plugin.

    This is in units of arcseconds, and assumes the hsc_rings_v1 skymap pixel
    scale of 0.168 arcseconds/pixel.

    Notes
    -----
    This conversion assumes the PSF is Gaussian, which is not always the case.
    """
    name = 'HSM Psf FWHM'
    _columns = ('ext_shapeHSM_HsmPsfMoments_xx', 'ext_shapeHSM_HsmPsfMoments_yy')
    # TODO: DM-21403 pixel scale should be computed from the CD matrix or transform matrix
    pixelScale = 0.168
    SIGMA2FWHM = 2*np.sqrt(2*np.log(2))

    def _func(self, df):
        return (self.pixelScale*self.SIGMA2FWHM*np.sqrt(
            0.5*(df['ext_shapeHSM_HsmPsfMoments_xx']
                 + df['ext_shapeHSM_HsmPsfMoments_yy']))).astype(np.float32)


class E1(Functor):
    r"""Calculate :math:`e_1` ellipticity component for sources, defined as:

    .. math::
        e_1 &= (I_{xx}-I_{yy})/(I_{xx}+I_{yy})

    See Also
    --------
    E2
    """
    name = "Distortion Ellipticity (e1)"
    shortname = "Distortion"

    def __init__(self, colXX, colXY, colYY, **kwargs):
        self.colXX = colXX
        self.colXY = colXY
        self.colYY = colYY
        self._columns = [self.colXX, self.colXY, self.colYY]
        super().__init__(**kwargs)

    @property
    def columns(self):
        return [self.colXX, self.colXY, self.colYY]

    def _func(self, df):
        return ((df[self.colXX] - df[self.colYY]) / (
            df[self.colXX] + df[self.colYY])).astype(np.float32)


class E2(Functor):
    r"""Calculate :math:`e_2` ellipticity component for sources, defined as:

    .. math::
        e_2 &= 2I_{xy}/(I_{xx}+I_{yy})

    See Also
    --------
    E1
    """
    name = "Ellipticity e2"

    def __init__(self, colXX, colXY, colYY, **kwargs):
        self.colXX = colXX
        self.colXY = colXY
        self.colYY = colYY
        super().__init__(**kwargs)

    @property
    def columns(self):
        return [self.colXX, self.colXY, self.colYY]

    def _func(self, df):
        return (2*df[self.colXY] / (df[self.colXX] + df[self.colYY])).astype(np.float32)


class RadiusFromQuadrupole(Functor):
    """Calculate the radius from the quadrupole moments.

    This returns the fourth root of the determinant of the second moments
    tensor, which has units of pixels.

    See Also
    --------
    SdssTraceSize
    HsmTraceSize
    """

    def __init__(self, colXX, colXY, colYY, **kwargs):
        self.colXX = colXX
        self.colXY = colXY
        self.colYY = colYY
        super().__init__(**kwargs)

    @property
    def columns(self):
        return [self.colXX, self.colXY, self.colYY]

    def _func(self, df):
        return ((df[self.colXX]*df[self.colYY] - df[self.colXY]**2)**0.25).astype(np.float32)


class LocalWcs(Functor):
    """Computations using the stored localWcs."""
    name = "LocalWcsOperations"

    def __init__(self,
                 colCD_1_1,
                 colCD_1_2,
                 colCD_2_1,
                 colCD_2_2,
                 **kwargs):
        self.colCD_1_1 = colCD_1_1
        self.colCD_1_2 = colCD_1_2
        self.colCD_2_1 = colCD_2_1
        self.colCD_2_2 = colCD_2_2
        super().__init__(**kwargs)

    def computeDeltaRaDec(self, x, y, cd11, cd12, cd21, cd22):
        """Compute the dRA, dDec from dx, dy.

        Parameters
        ----------
        x : `~pandas.Series`
            X pixel coordinate.
        y : `~pandas.Series`
            Y pixel coordinate.
        cd11 : `~pandas.Series`
            [1, 1] element of the local Wcs affine transform.
        cd12 : `~pandas.Series`
            [1, 2] element of the local Wcs affine transform.
        cd21 : `~pandas.Series`
            [2, 1] element of the local Wcs affine transform.
        cd22 : `~pandas.Series`
            [2, 2] element of the local Wcs affine transform.

        Returns
        -------
        raDecTuple : tuple
            RA and Dec conversion of x and y given the local Wcs.
            Returned units are in radians.

        Notes
        -----
        If x and y are with respect to the CRVAL1, CRVAL2
        then this will return the RA, Dec for that WCS.
        """
        return (x * cd11 + y * cd12, x * cd21 + y * cd22)

    def computeSkySeparation(self, ra1, dec1, ra2, dec2):
        """Compute the local pixel scale conversion.

        Parameters
        ----------
        ra1 : `~pandas.Series`
            Ra of the first coordinate in radians.
        dec1 : `~pandas.Series`
            Dec of the first coordinate in radians.
        ra2 : `~pandas.Series`
            Ra of the second coordinate in radians.
        dec2 : `~pandas.Series`
            Dec of the second coordinate in radians.

        Returns
        -------
        dist : `~pandas.Series`
            Distance on the sphere in radians.
        """
        deltaDec = dec2 - dec1
        deltaRa = ra2 - ra1
        return 2 * np.arcsin(
            np.sqrt(
                np.sin(deltaDec / 2) ** 2
                + np.cos(dec2) * np.cos(dec1) * np.sin(deltaRa / 2) ** 2))

    def getSkySeparationFromPixel(self, x1, y1, x2, y2, cd11, cd12, cd21, cd22):
        """Compute the distance on the sphere from x2, y1 to x1, y1.

        Parameters
        ----------
        x1 : `~pandas.Series`
            X pixel coordinate.
        y1 : `~pandas.Series`
            Y pixel coordinate.
        x2 : `~pandas.Series`
            X pixel coordinate.
        y2 : `~pandas.Series`
            Y pixel coordinate.
        cd11 : `~pandas.Series`
            [1, 1] element of the local Wcs affine transform.
        cd12 : `~pandas.Series`
            [1, 2] element of the local Wcs affine transform.
        cd21 : `~pandas.Series`
            [2, 1] element of the local Wcs affine transform.
        cd22 : `~pandas.Series`
            [2, 2] element of the local Wcs affine transform.

        Returns
        -------
        Distance : `~pandas.Series`
            Arcseconds per pixel at the location of the local WC.
        """
        ra1, dec1 = self.computeDeltaRaDec(x1, y1, cd11, cd12, cd21, cd22)
        ra2, dec2 = self.computeDeltaRaDec(x2, y2, cd11, cd12, cd21, cd22)
        # Great circle distance for small separations.
        return self.computeSkySeparation(ra1, dec1, ra2, dec2)

    def computePositionAngle(self, ra1, dec1, ra2, dec2):
        """Compute position angle (E of N) from (ra1, dec1) to (ra2, dec2).

        Parameters
        ----------
        ra1 : iterable [`float`]
            RA of the first coordinate [radian].
        dec1 : iterable [`float`]
            Dec of the first coordinate [radian].
        ra2 : iterable [`float`]
            RA of the second coordinate [radian].
        dec2 : iterable [`float`]
            Dec of the second coordinate [radian].

        Returns
        -------
        Position Angle: `~pandas.Series`
            radians E of N

        Notes
        -----
        (ra1, dec1) -> (ra2, dec2) is interpreted as the shorter way around the sphere

        For a separation of 0.0001 rad, the position angle is good to 0.0009 rad
        all over the sphere.
        """
        # lsst.geom.SpherePoint has "bearingTo", which returns angle N of E
        # We instead want the astronomy convention of "Position Angle", which is angle E of N
        position_angle = np.zeros(len(ra1))
        for i, (r1, d1, r2, d2) in enumerate(zip(ra1, dec1, ra2, dec2)):
            point1 = geom.SpherePoint(r1, d1, geom.radians)
            point2 = geom.SpherePoint(r2, d2, geom.radians)
            bearing = point1.bearingTo(point2)
            pa_ref_angle = geom.Angle(np.pi/2, geom.radians)  # in bearing system
            pa = pa_ref_angle - bearing
            # Wrap around to get Delta_RA from -pi to +pi
            pa = pa.wrapCtr()
            position_angle[i] = pa.asRadians()

        return pd.Series(position_angle)

    def getPositionAngleFromDetectorAngle(self, theta, cd11, cd12, cd21, cd22):
        """Compute position angle (E of N) from detector angle (+y of +x).

        Parameters
        ----------
        theta : `float`
            detector angle [radian]
        cd11 : `float`
            [1, 1] element of the local Wcs affine transform.
        cd12 : `float`
            [1, 2] element of the local Wcs affine transform.
        cd21 : `float`
            [2, 1] element of the local Wcs affine transform.
        cd22 : `float`
            [2, 2] element of the local Wcs affine transform.

        Returns
        -------
        Position Angle: `~pandas.Series`
            Degrees E of N.
        """
        # Create a unit vector in (x, y) along da
        dx = np.cos(theta)
        dy = np.sin(theta)
        ra1, dec1 = self.computeDeltaRaDec(0, 0, cd11, cd12, cd21, cd22)
        ra2, dec2 = self.computeDeltaRaDec(dx, dy, cd11, cd12, cd21, cd22)
        # Position angle of vector from (RA1, Dec1) to (RA2, Dec2)
        return np.rad2deg(self.computePositionAngle(ra1, dec1, ra2, dec2))

    def getPositionAngleErrFromDetectorAngleErr(self, theta, theta_err, cd11, cd12, cd21, cd22):
        """Compute position angle error (E of N) from detector angle error.

        Parameters
        ----------
        theta : `float`
            detector angle [radian]
        theta_err : `float`
            detector angle err [radian]
        cd11 : `float`
            [1, 1] element of the local Wcs affine transform.
        cd12 : `float`
            [1, 2] element of the local Wcs affine transform.
        cd21 : `float`
            [2, 1] element of the local Wcs affine transform.
        cd22 : `float`
            [2, 2] element of the local Wcs affine transform.
        Returns
        -------
        Position Angle Error: `~pandas.Series`
            Position angle error in degrees
        """
        # Need to compute abs(dPA/dtheta)*theta_Err to get propogated errors

        # Get unit direction
        dx = np.cos(theta)
        dy = np.sin(theta)

        # Transform it using WCS?
        u = dx * cd11 + dy * cd12
        v = dx * cd21 + dy * cd22
        # Now we are computing the tangent
        ratio = u / v

        # Get derivative of theta
        du_dtheta = -np.sin(theta) * cd11 + np.cos(theta) * cd12
        dv_dtheta = -np.sin(theta) * cd21 + np.cos(theta) * cd22

        # Get derivative of tangent
        d_ratio_dtheta = (v * du_dtheta - u * dv_dtheta) / v ** 2
        dPA_dtheta = (1 / (1 + ratio ** 2)) * d_ratio_dtheta

        pa_err = np.rad2deg(np.abs(dPA_dtheta) * theta_err)

        logging.info("PA Error: %s" % pa_err)
        logging.info("theta_err: %s" % theta_err)

        return pa_err


class ComputePixelScale(LocalWcs):
    """Compute the local pixel scale from the stored CDMatrix.
    """
    name = "PixelScale"

    @property
    def columns(self):
        return [self.colCD_1_1,
                self.colCD_1_2,
                self.colCD_2_1,
                self.colCD_2_2]

    def pixelScaleArcseconds(self, cd11, cd12, cd21, cd22):
        """Compute the local pixel to scale conversion in arcseconds.

        Parameters
        ----------
        cd11 : `~pandas.Series`
            [1, 1] element of the local Wcs affine transform in radians.
        cd11 : `~pandas.Series`
            [1, 1] element of the local Wcs affine transform in radians.
        cd12 : `~pandas.Series`
            [1, 2] element of the local Wcs affine transform in radians.
        cd21 : `~pandas.Series`
            [2, 1] element of the local Wcs affine transform in radians.
        cd22 : `~pandas.Series`
            [2, 2] element of the local Wcs affine transform in radians.

        Returns
        -------
        pixScale : `~pandas.Series`
            Arcseconds per pixel at the location of the local WC.
        """
        return 3600 * np.degrees(np.sqrt(np.fabs(cd11 * cd22 - cd12 * cd21)))

    def _func(self, df):
        return self.pixelScaleArcseconds(df[self.colCD_1_1],
                                         df[self.colCD_1_2],
                                         df[self.colCD_2_1],
                                         df[self.colCD_2_2])


class ConvertPixelToArcseconds(ComputePixelScale):
    """Convert a value in units of pixels to units of arcseconds."""

    def __init__(self,
                 col,
                 colCD_1_1,
                 colCD_1_2,
                 colCD_2_1,
                 colCD_2_2,
                 **kwargs):
        self.col = col
        super().__init__(colCD_1_1,
                         colCD_1_2,
                         colCD_2_1,
                         colCD_2_2,
                         **kwargs)

    @property
    def name(self):
        return f"{self.col}_asArcseconds"

    @property
    def columns(self):
        return [self.col,
                self.colCD_1_1,
                self.colCD_1_2,
                self.colCD_2_1,
                self.colCD_2_2]

    def _func(self, df):
        return df[self.col] * self.pixelScaleArcseconds(df[self.colCD_1_1],
                                                        df[self.colCD_1_2],
                                                        df[self.colCD_2_1],
                                                        df[self.colCD_2_2])


class ConvertPixelSqToArcsecondsSq(ComputePixelScale):
    """Convert a value in units of pixels squared to units of arcseconds
    squared.
    """

    def __init__(self,
                 col,
                 colCD_1_1,
                 colCD_1_2,
                 colCD_2_1,
                 colCD_2_2,
                 **kwargs):
        self.col = col
        super().__init__(colCD_1_1,
                         colCD_1_2,
                         colCD_2_1,
                         colCD_2_2,
                         **kwargs)

    @property
    def name(self):
        return f"{self.col}_asArcsecondsSq"

    @property
    def columns(self):
        return [self.col,
                self.colCD_1_1,
                self.colCD_1_2,
                self.colCD_2_1,
                self.colCD_2_2]

    def _func(self, df):
        pixScale = self.pixelScaleArcseconds(df[self.colCD_1_1],
                                             df[self.colCD_1_2],
                                             df[self.colCD_2_1],
                                             df[self.colCD_2_2])
        return df[self.col] * pixScale * pixScale


class ConvertDetectorAngleToPositionAngle(LocalWcs):
    """Compute a position angle from a detector angle and the stored CDMatrix.

    Returns
    -------
    position angle : degrees
    """

    name = "PositionAngle"

    def __init__(
        self,
        theta_col,
        colCD_1_1,
        colCD_1_2,
        colCD_2_1,
        colCD_2_2,
        **kwargs
    ):
        self.theta_col = theta_col
        super().__init__(colCD_1_1, colCD_1_2, colCD_2_1, colCD_2_2, **kwargs)

    @property
    def columns(self):
        return [
            self.theta_col,
            self.colCD_1_1,
            self.colCD_1_2,
            self.colCD_2_1,
            self.colCD_2_2
        ]

    def _func(self, df):
        return self.getPositionAngleFromDetectorAngle(
            df[self.theta_col],
            df[self.colCD_1_1],
            df[self.colCD_1_2],
            df[self.colCD_2_1],
            df[self.colCD_2_2]
        )


class ConvertDetectorAngleErrToPositionAngleErr(LocalWcs):
    """Compute a position angle error from a detector angle error and the
    stored CDMatrix.

    Returns
    -------
    position angle error : degrees
    """

    name = "PositionAngleErr"

    def __init__(
        self,
        theta_col,
        theta_err_col,
        colCD_1_1,
        colCD_1_2,
        colCD_2_1,
        colCD_2_2,
        **kwargs
    ):
        self.theta_col = theta_col
        self.theta_err_col = theta_err_col
        super().__init__(colCD_1_1, colCD_1_2, colCD_2_1, colCD_2_2, **kwargs)

    @property
    def columns(self):
        return [
            self.theta_col,
            self.theta_err_col,
            self.colCD_1_1,
            self.colCD_1_2,
            self.colCD_2_1,
            self.colCD_2_2
        ]

    def _func(self, df):
        return self.getPositionAngleErrFromDetectorAngleErr(
            df[self.theta_col],
            df[self.theta_err_col],
            df[self.colCD_1_1],
            df[self.colCD_1_2],
            df[self.colCD_2_1],
            df[self.colCD_2_2]
        )


class ReferenceBand(Functor):
    """Return the band used to seed multiband forced photometry.

    This functor is to be used on Object tables.
    It converts the boolean merge_measurements_{band} columns into a single
    string representing the first band for which merge_measurements_{band}
    is True.

    Assumes the default priority order of i, r, z, y, g, u.
    """
    name = 'Reference Band'
    shortname = 'refBand'

    band_order = ("i", "r", "z", "y", "g", "u")

    @property
    def columns(self):
        # Build the actual input column list, not hardcoded ugrizy
        bands = [band for band in self.band_order if band in self.bands]
        # In the unlikely scenario that users attempt to add non-ugrizy bands
        bands += [band for band in self.bands if band not in self.band_order]
        return [f"merge_measurement_{band}" for band in bands]

    def _func(self, df: pd.DataFrame) -> pd.Series:
        def getFilterAliasName(row):
            # Get column name with the max value (True > False).
            colName = row.idxmax()
            return colName.replace('merge_measurement_', '')

        # Skip columns that are unavailable, because this functor requests the
        # superset of bands that could be included in the object table.
        columns = [col for col in self.columns if col in df.columns]
        # Makes a Series of dtype object if df is empty.
        return df[columns].apply(getFilterAliasName, axis=1,
                                 result_type='reduce').astype('object')

    def __init__(self, bands: tuple[str] | list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.bands = self.band_order if bands is None else tuple(bands)


class Photometry(Functor):
    """Base class for Object table calibrated fluxes and magnitudes."""
    # AB to NanoJansky (3631 Jansky).
    AB_FLUX_SCALE = (0 * u.ABmag).to_value(u.nJy)
    LOG_AB_FLUX_SCALE = 12.56
    FIVE_OVER_2LOG10 = 1.085736204758129569
    # TO DO: DM-21955 Replace hard coded photometic calibration values.
    COADD_ZP = 27

    def __init__(self, colFlux, colFluxErr=None, **kwargs):
        self.vhypot = np.vectorize(self.hypot)
        self.col = colFlux
        self.colFluxErr = colFluxErr

        self.fluxMag0 = 1./np.power(10, -0.4*self.COADD_ZP)
        self.fluxMag0Err = 0.

        super().__init__(**kwargs)

    @property
    def columns(self):
        return [self.col]

    @property
    def name(self):
        return f'mag_{self.col}'

    @classmethod
    def hypot(cls, a, b):
        """Compute sqrt(a^2 + b^2) without under/overflow."""
        if np.abs(a) < np.abs(b):
            a, b = b, a
        if a == 0.:
            return 0.
        q = b/a
        return np.abs(a) * np.sqrt(1. + q*q)

    def dn2flux(self, dn, fluxMag0):
        """Convert instrumental flux to nanojanskys."""
        return (self.AB_FLUX_SCALE * dn / fluxMag0).astype(np.float32)

    def dn2mag(self, dn, fluxMag0):
        """Convert instrumental flux to AB magnitude."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered')
            warnings.filterwarnings('ignore', r'divide by zero')
            return (-2.5 * np.log10(dn/fluxMag0)).astype(np.float32)

    def dn2fluxErr(self, dn, dnErr, fluxMag0, fluxMag0Err):
        """Convert instrumental flux error to nanojanskys."""
        retVal = self.vhypot(dn * fluxMag0Err, dnErr * fluxMag0)
        retVal *= self.AB_FLUX_SCALE / fluxMag0 / fluxMag0
        return retVal.astype(np.float32)

    def dn2MagErr(self, dn, dnErr, fluxMag0, fluxMag0Err):
        """Convert instrumental flux error to AB magnitude error."""
        retVal = self.dn2fluxErr(dn, dnErr, fluxMag0, fluxMag0Err) / self.dn2flux(dn, fluxMag0)
        return (self.FIVE_OVER_2LOG10 * retVal).astype(np.float32)


class NanoJansky(Photometry):
    """Convert instrumental flux to nanojanskys."""
    def _func(self, df):
        return self.dn2flux(df[self.col], self.fluxMag0)


class NanoJanskyErr(Photometry):
    """Convert instrumental flux error to nanojanskys."""
    @property
    def columns(self):
        return [self.col, self.colFluxErr]

    def _func(self, df):
        retArr = self.dn2fluxErr(df[self.col], df[self.colFluxErr], self.fluxMag0, self.fluxMag0Err)
        return pd.Series(retArr, index=df.index)


class LocalPhotometry(Functor):
    """Base class for calibrating the specified instrument flux column using
    the local photometric calibration.

    Parameters
    ----------
    instFluxCol : `str`
        Name of the instrument flux column.
    instFluxErrCol : `str`
        Name of the assocated error columns for ``instFluxCol``.
    photoCalibCol : `str`
        Name of local calibration column.
    photoCalibErrCol : `str`, optional
        Error associated with ``photoCalibCol``.  Ignored and deprecated; will
        be removed after v29.

    See Also
    --------
    LocalNanojansky
    LocalNanojanskyErr
    """
    logNJanskyToAB = (1 * u.nJy).to_value(u.ABmag)

    def __init__(self,
                 instFluxCol,
                 instFluxErrCol,
                 photoCalibCol,
                 photoCalibErrCol=None,
                 **kwargs):
        self.instFluxCol = instFluxCol
        self.instFluxErrCol = instFluxErrCol
        self.photoCalibCol = photoCalibCol
        # TODO[DM-49400]: remove this check and the argument it corresponds to.
        if photoCalibErrCol is not None:
            warnings.warn("The photoCalibErrCol argument is deprecated and will be removed after v29.",
                          category=FutureWarning)
        super().__init__(**kwargs)

    def instFluxToNanojansky(self, instFlux, localCalib):
        """Convert instrument flux to nanojanskys.

        Parameters
        ----------
        instFlux : `~numpy.ndarray` or `~pandas.Series`
            Array of instrument flux measurements.
        localCalib : `~numpy.ndarray` or `~pandas.Series`
            Array of local photometric calibration estimates.

        Returns
        -------
        calibFlux : `~numpy.ndarray` or `~pandas.Series`
            Array of calibrated flux measurements.
        """
        return instFlux * localCalib

    def instFluxErrToNanojanskyErr(self, instFlux, instFluxErr, localCalib, localCalibErr=None):
        """Convert instrument flux to nanojanskys.

        Parameters
        ----------
        instFlux : `~numpy.ndarray` or `~pandas.Series`
            Array of instrument flux measurements.  Ignored (accepted for
            backwards compatibility and consistency with magnitude-error
            calculation methods).
        instFluxErr : `~numpy.ndarray` or `~pandas.Series`
            Errors on associated ``instFlux`` values.
        localCalib : `~numpy.ndarray` or `~pandas.Series`
            Array of local photometric calibration estimates.
        localCalibErr : `~numpy.ndarray` or `~pandas.Series`, optional
            Errors on associated ``localCalib`` values.  Ignored and deprecated;
            will be removed after v29.

        Returns
        -------
        calibFluxErr : `~numpy.ndarray` or `~pandas.Series`
            Errors on calibrated flux measurements.
        """
        # TODO[DM-49400]: remove this check and the argument it corresponds to.
        if localCalibErr is not None:
            warnings.warn("The localCalibErr argument is deprecated and will be removed after v29.",
                          category=FutureWarning)
        return instFluxErr * localCalib

    def instFluxToMagnitude(self, instFlux, localCalib):
        """Convert instrument flux to nanojanskys.

        Parameters
        ----------
        instFlux : `~numpy.ndarray` or `~pandas.Series`
            Array of instrument flux measurements.
        localCalib : `~numpy.ndarray` or `~pandas.Series`
            Array of local photometric calibration estimates.

        Returns
        -------
        calibMag : `~numpy.ndarray` or `~pandas.Series`
            Array of calibrated AB magnitudes.
        """
        return -2.5 * np.log10(self.instFluxToNanojansky(instFlux, localCalib)) + self.logNJanskyToAB

    def instFluxErrToMagnitudeErr(self, instFlux, instFluxErr, localCalib, localCalibErr=None):
        """Convert instrument flux err to nanojanskys.

        Parameters
        ----------
        instFlux : `~numpy.ndarray` or `~pandas.Series`
            Array of instrument flux measurements.
        instFluxErr : `~numpy.ndarray` or `~pandas.Series`
            Errors on associated ``instFlux`` values.
        localCalib : `~numpy.ndarray` or `~pandas.Series`
            Array of local photometric calibration estimates.
        localCalibErr : `~numpy.ndarray` or `~pandas.Series`, optional
            Errors on associated ``localCalib`` values.  Ignored and deprecated;
            will be removed after v29.

        Returns
        -------
        calibMagErr: `~numpy.ndarray` or `~pandas.Series`
            Error on calibrated AB magnitudes.
        """
        # TODO[DM-49400]: remove this check and the argument it corresponds to.
        if localCalibErr is not None:
            warnings.warn("The localCalibErr argument is deprecated and will be removed after v29.",
                          category=FutureWarning)
        err = self.instFluxErrToNanojanskyErr(instFlux, instFluxErr, localCalib)
        return 2.5 / np.log(10) * err / self.instFluxToNanojansky(instFlux, instFluxErr)


class LocalNanojansky(LocalPhotometry):
    """Compute calibrated fluxes using the local calibration value.

    This returns units of nanojanskys.
    """

    @property
    def columns(self):
        return [self.instFluxCol, self.photoCalibCol]

    @property
    def name(self):
        return f'flux_{self.instFluxCol}'

    def _func(self, df):
        return self.instFluxToNanojansky(df[self.instFluxCol],
                                         df[self.photoCalibCol]).astype(np.float32)


class LocalNanojanskyErr(LocalPhotometry):
    """Compute calibrated flux errors using the local calibration value.

    This returns units of nanojanskys.
    """

    @property
    def columns(self):
        return [self.instFluxCol, self.instFluxErrCol, self.photoCalibCol]

    @property
    def name(self):
        return f'fluxErr_{self.instFluxCol}'

    def _func(self, df):
        return self.instFluxErrToNanojanskyErr(df[self.instFluxCol], df[self.instFluxErrCol],
                                               df[self.photoCalibCol]).astype(np.float32)


class LocalDipoleMeanFlux(LocalPhotometry):
    """Compute absolute mean of dipole fluxes.

    See Also
    --------
    LocalNanojansky
    LocalNanojanskyErr
    LocalDipoleMeanFluxErr
    LocalDipoleDiffFlux
    LocalDipoleDiffFluxErr
    """
    def __init__(self,
                 instFluxPosCol,
                 instFluxNegCol,
                 instFluxPosErrCol,
                 instFluxNegErrCol,
                 photoCalibCol,
                 # TODO[DM-49400]: remove this option; it's already deprecated (in super).
                 photoCalibErrCol=None,
                 **kwargs):
        self.instFluxNegCol = instFluxNegCol
        self.instFluxPosCol = instFluxPosCol
        self.instFluxNegErrCol = instFluxNegErrCol
        self.instFluxPosErrCol = instFluxPosErrCol
        self.photoCalibCol = photoCalibCol
        super().__init__(instFluxNegCol,
                         instFluxNegErrCol,
                         photoCalibCol,
                         photoCalibErrCol,
                         **kwargs)

    @property
    def columns(self):
        return [self.instFluxPosCol,
                self.instFluxNegCol,
                self.photoCalibCol]

    @property
    def name(self):
        return f'dipMeanFlux_{self.instFluxPosCol}_{self.instFluxNegCol}'

    def _func(self, df):
        return 0.5*(np.fabs(self.instFluxToNanojansky(df[self.instFluxNegCol], df[self.photoCalibCol]))
                    + np.fabs(self.instFluxToNanojansky(df[self.instFluxPosCol], df[self.photoCalibCol])))


class LocalDipoleMeanFluxErr(LocalDipoleMeanFlux):
    """Compute the error on the absolute mean of dipole fluxes.

    See Also
    --------
    LocalNanojansky
    LocalNanojanskyErr
    LocalDipoleMeanFlux
    LocalDipoleDiffFlux
    LocalDipoleDiffFluxErr
    """

    @property
    def columns(self):
        return [self.instFluxPosCol,
                self.instFluxNegCol,
                self.instFluxPosErrCol,
                self.instFluxNegErrCol,
                self.photoCalibCol]

    @property
    def name(self):
        return f'dipMeanFluxErr_{self.instFluxPosCol}_{self.instFluxNegCol}'

    def _func(self, df):
        return 0.5*np.hypot(df[self.instFluxNegErrCol], df[self.instFluxPosErrCol]) * df[self.photoCalibCol]


class LocalDipoleDiffFlux(LocalDipoleMeanFlux):
    """Compute the absolute difference of dipole fluxes.

    Calculated value is (abs(pos) - abs(neg)).

    See Also
    --------
    LocalNanojansky
    LocalNanojanskyErr
    LocalDipoleMeanFlux
    LocalDipoleMeanFluxErr
    LocalDipoleDiffFluxErr
    """

    @property
    def columns(self):
        return [self.instFluxPosCol,
                self.instFluxNegCol,
                self.photoCalibCol]

    @property
    def name(self):
        return f'dipDiffFlux_{self.instFluxPosCol}_{self.instFluxNegCol}'

    def _func(self, df):
        return (np.fabs(self.instFluxToNanojansky(df[self.instFluxPosCol], df[self.photoCalibCol]))
                - np.fabs(self.instFluxToNanojansky(df[self.instFluxNegCol], df[self.photoCalibCol])))


class LocalDipoleDiffFluxErr(LocalDipoleMeanFlux):
    """Compute the error on the absolute difference of dipole fluxes.

    See Also
    --------
    LocalNanojansky
    LocalNanojanskyErr
    LocalDipoleMeanFlux
    LocalDipoleMeanFluxErr
    LocalDipoleDiffFlux
    """

    @property
    def columns(self):
        return [self.instFluxPosCol,
                self.instFluxNegCol,
                self.instFluxPosErrCol,
                self.instFluxNegErrCol,
                self.photoCalibCol]

    @property
    def name(self):
        return f'dipDiffFluxErr_{self.instFluxPosCol}_{self.instFluxNegCol}'

    def _func(self, df):
        return np.hypot(df[self.instFluxPosErrCol], df[self.instFluxNegErrCol]) * df[self.photoCalibCol]


class Ebv(Functor):
    """Compute E(B-V) from dustmaps.sfd."""
    _defaultDataset = 'ref'
    name = "E(B-V)"
    shortname = "ebv"

    def __init__(self, **kwargs):
        # Import is only needed for Ebv.
        # Suppress unnecessary .dustmapsrc log message on import.
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull):
                from dustmaps.sfd import SFDQuery
        self._columns = ['coord_ra', 'coord_dec']
        self.sfd = SFDQuery()
        super().__init__(**kwargs)

    def _func(self, df):
        coords = SkyCoord(df['coord_ra'].values * u.rad, df['coord_dec'].values * u.rad)
        ebv = self.sfd(coords)
        return pd.Series(ebv, index=df.index).astype('float32')


class MomentsBase(Functor):
    """Base class for functors that use shape moments and localWCS

    Attributes
    ----------
    is_covariance : bool
        Whether the shape columns are terms of a covariance matrix. If False,
        they will be assumed to be terms of a correlation matrix instead.
    """

    is_covariance: bool = True

    def __init__(self,
                 shape_1_1,
                 shape_2_2,
                 shape_1_2,
                 colCD_1_1,
                 colCD_1_2,
                 colCD_2_1,
                 colCD_2_2,
                 **kwargs):
        self.shape_1_1 = shape_1_1
        self.shape_2_2 = shape_2_2
        self.shape_1_2 = shape_1_2
        self.colCD_1_1 = colCD_1_1
        self.colCD_1_2 = colCD_1_2
        self.colCD_2_1 = colCD_2_1
        self.colCD_2_2 = colCD_2_2
        super().__init__(**kwargs)

    @property
    def columns(self):
        return [
            self.shape_1_1,
            self.shape_2_2,
            self.shape_1_2,
        ] + self.columns_ref

    @property
    def columns_ref(self):
        """Return columns that are needed from the ref table."""
        return [
            self.colCD_1_1,
            self.colCD_1_2,
            self.colCD_2_1,
            self.colCD_2_2]

    def compute_ellipse_terms(self, df, sky: bool = True):
        r"""Return terms commonly used for ellipse parameterization conversions.

        Parameters
        ----------
        df
            The data frame.
        sky
            Whether to compute the terms in sky coordinates.
            If False, XX, YY and XY moments are used instead of
            UU, VV and UV.

        Returns
        -------
        xx_p_yy
            The sum of the diagonal terms of the covariance.
        xx_m_yy
            The difference of the diagonal terms of the covariance.
        t2
            A term similar to the discriminant of the quadratic formula.
        """
        xx = self.sky_uu(df) if sky else self.get_xx(df)
        yy = self.sky_vv(df) if sky else self.get_yy(df)
        xx_m_yy = xx - yy
        t2 = xx_m_yy**2 + 4.0*(self.sky_uv(df) if sky else self.get_xy(df))**2
        # TODO: Check alternative form that may be more stable for computing
        # the minor axis size (see gauss2d/src/ellipse.cc)
        # t2 = xx**2 + yy**2 - 2*(xx*yy - 2*xy**2)
        return xx + yy, xx_m_yy, t2

    def get_xx(self, df):
        xx = df[self.shape_1_1]
        return xx if self.is_covariance else xx**2

    def get_yy(self, df):
        yy = df[self.shape_2_2]
        return yy if self.is_covariance else yy**2

    def get_xy(self, df):
        xy = df[self.shape_1_2]
        return xy if self.is_covariance else xy*df[self.shape_1_1]*df[self.shape_2_2]

    # Each of sky_uu, sky_vv, sky_uv evaluates one element of
    # CD_matrix * moments_matrix * CD_matrix.T
    def sky_uu(self, df):
        """Return the component of the moments tensor aligned with the RA axis, in radians."""
        i_xx = self.get_xx(df)
        i_yy = self.get_yy(df)
        i_xy = self.get_xy(df)
        CD_1_1 = df[self.colCD_1_1]
        CD_1_2 = df[self.colCD_1_2]
        CD_2_1 = df[self.colCD_2_1]
        return (CD_1_1*(i_xx*CD_1_1 + i_xy*CD_2_1)
                + CD_1_2*(i_xy*CD_1_1 + i_yy*CD_2_1))

    def sky_vv(self, df):
        """Return the component of the moments tensor aligned with the dec axis, in radians."""
        i_xx = self.get_xx(df)
        i_yy = self.get_yy(df)
        i_xy = self.get_xy(df)
        CD_1_2 = df[self.colCD_1_2]
        CD_2_1 = df[self.colCD_2_1]
        CD_2_2 = df[self.colCD_2_2]
        return (CD_2_1*(i_xx*CD_1_2 + i_xy*CD_2_2)
                + CD_2_2*(i_xy*CD_1_2 + i_yy*CD_2_2))

    def sky_uv(self, df):
        """Return the covariance of the moments tensor in ra, dec coordinates, in radians."""
        i_xx = self.get_xx(df)
        i_yy = self.get_yy(df)
        i_xy = self.get_xy(df)
        CD_1_1 = df[self.colCD_1_1]
        CD_1_2 = df[self.colCD_1_2]
        CD_2_1 = df[self.colCD_2_1]
        CD_2_2 = df[self.colCD_2_2]
        return ((CD_1_1 * i_xx + CD_1_2 * i_xy) * CD_2_1
                + (CD_1_1 * i_xy + CD_1_2 * i_yy) * CD_2_2)

    def get_g1(self, df):
        """
        Calculate shear-type ellipticity parameter G1.
        """
        # TODO: Replace this with functionality from afwGeom, DM-54015
        sky_uu = self.sky_uu(df)
        sky_vv = self.sky_vv(df)
        sky_uv = self.sky_uv(df)
        denom = sky_uu + sky_vv + 2 * np.sqrt(sky_uu*sky_vv - sky_uv**2)
        return ((sky_uu - sky_vv) / denom).astype(np.float32)

    def get_g2(self, df):
        """
        Calculate shear-type ellipticity parameter G2.

        This has the opposite sign as sky_uv in order to maintain consistency with the HSM moments
        sign convention.
        """
        # TODO: Replace this with functionality from afwGeom, DM-54015
        sky_uu = self.sky_uu(df)
        sky_vv = self.sky_vv(df)
        sky_uv = self.sky_uv(df)
        denom = sky_uu + sky_vv + 2 * np.sqrt(sky_uu*sky_vv - sky_uv**2)
        return (-2*sky_uv / denom).astype(np.float32)

    def get_trace(self, df):
        sky_uu = self.sky_uu(df)
        sky_vv = self.sky_vv(df)
        return np.sqrt(0.5*(sky_uu + sky_vv)).astype(np.float32)


class MomentsG1Sky(MomentsBase):
    """Rotate pixel moments Ixx,Iyy,Ixy into RA/dec frame and G1/G2 reduced
    shear parameterization"""
    _defaultDataset = 'meas'
    name = "moments_g1"
    shortname = "moments_g1"

    def _func(self, df):
        sky_g1 = self.get_g1(df)

        return pd.Series(sky_g1.astype(np.float32), index=df.index)


class MomentsG2Sky(MomentsBase):
    """Rotate pixel moments Ixx,Iyy,Ixy into RA/dec frame and G1/G2 reduced
    shear parameterization"""
    _defaultDataset = 'meas'
    name = "moments_g2"
    shortname = "moments_g2"

    def _func(self, df):
        sky_g2 = self.get_g2(df)

        return pd.Series(sky_g2.astype(np.float32), index=df.index)


class MomentsTraceSky(MomentsBase):
    """Trace radius size in arcseconds from pixel moments Ixx,Iyy,Ixy

    The trace radius size is a measure of size equal to the square root of
    half of the trace of the second moments tensor.
    """
    _defaultDataset = 'meas'
    name = "moments_trace"
    shortname = "moments_trace"

    def _func(self, df):
        sky_trace_radians = self.get_trace(df)

        return pd.Series((sky_trace_radians*(180/np.pi)*3600).astype(np.float32), index=df.index)


class MomentsIuuSky(MomentsBase):
    """Rotate pixel moments Ixx,Iyy,Ixy into ra,dec frame and arcseconds"""
    _defaultDataset = 'meas'
    name = "moments_uu"
    shortname = "moments_uu"

    def _func(self, df):
        sky_uu_radians = self.sky_uu(df)

        return pd.Series((sky_uu_radians*((180/np.pi)*3600)**2).astype(np.float32), index=df.index)


class CorrelationIuuSky(MomentsIuuSky):
    """MomentsIuuSky but from sigma_x, sigma_y, rho correlation terms."""
    is_covariance = False


class MomentsIvvSky(MomentsBase):
    """Rotate pixel moments Ixx,Iyy,Ixy into ra,dec frame and arcseconds"""
    _defaultDataset = 'meas'
    name = "moments_vv"
    shortname = "moments_vv"

    def _func(self, df):
        sky_vv_radians = self.sky_vv(df)

        return pd.Series((sky_vv_radians*((180/np.pi)*3600)**2).astype(np.float32), index=df.index)


class CorrelationIvvSky(MomentsIvvSky):
    """MomentsIvvSky but from sigma_x, sigma_y, rho correlation terms."""
    is_covariance = False


class MomentsIuvSky(MomentsBase):
    """Rotate pixel moments Ixx,Iyy,Ixy into ra,dec frame and arcseconds"""
    _defaultDataset = 'meas'
    name = "moments_uv"
    shortname = "moments_uv"

    def _func(self, df):
        sky_uv_radians = self.sky_uv(df)

        return pd.Series((sky_uv_radians*((180/np.pi)*3600)**2).astype(np.float32), index=df.index)


class CorrelationIuvSky(MomentsIuvSky):
    """MomentsIuvSky but from sigma_x, sigma_y, rho correlation terms."""
    is_covariance = False


class PositionAngleFromMoments(MomentsBase):
    """Compute position angle relative to ra,dec frame, in degrees, from Ixx,Iyy,Ixy pixel moments."""
    _defaultDataset = 'meas'
    name = "moments_theta"
    shortname = "moments_theta"

    def _func(self, df):
        sky_uu = self.sky_uu(df)
        sky_vv = self.sky_vv(df)
        sky_uv = self.sky_uv(df)
        theta = 0.5*np.arctan2(2*sky_uv, sky_uu - sky_vv)

        return pd.Series((np.degrees(np.array(theta))).astype(np.float32), index=df.index)


class PositionAngleFromCorrelation(PositionAngleFromMoments):
    """PositionAngleFromMoments but from sigma_x, sigma_y, rho correlation terms."""
    is_covariance = False


class SemimajorAxisFromMoments(MomentsBase):
    """Compute the semimajor axis length in arcseconds, from Ixx,Iyy,Ixy pixel moments."""
    _defaultDataset = 'meas'
    name = "moments_a"
    shortname = "moments_a"

    def _func(self, df):
        xx_p_yy, _, t2 = self.compute_ellipse_terms(df)
        # This copies what is done (unvectorized) in afw.geom.ellipse
        a_radians = np.sqrt(0.5 * (xx_p_yy + np.sqrt(t2)))

        return pd.Series((np.degrees(a_radians)*3600).astype(np.float32), index=df.index)


class SemimajorAxisFromCorrelation(SemimajorAxisFromMoments):
    """SemimajorAxisFromMoments but from sigma_x, sigma_y, rho correlation terms."""
    is_covariance = False


class SemiminorAxisFromMoments(MomentsBase):
    """Compute the semiminor axis length in arcseconds, from Ixx,Iyy,Ixy pixel moments."""
    _defaultDataset = 'meas'
    name = "moments_b"
    shortname = "moments_b"

    def _func(self, df):
        xx_p_yy, _, t2 = self.compute_ellipse_terms(df)
        # This copies what is done (unvectorized) in afw.geom.ellipse
        b_radians = np.sqrt(0.5 * (xx_p_yy - np.sqrt(t2)))

        return pd.Series((np.degrees(b_radians)*3600).astype(np.float32), index=df.index)


class SemiminorAxisFromCorrelation(SemiminorAxisFromMoments):
    """SemiminorAxisFromMoments but from sigma_x, sigma_y, rho correlation terms."""
    is_covariance = False
