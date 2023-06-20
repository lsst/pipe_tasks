from __future__ import annotations

__all__ = ("SingleColumnAction", "MultiColumnAction", "CoordColumn", "MagColumnDN", "SumColumns", "AddColumn",
           "DivideColumns", "SubtractColumns", "MultiplyColumns", "FractionalDifferenceColumns",
           "MagColumnNanoJansky", "DiffOfDividedColumns", "PercentDiffOfDividedColumns",)

from typing import Iterable

import warnings
import numpy as np
import pandas as pd
from astropy import units

from lsst.pex.config.configurableActions import ConfigurableActionStructField, ConfigurableActionField
from ._baseDataFrameActions import DataFrameAction
from ._evalColumnExpression import makeColumnExpressionAction

from lsst.pex.config import Field


class SingleColumnAction(DataFrameAction):
    column = Field(doc="Column to load for this action", dtype=str, optional=False)

    @property
    def columns(self) -> Iterable[str]:
        return (self.column, )

    def __call__(self, df, **kwargs):
        return df[self.column]


class MultiColumnAction(DataFrameAction):
    actions = ConfigurableActionStructField(doc="Configurable actions to use in a joint action")

    @property
    def columns(self) -> Iterable[str]:
        yield from (column for action in self.actions for column in action.columns)


class CoordColumn(SingleColumnAction):
    inRadians = Field(doc="Return the column in radians if true", default=True, dtype=bool)

    def __call__(self, df):
        col = super().__call__(df)
        return col * 180 / np.pi if self.inRadians else col


class MagColumnDN(SingleColumnAction):
    coadd_zeropoint = Field(doc="Magnitude zero point", dtype=float, default=27)

    def __call__(self, df: pd.DataFrame, **kwargs):
        if not (fluxMag0 := kwargs.get('fluxMag0')):
            fluxMag0 = 1/np.power(10, -0.4*self.coadd_zeropoint)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered')
            warnings.filterwarnings('ignore', r'divide by zero')
            return -2.5 * np.log10(df[self.column] / fluxMag0)


class MagColumnNanoJansky(SingleColumnAction):

    def __call__(self, df: pd.DataFrame, **kwargs):

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered')
            warnings.filterwarnings('ignore', r'divide by zero')
            return -2.5 * np.log10((df[self.column] * 1e-9) / 3631.0)


class NanoJansky(SingleColumnAction):
    ab_flux_scale = Field(doc="Scaling of ab flux", dtype=float, default=(0*units.ABmag).to_value(units.nJy))
    coadd_zeropoint = Field(doc="Magnitude zero point", dtype=float, default=27)

    def __call__(self, df, **kwargs):
        dataNumber = super().__call__(df, **kwargs)
        if not (fluxMag0 := kwargs.get('fluxMag0')):
            fluxMag0 = 1/np.power(10, -0.4*self.coadd_zeropoint)
        return self.ab_flux_scale * dataNumber / fluxMag0

    def setDefaults(self):
        super().setDefaults()
        self.cache = True  # cache this action for future calls


class NanoJanskyErr(SingleColumnAction):
    flux_mag_err = Field(doc="Error in the magnitude zeropoint", dtype=float, default=0)
    flux_action = ConfigurableActionField(doc="Action to use if flux is not provided to the call method",
                                          default=NanoJansky, dtype=DataFrameAction)

    @property
    def columns(self):
        yield from zip((self.column,), self.flux_action.columns)

    def __call__(self, df, flux_column=None, flux_mag_err=None, **kwargs):
        if flux_column is None:
            flux_column = self.flux_action(df, **kwargs)
        if flux_mag_err is None:
            flux_mag_err = self.flux_mag_err


_docs = """This is a `DataFrameAction` that is designed to add two columns
together and return the result.
"""
SumColumns = makeColumnExpressionAction("SumColumns", "colA+colB",
                                        exprDefaults={"colA": SingleColumnAction,
                                                      "colB": SingleColumnAction},
                                        docstring=_docs)

_docs = """This is a `MultiColumnAction` that is designed to subtract two columns
together and return the result.
"""
SubtractColumns = makeColumnExpressionAction("SubtractColumns", "colA-colB",
                                             exprDefaults={"colA": SingleColumnAction,
                                                           "colB": SingleColumnAction},
                                             docstring=_docs)

_docs = """This is a `MultiColumnAction` that is designed to multiply two columns
together and return the result.
"""
MultiplyColumns = makeColumnExpressionAction("MultiplyColumns", "colA*colB",
                                             exprDefaults={"colA": SingleColumnAction,
                                                           "colB": SingleColumnAction},
                                             docstring=_docs)

_docs = """This is a `MultiColumnAction` that is designed to divide two columns
together and return the result.
"""
DivideColumns = makeColumnExpressionAction("DivideColumns", "colA/colB",
                                           exprDefaults={"colA": SingleColumnAction,
                                                         "colB": SingleColumnAction},
                                           docstring=_docs)

_docs = """This is a `MultiColumnAction` that is designed to divide two columns
together, subtract one and return the result.
"""
FractionalDifferenceColumns = makeColumnExpressionAction("FractionalDifferenceColumns", "(colA-colB)/colB",
                                                         exprDefaults={"colA": SingleColumnAction,
                                                                       "colB": SingleColumnAction},
                                                         docstring=_docs)

_docs = """This is a `MultiColumnAction` that is designed to subtract the division of two columns
from the division of two other columns and return the result (i.e. colA1/colB1 - colA2/colB2).
"""
DiffOfDividedColumns = makeColumnExpressionAction("DiffOfDividedColumns", "(colA1/colB1)-(colA2/colB2)",
                                                  exprDefaults={"colA1": SingleColumnAction,
                                                                "colB1": SingleColumnAction,
                                                                "colA2": SingleColumnAction,
                                                                "colB2": SingleColumnAction},
                                                  docstring=_docs)
_docs = """This is a `MultiColumnAction` that is designed to compute the percent difference
between the division of two columns and the division of two other columns and return the result
(i.e. 100*((colA1/colB1 - colA2/colB2)/(colA1/colB1))).
"""
PercentDiffOfDividedColumns = makeColumnExpressionAction("PercentDiffOfDividedColumns",
                                                         "100*(((colA1/colB1)-(colA2/colB2))/(colA1/colB1))",
                                                         exprDefaults={"colA1": SingleColumnAction,
                                                                       "colB1": SingleColumnAction,
                                                                       "colA2": SingleColumnAction,
                                                                       "colB2": SingleColumnAction},
                                                         docstring=_docs)


class AddColumn(DataFrameAction):
    aggregator = ConfigurableActionField(doc="This is an instance of a Dataframe action that will be used "
                                             "to create a new column", dtype=DataFrameAction)
    newColumn = Field(doc="Name of the new column to add", dtype=str)

    @property
    def columns(self) -> Iterable[str]:
        yield from self.aggregator.columns

    def __call__(self, df, **kwargs) -> pd.DataFrame:
        # do your calculation and and
        df[self.newColumn] = self.aggregator(df, kwargs)
        return df
