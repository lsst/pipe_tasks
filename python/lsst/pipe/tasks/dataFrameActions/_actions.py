from __future__ import annotations

__all__ = ("SingleColumnAction", "MultiColumnAction", "CoordColumn", "MagColumn", "SumColumns", "AddColumn",
           "DivideColumns", "SubtractColumns", "MultiplyColumns", "BaseBinOp")

from typing import Iterable

import numpy as np
import pandas as pd

from ..configurableActions import ConfigurableActionStructField, ConfigurableActionField
from ._baseDataFrameActions import DataFrameAction

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
        yield from (column for action, _ in self.actions for column in action.columns)


class CoordColumn(SingleColumnAction):
    inRadians = Field(doc="Return the column in radians if true", default=True, dtype=bool)

    def __call__(self, df):
        col = super().__call__(df)
        return col * 180 / np.pi if self.inRadians else col


class MagColumn(SingleColumnAction):
    def __call__(self, df: pd.DataFrame, calib=None):
        fluxMag0: float = calib.getFluxMag0()[0] if calib else 63095734448.0194
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value encountered')
            np.warnings.filterwarnings('ignore', r'divide by zero')
            return -2.5 * np.log10(df[self.column] / fluxMag0)


class BaseBinOp(MultiColumnAction):
    """This is a baseclass for various binary operator actions
    """
    def setDefaults(self):
        super().setDefaults()
        self.actions.ColA = SingleColumnAction
        self.actions.ColB = SingleColumnAction


class SumColumns(BaseBinOp):
    """This is a `MultiColumnAction` that is designed to add two columns
    together and return the result. The names in the `actions` field should
    be ColA and ColB
    """
    def __call__(self, df, **kwargs):
        return self.actions.ColA(df, kwargs) + self.actions.ColB(df, kwargs)


class SubtractColumns(BaseBinOp):
    """This is a `MultiColumnAction` that is designed to subtract two columns
    together and return the result. The names in the `actions` field should
    be ColA and ColB
    """
    def __call__(self, df, **kwargs):
        return self.actions.ColA(df, kwargs) - self.actions.ColB(df, kwargs)


class MultiplyColumns(BaseBinOp):
    """This is a `MultiColumnAction` that is designed to multiply two columns
    together and return the result. The names in the `actions` field should
    be ColA and ColB
    """
    def __call__(self, df, **kwargs):
        return self.actions.ColA(df, kwargs) * self.actions.ColB(df, kwargs)


class DivideColumns(BaseBinOp):
    """This is a `MultiColumnAction` that is designed to multiply two columns
    together and return the result. The names in the `actions` field should
    be ColA and ColB
    """
    def __call__(self, df, **kwargs):
        return self.actions.ColA(df, kwargs) / self.actions.ColB(df, kwargs)


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
