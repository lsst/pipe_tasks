from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..configurableActions import ConfigurableActionsField
from ._baseDataFrameActions import DataFrameAction

from lsst.pex.config import Field, ConfigField


class SingleColumnAction(DataFrameAction):
    column = Field(doc="Column to load for this action", dtype=str)

    @property
    def columns(self) -> Iterable[str]:
        return (self.column, )

    def __call__(self, df):
        return df[self.column]


class MultiColumnAction(DataFrameAction):
    actions = ConfigurableActionsField(doc="Configurable actions to use in a joint action")

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


class SumColumns(MultiColumnAction):
    """This is a `MultiColumnAction` that is designed to add two columns
    together and return the result. The names in the `actions` field should
    be ColA and ColB
    """
    def __call__(self, df, **kwargs):
        return self.actions.ColA(df, kwargs) + self.actions.ColB(df, kwargs)


class AddColumn(DataFrameAction):
    aggregator = ConfigField(doc="This is an instance of a Dataframe action that will be used to "
                             "create a new column", dtype=DataFrameAction)
    newColumn = Field(doc="Name of the new column to add", dtype=str)

    @property
    def columns(self) -> Iterable[str]:
        yield from self.aggregator.columns

    def __call__(self, df, **kwargs) -> pd.DataFrame:
        # do your calculation and and
        df[self.newColumn] = self.aggregator(df, kwargs)
        return df
