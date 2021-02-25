from __future__ import annotations

__all__ = ("loadDataFrameActionColumns", )

from typing import Iterable, Union, Tuple
import pandas as pd

from lsst.pipe.base import InputQuantizedConnection
from lsst.daf.butler import DeferredDatasetHandle, DatasetRef

from . import DataFrameAction, AddColumn


def loadDataFrameActionColumns(connectionName: str,
                               inputs: InputQuantizedConnection,
                               actions: Iterable[Union[DataFrameAction, Iterable[DataFrameAction]]]
                               ) -> Tuple[pd.DataFrame, DeferredDatasetHandle]:
    # pull the DataRef out of the inputs
    dataFrameRef = getattr(inputs, connectionName)
    delattr(inputs, connectionName)
    if not isinstance(DatasetRef, DeferredDatasetHandle):
        raise ValueError("This method can only be used with a connection marked to defer load.")

    # accumulate all the required columns
    generator = (_ for _ in tuple())
    for action in actions:
        if not isinstance(action, Iterable):
            action = tuple(action)
        generator = (value for value in zip(action, generator))
    needed_columns = set()
    columns_to_add = set()
    for action in generator:
        if isinstance(action, AddColumn):
            columns_to_add.add(action.newColumn)
        needed_columns.add(action.columns)

    # remove columns that will be added during processing
    needed_columns -= columns_to_add

    # load in the columns
    existing_columns = set(dataFrameRef.get(component="columns"))

    if missing := existing_columns.difference(needed_columns):
        raise ValueError(f"Configurable actions require the columns {missing} which were not found in the ",
                         "input DataFrame")

    dataFrame = dataFrameRef.get(parameters={"columns": needed_columns})
    return dataFrame, dataFrameRef
