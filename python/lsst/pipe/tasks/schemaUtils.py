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


"""Utilities for working with sdm_schemas.
"""
__all__ = ("convertDataFrameToSdmSchema", "readSdmSchemaFile",
           "dropEmptyColumns", "make_empty_catalog", "checkSdmSchemaColumns")

from collections.abc import Mapping
import os

import felis.datamodel
import numpy as np
import pandas as pd
from astropy.table import Table
import yaml


# The first entry in the returned mapping is for nullable columns,
# the second entry is for non-nullable columns.
_dtype_map: Mapping[felis.datamodel.DataType, tuple[str, str]] = {
    felis.datamodel.DataType.double: ("float64", "float64"),  # Cassandra utilities need np.nan not pd.NA
    felis.datamodel.DataType.float: ("float32", "float32"),  # Cassandra utilities need np.nan not pd.NA
    felis.datamodel.DataType.timestamp: ("datetime64[ms]", "datetime64[ms]"),
    felis.datamodel.DataType.long: ("Int64", "int64"),
    felis.datamodel.DataType.int: ("Int32", "int32"),
    felis.datamodel.DataType.short: ("Int16", "int16"),
    felis.datamodel.DataType.byte: ("Int8", "int8"),
    felis.datamodel.DataType.binary: ("object", "object"),
    felis.datamodel.DataType.char: ("object", "object"),
    felis.datamodel.DataType.text: ("object", "object"),
    felis.datamodel.DataType.string: ("object", "object"),
    felis.datamodel.DataType.unicode: ("object", "object"),
    felis.datamodel.DataType.boolean: ("boolean", "bool"),
}


def column_dtype(felis_type: felis.datamodel.DataType | schema_model.ExtraDataTypes,
                 nullable=False) -> str:
    """Return Pandas data type for a given Felis column type.

    Parameters
    ----------
    felis_type : `felis.datamodel.DataType`
        Felis type, on of the enums defined in `felis.datamodel` module.

    Returns
    -------
    column_dtype : `type` or `str`
        Type that can be used for columns in Pandas.

    Raises
    ------
    TypeError
        Raised if type is cannot be handled.
    """
    try:
        return _dtype_map[felis_type][0] if nullable else _dtype_map[felis_type][1]
    except KeyError:
        raise TypeError(f"Unexpected Felis type: {felis_type}")


def readSdmSchemaFile(schemaFile: str,
                      schemaName: str = "ApdbSchema",
                      ):
    """Read a schema file in YAML format.

    Parameters
    ----------
    schemaFile : `str`
        Fully specified path to the file to be read.
    schemaName : `str`, optional
        Name of the table of schemas to read from the file.

    Returns
    -------
    schemaTable : dict[str, schema_model.Table]
        A dict of the schemas in the given table defined in the specified file.

    Raises
    ------
    ValueError
        If the schema file can't be parsed.
    """
    schemaFile = os.path.expandvars(schemaFile)
    with open(schemaFile) as yaml_stream:
        schemas_list = list(yaml.load_all(yaml_stream, Loader=yaml.SafeLoader))
        schemas_list = [schema for schema in schemas_list if schema.get("name") == schemaName]
        if not schemas_list:
            raise ValueError(f"Schema file {schemaFile!r} does not define schema {schemaName!r}")
        elif len(schemas_list) > 1:
            raise ValueError(f"Schema file {schemaFile!r} defines multiple schemas {schemaName!r}")
        felis_schema = felis.datamodel.Schema.model_validate(schemas_list[0],
                                                             context={'id_generation': True}
                                                             )
        schema = schema_model.Schema.from_felis(felis_schema)
    schemaTable = {}

    for singleTable in schema.tables:
        schemaTable[singleTable.name] = singleTable
    return schemaTable


def checkSdmSchemaColumns(apdbSchema, colNames, tableName):
    """Check if supplied column names exists in the schema.

    Parameters
    ----------
    apdbSchema : `dict` [`str`, `lsst.dax.apdb.schema_model.Table`]
        Schema from ``sdm_schemas`` containing the table definition to use.
    colNames : `list` of ``str`
        Names of the columns to check for in the table.
    tableName : `str`
        Name of the table in the schema to use.

    Returns
    -------
    missing : `list` of `str`
        All column names that are not in the schema
    """
    table = apdbSchema[tableName]
    missing = []

    names = [columnDef.name for columnDef in table.columns]
    for col in colNames:
        if col not in names:
            missing.append(col)
    return missing


def convertDataFrameToSdmSchema(apdbSchema, sourceTable, tableName):
    """Force a table to conform to the schema defined by the APDB.

    This method uses the table definitions in ``sdm_schemas`` to
    load the schema of the APDB, and does not actually connect to the APDB.

    Parameters
    ----------
    apdbSchema : `dict` [`str`, `lsst.dax.apdb.schema_model.Table`]
        Schema from ``sdm_schemas`` containing the table definition to use.
    sourceTable : `pandas.DataFrame`
        The input table to convert.
    tableName : `str`
        Name of the table in the schema to use.

    Returns
    -------
    `pandas.DataFrame`
        A table with the correct schema for the APDB and data copied from
        the input ``sourceTable``.
    """
    if sourceTable.empty:
        make_empty_catalog(apdbSchema, tableName)
    table = apdbSchema[tableName]

    data = {}
    nSrc = len(sourceTable)

    for columnDef in table.columns:
        dtype = column_dtype(columnDef.datatype, nullable=columnDef.nullable)
        if columnDef.name in sourceTable.columns:
            data[columnDef.name] = pd.Series(sourceTable[columnDef.name], dtype=dtype,
                                             index=sourceTable.index)
        else:
            if columnDef.nullable:
                try:
                    data[columnDef.name] = pd.Series([pd.NA]*nSrc, dtype=dtype, index=sourceTable.index)
                except TypeError:
                    data[columnDef.name] = pd.Series([np.nan]*nSrc, dtype=dtype, index=sourceTable.index)
            else:
                data[columnDef.name] = pd.Series([0]*nSrc, dtype=dtype, index=sourceTable.index)
    return pd.DataFrame(data)


def convertTableToSdmSchema(apdbSchema, sourceTable, tableName):
    """Force a table to conform to the schema defined by the APDB.

    This method uses the table definitions in ``sdm_schemas`` to
    load the schema of the APDB, and does not actually connect to the APDB.

    Parameters
    ----------
    apdbSchema : `dict` [`str`, `lsst.dax.apdb.schema_model.Table`]
        Schema from ``sdm_schemas`` containing the table definition to use.
    sourceTable : `astropy.table.Table`
        The input table to convert.
    tableName : `str`
        Name of the table in the schema to use.

    Returns
    -------
    `astropy.table.Table`
        A table with the correct schema for the APDB and data copied from
        the input ``sourceTable``.
    """
    table = apdbSchema[tableName]

    data = {}
    nSrc = len(sourceTable)

    for columnDef in table.columns:
        dtype = column_dtype(columnDef.datatype, nullable=columnDef.nullable)
        if columnDef.name in sourceTable.columns:
            data[columnDef.name] = Table.Column(sourceTable[columnDef.name], dtype=dtype.lower())
        else:
            if columnDef.nullable:
                try:
                    data[columnDef.name] = Table.Column([pd.NA]*nSrc, dtype=object)
                except TypeError:
                    data[columnDef.name] = Table.Column([pd.nan]*nSrc, dtype=dtype)
            else:
                data[columnDef.name] = Table.Column([0]*nSrc, dtype=dtype)
    return Table(data)


def dropEmptyColumns(apdbSchema, sourceTable, tableName):
    """Drop empty columns that are nullable.

    This method uses the table definitions in ``sdm_schemas`` to
    load the schema of the APDB, and does not actually connect to the APDB.

    Parameters
    ----------
    apdbSchema : `dict` [`str`, `lsst.dax.apdb.schema_model.Table`]
        Schema from ``sdm_schemas`` containing the table definition to use.
    sourceTable : `pandas.DataFrame`
        The input table to remove missing data columns from.
    tableName : `str`
        Name of the table in the schema to use.
    """
    table = apdbSchema[tableName]

    nullableList = [columnDef.name for columnDef in table.columns if columnDef.nullable]
    nullColumns = sourceTable.isnull().all()
    nullColNames = nullColumns[nullColumns].index.tolist()
    dropColumns = list(set(nullColNames) & set(nullableList))
    return sourceTable.drop(columns=dropColumns)


def make_empty_catalog(apdbSchema, tableName):
    """Make an empty catalog for a table with a given name.

    Parameters
    ----------
    apdbSchema : `dict` [`str`, `lsst.dax.apdb.schema_model.Table`]
        Schema from ``sdm_schemas`` containing the table definition to use.
    tableName : `str`
        Name of the table in the schema to use.

    Returns
    -------
    catalog : `pandas.DataFrame`
        An empty catalog.
    """
    table = apdbSchema[tableName]

    data = {
        columnDef.name: pd.Series(dtype=column_dtype(columnDef.datatype, nullable=columnDef.nullable))
        for columnDef in table.columns
    }
    return pd.DataFrame(data)
