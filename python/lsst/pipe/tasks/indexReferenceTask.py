#
# LSST Data Management System
# Copyright 2008-2016 AURA/LSST.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

import numpy


class TextReaderConfig(pexConfig.Config):
    header_lines = pexConfig.Field(
        dtype=int,
        default=0,
        doc='Number of lines to skip when reading the text reference file.'
    )
    colnames = pexConfig.ListField(
        dtype=str,
        default=[],
        doc="""An ordered list of column names to use in ingesting the catalog.  With an empty
list, column names will be discovered from the first line after the skipped header lines."""
    )
    delimiter = pexConfig.Field(
        dtype=str,
        default=',',
        doc='Delimiter to use when reading text reference files.  Comma is default.'
    )


class TextReaderTask(pipeBase.Task):
    _DefaultName = 'TextReaderTask'
    ConfigClass = TextReaderConfig

    def readFile(self, filename):
            names = True
            if self.config.colnames:
                names = self.config.colnames
            arr = numpy.genfromtxt(filename, dtype=None, skip_header=self.config.header_lines,
                                   delimiter=self.config.delimiter,
                                   names=names)
            # Just in case someone has only one line in the file.
            return numpy.atleast_1d(arr)


class IngestIndexedReferenceConfig(pexConfig.Config):
    ref_dataset_name = pexConfig.Field(
        dtype=str,
        default='cal_ref_cat',
        doc='String to pass to the butler to retrieve persisted files.',
    )
    level = pexConfig.Field(
        dtype=int,
        default=8,
        doc='Default HTM level.  Level 8 gives ~0.08 sq deg per trixel.',
    )
    file_reader = pexConfig.ConfigurableField(
        target=TextReaderTask,
        doc='Task to use to read the files.  Default is to expect text files.'
    )
    ra_name = pexConfig.Field(
        dtype=str,
        doc="Name of RA column",
    )
    dec_name = pexConfig.Field(
        dtype=str,
        doc="Name of Dec column",
    )
    mag_column_list = pexConfig.ListField(
        dtype=str,
        doc="""The values in the reference catalog are assumed to be in AB magnitudes.
List of column names to use for photometric information.  At least one entry is required."""
    )
    mag_err_column_map = pexConfig.DictField(
        keytype=str,
        itemtype=str,
        default={},
        doc="A map of magnitude column name (key) to magnitude error column (value)."
    )
    is_photometric_name = pexConfig.Field(
        dtype=str,
        optional=True,
        doc='Name of column stating if satisfactory for photometric calibration (optional).'
    )
    is_resolved_name = pexConfig.Field(
        dtype=str,
        optional=True,
        doc='Name of column stating if the object is resolved (optional).'
    )
    is_variable_name = pexConfig.Field(
        dtype=str,
        optional=True,
        doc='Name of column stating if the object is measured to be variable (optional).'
    )
    id_name = pexConfig.Field(
        dtype=str,
        optional=True,
        doc='Name of column to use as an identifier (optional).'
    )
    extra_col_names = pexConfig.ListField(
        dtype=str,
        default=[],
        doc='Extra columns to add to the reference catalog.'
    )

    def validate(self):
        pexConfig.Config.validate(self)
        if not (self.ra_name and self.dec_name and self.mag_column_list):
            raise ValueError("ra_name and dec_name and at least one entry in mag_column_list must be" +
                             " supplied.")
        if len(self.mag_err_column_map) > 0 and not len(self.mag_column_list) == len(self.mag_err_column_map):
            raise ValueError("If magnitude errors are provided, all magnitudes must have an error column")
