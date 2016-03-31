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
