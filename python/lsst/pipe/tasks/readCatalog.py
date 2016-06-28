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
import numpy as np

from astropy.io import fits
from astropy.table import Table

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

__all__ = ["ReadTextCatalogConfig", "ReadTextCatalogTask", "ReadFitsCatalogConfig", "ReadFitsCatalogTask"]


class ReadTextCatalogConfig(pexConfig.Config):
    header_lines = pexConfig.Field(
        dtype=int,
        default=0,
        doc='Number of lines to skip when reading the text reference file.'
    )
    colnames = pexConfig.ListField(
        dtype=str,
        default=[],
        doc="An ordered list of column names to use in ingesting the catalog. "
            "With an empty list, column names will be discovered from the first line "
            "after the skipped header lines."
    )
    delimiter = pexConfig.Field(
        dtype=str,
        default=',',
        doc='Delimiter to use when reading text reference files.  Comma is default.'
    )


class ReadTextCatalogTask(pipeBase.Task):
    """Read an object catalog from a text file
    """
    _DefaultName = 'readCatalog'
    ConfigClass = ReadTextCatalogConfig

    def run(self, filename):
        """Read an object catalog from the specified text file

        @param[in] filename  path to text file
        @return a numpy structured array containing the specified columns
        """
        names = True
        if self.config.colnames:
            names = self.config.colnames
        arr = np.genfromtxt(filename, dtype=None, skip_header=self.config.header_lines,
                            delimiter=self.config.delimiter,
                            names=names)
        # Just in case someone has only one line in the file.
        return np.atleast_1d(arr)


class ReadFitsCatalogConfig(pexConfig.Config):
    hdu = pexConfig.Field(
        dtype=int,
        default=1,
        doc="HDU containing the desired binary table, 0-based but a binary table never occurs in HDU 0",
    )
    column_map = pexConfig.DictField(
        doc="Mapping of input column name: output column name; each specified column must exist, "
            "but additional columns in the input data are written using their original name. ",
        keytype=str,
        itemtype=str,
        default={},
    )


class ReadFitsCatalogTask(pipeBase.Task):
    """Read an object catalog from a FITS binary table
    """
    _DefaultName = 'readCatalog'
    ConfigClass = ReadFitsCatalogConfig

    def run(self, filename):
        """Read an object catalog from the specified FITS file

        @param[in] filename  path to FITS file
        @return a numpy structured array containing the specified columns
        """
        with fits.open(filename) as f:
            hdu = f[self.config.hdu]
            if hdu.data is None:
                raise RuntimeError("No data found in %s HDU %s" % (filename, self.config.hdu))
            if hdu.is_image:
                raise RuntimeError("%s HDU %s is an image" % (filename, self.config.hdu))

            if not self.config.column_map:
                # take the data as it is
                return hdu.data

            # some columns need to be renamed; use astropy table
            table = Table(hdu.data, copy=False)
            missingnames = set(self.config.column_map.keys()) - set(table.colnames)
            if missingnames:
                raise RuntimeError("Columns %s in column_map were not found in %s" % (missingnames, filename))
            for inname, outname in self.config.column_map.iteritems():
                table.rename_column(inname, outname)
            return np.array(table)  # convert the astropy table back to a numpy structured array
