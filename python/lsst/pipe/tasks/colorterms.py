# 
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011 LSST Corporation.
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
# see <http://www.lsstcorp.org/LegalNotices/>.
#

import fnmatch

import numpy as np

from lsst.pex.config import Config, Field, ConfigDictField
from lsst.afw.image import Filter

__all__ = ["ColortermNotFoundError", "Colorterm", "ColortermDictConfig", "ColortermLibraryConfig"]

class ColortermNotFoundError(LookupError):
    """Exception class indicating we couldn't find a colorterm
    """
    pass


class Colorterm(Config):
    """!Configuration describing a Colorterm
    
    The transformed magnitude p' is given by
        p' = primary + c0 + c1*(primary - secondary) + c2*(primary - secondary)**2
    """
    primary = Field(dtype=str, doc="name of primary filter")
    secondary = Field(dtype=str, doc="name of secondary filter")
    c0 = Field(dtype=float, default=0.0, doc="Constant parameter")
    c1 = Field(dtype=float, default=0.0, doc="First-order parameter")
    c2 = Field(dtype=float, default=0.0, doc="Second-order parameter")

    # the following is desired to allow positional data, but is impossible due to DM-2381
    # def __init__(self, primary, secondary, c0=0.0, c1=0.0, c2=0.0):
    #     """!Construct a Colorterm

    #     This overrides the default constructor for improved error detection
    #     and to allow positional arguments
    #     """
    #     Config.__init__(self, primary=primary, secondary=secondary, c0=c0, c1=c1, c2=c2)

    def transformSource(self, source):
        """!Transform the brightness of a source

        @param[in] source  source whose brightness is to be converted; must support get(filterName)
                    (e.g. source.get("r")) method, as do afw::table::Source and dicts.
        @return the transformed source magnitude
        """
        return self.transformMags(source.get(self.primary), source.get(self.secondary))

    def transformMags(self, primary, secondary):
        """!Transform primary and secondary magnitudes to a magnitude
        
        @param[in] primary  magnitude in primary filter
        @param[in] secondary  magnitude in secondary filter
        @return the transformed magnitude
        """
        color = primary - secondary
        return primary + self.c0 + color*(self.c1 + color*self.c2)

    def propagateFluxErrors(self, primaryFluxErr, secondaryFluxErr):
        return np.hypot((1 + self.c1)*primaryFluxErr, self.c1*secondaryFluxErr)


class ColortermDictConfig(Config):
    """!A config containing dict: a dict of filterName: Colorterm

    Different reference catalogs may need different ColortermDictConfigs; see ColortermLibrary
    """
    dict = ConfigDictField(
        doc="Mapping of filter name to Colorterm instance",
        keytype=str,
        itemtype=Colorterm,
        default={},
    )

    # the following is desired to allow positional data and avoid DM-2382, but is impossible due to DM-2381
    # def __init__(self, dict=None):
    #     """!Construct a ColortermDictConfig

    #     Overrides the default constructor for improved safety
    #     and to allow the data to be a positional argument
    #     """
    #     if dict is None: # mutable objects should not be used as defaults
    #         dict = {}
    #     Config.__init__(self, dict=dict)


class ColortermLibraryConfig(Config):
    """!A config containing library: a dict of catalog name: ColortermDictConfig

    This is intended to support a particular camera with a variety of reference catalogs
    """
    library = ConfigDictField(
        doc="Mapping of reference catalog name (or glob) to group of color terms",
        keytype=str,
        itemtype=ColortermDictConfig,
        default={},
    )

    # the following is desired to allow positional data and avoid DM-2382, but is impossible due to DM-2381
    # def __init__(self, library=None):
    #     """!Construct a ColortermLibraryConfig

    #     Overrides the default constructor for improved safety
    #     and to allow the data to be a positional argument
    #     """
    #     if library is None: # mutable objects should not be used as defaults
    #         library = {}
    #     Config.__init__(self, library=library)

    def getColorterm(self, filterName, refCatName, doRaise=True):
        """!Get the appropriate Colorterm from the library

        We use the group of color terms in the library that matches the refCatName.
        If the refCatName exactly matches an entry in the library, that
        group is used; otherwise if the refCatName matches a single glob (shell syntax,
        e.g., "sdss-*" will match "sdss-dr8"), then that is used.  If there is no
        exact match and no unique match to the globs, we raise an exception.

        @param filterName  name of filter
        @param refCatName  reference catalog name or glob expression; if a glob expression then
            there must be exactly one match in the library
        @param[in] doRaise  if True then raise ColortermNotFoundError if no suitable Colorterm found;
            if False then return a null Colorterm
        @return the appropriate Colorterm

        @throw ColortermNotFoundError if no suitable Colorterm found and doRaise true
        """
        try:
            trueRefCatName = None
            ctDictConfig = self.library.get(refCatName)
            if ctDictConfig is None:
                # try glob expression
                matchList = [glob for glob in self.library if fnmatch.fnmatch(refCatName, glob)]
                if len(matchList) == 1:
                    trueRefCatName = matchList[0]
                    ctDictConfig = self.library[trueRefCatName]
                elif len(matchList) > 1:
                    raise RuntimeError(
                        "Multiple library globs match refCatName %r: %s" % (refCatName, matchList))
                else:
                    raise RuntimeError("No colorterm dict found with refCatName %r" % refCatName)
            ctDict = ctDictConfig.dict
            if filterName not in ctDict:
                # Perhaps it's an alias
                filterName = Filter(Filter(filterName).getId()).getName()
                if filterName not in ctDict:
                    errMsg = "No colorterm found for filter %r with refCatName %r" % (filterName, refCatName)
                    if trueRefCatName is not None:
                        errMsg += " = catalog %r" % (trueRefCatName,)
                    raise ColortermNotFoundError(errMsg)
            return ctDict[filterName]
        except Exception:
            if doRaise:
                raise
            else:
                return Colorterm(filterName, filterName)
