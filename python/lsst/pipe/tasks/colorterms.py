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

import lsst.pex.exceptions as pexExcept
from lsst.pex.config import Config, Field, ConfigDictField
from lsst.afw.image import Filter

__all__ = ["ColortermNotFoundError", "Colorterm", "ColortermDict", "ColortermLibrary"]


class ColortermNotFoundError(LookupError):
    """Exception class indicating we couldn't find a colorterm
    """
    pass


class Colorterm(Config):
    """!Colorterm correction for one pair of filters

    The transformed magnitude p' is given by
        p' = primary + c0 + c1*(primary - secondary) + c2*(primary - secondary)**2

    To construct a Colorterm, use keyword arguments:
    Colorterm(primary=primaryFilterName, secondary=secondaryFilterName, c0=c0value, c1=c1Coeff, c2=c2Coeff)
    where c0-c2 are optional. For example (omitting c2):
    Colorterm(primary="g", secondary="r", c0=-0.00816446, c1=-0.08366937)

    This is subclass of Config. That is a bit of a hack to make it easy to store the data
    in an appropriate obs_* package as a config override file. In the long term some other
    means of persistence will be used, at which point the constructor can be simplified
    to not require keyword arguments. (Fixing DM-2831 will also allow making a custom constructor).
    """
    primary = Field(dtype=str, doc="name of primary filter")
    secondary = Field(dtype=str, doc="name of secondary filter")
    c0 = Field(dtype=float, default=0.0, doc="Constant parameter")
    c1 = Field(dtype=float, default=0.0, doc="First-order parameter")
    c2 = Field(dtype=float, default=0.0, doc="Second-order parameter")

    def transformSource(self, source):
        """!Transform the brightness of a source

        @param[in] source  source whose brightness is to be converted; must support get(filterName)
                    (e.g. source.get("r")) method, as do afw::table::Source and dicts.
        @return the transformed source magnitude
        """
        return self.transformMags(source.get(self.primary), source.get(self.secondary))

    def transformMags(self, primary, secondary):
        """!Transform brightness

        @param[in] primary  brightness in primary filter (magnitude)
        @param[in] secondary  brightness in secondary filter (magnitude)
        @return the transformed brightness (as a magnitude)
        """
        color = primary - secondary
        return primary + self.c0 + color*(self.c1 + color*self.c2)

    def propagateFluxErrors(self, primaryFluxErr, secondaryFluxErr):
        return np.hypot((1 + self.c1)*primaryFluxErr, self.c1*secondaryFluxErr)


class ColortermDict(Config):
    """!A mapping of filterName to Colorterm

    Different reference catalogs may need different ColortermDicts; see ColortermLibrary

    To construct a ColortermDict use keyword arguments:
    ColortermDict(data=dataDict)
    where dataDict is a Python dict of filterName: Colorterm
    For example:
    ColortermDict(data={
        'g':    Colorterm(primary="g", secondary="r", c0=-0.00816446, c1=-0.08366937, c2=-0.00726883),
        'r':    Colorterm(primary="r", secondary="i", c0= 0.00231810, c1= 0.01284177, c2=-0.03068248),
        'i':    Colorterm(primary="i", secondary="z", c0= 0.00130204, c1=-0.16922042, c2=-0.01374245),
    })
    The constructor will likely be simplified at some point.

    This is subclass of Config. That is a bit of a hack to make it easy to store the data
    in an appropriate obs_* package as a config override file. In the long term some other
    means of persistence will be used, at which point the constructor can be made saner.
    """
    data = ConfigDictField(
        doc="Mapping of filter name to Colorterm",
        keytype=str,
        itemtype=Colorterm,
        default={},
    )


class ColortermLibrary(Config):
    """!A mapping of photometric reference catalog name or glob to ColortermDict

    This allows photometric calibration using a variety of reference catalogs.

    To construct a ColortermLibrary, use keyword arguments:
    ColortermLibrary(data=dataDict)
    where dataDict is a Python dict of catalog_name_or_glob: ColortermDict

    For example:
    ColortermLibrary(data = {
        "hsc*": ColortermDict(data={
            'g': Colorterm(primary="g", secondary="g"),
            'r': Colorterm(primary="r", secondary="r"),
            ...
        }),
        "sdss*": ColortermDict(data={
            'g':    Colorterm(primary="g", secondary="r", c0=-0.00816446, c1=-0.08366937, c2=-0.00726883),
            'r':    Colorterm(primary="r", secondary="i", c0= 0.00231810, c1= 0.01284177, c2=-0.03068248),
            ...
        }),
    })

    This is subclass of Config. That is a bit of a hack to make it easy to store the data
    in an appropriate obs_* package as a config override file. In the long term some other
    means of persistence will be used, at which point the constructor can be made saner.
    """
    data = ConfigDictField(
        doc="Mapping of reference catalog name (or glob) to ColortermDict",
        keytype=str,
        itemtype=ColortermDict,
        default={},
    )

    def getColorterm(self, filterName, photoCatName, doRaise=True):
        """!Get the appropriate Colorterm from the library

        Use dict of color terms in the library that matches the photoCatName.
        If the photoCatName exactly matches an entry in the library, that
        dict is used; otherwise if the photoCatName matches a single glob (shell syntax,
        e.g., "sdss-*" will match "sdss-dr8"), then that is used. If there is no
        exact match and no unique match to the globs, raise an exception.

        @param filterName  name of filter
        @param photoCatName  name of photometric reference catalog from which to retrieve the data.
            This argument is not glob-expanded (but the catalog names in the library are,
            if no exact match is found).
        @param[in] doRaise  if True then raise ColortermNotFoundError if no suitable Colorterm found;
            if False then return a null Colorterm with filterName as the primary and secondary filter
        @return the appropriate Colorterm

        @throw ColortermNotFoundError if no suitable Colorterm found and doRaise true;
        other exceptions may be raised for unexpected errors, regardless of the value of doRaise
        """
        try:
            trueRefCatName = None
            ctDictConfig = self.data.get(photoCatName)
            if ctDictConfig is None:
                # try glob expression
                matchList = [libRefNameGlob for libRefNameGlob in self.data
                             if fnmatch.fnmatch(photoCatName, libRefNameGlob)]
                if len(matchList) == 1:
                    trueRefCatName = matchList[0]
                    ctDictConfig = self.data[trueRefCatName]
                elif len(matchList) > 1:
                    raise ColortermNotFoundError(
                        "Multiple library globs match photoCatName %r: %s" % (photoCatName, matchList))
                else:
                    raise ColortermNotFoundError(
                        "No colorterm dict found with photoCatName %r" % photoCatName)
            ctDict = ctDictConfig.data
            if filterName not in ctDict:
                # Perhaps it's an alias
                try:
                    filterName = Filter(Filter(filterName).getId()).getName()
                except pexExcept.NotFoundError:
                    pass  # this will be handled shortly
                if filterName not in ctDict:
                    errMsg = "No colorterm found for filter %r with photoCatName %r" % (
                        filterName, photoCatName)
                    if trueRefCatName is not None:
                        errMsg += " = catalog %r" % (trueRefCatName,)
                    raise ColortermNotFoundError(errMsg)
            return ctDict[filterName]
        except ColortermNotFoundError:
            if doRaise:
                raise
            else:
                return Colorterm(filterName, filterName)
