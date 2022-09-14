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

__all__ = ["ColortermNotFoundError", "Colorterm", "ColortermDict", "ColortermLibrary"]

import fnmatch
import warnings

import numpy as np
import astropy.units as u

from lsst.afw.image import abMagErrFromFluxErr
from lsst.pex.config import Config, Field, ConfigDictField


class ColortermNotFoundError(LookupError):
    """Exception class indicating we couldn't find a colorterm
    """
    pass


class Colorterm(Config):
    """Colorterm correction for one pair of filters

    Notes
    -----
    The transformed magnitude p' is given by:
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

    def getCorrectedMagnitudes(self, refCat, filterName="deprecatedArgument"):
        """Return the colorterm corrected magnitudes for a given filter.

        Parameters
        ----------
        refCat : `lsst.afw.table.SimpleCatalog`
            The reference catalog to apply color corrections to.
        filterName : `str`, deprecated
            The camera filter to correct the reference catalog into.
            The ``filterName`` argument is unused and will be removed in v23.

        Returns
        -------
        refMag : `np.ndarray`
            The corrected AB magnitudes.
        refMagErr : `np.ndarray`
            The corrected AB magnitude errors.

        Raises
        ------
        KeyError
            Raised if the reference catalog does not have a flux uncertainty
            for that filter.

        Notes
        -----
        WARNING: I do not know that we can trust the propagation of magnitude
        errors returned by this method. They need more thorough tests.
        """
        if filterName != "deprecatedArgument":
            msg = "Colorterm.getCorrectedMagnitudes() `filterName` arg is unused and will be removed in v23."
            warnings.warn(msg, category=FutureWarning)

        def getFluxes(fluxField):
            """Get the flux and fluxErr of this field from refCat.

            Parameters
            ----------
            fluxField : `str`
                Name of the source flux field to use.

            Returns
            -------
            refFlux : `Unknown`
            refFluxErr : `Unknown`

            Raises
            ------
            KeyError
                Raised if reference catalog does not have flux uncertainties for the given flux field.
            """
            fluxKey = refCat.schema.find(fluxField).key
            refFlux = refCat[fluxKey]
            try:
                fluxErrKey = refCat.schema.find(fluxField + "Err").key
                refFluxErr = refCat[fluxErrKey]
            except KeyError as e:
                raise KeyError("Reference catalog does not have flux uncertainties for %s" % fluxField) from e

            return refFlux, refFluxErr

        primaryFlux, primaryErr = getFluxes(self.primary + "_flux")
        secondaryFlux, secondaryErr = getFluxes(self.secondary + "_flux")

        primaryMag = u.Quantity(primaryFlux, u.nJy).to_value(u.ABmag)
        secondaryMag = u.Quantity(secondaryFlux, u.nJy).to_value(u.ABmag)

        refMag = self.transformMags(primaryMag, secondaryMag)
        refFluxErrArr = self.propagateFluxErrors(primaryErr, secondaryErr)

        # HACK convert to Jy until we have a replacement for this (DM-16903)
        refMagErr = abMagErrFromFluxErr(refFluxErrArr*1e-9, primaryFlux*1e-9)

        return refMag, refMagErr

    def transformSource(self, source):
        """Transform the brightness of a source

        Parameters
        ----------
        source : `Unknown`
            Source whose brightness is to be converted; must support get(filterName)
            (e.g. source.get("r")) method, as do afw::table::Source and dicts.

        Returns
        -------
        transformed : `float`
            The transformed source magnitude.
        """
        return self.transformMags(source.get(self.primary), source.get(self.secondary))

    def transformMags(self, primary, secondary):
        """Transform brightness

        Parameters
        ----------
        primary : `float`
            Brightness in primary filter (magnitude).
        secondary : `float`
            Brightness in secondary filter (magnitude).

        Returns
        -------
        transformed : `float`
            The transformed brightness (as a magnitude).
        """
        color = primary - secondary
        return primary + self.c0 + color*(self.c1 + color*self.c2)

    def propagateFluxErrors(self, primaryFluxErr, secondaryFluxErr):
        return np.hypot((1 + self.c1)*primaryFluxErr, self.c1*secondaryFluxErr)


class ColortermDict(Config):
    """A mapping of physical filter label to Colorterm

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
    """A mapping of photometric reference catalog name or glob to ColortermDict

    Notes
    -----
    This allows photometric calibration using a variety of reference catalogs.

    To construct a ColortermLibrary, use keyword arguments:
    ColortermLibrary(data=dataDict)
    where dataDict is a Python dict of catalog_name_or_glob: ColortermDict

    Examples
    --------

    .. code-block:: none

        ColortermLibrary(data = {
            "hsc": ColortermDict(data={
                'g': Colorterm(primary="g", secondary="g"),
                'r': Colorterm(primary="r", secondary="r"),
                ...
            }),
            "sdss": ColortermDict(data={
                'g':    Colorterm(primary="g",
                                  secondary="r",
                                  c0=-0.00816446,
                                  c1=-0.08366937,
                                  c2=-0.00726883),
                'r':    Colorterm(primary="r",
                                  secondary="i",
                                  c0= 0.00231810,
                                  c1= 0.01284177,
                                  c2=-0.03068248),
                ...
            }),
        })

    This is subclass of Config. That is a bit of a hack to make it easy to store the data
    in an appropriate obs package as a config override file. In the long term some other
    means of persistence will be used, at which point the constructor can be made saner.
    """
    data = ConfigDictField(
        doc="Mapping of reference catalog name (or glob) to ColortermDict",
        keytype=str,
        itemtype=ColortermDict,
        default={},
    )

    def getColorterm(self, physicalFilter, photoCatName, doRaise=True):
        """Get the appropriate Colorterm from the library

        Use dict of color terms in the library that matches the photoCatName.
        If the photoCatName exactly matches an entry in the library, that
        dict is used; otherwise if the photoCatName matches a single glob (shell syntax,
        e.g., "sdss-*" will match "sdss-dr8"), then that is used. If there is no
        exact match and no unique match to the globs, raise an exception.

        Parameters
        ----------
        physicalFilter : `str`
            Label of physical filter to correct to.
        photoCatName : `str`
            Name of photometric reference catalog from which to retrieve the data.
            This argument is not glob-expanded (but the catalog names in the library are,
            if no exact match is found).
        doRaise : `bool`
            If True then raise ColortermNotFoundError if no suitable Colorterm found;
            If False then return a null Colorterm with physicalFilter as the primary and secondary filter.

        Returns
        -------
        ctDict : `Unknown`
            The appropriate Colorterm.

        Raises
        ------
        ColortermNotFoundError
            If no suitable Colorterm found and doRaise true;
            other exceptions may be raised for unexpected errors, regardless of the value of doRaise.
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
            if physicalFilter not in ctDict:
                errMsg = "No colorterm found for filter %r with photoCatName %r" % (
                    physicalFilter, photoCatName)
                if trueRefCatName is not None:
                    errMsg += " = catalog %r" % (trueRefCatName,)
                raise ColortermNotFoundError(errMsg)
            return ctDict[physicalFilter]
        except ColortermNotFoundError:
            if doRaise:
                raise
            else:
                return Colorterm(physicalFilter, physicalFilter)
