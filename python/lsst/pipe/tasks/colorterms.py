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
import numpy as np

class Colorterm(object):
    """!A class to describe colour terms between photometric bands
    """
    _colorterms = {}                    # cached dictionary of dictionaries of Colorterms for devices
    _activeColorterms = None            # The set of Colorterms that are currently in use

    def __init__(self, primary, secondary, c0, c1=0.0, c2=0.0):
        """!Construct a Colorterm

        The transformed magnitude p' is given by
        p' = primary + c0 + c1*(primary - secondary) + c2*(primary - secondary)**2
        """
        self.primary = primary
        self.secondary = secondary
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2

    def __str__(self):
        return "%s %s [%g %g %g]" % (self.primary, self.secondary, self.c0, self.c1, self.c2)

    @staticmethod
    def setColorterms(colorterms, device=None):
        """!Replace or update the cached Colorterms dict

        @param[in,out] colorterms  a dict of device: Colorterm
        @param[in] device  device name, or None;
            if device is None then colorterms replaces the internal catched dict,
            else the internal cached dict is updated with device: colorterms[device]
        """
        if device:
            Colorterm._colorterms[device] = colorterms[device]
        else:
            Colorterm._colorterms = colorterms

        Colorterm.setActiveDevice(device, allowRaise=False)

    @staticmethod
    def setActiveDevice(device, allowRaise=True):
        """!Set or clear the default colour terms

        @param[in] device  device name, or None to clear
        @param[in] allowRaise  controls handling an unknown, non-None device:
            if true raise RuntimeError, else clear the default colorterms
        """
        if device is None:
            Colorterm._activeColorterms = None

        else:
            try:
                Colorterm._activeColorterms = Colorterm._colorterms[device]
            except KeyError:
                if allowRaise:
                    raise RuntimeError("No colour terms are available for %s" % device)

                Colorterm._activeColorterms = None

    @staticmethod
    def getColorterm(band):
        """!Return the Colorterm for the specified band (or None if unknown)
        """
        return Colorterm._activeColorterms.get(band) if Colorterm._activeColorterms else None

    @staticmethod
    def transformSource(band, source, reverse=False, colorterms=None):
        """!Transform the magnitudes in *source* to the specified *band* and return it.

        The *source* must support a get(band) (e.g. source.get("r")) method, as do afw::Source and dicts.
        Use the colorterms (or the cached set if colorterms is None); if no set is available,
        return the *band* flux.
        If reverse is True, return the inverse transformed magnitude

        @warning reverse is not yet implemented
        """
        if not colorterms:
            colorterms = Colorterm._activeColorterms

        if not colorterms:
            return source.get(band)

        ct = colorterms[band]
        
        return Colorterm.transformMags(band, source.get(ct.primary), source.get(ct.secondary),
                                       reverse, colorterms)

    @staticmethod
    def transformMags(band, primary, secondary, reverse=False, colorterms=None):
        """!Transform the magnitudes *primary* and *secondary* to the specified *band* and return it.

        Use the colorterms (or the cached set if colorterms is None); if no set is available,
        return the *band* flux.
        If reverse is True, return the inverse transformed magnitude

        @warning reverse is not yet implemented
        """
        if not colorterms:
            colorterms = Colorterm._activeColorterms

        if not colorterms:
            return primary

        ct = colorterms[band]
        
        p = primary
        s = secondary

        if reverse:
            raise NotImplemented("reverse photometric transformations are not implemented")
        else:
            return p + ct.c0 + (p - s)*(ct.c1 + (p - s)*ct.c2)

    @staticmethod
    def propagateFluxErrors(band, primaryFluxErr, secondaryFluxErr, reverse=False, colorterms=None):
        """!Transform flux errors
        """
        if not colorterms:
            colorterms = Colorterm._activeColorterms

        if not colorterms:
            return primaryFluxErr

        ct = colorterms[band]

        return np.hypot((1 + ct.c1)*primaryFluxErr, ct.c1*secondaryFluxErr)
        
