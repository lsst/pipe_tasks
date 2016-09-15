from eups import Eups
from fnmatch import fnmatch

from lsst.pipe.base.argumentParser import setDottedAttr

__all__ = ["getAndVersion", "setAstrometryConfigFromEups", "setPhotocalConfigFromEups", "setConfigFromEups", ]


def getEups():
    """Return a cached eups instance"""
    try:
        return getEups._eups
    except AttributeError:
        getEups._eups = Eups()
        return getEups._eups


def getAndVersion():
    """Return the version of astrometry_net_data in use"""
    return getEups().findSetupVersion("astrometry_net_data")[0]


def setAstrometryConfigFromEups(config, menu):
    """Set the astrometry configuration according to the astrometry_net_data being used

    The menu is a dict mapping the astrometry_net_data version to a dict of configuration
    values to apply.  The menu key may also be a glob.  For example:
    menu = { "ps1*": {}, # No changes
             "ps1-without-y": { "solver.filterMap": {"y": "z"} }, # No y-band in this specific version
             "sdss*": { "solver.filterMap": {"y": "z"} }, # No y-band, use z instead
             "2mass*": { "solver.filterMap": {"y": "J"} }, # No y-band, use J instead
           }
    """
    version = getAndVersion()

    if version in menu:
        selected = menu[version]
    else:
        # Treat keys in menu as glob; see if any match
        matchList = [key for key in menu if fnmatch(version, key)]
        if len(matchList) > 1:
            raise RuntimeError("Multiple menu keys match astrometry_net_data version %s: %s" %
                               (version, matchList))
        if len(matchList) == 0:
            raise RuntimeError("No menu key matches astrometry_net_data version %s" % version)
        selected = menu[matchList.pop()]
    for name, value in selected.iteritems():
        setDottedAttr(config, name, value)


def setPhotocalConfigFromEups(config):
    """Set the photocal configuration according to the astrometry_net_data being used"""
    config.photoCatName = getAndVersion()


def setConfigFromEups(photocalConfig=None, astrometryConfig=None, astrometryMenu=None):
    """Set the astrometry and photocal configuration according to the astrometry_net_data being used

    The 'astrometryMenu' is as defined for the 'menu' parameter for 'setAstrometryConfigFromEups'.
    """
    if photocalConfig:
        setPhotocalConfigFromEups(photocalConfig)
    if astrometryConfig:
        if astrometryMenu is None:
            raise RuntimeError("No astrometryMenu provided for astrometryConfig")
        setAstrometryConfigFromEups(astrometryConfig, astrometryMenu)
