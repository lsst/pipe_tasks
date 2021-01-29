#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008-2016 LSST/AURA
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import numpy as np
from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Task
from lsst.geom import Box2D


def getPatchInner(sources, patchInfo):
    """Set a flag for each source if it is in the innerBBox of a patch

    Parameters
    ----------
    sources : `lsst.afw.table.SourceCatalog`
        A sourceCatalog with pre-calculated centroids.
        If `key` is not `None` then the results are written into
        `sources[key]`.
    patchInfo : `lsst.skymap.PatchInfo`
        Patch object

    Returns
    --------
    isPatchInner: boolean array
        True for each source that has a centroid
        in the inner region of a patch.
    """
    # Extract the centroid position for all the sources
    centroidFlag = sources[sources.getCentroidSlot().getFlagKey()]
    centroidKey = sources.getCentroidSlot().getMeasKey()
    x = sources[centroidKey.getX()]
    y = sources[centroidKey.getY()]

    # set inner flags for each source and set primary flags for sources with no children
    # (or all sources if deblend info not available)
    innerFloatBBox = Box2D(patchInfo.getInnerBBox())
    xmin = innerFloatBBox.getMinX()
    xmax = innerFloatBBox.getMaxX()
    ymin = innerFloatBBox.getMinY()
    ymax = innerFloatBBox.getMaxY()
    inInner = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    # When the centroider fails, we can still fall back to the peak, but we don't trust
    # that quite as much - so we use a slightly smaller box for the patch comparison.
    # That's trickier for the tract comparison, so we just use the peak without extra
    # care there.
    shrunkInnerFloatBBox = Box2D(innerFloatBBox)
    shrunkInnerFloatBBox.grow(-1)
    xmin = shrunkInnerFloatBBox.getMinX()
    xmax = shrunkInnerFloatBBox.getMaxX()
    ymin = shrunkInnerFloatBBox.getMinY()
    ymax = shrunkInnerFloatBBox.getMaxY()
    inShrunkInner = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    # Flag sources contained in the inner region of a patch
    isPatchInner = (centroidFlag & inShrunkInner) | (~centroidFlag & inInner)
    return isPatchInner


def getTractInner(sources, tractInfo, skyMap):
    """Set a flag for each source if it is contained in a given tract

    Parameters
    ----------
    sources : `lsst.afw.table.SourceCatalog`
        A sourceCatalog with pre-calculated centroids.
        If `key` is not `None` then the results are written into
        `sources[key]`.
    tractInfo : `lsst.skymap.TractInfo`
        Tract object
    skyMap : `lsst.skymap.BaseSkyMap`
        Sky tessellation object

    Returns
    -------
    isTractInner: `bool` array
        True for each source that has a centroid in the same
        tract as `tractInfo`.
    """
    # Mark sources in the inner region of a tract
    tractId = tractInfo.getId()
    isTractInner = np.array([skyMap.findTract(s.getCoord()).getId() == tractId for s in sources])
    return isTractInner


def getPseudoSources(sources, pseudoFilterList, schema, log):
    """Get a flag that marks pseudo sources

    Some categories of sources, for example sky objects,
    are not really detected sources and should not be considered primary
    sources.

    Parameters
    ----------
    sources : `lsst.afw.table.SourceCatalog`
        A sourceCatalog.
    pseudoFilterList: `list` of `str`
        Names of filters which should never be primary

    Returns
    -------
    isPseudo: `bool` array
        True for each source that is a pseudo source.
        Note: to remove pseudo sources use `~isPseudo`.
    """
    # Filter out sources that should never be primary
    isPseudo = np.zeros(len(sources), dtype=bool)
    for filt in pseudoFilterList:
        try:
            pseudoFilterKey = schema.find("merge_peak_%s" % filt).getKey()
        except Exception:
            log.warn("merge_peak is not set for pseudo-filter %s" % filt)
        isPseudo |= sources[pseudoFilterKey]
    return isPseudo


def getDeblendPrimaryFlags(sources, nChildKey=None):
    """Get both types of primary sources flagged by the deblender

    scarlet is different than meas_deblender in that it is not
    (necessarily) flux conserving. For consistency in scarlet,
    all of the parents with only a single child (isolated sources)
    need to be deblended. This creates a question: which type
    of isolated source should we make measurements on, the
    undeblended "parent" or the deblended child?
    For that reason we distinguish between a SimpleLeaf,
    which is a source that has no children and uses the
    isolated parents, and a ModelLeaf, which uses
    the scarlet models for both isolated and blended sources.
    In the case of meas_deblender, a ModelLeaf is the same thing
    as a blended source, and a SimpleModel has the same definition.

    Parameters
    ----------
    sources : `lsst.afw.table.SourceCatalog`
        A sourceCatalog that has already been deblended using
        either meas_extensions_scarlet or meas_deblender.
    nChildKey: `str`
        Name of the column in `sources` that contains the
        number of deblended children for each source.

    Returns
    -------
    isSimpleLeaf: `bool` array
        True for each source that is a `SimpleLeaf` as defined above.
    isModelLeaf: `bool` array
        True for each sources that is a `ModelLeaf` as defined above.
    """
    if nChildKey is None:
        nChildKey = "deblend_nChild"
    nChild = sources[nChildKey]
    parent = sources["parent"]
    isModelLeaf = (parent != 0) & (nChild == 0)

    if "deblend_scarletFlux" in sources.schema:
        parentNChild = sources["deblend_parentNChild"]
        isBlend = isModelLeaf & (parentNChild > 1)
        isSimpleLeaf = ((parent == 0) & (nChild == 1)) | isBlend
    else:
        isSimpleLeaf = nChild == 0
    return isSimpleLeaf, isModelLeaf


class SetPrimaryFlagsConfig(Config):
    nChildKeyName = Field(dtype=str, default="deblend_nChild",
                          doc="Name of field in schema with number of deblended children")
    pseudoFilterList = ListField(dtype=str, default=['sky'],
                                 doc="Names of filters which should never be primary")


class SetPrimaryFlagsTask(Task):
    """Add isPrimaryKey to a given schema.

    Parameters
    ----------
    schema : `lsst.afw.table.Schema`
        The input schema.
    isSingleFrame : `bool`
        Flag specifying if task is operating with single frame imaging.
    includeDeblend : `bool`
        Include deblend information in isPrimary and
        add isSimpleLeaf and isModelLeaf fields?
    kwargs :
        Keyword arguments passed to the task.
    """

    ConfigClass = SetPrimaryFlagsConfig

    def __init__(self, schema, isSingleFrame=False, includeDeblend=True, **kwargs):
        Task.__init__(self, **kwargs)
        self.schema = schema
        self.isSingleFrame = isSingleFrame
        self.includeDeblend = includeDeblend
        if not self.isSingleFrame:
            primaryDoc = ("true if source has no children and is in the inner region of a coadd patch "
                          "and is in the inner region of a coadd tract "
                          "and is not \"detected\" in a pseudo-filter (see config.pseudoFilterList)")
            self.isPatchInnerKey = self.schema.addField(
                "detect_isPatchInner", type="Flag",
                doc="true if source is in the inner region of a coadd patch",
            )
            self.isTractInnerKey = self.schema.addField(
                "detect_isTractInner", type="Flag",
                doc="true if source is in the inner region of a coadd tract",
            )
        else:
            primaryDoc = "true if source has no children and is not a sky source"
        self.isPrimaryKey = self.schema.addField(
            "detect_isPrimary", type="Flag",
            doc=primaryDoc,
        )

        if self.includeDeblend:
            self.isSimpleLeafKey = self.schema.addField(
                "detect_isSimpleLeaf", type="Flag",
                doc=primaryDoc + " and is either an unblended isolated source or a"
                                 "deblended child from a parent with 'deblend_nChild' > 1")
            self.isModelLeafKey = self.schema.addField(
                "detect_isModelLeaf", type="Flag",
                doc=primaryDoc + " and is a deblended child")

    def run(self, sources, skyMap=None, tractInfo=None, patchInfo=None):
        """Set is-patch-inner, is-tract-inner and is-primary flags on sources.
        For coadded imaging, the is-primary flag returns True when an object
        has no children, is in the inner region of a coadd patch, is in the
        inner region of a coadd trach, and is not detected in a pseudo-filter
        (e.g., a sky_object).
        For single frame imaging, the is-primary flag returns True when a
        source has no children and is not a sky source.

        Parameters
        ----------
        sources : `lsst.afw.table.SourceCatalog`
            A sourceTable. Reads in centroid fields and an nChild field.
            Writes is-patch-inner, is-tract-inner, and is-primary flags.
        skyMap : `lsst.skymap.BaseSkyMap`
            Sky tessellation object
        tractInfo : `lsst.skymap.TractInfo`
            Tract object
        patchInfo : `lsst.skymap.PatchInfo`
            Patch object
        """
        # Mark the sources inside of a tract/patch
        if not self.isSingleFrame:
            isPatchInner = getPatchInner(sources, patchInfo)
            isTractInner = getTractInner(sources, tractInfo, skyMap)
            isPseudo = getPseudoSources(sources, self.config.pseudoFilterList, self.schema, self.log)
            isPrimary = isTractInner & isPatchInner & ~isPseudo

            sources[self.isPatchInnerKey] = isPatchInner
            sources[self.isTractInnerKey] = isTractInner
        else:
            # Mark all of the sky sources in SingleFrame images (if they were added)
            if "sky_source" in sources.schema:
                isSky = sources["sky_source"]
            else:
                isSky = np.zeros(len(sources), dtype=bool)
            isPrimary = ~isSky

        if self.includeDeblend:
            isSimpleLeaf, isModelLeaf = getDeblendPrimaryFlags(sources, self.config.nChildKeyName)
            sources[self.isSimpleLeafKey] = isSimpleLeaf & isPrimary
            sources[self.isModelLeafKey] = isModelLeaf & isPrimary
            isPrimary = isSimpleLeaf & isPrimary

        sources[self.isPrimaryKey] = isPrimary
