#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008-2013 LSST Corporation.
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

import argparse
import collections

import lsst.pex.logging
import lsst.pex.exceptions
import lsst.afw.table
import lsst.afw.image
from lsst.geom import convexHull

from .coaddBase import CoaddDataIdContainer


class PerTractCcdDataIdContainer(CoaddDataIdContainer):
    """A version of lsst.pipe.base.DataIdContainer that combines raw data IDs (defined as whatever we use
    for 'src') with a tract.
    """

    def castDataIds(self, butler):
        """Validate data IDs and cast them to the correct type (modify idList in place).

        @param butler: data butler
        """
        try:
            idKeyTypeDict = butler.getKeys(datasetType="src", level=self.level)
        except KeyError as e:
            raise KeyError("Cannot get keys for datasetType %s at level %s" % ("src", self.level))

        idKeyTypeDict = idKeyTypeDict.copy()
        idKeyTypeDict["tract"] = int

        for dataDict in self.idList:
            for key, strVal in dataDict.iteritems():
                try:
                    keyType = idKeyTypeDict[key]
                except KeyError:
                    validKeys = sorted(idKeyTypeDict.keys())
                    raise KeyError("Unrecognized ID key %r; valid keys are: %s" % (key, validKeys))
                if keyType != str:
                    try:
                        castVal = keyType(strVal)
                    except Exception:
                        raise TypeError("Cannot cast value %r to %s for ID key %r" % (strVal, keyType, key,))
                    dataDict[key] = castVal

    def _addDataRef(self, namespace, dataId, tract):
        """Construct a dataRef based on dataId, but with an added tract key"""
        forcedDataId = dataId.copy()
        forcedDataId['tract'] = tract
        dataRef = namespace.butler.dataRef(datasetType=self.datasetType, dataId=forcedDataId)
        self.refList.append(dataRef)

    def makeDataRefList(self, namespace):
        """Make self.refList from self.idList
        """
        if self.datasetType is None:
            raise RuntimeError("Must call setDatasetType first")
        skymap = None
        log = None
        visitTract = {} # Set of tracts for each visit
        visitRefs = {} # List of data references for each visit
        for dataId in self.idList:
            if "tract" not in dataId:
                # Discover which tracts the data overlaps
                if log is None:
                    log = lsst.pex.logging.Log.getDefaultLog()
                log.info("Reading WCS for components of dataId=%s to determine tracts" % (dict(dataId),))
                if skymap is None:
                    skymap = self.getSkymap(namespace, "deepCoadd")

                for ref in namespace.butler.subset("calexp", dataId=dataId):
                    if not ref.datasetExists("calexp"):
                        continue

                    # XXX fancier mechanism to select an individual exposure than just pulling out "visit"?
                    visit = ref.dataId["visit"]
                    if visit not in visitRefs:
                        visitRefs[visit] = list()
                    visitRefs[visit].append(ref)

                    md = ref.get("calexp_md", immediate=True)
                    wcs = lsst.afw.image.makeWcs(md)
                    box = lsst.afw.geom.Box2D(lsst.afw.geom.Point2D(0, 0),
                                              lsst.afw.geom.Point2D(md.get("NAXIS1"), md.get("NAXIS2")))
                    # Going with just the nearest tract.  Since we're throwing all tracts for the visit
                    # together, this shouldn't be a problem unless the tracts are much smaller than a CCD.
                    tract = skymap.findTract(wcs.pixelToSky(box.getCenter()))
                    if overlapsTract(tract, wcs, box):
                        if visit not in visitTract:
                            visitTract[visit] = set()
                        visitTract[visit].add(tract.getId())
            else:
                tract = dataId.pop("tract")
                # making a DataRef for src fills out any missing keys and allows us to iterate
                for ref in namespace.butler.subset("src", dataId=dataId):
                    self._addDataRef(namespace, ref.dataId, tract)

        # Ensure all components of a visit are kept together by putting them all in the same set of tracts
        for visit, tractSet in visitTract.iteritems():
            for ref in visitRefs[visit]:
                for tract in tractSet:
                    self._addDataRef(namespace, ref.dataId, tract)
        if visitTract:
            tractCounter = collections.Counter()
            for tractSet in visitTract.itervalues():
                tractCounter.update(tractSet)
            log.info("Number of visits for each tract: %s" % (dict(tractCounter),))


def overlapsTract(tract, imageWcs, imageBox):
    """Return whether the image (specified by Wcs and bounding box) overlaps the tract

    @param tract: TractInfo specifying a tract
    @param imageWcs: Wcs for image
    @param imageBox: Bounding box for image
    @return bool
    """
    tractWcs = tract.getWcs()
    tractCorners = [tractWcs.pixelToSky(lsst.afw.geom.Point2D(coord)).getVector() for
                    coord in tract.getBBox().getCorners()]
    tractPoly = convexHull(tractCorners)

    try:
        imageCorners = [imageWcs.pixelToSky(lsst.afw.geom.Point2D(pix)) for pix in imageBox.getCorners()]
    except lsst.pex.exceptions.LsstCppException, e:
        # Protecting ourselves from awful Wcs solutions in input images
        if (not isinstance(e.message, lsst.pex.exceptions.DomainErrorException) and
            not isinstance(e.message, lsst.pex.exceptions.RuntimeErrorException)):
            raise
        return False

    imagePoly = convexHull([coord.getVector() for coord in imageCorners])
    if imagePoly is None:
        return False
    return tractPoly.intersects(imagePoly) # "intersects" also covers "contains" or "is contained by"
