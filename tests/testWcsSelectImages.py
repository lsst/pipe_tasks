from builtins import map
from builtins import object
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

import unittest

import lsst.utils.tests
import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
from lsst.pipe.tasks.selectImages import WcsSelectImagesTask, SelectStruct
from lsst.pipe.tasks.coaddBase import CoaddBaseTask


class KeyValue(object):

    """Mixin to provide __getitem__ of key/value pair"""

    def __init__(self, key, value):
        self._key = key
        self._value = value

    def __getitem__(self, key):
        if key != self._key:
            raise KeyError("Unrecognised key in %s: %s vs %s" % (self.__class__.__name__, key, self._key))
        return self._value


class DummyPatch(object):

    """Quacks like a lsst.skymap.PatchInfo"""

    def __init__(self, xy0, dims):
        self._outerBBox = afwGeom.Box2I(xy0, dims)

    def getOuterBBox(self):
        return self._outerBBox


class DummyTract(KeyValue):

    """Quacks like a lsst.skymap.TractInfo"""

    def __init__(self, patchId, patch, wcs):
        super(DummyTract, self).__init__(patchId, patch)
        self._wcs = wcs

    def getPatchInfo(self, patchId):
        return self[patchId]

    def getWcs(self):
        return self._wcs


class DummySkyMap(KeyValue):

    """Quacks like a lsst.skymap.BaseSkyMap"""

    def __init__(self, tractId, tract):
        super(DummySkyMap, self).__init__(tractId, tract)


class DummyDataRef(object):

    """Quacks like a lsst.daf.persistence.ButlerDataRef"""

    def __init__(self, dataId, **data):
        self.dataId = dataId
        self._data = data

    def get(self, dataType):
        return self._data[dataType]

# Common defaults for createPatch and createImage
CENTER = afwCoord.Coord(0*afwGeom.degrees, 90*afwGeom.degrees)
ROTATEAXIS = afwCoord.Coord(0*afwGeom.degrees, 0*afwGeom.degrees)
DIMS = afwGeom.Extent2I(3600, 3600)
SCALE = 0.5*afwGeom.arcseconds


def createPatch(
    tractId=1, patchId=(2, 3),  # Tract and patch identifier, for dataId
    dims=DIMS,                # Patch dimensions (Extent2I)
    xy0=afwGeom.Point2I(1234, 5678),  # Patch xy0 (Point2I)
    center=CENTER,                  # Celestial coordinates of center (Coord)
    scale=SCALE                     # Pixel scale (Angle)
):
    crpix = afwGeom.Point2D(xy0) + afwGeom.Extent2D(dims)*0.5
    wcs = afwImage.makeWcs(center, crpix, scale.asDegrees(), 0.0, 0.0, scale.asDegrees())
    patch = DummyPatch(xy0, dims)
    tract = DummyTract(patchId, patch, wcs)
    skymap = DummySkyMap(tractId, tract)
    dataRef = DummyDataRef({'tract': tractId, 'patch': ",".join(map(str, patchId))}, deepCoadd_skyMap=skymap)
    return dataRef


def createImage(
    dataId={"name": "foobar"},          # Data identifier
    center=CENTER,                      # Celestial coordinates of center (Coord)
    rotateAxis=ROTATEAXIS,              # Rotation axis (Angle)
    rotateAngle=0*afwGeom.degrees,      # Rotation angle/distance to move (Angle)
    dims=DIMS,                          # Image dimensions (Extent2I)
    scale=SCALE                         # Pixel scale (Angle)
):
    crpix = afwGeom.Point2D(afwGeom.Extent2D(dims)*0.5)
    center = center.clone()  # Ensure user doesn't need it, because we're mangling it
    center.rotate(rotateAxis, rotateAngle)
    wcs = afwImage.makeWcs(center, crpix, scale.asDegrees(), 0.0, 0.0, scale.asDegrees())
    return SelectStruct(DummyDataRef(dataId), wcs, (dims[0], dims[1]))


class WcsSelectImagesTestCase(unittest.TestCase):

    def check(self, patchRef, selectData, doesOverlap):
        config = CoaddBaseTask.ConfigClass()
        config.select.retarget(WcsSelectImagesTask)
        task = CoaddBaseTask(config=config, name="CoaddBase")
        dataRefList = task.selectExposures(patchRef, selectDataList=[selectData])
        numExpected = 1 if doesOverlap else 0
        self.assertEqual(len(dataRefList), numExpected)

    def testIdentical(self):
        self.check(createPatch(), createImage(), True)

    def testImageContains(self):
        self.check(createPatch(), createImage(scale=2*SCALE), True)

    def testImageContained(self):
        self.check(createPatch(), createImage(scale=0.5*SCALE), True)

    def testDisjoint(self):
        self.check(createPatch(), createImage(center=afwCoord.Coord(0*afwGeom.degrees, -90*afwGeom.degrees)),
                   False)

    def testIntersect(self):
        self.check(createPatch(), createImage(rotateAngle=0.5*afwGeom.Extent2D(DIMS).computeNorm()*SCALE),
                   True)


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
