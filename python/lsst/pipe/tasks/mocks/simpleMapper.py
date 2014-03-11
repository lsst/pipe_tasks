# 
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011, 2012 LSST Corporation.
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

"""Mapper and cameraGeom definition for extremely simple mock data.

SimpleMapper inherits directly from Mapper, not CameraMapper.  This means
we can avoid any problems with paf files at the expense of reimplementing
some parts of CameraMapper here.  Jim is not sure this was the best
possible approach, but it gave him an opportunity to play around with
prototyping a future paf-free mapper class, and it does everything it
needs to do right now.
"""

import os
import shutil
import re

import lsst.daf.persistence
import lsst.afw.cameraGeom
from lsst.afw.cameraGeom.testUtils import DetectorWrapper

__all__ = ("SimpleMapper", "makeSimpleCamera", "makeDataRepo")

class PersistenceType(object):
    """Base class of a hierarchy used by SimpleMapper to defined different kinds of types of objects
    to persist.

    PersistenceType objects are never instantiated; only the type objects are used (we needed a
    simple singleton struct that could be inherited, which is exactly what a Python type is).
    """
    python = None
    cpp = "ignored"
    storage = None
    ext = ""
    suffixes = ()

    @classmethod
    def makeButlerLocation(cls, path, dataId, suffix=None):
        """Method called by SimpleMapping to implement a map_ method."""
        return lsst.daf.persistence.ButlerLocation(cls.python, cls.cpp, cls.storage, [path], dataId)

class BypassPersistenceType(PersistenceType):
    """Persistence type for things that don't actually use daf_persistence.
    """

    python = "lsst.daf.base.PropertySet"  # something to import even when we don't need to

    @classmethod
    def makeButlerLocation(cls, path, dataId, suffix=None):
        """Method called by SimpleMapping to implement a map_ method; overridden to not use the path."""
        return lsst.daf.persistence.ButlerLocation(cls.python, cls.cpp, cls.storage, [], dataId)

class ExposurePersistenceType(PersistenceType):
    """Persistence type of Exposure images.
    """

    python = "lsst.afw.image.ExposureF"
    cpp = "ExposureF"
    storage = "FitsStorage"
    ext = ".fits"
    suffixes = ("_sub",)

    @classmethod
    def makeButlerLocation(cls, path, dataId, suffix=None):
        """Method called by SimpleMapping to implement a map_ method; overridden to support subimages."""
        if suffix is None:
            loc = super(ExposurePersistenceType, cls).makeButlerLocation(path, dataId, suffix=None)
        elif suffix == "_sub":
            subId = dataId.copy()
            bbox = subId.pop('bbox')
            loc = super(ExposurePersistenceType, cls).makeButlerLocation(path, subId, suffix=None)
            loc.additionalData.set('llcX', bbox.getMinX())
            loc.additionalData.set('llcY', bbox.getMinY())
            loc.additionalData.set('width', bbox.getWidth())
            loc.additionalData.set('height', bbox.getHeight())
            if 'imageOrigin' in dataId:
                loc.additionalData.set('imageOrigin',
                                       dataId['imageOrigin'])
        return loc

class SkyMapPersistenceType(PersistenceType):
    python = "lsst.skymap.BaseSkyMap"
    storage = "PickleStorage"
    ext = ".pickle"

class CatalogPersistenceType(PersistenceType):
    python = "lsst.afw.table.BaseCatalog"
    cpp = "BaseCatalog"
    storage = "FitsCatalogStorage"
    ext = ".fits"

class SimpleCatalogPersistenceType(CatalogPersistenceType):
    python = "lsst.afw.table.SimpleCatalog"
    cpp = "SimpleCatalog"

class SourceCatalogPersistenceType(SimpleCatalogPersistenceType):
    python = "lsst.afw.table.SimpleCatalog"
    cpp = "SimpleCatalog"

class ExposureCatalogPersistenceType(CatalogPersistenceType):
    python = "lsst.afw.table.ExposureCatalog"
    cpp = "ExposureCatalog"

class SimpleMapping(object):
    """Mapping object used to implement SimpleMapper, similar in intent to lsst.daf.peristence.Mapping.
    """

    template = None
    keys = {}

    def __init__(self, persistence, template=None, keys=None):
        self.persistence = persistence
        if template is not None:
            self.template = template
        if keys is not None:
            self.keys = keys

    def map(self, dataset, root, dataId, suffix=None):
        if self.template is not None:
            path = os.path.join(root, self.template.format(dataset=dataset, ext=self.persistence.ext,
                                                           **dataId))
        else:
            path = None
        return self.persistence.makeButlerLocation(path, dataId, suffix=suffix)

class RawMapping(SimpleMapping):
    """Mapping for dataset types that are organized the same way as raw data (i.e. by CCD)."""

    template = "{dataset}-{visit:04d}-{ccd:01d}{ext}"
    keys = dict(visit=int, ccd=int)

    def query(self, index, level, format, dataId):
        results = []
        visit = dataId.get('visit', None)
        dictList = self.index[level][visit]
        ccd = dataId.get('ccd', None)
        for d in dictList.values():
            if ccd is not None and d['ccd'] != ccd:
                continue
            results.append(tuple(d[f] for f in format))
        return results

class SkyMapping(SimpleMapping):
    """Mapping for dataset types that are organized according to a SkyMap subdivision of the sky."""

    template = "{dataset}-{filter}-{tract:02d}-{patch}{ext}"
    keys = dict(filter=str, tract=int, patch=str)

class TempExpMapping(SimpleMapping):
    """Mapping for CoaddTempExp datasets."""
    
    template = "{dataset}-{tract:02d}-{patch}-{visit:04d}{ext}"
    keys = dict(tract=int, patch=str, visit=int)

class MapperMeta(type):
    """Metaclass for SimpleMapper that creates map_ and query_ methods for everything found in the
    'mappings' class variable.
    """

    @staticmethod
    def _makeMapClosure(dataset, mapping, suffix=None):
        def mapClosure(self, dataId, write=False):
            return mapping.map(dataset, self.root, dataId, suffix=suffix)
        return mapClosure

    @staticmethod
    def _makeQueryClosure(dataset, mapping):
        def queryClosure(self, level, format, dataId):
            return mapping.query(self.index, level, format, dataId)
        return queryClosure

    def __init__(cls, name, bases, dict_):
        type.__init__(cls, name, bases, dict_)
        cls.keyDict = dict()
        for dataset, mapping in cls.mappings.iteritems():
            setattr(cls, "map_" + dataset, MapperMeta._makeMapClosure(dataset, mapping, suffix=None))
            for suffix in mapping.persistence.suffixes:
                setattr(cls, "map_" + dataset + suffix,
                        MapperMeta._makeMapClosure(dataset, mapping, suffix=suffix))
            if hasattr(mapping, "query"):
                setattr(cls, "query_" + dataset, MapperMeta._makeQueryClosure(dataset, mapping))
        cls.keyDict.update(mapping.keys)

class SimpleMapper(lsst.daf.persistence.Mapper):
    """
    An extremely simple mapper for an imaginary camera for use in integration tests.
    
    As SimpleMapper does not inherit from daf.butlerUtils.CameraMapper, it does not
    use a policy file to set mappings or a registry; all the information is here
    (in the map_* and query_* methods).

    The imaginary camera's raw data format has only 'visit' and 'ccd' keys, with
    two CCDs per visit (by default).
    """

    __metaclass__ = MapperMeta

    mappings = dict(
        calexp = RawMapping(ExposurePersistenceType),
        truth = SimpleMapping(SimpleCatalogPersistenceType, template="{dataset}-{tract:02d}{ext}",
                              keys={"tract":int}),
        simsrc = RawMapping(SimpleCatalogPersistenceType, template="{dataset}-{tract:02d}{ext}",
                            keys={"tract":int}),
        observations = SimpleMapping(ExposureCatalogPersistenceType, template="{dataset}-{tract:02d}{ext}",
                                     keys={"tract":int}),
        ccdExposureId = RawMapping(BypassPersistenceType),
        ccdExposureId_bits = SimpleMapping(BypassPersistenceType),
        deepCoadd_skyMap = SimpleMapping(SkyMapPersistenceType, template="{dataset}{ext}", keys={}),
        deepCoadd = SkyMapping(ExposurePersistenceType),
        deepCoadd_mock = SkyMapping(ExposurePersistenceType),
        deepCoadd_tempExp = TempExpMapping(ExposurePersistenceType),
        deepCoadd_tempExp_mock = TempExpMapping(ExposurePersistenceType),
    )

    levels = dict(
        visit = ['ccd'],
        ccd = [],
    )

    def __init__(self, root):
        super(SimpleMapper, self).__init__()
        self.root = root
        self.camera = makeSimpleCamera(nX=1, nY=2, sizeX=400, sizeY=200, gapX=2, gapY=2)
        self.update()

    def getDefaultLevel(self): return "ccd"

    def getKeys(self, datasetType, level):
        if datasetType is None:
            keyDict = self.keyDict
        else:
            keyDict = self.mappings[datasetType].keys
        if level is not None and level in self.levels:
            keyDict  = dict(keyDict)
            for l in self.levels[level]:
                if l in keyDict:
                    del keyDict[l]
        return keyDict

    def update(self):
        filenames = os.listdir(self.root)
        rawRegex = re.compile(r"(?P<dataset>\w+)-(?P<visit>\d+)-(?P<ccd>\d).*")
        self.index = {}
        for filename in filenames:
            m = rawRegex.match(filename)
            if not m:
                continue
            index = self.index.setdefault(m.group('dataset'), dict(ccd={None:[]}, visit={None:[]}))
            visit = int(m.group('visit'))
            ccd = int(m.group('ccd'))
            d1 = dict(visit=visit, ccd=ccd)
            d2 = dict(visit=visit)
            index['ccd'].setdefault(visit, []).append(d1)
            index['ccd'][None].append(d1)
            index['visit'][visit] = [d2]
            index['visit'][None].append(d1)

    def keys(self):
        return self.keyDict

    def bypass_camera(self, datasetType, pythonType, location, dataId):
        return self.camera

    def map_camera(self, dataId, write=False):
        return lsst.daf.persistence.ButlerLocation(
            "lsst.afw.cameraGeom.Camera", "Camera", None, [], dataId
            )

    def std_calexp(self, item, dataId):
        ccdId = lsst.afw.cameraGeom.Id(dataId["ccd"])
        item.setDetector(lsst.afw.cameraGeom.utils.findCcd(self.camera, ccdId))
        return item

    def _computeCcdExposureId(self, dataId):
        return long(dataId["visit"]) * 10 + long(dataId["ccd"])

    def splitCcdExposureId(ccdExposureId):
        return dict(visit=(long(ccdExposureId) // 10), ccd=(long(ccdExposureId) % 10))

    def bypass_ccdExposureId(self, datasetType, pythonType, location, dataId):
        return self._computeCcdExposureId(dataId)

    def bypass_ccdExposureId_bits(self, datasetType, pythonType, location, dataId):
        return 32

def makeSimpleCamera(
    nX, nY,
    sizeX, sizeY,
    gapX, gapY,
    pixelSize=1.0,
    plateScale=20.0,
    radialDistortion=0.925,
):
    """Create a camera

    @param[in] nx: number of detectors in x
    @param[in] ny: number of detectors in y
    @param[in] sizeX: detector size in x (pixels)
    @param[in] sizeY: detector size in y (pixels)
    @param[in] gapX: gap between detectors in x (mm)
    @param[in] gapY: gap between detectors in y (mm)
    @param[in] pixelSize: pixel size (mm) (a float)
    @param[in] plateScale: plate scale in arcsec/mm; 20.0 is for LSST
    @param[in] radialDistortion: radial distortion, in mm/rad^2
        (the r^3 coefficient of the radial distortion polynomial
        that converts PUPIL in radians to FOCAL_PLANE in mm);
        0.925 is the value Dave Monet measured for lsstSim data

    Each detector will have one amplifier (with no raw information).
    """
    pScaleRad = lsst.afw.geom.arcsecToRad(plateScale)
    radialDistortCoeffs = [0.0, 1.0/pScaleRad, 0.0, radialDistortion/pScaleRad]
    focalPlaneToPupil = lsst.afw.geom.RadialXYTransform(radialDistortCoeffs)
    transformMap = {
        lsst.afw.cameraGeom.PUPIL: focalPlaneToPupil
    }

    detectorList = []
    ccdBBox = lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(), lsst.afw.geom.Extent2I(sizeX, sizeY))
    for iY in range(nY):
        cY = (iY - 0.5 * (nY - 1)) * (pixelSize * sizeY + gapY)
        for iX in range(nX):
            cX = (iX - 0.5 * (nX - 1)) * (pixelSize * sizeY + gapX)
            fpPos = lsst.afw.geom.Point2D(cX, cY)
            detectorName = "detector %d,%d" % (iX, iY)
            detectorId = len(detectorList) + 1
            detectorList.append(DetectorWrapper(
                name = detectorName,
                id = detectorId,
                serial = detectorName + " serial",
                bbox = ccdBBox,
                ampExtent = ccdBBox.getDimensions(),
                numAmps = 1,
                pixelSize = lsst.afw.geom.Extent2D(pixelSize, pixelSize),
                orientation = lsst.afw.cameraGeom.Orientation(fpPos),
                plateScale = plateScale,
                radialDistortion = radialDistortion,
            ).detector)

    return lsst.afw.cameraGeom.Camera(
        name = "Simple Camera",
        detectorList = detectorList,
        transformMap = transformMap,
    )

def makeDataRepo(root):
    """
    Create a data repository for SimpleMapper and return a butler for it.

    Clobbers anything already in the given path.
    """
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)
    with open(os.path.join(root, "_mapper"), "w") as f:
        f.write("lsst.pipe.tasks.mocks.SimpleMapper\n")
    return lsst.daf.persistence.Butler(root=root)
