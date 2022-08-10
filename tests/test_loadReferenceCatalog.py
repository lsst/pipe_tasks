import unittest
import copy

import astropy.units as u

import lsst.utils.tests
import lsst.afw.geom
import lsst.pipe.base as pipeBase
from lsst.meas.algorithms import (LoadReferenceObjectsTask,
                                  getRefFluxField,
                                  ReferenceObjectLoader)
from lsst.pipe.tasks.loadReferenceCatalog import LoadReferenceCatalogConfig, LoadReferenceCatalogTask
from lsst.pipe.tasks.colorterms import Colorterm, ColortermDict, ColortermLibrary


synthTerms = ColortermLibrary(data={
    "synth*": ColortermDict(data={
        "filter1": Colorterm(primary="ref1", secondary="ref2", c0=0.0, c1=0.01),
        "filter2": Colorterm(primary="ref2", secondary="ref3", c0=0.0, c1=-0.01),
    })
})


_synthFlux = 100.0
_synthCenter = lsst.geom.SpherePoint(30, -30, lsst.geom.degrees)


def setup_module(module):
    lsst.utils.tests.init()


class TrivialLoader(ReferenceObjectLoader):
    """Minimal subclass of LoadReferenceObjectsTask"""
    def make_synthetic_refcat(self, center, flux):
        """Make a synthetic reference catalog."""
        filters = ["ref1", "ref2", "ref3"]
        schema = LoadReferenceObjectsTask.makeMinimalSchema(filters)
        schema.addField('pm_ra', 'D')
        schema.addField('pm_dec', 'D')

        catalog = lsst.afw.table.SimpleCatalog(schema)
        record = catalog.addNew()
        record.setCoord(center)
        record[filters[0] + '_flux'] = flux
        record[filters[0] + '_fluxErr'] = flux*0.1
        record[filters[1] + '_flux'] = flux*10
        record[filters[1] + '_fluxErr'] = flux*10*0.1
        record[filters[2] + '_flux'] = flux*100
        record[filters[2] + '_fluxErr'] = flux*100*0.1
        record['pm_ra'] = 0.0
        record['pm_dec'] = 0.0

        return catalog

    def loadSkyCircle(self, ctrCoord, radius, filterName, **kwargs):
        refCat = self.make_synthetic_refcat(_synthCenter, _synthFlux)
        fluxField = getRefFluxField(schema=refCat.schema, filterName=filterName)
        return pipeBase.Struct(
            refCat=self.make_synthetic_refcat(_synthCenter, _synthFlux),
            fluxField=fluxField
        )

    def loadPixelBox(self, bbox, wcs, referenceFilter, **kwargs):
        return self.loadSkyCircle(None, None, referenceFilter)


class LoadReferenceCatalogTestCase(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = LoadReferenceCatalogConfig()
        cls.config.refObjLoader.filterMap = {"filter1": "ref1",
                                             "filter2": "ref2"}
        cls.config.refObjLoader.ref_dataset_name = 'synthCam'
        cls.config.colorterms = synthTerms
        cls.config.referenceSelector.doSignalToNoise = True
        cls.config.referenceSelector.signalToNoise.fluxField = 'ref1_flux'
        cls.config.referenceSelector.signalToNoise.errField = 'ref1_fluxErr'
        cls.config.referenceSelector.signalToNoise.minimum = 20.0

        cls.config.doApplyColorTerms = False
        cls.config.doReferenceSelection = False

        cls.synthMag1 = (_synthFlux*u.nanojansky).to(u.ABmag).value
        cls.synthMag2 = ((_synthFlux*10)*u.nanojansky).to(u.ABmag).value
        cls.synthMag3 = ((_synthFlux*100)*u.nanojansky).to(u.ABmag).value

        cls.synthMag1Corr = cls.synthMag1 + 0.01*(cls.synthMag1 - cls.synthMag2)
        cls.synthMag2Corr = cls.synthMag2 - 0.01*(cls.synthMag2 - cls.synthMag3)

        cls.trivialLoader = TrivialLoader(dataIds=[],
                                          refCats=[],
                                          name="synthCam",
                                          config=cls.config.refObjLoader)

    def testGetReferenceCatalogCircle(self):
        """Get a reference catalog skycircle."""
        config = copy.copy(self.config)
        config.freeze()

        loaderTask = LoadReferenceCatalogTask(config=config, dataIds=[], refCats=[], name="synthCam")
        # Monkey-patch our testing trivial loader to bypass the butler
        loaderTask.refObjLoader = self.trivialLoader

        cat = loaderTask.getSkyCircleCatalog(_synthCenter,
                                             1.0*lsst.geom.degrees,
                                             ['filter1', 'filter2'])

        self.assertAlmostEqual(cat['ra'], _synthCenter.getRa().asDegrees())
        self.assertAlmostEqual(cat['dec'], _synthCenter.getDec().asDegrees())
        self.assertFloatsAlmostEqual(cat['refMag'][0, 0], self.synthMag1, rtol=1e-7)
        self.assertFloatsAlmostEqual(cat['refMag'][0, 1], self.synthMag2, rtol=1e-7)

    def testGetReferenceCatalogBox(self):
        """Get a reference catalog box."""
        config = copy.copy(self.config)
        config.freeze()

        loaderTask = LoadReferenceCatalogTask(config=config, dataIds=[], refCats=[], name="synthCam")
        # Monkey-patch our testing trivial loader to bypass the butler
        loaderTask.refObjLoader = self.trivialLoader

        bbox = lsst.geom.Box2I(corner=lsst.geom.Point2I(0, 0),
                               dimensions=lsst.geom.Extent2I(100, 100))
        crpix = lsst.geom.Point2D(50, 50)
        crval = _synthCenter
        cdMatrix = lsst.afw.geom.makeCdMatrix(scale=1.0*lsst.geom.arcseconds)
        wcs = lsst.afw.geom.makeSkyWcs(crpix, crval, cdMatrix)

        cat = loaderTask.getPixelBoxCatalog(bbox,
                                            wcs,
                                            ['filter1', 'filter2'])

        self.assertAlmostEqual(cat['ra'], _synthCenter.getRa().asDegrees())
        self.assertAlmostEqual(cat['dec'], _synthCenter.getDec().asDegrees())
        self.assertFloatsAlmostEqual(cat['refMag'][0, 0], self.synthMag1, rtol=1e-7)
        self.assertFloatsAlmostEqual(cat['refMag'][0, 1], self.synthMag2, rtol=1e-7)

    def testGetReferenceCatalogCircleColorterms(self):
        """Get a reference catalog circle, with color terms applied."""
        config = copy.copy(self.config)
        config.doApplyColorTerms = True
        config.freeze()

        loaderTask = LoadReferenceCatalogTask(config=config, dataIds=[], refCats=[], name="synthCam")
        # Monkey-patch our testing trivial loader to bypass the butler
        loaderTask.refObjLoader = self.trivialLoader

        cat = loaderTask.getSkyCircleCatalog(_synthCenter,
                                             1.0*lsst.geom.degrees,
                                             ['filter1', 'filter2'])

        self.assertAlmostEqual(cat['ra'], _synthCenter.getRa().asDegrees())
        self.assertAlmostEqual(cat['dec'], _synthCenter.getDec().asDegrees())
        self.assertFloatsAlmostEqual(cat['refMag'][0, 0], self.synthMag1Corr, rtol=1e-7)
        self.assertFloatsAlmostEqual(cat['refMag'][0, 1], self.synthMag2Corr, rtol=1e-7)

    def testGetReferenceCatalogCircleSelection(self):
        """Get a reference catalog circle, apply selection."""
        config = copy.copy(self.config)
        config.doReferenceSelection = True
        config.freeze()

        loaderTask = LoadReferenceCatalogTask(config=config, dataIds=[], refCats=[], name="synthCam")
        # Monkey-patch our testing trivial loader to bypass the butler
        loaderTask.refObjLoader = self.trivialLoader

        cat = loaderTask.getSkyCircleCatalog(_synthCenter,
                                             1.0*lsst.geom.degrees,
                                             ['filter1', 'filter2'])

        # The selection removed all the objects.
        self.assertEqual(len(cat), 0)

    def testGetReferenceCatalogCircleSingleFilter(self):
        """Get a reference catalog circle, single filter."""
        config = copy.copy(self.config)
        config.freeze()

        loaderTask = LoadReferenceCatalogTask(config=config, dataIds=[], refCats=[], name="synthCam")
        # Monkey-patch our testing trivial loader to bypass the butler
        loaderTask.refObjLoader = self.trivialLoader

        cat = loaderTask.getSkyCircleCatalog(_synthCenter,
                                             1.0*lsst.geom.degrees,
                                             ['filter1'])

        self.assertAlmostEqual(cat['ra'], _synthCenter.getRa().asDegrees())
        self.assertAlmostEqual(cat['dec'], _synthCenter.getDec().asDegrees())
        self.assertFloatsAlmostEqual(cat['refMag'][0, 0], self.synthMag1, rtol=1e-7)

    def testGetReferenceCatalogAnyFilter(self):
        """Get a reference catalog circle, using anyFilterMapsToThis."""
        config = LoadReferenceCatalogConfig()
        config.refObjLoader.anyFilterMapsToThis = 'ref1'
        config.refObjLoader.ref_dataset_name = 'synthCam'

        config.doApplyColorTerms = False
        config.doReferenceSelection = False
        config.freeze()

        loaderTask = LoadReferenceCatalogTask(config=config, dataIds=[], refCats=[], name="synthCam")
        # Monkey-patch our testing trivial loader to bypass the butler
        loaderTask.refObjLoader = self.trivialLoader

        cat = loaderTask.getSkyCircleCatalog(_synthCenter,
                                             1.0*lsst.geom.degrees,
                                             ['filter1', 'filter2'])

        self.assertFloatsAlmostEqual(cat['refMag'][0, 0], self.synthMag1, rtol=1e-7)
        self.assertFloatsAlmostEqual(cat['refMag'][0, 1], self.synthMag1, rtol=1e-7)

    def testGetReferenceCatalogRequirePm(self):
        """Get a reference catalog circle, requiring proper motion."""
        config = copy.copy(self.config)
        config.refObjLoader.requireProperMotion = True
        config.freeze()

        loaderTask = LoadReferenceCatalogTask(config=config, dataIds=[], refCats=[], name="synthCam")
        # Monkey-patch our testing trivial loader to bypass the butler
        trivialLoader2 = TrivialLoader(dataIds=[], refCats=[], name="synthCam", config=config.refObjLoader)
        loaderTask.refObjLoader = trivialLoader2

        cat = loaderTask.getSkyCircleCatalog(_synthCenter,
                                             1.0*lsst.geom.degrees,
                                             ['filter1'])

        self.assertAlmostEqual(cat['ra'], _synthCenter.getRa().asDegrees())
        self.assertAlmostEqual(cat['dec'], _synthCenter.getDec().asDegrees())
        self.assertFloatsAlmostEqual(cat['refMag'][0, 0], self.synthMag1, rtol=1e-7)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
