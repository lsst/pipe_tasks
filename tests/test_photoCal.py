#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
import os
import unittest
import logging
import glob

import numpy as np
import astropy.units as u

import lsst.geom as geom
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.utils.tests
from lsst.utils import getPackageDir
from lsst.pipe.tasks.photoCal import PhotoCalTask, PhotoCalConfig
from lsst.pipe.tasks.colorterms import Colorterm, ColortermDict, ColortermLibrary
from lsst.utils.logging import TRACE
from lsst.meas.algorithms.testUtils import MockReferenceObjectLoaderFromFiles

RefCatDir = os.path.join(getPackageDir("pipe_tasks"), "tests", "data", "sdssrefcat")

testColorterms = ColortermLibrary(data={
    "test*": ColortermDict(data={
        "test-g": Colorterm(primary="g", secondary="r", c0=0.00, c1=0.00),
        "test-r": Colorterm(primary="r", secondary="i", c0=0.00, c1=0.00, c2=0.00),
        "test-i": Colorterm(primary="i", secondary="z", c0=1.00, c1=0.00, c2=0.00),
        "test-z": Colorterm(primary="z", secondary="i", c0=0.00, c1=0.00, c2=0.00),
    })
})


def setup_module(module):
    lsst.utils.tests.init()


class PhotoCalTest(unittest.TestCase):

    def setUp(self):

        # Load sample input from disk
        testDir = os.path.dirname(__file__)
        self.srcCat = afwTable.SourceCatalog.readFits(
            os.path.join(testDir, "data", "v695833-e0-c000.xy.fits"))

        self.srcCat["slot_ApFlux_instFluxErr"] = 1
        self.srcCat["slot_PsfFlux_instFluxErr"] = 1

        # The .xy.fits file has sources in the range ~ [0,2000],[0,4500]
        # which is bigger than the exposure
        self.bbox = geom.Box2I(geom.Point2I(0, 0), geom.Extent2I(2048, 4612))
        smallExposure = afwImage.ExposureF(os.path.join(testDir, "data", "v695833-e0-c000-a00.sci.fits"))
        self.exposure = afwImage.ExposureF(self.bbox)
        self.exposure.setWcs(smallExposure.getWcs())
        self.exposure.setFilter(afwImage.FilterLabel(band="i", physical="test-i"))
        self.exposure.setPhotoCalib(smallExposure.getPhotoCalib())

        coordKey = self.srcCat.getCoordKey()
        centroidKey = self.srcCat.getCentroidSlot().getMeasKey()
        wcs = self.exposure.getWcs()
        for src in self.srcCat:
            src.set(coordKey, wcs.pixelToSky(src.get(centroidKey)))

        # Make a reference loader
        filenames = sorted(glob.glob(os.path.join(RefCatDir, 'ref_cats', 'cal_ref_cat', '??????.fits')))
        self.refObjLoader = MockReferenceObjectLoaderFromFiles(filenames, htmLevel=8)
        self.log = logging.getLogger('lsst.testPhotoCal')
        self.log.setLevel(TRACE)

        self.config = PhotoCalConfig()
        self.config.match.matchRadius = 0.5
        self.config.match.referenceSelection.doMagLimit = True
        self.config.match.referenceSelection.magLimit.maximum = 22.0
        self.config.match.referenceSelection.magLimit.fluxField = "i_flux"
        self.config.match.referenceSelection.doFlags = True
        self.config.match.referenceSelection.flags.good = ['photometric']
        self.config.match.referenceSelection.flags.bad = ['resolved']
        self.config.match.sourceSelection.doUnresolved = False  # Don't have star/galaxy in the srcCat

        # The test and associated data have been prepared on the basis that we
        # use the PsfFlux to perform photometry.
        self.config.fluxField = "base_PsfFlux_instFlux"

    def tearDown(self):
        del self.srcCat
        del self.exposure
        del self.refObjLoader
        del self.log

    def _runTask(self):
        """All the common setup to actually test the results"""
        task = PhotoCalTask(self.refObjLoader, config=self.config, schema=self.srcCat.schema)
        pCal = task.run(exposure=self.exposure, sourceCat=self.srcCat)
        matches = pCal.matches
        refFluxField = pCal.arrays.refFluxFieldList[0]

        # These are *all* the matches; we don't really expect to do that well.
        diff = []
        for m in matches:
            refFlux = m[0].get(refFluxField)  # reference catalog flux
            if refFlux <= 0:
                continue
            refMag = u.Quantity(refFlux, u.nJy).to_value(u.ABmag)
            instFlux = m[1].getPsfInstFlux()  # Instrumental Flux
            if instFlux <= 0:
                continue
            instMag = pCal.photoCalib.instFluxToMagnitude(instFlux)  # Instrumental mag
            diff.append(instMag - refMag)
        self.diff = np.array(diff)
        # Differences of matched objects that were used in the fit.
        self.zp = pCal.photoCalib.instFluxToMagnitude(1.)
        self.fitdiff = pCal.arrays.srcMag + self.zp - pCal.arrays.refMag

    def testFlags(self):
        """test that all the calib_photometry flags are set to reasonable values"""
        schema = self.srcCat.schema
        task = PhotoCalTask(self.refObjLoader, config=self.config, schema=schema)
        mapper = afwTable.SchemaMapper(self.srcCat.schema, schema)
        cat = afwTable.SourceCatalog(schema)
        for name in self.srcCat.schema.getNames():
            mapper.addMapping(self.srcCat.schema.find(name).key)
        cat.extend(self.srcCat, mapper=mapper)

        # test that by default, no stars are reserved and all used are candidates
        task.run(exposure=self.exposure, sourceCat=cat)
        used = 0
        for source in cat:
            if source.get("calib_photometry_used"):
                used += 1
            self.assertFalse(source.get("calib_photometry_reserved"))
        # test that some are actually used
        self.assertGreater(used, 0)

    def testZeroPoint(self):
        """ Test to see if we can compute a photometric zeropoint given a reference task"""
        self._runTask()
        self.assertGreater(len(self.diff), 50)
        self.log.info('%i magnitude differences; mean difference %g; mean abs diff %g' %
                      (len(self.diff), np.mean(self.diff), np.mean(np.abs(self.diff))))
        self.assertLess(np.mean(self.diff), 0.6)

        # Differences of matched objects that were used in the fit.
        self.log.debug('zeropoint: %g', self.zp)
        self.log.debug('number of sources used in fit: %i', len(self.fitdiff))
        self.log.debug('rms diff: %g', np.mean(self.fitdiff**2)**0.5)
        self.log.debug('median abs(diff): %g', np.median(np.abs(self.fitdiff)))

        # zeropoint: 31.3145
        # number of sources used in fit: 65
        # median diff: -0.009681
        # mean diff: 0.00331871
        # median abs(diff): 0.0368904
        # mean abs(diff): 0.0516589

        self.assertLess(abs(self.zp - 31.3145), 0.05)
        self.assertGreater(len(self.fitdiff), 50)
        # Tolerances are somewhat arbitrary; they're set simply to avoid regressions, and
        # are not based on we'd expect to get given the data quality.
        lq, uq = np.percentile(self.fitdiff, (25, 75))
        rms = 0.741*(uq - lq)  # Convert IQR to stdev assuming a Gaussian
        self.assertLess(rms, 0.07)    # rms difference
        self.assertLess(np.median(np.abs(self.fitdiff)), 0.06)  # median absolution difference

    def testColorTerms(self):
        """ Test to see if we can apply colorterm corrections while computing photometric zeropoints"""
        # Turn colorterms on. The colorterm library used here is simple - we just apply a 1 mag
        # color-independentcolorterm correction to everything. This should change the photometric zeropoint.
        # by 1 mag.
        self.config.applyColorTerms = True
        self.config.colorterms = testColorterms
        self.config.photoCatName = "testglob"  # Check glo expansion
        # zerPointOffset is the offset in the zeropoint that we expect from a uniform (i.e. color-independent)
        # colorterm correction.
        zeroPointOffset = testColorterms.data['test*'].data['test-i'].c0
        self._runTask()

        self.assertLess(np.mean(self.diff), 0.6 + zeroPointOffset)
        self.log.debug('zeropoint: %g', self.zp)
        # zeropoint: 32.3145
        self.assertLess(abs(self.zp - (31.3145 + zeroPointOffset)), 0.05)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
