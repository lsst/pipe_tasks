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

import os
import glob
import unittest

import numpy as np

import lsst.afw.image as afwImage
import lsst.geom as geom
from lsst.meas.algorithms.testUtils import MockReferenceObjectLoaderFromFiles
import lsst.meas.base.tests
from lsst.pipe.tasks.diffractionSpikeMask import DiffractionSpikeMaskTask, DiffractionSpikeMaskConfig
from lsst.pipe.tasks.colorterms import Colorterm, ColortermDict, ColortermLibrary
import lsst.utils.tests

from utils import makeTestVisitInfo


TESTDIR = os.path.abspath(os.path.dirname(__file__))
RefCatDir = os.path.join(TESTDIR, "data", "sdssrefcat")

testColorterms = ColortermLibrary(data={
    "test*": ColortermDict(data={
        "test-g": Colorterm(primary="g", secondary="r", c0=0.00, c1=0.00),
        "test-r": Colorterm(primary="r", secondary="i", c0=0.00, c1=0.00, c2=0.00),
        "test-i": Colorterm(primary="i", secondary="z", c0=1.00, c1=0.00, c2=0.00),
        "test-z": Colorterm(primary="z", secondary="i", c0=0.00, c1=0.00, c2=0.00),
    })
})


class DiffractionSpikeMaskTest(lsst.utils.tests.TestCase):

    def setUp(self):

        # Load sample input from disk
        testDir = os.path.dirname(__file__)

        # The .xy.fits file has sources in the range ~ [0,2000],[0,4500]
        # which is bigger than the exposure
        self.bbox = geom.Box2I(geom.Point2I(0, 0), geom.Extent2I(2048, 4612))

        smallExposure = afwImage.ExposureF(os.path.join(testDir, "data", "v695833-e0-c000-a00.sci.fits"))
        self.exposure = afwImage.ExposureF(self.bbox)
        self.exposure.setWcs(smallExposure.getWcs())
        self.exposure.setFilter(afwImage.FilterLabel(band="i", physical="test-i"))
        self.exposure.info.setVisitInfo(makeTestVisitInfo())

        # Make a reference loader
        filenames = sorted(glob.glob(os.path.join(RefCatDir, 'ref_cats', 'cal_ref_cat', '??????.fits')))
        self.refObjLoader = MockReferenceObjectLoaderFromFiles(filenames, htmLevel=8)

    def tearDown(self):
        del self.exposure
        del self.refObjLoader

    def test_raiseWithoutLoader(self):
        """The task should raise an error if no reference catalog loader is
        configured.
        """
        config = DiffractionSpikeMaskConfig()
        task = DiffractionSpikeMaskTask(config=config)
        with self.assertRaises(RuntimeError):
            task.run(self.exposure)

    def test_loadAndMaskStars(self):
        """Run the bright star mask with a selection of reference sources."""

        config = DiffractionSpikeMaskConfig(magnitudeThreshold=16)
        task = DiffractionSpikeMaskTask(self.refObjLoader, config=config)
        exposure = self.exposure.clone()
        # Set the saturated mask plane in half of the image
        saturatedMaskBit = exposure.mask.getPlaneBitMask(config.saturatedMaskPlane)
        bbox = exposure.getBBox()
        bbox.grow(-lsst.geom.Extent2I(0, bbox.height//4))
        exposure[bbox].mask.array |= saturatedMaskBit
        brightCat = task.run(exposure=exposure)
        self.assertGreater(len(brightCat), 0)
        # Verify that the new mask plane has been added
        spikeMaskBit = exposure.mask.getPlaneBitMask(config.spikeMask)
        # The images should not be modified
        self.assertImagesEqual(self.exposure.image, exposure.image)
        self.assertImagesEqual(self.exposure.variance, exposure.variance)
        # Ensure that the mask has changed
        self.assertFloatsNotEqual(self.exposure.mask.array, exposure.mask.array)

        # Check that the mask is set for the bright sources inside the image
        # Note that the catalog will include bright sources *off* the image that
        # have long enough diffraction spikes to overlap the edge of the image
        xvals, yvals = exposure.wcs.skyToPixelArray(brightCat[config.raKey], brightCat[config.decKey])
        bbox = exposure.getBBox()
        # Shrink the bounding box so that the bbox.contains check below is
        # sufficient to avoid errors.
        bbox.grow(-2)
        points = [geom.Point2D(xv, yv) for xv, yv in zip(xvals, yvals)]
        inside = 0
        outside = 0
        for pt in points:
            if bbox.contains(int(pt.getX()), int(pt.getY())):
                ptBox = geom.Box2I.makeCenteredBox(pt, geom.Extent2I(3, 3))
                maskSet = exposure[ptBox].mask.array & spikeMaskBit > 0
                self.assertTrue(np.all(maskSet))
                inside += 1
            else:
                outside += 1
        self.assertGreater(inside, 0)
        self.assertGreater(outside, 0)

    def test_noBrightStars(self):
        """Run the bright star mask with no bright stars."""

        # Set a very high magnitude limit so that no stars are selected
        config = DiffractionSpikeMaskConfig(magnitudeThreshold=0)
        task = DiffractionSpikeMaskTask(self.refObjLoader, config=config)
        exposure = self.exposure.clone()
        brightCat = task.run(exposure=exposure)
        self.assertEqual(len(brightCat), 0)
        # Verify that the new mask plane has been added
        exposure.mask.getPlaneBitMask(config.spikeMask)
        # The images should not be modified
        self.assertImagesEqual(self.exposure.image, exposure.image)

    def test_maskSources(self):
        """Verify that sources on and off the image are masked correctly."""
        task = DiffractionSpikeMaskTask(self.refObjLoader, config=DiffractionSpikeMaskConfig())
        task.set_diffraction_angle(self.exposure)
        self.exposure.mask.addMaskPlane(task.config.spikeMask)

        xSize, ySize = self.bbox.getDimensions()
        x0, y0 = self.bbox.getBegin()
        x1, y1 = self.bbox.getEnd()

        nBright = 50
        rng = np.random.RandomState(3)
        xLoc = np.arange(x0 - xSize/4, x1 + xSize/4)
        rng.shuffle(xLoc)
        xLoc = xLoc[:nBright]
        yLoc = np.arange(y0 - ySize/4, y1 + ySize/4)
        rng.shuffle(yLoc)
        yLoc = yLoc[:nBright]
        spikeRadii = np.arange(10, 200)
        rng.shuffle(spikeRadii)
        spikeRadii = spikeRadii[:nBright]
        saturatedBox = geom.Box2I(self.bbox.getBegin(), geom.Extent2I(xSize, ySize//2))
        baseMask = self.exposure.mask.clone()
        baseMask[saturatedBox].array |= baseMask.getPlaneBitMask(task.config.saturatedMaskPlane)
        # There are four classes of sources:
        # 1. Bright sources on the image with saturated cores - masked
        # 2. Sources on the image without saturated cores - not masked
        # 3. Bright sources off the image with predicted diffraction spikes that
        # overlap the image - masked
        # 4. Bright sources off the image that are far away enough that any
        # diffraction spikes would not overlap the image - not masked
        nClass1 = 0
        nClass2 = 0
        nClass3 = 0
        nClass4 = 0
        selectedSources = task.selectSources(xLoc, yLoc, spikeRadii, baseMask)
        for x, y, r, selected in zip(xLoc, yLoc, spikeRadii, selectedSources):
            mask = baseMask.clone()
            isInImage = self.bbox.contains(geom.Point2I(x, y))
            # Bright sources on the image with saturated cores.
            if isInImage and selected:
                nClass1 += 1
                task.maskSources([x], [y], [r], mask)
                self.assertGreater(np.sum(mask.array & mask.getPlaneBitMask(task.config.spikeMask) > 0), 0)

            # Bright sources on the image without saturated cores.
            if isInImage and not selected:
                nClass2 += 1
                # Do *not* run task.maskSources in this case, since these are
                # skipped.

            # Bright sources off the image that we predict should overlap the
            # image, should set the SPIKE mask for some pixels.
            if not isInImage and selected:
                nClass3 += 1
                task.maskSources([x], [y], [r], mask)
                self.assertGreater(np.sum(mask.array & mask.getPlaneBitMask(task.config.spikeMask) > 0), 0)

            # Sources off the image that are skipped in source selection should
            # not change the mask even if we do calculate their SPIKE mask.
            if not isInImage and not selected:
                nClass4 += 1
                task.maskSources([x], [y], [r], mask)
                self.assertMasksEqual(mask, baseMask)
        # Verify that the test points were sufficient to exercise all classes.
        self.assertGreater(nClass1, 0)
        self.assertGreater(nClass2, 0)
        self.assertGreater(nClass3, 0)
        self.assertGreater(nClass4, 0)


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
