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


def setup_module(module):
    lsst.utils.tests.init()


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

    def test_loadAndMaskStars(self):
        """Run the bright star mask with a selection of reference sources."""

        config = DiffractionSpikeMaskConfig(magnitudeThreshold=16)
        task = DiffractionSpikeMaskTask(self.refObjLoader, config=config)
        exposure = self.exposure.clone()
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
        """Run the bright star mask with no bright stars"""

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
