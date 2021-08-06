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
import pandas as pd
import PIL
import unittest
import tempfile

import lsst.afw.table
import lsst.geom
import lsst.meas.base.tests
from lsst.pipe.tasks.zooniverseCutouts import ZooniverseCutoutsTask
import lsst.utils.tests


class TestZooniverseCutouts(lsst.utils.tests.TestCase):
    """Test that ZooniverseCutoutsTask generates images and manifest files
    correctly.
    """
    def setUp(self):
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Point2I(100, 100))
        self.centroid = lsst.geom.Point2D(65, 70)
        dataset = lsst.meas.base.tests.TestDataset(bbox)
        dataset.addSource(instFlux=1e5, centroid=self.centroid)
        self.science, self.scienceCat = dataset.realize(noise=1000.0, schema=dataset.makeMinimalSchema())
        lsst.afw.table.updateSourceCoords(self.science.wcs, self.scienceCat)
        self.skyCenter = self.scienceCat[0].getCoord()
        self.template, self.templateCat = dataset.realize(noise=5.0, schema=dataset.makeMinimalSchema())
        # A simple and incorrect image difference to have something to plot.
        self.difference = lsst.afw.image.ExposureF(self.science, deep=True)
        self.difference.image -= self.template.image

    def test_generate_image(self):
        """Test that we get some kind of image out.

        It's useful to have a person look at the output via:
            im.show()
        """
        zooniverseCutouts = ZooniverseCutoutsTask()
        cutout = zooniverseCutouts.generate_image(self.science, self.template, self.difference,
                                                  self.skyCenter)
        im = PIL.Image.open(cutout)
        # NOTE: uncomment this to show the resulting image.
        # im.show()
        # NOTE: the dimensions here are determined by the matplotlib figure
        # size (in inches) and the dpi (default=100), plus borders.
        self.assertEqual(im.height, 233)
        self.assertEqual(im.width, 630)

    def test_generate_image_larger_cutout(self):
        """A different cutout size: the resulting cutout image is the same
        size but shows more pixels.
        """
        config = ZooniverseCutoutsTask.ConfigClass()
        config.size = 100
        zooniverseCutouts = ZooniverseCutoutsTask(config=config)
        cutout = zooniverseCutouts.generate_image(self.science, self.template, self.difference,
                                                  self.skyCenter)
        im = PIL.Image.open(cutout)
        # NOTE: uncomment this to show the resulting image.
        # im.show()
        # NOTE: the dimensions here are determined by the matplotlib figure
        # size (in inches) and the dpi (default=100), plus borders.
        self.assertEqual(im.height, 233)
        self.assertEqual(im.width, 630)

    def test_write_images(self):
        """Test that images get written to a temporary directory."""
        data = pd.DataFrame(data={"diaSourceId": [5, 10],
                                  "ra": [45.001, 45.002],
                                  "decl": [45.0, 45.001],
                                  "ccd": [50, 60],
                                  "visit": [1234, 5678]})
        butler = unittest.mock.Mock(spec=lsst.daf.butler.Butler)
        # we don't care what the output images look like here, just that
        # butler.get() returns an Exposure for every call.
        butler.get.return_value = self.science

        with tempfile.TemporaryDirectory() as path:
            config = ZooniverseCutoutsTask.ConfigClass()
            config.outputPath = path
            zooniverseCutouts = ZooniverseCutoutsTask(config)
            zooniverseCutouts.write_images(data, butler)
            for file in ("images/5.png", "images/10.png"):
                filename = os.path.join(path, file)
                self.assertTrue(os.path.exists(filename))
                image = PIL.Image.open(filename)
                self.assertEqual(image.format, "PNG")

    def check_make_manifest(self, url_root, url_list):
        """Check that make_manifest returns an appropriate DataFrame.
        """
        data = pd.DataFrame(data={"diaSourceId": [5, 10, 20], "otherField": [3, 2, 1]})
        config = ZooniverseCutoutsTask.ConfigClass()
        config.urlRoot = url_root
        zooniverseCutouts = ZooniverseCutoutsTask(config=config)
        manifest = zooniverseCutouts.make_manifest(data)
        self.assertEqual(manifest['metadata:diaSourceId'].to_list(),
                         [5, 10, 20])
        self.assertEqual(manifest['location:1'].to_list(), url_list)

    def test_make_manifest(self):
        # check without an ending slash
        root = "http://example.org/zooniverse"
        url_list = [f"{root}/images/5.png",
                    f"{root}/images/10.png",
                    f"{root}/images/20.png"]
        self.check_make_manifest(root, url_list)

        # check with an ending slash
        root = "http://example.org/zooniverse/"
        url_list = [f"{root}images/5.png",
                    f"{root}images/10.png",
                    f"{root}images/20.png"]
        self.check_make_manifest(root, url_list)
