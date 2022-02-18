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
import unittest
import numpy as np
import logging

from lsst.geom import Point2I, Box2I, Extent2I
from lsst.skymap import TractInfo
from lsst.skymap.patchInfo import PatchInfo
import lsst.afw.image as afwImage
import lsst.utils.tests
from lsst.utils import getPackageDir
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig
from lsst.pipe.tasks.calibrate import CalibrateTask, CalibrateConfig
from lsst.meas.algorithms import SourceDetectionTask, SkyObjectsTask
from lsst.meas.extensions.scarlet.scarletDeblendTask import ScarletDeblendTask
from lsst.meas.base import SingleFrameMeasurementTask
from lsst.pipe.tasks.setPrimaryFlags import SetPrimaryFlagsTask, getPseudoSources
from lsst.afw.table import SourceCatalog


class NullTract(TractInfo):
    """A Tract not contained in the MockSkyMap.

    BaseSkyMap.findTract(coord) will always return a Tract,
    even if the coord isn't located in the Tract.
    In order to mimick this functionality we create a
    NullTract for regions of the MockSkyMap that
    aren't contained in any of the tracts.
    """
    def __init__(self):
        pass

    def getId(self):
        return None


class MockTractInfo:
    """A Tract based on a bounding box and WCS.

    Testing is made easier when we can specifically define
    a Tract in terms of its bounding box in pixel coordinates
    along with a WCS for the exposure.

    Only the relevant methods from `TractInfo` needed to make
    test pass are implemented here. Since this is just for
    testing, it isn't sophisticated and requires developers to
    ensure that the size of the bounding box is evenly divisible
    by the number of patches in the Tract.
    """
    def __init__(self, name, bbox, wcs, numPatches):
        self.name = name
        self.bbox = bbox
        self.wcs = wcs
        self._numPatches = numPatches
        assert bbox.getWidth()%numPatches[0] == 0
        assert bbox.getHeight()%numPatches[1] == 0
        self.patchWidth = bbox.getWidth()//numPatches[0]
        self.patchHeight = bbox.getHeight()//numPatches[1]

    def contains(self, coord):
        pixel = self.wcs.skyToPixel(coord)
        return self.bbox.contains(Point2I(pixel))

    def getId(self):
        return self.name

    def getNumPatches(self):
        return self._numPatches

    def getPatchInfo(self, index):
        x, y = index
        width = self.patchWidth
        height = self.patchHeight

        x = x*self.patchWidth
        y = y*self.patchHeight

        bbox = Box2I(Point2I(x, y), Extent2I(width, height))

        nx, ny = self._numPatches
        sequentialIndex = nx*y + x

        patchInfo = PatchInfo(
            index=index,
            innerBBox=bbox,
            outerBBox=bbox,
            sequentialIndex=sequentialIndex,
            tractWcs=self.wcs
        )
        return patchInfo

    def __getitem__(self, index):
        return self.getPatchInfo(index)

    def __iter__(self):
        xNum, yNum = self.getNumPatches()
        for y in range(yNum):
            for x in range(xNum):
                yield self.getPatchInfo((x, y))


class MockSkyMap:
    """A SkyMap based on a list of bounding boxes.

    Testing is made easier when we can specifically define
    a Tract in terms of its bounding box in pixel coordinates
    along with a WCS for the exposure. This class allows us
    to define the tract(s) in the SkyMap and create
    them.
    """
    def __init__(self, bboxes, wcs, numPatches):
        self.bboxes = bboxes
        self.wcs = wcs
        self.numPatches = numPatches

    def __iter__(self):
        for b, bbox in enumerate(self.bboxes):
            yield self.generateTract(b)

    def __getitem__(self, index):
        return self.generateTract(index)

    def generateTract(self, index):
        return MockTractInfo(index, self.bboxes[index], self.wcs, self.numPatches)

    def findTract(self, coord):
        for tractInfo in self:
            if tractInfo.contains(coord):
                return tractInfo

        return NullTract()


class IsPrimaryTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        # Load sample input from disk
        expPath = os.path.join(getPackageDir("pipe_tasks"), "tests", "data", "v695833-e0-c000-a00.sci.fits")
        self.exposure = afwImage.ExposureF(expPath)

        # Characterize the image (create PSF, etc.)
        charImConfig = CharacterizeImageConfig()
        charImTask = CharacterizeImageTask(config=charImConfig)
        self.charImResults = charImTask.run(self.exposure)

        # set log level so that warnings do not display
        logging.getLogger("lsst.calibrate").setLevel(logging.ERROR)

    def tearDown(self):
        del self.exposure
        self.charImResults

    def testIsSinglePrimaryFlag(self):
        """Tests detect_isPrimary column gets added when run, and that sources
        labelled as detect_isPrimary are not sky sources and have no children.
        """
        calibConfig = CalibrateConfig()
        calibConfig.doAstrometry = False
        calibConfig.doPhotoCal = False
        calibTask = CalibrateTask(config=calibConfig)
        calibResults = calibTask.run(self.charImResults.exposure)
        outputCat = calibResults.outputCat
        self.assertTrue("detect_isPrimary" in outputCat.schema.getNames())
        # make sure all sky sources are flagged as not primary
        self.assertEqual(sum((outputCat["detect_isPrimary"]) & (outputCat["sky_source"])), 0)
        # make sure all parent sources are flagged as not primary
        self.assertEqual(sum((outputCat["detect_isPrimary"]) & (outputCat["deblend_nChild"] > 0)), 0)

        with self.assertRaises(KeyError):
            outputCat.getSchema().find("detect_isDelendedModelPrimary")

    def testIsScarletPrimaryFlag(self):
        """Test detect_isPrimary column when scarlet is used as the deblender
        """
        # We need a multiband coadd for scarlet,
        # even though there is only one band
        coadds = afwImage.MultibandExposure.fromExposures(["test"], [self.exposure])

        # Create a SkyMap with a tract that contains a portion of the image,
        # subdivided into 3x3 patches
        wcs = self.exposure.getWcs()
        tractBBox = Box2I(Point2I(100, 100), Extent2I(900, 900))
        skyMap = MockSkyMap([tractBBox], wcs, (3, 3))
        tractInfo = skyMap[0]
        patchInfo = tractInfo[0, 0]
        patchBBox = patchInfo.getInnerBBox()

        schema = SourceCatalog.Table.makeMinimalSchema()
        # Initialize the detection task
        detectionTask = SourceDetectionTask(schema=schema)

        # Initialize the fake source injection task
        skyConfig = SkyObjectsTask.ConfigClass()
        skySourcesTask = SkyObjectsTask(name="skySources", config=skyConfig)
        schema.addField("merge_peak_sky", type="Flag")

        # Initialize the deblender task
        scarletConfig = ScarletDeblendTask.ConfigClass()
        scarletConfig.maxIter = 20
        scarletConfig.columnInheritance["merge_peak_sky"] = "merge_peak_sky"
        deblendTask = ScarletDeblendTask(schema=schema, config=scarletConfig)

        # We'll customize the configuration of measurement to just run the
        # minimal number of plugins to make setPrimaryFlags work.
        measureConfig = SingleFrameMeasurementTask.ConfigClass()
        measureConfig.plugins.names = ["base_SdssCentroid", "base_SkyCoord"]
        measureConfig.slots.psfFlux = None
        measureConfig.slots.apFlux = None
        measureConfig.slots.shape = None
        measureConfig.slots.modelFlux = None
        measureConfig.slots.calibFlux = None
        measureConfig.slots.gaussianFlux = None
        measureTask = SingleFrameMeasurementTask(config=measureConfig, schema=schema)
        primaryConfig = SetPrimaryFlagsTask.ConfigClass()
        setPrimaryTask = SetPrimaryFlagsTask(config=primaryConfig, schema=schema,
                                             name="setPrimaryFlags", isSingleFrame=False)

        table = SourceCatalog.Table.make(schema)
        # detect sources
        detectionResult = detectionTask.run(table, coadds["test"])
        catalog = detectionResult.sources
        # add fake sources
        skySources = skySourcesTask.run(mask=self.exposure.mask, seed=0)
        for foot in skySources[:5]:
            src = catalog.addNew()
            src.setFootprint(foot)
            src.set("merge_peak_sky", True)
        # deblend
        result, fluxResult = deblendTask.run(coadds, catalog)
        # measure
        measureTask.run(result["test"], self.exposure)
        outputCat = result["test"]
        # Set the primary flags
        setPrimaryTask.run(outputCat, skyMap=skyMap, tractInfo=tractInfo, patchInfo=patchInfo)

        # There should be the same number of deblenedPrimary and
        # deblendedModelPrimary sources,
        # since they both have the same blended sources and only differ
        # over which model to use for the isolated sources.
        isPseudo = getPseudoSources(outputCat, primaryConfig.pseudoFilterList, schema, setPrimaryTask.log)
        self.assertEqual(
            np.sum(outputCat["detect_isDeblendedSource"] & ~isPseudo),
            np.sum(outputCat["detect_isDeblendedModelSource"]))

        # Check that the sources contained in a tract are all marked appropriately
        x = outputCat["slot_Centroid_x"]
        y = outputCat["slot_Centroid_y"]
        tractInner = tractBBox.contains(x, y)
        np.testing.assert_array_equal(outputCat["detect_isTractInner"], tractInner)

        # Check that the sources contained in a patch are all marked appropriately
        patchInner = patchBBox.contains(x, y)
        np.testing.assert_array_equal(outputCat["detect_isPatchInner"], patchInner)

        # make sure all sky sources are flagged as not primary
        self.assertEqual(sum((outputCat["detect_isPrimary"]) & (outputCat["merge_peak_sky"])), 0)

        # Check that sky objects have not been deblended
        np.testing.assert_array_equal(
            isPseudo,
            isPseudo & (outputCat["deblend_nChild"] == 0)
        )


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
