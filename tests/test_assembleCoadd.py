# This file is part of pipe_tasks.
#
# LSST Data Management System
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
# See COPYRIGHT file at the top of the source tree.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
"""Test AssembleCoaddTask and its variants.

This uses
"""
import unittest
import numpy as np

import lsst.utils.tests

import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.assembleCoadd import (AssembleCoaddTask, AssembleCoaddConfig,
                                           CompareWarpAssembleCoaddTask, CompareWarpAssembleCoaddConfig)
from lsst.pipe.tasks.dcrAssembleCoadd import DcrAssembleCoaddTask, DcrAssembleCoaddConfig
from assembleCoaddTestUtils import makeMockSkyInfo, MockCoaddTestData

__all__ = ["MockAssembleCoaddConfig", "MockAssembleCoaddTask",
           "MockCompareWarpAssembleCoaddConfig", "MockCompareWarpAssembleCoaddTask"]


class MockAssembleCoaddConfig(AssembleCoaddConfig):

    def setDefaults(self):
        super().setDefaults()
        self.doWrite = False


class MockAssembleCoaddTask(AssembleCoaddTask):
    """Lightly modified version of `AssembleCoaddTask` for use with unit tests.

    The modifications bypass the usual middleware for loading data and setting
    up the Task, and instead supply in-memory mock data references to the `run`
    method so that the coaddition algorithms can be tested without a Butler.
    """
    ConfigClass = MockAssembleCoaddConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.warpType = self.config.warpType
        self.makeSubtask("interpImage")
        self.makeSubtask("scaleZeroPoint")

    def processResults(self, *args, **kwargs):
        "This should be tested separately."
        pass

    def runQuantum(self, mockSkyInfo, warpRefList, *args):
        """Modified interface for testing coaddition algorithms without a Butler.

        Parameters
        ----------
        mockSkyInfo : `lsst.pipe.base.Struct`
            A simple container that supplies a bounding box and WCS in the
            same format as the output of
            `lsst.pipe.tasks.CoaddBaseTask.getSkyInfo`
        warpRefList : `list` of `lsst.pipe.tasks.MockExposureReference`
            Data references to the test exposures that will be coadded,
            using the Gen 3 API.

        Returns
        -------
        retStruct : `lsst.pipe.base.Struct`
            The coadded exposure and associated metadata.
        """
        inputs = self.prepareInputs(warpRefList)

        retStruct = self.run(mockSkyInfo, inputs.tempExpRefList, inputs.imageScalerList,
                             inputs.weightList, supplementaryData=pipeBase.Struct())
        return retStruct


class MockCompareWarpAssembleCoaddConfig(CompareWarpAssembleCoaddConfig):

    def setDefaults(self):
        super().setDefaults()
        self.assembleStaticSkyModel.retarget(MockAssembleCoaddTask)
        self.assembleStaticSkyModel.doWrite = False
        self.doWrite = False


class MockCompareWarpAssembleCoaddTask(MockAssembleCoaddTask, CompareWarpAssembleCoaddTask):
    """Lightly modified version of `CompareWarpAssembleCoaddTask`
    for use with unit tests.

    The modifications bypass the usual middleware for loading data and setting
    up the Task, and instead supply in-memory mock data references to the `run`
    method so that the coaddition algorithms can be tested without a Butler.
    """
    ConfigClass = MockCompareWarpAssembleCoaddConfig
    _DefaultName = "compareWarpAssembleCoadd"

    def __init__(self, *args, **kwargs):
        CompareWarpAssembleCoaddTask.__init__(self, *args, **kwargs)

    def runQuantum(self, mockSkyInfo, warpRefList, *args):
        inputs = self.prepareInputs(warpRefList)

        assembleStaticSkyModel = MockAssembleCoaddTask(config=self.config.assembleStaticSkyModel)
        templateCoadd = assembleStaticSkyModel.runQuantum(mockSkyInfo, warpRefList)

        supplementaryData = pipeBase.Struct(
            templateCoadd=templateCoadd.coaddExposure,
            nImage=templateCoadd.nImage,
            warpRefList=templateCoadd.warpRefList,
            imageScalerList=templateCoadd.imageScalerList,
            weightList=templateCoadd.weightList)

        retStruct = self.run(mockSkyInfo, inputs.tempExpRefList, inputs.imageScalerList,
                             inputs.weightList, supplementaryData=supplementaryData)
        return retStruct


class MockDcrAssembleCoaddConfig(DcrAssembleCoaddConfig):

    def setDefaults(self):
        super().setDefaults()
        self.assembleStaticSkyModel.retarget(MockCompareWarpAssembleCoaddTask)
        self.assembleStaticSkyModel.doWrite = False
        self.doWrite = False
        self.effectiveWavelength = 476.31  # Use LSST g band values for the test.
        self.bandwidth = 552. - 405.


class MockDcrAssembleCoaddTask(MockCompareWarpAssembleCoaddTask, DcrAssembleCoaddTask):
    """Lightly modified version of `DcrAssembleCoaddTask`
    for use with unit tests.

    The modifications bypass the usual middleware for loading data and setting
    up the Task, and instead supply in-memory mock data references to the `run`
    method so that the coaddition algorithms can be tested without a Butler.
    """
    ConfigClass = MockDcrAssembleCoaddConfig
    _DefaultName = "dcrAssembleCoadd"

    def __init__(self, *args, **kwargs):
        DcrAssembleCoaddTask.__init__(self, *args, **kwargs)


class MockInputMapAssembleCoaddConfig(MockCompareWarpAssembleCoaddConfig):

    def setDefaults(self):
        super().setDefaults()
        self.doInputMap = True


class MockInputMapAssembleCoaddTask(MockCompareWarpAssembleCoaddTask):
    """Lightly modified version of `CompareWarpAssembleCoaddTask`
    for use with unit tests.

    The modifications bypass the usual middleware for loading data and setting
    up the Task, and instead supply in-memory mock data references to the `run`
    method so that the coaddition algorithms can be tested without a Butler.
    """
    ConfigClass = MockInputMapAssembleCoaddConfig
    _DefaultName = "inputMapAssembleCoadd"

    def __init__(self, *args, **kwargs):
        CompareWarpAssembleCoaddTask.__init__(self, *args, **kwargs)


class AssembleCoaddTestCase(lsst.utils.tests.TestCase):
    """Tests of AssembleCoaddTask and its derived classes.

    These tests bypass the middleware used for accessing data and managing Task
    execution.
    """

    def setUp(self):
        patch = 42
        patchGen2 = "2,3"
        tract = 0
        testData = MockCoaddTestData(fluxRange=1e4)
        exposures = {}
        matchedExposures = {}
        for expId in range(100, 110):
            exposures[expId], matchedExposures[expId] = testData.makeTestImage(expId)
        self.gen2DataRefList = testData.makeGen2DataRefList(exposures, matchedExposures,
                                                            patch=patchGen2, tract=tract)
        self.dataRefList = testData.makeDataRefList(exposures, matchedExposures,
                                                    'direct', patch=patch, tract=tract)
        self.dataRefListPsfMatched = testData.makeDataRefList(exposures, matchedExposures,
                                                              'psfMatched', patch=patch, tract=tract)
        self.skyInfo = makeMockSkyInfo(testData.bbox, testData.wcs, patch=patch)

    def checkRun(self, assembleTask, warpType="direct"):
        """Check that the task runs successfully."""
        dataRefList = self.dataRefListPsfMatched if warpType == "psfMatched" else self.dataRefList
        result = assembleTask.runQuantum(self.skyInfo, dataRefList)

        # Check that we produced an exposure.
        self.assertTrue(result.coaddExposure is not None)

    def testAssembleBasic(self):
        config = MockAssembleCoaddConfig()
        config.validate()
        assembleTask = MockAssembleCoaddTask(config=config)
        self.checkRun(assembleTask)

    def testAssemblePsfMatched(self):
        config = MockAssembleCoaddConfig(warpType="psfMatched")
        config.validate()
        assembleTask = MockAssembleCoaddTask(config=config)
        self.checkRun(assembleTask, warpType="psfMatched")

    def testAssembleCompareWarp(self):
        config = MockCompareWarpAssembleCoaddConfig()
        config.validate()
        assembleTask = MockCompareWarpAssembleCoaddTask(config=config)
        self.checkRun(assembleTask)

    def testAssembleDCR(self):
        config = MockDcrAssembleCoaddConfig()
        config.validate()
        assembleTask = MockDcrAssembleCoaddTask(config=config)
        self.checkRun(assembleTask)

    def testOnlineCoadd(self):
        config = MockInputMapAssembleCoaddConfig()
        config.statistic = "MEAN"
        config.validate()
        assembleTask = MockInputMapAssembleCoaddTask(config=config)

        dataRefList = self.dataRefList
        results = assembleTask.runQuantum(self.skyInfo, dataRefList)
        coadd = results.coaddExposure

        configOnline = MockInputMapAssembleCoaddConfig()
        configOnline.statistic = "MEAN"
        configOnline.doOnlineForMean = True
        configOnline.validate()
        assembleTaskOnline = MockInputMapAssembleCoaddTask(config=configOnline)

        resultsOnline = assembleTaskOnline.runQuantum(self.skyInfo, dataRefList)
        coaddOnline = resultsOnline.coaddExposure

        self.assertFloatsAlmostEqual(coaddOnline.image.array,
                                     coadd.image.array, rtol=1e-3)
        self.assertFloatsAlmostEqual(coaddOnline.variance.array,
                                     coadd.variance.array, rtol=1e-6)
        self.assertMasksEqual(coaddOnline.mask, coadd.mask)

    def testInputMap(self):
        config = MockInputMapAssembleCoaddConfig()
        config.validate()
        assembleTask = MockInputMapAssembleCoaddTask(config=config)

        # Make exposures where one of them has a bad region.
        patch = 42
        tract = 0
        testData = MockCoaddTestData(fluxRange=1e4)
        exposures = {}
        matchedExposures = {}
        for expId in range(100, 110):
            if expId == 105:
                badBox = lsst.geom.Box2I(lsst.geom.Point2I(testData.bbox.beginX + 10,
                                                           testData.bbox.beginY + 10),
                                         lsst.geom.Extent2I(100, 100))
            else:
                badBox = None
            exposures[expId], matchedExposures[expId] = testData.makeTestImage(expId,
                                                                               badRegionBox=badBox)
        dataRefList = testData.makeDataRefList(exposures, matchedExposures,
                                               'direct', patch=patch, tract=tract)

        results = assembleTask.runQuantum(self.skyInfo, dataRefList)

        inputMap = results.inputMap
        validPix, raPix, decPix = inputMap.valid_pixels_pos(return_pixels=True)

        # Confirm that all the map pixels are in the bounding box
        # Exposure 100 is the first one and they all have the same WCS in the tests.
        xPix, yPix = exposures[100].getWcs().skyToPixelArray(raPix, decPix, degrees=True)
        self.assertGreater(xPix.min(), testData.bbox.beginX)
        self.assertGreater(yPix.min(), testData.bbox.beginY)
        self.assertLess(xPix.max(), testData.bbox.endX)
        self.assertLess(xPix.max(), testData.bbox.endY)

        # Confirm that all exposures except 105 are completely covered
        # This assumes we have one input per visit in the mock data.
        metadata = inputMap.metadata
        visitBitDict = {}
        for bit in range(inputMap.wide_mask_maxbits):
            if f'B{bit:04d}VIS' in metadata:
                visitBitDict[metadata[f'B{bit:04d}VIS']] = bit
        for expId in range(100, 110):
            if expId == 105:
                self.assertFalse(np.all(inputMap.check_bits_pix(validPix, [visitBitDict[expId]])))
            else:
                self.assertTrue(np.all(inputMap.check_bits_pix(validPix, [visitBitDict[expId]])))


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
