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

import lsst.utils.tests

from lsst.pipe.tasks.assembleCoadd import (AssembleCoaddTask, AssembleCoaddConfig,
                                           SafeClipAssembleCoaddTask, SafeClipAssembleCoaddConfig,
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

    def _dataRef2DebugPath(self, *args, **kwargs):
        raise NotImplementedError("This lightweight version of the task is not "
                                  "meant to test debugging options.")

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
        supplementaryData = self.makeSupplementaryData(mockSkyInfo, warpRefList=inputs.tempExpRefList)
        retStruct = self.run(mockSkyInfo, inputs.tempExpRefList, inputs.imageScalerList,
                             inputs.weightList, supplementaryData=supplementaryData)
        return retStruct

    def runDataRef(self, mockSkyInfo, selectDataList=None, warpRefList=None):
        """Modified interface for testing coaddition algorithms without a Butler.

        Notes
        -----
        This tests the coaddition algorithms using Gen 2 Butler data references,
        and can be removed once that is fully deprecated.

        Both `runDataRef` and `runQuantum` are needed even those their
        implementation here is identical, because the Gen 2 and Gen 3 versions
        of `makeSupplementaryData` call `runDataRef` and `runQuantum` to build
        initial templates, respectively.

        Parameters
        ----------
        mockSkyInfo : `lsst.pipe.base.Struct`
            A simple container that supplies a bounding box and WCS in the
            same format as the output of
            `lsst.pipe.tasks.CoaddBaseTask.getSkyInfo`
        warpRefList : `list` of `lsst.pipe.tasks.MockGen2ExposureReference`
            Data references to the test exposures that will be coadded,
            using the Gen 2 API.

        Returns
        -------
        retStruct : `lsst.pipe.base.Struct`
            The coadded exposure and associated metadata.
        """
        inputData = self.prepareInputs(warpRefList)
        supplementaryData = self.makeSupplementaryData(mockSkyInfo, warpRefList=inputData.tempExpRefList)
        retStruct = self.run(mockSkyInfo, inputData.tempExpRefList, inputData.imageScalerList,
                             inputData.weightList, supplementaryData=supplementaryData)
        return retStruct


class MockSafeClipAssembleCoaddConfig(SafeClipAssembleCoaddConfig):

    def setDefaults(self):
        super().setDefaults()
        self.doWrite = False


class MockSafeClipAssembleCoaddTask(MockAssembleCoaddTask, SafeClipAssembleCoaddTask):
    """Lightly modified version of `SafeClipAssembleCoaddTask`
    for use with unit tests.

    The modifications bypass the usual middleware for loading data and setting
    up the Task, and instead supply in-memory mock data references to the `run`
    method so that the coaddition algorithms can be tested without a Butler.
    """
    ConfigClass = MockSafeClipAssembleCoaddConfig
    _DefaultName = "safeClipAssembleCoadd"

    def __init__(self, *args, **kwargs):
        SafeClipAssembleCoaddTask.__init__(self, *args, **kwargs)


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


class MockDcrAssembleCoaddConfig(DcrAssembleCoaddConfig):

    def setDefaults(self):
        super().setDefaults()
        self.assembleStaticSkyModel.retarget(MockCompareWarpAssembleCoaddTask)
        self.assembleStaticSkyModel.doWrite = False
        self.doWrite = False
        self.effectiveWavelength = 476.31  # Use LSST g band values for the test.
        self.bandwidth = 552. - 405.


class MockDcrAssembleCoaddTask(MockAssembleCoaddTask, DcrAssembleCoaddTask):
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


# class MockInputMapAssembleCoaddConfig(MockSafeClipAssembleCoaddConfig):
class MockInputMapAssembleCoaddConfig(MockCompareWarpAssembleCoaddConfig):

    def setDefaults(self):
        super().setDefaults()
        self.doInputMap = True


# class MockInputMapAssembleCoaddTask(MockSafeClipAssembleCoaddTask):
# class MockInputMapAssembleCoaddTask(MockAssembleCoaddTask, SafeClipAssembleCoaddTask):
class MockInputMapAssembleCoaddTask(MockCompareWarpAssembleCoaddTask):
    """Lightly modified version of `SafeClipAssembleCoaddTask`
    for use with unit tests.

    The modifications bypass the usual middleware for loading data and setting
    up the Task, and instead supply in-memory mock data references to the `run`
    method so that the coaddition algorithms can be tested without a Butler.
    """
    ConfigClass = MockInputMapAssembleCoaddConfig
    _DefaultName = "inputMapAssembleCoadd"

    def __init__(self, *args, **kwargs):
        # SafeClipAssembleCoaddTask.__init__(self, *args, **kwargs)
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
        self.skyInfoGen2 = makeMockSkyInfo(testData.bbox, testData.wcs, patch=patchGen2)
        self.skyInfo = makeMockSkyInfo(testData.bbox, testData.wcs, patch=patch)

    def checkGen2Gen3Compatibility(self, assembleTask, warpType="direct"):
        dataRefList = self.dataRefListPsfMatched if warpType == "psfMatched" else self.dataRefList
        resultsGen3 = assembleTask.runQuantum(self.skyInfo, dataRefList)
        resultsGen2 = assembleTask.runDataRef(self.skyInfoGen2, warpRefList=self.gen2DataRefList)
        coaddGen2 = resultsGen2.coaddExposure
        coaddGen3 = resultsGen3.coaddExposure
        self.assertFloatsEqual(coaddGen2.image.array, coaddGen3.image.array)

    def testGen2Gen3Compatibility(self):
        config = MockAssembleCoaddConfig()
        config.validate()
        assembleTask = MockAssembleCoaddTask(config=config)
        self.checkGen2Gen3Compatibility(assembleTask)

    def testPsfMatchedGen2Gen3Compatibility(self):
        config = MockAssembleCoaddConfig(warpType="psfMatched")
        config.validate()
        assembleTask = MockAssembleCoaddTask(config=config)
        self.checkGen2Gen3Compatibility(assembleTask, warpType="psfMatched")

    def testSafeClipGen2Gen3Compatibility(self):
        config = MockSafeClipAssembleCoaddConfig()
        config.validate()
        assembleTask = MockSafeClipAssembleCoaddTask(config=config)
        self.checkGen2Gen3Compatibility(assembleTask)

    def testCompareWarpGen2Gen3Compatibility(self):
        config = MockCompareWarpAssembleCoaddConfig()
        config.validate()
        assembleTask = MockCompareWarpAssembleCoaddTask(config=config)
        self.checkGen2Gen3Compatibility(assembleTask)

    def testDcrGen2Gen3Compatibility(self):
        config = MockDcrAssembleCoaddConfig()
        config.validate()
        assembleTask = MockDcrAssembleCoaddTask(config=config)
        self.checkGen2Gen3Compatibility(assembleTask)

    def testInputMapGen3(self):
        import numpy as np

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
