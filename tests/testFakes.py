#!/usr/bin/env python
from __future__ import division, print_function, absolute_import
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

import lsst.utils
import lsst.afw.image
import lsst.utils.tests
import lsst.pipe.tasks.fakes as fakes


class TrialFakeSourcesConfig(fakes.BaseFakeSourcesConfig):
    pass


class TrialFakeSourcesTask(fakes.BaseFakeSourcesTask):
    ConfigClass = TrialFakeSourcesConfig

    def __init__(self, **kwargs):
        fakes.BaseFakeSourcesTask.__init__(self, **kwargs)

    def run(self, exposure, background):
        pass


class TestFakes(lsst.utils.tests.TestCase):

    def testFakeMaskAdded(self):
        '''
        Test that the FAKE mask plane does not exist, that it is added, and the instance's
        bitmask points to the correct plane
        '''
        maskPlaneName = "FAKE"
        maskKeysBefore = list(lsst.afw.image.Mask().getMaskPlaneDict())
        trialInstance = TrialFakeSourcesTask()
        maskKeysAfter = list(lsst.afw.image.Mask().getMaskPlaneDict())
        maskPlaneName = trialInstance.config.maskPlaneName
        self.assertNotIn(maskPlaneName, maskKeysBefore)
        self.assertIn(maskPlaneName, maskKeysAfter)
        self.assertEqual(trialInstance.bitmask, lsst.afw.image.Mask[lsst.afw.image.MaskPixel]\
                         .getPlaneBitMask(maskPlaneName))
        lsst.afw.image.Mask[lsst.afw.image.MaskPixel].removeMaskPlane(maskPlaneName)
        del trialInstance

    @unittest.skip()
    def testFakeMasksed(self):
        '''
        Test that if the FAKE mask plane exists, it is used, and that the instance's bitmask
        points to the correct plane
        '''
        maskPlaneName = "FAKE"
        lsst.afw.image.Mask[lsst.afw.image.MaskPixel].addMaskPlane(maskPlaneName)
        maskKeysBefore = list(lsst.afw.image.Mask().getMaskPlaneDict())
        self.assertIn(maskPlaneName, maskKeysBefore)
        maskPlaneBitMask = lsst.afw.image.Mask[lsst.afw.image.MaskPixel].getPlaneBitMask(maskPlaneName)
        trialInstance = TrialFakeSourcesTask()
        self.assertEqual(maskPlaneBitMask, trialInstance.bitmask)
        lsst.afw.image.Mask[lsst.afw.image.MaskPixel].removeMaskPlane(maskPlaneName)
        maskKeysAfter = list(lsst.afw.image.Mask().getMaskPlaneDict())
        self.assertNotIn(maskPlaneName, maskKeysAfter)
        del trialInstance


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
