#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008-2015 AURA/LSST.
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
# see <https://www.lsstcorp.org/LegalNotices/>.
#

"""
Test the basic mechanics of covariance propagation through coaddition, coadd processing,
and forced photometry.

In this test, we build a mock calexps using perfectly knowns WCSs, with the only sources
being stars created from a perfectly known PSF and then coadd them, propagating the
covariance matrix generated during the warping process all the way through to the final
coadd.
"""

import unittest
import shutil
import os
import numbers
from collections import Iterable
from types import StringTypes
import pdb
import numpy as np
import matplotlib.pyplot as plt

import lsst.utils.tests
import lsst.afw.math
import lsst.afw.geom
import lsst.afw.image
import lsst.afw.table.io
import lsst.meas.algorithms
import lsst.pipe.tasks.mocks
import lsst.daf.persistence

try:
    import lsst.meas.base
except ImportError:
    haveMeasBase = False
else:
    haveMeasBase = True

from lsst.pipe.tasks.assembleCoadd import AssembleCoaddConfig, SafeClipAssembleCoaddConfig
from lsst.pipe.tasks.multiBand import (DetectCoaddSourcesTask, MergeDetectionsTask,
                                       MeasureMergedCoaddSourcesTask, MergeMeasurementsTask)

DATAREPO_ROOT = os.path.join(os.path.dirname(__file__), ".tests", "testCoadds-data")

plt.ion()


def setup_module(module):
    lsst.utils.tests.init()

    if os.path.exists(DATAREPO_ROOT):
        print("Deleting existing repo: %r" % (DATAREPO_ROOT,))
        shutil.rmtree(DATAREPO_ROOT)

    if not haveMeasBase:
        raise unittest.SkipTest("meas_base could not be imported")

    # Create a task that creates simulated images and builds a coadd from them
    mocksTask = lsst.pipe.tasks.mocks.MockCoaddTask()

    # Create an instance of DetectCoaddSourcesTask to measure on the coadd.
    # There's no noise in these images, so we set a direct-value threshold,
    # and the background weighting (when using Approximate) to False

    detectConfig = DetectCoaddSourcesTask.ConfigClass()
    detectConfig.detection.thresholdType = "value"
    detectConfig.detection.thresholdValue = 0.01
    detectConfig.detection.background.weighting = False
    detectTask = DetectCoaddSourcesTask(config=detectConfig)

    butler = lsst.pipe.tasks.mocks.makeDataRepo(DATAREPO_ROOT)
    mocksTask.buildAllInputs(butler)

    addMaskPlanes(butler)
    mocksTask.buildCoadd(butler)
    mocksTask.buildMockCoadd(butler)
    detectTask.writeSchemas(butler)
    # Now run the seperate multiband tasks on the Coadd to make the reference
    # catalog for the forced photometry tests.
    runTaskOnPatches(butler, detectTask, mocksTask)

    mergeDetConfig = MergeDetectionsTask.ConfigClass()
    mergeDetConfig.priorityList = ['r', ]
    mergeDetTask = MergeDetectionsTask(config=mergeDetConfig, butler=butler)
    mergeDetTask.writeSchemas(butler)
    runTaskOnPatchList(butler, mergeDetTask, mocksTask)

    measMergedConfig = MeasureMergedCoaddSourcesTask.ConfigClass()
    measMergedConfig.measurement.slots.shape = "base_SdssShape"
    measMergedConfig.measurement.plugins['base_PixelFlags'].masksFpAnywhere = []
    measMergedConfig.propagateFlags.flags = {}  # Disable flag propagation: no flags to propagate
    measMergedConfig.doMatchSources = False  # We don't have a reference catalog available
    measMergedConfig.doApCorr = False
    measMergedTask = MeasureMergedCoaddSourcesTask(config=measMergedConfig, butler=butler)
    measMergedTask.writeSchemas(butler)
    runTaskOnPatches(butler, measMergedTask, mocksTask)

    mergeMeasConfig = MergeMeasurementsTask.ConfigClass()
    mergeMeasConfig.priorityList = ['r', ]
    mergeMeasTask = MergeMeasurementsTask(config=mergeMeasConfig, butler=butler)
    mergeMeasTask.writeSchemas(butler)
    runTaskOnPatchList(butler, mergeMeasTask, mocksTask)

    runForcedPhotCoaddTask(butler, mocksTask)
    runForcedPhotCcdTask(butler)


def getCalexpIds(butler, tract=0):
    catalog = butler.get("observations", tract=tract, immediate=True)
    return [{"visit": int(visit), "ccd": int(ccd)} for visit, ccd in zip(catalog["visit"], catalog["ccd"])]


def addMaskPlanes(butler):
    # Get the dataId for each calexp in the repository
    calexpDataIds = getCalexpIds(butler)
    # Loop over each of the calexp and add the CROSSTALK and NOT_DEBLENDED mask planes
    for Id in calexpDataIds:
        image = butler.get('calexp', Id)
        mask = image.getMaskedImage().getMask()
        mask.addMaskPlane("CROSSTALK")
        mask.addMaskPlane("NOT_DEBLENDED")
        butler.put(image, 'calexp', dataId=Id)


def runTaskOnPatches(butler, task, mocksTask, tract=0):
    skyMap = butler.get(mocksTask.config.coaddName + "Coadd_skyMap", immediate=True)
    tractInfo = skyMap[tract]
    for dataRef in mocksTask.iterPatchRefs(butler, tractInfo):
        task.run(dataRef)


def runTaskOnPatchList(butler, task, mocksTask, tract=0, rerun=None):
    skyMap = butler.get(mocksTask.config.coaddName + "Coadd_skyMap", immediate=True)
    tractInfo = skyMap[tract]
    for dataRef in mocksTask.iterPatchRefs(butler, tractInfo):
        task.run([dataRef])


def runTaskOnCcds(butler, task, tract=0):
    catalog = butler.get("observations", tract=tract, immediate=True)
    visitKey = catalog.getSchema().find("visit").key
    ccdKey = catalog.getSchema().find("ccd").key
    for record in catalog:
        dataRef = butler.dataRef("forced_src", tract=tract, visit=record.getI(visitKey),
                                 ccd=record.getI(ccdKey))
        task.run(dataRef)


def getObsDict(butler, tract=0):
    catalog = butler.get("observations", tract=tract, immediate=True)
    visitKey = catalog.getSchema().find("visit").key
    ccdKey = catalog.getSchema().find("ccd").key
    obsDict = {}
    for record in catalog:
        visit = record.getI(visitKey)
        ccd = record.getI(ccdKey)
        obsDict.setdefault(visit, {})[ccd] = record
    return obsDict


def runForcedPhotCoaddTask(butler, mocksTask):
    config = lsst.meas.base.ForcedPhotCoaddConfig()
    config.references.filter = 'r'
    task = lsst.meas.base.ForcedPhotCoaddTask(config=config, butler=butler)
    task.writeSchemas(butler)
    runTaskOnPatches(butler, task, mocksTask)


def runForcedPhotCcdTask(butler):
    config = lsst.meas.base.ForcedPhotCcdConfig()
    config.references.filter = 'r'
    # There is no reference flux for the mocks, so turn off ap corrections
    config.doApCorr = False
    task = lsst.meas.base.ForcedPhotCcdTask(config=config, butler=butler)
    runTaskOnCcds(butler, task)


class CoaddsTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        if not haveMeasBase:
            raise unittest.SkipTest("meas_base could not be imported; skipping this test")
        self.mocksTask = lsst.pipe.tasks.mocks.MockCoaddTask()
        self.butler = lsst.daf.persistence.Butler(DATAREPO_ROOT)

    def tearDown(self):
        del self.mocksTask
        del self.butler

    def comparePsfs(self, a, b):
        if a is None and b is None:
            return
        ak = lsst.meas.algorithms.KernelPsf.swigConvert(a).getKernel()
        bk = lsst.meas.algorithms.KernelPsf.swigConvert(b).getKernel()
        self.assertEqual(type(ak), type(bk))
        self.assertEqual(ak.getDimensions(), bk.getDimensions())
        self.assertEqual(ak.getNKernelParameters(), ak.getNKernelParameters())
        self.assertEqual(ak.getNSpatialParameters(), ak.getNSpatialParameters())
        for aFuncParams, bFuncParams in zip(ak.getSpatialParameters(), bk.getSpatialParameters()):
            for aParam, bParam in zip(aFuncParams, bFuncParams):
                self.assertEqual(aParam, bParam)


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    import sys
    setup_module(sys.modules[__name__])
    unittest.main()
