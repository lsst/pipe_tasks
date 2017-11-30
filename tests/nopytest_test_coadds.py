#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008-2017 AURA/LSST.
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
Test the basic mechanics of coaddition, coadd processing, and forced photometry.

In this test, we build a mock calexps using perfectly knowns WCSs, with the only sources
being stars created from a perfectly known PSF, then coadd them, process the coadd (using
the new measurement framework in meas_base), and then run forced photometry (again, using
the new forced measurement tasks in meas_base).

We do not check that the results of this processing is exactly what we'd expect, except in
some cases where it's easy and/or particularly important to do so (e.g. CoaddPsf); we mostly
just check that everything runs, and that the results make enough sense to let us proceed
to the next step.

NOTE: if this test fails with what looks like a failure to load a FITS file, try changing
the REUSE_DATAREPO variable below to False, as sometimes this error message indicates a
different problem that's revealed when we're not trying to cache the mock data between
tests (but set REUSE_DATAREPO back to True when done debugging, or this test will be very
slow).

WARNING: This test should not be run with other tests using pytest, and should
not be discoverable automatically by pytest. The reason for this is that the
tests rely on 200 MB of data generated on module load, using a single
directory visible to all the tests. When run in parallel with pytest-xdist
this data will be created for every sub-process, leading to excessive disk
usage, excessive test execution times and possible failure.
"""
from __future__ import division, print_function, absolute_import
from builtins import zip
from past.builtins import basestring

import unittest
import shutil
import os
import numbers
from collections import Iterable

import lsst.utils.tests
import lsst.afw.math
import lsst.afw.geom
import lsst.afw.image
import lsst.afw.table.io
import lsst.afw.table.testUtils
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
        # TODO: pybind11 remvoe `immediate=True` once DM-9112 is resolved
        image = butler.get('calexp', Id, immediate=True)
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
    task = lsst.meas.base.ForcedPhotCcdTask(config=config, butler=butler)
    task.writeSchemas(butler)
    runTaskOnCcds(butler, task)


class CoaddsTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        if not haveMeasBase:
            raise unittest.SkipTest("meas_base could not be imported; skipping this test")
        self.mocksTask = lsst.pipe.tasks.mocks.MockCoaddTask()
        self.butler = lsst.daf.persistence.Butler(DATAREPO_ROOT)
        self.coaddNameList = ["Coadd", "CoaddPsfMatched"]
        self.warpNameList = ["Coadd_directWarp", "Coadd_psfMatchedWarp"]

    def tearDown(self):
        del self.mocksTask
        del self.butler

    def testMaskPlanesExist(self):
        # Get the dataId for each calexp in the repository
        calexpDataIds = getCalexpIds(self.butler)
        # Loop over each Id and verify the mask planes were added
        for ID in calexpDataIds:
            image = self.butler.get('calexp', ID)
            mask = image.getMaskedImage().getMask()
            self.assertIn('CROSSTALK', list(mask.getMaskPlaneDict().keys()))
            self.assertIn('NOT_DEBLENDED', list(mask.getMaskPlaneDict().keys()))

    def comparePsfs(self, a, b):
        if a is None and b is None:
            return
        ak = a.getKernel()
        bk = b.getKernel()
        self.assertEqual(type(ak), type(bk))
        self.assertEqual(ak.getDimensions(), bk.getDimensions())
        self.assertEqual(ak.getNKernelParameters(), ak.getNKernelParameters())
        self.assertEqual(ak.getNSpatialParameters(), ak.getNSpatialParameters())
        for aFuncParams, bFuncParams in zip(ak.getSpatialParameters(), bk.getSpatialParameters()):
            for aParam, bParam in zip(aFuncParams, bFuncParams):
                self.assertEqual(aParam, bParam)

    @unittest.skip("Remove test until DM-5174 is complete")
    def testMasksRemoved(self):
        for dataProduct in self.coaddNameList:
            image = self.butler.get(self.mocksTask.config.coaddName + dataProduct + "_mock",
                                    {'filter': 'r', 'tract': 0, 'patch': '0,0'})
            keys = image.getMaskedImage().getMask().getMaskPlaneDict().keys()
            self.assertNotIn('CROSSTALK', keys)
            self.assertNotIn('NOT_DEBLENDED', keys)

    def testTempExpInputs(self, tract=0):
        skyMap = self.butler.get(self.mocksTask.config.coaddName + "Coadd_skyMap", immediate=True)
        tractInfo = skyMap[tract]
        for dataProduct in self.warpNameList:
            for visit, obsVisitDict in getObsDict(self.butler, tract).items():
                foundOneTempExp = False
                for patchRef in self.mocksTask.iterPatchRefs(self.butler, tractInfo):
                    try:
                        tempExp = patchRef.get(self.mocksTask.config.coaddName + dataProduct, visit=visit,
                                               immediate=True)
                        foundOneTempExp = True
                    except:
                        continue
                    self.assertEqual(tractInfo.getWcs(), tempExp.getWcs())
                    coaddInputs = tempExp.getInfo().getCoaddInputs()
                    self.assertEqual(len(coaddInputs.visits), 1)
                    visitRecord = coaddInputs.visits[0]
                    self.assertEqual(visitRecord.getWcs(), tempExp.getWcs())
                    self.assertEqual(visitRecord.getBBox(), tempExp.getBBox())
                    self.assertGreater(len(coaddInputs.ccds), 0)
                    ccdKey = coaddInputs.ccds.getSchema().find("ccd").key
                    for ccdRecord in coaddInputs.ccds:
                        ccd = ccdRecord.getI(ccdKey)
                        obsRecord = obsVisitDict[ccd]
                        self.assertEqual(obsRecord.getId(), ccdRecord.getId())
                        self.assertEqual(obsRecord.getWcs(), ccdRecord.getWcs())
                        self.assertEqual(obsRecord.getBBox(), ccdRecord.getBBox())
                        self.comparePsfs(obsRecord.getPsf(), ccdRecord.getPsf())
                self.assertTrue(foundOneTempExp)

    def testCoaddInputs(self, tract=0):
        skyMap = self.butler.get(self.mocksTask.config.coaddName + "Coadd_skyMap", immediate=True)
        tractInfo = skyMap[tract]
        obsCatalog = self.butler.get("observations", tract=tract, immediate=True)
        for patchRef in self.mocksTask.iterPatchRefs(self.butler, tractInfo):
            for dataProduct in self.coaddNameList:
                coaddExp = patchRef.get(self.mocksTask.config.coaddName + dataProduct, immediate=True)
                self.assertEqual(tractInfo.getWcs(), coaddExp.getWcs())
                coaddInputs = coaddExp.getInfo().getCoaddInputs()
                try:
                    ccdVisitKey = coaddInputs.ccds.getSchema().find("visit").key
                except:
                    print(patchRef.dataId)
                    print(coaddInputs.ccds.getSchema())
                    raise
                for ccdRecord in coaddInputs.ccds:
                    obsRecord = obsCatalog.find(ccdRecord.getId())
                    self.assertEqual(obsRecord.getId(), ccdRecord.getId())
                    self.assertEqual(obsRecord.getWcs(), ccdRecord.getWcs())
                    self.assertEqual(obsRecord.getBBox(), ccdRecord.getBBox())
                    self.assertEqual(obsRecord.get("filter"), ccdRecord.get("filter"))
                    self.comparePsfs(obsRecord.getPsf(), ccdRecord.getPsf())
                    self.assertIsNotNone(coaddInputs.visits.find(ccdRecord.getL(ccdVisitKey)))
                for visitRecord in coaddInputs.visits:
                    nCcds = len([ccdRecord for ccdRecord in coaddInputs.ccds
                                 if ccdRecord.getL(ccdVisitKey) == visitRecord.getId()])
                    self.assertGreaterEqual(nCcds, 1)
                    self.assertLessEqual(nCcds, 2)

    def testPsfInstallation(self, tract=0):
        skyMap = self.butler.get(self.mocksTask.config.coaddName + "Coadd_skyMap", immediate=True)
        tractInfo = skyMap[tract]
        for patchRef in self.mocksTask.iterPatchRefs(self.butler, tractInfo):
            coaddExp = patchRef.get(self.mocksTask.config.coaddName + "Coadd", immediate=True)
            ccdCat = coaddExp.getInfo().getCoaddInputs().ccds
            savedPsf = coaddExp.getPsf()
            newPsf = lsst.meas.algorithms.CoaddPsf(ccdCat, coaddExp.getWcs())
            self.assertEqual(savedPsf.getComponentCount(), len(ccdCat))
            self.assertEqual(newPsf.getComponentCount(), len(ccdCat))
            for n, record in enumerate(ccdCat):
                self.assertIs(savedPsf.getPsf(n), record.getPsf())
                self.assertIs(newPsf.getPsf(n), record.getPsf())
                self.assertIs(savedPsf.getWcs(n), record.getWcs())
                self.assertIs(newPsf.getWcs(n), record.getWcs())
                self.assertEqual(savedPsf.getBBox(n), record.getBBox())
                self.assertEqual(newPsf.getBBox(n), record.getBBox())

    def testCoaddPsf(self, tract=0):
        """Test that stars on the coadd are well represented by the attached PSF

        in both direct and PSF-matched coadds. The attached PSF is a "CoaddPsf"
        for direct coadds and a Model Psf for PSF-matched Coadds
        """
        skyMap = self.butler.get(self.mocksTask.config.coaddName + "Coadd_skyMap", immediate=True)
        tractInfo = skyMap[tract]
        # Start by finding objects that never appeared on the edge of an image
        simSrcCat = self.butler.get("simsrc", tract=tract, immediate=True)
        simSrcSchema = simSrcCat.getSchema()
        objectIdKey = simSrcSchema.find("objectId").key
        centroidInBBoxKey = simSrcSchema.find("centroidInBBox").key
        partialOverlapKey = simSrcSchema.find("partialOverlap").key
        simSrcByObject = {}
        for simSrcRecord in simSrcCat:
            simSrcByObject.setdefault(simSrcRecord.getL(objectIdKey), []).append(simSrcRecord)
        pureObjectIds = set()  # set will contain objects that never appear on edges
        for objectId, simSrcRecords in simSrcByObject.items():
            inAnyImages = False
            for simSrcRecord in simSrcRecords:
                if simSrcRecord.getFlag(centroidInBBoxKey):
                    if simSrcRecord.getFlag(partialOverlapKey):
                        break
                    inAnyImages = True
            else:  # only get here if we didn't break
                if inAnyImages:
                    pureObjectIds.add(objectId)

        truthCatalog = self.butler.get("truth", tract=tract, immediate=True)
        truthCatalog.sort()
        for dataProduct in self.coaddNameList:
            nTested = 0
            for patchRef in self.mocksTask.iterPatchRefs(self.butler, tractInfo):
                coaddExp = patchRef.get(self.mocksTask.config.coaddName + dataProduct, immediate=True)
                coaddWcs = coaddExp.getWcs()
                coaddPsf = coaddExp.getPsf()
                coaddBBox = lsst.afw.geom.Box2D(coaddExp.getBBox())
                for objectId in pureObjectIds:
                    truthRecord = truthCatalog.find(objectId)
                    position = coaddWcs.skyToPixel(truthRecord.getCoord())
                    if not coaddBBox.contains(position):
                        continue
                    try:
                        psfImage = coaddPsf.computeImage(position)
                    except:
                        continue
                    psfImageBBox = psfImage.getBBox()
                    if not coaddExp.getBBox().contains(psfImageBBox):
                        continue
                    starImage = lsst.afw.image.ImageF(coaddExp.getMaskedImage().getImage(),
                                                      psfImageBBox).convertD()
                    starImage /= starImage.getArray().sum()
                    psfImage /= psfImage.getArray().sum()
                    residuals = lsst.afw.image.ImageD(starImage, True)
                    residuals -= psfImage
                    self.assertFloatsAlmostEqual(starImage.getArray(), psfImage.getArray(),
                                                 rtol=1E-3, atol=1E-2)
                    nTested += 1
        if nTested == 0:
            print("WARNING: CoaddPsf test inconclusive (this can occur randomly, but very rarely; "
                  "first try running the test again)")

    def testSchemaConsistency(self):
        """Test that _schema catalogs are consistent with the data catalogs.
        """
        det_schema = self.butler.get("deepCoadd_det_schema").schema
        meas_schema = self.butler.get("deepCoadd_meas_schema").schema
        mergeDet_schema = self.butler.get("deepCoadd_mergeDet_schema").schema
        ref_schema = self.butler.get("deepCoadd_ref_schema").schema
        coadd_forced_schema = self.butler.get("deepCoadd_forced_src_schema").schema
        ccd_forced_schema = self.butler.get("forced_src_schema").schema
        patchList = ['0,0', '0,1', '1,0', '1,1']
        for patch in patchList:
            det = self.butler.get("deepCoadd_det", filter='r', tract=0, patch=patch)
            self.assertSchemasEqual(det.schema, det_schema)
            mergeDet = self.butler.get("deepCoadd_mergeDet", filter='r', tract=0, patch=patch)
            self.assertSchemasEqual(mergeDet.schema, mergeDet_schema)
            meas = self.butler.get("deepCoadd_meas", filter='r', tract=0, patch=patch)
            self.assertSchemasEqual(meas.schema, meas_schema)
            ref = self.butler.get("deepCoadd_ref", filter='r', tract=0, patch=patch)
            self.assertSchemasEqual(ref.schema, ref_schema)
            coadd_forced_src = self.butler.get("deepCoadd_forced_src", filter='r', tract=0, patch=patch)
            self.assertSchemasEqual(coadd_forced_src.schema, coadd_forced_schema)
        for visit, obsVisitDict in getObsDict(self.butler, 0).items():
            for ccd in obsVisitDict:
                ccd_forced_src = self.butler.get("forced_src", tract=0, visit=visit, ccd=ccd)
                self.assertSchemasEqual(ccd_forced_src.schema, ccd_forced_schema)

    def testAlgMetadataOutput(self):
        """Test to see if algMetadata is persisted correctly from MeasureMergedCoaddSourcesTask.

        This test fails with a NotFoundError if the algorithm metadata is not persisted"""
        patchList = ['0,0', '0,1', '1,0', '1,1']
        for patch in patchList:
            cat = self.butler.get("deepCoadd_meas", filter='r', tract=0, patch=patch)
            meta = cat.getTable().getMetadata()
            for circApertureFluxRadius in meta.get('base_CircularApertureFlux_radii'):
                self.assertIsInstance(circApertureFluxRadius, numbers.Number)
            # Each time the run method of a measurement task is executed, algorithm metadata is appended
            # to the algorithm metadata object. Depending on how many times a measurement task is run,
            # a metadata entry may be a single value or multiple values, this test ensures that in either
            # case the value can properly be extracted and compared.

            def ensureIterable(x):
                if isinstance(x, Iterable) and not isinstance(x, basestring):
                    return x
                return [x]
            for nOffset in ensureIterable(meta.get('NOISE_OFFSET')):
                self.assertIsInstance(nOffset, numbers.Number)
            for noiseSrc in ensureIterable(meta.get('NOISE_SOURCE')):
                self.assertEqual(noiseSrc, 'measure')
            for noiseExpID in ensureIterable(meta.get('NOISE_EXPOSURE_ID')):
                self.assertIsInstance(noiseExpID, numbers.Number)
            noiseSeedMul = meta.get('NOISE_SEED_MULTIPLIER')
            for noiseSeedMul in ensureIterable(meta.get('NOISE_SEED_MULTIPLIER')):
                self.assertIsInstance(noiseSeedMul, numbers.Number)

    def testForcedIdNames(self):
        """Test that forced photometry ID fields are named as we expect
        (DM-8210).

        Specifically, coadd forced photometry should have only "id" and "parent"
        fields, while CCD forced photometry should have those, "objectId", and
        "parentObjectId".
        """
        coaddSchema = self.butler.get("deepCoadd_forced_src_schema", immediate=True).schema
        self.assertIn("id", coaddSchema)
        self.assertIn("parent", coaddSchema)
        self.assertNotIn("objectId", coaddSchema)
        self.assertNotIn("parentObjectId", coaddSchema)
        ccdSchema = self.butler.get("forced_src_schema", immediate=True).schema
        self.assertIn("id", ccdSchema)
        self.assertIn("parent", ccdSchema)
        self.assertIn("objectId", ccdSchema)
        self.assertIn("parentObjectId", ccdSchema)


class AssembleCoaddTestCase(lsst.utils.tests.TestCase):

    def testSafeClipConfig(self):
        # Test for DM-4797: ensure that AssembleCoaddConfig.setDefaults() is
        # run when SafeClipAssembleCoaddConfig.setDefaults() is run. This
        # simply sets the default value for badMaskPlanes.
        self.assertEqual(AssembleCoaddConfig().badMaskPlanes, SafeClipAssembleCoaddConfig().badMaskPlanes)


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    import sys
    setup_module(sys.modules[__name__])
    unittest.main()
