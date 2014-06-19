#!/usr/bin/env python
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
import numpy
import shutil
import os
import sys

import lsst.utils.tests as utilsTests
import lsst.afw.math
import lsst.afw.geom
import lsst.afw.image
import lsst.afw.table.io
import lsst.meas.algorithms
import lsst.pipe.tasks.mocks
import lsst.daf.persistence

try:
    REUSE_DATAREPO
except NameError:                       # only set variables on initial import
    REUSE_DATAREPO = True      # If mocks are found (for each test), they will be used instead of regenerated
    CLEANUP_DATAREPO = True    # Delete mocks after all tests (if REUSE_DATAREPO) or after each one (else).
    DATAREPO_ROOT = "testCoaddInputs-data"

class CoaddInputsTestCase(unittest.TestCase):
    """A test case for CoaddInputsTask."""

    def assertClose(self, a, b, rtol=1E-5, atol=1E-8):
        self.assert_(numpy.allclose(a, b, rtol=rtol, atol=atol), "\n%s\n!=\n%s" % (a, b))

    def assertNotClose(self, a, b, rtol=1E-5, atol=1E-8):
        self.assert_(not numpy.allclose(a, b, rtol=rtol, atol=atol), "\n%s\n==\n%s" % (a, b))

    def setUp(self):
        self.task = lsst.pipe.tasks.mocks.MockCoaddTask()
        if REUSE_DATAREPO and os.path.exists(os.path.join(DATAREPO_ROOT, "_mapper")):
            self.butler = lsst.daf.persistence.Butler(DATAREPO_ROOT)
        else:
            self.butler = lsst.pipe.tasks.mocks.makeDataRepo(DATAREPO_ROOT)

            self.task.buildAllInputs(self.butler)
            self.task.buildCoadd(self.butler)
            self.task.buildMockCoadd(self.butler)

    def tearDown(self):
        if CLEANUP_DATAREPO and not REUSE_DATAREPO:
            shutil.rmtree(DATAREPO_ROOT)
        del self.task
        del self.butler

    def getObsDict(self, tract=0):
        catalog = self.butler.get("observations", tract=tract, immediate=True)
        visitKey = catalog.getSchema().find("visit").key
        ccdKey = catalog.getSchema().find("ccd").key
        obsDict = {}
        for record in catalog:
            visit = record.getI(visitKey)
            ccd = record.getI(ccdKey)
            obsDict.setdefault(visit, {})[ccd] = record
        return obsDict

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

    def testTempExpInputs(self, tract=0):
        skyMap = self.butler.get(self.task.config.coaddName + "Coadd_skyMap", immediate=True)
        tractInfo = skyMap[tract]
        for visit, obsVisitDict in self.getObsDict(tract).iteritems():
            foundOneTempExp = False
            for patchRef in self.task.iterPatchRefs(self.butler, tractInfo):
                try:
                    tempExp = patchRef.get(self.task.config.coaddName + "Coadd_tempExp", visit=visit,
                                           immediate=True)
                    foundOneTempExp = True
                except:
                    continue
                self.assertEqual(tractInfo.getWcs(), tempExp.getWcs())
                coaddInputs = tempExp.getInfo().getCoaddInputs()
                self.assertEqual(len(coaddInputs.visits), 1)
                visitRecord = coaddInputs.visits[0]
                self.assertEqual(visitRecord.getWcs(), tempExp.getWcs())
                self.assertEqual(visitRecord.getBBox(), tempExp.getBBox(lsst.afw.image.PARENT))
                self.assert_(len(coaddInputs.ccds) > 0)
                ccdKey = coaddInputs.ccds.getSchema().find("ccd").key
                for ccdRecord in coaddInputs.ccds:
                    ccd = ccdRecord.getI(ccdKey)
                    obsRecord = obsVisitDict[ccd]
                    self.assertEqual(obsRecord.getId(), ccdRecord.getId())
                    self.assertEqual(obsRecord.getWcs(), ccdRecord.getWcs())
                    self.assertEqual(obsRecord.getBBox(), ccdRecord.getBBox())
                    self.comparePsfs(obsRecord.getPsf(), ccdRecord.getPsf())
            self.assert_(foundOneTempExp)

    def testCoaddInputs(self, tract=0):
        skyMap = self.butler.get(self.task.config.coaddName + "Coadd_skyMap", immediate=True)
        tractInfo = skyMap[tract]
        obsCatalog = self.butler.get("observations", tract=tract, immediate=True)
        for patchRef in self.task.iterPatchRefs(self.butler, tractInfo):
            coaddExp = patchRef.get(self.task.config.coaddName + "Coadd", immediate=True)
            self.assertEqual(tractInfo.getWcs(), coaddExp.getWcs())
            coaddInputs = coaddExp.getInfo().getCoaddInputs()
            try:
                ccdVisitKey = coaddInputs.ccds.getSchema().find("visit").key
            except:
                print patchRef.dataId
                print coaddInputs.ccds.getSchema()
                raise
            for ccdRecord in coaddInputs.ccds:
                obsRecord = obsCatalog.find(ccdRecord.getId())
                self.assertEqual(obsRecord.getId(), ccdRecord.getId())
                self.assertEqual(obsRecord.getWcs(), ccdRecord.getWcs())
                self.assertEqual(obsRecord.getBBox(), ccdRecord.getBBox())
                self.comparePsfs(obsRecord.getPsf(), ccdRecord.getPsf())
                self.assert_(coaddInputs.visits.find(ccdRecord.getL(ccdVisitKey)) is not None)
            for visitRecord in coaddInputs.visits:
                nCcds = len([ccdRecord for ccdRecord in coaddInputs.ccds
                             if ccdRecord.getL(ccdVisitKey) == visitRecord.getId()])
                self.assert_(nCcds >= 1)
                self.assert_(nCcds <= 2)

    def testPsfInstallation(self, tract=0):
        skyMap = self.butler.get(self.task.config.coaddName + "Coadd_skyMap", immediate=True)
        tractInfo = skyMap[tract]
        for patchRef in self.task.iterPatchRefs(self.butler, tractInfo):
            coaddExp = patchRef.get(self.task.config.coaddName + "Coadd", immediate=True)
            ccdCat = coaddExp.getInfo().getCoaddInputs().ccds
            savedPsf = lsst.meas.algorithms.CoaddPsf.swigConvert(coaddExp.getPsf())
            newPsf = lsst.meas.algorithms.CoaddPsf(ccdCat, coaddExp.getWcs())
            self.assertEqual(savedPsf.getComponentCount(), len(ccdCat))
            self.assertEqual(newPsf.getComponentCount(), len(ccdCat))
            for n, record in enumerate(ccdCat):
                self.assert_(lsst.afw.table.io.comparePersistablePtrs(savedPsf.getPsf(n), record.getPsf()))
                self.assert_(lsst.afw.table.io.comparePersistablePtrs(newPsf.getPsf(n), record.getPsf()))
                self.assert_(lsst.afw.table.io.comparePersistablePtrs(savedPsf.getWcs(n), record.getWcs()))
                self.assert_(lsst.afw.table.io.comparePersistablePtrs(newPsf.getWcs(n), record.getWcs()))
                self.assertEqual(savedPsf.getBBox(n), record.getBBox())
                self.assertEqual(newPsf.getBBox(n), record.getBBox())

    def testCoaddPsf(self, tract=0):
        skyMap = self.butler.get(self.task.config.coaddName + "Coadd_skyMap", immediate=True)
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
        pureObjectIds = set() # set will contain objects that never appear on edges
        for objectId, simSrcRecords in simSrcByObject.iteritems():
            inAnyImages = False
            for simSrcRecord in simSrcRecords:
                if simSrcRecord.getFlag(centroidInBBoxKey):
                    if simSrcRecord.getFlag(partialOverlapKey):
                        break
                    inAnyImages = True
            else: # only get here if we didn't break
                if inAnyImages:
                    pureObjectIds.add(objectId)

        truthCatalog = self.butler.get("truth", tract=tract, immediate=True)
        truthCatalog.sort()
        nTested = 0
        for patchRef in self.task.iterPatchRefs(self.butler, tractInfo):
            coaddExp = patchRef.get(self.task.config.coaddName + "Coadd", immediate=True)
            coaddWcs = coaddExp.getWcs()
            coaddPsf = coaddExp.getPsf()
            coaddBBox = lsst.afw.geom.Box2D(coaddExp.getBBox(lsst.afw.image.PARENT))
            for objectId in pureObjectIds:
                truthRecord = truthCatalog.find(objectId)
                position = coaddWcs.skyToPixel(truthRecord.getCoord())
                if not coaddBBox.contains(position):
                    continue
                try:
                    psfImage = coaddPsf.computeImage(position)
                except:
                    continue
                psfImageBBox = psfImage.getBBox(lsst.afw.image.PARENT)
                if not coaddExp.getBBox(lsst.afw.image.PARENT).contains(psfImageBBox):
                    continue
                starImage = lsst.afw.image.ImageF(coaddExp.getMaskedImage().getImage(),
                                                  psfImageBBox, lsst.afw.image.PARENT).convertD()
                starImage /= starImage.getArray().sum()
                psfImage /= psfImage.getArray().sum()
                residuals = lsst.afw.image.ImageD(starImage, True)
                residuals -= psfImage

                self.assertClose(starImage.getArray(), psfImage.getArray(), rtol=1E-3, atol=1E-2)
                nTested += 1
        if nTested == 0:
            print ("WARNING: CoaddPsf test inconclusive (this can occur randomly, but very rarely; "
                   "first try running the test again)")

def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(CoaddInputsTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    status = utilsTests.run(suite(), False)
    if os.path.exists(DATAREPO_ROOT):
        if CLEANUP_DATAREPO:
            if status == 0:
                shutil.rmtree(DATAREPO_ROOT)
            else:
                print >> sys.stderr, "Tests failed; not cleaning up %s" % os.path.abspath(DATAREPO_ROOT)
        else:
            print >> sys.stderr, "CLEANUP_DATAREPO is False; not cleaning up %s" % \
                os.path.abspath(DATAREPO_ROOT)
        if shouldExit:
            sys.exit(status)

    return status

if __name__ == "__main__":
    run(True)
