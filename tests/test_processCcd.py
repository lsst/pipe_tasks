#
# LSST Data Management System
# Copyright 2008-2016 AURA/LSST.
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
"""Test ProcessCcdTask and its immediate subtasks.

Run the task on one obs_test image and perform various checks on the results
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

import lsst.utils
import lsst.afw.geom as afwGeom
import lsst.utils.tests
from lsst.ip.isr import IsrTask  # we assume obs_test uses base IsrTask here; may change in future.
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask
from lsst.pipe.tasks.calibrate import CalibrateTask
from lsst.pipe.tasks.processCcd import ProcessCcdTask

obsTestDir = lsst.utils.getPackageDir('obs_test')
InputDir = os.path.join(obsTestDir, "data", "input")

OutputName = None  # specify a name (as a string) to save the output repository


def getObsTestConfig(TaskClass):
    """Helper function to get a command-line task config customized by obs_test.

    This duplicates the config override code in pipe_base's ArgumentParser, but
    essentially in order to test it.
    """
    config = TaskClass.ConfigClass()
    filename = os.path.join(obsTestDir, "config", TaskClass._DefaultName + ".py")
    if os.path.exists(filename):
        config.load(filename)
    return config


class ProcessCcdTestCase(lsst.utils.tests.TestCase):

    def testProcessCcd(self):
        """test ProcessCcdTask via parseAndRun (simulating the command line)

        This is intended as a sanity check of the task, not a detailed test of its sub-tasks. As such
        comparisons are intentionally loose, so as to allow evolution of the sub-tasks over time
        without breaking this test.
        """
        outPath = tempfile.mkdtemp() if OutputName is None else "{}-ProcessCcd".format(OutputName)
        try:
            dataId = dict(visit=1)
            dataIdStrList = ["%s=%s" % (key, val) for key, val in dataId.items()]
            fullResult = ProcessCcdTask.parseAndRun(
                args=[InputDir, "--output", outPath, "--clobber-config", "--doraise",
                      "-c", "charImage.doWriteExposure=True", "--id"] + dataIdStrList,
                doReturnResults=True
            )
            butler = fullResult.parsedCmd.butler
            self.assertEqual(len(fullResult.resultList), 1)
            runResult = fullResult.resultList[0]
            fullDataId = runResult.dataRef.dataId
            self.assertEqual(len(fullDataId), 2)
            self.assertEqual(fullDataId["visit"], dataId["visit"])
            self.assertEqual(fullDataId["filter"], "g")
            result = runResult.result

            icExpBackground = butler.get("icExpBackground", dataId, immediate=True)
            bg0Arr = icExpBackground.getImage().getArray()
            bgMean = bg0Arr.mean(dtype=np.float64)
            bgStdDev = bg0Arr.std(dtype=np.float64)

            icSrc = butler.get("icSrc", dataId)
            src = butler.get("src", dataId)

            # the following makes pyflakes linter happy and the code more robust
            oldImMean = None
            oldImStdDev = None
            oldVarMean = None
            oldVarStdDev = None
            oldPsfIxx = None
            oldPsfIyy = None
            oldPsfIxy = None

            for i, exposure in enumerate((butler.get("calexp", dataId), result.exposure)):
                self.assertEqual(exposure.getBBox(),
                                 afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(1018, 2000)))
                maskedImage = exposure.getMaskedImage()
                maskArr = maskedImage.getMask().getArray()
                numGoodPix = np.sum(maskArr == 0)

                imageArr = maskedImage.getImage().getArray()
                imMean = imageArr.mean(dtype=np.float64)
                imStdDev = imageArr.std(dtype=np.float64)
                varArr = maskedImage.getVariance().getArray()
                varMean = varArr.mean(dtype=np.float64)
                varStdDev = varArr.std(dtype=np.float64)

                psfShape = exposure.getPsf().computeShape()
                psfIxx = psfShape.getIxx()
                psfIyy = psfShape.getIyy()
                psfIxy = psfShape.getIxy()

                if i == 0:
                    print("\nMeasured results:")
                    print("background mean = %r, stdDev = %r" % (bgMean, bgStdDev))
                    print("len(icSrc) :", len(icSrc))
                    print("len(src) :", len(src))

                    print("numGoodPix =", numGoodPix)
                    print("image mean = %r, stdDev = %r" % (imMean, imStdDev))
                    print("variance mean = %r, stdDev = %r" % (varMean, varStdDev))
                    print("psf Ixx = %r, Iyy = %r, Ixy = %r" % (psfIxx, psfIyy, psfIxy))

                    self.assertEqual(len(icSrc), 28)
                    self.assertEqual(len(src), 186)

                    expectedPlaces = 7  # Tolerance for numerical comparisons
                    for name, var, val in [
                        ("bgMean", bgMean, 191.4862217336029),
                        ("bgStdDev", bgStdDev, 0.23986511599945562),
                        ("numGoodPix", numGoodPix, 1965471),
                        ("imMean", imMean, 1.1239582066430034),
                        ("imStdDev", imStdDev, 85.81319381115661),
                        ("varMean", varMean, 131.23984767404193),
                        ("varStdDev", varStdDev, 55.9802472085537),
                        ("psfIxx", psfIxx, 2.8540512421637554),
                        ("psfIyy", psfIyy, 2.1738662399061064),
                        ("psfIxy", psfIxy, 0.1439765855869371)
                    ]:
                        self.assertAlmostEqual(var, val, places=expectedPlaces, msg=name)

                else:
                    self.assertEqual(imMean, oldImMean)
                    self.assertEqual(imStdDev, oldImStdDev)
                    self.assertEqual(varMean, oldVarMean)
                    self.assertEqual(varStdDev, oldVarStdDev)
                    self.assertEqual(psfIxx, oldPsfIxx)
                    self.assertEqual(psfIyy, oldPsfIyy)
                    self.assertEqual(psfIxy, oldPsfIxy)

                oldImMean = imMean
                oldImStdDev = imStdDev
                oldVarMean = varMean
                oldVarStdDev = varStdDev
                oldPsfIxx = psfIxx
                oldPsfIyy = psfIyy
                oldPsfIxy = psfIxy

        finally:
            if OutputName is None:
                shutil.rmtree(outPath)
            else:
                print("testProcessCcd.py's output data saved to %r" % (OutputName,))

    def assertCatalogsEqual(self, catalog1, catalog2, skipCols=()):
        """Compare two Catalogs for equality.

        This should only be used in contexts where it's unlikely that the catalogs will be subtly different;
        instead of comparing all values we simply do a spot check of a few cells.

        This does not require that either catalog be contiguous (which is why we can't use column access).
        """
        self.assertEqual(catalog1.schema, catalog2.schema)
        self.assertEqual(len(catalog1), len(catalog2))
        d = catalog1.schema.extract("*")

        def fixNaN(x):
            if x != x:
                return "NaN"

        for record1, record2 in zip(catalog1, catalog2):
            for name, item in d.items():
                if name not in skipCols:
                    self.assertEqual(
                        fixNaN(record1.get(item.key)), fixNaN(record2.get(item.key)),
                        "{} != {} in field {}".format(record1.get(item.key), record2.get(item.key), name)
                    )

    def assertBackgroundListsEqual(self, bkg1, bkg2):
        """Compare two BackgroundLists for equality.

        This should only be used in contexts where it's unlikely that the catalogs will be subtly different;
        instead of comparing all values we simply do a spot check of a few cells.

        This does not require that either catalog be contiguous (which is why we can't use column access).
        """
        im1 = bkg1.getImage()
        im2 = bkg2.getImage()
        self.assertEqual(im1.getBBox(), im2.getBBox())
        self.assertImagesEqual(im1, im2)

    def testComponents(self):
        """Test that we can run the first-level subtasks of ProcessCcdTasks.

        This tests that we can run these subtasks from the command-line independently (they're all
        CmdLineTasks) as well as directly from Python (without giving them access to a Butler).

        Aside from verifying that no exceptions are raised, we simply tests that most persisted results are
        present and equivalent to both in-memory results.
        """
        outPath = tempfile.mkdtemp() if OutputName is None else "{}-Components".format(OutputName)
        # We'll use an input butler to get data for the tasks we call from Python, but we won't ever give it
        # to those tasks.
        inputButler = lsst.daf.persistence.Butler(InputDir)
        # Construct task instances we can use directly from Python
        isrTask = IsrTask(
            config=getObsTestConfig(IsrTask),
            name="isr2"
        )
        # If we ever enable astrometry and photocal in obs_test, we'll need to pass a refObjLoader to these
        # tasks.  To maintain the spirit of these tests, we'd ideally have a LoadReferenceObjectsTask class
        # that doesn't require a Butler.  If we don't, we should construct a butler-based on outside these
        # task constructors and pass the LoadReferenceObjectsTask instance to the task constructors.
        charImageTask = CharacterizeImageTask(
            config=getObsTestConfig(CharacterizeImageTask),
            name="charImage2"
        )
        calibrateTask = CalibrateTask(
            config=getObsTestConfig(CalibrateTask),
            name="calibrate2",
            icSourceSchema=charImageTask.schema
        )
        try:
            dataId = dict(visit=1)
            dataIdStrList = ["%s=%s" % (key, val) for key, val in dataId.items()]

            isrResult1 = IsrTask.parseAndRun(
                args=[InputDir, "--output", outPath, "--clobber-config", "--doraise", "--id"] + dataIdStrList,
                doReturnResults=True,
            )
            # We'll just use the butler to get the original image and calibration frames; it's not clear
            # extending the test coverage to include that is worth it.
            dataRef = inputButler.dataRef("raw", dataId=dataId)
            rawExposure = dataRef.get("raw", immediate=True)
            camera = dataRef.get("camera")
            isrData = isrTask.readIsrData(dataRef, rawExposure)
            isrResult2 = isrTask.run(
                rawExposure,
                bias=isrData.bias,
                linearizer=isrData.linearizer,
                flat=isrData.flat,
                defects=isrData.defects,
                fringes=isrData.fringes,
                bfKernel=isrData.bfKernel,
                camera=camera,
            )
            self.assertMaskedImagesEqual(
                isrResult1.parsedCmd.butler.get("postISRCCD", dataId, immediate=True).getMaskedImage(),
                isrResult1.resultList[0].result.exposure.getMaskedImage()
            )
            self.assertMaskedImagesEqual(
                isrResult2.exposure.getMaskedImage(),
                isrResult1.resultList[0].result.exposure.getMaskedImage()
            )

            icResult1 = CharacterizeImageTask.parseAndRun(
                args=[InputDir, "--output", outPath, "--clobber-config", "--doraise", "--id"] + dataIdStrList,
                doReturnResults=True,
            )
            icResult2 = charImageTask.run(isrResult2.exposure)
            self.assertMaskedImagesEqual(
                icResult1.parsedCmd.butler.get("icExp", dataId, immediate=True).getMaskedImage(),
                icResult1.resultList[0].result.exposure.getMaskedImage()
            )
            self.assertMaskedImagesEqual(
                icResult2.exposure.getMaskedImage(),
                icResult1.resultList[0].result.exposure.getMaskedImage()
            )
            self.assertCatalogsEqual(
                icResult1.parsedCmd.butler.get("icSrc", dataId, immediate=True),
                icResult1.resultList[0].result.sourceCat
            )
            self.assertCatalogsEqual(
                icResult2.sourceCat,
                icResult1.resultList[0].result.sourceCat,
                skipCols=("id", "parent")  # since we didn't want to pass in an ExposureIdInfo, IDs disagree
            )
            self.assertBackgroundListsEqual(
                icResult1.parsedCmd.butler.get("icExpBackground", dataId, immediate=True),
                icResult1.resultList[0].result.background
            )
            self.assertBackgroundListsEqual(
                icResult2.background,
                icResult1.resultList[0].result.background
            )

            calResult1 = CalibrateTask.parseAndRun(
                args=[InputDir, "--output", outPath, "--clobber-config", "--doraise", "--id"] + dataIdStrList,
                doReturnResults=True,
            )
            calResult2 = calibrateTask.run(
                icResult2.exposure,
                background=icResult2.background,
                icSourceCat=icResult2.sourceCat
            )
            self.assertMaskedImagesEqual(
                calResult1.parsedCmd.butler.get("calexp", dataId, immediate=True).getMaskedImage(),
                calResult1.resultList[0].result.exposure.getMaskedImage()
            )
            self.assertMaskedImagesEqual(
                calResult2.exposure.getMaskedImage(),
                calResult1.resultList[0].result.exposure.getMaskedImage()
            )
            self.assertCatalogsEqual(
                calResult1.parsedCmd.butler.get("src", dataId, immediate=True),
                calResult1.resultList[0].result.sourceCat
            )
            self.assertCatalogsEqual(
                calResult2.sourceCat,
                calResult1.resultList[0].result.sourceCat,
                skipCols=("id", "parent")
            )
            self.assertBackgroundListsEqual(
                calResult1.parsedCmd.butler.get("calexpBackground", dataId, immediate=True),
                calResult1.resultList[0].result.background
            )
            self.assertBackgroundListsEqual(
                calResult2.background,
                calResult1.resultList[0].result.background
            )

        finally:
            if OutputName is None:
                shutil.rmtree(outPath)
            else:
                print("testProcessCcd.py's output data saved to %r" % (OutputName,))


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
