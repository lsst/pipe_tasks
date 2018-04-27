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

import numpy as np
import os
import shutil
import tempfile
import unittest

from collections import namedtuple

import lsst.utils.tests
import lsst.afw.geom as afwGeom

from lsst.pipe.tasks.processCcd import ProcessCcdTask
from lsst.pipe.tasks.fakes import BaseFakeSourcesConfig, BaseFakeSourcesTask

obsTestDir = lsst.utils.getPackageDir('obs_test')
InputDir = os.path.join(obsTestDir, "data", "input")

OutputName = None  # specify a name (as a string) to save the output repository

positionTuple = namedtuple("positionTuple", "y x")


class FakeSourcesTestConfig(BaseFakeSourcesConfig):
    pass


class FakeSourcesTestTask(BaseFakeSourcesTask):
    '''
    A task to insert fake objects into test data to verify the hooks for the
    fake object pipeline work.
    '''

    ConfigClass = FakeSourcesTestConfig
    _DefaultName = "fakeSourcesTest"

    # Ground truth position and intensities for the fake sources
    fakeSources = [(positionTuple(800, 435), 11342),
                   (positionTuple(400, 350), 18235),
                   (positionTuple(1834, 379), 13574),
                   (positionTuple(1234, 642), 12456)]

    def __init__(self, **kwargs):
        BaseFakeSourcesTask.__init__(self, **kwargs)

    def run(self, exposure, background):
        if not exposure.hasPsf():
            raise RuntimeError("Exposure object must have a PSF")
        # Fetch objects from the exposure
        psf = exposure.getPsf()
        image = exposure.getMaskedImage().getImage()
        mask = exposure.getMaskedImage().getMask()
        variance = exposure.getMaskedImage().getVariance()

        y0 = image.getY0()
        x0 = image.getX0()

        # Bitplane to set corresponding to the FAKE bit
        fakeMaskValue = 2**mask.getMaskPlaneDict()['FAKE']

        # At each position create a star with the given intensity and add it
        # to the image.
        for pos, intensity in self.fakeSources:
            objArray, noiseArray = self.makeFakeStar(pos, intensity, psf)
            psfRad = int((objArray.shape[0]-1)/2.)
            yslice, xslice = slice(pos.y-psfRad-y0, pos.y+psfRad+y0+1),\
                slice(pos.x-psfRad-x0, pos.x+psfRad+x0+1)

            image.getArray()[yslice, xslice] += objArray
            mask.getArray()[yslice, xslice] += fakeMaskValue
            variance.getArray()[yslice, xslice] += noiseArray**2

    # make stars at a given position with a given intensity
    @staticmethod
    def makeFakeStar(position, intensity, psf):
        psfImage = psf.computeImage(afwGeom.Point2D(position.x, position.y)).getArray()
        psfImage *= intensity
        noise = np.random.normal(0, np.sqrt(abs(psfImage)))
        return psfImage + noise, noise


def getObsTestConfig(TaskClass):
    """Helper function to get a command-line task config customized by obs_test.

    This duplicates the config override code in pipe_base's ArgumentParser, but
    essentially in order to test it.
    """
    config = TaskClass.ConfigClass()
    filename = os.path.join(obsTestDir, "config",
                            TaskClass._DefaultName + ".py")
    if os.path.exists(filename):
        config.load(filename)
    return config


class FakeProcessingTestCase(lsst.utils.tests.TestCase):
    def testFakeProcessing(self):
        # Set the random seed for predictability
        np.random.seed(500)

        # Set ouput path and create a dataId
        outPath = tempfile.mkdtemp() if OutputName is None \
            else "{}-ProcessCcd".format(OutputName)
        dataId = dict(visit=1)
        dataIdStrList = ["{}={}".format(*item) for item in dataId.items()]
        mask = None
        maskPlaneName = "FAKE"

        try:
            # Set the configurations for running the fake object piepline
            processCcdConfig = getObsTestConfig(ProcessCcdTask)
            processCcdConfig.calibrate.doInsertFakes = True
            processCcdConfig.calibrate.insertFakes.retarget(FakeSourcesTestTask)

            # Run ProcessCcd
            pCcdResult = ProcessCcdTask.parseAndRun(
                args=[InputDir, "--output", outPath,
                      "--clobber-config", "--doraise", "--id"] +
                dataIdStrList,
                doReturnResults=True, config=processCcdConfig)

            # Check the Catalog contains properly measured fake sources
            sourceCat = pCcdResult.resultList[0].result.calibRes.sourceCat
            self.checkSourceCatalog(sourceCat)

            exposure = pCcdResult.resultList[0].result.calibRes.exposure
            mask = exposure.getMaskedImage().getMask()
            maskBit = 2**mask.getMaskPlaneDict()[maskPlaneName]
            fakeMask = np.bitwise_and(mask.getArray(), maskBit)
            self.assertGreater(fakeMask.sum(), 0)

        # Clean up temporary directories if needed
        finally:
            if mask:
                # Remove FAKE so as not to contaminate later tests
                lsst.afw.image.Mask[lsst.afw.image.MaskPixel].removeMaskPlane(maskPlaneName)
            if OutputName is None:
                shutil.rmtree(outPath)
            else:
                message = "testFakeProcessing.py's output data saved to {}"
                print(message.format(OutputName))

    def checkSourceCatalog(self, srcCat, thresh=5):
        # Find the fake objects in the output source catalog, verify the
        # measured flux is plausable given ground truth
        fakeSourceCounter = 0
        for source in srcCat:
            srcX = source.getX()
            srcY = source.getY()
            for fakePos, fakeIntensity in FakeSourcesTestTask.fakeSources:
                distSq = (srcX - fakePos.x)**2 + (srcY - fakePos.y)**2
                if distSq <= thresh**2:
                    fluxDiff = abs(source.getPsfFlux() - fakeIntensity)
                    self.assertLessEqual(fluxDiff, 5*np.sqrt(fakeIntensity))
                    fakeSourceCounter += 1
        # Verify the correct number of fake sources were found
        self.assertEqual(fakeSourceCounter,
                         len(FakeSourcesTestTask.fakeSources))


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
