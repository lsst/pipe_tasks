import time
import unittest

import lsst.utils.tests as utilsTests
import lsst.pipe.tasks as pipeTasks
import lsst.pipe.tasks.calibrate

class CalibrateTestCase(unittest.TestCase):
    conf = pipeTasks.calibrate.CalibrateConfig()
    conf.validate()
    # Demonstrate typo
    conf.doPhotoCal = False
    conf.validate()
    conf.doComputeApCorr = False
    conf.doPsf = True
    conf.measurement.doApplyApCorr = False
    conf.validate()
    conf.doAstrometry = False
    conf.validate()












def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(CalibrateTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
