import unittest

import eups

from lsst.meas.algorithms import *
import lsst.pipe.tasks as pipeTasks
import lsst.pipe.tasks.astrometry as pipeTasksAstrom
import lsst.utils.tests as utilsTests
import lsst.afw.image   as afwImage
import lsst.pex.logging as pexLog

class TestForceWcs(unittest.TestCase):
    def setUp(self):
        eupsObj = eups.Eups()
        #ver = 'imsim-2011-08-01-0'
        # ver = 'sdss-2012-05-01-0'
        # ok, version, reason = eupsObj.setup('astrometry_net_data', versionName=ver)
        # if not ok:
        #     raise ValueError("Failed to setup astrometry_net_data version '%s': %s" % (ver, reason))
        # print 'Setup astrometry_net_data', ver
        mypath = eups.productDir("pipe_tasks")
        datapath = os.path.join(mypath, 'tests', 'astrometry_net_data', 't2155')
        eupsObj = eups.Eups(root=datapath)
        ok, version, reason = eupsObj.setup('astrometry_net_data')
        if not ok:
            raise ValueError("Need local version of astrometry_net_data (from path: %s): %s" %
                             (datapath, reason))

    def test1(self):
        #exposure = afwImage.ExposureF('mini-v85408556-fr-R23-S11.fits')
        #exposure = afwImage.ExposureF('../afwdata/ImSim/calexp/v85408556-fr/R23/S11.fits')
        #bb = afwGeom.Box2I(afwGeom.Point2I(0,0), afwGeom.Point2I(511,511))
        #exposure = afwImage.ExposureF('data/goodSeeingCoadd/r/3/113,0/coadd-r-3-113,0.fits', 0, bb)
        #exposure.writeFits('mini-r-3-113,0.fits')
        fn = os.path.join(os.path.dirname(__file__), 'data', 'mini-r-3-113,0.fits.gz')
        print 'Reading image', fn
        exposure = afwImage.ExposureF(fn)
        schema = afwTable.SourceTable.makeMinimalSchema()
        idFactory = afwTable.IdFactory.makeSimple()

        dconf = SourceDetectionConfig()
        dconf.reEstimateBackground = False
        dconf.includeThresholdMultiplier = 5.

        mconf = SourceMeasurementConfig()

        aconf = pipeTasksAstrom.AstrometryConfig()
        aconf.forceKnownWcs = True

        det = SourceDetectionTask(schema=schema, config=dconf)
        meas = SourceMeasurementTask(schema, config=mconf)
        astrom = pipeTasksAstrom.AstrometryTask(schema, config=aconf, name='astrom')

        astrom.log.setThreshold(pexLog.Log.DEBUG)

        inwcs = exposure.getWcs()
        print 'inwcs:', inwcs
        instr = inwcs.getFitsMetadata().toString()
        print 'inwcs:', instr
        table = afwTable.SourceTable.make(schema, idFactory)
        sources = det.makeSourceCatalog(table, exposure, doSmooth=False).sources
        meas.measure(exposure, sources)

        for dosip in [False, True]:
            aconf.solver.calculateSip = dosip
            ast = astrom.run(exposure, sources)
            outwcs = exposure.getWcs()
            outstr = outwcs.getFitsMetadata().toString()
            if dosip is False:
                self.assertEqual(inwcs, outwcs)
                self.assertEqual(instr, outstr)
            print 'inwcs:', instr
            print 'outwcs:', outstr
            print len(ast.matches), 'matches'
            self.assertTrue(len(ast.matches) > 10)
        #exposure.writeFits('out-2155.fits')

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(TestForceWcs)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)
 
if __name__ == "__main__":
    run(True)
    
