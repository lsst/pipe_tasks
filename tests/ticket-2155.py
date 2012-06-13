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
        ver = 'sdss-2012-05-01-0'
        ok, version, reason = eupsObj.setup('astrometry_net_data', versionName=ver)
        if not ok:
            raise ValueError("Failed to setup astrometry_net_data version '%s': %s" % (ver, reason))
        print 'Setup astrometry_net_data', ver

    def test1(self):
        '''
        > bb = afwGeom.Box2I(afwGeom.Point2I(0,0), afwGeom.Point2I(255,255))
        > e2 = afwImg.ExposureF('../afwdata/ImSim/calexp/v85408556-fr/R23/S11.fits', 0, bb)
        > e2.writeFits('mini-v85408556-fr-R23-S11.fits')
        '''
        #exposure = afwImage.ExposureF('mini-v85408556-fr-R23-S11.fits')
        #exposure = afwImage.ExposureF('../afwdata/ImSim/calexp/v85408556-fr/R23/S11.fits')
        exposure = afwImage.ExposureF('data/goodSeeingCoadd/r/3/113,0/coadd-r-3-113,0.fits')
        schema = afwTable.SourceTable.makeMinimalSchema()
        idFactory = afwTable.IdFactory.makeSimple()

        dconf = SourceDetectionConfig()
        dconf.reEstimateBackground = False
        dconf.includeThresholdMultiplier = 10.

        mconf = SourceMeasurementConfig()

        aconf = pipeTasksAstrom.AstrometryConfig()
        aconf.forceKnownWcs = True
        aconf.solver.calculateSip = False

        det = SourceDetectionTask(schema=schema, config=dconf)
        meas = SourceMeasurementTask(schema, config=mconf)
        astrom = pipeTasksAstrom.AstrometryTask(schema, config=aconf, name='astrom')

        astrom.log.setThreshold(pexLog.Log.DEBUG)

        inwcs = exposure.getWcs()
        instr = inwcs.getFitsMetadata().toString()
        table = afwTable.SourceTable.make(schema, idFactory)
        sources = det.makeSourceCatalog(table, exposure).sources
        meas.measure(exposure, sources)
        ast = astrom.run(exposure, sources)
        outwcs = exposure.getWcs()
        outstr = outwcs.getFitsMetadata().toString()
        self.assertEqual(inwcs, outwcs)
        self.assertEqual(instr, outstr)
        print 'inwcs:', instr
        print 'outwcs:', outstr
        print len(ast.matches), 'matches'
        self.assertTrue(len(ast.matches) > 200)

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
    
