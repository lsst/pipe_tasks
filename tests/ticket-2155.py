#!/usr/bin/env python
# 
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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

import os
import eups

from lsst.meas.algorithms import *
import lsst.pipe.tasks as pipeTasks
import lsst.pipe.tasks.astrometry as pipeTasksAstrom
import lsst.utils.tests as utilsTests
import lsst.afw.image   as afwImage
import lsst.afw.table   as afwTable
import lsst.pex.logging as pexLog

display = False

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
        sources = det.makeSourceCatalog(table, exposure, sigma=1).sources
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

            if display:
                import lsst.afw.display.ds9 as ds9
                ds9.mtv(exposure, frame=1)
                x0, y0 = exposure.getXY0()
                with ds9.Buffering():
                    for ss in astrom.astrometer.getReferenceSourcesForWcs(inwcs, exposure.getDimensions(), "r",
                                                                          x0=x0, y0=y0):
                        xy = inwcs.skyToPixel(ss.getCoord())
                        ds9.dot("+", xy[0] - x0, xy[1] - y0, ctype='red', frame=1)
                    for ss in sources:
                        xy = ss.getCentroid()
                        ds9.dot("x", xy[0] - x0, xy[1] - y0, ctype='red', frame=1)
                    for mm in ast.matches:
                        xy = mm.second.getCentroid()
                        ds9.dot('o', xy[0] - x0, xy[1] - y0, ctype='blue', frame=1)
                import pdb;pdb.set_trace() # Pause for inspection

            self.assertTrue(len(ast.matches) >= 10)

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
    
