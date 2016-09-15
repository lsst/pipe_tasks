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
from __future__ import print_function
import unittest
import os

import lsst.meas.algorithms as measAlg
from lsst.meas.astrom import ANetAstrometryTask, ANetAstrometryConfig
from lsst.meas.base import SingleFrameMeasurementTask, SingleFrameMeasurementConfig
import lsst.utils.tests
import lsst.afw.detection as afwDetection
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable

import testFindAstrometryNetDataDir as helper


class TestForceWcs(unittest.TestCase):

    def setUp(self):
        helper.setupAstrometryNetDataDir('t2155')

    def test1(self):
        fn = os.path.join(os.path.dirname(__file__), 'data', 'mini-r-3-113,0.fits.gz')
        print('Reading image', fn)
        exposure = afwImage.ExposureF(fn)
        exposure.setPsf(afwDetection.GaussianPsf(15, 15, 3))
        schema = afwTable.SourceTable.makeMinimalSchema()
        idFactory = afwTable.IdFactory.makeSimple()

        dconf = measAlg.SourceDetectionConfig()
        dconf.reEstimateBackground = False
        dconf.includeThresholdMultiplier = 5.

        mconf = SingleFrameMeasurementConfig()

        aconf = ANetAstrometryConfig()
        aconf.forceKnownWcs = True

        det = measAlg.SourceDetectionTask(schema=schema, config=dconf)
        meas = SingleFrameMeasurementTask(schema, config=mconf)
        astrom = ANetAstrometryTask(schema, config=aconf, name='astrom')

        astrom.log.setLevel(astrom.log.TRACE)

        inwcs = exposure.getWcs()
        print('inwcs:', inwcs)
        instr = inwcs.getFitsMetadata().toString()
        print('inwcs:', instr)
        table = afwTable.SourceTable.make(schema, idFactory)
        sources = det.makeSourceCatalog(table, exposure, sigma=1).sources
        meas.measure(exposure, sources)

        for dosip in [False, True]:
            aconf.solver.calculateSip = dosip
            ast = astrom.run(sourceCat=sources, exposure=exposure)
            outwcs = exposure.getWcs()
            outstr = outwcs.getFitsMetadata().toString()
            if dosip is False:
                self.assertEqual(inwcs, outwcs)
                self.assertEqual(instr, outstr)
            print('inwcs:', instr)
            print('outwcs:', outstr)
            print(len(ast.matches), 'matches')
            self.assertGreater(len(ast.matches), 10)


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
