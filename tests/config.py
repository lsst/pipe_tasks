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

import lsst.utils.tests as utilsTests
import lsst.pipe.tasks as pipeTasks
import lsst.pipe.tasks.calibrate

class CalibrateTestCase(unittest.TestCase):
    conf = pipeTasks.calibrate.CalibrateConfig()
    conf.validate()
    # Demonstrate typo
    conf.doPhotoCal = False
    conf.validate()
    conf.doPsf = True
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
