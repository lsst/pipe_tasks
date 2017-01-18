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
from __future__ import division, print_function, absolute_import

import unittest

import lsst.utils.tests
from lsst.pipe.tasks.calibrate import CalibrateConfig


class CalibrateTestCase(unittest.TestCase):

    def testGood(self):
        conf = CalibrateConfig()
        conf.validate()
        conf.doPhotoCal = False
        conf.validate()
        conf.doAstrometry = False
        conf.validate()

    def testBad(self):
        conf = CalibrateConfig()
        with self.assertRaises(Exception):
            conf.invalidField = True


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
