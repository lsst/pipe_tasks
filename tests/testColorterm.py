#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
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

"""
Tests for Colorterm

Run with:
   python colorterm.py
or
   python
   >>> import colorterm; colorterm.run()
"""

import unittest
import pickle

import lsst.utils.tests as utilsTests
from lsst.pipe.tasks.colorterms import Colorterm, ColortermDict, ColortermLibrary

# From the last page of http://www.naoj.org/staff/nakata/suprime/illustration/colorterm_report_ver3.pdf
# Transformation for griz band between SDSS and SC (estimated with GS83 SEDs)
mitll = ColortermLibrary(data={
    "*": ColortermDict(data={
        "g": Colorterm(primary="g", secondary="r", c0=-0.00569, c1=-0.0427),
        "r": Colorterm(primary="r", secondary="g", c0=0.00261,  c1=0.0304),
        "i": Colorterm(primary="i", secondary="r", c0=0.00586,  c1=0.0827, c2=-0.0118),
        "z": Colorterm(primary="z", secondary="i", c0=0.000329, c1=0.0608, c2=0.0219),
    })
})
hamamatsu = ColortermLibrary(data={
    "*": ColortermDict(data={
        "g": Colorterm(primary="g", secondary="r", c0=-0.00928, c1=-0.0824),
        "r": Colorterm(primary="r", secondary="i", c0=-0.00282, c1=-0.0498, c2=-0.0149),
        "i": Colorterm(primary="i", secondary="z", c0=0.00186,  c1=-0.140,  c2=-0.0196),
        "z": Colorterm(primary="z", secondary="i", c0=-4.03e-4, c1=0.0967,  c2=0.0210),
    })
})

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ColortermTestCase(unittest.TestCase):
    """A test case for MaskedImage"""
    def setUp(self):
        self.sources = dict(g=0.0, r=0.0, true_g=-0.00928), dict(g=0.0, r=-1.0, true_g=-0.09168)
        self.colorterms = hamamatsu

    def tearDown(self):
        import lsst.meas.astrom.astrometry_net as an
        an.finalize()

    def testTransformSource(self):
        """Check if we can use colour terms"""

        ct = self.colorterms.getColorterm("g", refCatName="foo")

        for s in self.sources:
            self.assertEqual(ct.transformSource(s), s["true_g"])

    def testTransformMags(self):
        """Check if we can use colour terms via transformMags"""

        ct = self.colorterms.getColorterm("g", refCatName="bar")

        for s in self.sources:
            self.assertEqual(ct.transformMags(s[ct.primary], s[ct.secondary]), s["true_g"])

    def testPickle(self):
        """Ensure color terms can be pickled"""
        colorterms = pickle.loads(pickle.dumps(self.colorterms))
        self.assertEqual(colorterms, self.colorterms)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ColortermTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
