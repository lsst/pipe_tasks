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
import pickle

import astropy.units as u

import lsst.utils.tests
from lsst.meas.algorithms import convertReferenceCatalog
from lsst.pipe.tasks.colorterms import Colorterm, ColortermDict, ColortermLibrary, ColortermNotFoundError

# From the last page of http://www.naoj.org/staff/nakata/suprime/illustration/colorterm_report_ver3.pdf
# Transformation for griz band between SDSS and SC (estimated with GS83 SEDs)
hamamatsu = ColortermLibrary(data={
    "ham*": ColortermDict(data={
        "g": Colorterm(primary="g", secondary="r", c0=-0.00928, c1=-0.0824),
        "r": Colorterm(primary="r", secondary="i", c0=-0.00282, c1=-0.0498, c2=-0.0149),
        "i": Colorterm(primary="i", secondary="z", c0=0.00186, c1=-0.140, c2=-0.0196),
        "z": Colorterm(primary="z", secondary="i", c0=-4.03e-4, c1=0.0967, c2=0.0210),
    })
})


def setup_module(module):
    lsst.utils.tests.init()


class ColortermTestCase(unittest.TestCase):
    """A test case for MaskedImage"""

    def setUp(self):
        # A list of simple fake sources. The values are chosen so that the colorterm corrections are
        # predictable.
        self.sources = (dict(g=0.0, r=0.0, fluxErr_g=0.0, fluxErr_r=0.0, true_g=-0.00928, true_fluxErr_g=0.0),
                        dict(g=0.0, r=-1.0, fluxErr_g=1.0, fluxErr_r=1.0, true_g=-0.09168,
                             true_fluxErr_g=0.92129230974756315))
        self.colorterms = hamamatsu

    def testTransformSource(self):
        """Check if we can use colour terms"""

        ct = self.colorterms.getColorterm("g", photoCatName="ham")

        for s in self.sources:
            self.assertEqual(ct.transformSource(s), s["true_g"])

    def testLibraryAccess(self):
        """Test ColortermLibrary.getColorterm"""
        ctg = self.colorterms.getColorterm("g", photoCatName="ham")  # exact match
        self.assertEqual(ctg.primary, "g")
        self.assertEqual(ctg.secondary, "r")
        self.assertAlmostEqual(ctg.c0, -0.00928)
        self.assertAlmostEqual(ctg.c1, -0.0824)
        self.assertAlmostEqual(ctg.c2, 0)

        ctr = self.colorterms.getColorterm("r", photoCatName="hambone")  # glob should expand
        self.assertEqual(ctr.primary, "r")
        self.assertEqual(ctr.secondary, "i")
        self.assertAlmostEqual(ctr.c0, -0.00282)
        self.assertAlmostEqual(ctr.c1, -0.0498)
        self.assertAlmostEqual(ctr.c2, -0.0149)

        # bad filter name
        self.assertRaises(ColortermNotFoundError, self.colorterms.getColorterm, "x", photoCatName="ham")

        # bad catalog name: not in library
        self.assertRaises(ColortermNotFoundError, self.colorterms.getColorterm, "r", photoCatName="eggs")

        # bad catalog name: glob expression
        self.assertRaises(ColortermNotFoundError, self.colorterms.getColorterm, "r", photoCatName="ha*")

    def testTransformMags(self):
        """Check if we can use colour terms via transformMags"""

        ct = self.colorterms.getColorterm("g", photoCatName="ham")

        for s in self.sources:
            self.assertEqual(ct.transformMags(s[ct.primary], s[ct.secondary]), s["true_g"])

    def testPropagateFluxErrors(self):
        """Check if we can propagate flux errors"""

        ct = self.colorterms.getColorterm("g", photoCatName="ham")
        for s in self.sources:
            self.assertEqual(ct.propagateFluxErrors(s["fluxErr_g"], s["fluxErr_r"]), s["true_fluxErr_g"])

    def testPickle(self):
        """Ensure color terms can be pickled"""
        colorterms = pickle.loads(pickle.dumps(self.colorterms))
        self.assertEqual(colorterms, self.colorterms)


def make_fake_refcat(center, flux):
    """Make a fake reference catalog."""
    filters = ['f1', 'f2']
    schema = convertReferenceCatalog._makeSchema(filters)
    catalog = lsst.afw.table.SimpleCatalog(schema)
    record = catalog.addNew()
    record.setCoord(center)
    record[filters[0] + '_flux'] = flux
    record[filters[0] + '_fluxErr'] = flux*0.1
    record[filters[1] + '_flux'] = flux*10
    record[filters[1] + '_fluxErr'] = flux*10*0.1
    return catalog


class ApplyColortermsTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.colorterm = Colorterm(primary="f1", secondary="f2", c0=2.0, c1=3.0)

    def testGetCorrectedMagnitudes(self):
        center = lsst.geom.SpherePoint(30, -30, lsst.geom.degrees)
        flux = 100
        refCat = make_fake_refcat(center, flux)

        expectMag = self.colorterm.transformMags((u.nJy*refCat['f1_flux']).to_value(u.ABmag),
                                                 (u.nJy*refCat['f2_flux']).to_value(u.ABmag))
        refMag, refMagErr = self.colorterm.getCorrectedMagnitudes(refCat)
        self.assertEqual(refMag, expectMag)
        # TODO DM-17692: Not testing the returned errors, as I do not trust `propagateFluxErrors()`
        # and there is some interesting logic involved in how the errors are propagated.


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
