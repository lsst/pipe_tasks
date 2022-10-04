# This file is part of pipe_tasks.
#
# LSST Data Management System
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
# See COPYRIGHT file at the top of the source tree.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
"""Test HealSparsePropertyMapTask.
"""
import unittest
import numpy as np
import healsparse as hsp
import esutil
import warnings

import lsst.utils.tests
import lsst.daf.butler
import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
from lsst.skymap.discreteSkyMap import DiscreteSkyMap
import lsst.geom as geom

from lsst.pipe.tasks.healSparseMapping import HealSparsePropertyMapTask
from lsst.pipe.tasks.healSparseMappingProperties import (register_property_map,
                                                         BasePropertyMap)

from surveyPropertyMapsTestUtils import (makeMockVisitSummary,
                                         MockVisitSummaryReference,
                                         MockCoaddReference,
                                         MockInputMapReference)


# Test creation of an arbitrary new property map by registering it here
# and using it in the test class.
@register_property_map("dist_times_psfarea")
class DistTimesPsfAreaPropertyMap(BasePropertyMap):
    """Property map to compute the distance from the boresight center
    by the psf area. Do not try this at home."""
    requires_psf = True

    def _compute(self, row, ra, dec, scalings, psf_array=None):
        boresight = row.getVisitInfo().getBoresightRaDec()
        dist = esutil.coords.sphdist(ra, dec,
                                     boresight.getRa().asDegrees(), boresight.getDec().asDegrees())
        return np.deg2rad(dist)*psf_array['psf_area']


class HealSparsePropertyMapTaskTestCase(lsst.utils.tests.TestCase):
    """Test of HealSparsePropertyMapTask.

    These tests bypass the middleware used for accessing data and
    managing Task execution.
    """
    def setUp(self):
        tract = 0
        band = 'r'
        patch = 0
        visits = [100, 101]
        # Good to test crossing 0.
        ra_center = 0.0
        dec_center = -45.0
        pixel_scale = 0.2
        coadd_zp = 27.0

        # Generate a mock skymap with one patch
        config = DiscreteSkyMap.ConfigClass()
        config.raList = [ra_center]
        config.decList = [dec_center]
        config.radiusList = [150*pixel_scale/3600.]
        config.patchInnerDimensions = (350, 350)
        config.patchBorder = 50
        config.tractOverlap = 0.0
        config.pixelScale = pixel_scale
        sky_map = DiscreteSkyMap(config)

        visit_summaries = [makeMockVisitSummary(visit,
                                                ra_center=ra_center,
                                                dec_center=dec_center)
                           for visit in visits]
        visit_summary_refs = [MockVisitSummaryReference(visit_summary, visit)
                              for visit_summary, visit in zip(visit_summaries, visits)]
        self.visit_summary_dict = {visit: ref.get()
                                   for ref, visit in zip(visit_summary_refs, visits)}

        # Generate an input map.  Note that this does not need to be consistent
        # with the visit_summary projections, we're just tracking values.
        with warnings.catch_warnings():
            # Healsparse will emit a warning if nside coverage is greater than
            # 128.  In the case of generating patch input maps, and not global
            # maps, high nside coverage works fine, so we can suppress this
            # warning.
            warnings.simplefilter("ignore")
            input_map = hsp.HealSparseMap.make_empty(nside_coverage=256,
                                                     nside_sparse=32768,
                                                     dtype=hsp.WIDE_MASK,
                                                     wide_mask_maxbits=len(visits)*2)

        patch_poly = afwGeom.Polygon(geom.Box2D(sky_map[tract][patch].getOuterBBox()))
        sph_pts = sky_map[tract].getWcs().pixelToSky(patch_poly.convexHull().getVertices())
        patch_poly_radec = np.array([(sph.getRa().asDegrees(), sph.getDec().asDegrees())
                                     for sph in sph_pts])
        poly = hsp.Polygon(ra=patch_poly_radec[: -1, 0],
                           dec=patch_poly_radec[: -1, 1],
                           value=[0])
        poly_pixels = poly.get_pixels(nside=input_map.nside_sparse)
        # The input map has full coverage for bits 0 and 1
        input_map.set_bits_pix(poly_pixels, [0])
        input_map.set_bits_pix(poly_pixels, [1])

        input_map_ref = MockInputMapReference(input_map, patch=patch, tract=tract)
        self.input_map_dict = {patch: input_map_ref}

        coadd = afwImage.ExposureF(sky_map[tract][patch].getOuterBBox(),
                                   sky_map[tract].getWcs())
        instFluxMag0 = 10.**(coadd_zp/2.5)
        pc = afwImage.makePhotoCalibFromCalibZeroPoint(instFluxMag0)
        coadd.setPhotoCalib(pc)

        # Mock the coadd input ccd table
        schema = afwTable.ExposureTable.makeMinimalSchema()
        schema.addField("ccd", type="I")
        schema.addField("visit", type="I")
        schema.addField("weight", type="F")
        ccds = afwTable.ExposureCatalog(schema)
        ccds.resize(2)
        ccds['id'] = np.arange(2)
        ccds['visit'][0] = visits[0]
        ccds['visit'][1] = visits[1]
        ccds['ccd'][0] = 0
        ccds['ccd'][1] = 1
        ccds['weight'] = 10.0
        for ccd_row in ccds:
            summary = self.visit_summary_dict[ccd_row['visit']].find(ccd_row['ccd'])
            ccd_row.setWcs(summary.getWcs())
            ccd_row.setPsf(summary.getPsf())
            ccd_row.setBBox(summary.getBBox())
            ccd_row.setPhotoCalib(summary.getPhotoCalib())

        inputs = afwImage.CoaddInputs()
        inputs.ccds = ccds
        coadd.getInfo().setCoaddInputs(inputs)

        coadd_ref = MockCoaddReference(coadd, patch=patch, tract=tract)
        self.coadd_dict = {patch: coadd_ref}

        self.tract = tract
        self.band = band
        self.sky_map = sky_map
        self.input_map = input_map

    def testPropertyMapCreation(self):
        """Test creation of property maps."""
        config = HealSparsePropertyMapTask.ConfigClass()

        # Add our new test map to the set of maps
        config.property_maps.names |= ['dist_times_psfarea']
        config.property_maps['dist_times_psfarea'].do_min = True
        config.property_maps['dist_times_psfarea'].do_max = True
        config.property_maps['dist_times_psfarea'].do_mean = True

        property_task = HealSparsePropertyMapTask(config=config)

        property_task.run(self.sky_map,
                          self.tract,
                          self.band,
                          self.coadd_dict,
                          self.input_map_dict,
                          self.visit_summary_dict)

        valid_pixels = self.input_map.valid_pixels

        # Verify each map exists and has the correct pixels set.
        for name, map_config, PropertyMapClass in config.property_maps.apply():
            self.assertTrue(name in property_task.property_maps)
            property_map = property_task.property_maps[name]
            if map_config.do_min:
                self.assertTrue(hasattr(property_map, 'min_map'))
                np.testing.assert_array_equal(property_map.min_map.valid_pixels, valid_pixels)
            else:
                self.assertFalse(hasattr(property_map, 'min_map'))
            if map_config.do_max:
                self.assertTrue(hasattr(property_map, 'max_map'))
                np.testing.assert_array_equal(property_map.max_map.valid_pixels, valid_pixels)
            else:
                self.assertFalse(hasattr(property_map, 'max_map'))
            if map_config.do_mean:
                self.assertTrue(hasattr(property_map, 'mean_map'))
                np.testing.assert_array_equal(property_map.mean_map.valid_pixels, valid_pixels)
            else:
                self.assertFalse(hasattr(property_map, 'mean_map'))
            if map_config.do_weighted_mean:
                self.assertTrue(hasattr(property_map, 'weighted_mean_map'))
                np.testing.assert_array_equal(property_map.weighted_mean_map.valid_pixels, valid_pixels)
            else:
                self.assertFalse(hasattr(property_map, 'weighted_mean_map'))
            if map_config.do_sum:
                self.assertTrue(hasattr(property_map, 'sum_map'))
                np.testing.assert_array_equal(property_map.sum_map.valid_pixels, valid_pixels)
            else:
                self.assertFalse(hasattr(property_map, 'sum_map'))


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
