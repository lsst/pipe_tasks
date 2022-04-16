#
# LSST Data Management System
# Copyright 2012 LSST Corporation.
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

import lsst.geom
import lsst.afw.geom as afwGeom
import lsst.skymap
from lsst.pipe.tasks.makeDiscreteSkyMap import MakeDiscreteSkyMapTask, MakeDiscreteSkyMapConfig


class MakeDiscreteSkyMapTestCase(unittest.TestCase):
    """Test MakeDiscreteSkyMapTask."""
    def setUp(self):
        self.config = MakeDiscreteSkyMapConfig()
        self.task = MakeDiscreteSkyMapTask(config=self.config)

        self.cd_matrix = afwGeom.makeCdMatrix(scale=0.2*lsst.geom.arcseconds)
        self.crpix = lsst.geom.Point2D(100, 100)
        self.crval1 = lsst.geom.SpherePoint(10.0*lsst.geom.degrees, 0.0*lsst.geom.degrees)
        self.wcs1 = afwGeom.makeSkyWcs(crpix=self.crpix, crval=self.crval1, cdMatrix=self.cd_matrix)
        self.bbox = lsst.geom.Box2I(corner=lsst.geom.Point2I(0, 0), dimensions=lsst.geom.Extent2I(200, 200))
        self.crval2 = lsst.geom.SpherePoint(11.0*lsst.geom.degrees, 1.0*lsst.geom.degrees)
        self.wcs2 = afwGeom.makeSkyWcs(crpix=self.crpix, crval=self.crval2, cdMatrix=self.cd_matrix)
        self.crval3 = lsst.geom.SpherePoint(20.0*lsst.geom.degrees, 10.0*lsst.geom.degrees)
        self.wcs3 = afwGeom.makeSkyWcs(crpix=self.crpix, crval=self.crval3, cdMatrix=self.cd_matrix)

    def test_run(self):
        """Test running the MakeDiscreteSkyMapTask."""
        wcs_bbox_tuple_list = [
            (self.wcs1, self.bbox),
            (self.wcs2, self.bbox)
        ]
        results = self.task.run(wcs_bbox_tuple_list)

        skymap = results.skyMap
        self.assertIsInstance(skymap, lsst.skymap.DiscreteSkyMap)
        self.assertEqual(len(skymap), 1)
        tract_info = skymap[0]

        # Ensure that the tract contains our points.
        self.assertTrue(tract_info.contains(self.crval1))
        self.assertTrue(tract_info.contains(self.crval2))

    def test_append(self):
        wcs_bbox_tuple_list = [
            (self.wcs1, self.bbox),
            (self.wcs2, self.bbox)
        ]
        results = self.task.run(wcs_bbox_tuple_list)

        skymap = results.skyMap

        wcs_bbox_tuple_list2 = [
            (self.wcs3, self.bbox)
        ]
        results2 = self.task.run(wcs_bbox_tuple_list2, oldSkyMap=skymap)

        skymap = results2.skyMap
        self.assertIsInstance(skymap, lsst.skymap.DiscreteSkyMap)
        self.assertEqual(len(skymap), 2)

        tract_info1 = skymap[0]

        # Ensure that the tract contains our points.
        self.assertTrue(tract_info1.contains(self.crval1))
        self.assertTrue(tract_info1.contains(self.crval2))

        tract_info2 = skymap[1]

        # Ensure that the tract contains our points.
        self.assertTrue(tract_info2.contains(self.crval3))


if __name__ == "__main__":
    unittest.main()
