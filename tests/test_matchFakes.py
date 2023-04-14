#
# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
import pandas as pd
import unittest
import uuid

import lsst.sphgeom as sphgeom
import lsst.geom as geom
import lsst.meas.base.tests as measTests
import lsst.skymap as skyMap
import lsst.utils.tests

from lsst.pipe.tasks.matchFakes import MatchFakesTask, MatchFakesConfig


class TestMatchFakes(lsst.utils.tests.TestCase):

    def setUp(self):
        """Create fake data to use in the tests.
        """
        self.bbox = geom.Box2I(geom.Point2I(0, 0),
                               geom.Extent2I(1024, 1153))
        dataset = measTests.TestDataset(self.bbox)
        self.exposure = dataset.exposure

        simpleMapConfig = skyMap.discreteSkyMap.DiscreteSkyMapConfig()
        simpleMapConfig.raList = [dataset.exposure.getWcs().getSkyOrigin().getRa().asDegrees()]
        simpleMapConfig.decList = [dataset.exposure.getWcs().getSkyOrigin().getDec().asDegrees()]
        simpleMapConfig.radiusList = [0.1]

        self.simpleMap = skyMap.DiscreteSkyMap(simpleMapConfig)
        self.tractId = 0
        bCircle = self.simpleMap.generateTract(self.tractId).getInnerSkyPolygon().getBoundingCircle()
        bCenter = sphgeom.LonLat(bCircle.getCenter())
        bRadius = bCircle.getOpeningAngle().asRadians()
        targetSources = 10000
        self.sourceDensity = (targetSources
                              / (bCircle.getArea() * (180 / np.pi) ** 2))
        self.rng = np.random.default_rng(1234)

        self.fakeCat = pd.DataFrame({
            "fakeId": [uuid.uuid4().int & (1 << 64) - 1 for n in range(targetSources)],
            # Quick-and-dirty values for testing
            "ra": bCenter.getLon().asRadians() + bRadius * (2.0 * self.rng.random(targetSources) - 1.0),
            "dec": bCenter.getLat().asRadians() + bRadius * (2.0 * self.rng.random(targetSources) - 1.0),
            "isVisitSource": np.concatenate([np.ones(targetSources//2, dtype="bool"),
                                             np.zeros(targetSources - targetSources//2, dtype="bool")]),
            "isTemplateSource": np.concatenate([np.zeros(targetSources//2, dtype="bool"),
                                                np.ones(targetSources - targetSources//2, dtype="bool")]),
            **{band: self.rng.uniform(20, 30, size=targetSources)
               for band in {"u", "g", "r", "i", "z", "y"}},
            "DiskHalfLightRadius": np.ones(targetSources, dtype="float"),
            "BulgeHalfLightRadius": np.ones(targetSources, dtype="float"),
            "disk_n": np.ones(targetSources, dtype="float"),
            "bulge_n": np.ones(targetSources, dtype="float"),
            "a_d": np.ones(targetSources, dtype="float"),
            "a_b": np.ones(targetSources, dtype="float"),
            "b_d": np.ones(targetSources, dtype="float"),
            "b_b": np.ones(targetSources, dtype="float"),
            "pa_disk": np.ones(targetSources, dtype="float"),
            "pa_bulge": np.ones(targetSources, dtype="float"),
            "sourceType": targetSources * ["star"],
        })

        self.inExp = np.zeros(len(self.fakeCat), dtype=bool)
        bbox = geom.Box2D(self.exposure.getBBox())
        for idx, row in self.fakeCat.iterrows():
            coord = geom.SpherePoint(row["ra"],
                                     row["dec"],
                                     geom.radians)
            cent = self.exposure.getWcs().skyToPixel(coord)
            self.inExp[idx] = bbox.contains(cent)

        tmpCat = self.fakeCat[self.inExp].iloc[:int(self.inExp.sum() / 2)]
        extraColumnData = self.rng.integers(0, 100, size=len(tmpCat))
        self.sourceCat = pd.DataFrame(
            data={"ra": np.degrees(tmpCat["ra"]),
                  "dec": np.degrees(tmpCat["dec"]),
                  "diaObjectId": np.arange(1, len(tmpCat) + 1, dtype=int),
                  "band": "g",
                  "diaSourceId": np.arange(1, len(tmpCat) + 1, dtype=int),
                  "extraColumn": extraColumnData})
        self.sourceCat.set_index(["diaObjectId", "band", "extraColumn"],
                                 drop=False,
                                 inplace=True)

    def testProcessFakes(self):
        """Test the run method.
        """
        matchFakesConfig = MatchFakesConfig()
        matchFakesConfig.matchDistanceArcseconds = 0.1
        matchFakes = MatchFakesTask(config=matchFakesConfig)
        result = matchFakes._processFakes(self.fakeCat,
                                          self.exposure,
                                          self.sourceCat)
        self.assertEqual(self.inExp.sum(), len(result.matchedDiaSources))
        self.assertEqual(
            len(self.sourceCat),
            np.sum(np.isfinite(result.matchedDiaSources["extraColumn"])))

    def testTrimCat(self):
        """Test that the correct number of sources are in the ccd area.
        """
        matchTask = MatchFakesTask()
        result = matchTask._trimFakeCat(self.fakeCat, self.exposure)
        self.assertEqual(len(result), self.inExp.sum())


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
