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
import shutil
import tempfile
import unittest
import uuid

import lsst.daf.butler.tests as butlerTests
import lsst.sphgeom as sphgeom
import lsst.geom as geom
import lsst.meas.base.tests as measTests
from lsst.pipe.base import testUtils
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
            "raJ2000": bCenter.getLon().asRadians() + bRadius * (2.0 * self.rng.random(targetSources) - 1.0),
            "decJ2000": bCenter.getLat().asRadians() + bRadius * (2.0 * self.rng.random(targetSources) - 1.0),
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
            coord = geom.SpherePoint(row["raJ2000"],
                                     row["decJ2000"],
                                     geom.radians)
            cent = self.exposure.getWcs().skyToPixel(coord)
            self.inExp[idx] = bbox.contains(cent)

        tmpCat = self.fakeCat[self.inExp].iloc[:int(self.inExp.sum() / 2)]
        extraColumnData = self.rng.integers(0, 100, size=len(tmpCat))
        self.sourceCat = pd.DataFrame(
            data={"ra": np.degrees(tmpCat["raJ2000"]),
                  "decl": np.degrees(tmpCat["decJ2000"]),
                  "diaObjectId": np.arange(1, len(tmpCat) + 1, dtype=int),
                  "filterName": "g",
                  "diaSourceId": np.arange(1, len(tmpCat) + 1, dtype=int),
                  "extraColumn": extraColumnData})
        self.sourceCat.set_index(["diaObjectId", "filterName", "extraColumn"],
                                 drop=False,
                                 inplace=True)

    def testRunQuantum(self):
        """Test the run quantum method with a gen3 butler.
        """
        root = tempfile.mkdtemp()
        dimensions = {"instrument": ["notACam"],
                      "skymap": ["deepCoadd_skyMap"],
                      "tract": [0, 42],
                      "visit": [1234, 4321],
                      "detector": [25, 26]}
        testRepo = butlerTests.makeTestRepo(root, dimensions)
        matchTask = MatchFakesTask()
        connections = matchTask.config.ConnectionsClass(
            config=matchTask.config)

        fakesDataId = {"skymap": "deepCoadd_skyMap",
                       "tract": 0}
        imgDataId = {"instrument": "notACam",
                     "visit": 1234,
                     "detector": 25}
        butlerTests.addDatasetType(
            testRepo,
            connections.fakeCat.name,
            connections.fakeCat.dimensions,
            connections.fakeCat.storageClass)
        butlerTests.addDatasetType(
            testRepo,
            connections.image.name,
            connections.image.dimensions,
            connections.image.storageClass)
        butlerTests.addDatasetType(
            testRepo,
            connections.detectedSources.name,
            connections.detectedSources.dimensions,
            connections.detectedSources.storageClass)
        butlerTests.addDatasetType(
            testRepo,
            connections.matchedSources.name,
            connections.matchedSources.dimensions,
            connections.matchedSources.storageClass)
        butler = butlerTests.makeTestCollection(testRepo)

        butler.put(self.fakeCat,
                   connections.fakeCat.name,
                   {"tract": fakesDataId["tract"],
                    "skymap": fakesDataId["skymap"]})
        butler.put(self.exposure,
                   connections.image.name,
                   {"instrument": imgDataId["instrument"],
                    "visit": imgDataId["visit"],
                    "detector": imgDataId["detector"]})
        butler.put(self.sourceCat,
                   connections.detectedSources.name,
                   {"instrument": imgDataId["instrument"],
                    "visit": imgDataId["visit"],
                    "detector": imgDataId["detector"]})

        quantumDataId = imgDataId.copy()
        quantumDataId.update(fakesDataId)
        quantum = testUtils.makeQuantum(
            matchTask, butler, quantumDataId,
            {"fakeCat": fakesDataId,
             "image": imgDataId,
             "detectedSources": imgDataId,
             "matchedSources": imgDataId})
        run = testUtils.runTestQuantum(matchTask, butler, quantum)
        # Actual input dataset omitted for simplicity
        run.assert_called_once()
        shutil.rmtree(root, ignore_errors=True)

    def testRun(self):
        """Test the run method.
        """
        matchFakesConfig = MatchFakesConfig()
        matchFakesConfig.matchDistanceArcseconds = 0.1
        matchFakes = MatchFakesTask(config=matchFakesConfig)
        result = matchFakes.run(self.fakeCat,
                                self.exposure,
                                self.sourceCat)
        self.assertEqual(self.inExp.sum(), len(result.matchedSources))
        self.assertEqual(
            len(self.sourceCat),
            np.sum(np.isfinite(result.matchedSources["extraColumn"])))

    def testTrimCat(self):
        """Test that the correct number of sources are in the ccd area.
        """
        matchTask = MatchFakesTask()
        result = matchTask._trimFakeCat(self.fakeCat, self.exposure)
        self.assertEqual(len(result), self.inExp.sum())

    def testNonstandardCat(self):
        customCat = pd.DataFrame(
            data={"ra2k": self.sourceCat["ra"],
                  "de2k": self.sourceCat["decl"],
                  "filterName": self.sourceCat["filterName"],
                  "unsourceId": np.arange(1, len(self.sourceCat) + 1, dtype=int),
                  "extraColumn": self.sourceCat["extraColumn"]})
        customCat.set_index(["unsourceId", "filterName", "extraColumn"],
                            drop=False,
                            inplace=True)
        matchFakesConfig = MatchFakesConfig()
        matchFakesConfig.matchDistanceArcseconds = 0.1
        matchFakesConfig.src_id_col = "unsourceId"
        matchFakesConfig.src_ra_col = "ra2k"
        matchFakesConfig.src_dec_col = "de2k"

        matchFakes = MatchFakesTask(config=matchFakesConfig)
        result = matchFakes.run(self.fakeCat,
                                self.exposure,
                                customCat)
        self.assertEqual(self.inExp.sum(), len(result.matchedSources))
        self.assertEqual(
            len(self.sourceCat),
            np.sum(np.isfinite(result.matchedSources["extraColumn"])))

    def testNoIdCat(self):
        cat = self.sourceCat.set_index("diaObjectId", drop=True)
        matchFakesConfig = MatchFakesConfig()
        matchFakesConfig.matchDistanceArcseconds = 0.1
        matchFakesConfig.src_id_col = None

        matchFakes = MatchFakesTask(config=matchFakesConfig)
        result = matchFakes.run(self.fakeCat,
                                self.exposure,
                                cat)
        self.assertEqual(self.inExp.sum(), len(result.matchedSources))
        self.assertEqual(
            len(self.sourceCat),
            np.sum(np.isfinite(result.matchedSources["extraColumn"])))

    def testMultiIndexCat(self):
        cat = self.sourceCat.set_index(["diaObjectId", "filterName", "extraColumn"],
                                       drop=True)
        matchFakesConfig = MatchFakesConfig()
        matchFakesConfig.matchDistanceArcseconds = 0.1
        matchFakesConfig.src_id_col = None

        matchFakes = MatchFakesTask(config=matchFakesConfig)
        with self.assertRaises(ValueError):
            matchFakes.run(self.fakeCat,
                           self.exposure,
                           cat)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
