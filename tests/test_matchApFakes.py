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

import lsst.daf.butler.tests as butlerTests
import lsst.geom as geom
import lsst.meas.base.tests as measTests
from lsst.pipe.base import testUtils
import lsst.skymap as skyMap
import lsst.utils.tests

from lsst.pipe.tasks.matchApFakes import MatchApFakesTask, MatchApFakesConfig
from lsst.ap.pipe.createApFakes import CreateRandomApFakesTask, CreateRandomApFakesConfig


class TestMatchApFakes(lsst.utils.tests.TestCase):

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
        targetSources = 10000
        self.sourceDensity = (targetSources
                              / (bCircle.getArea() * (180 / np.pi) ** 2))
        self.rng = np.random.default_rng(1234)

        fakesConfig = CreateRandomApFakesConfig()
        fakesConfig.fraction = 0.0
        fakesConfig.fakeDensity = self.sourceDensity
        fakesTask = CreateRandomApFakesTask(config=fakesConfig)
        self.fakeCat = fakesTask.run(self.tractId, self.simpleMap).fakeCat

        self.inExp = np.zeros(len(self.fakeCat), dtype=bool)
        bbox = geom.Box2D(self.exposure.getBBox())
        for idx, row in self.fakeCat.iterrows():
            coord = geom.SpherePoint(row[fakesConfig.raColName],
                                     row[fakesConfig.decColName],
                                     geom.radians)
            cent = self.exposure.getWcs().skyToPixel(coord)
            self.inExp[idx] = bbox.contains(cent)

        tmpCat = self.fakeCat[self.inExp].iloc[:int(self.inExp.sum() / 2)]
        extraColumnData = self.rng.integers(0, 100, size=len(tmpCat))
        self.sourceCat = pd.DataFrame(
            data={"ra": np.degrees(tmpCat[fakesTask.config.raColName]),
                  "decl": np.degrees(tmpCat[fakesTask.config.decColName]),
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
        matchTask = MatchApFakesTask()
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
            connections.diffIm.name,
            connections.diffIm.dimensions,
            connections.diffIm.storageClass)
        butlerTests.addDatasetType(
            testRepo,
            connections.associatedDiaSources.name,
            connections.associatedDiaSources.dimensions,
            connections.associatedDiaSources.storageClass)
        butlerTests.addDatasetType(
            testRepo,
            connections.matchedDiaSources.name,
            connections.matchedDiaSources.dimensions,
            connections.matchedDiaSources.storageClass)
        butler = butlerTests.makeTestCollection(testRepo)

        butler.put(self.fakeCat,
                   connections.fakeCat.name,
                   {"tract": fakesDataId["tract"],
                    "skymap": fakesDataId["skymap"]})
        butler.put(self.exposure,
                   connections.diffIm.name,
                   {"instrument": imgDataId["instrument"],
                    "visit": imgDataId["visit"],
                    "detector": imgDataId["detector"]})
        butler.put(self.sourceCat,
                   connections.associatedDiaSources.name,
                   {"instrument": imgDataId["instrument"],
                    "visit": imgDataId["visit"],
                    "detector": imgDataId["detector"]})

        quantumDataId = imgDataId.copy()
        quantumDataId.update(fakesDataId)
        quantum = testUtils.makeQuantum(
            matchTask, butler, quantumDataId,
            {"fakeCat": fakesDataId,
             "diffIm": imgDataId,
             "associatedDiaSources": imgDataId,
             "matchedDiaSources": imgDataId})
        run = testUtils.runTestQuantum(matchTask, butler, quantum)
        # Actual input dataset omitted for simplicity
        run.assert_called_once()
        shutil.rmtree(root, ignore_errors=True)

    def testRun(self):
        """Test the run method.
        """
        matchFakesConfig = MatchApFakesConfig()
        matchFakesConfig.matchDistanceArcseconds = 0.1
        matchFakes = MatchApFakesTask(config=matchFakesConfig)
        result = matchFakes.run(self.fakeCat,
                                self.exposure,
                                self.sourceCat)
        self.assertEqual(self.inExp.sum(), len(result.matchedDiaSources))
        self.assertEqual(
            len(self.sourceCat),
            np.sum(np.isfinite(result.matchedDiaSources["extraColumn"])))

    def testTrimCat(self):
        """Test that the correct number of sources are in the ccd area.
        """
        matchTask = MatchApFakesTask()
        result = matchTask._trimFakeCat(self.fakeCat, self.exposure)
        self.assertEqual(len(result), self.inExp.sum())


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
