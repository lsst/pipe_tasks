# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import unittest
import tempfile

import astropy.time

from lsst.afw.table import SourceCatalog
import lsst.daf.butler
import lsst.daf.butler.tests as butlerTests
import lsst.pipe.base.testUtils as pipeTests
from lsst.pipe.tasks.getRegionTimeFromVisit import GetRegionTimeFromVisitTask
from lsst.sphgeom import ConvexPolygon, UnitVector3d
import lsst.utils.tests


class GetRegionTimeFromVisitTests(lsst.utils.tests.TestCase):
    def setUp(self):
        instrument = "NotACam"
        detector = 42
        group = "groupy"
        exposure = 1011
        filter = "k2024"
        # Coordinates taken from LATISS exposure 2024040800445
        day_obs = 20240408
        ra = 122.47171635551595
        dec = -36.20378247543336
        rot = 359.99623587800414
        self.region = ConvexPolygon(
            [UnitVector3d(-0.43197476135299717, 0.6808244361827491, -0.5915030791555212),
             UnitVector3d(-0.4337643437542999, 0.6796857349601265, -0.5915029972697637),
             UnitVector3d(-0.4344262837761736, 0.6807261608742085, -0.5898183600617098),
             UnitVector3d(-0.43263670132940496, 0.6818648620502468, -0.5898184420345037),
             ])
        self.times = lsst.daf.butler.Timespan(astropy.time.Time("2024-04-09T03:50:00", scale="tai"),
                                              astropy.time.Time("2024-04-09T03:50:30", scale="tai"))

        repo_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tempfile.TemporaryDirectory.cleanup, repo_dir)
        self.repo = butlerTests.makeTestRepo(repo_dir.name)
        self.enterContext(self.repo)

        butlerTests.addDataIdValue(self.repo, "instrument", instrument)
        butlerTests.addDataIdValue(self.repo, "day_obs", day_obs)
        butlerTests.addDataIdValue(self.repo, "physical_filter", filter)
        butlerTests.addDataIdValue(self.repo, "detector", detector)
        butlerTests.addDataIdValue(self.repo, "group", group)
        # addDataIdValue can't handle metadata, or tables that don't have an ID column
        self.repo.registry.insertDimensionData("exposure", {"id": exposure,
                                                            "instrument": instrument,
                                                            "group": group,
                                                            "day_obs": day_obs,
                                                            "physical_filter": filter,
                                                            "tracking_ra": ra,
                                                            "tracking_dec": dec,
                                                            "sky_angle": rot,
                                                            "timespan": self.times,
                                                            })
        self.repo.registry.insertDimensionData("visit", {"id": exposure,
                                                         "instrument": instrument,
                                                         "day_obs": day_obs,
                                                         "physical_filter": filter,
                                                         "timespan": self.times,
                                                         })
        self.repo.registry.insertDimensionData("visit_definition", {"instrument": instrument,
                                                                    "exposure": exposure,
                                                                    "visit": exposure,
                                                                    })
        self.repo.registry.insertDimensionData("visit_detector_region", {"instrument": instrument,
                                                                         "visit": exposure,
                                                                         "detector": detector,
                                                                         "region": self.region,
                                                                         })

        butlerTests.addDatasetType(self.repo, "regionTimeInfo", {"instrument", "group", "detector"},
                                   "RegionTimeInfo")
        butlerTests.addDatasetType(self.repo, "initial_stars_footprints_detector",
                                   {"instrument", "visit", "detector"}, "SourceCatalog")
        # pipeTests.makeQuantum needs outputs registered even if graph generation does not.
        butlerTests.addDatasetType(self.repo, "getRegionTimeFromVisit_dummy2",
                                   {"instrument", "exposure", "detector"}, "int")

        self.group_id = self.repo.registry.expandDataId(
            {"instrument": instrument, "group": group, "detector": detector})
        self.exposure_id = self.repo.registry.expandDataId(
            {"instrument": instrument, "exposure": exposure, "detector": detector})
        self.visit_id = self.repo.registry.expandDataId(
            {"instrument": instrument, "visit": exposure, "detector": detector})

    def test_runQuantum(self):
        butler = butlerTests.makeTestCollection(self.repo, uniqueId=self.id())
        minimal_source_catalog = SourceCatalog()
        butler.put(minimal_source_catalog, "initial_stars_footprints_detector", self.visit_id)

        task = GetRegionTimeFromVisitTask()
        quantum = pipeTests.makeQuantum(
            task, butler, self.group_id,
            {"output": self.group_id,
             "dummy_visit": self.visit_id,
             "dummy_exposure": [self.exposure_id],
             })

        pipeTests.runTestQuantum(task, butler, quantum, mockRun=False)

        # Not exactly round-tripping, because these objects came from the dimension records.
        info = butler.get("regionTimeInfo", self.group_id)
        self.assertEqual(info.region, self.region)
        self.assertEqual(info.timespan, self.times)

    def test_connections(self):
        pipeTests.lintConnections(GetRegionTimeFromVisitTask.ConfigClass.ConnectionsClass)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
