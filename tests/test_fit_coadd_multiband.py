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

import lsst.utils.tests
from lsst.daf.butler import DataCoordinate, DatasetRef, DatasetType, DimensionUniverse
from lsst.pipe.base import NoWorkFound
from lsst.pipe.tasks.fit_coadd_multiband import (
    CoaddMultibandFitConfig, CoaddMultibandFitConnections,
    CoaddMultibandFitSubConfig, CoaddMultibandFitSubTask,
)


class CoaddMultibandFitDummySubTask(CoaddMultibandFitSubTask):
    ConfigClass = CoaddMultibandFitSubConfig
    _DefaultName = "test"

    def run(self, catexps, cat_ref):
        return None


class CoaddMultibandFitTestCase(lsst.utils.tests.TestCase):
    """Tests adjustQuantum for now. Could run the task with mock data."""
    def setUp(self):
        self.config = CoaddMultibandFitConfig()
        self.config.fit_coadd_multiband.retarget(CoaddMultibandFitDummySubTask)
        self.config.fit_coadd_multiband.bands_fit = ("g", "r")
        self.connections = CoaddMultibandFitConnections(config=self.config)
        self.config.freeze()

        self.universe = DimensionUniverse()
        self.datasetType_coadd, self.datasetType_cat_meas = (
            DatasetType(
                name=connection.name,
                dimensions=connection.dimensions,
                storageClass=connection.storageClass,
                universe=self.universe,
            )
            for connection in (self.connections.coadds, self.connections.cats_meas)
        )
        self.skymap = "test"
        self.tract = 0
        self.patch = 0
        self.run = "test"
        kwargs_patch = {"skymap": self.skymap, "tract": self.tract, "patch": self.patch}

        self.inputs = {
            "coadds": (
                self.connections.coadds,
                tuple(
                    DatasetRef(
                        self.datasetType_coadd,
                        DataCoordinate.standardize(universe=self.universe, band=band, **kwargs_patch),
                        self.run,
                    )
                    for band in ("g", "r")
                ),
            ),
            "cats_meas": (
                self.connections.cats_meas,
                tuple(
                    DatasetRef(
                        self.datasetType_cat_meas,
                        DataCoordinate.standardize(universe=self.universe, band=band, **kwargs_patch),
                        self.run,
                    )
                    for band in ("r",)
                ),
            )
        }
        self.universe = DimensionUniverse()
        self.dataId = DataCoordinate.standardize(universe=self.universe, **kwargs_patch)

    def testAdjustQuantum(self):
        inputs, outputs = self.connections.adjustQuantum(
            self.inputs, outputs={}, label="test", data_id=self.dataId,
        )
        self.assertEqual(len(outputs), 0)

        for name, (connection, refs) in inputs.items():
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0].dataId["band"], "r")

    def testAdjustQuantumMissingAll(self):
        inputs = {
            "coadds": self.inputs["coadds"],
            "cats_meas": (self.connections.cats_meas, tuple()),
        }
        with self.assertRaises(NoWorkFound):
            self.connections.adjustQuantum(inputs, outputs={}, label="test", data_id=self.dataId)

    def testAdjustQuantumStrict(self):
        config = self.config.copy()
        config.allow_missing_bands = False
        connections = CoaddMultibandFitConnections(config=config)

        with self.assertRaises(NoWorkFound):
            connections.adjustQuantum(self.inputs, outputs={}, label="test", data_id=self.dataId)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
