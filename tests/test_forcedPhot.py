# This file is part of meas_base.
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

"""Tests of the various forced photometry tasks.

These tests primarily confirm that their respective Tasks can be configured and
run without errors, but do not check anything about their algorithmic quality.
"""


import dataclasses
import unittest

import numpy as np

import lsst.afw.image
from lsst.afw.geom import SkyWcs
from lsst.afw.math import ChebyshevBoundedField
from lsst.daf.butler import DataCoordinate, DatasetRef, DimensionUniverse
from lsst.pipe.tasks.forcedPhotDetector import ForcedPhotDetectorTask
from lsst.pipe.base import InMemoryDatasetHandle, PipelineGraph, Struct
import lsst.meas.base.tests
import lsst.utils.tests

skyCenter = lsst.geom.SpherePoint(245.0, -45.0, lsst.geom.degrees)


@dataclasses.dataclass
class _MockQuantum:
    dataId: DataCoordinate


class _MockRefsStruct:

    def __init__(self, datasets: dict[str, object], refs: dict[str, DatasetRef | list[DatasetRef]]):
        self._datasets = datasets
        self._refs = refs

    def __getattr__(self, name):
        return self._refs[name]


@dataclasses.dataclass
class _MockQuantumContext:

    quantum: _MockQuantum
    outputs: dict

    def get(self, inputs: _MockRefsStruct) -> object:
        return inputs._datasets

    def put(self, datasets: Struct, outputs: _MockRefsStruct) -> None:
        outputs._datasets = datasets.__dict__.copy()


@dataclasses.dataclass
class _MockTractInfo:

    wcs: SkyWcs

    def getWcs(self):
        return self.wcs


@dataclasses.dataclass
class _MockSkyMap:

    wcs: SkyWcs

    def __getitem__(self, tract):
        return _MockTractInfo(self.wcs)


class ForcedPhotometryTests:
    """Base class for tests of forced photometry tasks.

    Creates a simple test image and catalog to run forced photometry on.
    """
    def setUp(self):
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Point2I(100, 100))
        dataset = lsst.meas.base.tests.TestDataset(bbox, crval=skyCenter)
        dataset.addSource(instFlux=1000, centroid=lsst.geom.Point2D(30, 30))
        dataset.addSource(instFlux=10000, centroid=lsst.geom.Point2D(60, 70))

        schema = dataset.makeMinimalSchema()
        self.exposure, self.refCat = dataset.realize(noise=10, schema=schema)
        # Simple aperture correction map in case the task needs it.
        apCorrMap = lsst.afw.image.ApCorrMap()
        apCorrMap["base_PsfFlux_instFlux"] = ChebyshevBoundedField(bbox, np.array([[2.0]]))
        apCorrMap["base_PsfFlux_instFluxErr"] = ChebyshevBoundedField(bbox, np.array([[3.0]]))
        self.exposure.info.setApCorrMap(apCorrMap)
        table = self.refCat.asAstropy(copy=True)
        table.rename_column("id", "diaObjectId")
        table.rename_column("coord_ra", "ra")
        table.rename_column("coord_dec", "dec")
        table.rename_column("slot_Centroid_x", "x")
        table.rename_column("slot_Centroid_y", "y")
        table["ra"] = table["ra"].to("deg")
        table["dec"] = table["dec"].to("deg")
        self.table = table

        # Offset WCS so that the forced coordinates don't match the truth.
        self.offsetWcs = dataset.makePerturbedWcs(self.exposure.wcs)

        self.universe = DimensionUniverse()
        self.data_id = DataCoordinate.standardize(
            instrument="cam", skymap="map", tract=0, visit=1, detector=2, universe=self.universe,
        )
        self.quantum_context = _MockQuantumContext(
            quantum=_MockQuantum(dataId=self.data_id),
            outputs={},
        )
        self.inputs = {
            "exposure": self.exposure,
            "skyMap": _MockSkyMap(self.offsetWcs),
        }


class ForcedPhotDetectorTaskTestCase(ForcedPhotometryTests, lsst.utils.tests.TestCase):
    def testRun(self):
        """Test ForcedPhotDetectorTask.run."""
        config = ForcedPhotDetectorTask.ConfigClass()
        task = ForcedPhotDetectorTask(config=config)
        table = self.table
        result = task.run(table, self.exposure, self.offsetWcs)
        measTable = result.measTable

        # Check that something was measured.
        self.assertTrue(np.isfinite(measTable["base_TransformedCentroidFromCoord_x"]).all())
        self.assertTrue(np.isfinite(measTable["base_TransformedCentroidFromCoord_y"]).all())
        self.assertTrue(np.isfinite(measTable["base_PsfFlux_instFlux"]).all())
        # We use an offset WCS, so the transformed centroids should not exactly
        # match the original positions.
        self.assertFloatsNotEqual(measTable["base_TransformedCentroidFromCoord_x"], self.refCat['truth_x'])
        self.assertFloatsNotEqual(measTable["base_TransformedCentroidFromCoord_y"], self.refCat['truth_y'])

    def testRunQuantum(self):
        """Test ForcedPhotDetectorTask.runQuantum."""
        config = ForcedPhotDetectorTask.ConfigClass()
        print("columns before", self.table.columns)
        self.checkRunQuantum(
            config,
            None,
            InMemoryDatasetHandle(self.table, storageClass="ArrowAstropy"),
            "base_TransformedCentroidFromCoord",
        )

    def checkRunQuantum(self, config, get_init_input, ref_cat_input, centroid_name,
                        task_class=ForcedPhotDetectorTask):
        """Run tests on a runQuantum method.

        Parameters
        ----------
        config : `lsst.meas.base.ForcedPhotDetectorConfig`
            Configuration for the task.
        get_init_input : callable or `None`
            Callable that takes a single ignored argument and returns the
            init-input object expected by the task at construction.
        ref_cat_input : `lsst.pipe.base.InMemoryDatasetHandle`
            Handle holding the input reference catalog.
        centroid_name : `str`
            Base name of the centroid plugin.
        task_class : `type`, optional
            Subclass of `ForcedPhotDetectorTask` to use.
        """
        pipeline_graph = PipelineGraph(universe=self.universe)
        pipeline_graph.add_task("ForcedPhotDetector", task_class, config)
        pipeline_graph.resolve(dimensions=self.universe)
        init_outputs = []
        (task,) = pipeline_graph.instantiate_tasks(get_init_input=get_init_input, init_outputs=init_outputs)
        self.inputs["refCat"] = [ref_cat_input]
        exposure_dataset_type = pipeline_graph.dataset_types["calexp"].dataset_type
        input_refs = _MockRefsStruct(
            self.inputs,
            {
                # This particular runQuantum mostly just gets all inputs at
                # once, but it does need one DatasetRef with a proper data ID.
                "exposure": DatasetRef(
                    exposure_dataset_type,
                    self.quantum_context.quantum.dataId.subset(exposure_dataset_type.dimensions),
                    run="arbitrary",
                )
            }
        )
        output_refs = _MockRefsStruct({}, {})
        task.runQuantum(self.quantum_context, input_refs, output_refs)
        table = output_refs._datasets["measTable"]
        # Check that something was measured.
        self.assertTrue(np.isfinite(table[f"{centroid_name}_x"]).all())
        self.assertTrue(np.isfinite(table[f"{centroid_name}_y"]).all())
        self.assertTrue(np.isfinite(table["base_PsfFlux_instFlux"]).all())
        # We use an offset WCS, so the transformed centroids should not exactly
        # match the original positions.
        self.assertFloatsNotEqual(table[f"{centroid_name}_x"], self.refCat['truth_x'])
        self.assertFloatsNotEqual(table[f"{centroid_name}_y"], self.refCat['truth_y'])


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
