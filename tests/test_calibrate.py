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

"""Test ProcessCcdTask and its immediate subtasks.
"""
import logging
import os
import shutil
import tempfile
import unittest

import lsst.utils.tests
import lsst.afw.image
import lsst.afw.math
import lsst.afw.table
import lsst.daf.butler.tests as butlerTests
from lsst.pipe.base import testUtils
from lsst.pipe.tasks.calibrate import CalibrateTask, CalibrateConfig
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig
import lsst.meas.extensions.piff.piffPsfDeterminer

TESTDIR = os.path.abspath(os.path.dirname(__file__))


class CalibrateTaskTestCaseWithButler(lsst.utils.tests.TestCase):

    @classmethod
    def _makeTestRepo(cls, root):
        """Create a repository with the metadata assumed by CalibrateTask.
        """
        # In-memory for performance
        config = lsst.daf.butler.Config()
        config["datastore", "cls"] = "lsst.daf.butler.datastores.inMemoryDatastore.InMemoryDatastore"
        config["datastore", "checksum"] = False
        config["registry", "db"] = "sqlite:///:memory:"

        butler = lsst.daf.butler.Butler(lsst.daf.butler.Butler.makeRepo(root, config=config), writeable=True)
        butler.registry.insertDimensionData(
            "instrument",
            {"name": "notACam", "visit_max": 256, "exposure_max": 256, "detector_max": 64})
        butler.registry.insertDimensionData(
            "physical_filter",
            {"instrument": "notACam", "name": "r", "band": "r"},
        )
        if "day_obs" in butler.dimensions:
            butler.registry.insertDimensionData(
                "day_obs",
                {"id": 20240201, "instrument": "notACam"},
            )
        butler.registry.insertDimensionData(
            "visit",
            {"instrument": "notACam", "id": 101, "name": "101", "physical_filter": "r", "day_obs": 20240201},
        )
        butler.registry.insertDimensionData("detector",
                                            {"instrument": "notACam", "id": 42, "full_name": "42"})
        return butler

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.root = tempfile.mkdtemp()
        cls.repo = cls._makeTestRepo(cls.root)

        butlerTests.addDatasetType(
            cls.repo, "icExp", {"instrument", "visit", "detector"},
            "ExposureF")
        butlerTests.addDatasetType(
            cls.repo, "icExpBackground", {"instrument", "visit", "detector"},
            "Background")
        butlerTests.addDatasetType(
            cls.repo, "icSrc", {"instrument", "visit", "detector"},
            "SourceCatalog")
        butlerTests.addDatasetType(
            cls.repo, "cal_ref_cat", {"htm7"},
            "SimpleCatalog")
        butlerTests.addDatasetType(
            cls.repo, "calexp", {"instrument", "visit", "detector"},
            "ExposureF")
        butlerTests.addDatasetType(
            cls.repo, "src", {"instrument", "visit", "detector"},
            "SourceCatalog")
        butlerTests.addDatasetType(
            cls.repo, "calexpBackground", {"instrument", "visit", "detector"},
            "Background")
        butlerTests.addDatasetType(
            cls.repo, "srcMatch", {"instrument", "visit", "detector"},
            "Catalog")
        butlerTests.addDatasetType(
            cls.repo, "srcMatchFull", {"instrument", "visit", "detector"},
            "Catalog")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        self.butler = butlerTests.makeTestCollection(self.repo, uniqueId=self.id())

        self.dataId = {"instrument": "notACam", "visit": 101, "detector": 42}
        # CalibrateTask absolutely requires an ExpandedDataCoordinate
        self.dataId = self.butler.registry.expandDataId(self.dataId)
        self.refcatId = {"htm7": 189584}

        # Tests do no processing, so we don't need real data
        self.exposure = lsst.afw.image.ExposureF(10, 10)
        background = lsst.afw.math.BackgroundMI(self.exposure.getBBox(), self.exposure.getMaskedImage())
        self.backgroundlist = lsst.afw.math.BackgroundList(
            (background, lsst.afw.math.Interpolate.UNKNOWN, lsst.afw.math.UndersampleStyle.THROW_EXCEPTION,
             lsst.afw.math.ApproximateControl.UNKNOWN, 0, 0, 1))
        self.icSrc = lsst.afw.table.SourceCatalog()
        self.refcat = lsst.afw.table.SimpleCatalog()

        self.butler.put(self.exposure, "icExp", self.dataId)
        self.butler.put(self.backgroundlist, "icExpBackground", self.dataId)
        self.butler.put(self.icSrc, "icSrc", self.dataId)
        self.butler.put(self.refcat, "cal_ref_cat", self.refcatId)

    def testDoAstrometry(self):
        """Ensure correct inputs passed to run whether or not doAstrometry
        is set.
        """
        allIds = {key: self.dataId for key in {
            "exposure", "background", "icSourceCat", "outputExposure", "outputCat", "outputBackground",
            "matches", "matchesDenormalized"
        }}
        allIds.update({key: [self.refcatId] for key in {"astromRefCat", "photoRefCat"}})

        self._checkDoRefcats(doAstrometry=True, doPhotoCal=True, ids=allIds)
        self._checkDoRefcats(doAstrometry=False, doPhotoCal=True, ids=allIds)

    def testDoPhotoCal(self):
        """Ensure correct inputs passed to run whether or not doPhotoCal
        is set.
        """
        allIds = {key: self.dataId for key in {
            "exposure", "background", "icSourceCat", "outputExposure", "outputCat", "outputBackground",
            "matches", "matchesDenormalized"
        }}
        allIds.update({key: [self.refcatId] for key in {"astromRefCat", "photoRefCat"}})

        self._checkDoRefcats(doAstrometry=True, doPhotoCal=True, ids=allIds)
        self._checkDoRefcats(doAstrometry=True, doPhotoCal=False, ids=allIds)

    def _checkDoRefcats(self, doAstrometry, doPhotoCal, ids):
        """Test whether run is called with the correct arguments.

        In the case of `CalibrateTask`, the inputs should not depend on the
        task configuration.

        Parameters
        ----------
        doAstrometry, doPhotoCal : `bool`
            Values of the config flags of the same name.
        ids : `dict` [`str`]
            A mapping from the input dataset type to the data ID of the
            dataset to process.
        """
        config = CalibrateConfig()
        config.doWriteMatches = False  # no real output to write
        config.doAstrometry = doAstrometry
        config.doPhotoCal = doPhotoCal
        config.connections.photoRefCat = "cal_ref_cat"
        config.connections.astromRefCat = "cal_ref_cat"
        config.idGenerator.packer.name = "observation"
        task = CalibrateTask(config=config)
        quantumId = ids["exposure"]

        quantum = testUtils.makeQuantum(task, self.butler, quantumId, ids)
        run = testUtils.runTestQuantum(task, self.butler, quantum)

        run.assert_called_once()
        self.assertEqual(run.call_args[0], ())
        # Some arguments unprintable because we don't have a full environment
        #     So just check which ones were passed in
        self.assertEqual(run.call_args[1].keys(),
                         {"exposure", "idGenerator", "background", "icSourceCat"})


class CalibrateTaskTestCaseNoButler(lsst.utils.tests.TestCase):

    def testNoAperCorrMap(self):
        expPath = os.path.join(TESTDIR, "data", "v695833-e0-c000-a00.sci.fits")
        exposure = lsst.afw.image.ExposureF(expPath)

        charImConfig = CharacterizeImageConfig()
        charImConfig.measurePsf.psfDeterminer = 'piff'
        charImConfig.measurePsf.psfDeterminer['piff'].spatialOrder = 0
        charImConfig.measureApCorr.sourceSelector["science"].doSignalToNoise = False
        charImTask = CharacterizeImageTask(config=charImConfig)
        charImResults = charImTask.run(exposure)
        calibConfig = CalibrateConfig()
        calibConfig.doAstrometry = False
        calibConfig.doPhotoCal = False
        calibConfig.doSkySources = False
        calibConfig.doComputeSummaryStats = False

        # Force the aperture correction map to None (DM-39626)
        exposure.info.setApCorrMap(None)
        calibTask = CalibrateTask(config=calibConfig)
        with self.assertLogs(level=logging.WARNING) as cm:
            _ = calibTask.run(charImResults.exposure)
        # Other warnings may also be emitted.
        warnings = '\n'.join(cm.output)
        self.assertIn("Image does not have valid aperture correction map", warnings)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
